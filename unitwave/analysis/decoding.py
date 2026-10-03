"""Decoding in Studio (S3; any task since step 8a): one session, the shown units, the
evaluation contract.

The same path as the CLI (cli.evaluate: build_splits, evaluate_target, write_target),
so Studio's table is the CLI's for the same session, units, split and seed:
- **Split (R1, R2):** within-session, early trials train and late trials test with a
  gap; block uses leave-one-block-out. Both come from the split registry. A BWM
  session is registered with the release manifest; another session with a one-session
  catalogue (its eid, subject and lab unknown), which within-session splits need no
  more than that.
- **Targets:** IBL's (choice, stimulus side, block, movement state), or a task's own
  ("task:<name>", targets/task.py): its decoding section names the condition, levels,
  window, trial filters and the shuffle null's minimum shift. Their no-spikes baseline
  uses the task's trialstruct variables.
- **Normalisation (R3):** fit on the training data only, stored with the run.
- **Rows:** every contract row that applies to one session. baseline_rrr is
  multi-session by design and not run; ceiling_within is the model itself, since
  the split is within-session; model and baseline_ridge are the same logistic
  decoder, as no deep model is run.
- **Verdicts:** within-session tests with BH (evaluation.single_session), because the
  contract's across-session test needs at least 5 sessions.
- **Units:** decoding uses the shown units that pass the session's own unit QC (IBL's
  configs/qc.yaml, or the Phy, spike-time or NWB rule: the page's); shown units
  failing QC are left out and counted.
- **Run log:** written to runs/<run_id>/ with the section 7 manifest: git SHA, seed,
  config hashes, preprocessing and target fingerprints, split hash, units.
"""

import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.cli.evaluate import (
    REPO,
    _git,
    _jsonable,
    _sha256,
    build_splits,
    evaluate_target,
    write_target,
)
from unitwave.data.load import load_data_config, load_session
from unitwave.data.manifest import Manifest, build_manifest
from unitwave.evaluation.contract import _FAILURE, GATE, ROWS, ContractResult
from unitwave.evaluation.data import KINDS, SplitData
from unitwave.evaluation.single_session import SessionTest, session_tests
from unitwave.models.baselines.features import load_baseline_config
from unitwave.preprocess.binning import PREPROC_VERSION, PreprocConfig, load_preproc_config
from unitwave.splits.registry import save_split
from unitwave.targets.config import load_target_config
from unitwave.targets.task import PREFIX as TASK_PREFIX

DEFAULT_CONFIG = REPO / "configs" / "studio_decoding.yaml"
CONFIG_FILES = ("qc", "preprocess", "targets", "nulls", "evaluation", "baselines")
_KEYS = {
    "seed",
    "split",
    "train_stride",
    "targets",
    "model",
    "n_shifts",
    "n_bootstrap",
    "alpha",
}
SAME_AS_MODEL = {
    "baseline_ridge": "the same logistic decoder as model: no deep model is run in Studio",
    "ceiling_within": "the model itself: the split is already within-session",
}
NOTE = (
    "model and baseline_ridge are the same decoder in this run (model: ridge), so the "
    "baseline_ridge verdict is vacuous. One session: verdicts are the within-session "
    "tests in single_session.json (the contract's across-session test needs 5 sessions)."
)


@dataclass(frozen=True)
class DecodingConfig:
    seed: int
    train_fraction: float
    gap_s: float
    leave_one_block_out: tuple[str, ...]
    train_stride: dict
    targets: tuple[str, ...]
    model: str
    n_shifts: int
    n_bootstrap: int
    alpha: float


def load_decoding_config(path: str | os.PathLike = DEFAULT_CONFIG) -> DecodingConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if set(raw) != _KEYS:
        raise ValueError(f"{path}: keys {sorted(raw)}, expected {sorted(_KEYS)}")
    unknown = sorted(set(raw["targets"]) - set(KINDS))
    if unknown:
        raise ValueError(f"{path}: unknown targets {unknown}")
    return DecodingConfig(
        seed=int(raw["seed"]),
        train_fraction=float(raw["split"]["train_fraction"]),
        gap_s=float(raw["split"]["gap_s"]),
        leave_one_block_out=tuple(raw["split"]["leave_one_block_out"]),
        train_stride=dict(raw["train_stride"]),
        targets=tuple(raw["targets"]),
        model=str(raw["model"]),
        n_shifts=int(raw["n_shifts"]),
        n_bootstrap=int(raw["n_bootstrap"]),
        alpha=float(raw["alpha"]),
    )


def _line(t: SessionTest) -> str:
    who = f"{t.subject} vs {t.row}"
    if not np.isfinite(t.p):
        return f"{who}: undefined, {t.note}."
    detail = f"ΔAUROC {t.difference:+.3f}, p = {t.p:.3g}, q = {t.q:.3g}; {t.method}"
    if t.note:
        return f"{who}: {t.note}, so this comparison can't show a difference ({detail})."
    if t.beats:
        return f"{t.subject} beats {t.row} ({detail})."
    return f"{t.subject} does NOT beat {t.row} ({detail}): {_FAILURE[(t.subject, t.row)]}."


@dataclass
class DecodingRun:
    """One target decoded on one session: the contract result, its single-session
    tests, the units used and the run folder."""

    folder: Path
    target: str
    eid: str
    result: ContractResult
    tests: tuple[SessionTest, ...]
    provider: SplitData
    units_used: list[str]
    units_excluded: list[str]
    cfg: DecodingConfig
    seconds: float = 0.0
    excluded_trials: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Everything the page shows, as strict JSON (NaN -> null)."""
        r, eid = self.result, self.eid
        rows = []
        for row in ROWS:
            if row not in r.per_session:
                continue
            values = r.per_session[row].loc[eid].to_dict()
            rows.append({"row": row, **values, "same_as": SAME_AS_MODEL.get(row, "")})
        gate = next(t for t in self.tests if (t.subject, t.row) == GATE)
        split = self.provider.split
        record = split.sessions[eid]
        if split.kind == "leave_one_block_out":
            split_info = {"kind": split.kind, "n_folds": len(record["folds"])}
        else:
            split_info = {
                "kind": split.kind,
                "train_trials": len(record["trials"]["train"]),
                "test_trials": len(record["trials"]["test"]),
            }
        dropped = {k: v for (e, k), v in self.provider.dropped.items() if e == eid and v}
        out = {
            "target": self.target,
            "eid": eid,
            "kind": r.kind,
            "primary": r.primary,
            "rows": rows,
            "not_applicable": dict(r.not_run),
            "tests": [{**vars(t), "is_gate": t.is_gate} for t in self.tests],
            "lines": [_line(t) for t in self.tests],
            "gate": {
                "passed": gate.beats,
                "line": (
                    f"GATE ({gate.subject} vs {gate.row}): "
                    + ("PASSED." if gate.beats else f"NOT PASSED: {_FAILURE[GATE]}.")
                ),
            },
            "null": {
                "n_shifts": r.n_shifts,
                # Draws that exist for this session (a short one allows fewer shifts).
                "n_shifts_used": int(np.isfinite(r.shuffle.loc[eid]).sum()),
                "n_pseudo": r.n_pseudo,
                "n_pseudo_used": (
                    0 if r.pseudo is None else int(np.isfinite(r.pseudo.loc[eid]).sum())
                ),
                "n_bootstrap": self.cfg.n_bootstrap,
                "seed": r.seed,
                "alpha": self.cfg.alpha,
                "correction": f"Benjamini-Hochberg across these {len(self.tests)} comparisons",
            },
            "split": {**split_info, "hash": split.hash, "gap_s": self.cfg.gap_s},
            "trials": {"excluded": self.excluded_trials, "dropped_windows": dropped},
            "units": {
                "used": len(self.units_used),
                "excluded_failing_qc": len(self.units_excluded),
            },
            "run": self.folder.name,
            "seconds": round(self.seconds, 1),
        }
        return _jsonable(out)


def one_session_catalog(eid: str) -> Manifest:
    """A catalogue of one session that isn't in the BWM release, for the split registry:
    within-session splits read each session's subject and lab, unknown here."""
    sessions = pd.DataFrame({"eid": [eid], "subject": ["unknown"], "lab": ["unknown"]})
    provenance = {"catalog": "one session, not a release manifest", "eid": eid}
    return Manifest(sessions=sessions, insertions=pd.DataFrame(), provenance=provenance)


def decode(
    eid: str,
    target: str,
    *,
    unit_ids: Sequence[str] | None = None,
    cfg: DecodingConfig | None = None,
    manifest: Manifest | None = None,
    load: Callable | None = None,
    runs_dir: Path = REPO / "runs",
    progress: Callable[[str, int, int], None] | None = None,
    session=None,
    qc=None,
    task=None,
) -> DecodingRun:
    """Decode target from one session's units (unit_ids: the shown units, or None for
    every QC-passing unit). progress(stage, done, total) is told as fits finish.

    IBL's targets need a BWM session. A task's own target ("task:<name>") needs the
    open session, its unit QC rule and its task definition."""
    cfg = cfg or load_decoding_config()
    start = time.time()
    preproc, targets_cfg, baselines = (
        load_preproc_config(),
        load_target_config(),
        load_baseline_config(),
    )
    if target.startswith(TASK_PREFIX):
        name = target.removeprefix(TASK_PREFIX)
        if session is None or qc is None or task is None:
            raise ValueError(f"{target}: decoding a task's target needs the session, QC and task")
        if task.decoding is None or name not in task.decoding.targets:
            raise ValueError(f"the {task.label} definition declares no decoding target {name!r}")
        manifest = one_session_catalog(eid)

        def load(e, opened=session):  # the one session, already open
            return opened

        preproc = PreprocConfig(bin_ms=preproc.bin_ms, qc=qc)  # the session's own rule
        targets_fingerprint = task.fingerprint
    else:
        if target not in cfg.targets:
            raise ValueError(
                f"{target} is not a Studio decoding target: one of {list(cfg.targets)}"
            )
        data = load_data_config()
        manifest = manifest or build_manifest(data.bwm_ephys_root, data.bwm_behavior_root)
        load = load or (lambda e: load_session(e, "bwm"))
        if eid not in set(manifest.sessions["eid"]):
            raise ValueError("decoding needs a session from the Brain Wide Map release")
        targets_fingerprint = targets_cfg.fingerprint()
    lobo = tuple(t for t in cfg.leave_one_block_out if t == target)
    split, lobo_split = build_splits(
        manifest,
        {eid: load(eid).trials},
        preproc,
        train_fraction=cfg.train_fraction,
        gap_s=cfg.gap_s,
        leave_one_block_out=lobo,
    )
    tag = Path(eid.removeprefix("nwb:").removeprefix("phy:")).stem[:16] if ":" in eid else eid[:8]
    slug = target.removeprefix(TASK_PREFIX)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_studio_decode_{slug}_{tag}"
    out = Path(runs_dir) / run_id
    out.mkdir(parents=True)
    save_split(split, out / "split.json")
    if lobo_split is not None:
        save_split(lobo_split, out / "split_leave_one_block_out.json")
    files = {"studio_decoding": DEFAULT_CONFIG}
    files |= {name: REPO / "configs" / f"{name}.yaml" for name in CONFIG_FILES}
    used_split = lobo_split if lobo else split
    run_manifest = {
        "run_id": run_id,
        "status": "running",
        "created": datetime.now(UTC).isoformat(),
        "command": "UnitWave Studio: decoding",
        "git": _git(),
        "seed": cfg.seed,
        "configs": {
            name: {"path": str(p), "sha256": _sha256(p), "content": yaml.safe_load(p.read_text())}
            for name, p in files.items()
        },
        "studio_config": vars(cfg),
        "preproc_version": PREPROC_VERSION,
        "preproc_fingerprint": preproc.fingerprint(),
        "targets_fingerprint": targets_fingerprint,
        "task": None if task is None else {"name": task.name, "fingerprint": task.fingerprint},
        "split_hash": used_split.hash,
        "manifest_provenance": manifest.provenance,
        "sessions": [eid],
        "target": target,
        "versions": {
            "python": platform.python_version(),
            **{m: sys.modules[m].__version__ for m in ("numpy", "pandas", "scipy", "sklearn")},
        },
    }

    def save():
        (out / "manifest.json").write_text(json.dumps(_jsonable(run_manifest), indent=1))

    save()
    result, provider = evaluate_target(
        target,
        split,
        lobo_split,
        leave_one_block_out=lobo,
        seed=cfg.seed,
        n_shifts=cfg.n_shifts,
        train_stride=cfg.train_stride,
        baselines=baselines,
        load=load,
        unit_ids=None if unit_ids is None else {eid: list(unit_ids)},
        rrr=False,
        progress=None if progress is None else (lambda d, t: progress("fitting", d, t)),
        task=task if target.startswith(TASK_PREFIX) else None,
        preproc=preproc,
    )
    if progress is not None:
        progress("bootstrap", 0, 1)
    tests = session_tests(
        result, provider.sample_trials, n_bootstrap=cfg.n_bootstrap, alpha=cfg.alpha, seed=cfg.seed
    )
    units = provider._prepared[eid].units.index.tolist()
    excluded = provider.units_excluded.get(eid, [])
    # Trial targets count their exclusions; per-bin targets have none to count.
    target_excluded = dict(getattr(provider._prepared[eid].target, "excluded", {}))
    run = DecodingRun(
        folder=out,
        target=target,
        eid=eid,
        result=result,
        tests=tests,
        provider=provider,
        units_used=units,
        units_excluded=excluded,
        cfg=cfg,
        excluded_trials=target_excluded,
    )
    write_target(out / target, target, result, provider, NOTE)
    run.seconds = time.time() - start
    summary = run.summary()
    (out / target / "single_session.json").write_text(
        json.dumps(summary, indent=1, allow_nan=False)
    )
    run_manifest.update(
        status="complete",
        seconds=round(run.seconds, 1),
        units={"used": units, "excluded_failing_qc": excluded},
        gate_passed=summary["gate"]["passed"],
    )
    save()
    if progress is not None:
        progress("done", 1, 1)
    return run
