"""Run the evaluation contract for a run config, logged under runs/<run_id>/.  [CLAUDE.md §5, §7]

    python -m unitwave.cli.evaluate configs/runs/phase3_first_table.yaml

Writes manifest.json first (status "running", updated after each target):
- git SHA, branch and whether the tree was dirty;
- the seed;
- every config file used, with its sha256;
- the preprocessing and target fingerprints and the split hash;
- the sessions, library versions and per-target timings.

Also split.json. Per target, <target>/ holds:
- report.txt: the six-row table and verdicts;
- normalizer.json (normalizers.json, one per fold, for a leave_one_block_out split):
  the Normalizer fit on the split's training data, with its hash (R3);
- pseudo.parquet: per-session scores per pseudo-session, when the target has them;
- metrics.json: every per-session metric, the summary, the verdicts and any dropped
  trials; the numbers any later report must trace back to;
- per_session.parquet and shuffle.parquet.

Studio (S3) decodes one session through the same functions: `build_splits`,
`evaluate_target` and `write_target`, so its table is the CLI's for the same session,
units, split and seed.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.data.load import load_data_config, load_session
from unitwave.data.manifest import Manifest, build_manifest
from unitwave.evaluation.contract import ContractResult, evaluate
from unitwave.evaluation.data import KINDS, PER_BIN, SplitData
from unitwave.evaluation.nulls import load_null_config
from unitwave.models.baselines.features import BaselineConfig, load_baseline_config
from unitwave.models.baselines.linear import (
    LogisticDecoder,
    RidgeDecoder,
    SpikesAndTaskLogistic,
    SpikesAndTaskRidge,
    TrialStructureLogistic,
    TrialStructureRidge,
)
from unitwave.models.baselines.rrr import RRRClassification, RRRRegression
from unitwave.preprocess.binning import PREPROC_VERSION, load_preproc_config
from unitwave.splits.registry import leave_one_block_out as leave_one_block_out_split
from unitwave.splits.registry import save_split, within_session
from unitwave.targets.config import load_target_config

REPO = Path(__file__).resolve().parents[2]
CONFIG_FILES = ("qc", "preprocess", "targets", "nulls", "evaluation", "baselines")
MODELS = ("ridge",)
RRR_ONE_SESSION = "multi-session by design; it is not fit on one session"


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    fixed_sessions: tuple[str, ...]
    random_sessions: int
    train_fraction: float
    gap_s: float
    targets: tuple[str, ...]
    train_stride: dict
    model: str
    n_shifts: int
    # Targets evaluated on a leave_one_block_out split instead of the within-session one.
    leave_one_block_out: tuple[str, ...] = ()


def load_run_config(path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    expected = {"name", "seed", "sessions", "split", "targets", "train_stride", "model", "n_shifts"}
    if set(raw) != expected:
        raise ValueError(f"{path}: keys {sorted(raw)}, expected {sorted(expected)}")
    unknown = sorted(set(raw["targets"]) - set(KINDS))
    if unknown:
        raise ValueError(f"{path}: unknown targets {unknown}")
    if raw["model"] not in MODELS:
        raise ValueError(f"{path}: model must be one of {MODELS}")
    split_keys = set(raw["split"]) - {"train_fraction", "gap_s", "leave_one_block_out"}
    if split_keys:
        raise ValueError(f"{path}: unknown split keys {sorted(split_keys)}")
    lobo = tuple(raw["split"].get("leave_one_block_out", []))
    if set(lobo) - {"choice", "block"}:
        raise ValueError(f"{path}: leave_one_block_out is for trial targets (choice, block)")
    return RunConfig(
        name=raw["name"],
        seed=int(raw["seed"]),
        fixed_sessions=tuple(raw["sessions"]["fixed"]),
        random_sessions=int(raw["sessions"]["random"]),
        train_fraction=float(raw["split"]["train_fraction"]),
        gap_s=float(raw["split"]["gap_s"]),
        targets=tuple(raw["targets"]),
        train_stride=dict(raw["train_stride"]),
        model=raw["model"],
        n_shifts=int(raw["n_shifts"]),
        leave_one_block_out=lobo,
    )


def select_sessions(manifest: Manifest, fixed, n_random: int, *, seed: int) -> list[str]:
    """The fixed sessions, then n_random others drawn from the manifest with the seed."""
    available = set(manifest.sessions["eid"])
    missing = [e for e in fixed if e not in available]
    if missing:
        raise ValueError(f"fixed sessions not in the manifest: {missing}")
    pool = sorted(available - set(fixed))
    extra = np.random.default_rng(seed).choice(pool, size=n_random, replace=False)
    return list(fixed) + sorted(str(e) for e in extra)


def decoders(target: str, provider: SplitData, cfg: BaselineConfig) -> dict:
    """The row factories for one target: model and baseline_ridge are the same baseline;
    model_with_task is that baseline given the task features too."""
    jobs, ratios = cfg.n_jobs, cfg.task_penalty_ratios
    if target == "wheel_velocity":
        chunks = cfg.per_bin_chunks
        return {
            "model": lambda: RidgeDecoder(chunks, cfg.cv, n_jobs=jobs),
            "model_with_task": lambda: SpikesAndTaskRidge(chunks, cfg.cv, ratios, n_jobs=jobs),
            "baseline_ridge": lambda: RidgeDecoder(chunks, cfg.cv, n_jobs=jobs),
            "baseline_rrr": lambda: RRRRegression(chunks, cfg.cv),
            "trialstruct": lambda: TrialStructureRidge(cfg.cv, n_jobs=jobs),
        }
    chunks = cfg.per_bin_chunks if target in PER_BIN else cfg.trial_chunks
    rrr_chunks = cfg.per_bin_chunks if target in PER_BIN else provider.context_bins
    return {
        "model": lambda: LogisticDecoder(chunks, cfg.cv, n_jobs=jobs),
        "model_with_task": lambda: SpikesAndTaskLogistic(chunks, cfg.cv, ratios, n_jobs=jobs),
        "baseline_ridge": lambda: LogisticDecoder(chunks, cfg.cv, n_jobs=jobs),
        "baseline_rrr": lambda: RRRClassification(rrr_chunks, cfg.cv),
        "trialstruct": lambda: TrialStructureLogistic(cfg.cv, n_jobs=jobs),
    }


def build_splits(manifest, trials, preproc, *, train_fraction, gap_s, leave_one_block_out=()):
    """The within-session split of the sessions in trials, and the leave-one-block-out
    one when any target uses it (else None)."""
    split = within_session(manifest, trials, preproc, train_fraction=train_fraction, gap_s=gap_s)
    lobo = None
    if leave_one_block_out:
        lobo = leave_one_block_out_split(manifest, trials, preproc, gap_s=gap_s)
    return split, lobo


def evaluate_target(
    target: str,
    split,
    lobo_split,
    *,
    leave_one_block_out,
    seed: int,
    n_shifts: int,
    train_stride: dict,
    baselines: BaselineConfig,
    load,
    unit_ids=None,
    rrr: bool = True,
    progress=None,
    task=None,
    preproc=None,
) -> tuple[ContractResult, SplitData]:
    """Run the contract for one target; unit_ids, rrr and progress are Studio's, and
    task and preproc (with the session's own QC rule) are for a task's own targets."""
    provider = SplitData(
        lobo_split if target in leave_one_block_out else split,
        target,
        context_bins=baselines.per_bin_context_bins if target in PER_BIN else None,
        train_stride=train_stride.get(target, 1) if target in PER_BIN else 1,
        load=load,
        unit_ids=unit_ids,
        task=task,
        preproc=preproc,
    )
    rows = decoders(target, provider, baselines)
    not_run = None
    if not rrr:
        rows["baseline_rrr"], not_run = None, {"baseline_rrr": RRR_ONE_SESSION}
    nulls = load_null_config()
    result = evaluate(
        provider,
        ceiling=None,
        seed=seed,
        n_shifts=n_shifts,
        n_pseudo=nulls.n_pseudo_sessions if provider.pseudo_sessions else None,
        not_run=not_run,
        progress=progress,
        **rows,
    )
    return result, provider


def _git() -> dict:
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    """NaN and infinities become null; numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(value) else float(value)
    return value


def write_target(out: Path, target: str, result: ContractResult, provider: SplitData, note: str):
    out.mkdir(parents=True, exist_ok=True)
    # R3: the normalisation statistics, fit on training data only, travel with the run
    # (one per fold for a leave_one_block_out split).
    if not provider.lobo:
        normalizer = provider.normalizers[0].to_dict()
        (out / "normalizer.json").write_text(json.dumps(_jsonable(normalizer), allow_nan=False))
    else:
        normalizers = [n.to_dict() for n in provider.normalizers]
        (out / "normalizers.json").write_text(json.dumps(_jsonable(normalizers), allow_nan=False))
    (out / "report.txt").write_text(f"{target}\n{note}\n{result.report()}\n")
    per_session = pd.concat(result.per_session, names=["row", "eid"])
    per_session.to_parquet(out / "per_session.parquet")
    result.shuffle.rename(columns=str).to_parquet(out / "shuffle.parquet")
    if result.pseudo is not None:
        result.pseudo.rename(columns=str).to_parquet(out / "pseudo.parquet")
    metrics = {
        "target": target,
        "kind": result.kind,
        "primary": result.primary,
        "seed": result.seed,
        "n_shifts": result.n_shifts,
        "split_hash": result.split_hash,
        "ceiling_is_model": result.ceiling_is_model,
        "rows": {row: table.to_dict(orient="index") for row, table in result.per_session.items()},
        "summary": result.summary().to_dict(orient="index"),
        "verdicts": [vars(v) for v in result.verdicts],
        "gate": vars(result.gate),
        "n_folds": result.n_folds,
        "n_pseudo": result.n_pseudo,
        "dropped": {f"{e}/{p}": n for (e, p), n in provider.dropped.items()},
        "normalizer_hashes": [n.hash for n in provider.normalizers],
    }
    (out / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=1, allow_nan=False))


def run(
    config: RunConfig,
    *,
    config_path: Path,
    runs_dir: Path = REPO / "runs",
    manifest: Manifest | None = None,
    load=None,
) -> Path:
    data = load_data_config()
    manifest = manifest or build_manifest(data.bwm_ephys_root, data.bwm_behavior_root)
    load = load or (lambda eid: load_session(eid, "bwm"))
    preproc, targets_cfg, baselines = (
        load_preproc_config(),
        load_target_config(),
        load_baseline_config(),
    )

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{config.name}"
    out = Path(runs_dir) / run_id
    out.mkdir(parents=True)
    eids = select_sessions(
        manifest, config.fixed_sessions, config.random_sessions, seed=config.seed
    )
    trials = {eid: load(eid).trials for eid in eids}
    split, lobo_split = build_splits(
        manifest,
        trials,
        preproc,
        train_fraction=config.train_fraction,
        gap_s=config.gap_s,
        leave_one_block_out=config.leave_one_block_out,
    )
    save_split(split, out / "split.json")
    if lobo_split is not None:
        save_split(lobo_split, out / "split_leave_one_block_out.json")

    config_files = {"run": Path(config_path)}
    config_files |= {name: REPO / "configs" / f"{name}.yaml" for name in CONFIG_FILES}
    run_manifest = {
        "run_id": run_id,
        "status": "running",
        "created": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git": _git(),
        "seed": config.seed,
        "configs": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "content": yaml.safe_load(path.read_text()),
            }
            for name, path in config_files.items()
        },
        "preproc_version": PREPROC_VERSION,
        "preproc_fingerprint": preproc.fingerprint(),
        "targets_fingerprint": targets_cfg.fingerprint(),
        "split_hash": split.hash,
        "split_leave_one_block_out_hash": None if lobo_split is None else lobo_split.hash,
        "manifest_provenance": manifest.provenance,
        "sessions": eids,
        "versions": {
            "python": platform.python_version(),
            **{m: sys.modules[m].__version__ for m in ("numpy", "pandas", "scipy", "sklearn")},
        },
        "targets": {},
    }

    def save_manifest():
        (out / "manifest.json").write_text(json.dumps(_jsonable(run_manifest), indent=1))

    save_manifest()
    note = (
        "model and baseline_ridge are the same decoder in this run (model: ridge), "
        "so the baseline_ridge verdict is vacuous."
    )
    for target in config.targets:
        start = time.time()
        result, provider = evaluate_target(
            target,
            split,
            lobo_split,
            leave_one_block_out=config.leave_one_block_out,
            seed=config.seed,
            n_shifts=config.n_shifts,
            train_stride=config.train_stride,
            baselines=baselines,
            load=load,
        )
        write_target(out / target, target, result, provider, note)
        run_manifest["targets"][target] = {
            "seconds": round(time.time() - start, 1),
            "split_hash": result.split_hash,
            "gate_passed": result.gate.beats,
            "normalizer_hashes": [n.hash for n in provider.normalizers],
        }
        save_manifest()
        print(f"{target}: done in {time.time() - start:.0f} s", flush=True)
    run_manifest["status"] = "complete"
    save_manifest()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation contract for a run config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    args = parser.parse_args(argv)
    out = run(load_run_config(args.config), config_path=args.config, runs_dir=args.runs_dir)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
