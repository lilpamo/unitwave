"""Region summaries across a session set, logged under runs/<run_id>/.  [S5, §5, §7]

    python -m unitwave.cli.summarise SET.unitwave-set.json --label responsive --event stim_on
    python -m unitwave.cli.summarise SET.unitwave-set.json --label selective --event stim_on --split choice
    python -m unitwave.cli.summarise SET.unitwave-set.json --label locked
    python -m unitwave.cli.summarise --nwb A.nwb B.nwb ... --layout steinmetz_2019 \
        --name "Steinmetz six" --label responsive --event stim_on --trial-filter '{"included": true}'

A saved set holds IBL sessions. NWB files are named on the command line with their
layout; the task is the layout's unless --task says otherwise, and each file keeps its
layout's unit QC. A layout without brain regions is refused before anything is read.

Slow (every session of the set is loaded and tested), so it runs here rather than in
the page. The homepage reads the finished runs. Each run writes:
- manifest.json:
  - the git SHA, the label (with its task) and level;
  - the set (name, file, hash, sessions, trial filter), or for NWB files their
    paths and sha256, the layout and the task;
  - every config used, with its sha256;
  - the sessions used, and the sessions left out with why;
- units.parquet: every unit's region and label (analysis.summary.session_labels);
- regions.parquet and sessions.parquet: analysis.summary.region_summary, with CSV
  copies for reading by eye;
- flatmap.svg/.pdf and spread.svg/.pdf: the Swanson flatmap and the per-session
  spread (viz.summary_plots), drawn light. The page draws them in its own theme.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from unitwave.analysis.conditions import TrialFilter
from unitwave.analysis.summary import (
    DEFAULT_CONFIG,
    LabelSpec,
    SummaryConfig,
    load_summary_config,
    region_summary,
    session_labels,
)
from unitwave.analysis.tasks import task_path
from unitwave.cli.evaluate import REPO, _git, _jsonable, _sha256
from unitwave.data.load import load_data_config, load_session
from unitwave.data.manifest import MANIFEST_VERSION
from unitwave.nwb.intake import layout_path, load_layout
from unitwave.qc.nwb import load_nwb_qc_config
from unitwave.qc.spike_times import load_spike_qc_config
from unitwave.qc.units import load_qc_config
from unitwave.studio.plans import files_identity, set_status
from unitwave.studio.project import Source, load_source, qc_config_path
from unitwave.studio.sets import read_set
from unitwave.viz.studio_plots import save_vector
from unitwave.viz.summary_plots import build_region_flatmap, build_region_spread

SUMMARY_STEP = "regions_differ/summary"  # recipe 3's step a plan can name
CONFIG_FILES = {
    "summary": DEFAULT_CONFIG,
    "qc": REPO / "configs" / "qc.yaml",
    "analysis": REPO / "configs" / "analysis.yaml",
    "selectivity": REPO / "configs" / "selectivity.yaml",
    "movement": REPO / "configs" / "movement.yaml",
}


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text).strip("-").lower()[:40]


def write_results(out: Path, units, regions, sessions) -> None:
    """The run's tables (parquet, with CSV copies) and figures (SVG and PDF)."""
    units.to_parquet(out / "units.parquet")
    regions.to_parquet(out / "regions.parquet")
    sessions.to_parquet(out / "sessions.parquet")
    regions.to_csv(out / "regions.csv")
    sessions.to_csv(out / "sessions.csv", index=False)
    for name, fig in (
        ("flatmap", build_region_flatmap(regions, "light")),
        ("spread", build_region_spread(regions, sessions, "light")),
    ):
        for suffix in (".svg", ".pdf"):
            save_vector(fig, out / f"{name}{suffix}")


def run(
    set_path,
    label: LabelSpec,
    *,
    cfg: SummaryConfig | None = None,
    runs_dir: Path = REPO / "runs",
    load=None,
    progress=print,
    plans_dir: Path | None = None,
) -> Path:
    """Label every IBL session of a saved set, summarise by region, write the run; its
    folder. load(eid) -> Session (default: the BWM release)."""
    load = load or (lambda eid: load_session(eid, "bwm"))
    the_set, warnings = read_set(set_path, MANIFEST_VERSION)
    record = {
        "name": the_set["name"],
        "file": str(Path(set_path).resolve()),
        "hash": the_set["hash"],
        "eids": the_set["eids"],
        "trial_filter": the_set["trial_filter"],
        "warnings": warnings,
    }
    configs = {**CONFIG_FILES, "task": task_path(label.task)}
    sessions = {eid: (lambda eid=eid: (load(eid), load_qc_config())) for eid in the_set["eids"]}
    identities = [(eid, None) for eid in the_set["eids"]]  # the release is fixed
    return _run(
        record,
        sessions,
        label,
        configs,
        identities,
        cfg=cfg,
        runs_dir=runs_dir,
        progress=progress,
        plans_dir=plans_dir,
    )


def run_nwb(
    files,
    label: LabelSpec,
    *,
    layout: str,
    name: str,
    trial_filter: dict | None = None,
    cfg: SummaryConfig | None = None,
    runs_dir: Path = REPO / "runs",
    progress=print,
    plans_dir: Path | None = None,
) -> Path:
    """Label every NWB file under one layout, summarise by region, write the run; its
    folder. The label's task reads the trials; trial_filter defaults to the filters the
    task turns on. A layout without brain regions is refused before any file is read."""
    lay = load_layout(layout)
    if lay.location is not None and lay.location[0] == "refused":
        raise ValueError(
            f"the {lay.label} layout has no brain regions: {lay.location[1]['reason']}. "
            "Region summaries need them"
        )
    paths = [Path(f).expanduser().resolve() for f in files]
    absent = [str(p) for p in paths if not p.is_file()]
    if absent:
        raise ValueError(f"no such NWB file: {', '.join(absent)}")
    if len(set(paths)) != len(paths):
        raise ValueError("a file is named twice")
    task = label.definition()
    if trial_filter is None:
        trial_filter = {k: True for k, f in task.trial_filters.items() if f.default}
    trial_filter = TrialFilter.from_dict(trial_filter, task).to_dict()
    progress(f"hashing {len(paths)} files")
    hashes = {str(p): _sha256(p) for p in paths}
    content = {"files": hashes, "layout": lay.name, "task": task.name, "trial_filter": trial_filter}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"))
    record = {
        "name": name,
        "kind": "nwb",
        "files": [{"path": p, "sha256": h} for p, h in hashes.items()],
        "layout": lay.name,
        "layout_label": lay.label,
        "task": task.name,
        "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "eids": [f"nwb:{p}" for p in paths],  # as intake names them
        "trial_filter": trial_filter,
        "warnings": [],
    }
    qc = _nwb_qc(lay)
    configs = {
        **{k: v for k, v in CONFIG_FILES.items() if k != "qc"},
        "qc": qc_config_path(qc),
        "task": task_path(label.task),
        "layout": layout_path(layout),
    }

    def opener(path):
        source = Source(kind="nwb", file=str(path), layout=lay.name, task=label.task)
        return lambda: load_source(source)

    sessions = {f"nwb:{p}": opener(p) for p in paths}
    identities = [(f"nwb:{p}", files_identity({p.name: hashes[str(p)]})) for p in paths]
    return _run(
        record,
        sessions,
        label,
        configs,
        identities,
        cfg=cfg,
        runs_dir=runs_dir,
        progress=progress,
        plans_dir=plans_dir,
    )


def _nwb_qc(layout):
    """The unit QC rule a layout's files are judged by (as studio.project.load_source)."""
    return load_spike_qc_config() if layout.quality is None else load_nwb_qc_config(layout.quality)


def _run(
    record, sessions, label, configs, identities, *, cfg, runs_dir, progress, plans_dir=None
) -> Path:
    """sessions: eid -> a function returning (Session, unit QC). Each is labelled on the
    record's trial filter; one that can't be is left out, with why. identities: (eid,
    sha256 of its files) per session, as held-out plans name them (studio.plans)."""
    cfg = cfg or load_summary_config()
    started = datetime.now(UTC)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_summary_{label.kind}_{_slug(record['name'])}"
    out = Path(runs_dir) / run_id
    out.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "status": "running",
        "created": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git": _git(),
        "label": {"kind": label.kind, "event": label.event, "split": label.split},
        "task": label.definition().name,
        "label_text": label.describe(),
        "level": cfg.level,
        "summary_config": vars(cfg),
        "set": record,
        "configs": {
            name: {
                "path": str(path),
                "sha256": _sha256(Path(path)),
                "content": yaml.safe_load(Path(path).read_text()),
            }
            for name, path in configs.items()
        },
        "versions": {
            "python": platform.python_version(),
            **{m: sys.modules[m].__version__ for m in ("numpy", "pandas", "scipy")},
        },
        "sessions_used": [],
        "sessions_failed": {},
    }

    def save():
        (out / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=1))

    save()
    tables = []
    for i, (eid, open_session) in enumerate(sessions.items(), 1):
        start = time.time()
        short = eid[:8] if record.get("kind") != "nwb" else Path(eid).name
        try:
            session, qc = open_session()
            labels = session_labels(session, label, record["trial_filter"], level=cfg.level, qc=qc)
        except (ValueError, KeyError, OSError) as e:  # left out, and counted with why
            manifest["sessions_failed"][eid] = str(e)
            progress(f"{i}/{len(sessions)} {short}: left out ({e})")
        else:
            tables.append(labels)
            manifest["sessions_used"].append(eid)
            progress(f"{i}/{len(sessions)} {short}: {time.time() - start:.0f} s")
        save()
    if not tables:
        manifest["status"] = "failed: no session could be labelled"
        save()
        raise ValueError("no session of the set could be labelled")
    units = pd.concat(tables, ignore_index=True)
    # Confirmatory only as the first run of a plan naming exactly these sessions and
    # recipe 3's summary step (step 9b); claimed now that the run has its results.
    tf = TrialFilter.from_dict(record["trial_filter"], label.definition()).to_dict()
    manifest["plan_status"] = set_status(
        plans_dir or load_data_config().data_root / "plans",
        sessions=identities,
        step=SUMMARY_STEP,
        trial_filter=tf,
        label=(label.kind, label.event),
        at=started,
        claim=not manifest["sessions_failed"],
    )
    if manifest["sessions_failed"] and manifest["plan_status"]["status"] == "confirmatory":
        manifest["plan_status"] = {
            "status": "exploratory",
            "why": "sessions were left out, so the planned set wasn't run whole",
        }
    regions, sessions_table = region_summary(units, cfg)
    write_results(out, units, regions, sessions_table)
    manifest.update(
        status="complete",
        n_units=len(units),
        n_regions=len(regions),
        n_tested=int(regions["n_tests"].iloc[0]) if len(regions) else 0,
        n_claims=int(regions["claim"].sum()),
    )
    save()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("set", type=Path, nargs="?", help="a saved IBL set (*.unitwave-set.json)")
    ap.add_argument("--nwb", type=Path, nargs="+", help="NWB files instead of a set")
    ap.add_argument("--layout", default="", help="with --nwb: their layout, e.g. steinmetz_2019")
    ap.add_argument("--task", default="", help="with --nwb: the task (default: the layout's)")
    ap.add_argument("--name", default="", help="with --nwb: a name for the files, as a set's")
    ap.add_argument(
        "--trial-filter", default=None, help="with --nwb: JSON, e.g. '{\"included\": true}'"
    )
    ap.add_argument("--label", required=True, choices=("responsive", "selective", "locked"))
    ap.add_argument("--event", default="", help="for responsive and selective, e.g. stim_on")
    ap.add_argument("--split", default="", help="for selective: the condition, e.g. choice")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    args = ap.parse_args(argv)
    if (args.set is None) == (args.nwb is None):
        ap.error("give either a saved set or --nwb files")
    if args.set is not None:
        label = LabelSpec(args.label, args.event, args.split)
        out = run(args.set, label, runs_dir=args.runs_dir)
    else:
        if not args.layout or not args.name:
            ap.error("--nwb needs --layout and --name")
        task = args.task or load_layout(args.layout).task
        if not task:
            ap.error(f"the {args.layout} layout names no task: give --task")
        label = LabelSpec(args.label, args.event, args.split, task=task)
        tf = None if args.trial_filter is None else json.loads(args.trial_filter)
        out = run_nwb(
            args.nwb,
            label,
            layout=args.layout,
            name=args.name,
            trial_filter=tf,
            runs_dir=args.runs_dir,
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
