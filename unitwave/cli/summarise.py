"""Region summaries across a session set, logged under runs/<run_id>/.  [S5, §5, §7]

    python -m unitwave.cli.summarise SET.unitwave-set.json --label responsive --event stim_on
    python -m unitwave.cli.summarise SET.unitwave-set.json --label selective --event stim_on --split choice
    python -m unitwave.cli.summarise SET.unitwave-set.json --label locked

Slow (every session of the set is loaded and tested), so it runs here rather than in
the page. The homepage reads the finished runs. Each run writes:
- manifest.json:
  - the git SHA, the label and level;
  - the set (name, file, hash, sessions, trial filter);
  - every config used, with its sha256;
  - the sessions used, and the sessions left out with why;
- units.parquet: every unit's region and label (analysis.summary.session_labels);
- regions.parquet and sessions.parquet: analysis.summary.region_summary, with CSV
  copies for reading by eye;
- flatmap.svg/.pdf and spread.svg/.pdf: the Swanson flatmap and the per-session
  spread (viz.summary_plots), drawn light. The page draws them in its own theme.
"""

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from unitwave.analysis.summary import (
    DEFAULT_CONFIG,
    LabelSpec,
    SummaryConfig,
    load_summary_config,
    region_summary,
    session_labels,
)
from unitwave.cli.evaluate import REPO, _git, _jsonable, _sha256
from unitwave.data.load import load_session
from unitwave.data.manifest import MANIFEST_VERSION
from unitwave.studio.sets import read_set
from unitwave.viz.studio_plots import save_vector
from unitwave.viz.summary_plots import build_region_flatmap, build_region_spread

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
) -> Path:
    """Label every session of the set, summarise by region, write the run; its folder."""
    cfg = cfg or load_summary_config()
    load = load or (lambda eid: load_session(eid, "bwm"))
    the_set, warnings = read_set(set_path, MANIFEST_VERSION)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_summary_{label.kind}_{_slug(the_set['name'])}"
    out = Path(runs_dir) / run_id
    out.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "status": "running",
        "created": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git": _git(),
        "label": {"kind": label.kind, "event": label.event, "split": label.split},
        "label_text": label.describe(),
        "level": cfg.level,
        "summary_config": vars(cfg),
        "set": {
            "name": the_set["name"],
            "file": str(Path(set_path).resolve()),
            "hash": the_set["hash"],
            "eids": the_set["eids"],
            "trial_filter": the_set["trial_filter"],
            "warnings": warnings,
        },
        "configs": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "content": yaml.safe_load(path.read_text()),
            }
            for name, path in CONFIG_FILES.items()
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
    for i, eid in enumerate(the_set["eids"], 1):
        start = time.time()
        try:
            labels = session_labels(load(eid), label, the_set["trial_filter"], level=cfg.level)
        except (ValueError, KeyError, OSError) as e:  # left out, and counted with why
            manifest["sessions_failed"][eid] = str(e)
            progress(f"{i}/{len(the_set['eids'])} {eid[:8]}: left out ({e})")
        else:
            tables.append(labels)
            manifest["sessions_used"].append(eid)
            progress(f"{i}/{len(the_set['eids'])} {eid[:8]}: {time.time() - start:.0f} s")
        save()
    if not tables:
        manifest["status"] = "failed: no session could be labelled"
        save()
        raise ValueError("no session of the set could be labelled")
    units = pd.concat(tables, ignore_index=True)
    regions, sessions = region_summary(units, cfg)
    write_results(out, units, regions, sessions)
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
    ap.add_argument("set", type=Path, help="a saved session set (*.unitwave-set.json)")
    ap.add_argument("--label", required=True, choices=("responsive", "selective", "locked"))
    ap.add_argument("--event", default="", help="for responsive and selective, e.g. stim_on")
    ap.add_argument("--split", default="", help="for selective: the condition, e.g. choice")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    args = ap.parse_args(argv)
    out = run(args.set, LabelSpec(args.label, args.event, args.split), runs_dir=args.runs_dir)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
