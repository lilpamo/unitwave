"""Region-summary runs for the homepage (S5): list them and read one by name.

The runs are written by `python -m unitwave.cli.summarise` (analysis.summary); this
module only reads them. A run is named by its folder under runs/, and a name that
isn't a summary run's (or tries to leave runs/) is refused.
"""

import json
import os
import re
from pathlib import Path

import pandas as pd

_NAME = re.compile(r"^\d{8}T\d{6,12}Z_summary_[a-z]+_[a-z0-9-]*$")


def list_summaries(runs_dir: str | os.PathLike) -> list[dict]:
    """Finished summary runs, newest first: run, label, set, level, created, sessions
    used and left out, regions tested, claims."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    rows = []
    for folder in runs_dir.iterdir():
        if not _NAME.match(folder.name) or not (folder / "manifest.json").is_file():
            continue
        m = json.loads((folder / "manifest.json").read_text())
        if m.get("status") != "complete":
            continue
        rows.append(
            {
                "run": folder.name,
                "label": m["label_text"],
                "set": m["set"]["name"],
                "level": m["level"],
                "created": m["created"],
                "n_sessions": len(m["sessions_used"]),
                "n_failed": len(m["sessions_failed"]),
                "n_tested": m["n_tested"],
                "n_claims": m["n_claims"],
            }
        )
    return sorted(rows, key=lambda r: r["run"], reverse=True)


def read_summary(runs_dir: str | os.PathLike, run: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """(manifest, regions, sessions) of a finished summary run, by its folder name."""
    folder = Path(runs_dir) / str(run)
    if not _NAME.match(str(run)) or not (folder / "manifest.json").is_file():
        raise ValueError(f"no region summary {run!r} in {Path(runs_dir).name}/")
    manifest = json.loads((folder / "manifest.json").read_text())
    return (
        manifest,
        pd.read_parquet(folder / "regions.parquet"),
        pd.read_parquet(folder / "sessions.parquet"),
    )
