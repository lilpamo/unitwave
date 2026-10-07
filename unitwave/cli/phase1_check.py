"""ROADMAP Phase 1 success check.

"load_session works for 50 sessions across >=10 subjects and >=3 labs, from cache,
in under 5 s each." Run with `python -m unitwave.cli.phase1_check`. It writes
runs/<UTC time>_phase1_check/report.json and exits non-zero if any criterion fails.
"""

import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from unitwave.data.load import load_data_config, load_session
from unitwave.data.manifest import build_manifest
from unitwave.data.session import Session

N_SESSIONS = 50
MIN_SUBJECTS = 10
MIN_LABS = 3
BUDGET_S = 5.0
REPO_ROOT = Path(__file__).resolve().parents[2]


def select_sessions(
    sessions: pd.DataFrame,
    n: int = N_SESSIONS,
    min_subjects: int = MIN_SUBJECTS,
    min_labs: int = MIN_LABS,
) -> pd.DataFrame:
    """Each subject's earliest session, taking labs in turn, until n. No randomness."""
    earliest = sessions.sort_values(["subject", "date", "session_number", "eid"]).drop_duplicates(
        "subject"
    )
    earliest = earliest.sort_values(["lab", "subject"])
    earliest = earliest.assign(turn=earliest.groupby("lab").cumcount())
    chosen = earliest.sort_values(["turn", "lab", "subject"]).head(n).drop(columns="turn")
    if chosen["subject"].nunique() < min_subjects or len(chosen) < n:
        raise ValueError(
            f"only {chosen['subject'].nunique()} subjects available; need {max(n, min_subjects)}"
        )
    if chosen["lab"].nunique() < min_labs:
        raise ValueError(f"only {chosen['lab'].nunique()} labs available; need {min_labs}")
    return chosen.reset_index(drop=True)


def _stats(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "max": float(v.max()),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "total": float(v.sum()),
    }


def run_check(
    selected: pd.DataFrame,
    load: Callable[[str], Session],
    budget_s: float = BUDGET_S,
    min_subjects: int = MIN_SUBJECTS,
    min_labs: int = MIN_LABS,
) -> dict:
    """Load every session twice (the second read from the cache) and check the criteria.

    Counts are checked against the manifest: units against its good-unit count and
    trials against its trial count.
    """
    first = {}
    for eid in selected["eid"]:
        start = time.perf_counter()
        load(eid)
        first[eid] = time.perf_counter() - start

    rows = []
    for rec in selected.itertuples(index=False):
        start = time.perf_counter()
        session = load(rec.eid)
        cached = time.perf_counter() - start
        rows.append(
            {
                "eid": rec.eid,
                "subject": rec.subject,
                "lab": rec.lab,
                "first_load_s": first[rec.eid],
                "cached_load_s": cached,
                "n_units": int(session.n_units),
                "n_trials": int(session.n_trials),
                "counts_match": bool(
                    session.eid == rec.eid
                    and session.n_units == rec.n_good_units
                    and session.n_trials == rec.n_trials
                ),
            }
        )

    failures = []
    over = [r["eid"] for r in rows if r["cached_load_s"] > budget_s]
    if over:
        failures.append(f"{len(over)} cached reads over budget ({budget_s} s): {over[:5]}")
    wrong = [r["eid"] for r in rows if not r["counts_match"]]
    if wrong:
        failures.append(f"{len(wrong)} sessions disagree with the manifest's counts: {wrong[:5]}")
    n_subjects, n_labs = selected["subject"].nunique(), selected["lab"].nunique()
    if n_subjects < min_subjects:
        failures.append(f"{n_subjects} subjects, fewer than {min_subjects}")
    if n_labs < min_labs:
        failures.append(f"{n_labs} labs, fewer than {min_labs}")

    return {
        "criteria": {
            "n_sessions": int(len(selected)),
            "min_subjects": min_subjects,
            "min_labs": min_labs,
            "cached_budget_s": budget_s,
        },
        "summary": {
            "n_sessions": len(rows),
            "n_subjects": int(n_subjects),
            "n_labs": int(n_labs),
            "first_load_s": _stats([r["first_load_s"] for r in rows]),
            "cached_load_s": _stats([r["cached_load_s"] for r in rows]),
        },
        "sessions": rows,
        "failures": failures,
        "passed": not failures,
    }


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout.strip()


def main() -> int:
    config = load_data_config()
    manifest = build_manifest(config.bwm_ephys_root, config.bwm_behavior_root)
    selected = select_sessions(manifest.sessions)
    report = run_check(selected, load=lambda eid: load_session(eid, "bwm", config=config))
    report["provenance"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "backend": "bwm",
        "cache_root": str(config.cache_root),
        "manifest": manifest.provenance,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / "runs" / f"{stamp}_phase1_check"
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))

    s = report["summary"]
    print(f"{s['n_sessions']} sessions, {s['n_subjects']} subjects, {s['n_labs']} labs")
    print(
        f"first load: median {s['first_load_s']['median']:.2f} s, max {s['first_load_s']['max']:.2f} s"
    )
    print(
        f"cached:     median {s['cached_load_s']['median']:.2f} s, max {s['cached_load_s']['max']:.2f} s"
    )
    print("PASSED" if report["passed"] else "FAILED:\n  " + "\n  ".join(report["failures"]))
    print(f"report: {out / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
