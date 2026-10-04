"""The analysis log (step 9b; docs/DECISIONS.md, "Step 9b design"): every test Studio
computes, written to the project file the moment it runs.

An entry records what was tested, not what came out (the project still holds no
results): when, the kind, the parameters, the size of the family it was corrected over
(`n_tests`), its units and trials, the recipe step that ran it, whether it is a repeat
of a test already in the log (`key`: the session fingerprint, kind, parameters and
configs), and whether it is confirmatory (studio.plans) or exploratory, with why.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def log_key(fingerprint: str, kind: str, params: dict, configs: dict) -> str:
    """sha256 identifying a test: the same key, the same numbers (every null is seeded
    or exact)."""
    text = json.dumps(
        {"session": fingerprint, "kind": kind, "params": params, "configs": configs},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_entry(
    log: list[dict],
    *,
    at: datetime,
    kind: str,
    what: str,
    params: dict,
    n_tests: int,
    n_units: int,
    n_trials: int | None,
    recipe: str | None,
    key: str,
    status: dict,
) -> dict:
    """One log entry; `repeat` when the key is already in the log."""
    return {
        "at": at.isoformat(),
        "kind": kind,
        "what": what,
        "params": params,
        "n_tests": int(n_tests),
        "n_units": int(n_units),
        "n_trials": None if n_trials is None else int(n_trials),
        "recipe": recipe,
        "key": key,
        "repeat": any(e["key"] == key for e in log),
        **status,
    }


def running_total(log: list[dict]) -> dict:
    """Runs logged; distinct tests and the hypotheses they tested (repeats counted
    once); repeats; confirmatory runs."""
    distinct = [e for e in log if not e["repeat"]]
    return {
        "n_runs": len(log),
        "n_tests": len(distinct),
        "n_hypotheses": sum(e["n_tests"] for e in distinct),
        "n_repeats": len(log) - len(distinct),
        "n_confirmatory": sum(e["status"] == "confirmatory" for e in log),
    }


def write_log(path: str | os.PathLike, log: list[dict], make_project) -> None:
    """Write the log into the project file at once: into its `analysis_log` when the
    file exists (its saved view untouched), else a new project made by make_project()."""
    from unitwave.studio.project import save_project

    path = Path(path)
    project = json.loads(path.read_text()) if path.exists() else make_project()
    project["analysis_log"] = log
    save_project(project, path)
