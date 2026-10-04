"""Held-out plans (step 9b; docs/DECISIONS.md, "Step 9b design"): the only way a result
is confirmatory.

A plan (`<data_root>/plans/<name>.unitwave-plan.json`, written by `python -m
unitwave.cli.plan`) names, before anything runs:
- the task, and the held-out sessions: an IBL eid, or an NWB or Phy source with the
  sha256 of its files (`source_identity`);
- the recipe steps to run on them ("recipe/step"), and the trial filter;
- `saved_at`, and a sha256 over all of it. A plan is never edited: a name already used
  is refused, and an edited file is refused with the reason.

The unit set is fixed: QC-passing units of every probe, as the recipes run by default.

Claims (`<name>.runs.jsonl` beside the plan, append-only): the first planned run of a
step on a session claims it. A later run, from any project, finds the claim and is
exploratory, saying when the planned run happened.
"""

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from unitwave.analysis.conditions import TrialFilter
from unitwave.analysis.recipes import load_recipe
from unitwave.analysis.tasks import load_task

PLAN_SUFFIX = ".unitwave-plan.json"
PLAN_VERSION = 1
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")
_CONTENT = ("version", "name", "task", "sessions", "steps", "trial_filter", "saved_at")


def _hash(content: dict) -> str:
    text = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def files_identity(hashes: dict) -> str:
    """The sha256 a plan records for a source's files: name -> sha256 of each
    (studio.project.file_hashes)."""
    return _hash(hashes)


def source_identity(source, session=None) -> tuple[str, str | None]:
    """(eid, sha256 of its files) as a plan names a session: an IBL session by its eid
    (the release is fixed), an NWB file or Phy folder also by its files' sha256. The eid
    is the loaded session's, or the one its loader gives (without loading it)."""
    from unitwave.studio.project import file_hashes

    if session is not None:
        eid = session.eid
    elif source.kind == "nwb":
        eid = f"nwb:{Path(source.file).resolve()}"
    elif source.kind == "phy":
        eid = f"phy:{Path(source.folder).resolve()}"
    else:
        eid = source.eid
    if source is None or source.kind == "ibl":
        return eid, None
    return eid, files_identity(file_hashes(source))


def save_plan(
    directory: str | os.PathLike, name: str, task: str, sessions, steps, trial_filter: dict
) -> Path:
    """Write a plan once; a name already used is refused. sessions: [{eid, sha256}]."""
    if not isinstance(name, str) or not _NAME.match(name) or ".." in name:
        raise ValueError("a plan name is 1-80 letters, digits, spaces, _ . or -")
    path = Path(directory) / f"{name}{PLAN_SUFFIX}"
    if path.exists():
        raise ValueError(f"a plan named {name!r} already exists; plans are never edited")
    definition = load_task(task)
    for step in steps:
        recipe, _, step_id = str(step).partition("/")
        if step_id not in {s.id for s in load_recipe(recipe).steps}:
            raise ValueError(f"recipe {recipe} has no step {step_id!r}")
    sessions = [{"eid": str(s["eid"]), "sha256": s.get("sha256")} for s in sessions]
    if not sessions or not steps:
        raise ValueError("a plan names at least one session and one step")
    content = {
        "version": PLAN_VERSION,
        "name": name,
        "task": definition.name,
        "sessions": sessions,
        "steps": sorted(set(steps)),
        "trial_filter": TrialFilter.from_dict(trial_filter, definition).to_dict(),
        "saved_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({**content, "hash": _hash(content)}, indent=1))
    tmp.replace(path)
    return path


def read_plan(path: str | os.PathLike) -> dict:
    """A plan, refused if its version is unknown or its hash doesn't match its content."""
    path = Path(path)
    raw = json.loads(path.read_text())
    if raw.get("version") != PLAN_VERSION:
        raise ValueError(f"{path.name}: plan version {raw.get('version')} is not supported")
    if _hash({k: raw.get(k) for k in _CONTENT}) != raw.get("hash"):
        raise ValueError(f"{path.name}: the hash doesn't match; the file was edited")
    return raw


def list_plans(directory: str | os.PathLike) -> list[dict]:
    """Every plan file: the plan, or {file, refused} for one that can't be trusted."""
    rows = []
    for path in sorted(Path(directory).glob(f"*{PLAN_SUFFIX}")):
        try:
            rows.append({"file": path.name, **read_plan(path)})
        except (ValueError, OSError, json.JSONDecodeError) as e:
            rows.append({"file": path.name, "refused": str(e)})
    return rows


def _claims_path(directory: Path, plan: dict) -> Path:
    return directory / f"{plan['name']}.runs.jsonl"


def _claim(directory: Path, plan: dict, eid: str, step: str) -> dict | None:
    path = _claims_path(directory, plan)
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        c = json.loads(line)
        if (c["eid"], c["step"], c["plan_hash"]) == (eid, step, plan["hash"]):
            return c
    return None


def plan_status(
    directory: str | os.PathLike,
    *,
    eid: str,
    sha256: str | None,
    step: str | None,
    trial_filter: dict,
    default_units: bool,
    at: datetime,
    claim: bool = False,
) -> dict:
    """{"status": "confirmatory", "plan", "plan_hash"} or {"status": "exploratory",
    "why"}. trial_filter: normalised (TrialFilter.to_dict). claim: write the claim when
    the run is confirmatory (the run is happening)."""
    if step is None:
        return {
            "status": "exploratory",
            "why": "a manual test: only planned recipe steps are confirmatory",
        }
    directory = Path(directory)
    plans = [p for p in list_plans(directory) if "refused" not in p]
    named = [p for p in plans if step in p["steps"] and any(s["eid"] == eid for s in p["sessions"])]
    if not named:
        return {"status": "exploratory", "why": "no held-out plan names this session and step"}
    whys = []
    for plan in named:
        name = plan["name"]
        held = next(s for s in plan["sessions"] if s["eid"] == eid)
        if held["sha256"] != sha256:
            whys.append(f"the data differ from plan '{name}' (the files' sha256)")
        elif at < datetime.fromisoformat(plan["saved_at"]):
            whys.append(f"this ran before plan '{name}' was saved")
        elif trial_filter != plan["trial_filter"]:
            whys.append(f"the trial filter differs from plan '{name}'s")
        elif not default_units:
            whys.append(f"plan '{name}' runs on QC-passing units of every probe")
        elif (c := _claim(directory, plan, eid, step)) is not None:
            whys.append(f"already run under plan '{name}' on {c['at'][:16].replace('T', ' ')} UTC")
        else:
            if claim:
                with _claims_path(directory, plan).open("a") as f:
                    record = {
                        "eid": eid,
                        "step": step,
                        "plan_hash": plan["hash"],
                        "at": at.isoformat(),
                    }
                    f.write(json.dumps(record) + "\n")
            return {"status": "confirmatory", "plan": name, "plan_hash": plan["hash"]}
    return {"status": "exploratory", "why": "; ".join(whys)}


def set_status(
    directory: str | os.PathLike,
    *,
    sessions: list[tuple[str, str | None]],
    step: str,
    trial_filter: dict,
    label: tuple[str, str],
    at: datetime,
    claim: bool = False,
) -> dict:
    """A region summary's status (recipe 3, step 2): confirmatory when a plan names
    exactly these sessions (eid, sha256) and this step, with this trial filter, the
    label the step runs (kind, event), saved before `at`, and nothing claimed yet."""
    recipe, _, step_id = step.partition("/")
    r = load_recipe(recipe)
    run = next(x.run for x in r.steps if x.id == step_id)
    if label != (run.get("label"), r.needs.get("stimulus")):
        return {
            "status": "exploratory",
            "why": f"recipe step {step} labels units {run.get('label')} at {r.needs.get('stimulus')}",
        }
    directory = Path(directory)
    plans = [p for p in list_plans(directory) if "refused" not in p and step in p["steps"]]
    named = [p for p in plans if {s["eid"] for s in p["sessions"]} == {e for e, _ in sessions}]
    if not named:
        return {
            "status": "exploratory",
            "why": "no held-out plan names exactly these sessions and this step",
        }
    key = "set:" + _hash(sorted(e for e, _ in sessions))
    whys = []
    for plan in named:
        name, held = plan["name"], {s["eid"]: s["sha256"] for s in plan["sessions"]}
        if any(held[e] != sha for e, sha in sessions):
            whys.append(f"the data differ from plan '{name}' (the files' sha256)")
        elif at < datetime.fromisoformat(plan["saved_at"]):
            whys.append(f"this ran before plan '{name}' was saved")
        elif trial_filter != plan["trial_filter"]:
            whys.append(f"the trial filter differs from plan '{name}'s")
        elif (c := _claim(directory, plan, key, step)) is not None:
            whys.append(f"already run under plan '{name}' on {c['at'][:16].replace('T', ' ')} UTC")
        else:
            if claim:
                with _claims_path(directory, plan).open("a") as f:
                    record = {
                        "eid": key,
                        "step": step,
                        "plan_hash": plan["hash"],
                        "at": at.isoformat(),
                    }
                    f.write(json.dumps(record) + "\n")
            return {"status": "confirmatory", "plan": name, "plan_hash": plan["hash"]}
    return {"status": "exploratory", "why": "; ".join(whys)}
