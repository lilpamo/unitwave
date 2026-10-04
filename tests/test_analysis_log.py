"""Step 9b: the analysis log and held-out plans (docs/DECISIONS.md, "Step 9b design").
Every test Studio computes is written to the project file at once, with how many
hypotheses it tested; an identical re-run is logged as a repeat and counted once. A
result is confirmatory only as the first run of a planned recipe step on a planned
session, after the plan was saved; everything else is exploratory, with the reason.
Sessions here are hand-built Phy folders (test inputs)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from phy_folder import write_phy_folder

from unitwave.analysis.conditions import TrialFilter
from unitwave.analysis.tasks import load_task
from unitwave.studio.analysis_log import running_total
from unitwave.studio.plans import list_plans, plan_status, read_plan, save_plan
from unitwave.studio.project import Source, load_source, open_project
from unitwave.studio.server import Studio

STEP = "stimulus_beyond_movement/respond"
EVERY_TRIAL = TrialFilter.from_dict({}, load_task("ibl")).to_dict()
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)


def test_a_plan_is_saved_once_and_never_edited(tmp_path):
    path = save_plan(tmp_path, "held out", "ibl", [{"eid": "e1", "sha256": None}], [STEP], {})
    plan = read_plan(path)
    assert plan["steps"] == [STEP] and plan["task"] == "ibl" and len(plan["hash"]) == 64
    assert plan["trial_filter"] == EVERY_TRIAL
    with pytest.raises(ValueError, match="already exists"):
        save_plan(tmp_path, "held out", "ibl", [{"eid": "e1", "sha256": None}], [STEP], {})
    with pytest.raises(ValueError, match="no step 'nope'"):
        save_plan(
            tmp_path,
            "other",
            "ibl",
            [{"eid": "e1", "sha256": None}],
            ["stimulus_beyond_movement/nope"],
            {},
        )
    raw = json.loads(path.read_text())
    raw["sessions"].append({"eid": "e2", "sha256": None})  # added after the fact
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="edited"):
        read_plan(path)
    assert list_plans(tmp_path) == [
        {"file": path.name, "refused": f"{path.name}: the hash doesn't match; the file was edited"}
    ]


def _status(plans, **change):
    args = {
        "eid": "e1",
        "sha256": None,
        "step": STEP,
        "trial_filter": EVERY_TRIAL,
        "default_units": True,
        "at": NOW,
    }
    return plan_status(plans, **{**args, **change})


def test_only_the_first_planned_run_is_confirmatory(tmp_path):
    save_plan(tmp_path, "held out", "ibl", [{"eid": "e1", "sha256": "abc"}], [STEP], {})
    later = datetime.now(UTC) + timedelta(seconds=1)
    s = _status(tmp_path, sha256="abc", at=later, claim=True)
    assert s["status"] == "confirmatory" and s["plan"] == "held out"
    again = _status(tmp_path, sha256="abc", at=later + timedelta(seconds=5), claim=True)
    assert again["status"] == "exploratory" and "already run under plan 'held out'" in again["why"]


def test_every_other_run_is_exploratory_with_the_reason(tmp_path):
    save_plan(tmp_path, "held out", "ibl", [{"eid": "e1", "sha256": "abc"}], [STEP], {})
    after = datetime.now(UTC) + timedelta(seconds=1)
    cases = [
        ({"step": None}, "a manual test"),
        ({"eid": "e9"}, "no held-out plan names this session and step"),
        (
            {"step": "stimulus_beyond_movement/locked"},
            "no held-out plan names this session and step",
        ),
        ({"sha256": "changed"}, "the data differ from plan 'held out'"),
        ({"trial_filter": {**EVERY_TRIAL, "bwm_include": True}}, "trial filter"),
        ({"default_units": False}, "QC-passing units of every probe"),
        ({"at": NOW - timedelta(days=1)}, "before plan 'held out' was saved"),
    ]
    for change, why in cases:
        s = _status(tmp_path, **{"sha256": "abc", "at": after, **change})
        assert s["status"] == "exploratory" and why in s["why"], (change, s)
    assert not (tmp_path / "held out.runs.jsonl").exists()  # none of them claimed anything


def test_running_totals_count_repeats_once():
    log = [
        {"n_tests": 10, "repeat": False, "status": "exploratory"},
        {"n_tests": 10, "repeat": True, "status": "exploratory"},
        {"n_tests": 3, "repeat": False, "status": "confirmatory"},
    ]
    assert running_total(log) == {
        "n_runs": 3,
        "n_tests": 2,
        "n_hypotheses": 13,
        "n_repeats": 1,
        "n_confirmatory": 1,
    }


# ---------- in Studio, on a hand-built Phy folder ----------
def _phy_source(tmp_path) -> Source:
    """40 trials over 202 s; units 3 and 9 labelled good fire evenly (10 and 8 Hz, so no
    refractory violations), unit 7 isn't labelled. Long enough for the shift null."""
    trains = {
        3: np.arange(0.013, 210.0, 0.1),
        9: np.arange(0.05, 210.0, 0.125),
        7: np.arange(0.1, 210.0, 0.2),
    }
    times = np.concatenate(list(trains.values()))
    clusters = np.concatenate([np.full(len(v), k) for k, v in trains.items()])
    order = np.argsort(times)
    folder = write_phy_folder(
        tmp_path / "imec0",
        np.round(times[order] * 30000),
        clusters[order],
        ks_label={3: "good", 9: "good"},
    )
    starts = 2.0 + 5.0 * np.arange(40)
    events = pd.DataFrame(
        {"intervals_0": starts, "intervals_1": starts + 4.0, "stimOn_times": starts + 0.5}
    )
    events.to_csv(tmp_path / "events.csv", index=False)
    return Source(kind="phy", folder=str(folder), events=str(tmp_path / "events.csv"), task="ibl")


def _studio(tmp_path, source, project=None, log=()) -> Studio:
    session, qc = load_source(source)
    studio = Studio(
        session,
        qc,
        tmp_path / "atlas",
        source,
        project or tmp_path / "p.unitwave.json",
        analysis_log=list(log),
    )
    studio.plans_dir = tmp_path / "plans"
    return studio


Q = {"event": "stim_on", "all": "1", "probe": "", "tf": "{}", "movement_free": "0"}


def test_a_test_is_written_to_the_project_at_once_and_holds_no_outcome(tmp_path):
    source = _phy_source(tmp_path)
    studio = _studio(tmp_path, source)
    assert not studio.project_path.exists()
    result = studio.test_json(Q)
    saved = json.loads(studio.project_path.read_text())  # created at the first test
    (entry,) = saved["analysis_log"]
    assert entry["kind"] == "responsiveness" and entry["n_tests"] == result["n_tests"] == 3
    assert entry["status"] == "exploratory" and "a manual test" in entry["why"]
    assert entry["params"]["event"] == "stim_on" and entry["repeat"] is False
    assert "n_responsive" not in entry and "responsive" not in json.dumps(entry["params"])
    assert result["status"] == "exploratory"
    studio.test_json(Q)  # reused from the page's cache: the same test, not logged again
    assert len(json.loads(studio.project_path.read_text())["analysis_log"]) == 1
    assert studio.session_json({})["log"] == {
        "n_runs": 1,
        "n_tests": 1,
        "n_hypotheses": 3,
        "n_repeats": 0,
        "n_confirmatory": 0,
    }
    # Save project keeps the log; reopening restores it, and an identical re-run is a repeat.
    studio.save({"event": "stim_on"})
    project, *_ = open_project(studio.project_path)
    again = _studio(tmp_path, source, studio.project_path, project["analysis_log"])
    again.test_json(Q)
    log = json.loads(studio.project_path.read_text())["analysis_log"]
    assert [e["repeat"] for e in log] == [False, True] and log[0]["key"] == log[1]["key"]
    assert running_total(log)["n_hypotheses"] == 3


def test_a_planned_recipe_step_is_confirmatory_once(tmp_path):
    from unitwave.studio.plans import source_identity

    source = _phy_source(tmp_path)
    studio = _studio(tmp_path, source)
    eid, sha = source_identity(source, studio.session)
    save_plan(tmp_path / "plans", "held out", "ibl", [{"eid": eid, "sha256": sha}], [STEP], {})
    page = {"all": "0", "probe": "", "tf": "{}"}
    out = studio.recipe_run({"recipe": "stimulus_beyond_movement", "step": "respond", **page})
    assert out["result"]["status"] == "confirmatory" and out["result"]["plan"] == "held out"
    entry = json.loads(studio.project_path.read_text())["analysis_log"][-1]
    assert entry["recipe"] == STEP and entry["status"] == "confirmatory"
    # Another project on the same session: the plan's run has happened.
    other = _studio(tmp_path, source, tmp_path / "q.unitwave.json")
    out = other.recipe_run({"recipe": "stimulus_beyond_movement", "step": "respond", **page})
    assert out["result"]["status"] == "exploratory"
    assert "already run under plan 'held out'" in out["result"]["why"]
    # The same step on other units is exploratory, and says why.
    third = _studio(tmp_path, source, tmp_path / "r.unitwave.json")
    out = third.recipe_run(
        {"recipe": "stimulus_beyond_movement", "step": "respond", **page, "all": "1"}
    )
    assert "QC-passing units of every probe" in out["result"]["why"]


def test_the_log_counts_every_kind_of_test(tmp_path):
    studio = _studio(tmp_path, _phy_source(tmp_path))
    studio.connections_json({"all": "1", "probe": ""})
    studio.test_json(Q)
    kinds = [e["kind"] for e in json.loads(studio.project_path.read_text())["analysis_log"]]
    assert kinds == ["connections", "responsiveness"]
    assert isinstance(
        pd.Timestamp(json.loads(studio.project_path.read_text())["analysis_log"][0]["at"]),
        pd.Timestamp,
    )


RICHARDS = (
    Path.home() / "data/neurodecoder/dandi/000017/sub-Richards/sub-Richards_ses-20171031T120000.nwb"
)


@pytest.mark.skipif(not RICHARDS.exists(), reason="the Steinmetz Richards file isn't downloaded")
def test_a_planned_region_summary_is_confirmatory_once(tmp_path):
    import dataclasses

    from unitwave.analysis.summary import LabelSpec, load_summary_config
    from unitwave.cli.summarise import run_nwb

    source = Source(kind="nwb", file=str(RICHARDS), layout="steinmetz_2019", task="steinmetz")
    from unitwave.studio.plans import source_identity

    eid, sha = source_identity(source)
    plans = tmp_path / "plans"
    save_plan(
        plans,
        "held out",
        "steinmetz",
        [{"eid": eid, "sha256": sha}],
        ["regions_differ/summary"],
        {"included": True},
    )
    label = LabelSpec("responsive", "stim_on", task="steinmetz")
    cfg = dataclasses.replace(load_summary_config(), min_sessions=1)
    kw = {
        "layout": "steinmetz_2019",
        "name": "one",
        "cfg": cfg,
        "progress": lambda _: None,
        "plans_dir": plans,
    }
    first = run_nwb(
        [RICHARDS], label, trial_filter={"included": True}, runs_dir=tmp_path / "a", **kw
    )
    status = json.loads((first / "manifest.json").read_text())["plan_status"]
    assert status["status"] == "confirmatory" and status["plan"] == "held out"
    second = run_nwb(
        [RICHARDS], label, trial_filter={"included": True}, runs_dir=tmp_path / "b", **kw
    )
    status = json.loads((second / "manifest.json").read_text())["plan_status"]
    assert (
        status["status"] == "exploratory" and "already run under plan 'held out'" in status["why"]
    )
    other = run_nwb([RICHARDS], label, trial_filter={}, runs_dir=tmp_path / "c", **kw)
    assert "trial filter" in json.loads((other / "manifest.json").read_text())["plan_status"]["why"]
