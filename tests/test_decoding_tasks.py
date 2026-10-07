"""Step 8a: decoding for any task. Preprocessing applies the session's own unit QC (IBL's
unchanged), trial targets and their no-spikes baseline come from the task definition,
and each task declares its shuffle null's minimum shift. Trials and spikes here are
test inputs, except the last test, which decodes the downloaded Steinmetz session."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from unitwave.analysis.tasks import TASKS_DIR, load_task, task_from_dict
from unitwave.preprocess.binning import PreprocConfig, load_preproc_config, preprocess_session
from unitwave.qc.spike_times import load_spike_qc_config
from unitwave.targets.task import task_trial_target, task_trialstruct_features

IBL_FINGERPRINT = "317be1f821d84f34a0fb456dbc6abe5c2973037f080ab74800a917a9a5ae6c46"


def test_ibls_preprocessing_fingerprint_is_unchanged():
    assert load_preproc_config().fingerprint() == IBL_FINGERPRINT


def _session():
    from test_qc_spike_times import _session as hand_built
    from test_qc_spike_times import _trains

    trains = _trains()
    session = hand_built(trains, 100.0)
    trials = pd.DataFrame(
        {
            "intervals_0": 2.0 + 4.0 * np.arange(24),
            "intervals_1": 5.0 + 4.0 * np.arange(24),
            "stim_time": 2.5 + 4.0 * np.arange(24),
            "response_time": 3.0 + 4.0 * np.arange(24),
            "contrast": np.tile([0.0, 0.5, 1.0], 8),
            "response_choice": np.tile([1.0, -1.0, 0.0, 1.0], 6),
            "feedback_type": np.tile([1.0, -1.0], 12),
            "engaged": np.r_[np.ones(20), np.zeros(4)],
        }
    )
    return session.__class__(**{**session.__dict__, "trials": trials}), trains


TASK = {
    "name": "decode_test",
    "label": "Decoding test",
    "required_columns": ["intervals_0", "intervals_1"],
    "events": {
        "stim_on": {"label": "Stimulus onset", "column": "stim_time"},
        "response": {"label": "Response", "column": "response_time"},
    },
    "conditions": {
        "choice": {
            "label": "Choice",
            "type": "categorical",
            "column": "response_choice",
            "level_names": {-1: "right", 0: "no-go", 1: "left"},
            "excluded": "trials without a choice",
        },
    },
    "trialstruct": ["contrast", "response_choice", "feedback_type"],
    "trial_filters": {
        "engaged": {
            "label": "Engaged",
            "kind": "flag",
            "column": "engaged",
            "reason": "not engaged",
        }
    },
    "decoding": {
        "min_shift_trials": 3,
        "targets": {
            "choice": {
                "label": "Choice (left vs right)",
                "condition": "choice",
                "levels": [-1, 1],
                "window": {"event": "response", "start_s": -0.1, "stop_s": 0.0},
                "trial_filters": ["engaged"],
                "after": ["feedback_type"],
            }
        },
    },
}


def test_preprocessing_applies_the_sessions_own_qc_rule():
    session, _ = _session()
    spike_qc = load_spike_qc_config()
    binned = preprocess_session(session, PreprocConfig(bin_ms=20, qc=spike_qc))
    assert list(binned.unit_ids) == [
        "clean"
    ]  # the spike-time verdicts of tests/test_qc_spike_times
    assert binned.fingerprint != IBL_FINGERPRINT
    assert PreprocConfig(bin_ms=20, qc=spike_qc).fingerprint() == binned.fingerprint


def test_a_task_target_by_hand():
    session, _ = _session()
    task = task_from_dict(TASK)
    binned = preprocess_session(session, PreprocConfig(bin_ms=20, qc=load_spike_qc_config()))
    target = task_trial_target(session, binned, task, "choice")
    # Choices -1 or 1 (no-go excluded), engaged only (the last 4 trials aren't).
    expected = [i for i in range(20) if [1, -1, 0, 1][i % 4] != 0]
    assert target.table["trial"].tolist() == expected
    assert target.table["label"].tolist() == [
        1 if [1, -1, 0, 1][i % 4] == 1 else 0 for i in expected
    ]
    # Window: the 100 ms before the response; the last bin finishing by the response.
    np.testing.assert_array_equal(
        target.table["end_bin"], np.floor((3.0 + 4.0 * np.array(expected)) * 50).astype(int) - 1
    )
    assert target.context_bins == 5 and target.name == "task:choice"
    # The task's trial filters first, then the label's levels.
    assert target.excluded == {"not engaged": 4, "choice not right or left": 5}


def test_the_no_spikes_baseline_sees_the_present_before_the_window_and_the_past():
    session, _ = _session()
    task = task_from_dict(TASK)
    binned = preprocess_session(session, PreprocConfig(bin_ms=20, qc=load_spike_qc_config()))
    target = task_trial_target(session, binned, task, "choice")
    features, names = task_trialstruct_features(session.trials, target, task, history_trials=2)
    # Now: contrast only (the choice is the target; feedback comes after the window).
    assert names[0] == "contrast" and "response_choice" not in names[:1]
    assert "feedback_type" not in names[: names.index("contrast_lag1")]
    assert names[1:] == [
        "contrast_lag1",
        "contrast_lag2",
        "response_choice_lag1",
        "response_choice_lag2",
        "feedback_type_lag1",
        "feedback_type_lag2",
    ]
    row = target.table["trial"].tolist().index(5)
    contrast = session.trials["contrast"].to_numpy()
    choice = session.trials["response_choice"].to_numpy()
    assert features[row, 0] == contrast[5]
    assert features[row, names.index("response_choice_lag1")] == choice[4]
    assert features[0, names.index("feedback_type_lag2")] == 0.0  # before the first trial


def test_decoding_sections_are_validated():
    def broken(change):
        raw = copy.deepcopy(TASK)
        change(raw["decoding"])
        return raw

    cases = [
        (lambda d: d.pop("min_shift_trials"), "min_shift_trials"),
        (lambda d: d["targets"]["choice"].update(condition="mood"), "no condition 'mood'"),
        (lambda d: d["targets"]["choice"].update(levels=[-1, 0, 1]), "two levels"),
        (lambda d: d["targets"]["choice"]["window"].update(event="go"), "no event 'go'"),
        (lambda d: d["targets"]["choice"].update(after=["mood"]), "not in trialstruct"),
        (lambda d: d["targets"]["choice"].update(trial_filters=["mood"]), "no trial filter 'mood'"),
    ]
    for change, message in cases:
        with pytest.raises(ValueError, match=message):
            task_from_dict(broken(change))


def test_the_steinmetz_definition_declares_its_choice_target():
    task = load_task("steinmetz")
    d = task.decoding
    assert d.min_shift_trials == 20
    c = d.targets["choice"]
    assert (c.condition, c.levels, c.event) == ("choice", (-1.0, 1.0), "response")
    assert "feedback_type" in c.after
    assert yaml.safe_load((TASKS_DIR / "ibl.yaml").read_text()).get("decoding") is None


RICHARDS = (
    Path.home() / "data/neurodecoder/dandi/000017/sub-Richards/sub-Richards_ses-20171031T120000.nwb"
)


@pytest.mark.skipif(not RICHARDS.exists(), reason="the Steinmetz Richards file isn't downloaded")
def test_steinmetz_choice_decodes_with_the_full_contract(tmp_path):
    from dataclasses import replace

    from unitwave.analysis.decoding import decode, load_decoding_config
    from unitwave.studio.project import Source, load_source

    source = Source(kind="nwb", file=str(RICHARDS), layout="steinmetz_2019", task="steinmetz")
    session, qc = load_source(source)
    cfg = replace(load_decoding_config(), n_shifts=20, n_bootstrap=200)
    run = decode(
        session.eid,
        "task:choice",
        session=session,
        qc=qc,
        task=load_task("steinmetz"),
        cfg=cfg,
        runs_dir=tmp_path,
    )
    s = run.summary()
    rows = {r["row"] for r in s["rows"]}
    assert {"null_shuffle", "null_trialstruct", "baseline_ridge", "model", "ceiling_within"} <= rows
    assert "baseline_rrr" in s["not_applicable"]
    assert s["split"]["kind"] == "within_session" and s["split"]["gap_s"] == 2.0
    assert s["null"]["n_shifts_used"] == 20
    assert s["units"]["used"] > 0 and s["gate"]["line"].startswith("GATE")
    assert (run.folder / "split.json").exists()


@pytest.mark.skipif(not RICHARDS.exists(), reason="the Steinmetz Richards file isn't downloaded")
def test_studio_decodes_a_tasks_own_target_in_the_background(tmp_path, monkeypatch):
    import time
    from dataclasses import replace

    from unitwave.analysis import decoding
    from unitwave.data.load import load_data_config
    from unitwave.studio.project import Source, load_source
    from unitwave.studio.server import Studio

    source = Source(kind="nwb", file=str(RICHARDS), layout="steinmetz_2019", task="steinmetz")
    session, qc = load_source(source)
    studio = Studio(session, qc, load_data_config().data_root / "atlas", source)
    info = studio.session_json({})["decoding"]
    assert info["available"] is True
    assert [t["id"] for t in info["targets"]] == ["task:choice"]
    studio.decode_cfg = replace(studio.decode_cfg, n_shifts=20, n_bootstrap=200)
    real = decoding.decode  # the run goes to tmp, not runs/
    monkeypatch.setattr(
        "unitwave.studio.server.decode", lambda *a, **k: real(*a, runs_dir=tmp_path, **k)
    )
    with pytest.raises(ValueError, match="not a decoding target here"):
        studio.decode_start({"target": "choice", "query": {}}, manifest=None)
    studio.decode_start({"target": "task:choice", "query": {"all": "0"}}, manifest=None)
    for _ in range(600):
        status = studio.decode_status({})
        if status["state"] != "running":
            break
        time.sleep(0.5)
    assert status["state"] == "done", status
    assert status["summary"]["target"] == "task:choice"
    assert status["summary"]["split"]["kind"] == "within_session"
