import json

import numpy as np
import pandas as pd
import pytest
from phy_folder import write_phy_folder

from unitwave.analysis.events import event_times
from unitwave.analysis.psth import psth
from unitwave.studio.project import (
    DEFAULT_VIEW,
    Source,
    load_source,
    make_project,
    open_project,
    save_project,
    saving_path,
    view_to_query,
)

SAMPLES = [30, 60, 90, 150, 30000, 45000, 60000, 90000]
CLUSTERS = [3, 7, 3, 7, 3, 11, 9, 7]
EVENTS = pd.DataFrame(
    {"intervals_0": [0.5, 1.5], "intervals_1": [1.4, 2.9], "stimOn_times": [0.6, 1.6]}
)
VIEW = {**DEFAULT_VIEW, "event": "stim_on", "t0": -0.5, "t1": 0.5, "bin": 0.1, "unit": "imec0_3"}


def _source(tmp_path) -> Source:
    folder = write_phy_folder(tmp_path / "imec0", SAMPLES, CLUSTERS, ks_label={3: "good"})
    EVENTS.to_csv(tmp_path / "events.csv", index=False)
    return Source(kind="phy", folder=str(folder), events=str(tmp_path / "events.csv"))


def _saved(tmp_path):
    source = _source(tmp_path)
    session, qc = load_source(source)
    path = tmp_path / "study.unitwave.json"
    save_project(make_project(source, session, qc, VIEW), path)
    return source, session, path


def test_a_project_holds_settings_and_hashes_never_results(tmp_path):
    _, _, path = _saved(tmp_path)
    saved = json.loads(path.read_text())
    assert set(saved) == {
        "version",
        "source",
        "files",
        "fingerprint",
        "configs",
        "view",
        "saved_with",
    }
    assert saved["view"] == VIEW
    assert set(saved["files"]) == {
        "params.py",
        "spike_times.npy",
        "spike_clusters.npy",
        "cluster_KSLabel.tsv",
        "events.csv",
    }
    text = path.read_text()
    assert "mean" not in text and "psth" not in text


def test_reopening_recomputes_identical_psths(tmp_path):
    _, session, path = _saved(tmp_path)
    project, reopened, _qc, warnings = open_project(path)
    assert warnings == []
    assert project["view"] == VIEW
    for unit in session.spikes:
        a = psth(session.spikes[unit], event_times(session.trials, "stim_on"), (-0.5, 0.5), 0.1)
        b = psth(reopened.spikes[unit], event_times(reopened.trials, "stim_on"), (-0.5, 0.5), 0.1)
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.sem, b.sem)


def test_a_changed_spike_file_is_named_in_a_warning(tmp_path):
    _, _, path = _saved(tmp_path)
    np.save(tmp_path / "imec0" / "spike_times.npy", np.array(SAMPLES[:-1] + [89999], np.uint64))
    _, _, _, warnings = open_project(path)
    assert any("spike_times.npy changed" in w for w in warnings)
    assert any("data differ" in w for w in warnings)


def test_new_and_removed_files_are_named(tmp_path):
    _, _, path = _saved(tmp_path)
    (tmp_path / "imec0" / "cluster_KSLabel.tsv").unlink()
    pd.DataFrame({"cluster_id": [3], "group": ["good"]}).to_csv(
        tmp_path / "imec0" / "cluster_group.tsv", sep="\t", index=False
    )
    _, _, _, warnings = open_project(path)
    assert any("cluster_KSLabel.tsv is gone" in w for w in warnings)
    assert any("cluster_group.tsv is new" in w for w in warnings)


def test_refuses_unknown_view_keys_and_newer_versions(tmp_path):
    source = _source(tmp_path)
    session, qc = load_source(source)
    with pytest.raises(ValueError, match="colour"):
        make_project(source, session, qc, {**VIEW, "colour": "red"})
    _, _, path = _saved(tmp_path)
    saved = json.loads(path.read_text())
    path.write_text(json.dumps({**saved, "version": 99}))
    with pytest.raises(ValueError, match="version 99"):
        open_project(path)


def test_older_projects_without_a_probe_open_on_all_probes(tmp_path):
    from unitwave.studio.server import Studio

    _, _, path = _saved(tmp_path)
    saved = json.loads(path.read_text())
    del saved["view"]["probe"]
    path.write_text(json.dumps(saved))
    project, session, qc, warnings = open_project(path)
    assert warnings == []
    assert Studio(session, qc, tmp_path, view=project["view"]).view["probe"] == ""


def test_trial_filters_round_trip_through_a_project(tmp_path):
    source = _source(tmp_path)
    session, qc = load_source(source)
    view = {**VIEW, "trials": {"outcomes": [1.0], "exclude_nogo": True}}
    path = tmp_path / "filtered.unitwave.json"
    save_project(make_project(source, session, qc, view), path)
    project, _, _, warnings = open_project(path)
    assert warnings == [] and project["view"]["trials"] == {"outcomes": [1.0], "exclude_nogo": True}


def test_every_config_behind_a_labelled_result_is_hashed(tmp_path):
    _, _, path = _saved(tmp_path)
    saved = json.loads(path.read_text())
    assert set(saved["configs"]) == {
        "qc",
        "spike_qc",
        "analysis",
        "selectivity",
        "movement",
        "trial_view",
        "correlograms",
        "trajectories",
    }
    assert saved["configs"]["movement"]["path"] == "configs/movement.yaml"
    assert saved["configs"]["spike_qc"]["path"] == "configs/qc_spikes.yaml"
    # Files saved before a config was recorded say so, rather than failing to open.
    del saved["configs"]["movement"]
    path.write_text(json.dumps(saved))
    _, _, _, warnings = open_project(path)
    assert warnings == ["configs/movement.yaml was not recorded when the project was saved"]


def test_the_trial_view_state_round_trips_through_a_project(tmp_path):
    source = _source(tmp_path)
    session, qc = load_source(source)
    state = {
        "trial": 1,
        "trial_align": "stimOn_times",
        "trial_pre_s": 0.25,
        "trial_post_s": 1.0,
        "trial_n": 3,
        "trial_all": True,
        "trial_traces": ["pupil_left"],
    }
    path = tmp_path / "trial.unitwave.json"
    save_project(make_project(source, session, qc, {**VIEW, **state}), path)
    project, _, _, warnings = open_project(path)
    assert warnings == []
    assert {k: project["view"][k] for k in state} == state
    q = view_to_query(project["view"])
    assert (q["trial"], q["trial_align"], q["trial_n"], q["trial_all"]) == (
        "1",
        "stimOn_times",
        "3",
        "1",
    )
    assert q["trial_traces"] == "pupil_left"


def test_exports_send_the_views_trial_filters_as_the_page_does():
    # Before this, the view's trial filters reached the server under the wrong key, so
    # exports were computed on every trial.
    trials = {"bwm_include": True, "exclude_nogo": True}
    q = view_to_query({**VIEW, "trials": trials})
    assert json.loads(q["tf"]) == trials and "trials" not in q


def test_an_old_ending_project_opens_and_saves_beside_itself(tmp_path):
    from unitwave.studio.server import Studio

    source, session, path = _saved(tmp_path)
    old = path.rename(tmp_path / "study.ndstudio.json")  # saved before the rename
    before = old.read_bytes()
    project, _, qc, warnings = open_project(old)
    assert warnings == []
    studio = Studio(session, qc, tmp_path, source, old, project["view"])
    assert studio.project_path == tmp_path / "study.unitwave.json"
    studio.save(project["view"])
    assert (tmp_path / "study.unitwave.json").exists()
    assert old.read_bytes() == before  # the old file is kept as it was
    # Opening the old file again never overwrites the new one: the next free name.
    assert saving_path(old) == tmp_path / "study-2.unitwave.json"
    assert saving_path(tmp_path / "study.unitwave.json") == tmp_path / "study.unitwave.json"
    with pytest.raises(ValueError, match="written as"):
        save_project(project, old)
