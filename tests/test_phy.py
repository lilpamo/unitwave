from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from phy_folder import write_phy_folder

from unitwave.analysis.units import unit_table
from unitwave.data.backends.phy import load_session_phy, read_params
from unitwave.data.load import load_data_config
from unitwave.qc.phy import PhyUnitQC, load_phy_qc_config, phy_unit_qc, refractory_passes
from unitwave.qc.units import task_firing_rates

FS = 30000.0
# Hand-built example, globally sorted like Kilosort's output.
SAMPLES = [30, 60, 90, 150, 30000, 45000, 60000, 90000]
CLUSTERS = [3, 7, 3, 7, 3, 11, 9, 7]
SPIKE_TEMPLATES = [0, 1, 0, 1, 1, 0, 0, 1]
# Template 0 peaks (largest peak-to-peak) on channel 2, template 1 on channel 3.
TEMPLATES = np.zeros((2, 3, 4))
TEMPLATES[0, :, 2] = [-5, 3, 0]
TEMPLATES[0, :, 0] = [1, -1, 0]
TEMPLATES[1, :, 3] = [-8, 2, 0]
POSITIONS = np.array([[11, 0], [43, 20], [11, 40], [43, 60]], float)
EVENTS = pd.DataFrame(
    {
        "intervals_0": [0.5, 1.5],
        "intervals_1": [1.4, 2.9],
        "stimOn_times": [0.6, 1.6],
        "feedback_times": [1.0, 2.0],
        "feedbackType": [1.0, -1.0],
    }
)


def _folder(tmp_path, **kwargs) -> Path:
    args = {
        "group": {3: "good", 7: "mua"},
        "ks_label": {3: "mua", 7: "good", 9: "good"},
        "templates": TEMPLATES,
        "spike_templates": SPIKE_TEMPLATES,
        "channel_positions": POSITIONS,
    }
    args.update(kwargs)
    return write_phy_folder(tmp_path / "imec0", SAMPLES, CLUSTERS, FS, **args)


def _events(tmp_path, table=EVENTS) -> Path:
    path = tmp_path / "events.csv"
    table.to_csv(path, index=False)
    return path


def test_samples_become_seconds_per_cluster(tmp_path):
    s = load_session_phy(_folder(tmp_path), _events(tmp_path))
    assert list(s.units.index) == ["imec0_3", "imec0_7", "imec0_9", "imec0_11"]
    np.testing.assert_allclose(s.spikes["imec0_3"], [0.001, 0.003, 1.0])
    np.testing.assert_allclose(s.spikes["imec0_7"], [0.002, 0.005, 3.0])
    np.testing.assert_allclose(s.spikes["imec0_9"], [2.0])
    np.testing.assert_allclose(s.spikes["imec0_11"], [1.5])
    assert s.time_bounds == (0.0, 3.0)
    assert (s.units["probe_name"] == "imec0").all()
    assert list(s.units["cluster_id"]) == [3, 7, 9, 11]


def test_group_comes_from_cluster_group_then_kslabel_with_its_file(tmp_path):
    units = load_session_phy(_folder(tmp_path), _events(tmp_path)).units
    assert units["phy_group"].tolist()[:3] == ["good", "mua", "good"]
    assert units["group_file"].tolist()[:3] == [
        "cluster_group.tsv",
        "cluster_group.tsv",
        "cluster_KSLabel.tsv",
    ]
    assert pd.isna(units.at["imec0_11", "phy_group"])
    assert pd.isna(units.at["imec0_11", "group_file"])


def test_depth_is_the_peak_channel_of_the_dominant_template(tmp_path):
    # Cluster 3's spikes use templates 0, 0, 1 -> template 0 -> channel 2 -> y = 40.
    s = load_session_phy(_folder(tmp_path), _events(tmp_path))
    assert s.units["depths"].tolist() == [40.0, 60.0, 40.0, 40.0]
    assert "units.depths" in s.available.present


def test_without_templates_depth_is_declared_missing(tmp_path):
    s = load_session_phy(_folder(tmp_path, templates=None), _events(tmp_path))
    assert "depths" not in s.units
    assert "templates.npy" in s.available.missing["units.depths"]


def test_fields_phy_does_not_have_are_declared_missing_with_reasons(tmp_path):
    s = load_session_phy(_folder(tmp_path), _events(tmp_path))
    for field in ("acronym", "x", "y", "z", "label", "firing_rate"):
        assert f"units.{field}" in s.available.missing
    assert "trials.firstMovement_times" in s.available.missing
    assert "trials.stimOn_times" in s.available.present
    assert not s.behaviour and all(k in s.available.missing for k in ["behaviour.wheel"])
    pd.testing.assert_frame_equal(s.trials[EVENTS.columns], EVENTS, check_dtype=False)


def test_refuses_an_unknown_event_column(tmp_path):
    with pytest.raises(ValueError, match="lick_times"):
        load_session_phy(_folder(tmp_path), _events(tmp_path, EVENTS.assign(lick_times=1.0)))


def test_refuses_events_without_trial_start_and_end(tmp_path):
    with pytest.raises(ValueError, match="intervals_0"):
        load_session_phy(_folder(tmp_path), _events(tmp_path, EVENTS.drop(columns="intervals_0")))


@pytest.mark.parametrize("bad", [-0.2, 3.5])
def test_refuses_events_outside_the_recorded_spikes(tmp_path, bad):
    events = EVENTS.copy()
    events.loc[1, "feedback_times"] = bad
    with pytest.raises(ValueError, match="clock"):
        load_session_phy(_folder(tmp_path), _events(tmp_path, events))


def test_refuses_spike_times_that_are_not_samples(tmp_path):
    folder = _folder(tmp_path)
    np.save(folder / "spike_times.npy", np.asarray(SAMPLES, float) / FS)
    with pytest.raises(ValueError, match="samples"):
        load_session_phy(folder, _events(tmp_path))


def test_params_py_is_read_but_never_executed(tmp_path):
    marker = tmp_path / "executed"
    folder = _folder(tmp_path, params_extra=f"open({str(marker)!r}, 'w')\nsample_rate = 25000.0\n")
    assert read_params(folder)["sample_rate"] == 25000.0
    assert not marker.exists()
    (folder / "params.py").write_text("dtype = 'int16'\n")
    with pytest.raises(ValueError, match="sample_rate"):
        read_params(folder)


QC = PhyUnitQC(
    groups=("good",), min_firing_rate_hz=0.1, refractory_contamination=0.1, refractory_alpha=0.1
)


def test_phy_qc_passes_good_units_that_fire_during_the_task(tmp_path):
    # Task period [0.5, 2.9] s = 2.4 s. Cluster 3: 1 spike (1.0 s) -> 0.417 Hz, good.
    # Cluster 7: no spike inside -> 0 Hz, and mua. Cluster 9: 1 spike, good (KSLabel).
    # Cluster 11: 1 spike, but no label in either file. The refractory results are
    # given here (cluster 9 fails) to test the QC logic on its own.
    s = load_session_phy(_folder(tmp_path), _events(tmp_path))
    units = s.units.assign(
        task_firing_rate=task_firing_rates(s), sliding_rp_pass=[True, True, False, True]
    )
    np.testing.assert_allclose(units["task_firing_rate"], [1 / 2.4, 0, 1 / 2.4, 1 / 2.4])
    verdict = phy_unit_qc(units, QC)
    assert verdict["passed"].tolist() == [True, False, False, False]
    assert (
        verdict.at["imec0_7", "reason"]
        == "group mua not in ['good']; task firing rate 0 Hz < 0.1 Hz"
    )
    assert verdict.at["imec0_9", "reason"] == (
        "refractory violations: contamination below 0.1 not shown at 0.9 confidence"
    )
    assert verdict.at["imec0_11", "reason"] == "group missing"


def test_units_with_a_few_spikes_fail_the_refractory_test(tmp_path):
    # One to three spikes each: too few to show low contamination, as in IBL.
    s = load_session_phy(_folder(tmp_path), _events(tmp_path))
    assert not refractory_passes(s, QC).any()


def test_phy_qc_default_config():
    assert load_phy_qc_config() == QC


def test_unit_table_for_phy_data(tmp_path):
    s = load_session_phy(_folder(tmp_path), _events(tmp_path))
    table = unit_table(s, load_phy_qc_config())
    assert table["region"].isna().all()
    assert table["label"].tolist()[:3] == ["good", "mua", "good"]
    # Every hand-built unit has 1-3 spikes, so all fail the refractory test.
    assert not table["qc_passed"].any()
    assert (table["probe"] == "imec0").all() and table["lateral_um"].isna().all()


# Real data: IBL's own Kilosort output for d23a44ef (probe00, pinned revision), written
# into Phy's format. Checks the loader on real sizes and labels, not quirks of files
# Kilosort or Phy wrote themselves.
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
ALF = (
    load_data_config().one_cache_root
    / "danlab/Subjects/DY_016/2020-09-12/001/alf/probe00/pykilosort/#2024-05-06#"
)


@pytest.mark.skipif(not ALF.exists(), reason="ONE cache for d23a44ef not available")
def test_real_ibl_sorting_round_trips_through_the_phy_format(tmp_path):
    from unitwave.data.load import load_session

    times = np.load(ALF / "spikes.times.npy")
    clusters = np.load(ALF / "spikes.clusters.npy")
    metrics = pd.read_parquet(ALF / "clusters.metrics.pqt")
    folder = write_phy_folder(
        tmp_path / "probe00",
        np.round(times * FS),
        clusters,
        FS,
        ks_label=dict(zip(metrics["cluster_id"], metrics["ks2_label"])),
    )
    trials = load_session(EID, "bwm").trials
    events = trials[[c for c in trials if c in EVENTS.columns or c.endswith("_times")]]
    s = load_session_phy(folder, _events(tmp_path, events))

    ids = np.unique(clusters)
    assert s.n_units == ids.size
    assert s.units["cluster_id"].tolist() == ids.tolist()
    counts = np.bincount(clusters)[ids]
    assert [s.spikes[u].size for u in s.units.index] == counts.tolist()
    order = np.argsort(clusters, kind="stable")
    loaded = np.concatenate([s.spikes[u] for u in s.units.index])
    assert np.max(np.abs(loaded - times[order])) <= 0.5 / FS + 1e-9
    good = set(metrics.loc[metrics["ks2_label"] == "good", "cluster_id"])
    assert (s.units["phy_group"] == "good").sum() == len(good & set(ids))
