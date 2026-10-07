"""Unit quality panel. Spike trains, folders and waveforms here are test inputs, never
shown as data."""

import numpy as np
import pandas as pd
import pytest

from unitwave.analysis.unit_quality import (
    QualityConfig,
    autocorrelogram,
    isi_histogram,
    load_quality_config,
    presence_ratio,
    rate_over_session,
)
from unitwave.data.cluster_files import (
    ibl_criteria,
    ibl_session_folder,
    ibl_sorting_folder,
    ibl_waveform,
    phy_waveform,
)
from unitwave.data.load import load_data_config
from unitwave.qc.refractory import sliding_rp_details, sliding_rp_pass


def test_default_config():
    assert load_quality_config() == QualityConfig(0.05, 0.0005, 0.05, 0.001, 60.0, 10.0)


def test_isi_histogram_by_hand():
    # Intervals 1.2, 3.8, 0.1 and 94.9 ms.
    t = np.array([0.0, 0.0012, 0.0050, 0.0051, 0.1])
    edges, counts, beyond = isi_histogram(t, max_s=0.005, bin_s=0.001)
    np.testing.assert_allclose(edges, [0, 0.001, 0.002, 0.003, 0.004, 0.005])
    assert counts.tolist() == [1, 1, 0, 1, 0] and beyond == 1


def test_autocorrelogram_by_hand():
    # Lags between different spikes: 2, 3 and 1 ms (10, 8 and 7 ms fall outside +-5 ms),
    # each counted once per order, so the counts are symmetric with nothing at zero.
    lags, counts = autocorrelogram(np.array([0.0, 0.002, 0.003, 0.010]), 0.005, 0.001)
    np.testing.assert_allclose(lags, np.arange(-5, 6) * 0.001)
    assert counts.tolist() == [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    # Two spikes at the same time are a pair at zero lag (in both orders); a spike is
    # never paired with itself.
    _, same = autocorrelogram(np.array([1.0, 1.0]), 0.005, 0.001)
    assert same[5] == 2 and same.sum() == 2
    # The outer bins are [-5.5, -4.5) and [4.5, 5.5) ms; nothing overflows past them.
    _, inside = autocorrelogram(np.array([0.0, 0.0054]), 0.005, 0.001)
    assert inside[[0, -1]].tolist() == [1, 1] and inside.sum() == 2
    _, outside = autocorrelogram(np.array([0.0, 0.0056]), 0.005, 0.001)
    assert outside.sum() == 0


def test_presence_ratio_by_hand_with_ibls_bins():
    spikes = np.array([0.5, 12.0, 25.0])
    # IBL's bins: np.arange(start, end + window / 2, window), each [edge, edge + window).
    ratio, counts = presence_ratio(spikes, 0.0, 25.0, 10.0)
    assert ratio == 1.0 and counts.tolist() == [1, 1, 1]
    # Once the end is past the middle of a last 10 s bin, IBL counts that bin too, even
    # though no spike can fall in it: 3 of 4.
    ratio, counts = presence_ratio(np.append(spikes, 25.1), 0.0, 25.1, 10.0)
    assert ratio == 0.75 and counts.tolist() == [1, 1, 2, 0]


def test_rate_over_session_by_hand():
    # Bins [0, 1), [1, 2), [2, 2.5]: the last is half a second, so 1 spike is 2 Hz.
    centres, rate = rate_over_session(np.array([0.5, 1.5, 1.6, 2.5]), 0.0, 2.5, 1.0)
    np.testing.assert_allclose(centres, [0.5, 1.5, 2.25])
    np.testing.assert_allclose(rate, [1.0, 2.0, 2.0])


def test_refractory_details_by_hand():
    clean = np.arange(0, 100, 0.02)  # 50 Hz, every gap 20 ms
    d = sliding_rp_details(clean, contamination=0.1, alpha=0.1)
    assert d.passed and d.rp_s.size == 15 and (d.violations == 0).all()
    np.testing.assert_allclose(d.rp_s[[0, -1]], [0.00125 + 1e-6, 0.01 + 1e-6])
    assert d.first_pass_rp_s == d.rp_s[np.flatnonzero(d.violations <= d.max_acceptable)[0]]
    contaminated = np.sort(np.concatenate([clean, clean + 0.0005]))
    bad = sliding_rp_details(contaminated, contamination=0.1, alpha=0.1)
    assert not bad.passed and bad.first_pass_rp_s is None
    empty = sliding_rp_details(np.array([5.0]), contamination=0.1, alpha=0.1)
    assert not empty.passed and empty.rp_s.size == 0 and empty.why


# Real data: d23a44ef probe00 in the ONE cache (IBL's pykilosort, revision 2024-05-06).
ALF = (
    load_data_config().one_cache_root
    / "danlab/Subjects/DY_016/2020-09-12/001/alf/probe00/pykilosort/#2024-05-06#"
)
needs_alf = pytest.mark.skipif(not ALF.exists(), reason="ONE cache for d23a44ef not available")


def _clusters():
    times = np.load(ALF / "spikes.times.npy")
    clusters = np.load(ALF / "spikes.clusters.npy")
    order = np.argsort(clusters, kind="stable")
    ids, starts = np.unique(clusters[order], return_index=True)
    ends = np.append(starts[1:], order.size)
    return times, {int(c): times[order[a:b]] for c, a, b in zip(ids, starts, ends)}


@needs_alf
def test_the_panels_refractory_details_reproduce_the_qc_verdict_on_every_cluster():
    _, by_cluster = _clusters()
    for c, t in by_cluster.items():
        assert sliding_rp_details(t, 0.1, 0.1).passed == sliding_rp_pass(t, 0.1, 0.1), c


@needs_alf
def test_presence_ratio_equals_ibls_stored_metric_on_every_cluster():
    times, by_cluster = _clusters()
    stored = pd.read_parquet(ALF / "clusters.metrics.pqt").set_index("cluster_id")
    ours = pd.Series(
        {c: presence_ratio(t, times[0], times[-1], 10.0)[0] for c, t in by_cluster.items()}
    )
    np.testing.assert_array_equal(
        ours.to_numpy(), stored.loc[ours.index, "presence_ratio"].to_numpy()
    )


@needs_alf
def test_ibl_criteria_reproduce_ibls_label_and_waveforms_load():
    crit = ibl_criteria(ALF)
    stored = pd.read_parquet(ALF / "clusters.metrics.pqt").set_index("cluster_id")["label"]
    passed = crit[["refractory_pass", "noise_cutoff_pass", "amplitude_pass"]].mean(axis=1)
    np.testing.assert_allclose(passed.to_numpy(), stored.loc[crit.index].to_numpy())
    w = ibl_waveform(ALF, 3, sample_rate=30000.0)
    assert w.samples_uv.shape == (82, 32) and w.channels.shape == (32,)
    assert w.unit == "µV" and np.ptp(w.samples_uv[:, w.peak]) > 0


def test_ibl_folders_are_found_from_the_manifest(tmp_path):
    sessions = pd.DataFrame(  # the manifest's columns
        {
            "eid": ["eid-1"],
            "lab": ["danlab"],
            "subject": ["DY_016"],
            "date": ["2020-09-12"],
            "session_number": [1],
        }
    )
    alf = tmp_path / "danlab/Subjects/DY_016/2020-09-12/001/alf"
    assert ibl_session_folder(tmp_path, sessions, "eid-1") is None  # not downloaded
    (alf / "probe00/pykilosort/#2024-05-06#").mkdir(parents=True)
    assert ibl_session_folder(tmp_path, sessions, "eid-1") == alf
    assert ibl_session_folder(tmp_path, sessions, "other") is None
    assert ibl_sorting_folder(alf, "probe00") == alf / "probe00/pykilosort/#2024-05-06#"
    assert ibl_sorting_folder(alf, "probe01") is None


def test_phy_waveform_is_the_spike_weighted_unwhitened_template(tmp_path):
    from phy_folder import write_phy_folder

    templates = np.zeros((2, 3, 2))  # (n_templates, n_samples, n_channels), whitened
    templates[0, 1] = [1.0, 0.0]
    templates[1, 1] = [0.0, 1.0]
    folder = write_phy_folder(
        tmp_path / "imec0",
        [10, 20, 30, 40],
        [5, 5, 5, 6],
        templates=templates,
        spike_templates=np.array([0, 0, 1, 1]),
        channel_positions=np.array([[0.0, 0.0], [0.0, 20.0]]),
    )
    with pytest.raises(ValueError, match="whitening_mat_inv.npy"):
        phy_waveform(folder, 5)
    np.save(folder / "whitening_mat_inv.npy", np.array([[2.0, 0.0], [0.0, 3.0]]))
    w = phy_waveform(folder, 5)
    # Cluster 5: two spikes of template 0, one of template 1; unwhitened by W^-1.
    np.testing.assert_allclose(w.samples_uv[1], [2 * 2 / 3, 3 * 1 / 3])
    assert w.unit == "template units" and w.peak == 0
