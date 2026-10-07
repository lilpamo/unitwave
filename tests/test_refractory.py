import numpy as np
import pandas as pd
import pytest

from unitwave.data.load import load_data_config
from unitwave.qc.refractory import (
    max_acceptable_violations,
    pairs_closer_than,
    sliding_rp_pass,
)


def test_pairs_closer_than_counts_later_spikes_by_hand():
    # Samples 0, 4, 22, 60: pair gaps 4, 22, 60, 18, 56, 38.
    samples = np.array([0, 4, 22, 60])
    assert pairs_closer_than(samples, 5) == 1  # 4
    assert pairs_closer_than(samples, 30) == 3  # 4, 22, 18
    assert pairs_closer_than(samples, 61) == 6
    # Identical samples are a pair with gap 0.
    assert pairs_closer_than(np.array([7, 7, 7]), 1) == 3


def test_max_acceptable_violations_by_hand():
    # Expected violations at 100% acceptable contamination: 1 * 0.002 * 2 * 10 * 100 = 4.
    # Poisson(4): P(<=1) = 0.092, P(<=2) = 0.238, so the 10% quantile is 2.
    assert max_acceptable_violations(10.0, 0.002, 100.0, 1.0, 0.1) == 2
    # Expected 0.1 * 0.002 * 2 * 1 * 1 = 0.0004: the 10% quantile is 0, so even zero
    # violations would be unsurprising at 10% contamination. IBL returns -1 there, so no
    # count can pass at that refractory period.
    assert max_acceptable_violations(1.0, 0.002, 1.0, 0.1, 0.1) == -1


def test_a_clean_train_passes_and_a_contaminated_one_fails():
    clean = np.arange(0, 100, 0.02)  # 50 Hz, every gap 20 ms
    assert sliding_rp_pass(clean, contamination=0.1, alpha=0.1)
    contaminated = np.sort(np.concatenate([clean, clean + 0.0005]))  # a spike 0.5 ms later
    assert not sliding_rp_pass(contaminated, contamination=0.1, alpha=0.1)


def test_too_few_spikes_fail_as_in_ibl():
    assert not sliding_rp_pass(np.array([1.0, 2.0, 3.0]), contamination=0.1, alpha=0.1)
    assert not sliding_rp_pass(np.array([]), contamination=0.1, alpha=0.1)
    assert not sliding_rp_pass(np.array([5.0]), contamination=0.1, alpha=0.1)


# Real data: IBL's stored flag for d23a44ef probe00 (revision 2024-05-06). IBL computed
# it with the newer, GPL slidingRP package; this is the older MIT algorithm, so the
# two can differ on borderline units (docs/DECISIONS.md records the agreement).
ALF = (
    load_data_config().one_cache_root
    / "danlab/Subjects/DY_016/2020-09-12/001/alf/probe00/pykilosort/#2024-05-06#"
)


@pytest.mark.skipif(not ALF.exists(), reason="ONE cache for d23a44ef not available")
def test_agrees_with_ibls_stored_flags_on_real_units():
    times = np.load(ALF / "spikes.times.npy")
    clusters = np.load(ALF / "spikes.clusters.npy")
    stored = pd.read_parquet(ALF / "clusters.metrics.pqt").set_index("cluster_id")
    order = np.argsort(clusters, kind="stable")
    ids, starts = np.unique(clusters[order], return_index=True)
    ends = np.append(starts[1:], order.size)
    ours = {
        int(c): sliding_rp_pass(times[order[a:b]], contamination=0.1, alpha=0.1)
        for c, a, b in zip(ids, starts, ends)
    }
    theirs = stored.loc[list(ours), "slidingRP_viol"].astype(bool)
    # Measured on 2026-09-30 (docs/DECISIONS.md): 553 of 674 agree. The port itself was
    # checked against ibllib's original code (73 of 73 sampled clusters identical); the
    # rest is the difference between IBL's two algorithms. A change here means the port
    # changed.
    assert int((pd.Series(ours) == theirs).sum()) == 553
    assert len(ours) == 674
