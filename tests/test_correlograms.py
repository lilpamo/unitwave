"""Cross-correlograms and putative connections. Spike trains here are test inputs,
never shown as data."""

import time

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from unitwave.analysis.correlograms import (
    P_RESOLUTION,
    CorrelogramConfig,
    capped_sum_distribution,
    close_pairs,
    connections,
    cross_correlogram,
    jitter_expected_ccg,
    jitter_test,
    load_correlogram_config,
    window_pmfs,
)
from unitwave.data.cache import SessionCache
from unitwave.data.load import key_parts, load_data_config, load_session

CFG = CorrelogramConfig(0.025, 0.0005, 0.005, (0.001, 0.004), 30, 50.0)


def test_default_config():
    assert load_correlogram_config() == CFG


def test_cross_correlogram_by_hand():
    # Lags b - a within +-5 ms: 2 ms (from a = 0), -1 and +1 ms (from a = 10 ms).
    a = np.array([0.0, 0.010])
    b = np.array([0.002, 0.009, 0.011, 0.040])
    lags, counts = cross_correlogram(a, b, 0.005, 0.001)
    np.testing.assert_allclose(lags, np.arange(-5, 6) * 0.001)
    assert counts.tolist() == [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    # Swapping the units mirrors the correlogram.
    assert cross_correlogram(b, a, 0.005, 0.001)[1].tolist() == counts[::-1].tolist()


def test_expected_correlogram_under_jitter_by_hand():
    # b's spike at 1.2 ms is resampled uniformly in its window [0, 5) ms, so its lag
    # from a's spike at 0 is uniform on [0, 5) ms: half a bin at 0 and 5 ms, whole
    # bins at 1-4 ms, each 1/5 of the spike.
    expected = jitter_expected_ccg(np.array([0.0]), np.array([0.0012]), 0.005, 0.001, 0.005)
    np.testing.assert_allclose(expected, [0, 0, 0, 0, 0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def _poisson(rng, rate, duration, modulation=None):
    """Spike times on [0, duration); `modulation(t)` in [0, 1] thins a 2x rate."""
    t = np.sort(rng.uniform(0, duration, rng.poisson(rate * duration * (2 if modulation else 1))))
    if modulation:
        t = t[rng.random(t.size) < modulation(t)]
    return t


def test_the_exact_jitter_null_matches_monte_carlo_jitter():
    rng = np.random.default_rng(0)
    a, b = _poisson(rng, 20, 60), _poisson(rng, 20, 60)
    b = np.sort(np.concatenate([b, a[rng.random(a.size) < 0.1] + 0.002]))  # some coupling
    window = (0.001, 0.004)
    r = jitter_test(a, b, window, 0.005)
    starts = np.floor(b / 0.005) * 0.005
    draws = []
    for _ in range(4000):
        jittered = np.sort(starts + rng.uniform(0, 0.005, b.size))
        lo = np.searchsorted(jittered, a + window[0], side="left")
        hi = np.searchsorted(jittered, a + window[1], side="left")
        draws.append(int((hi - lo).sum()))
    draws = np.array(draws)
    assert abs(r.expected - draws.mean()) < 4 * draws.std() / np.sqrt(draws.size)
    mc = (draws >= r.observed).mean()
    assert abs(r.p_high - mc) < 4 * np.sqrt(max(mc, 1e-3) / draws.size) + 1e-3


def test_jitter_p_values_are_uniform_on_co_modulated_uncoupled_pairs():
    # A shared slow (1 Hz) rate modulation but no millisecond coupling: a shuffle null
    # would call these connected; the jitter null keeps the slow part and should not.
    rng = np.random.default_rng(1)

    def slow(t):
        return 0.5 + 0.5 * np.sin(2 * np.pi * t)

    p = []
    for _ in range(300):
        a, b = _poisson(rng, 15, 60, slow), _poisson(rng, 15, 60, slow)
        p.append(jitter_test(a, b, (0.001, 0.004), 0.005).p_high)
    assert kstest(p, "uniform").pvalue > 0.01
    assert (np.array(p) < 0.05).mean() <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / len(p))


def test_an_injected_2_ms_excitatory_coupling_is_detected_in_its_direction():
    rng = np.random.default_rng(2)
    a = _poisson(rng, 10, 300)
    b = _poisson(rng, 10, 300)
    follow = a[rng.random(a.size) < 0.3]
    b = np.sort(np.concatenate([b, follow + rng.normal(0.002, 0.0002, follow.size)]))
    units = pd.DataFrame(
        {"probe": ["p0", "p0"], "depth_um": [100.0, 400.0], "lateral_um": [0.0, 0.0]},
        index=["a", "b"],
    )
    result = connections({"a": a, "b": b}, ["a", "b"], units, CFG, alpha=0.05)
    assert result["n_tests"].iloc[0] == 2  # one pair, both directions
    ab = result.set_index(["pre", "post"])
    assert ab.loc[("a", "b"), "p"] < 1e-6 and ab.loc[("a", "b"), "connected"]
    # The jitter expectation smears a's real +2 ms peak into b -> a's window, so b -> a
    # shows a deficit. It is not labelled: only excess counts as a connection.
    ba = ab.loc[("b", "a")]
    assert ba["observed"] < ba["expected"] and ba["p"] > 0.5 and not ba["connected"]
    assert result["null"].iloc[0] == "interval jitter, 5 ms windows, exact"


def test_close_pairs_are_flagged_and_missing_positions_say_so():
    units = pd.DataFrame(
        {
            "probe": ["p0", "p0", "p0", "p1", "p0"],
            "depth_um": [100.0, 120.0, 600.0, 100.0, np.nan],
            "lateral_um": [16.0, 16.0, 16.0, 16.0, np.nan],
        },
        index=["a", "b", "c", "d", "e"],
    )
    close, why = close_pairs(units, [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e")], 50.0)
    assert close == [True, False, False, None]  # d is on another probe: never close
    assert why[3] == "no site position for e"


def test_connections_refuse_too_many_units():
    units = pd.DataFrame(
        {"probe": ["p0"] * 31, "depth_um": 0.0, "lateral_um": 0.0},
        index=[f"u{i}" for i in range(31)],
    )
    spikes = {u: np.array([1.0, 2.0]) for u in units.index}
    with pytest.raises(ValueError, match="at most 30 units"):
        connections(spikes, list(units.index), units, CFG)


def test_connections_table_equals_testing_each_direction_in_turn():
    # The tests run on threads (S1: 702 tests took 160 s in one 6a601cc5 region); each
    # is exact, so the table must be the same as testing the directions one by one.
    rng = np.random.default_rng(4)
    ids = [f"u{i}" for i in range(6)]
    spikes = {u: _poisson(rng, 15, 40) for u in ids}
    units = pd.DataFrame(
        {"probe": "p0", "depth_um": np.arange(6) * 100.0, "lateral_um": 0.0}, index=ids
    )
    table = connections(spikes, ids, units, CFG, alpha=0.05)
    assert table.shape[0] == 6 * 5
    for row in table.itertuples():
        r = jitter_test(spikes[row.pre], spikes[row.post], CFG.synaptic_window_s, CFG.jitter_s)
        assert (row.observed, row.expected, row.p) == (r.observed, r.expected, r.p_high)


def test_late_spikes_give_the_same_test_as_early_ones():
    # Jitter windows sit on a grid from time 0, so shifting both trains by a whole
    # number of windows must not change anything. Late in a session, floor(t / D)
    # misplaces boundaries by rounding; the test must not depend on it.
    rng = np.random.default_rng(3)
    a, b = _poisson(rng, 20, 30), _poisson(rng, 20, 30)
    early = jitter_test(a, b, (0.001, 0.004), 0.005)
    late = jitter_test(a + 3000.0, b + 3000.0, (0.001, 0.004), 0.005)
    assert late.observed == early.observed
    assert late.expected == pytest.approx(early.expected, rel=1e-9)
    assert late.p_high == pytest.approx(early.p_high, rel=1e-6)


def test_capped_sum_distribution_by_hand():
    # Two draws from (1/2, 1/2) and one from (0, 1/4, 3/4): the sum is 1 + Binomial(2, 1/2)
    # plus another Bernoulli(3/4), so P = (0, 1/16, 5/16, 7/16, 3/16) on 0..4.
    pmfs = np.array([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])
    full = capped_sum_distribution(pmfs, np.array([2, 1]), cap=4)
    np.testing.assert_allclose(full, [0, 1 / 16, 5 / 16, 7 / 16, 3 / 16], atol=1e-15)
    # Capped at 2: the last bin holds P(sum >= 2).
    capped = capped_sum_distribution(pmfs, np.array([2, 1]), cap=2)
    np.testing.assert_allclose(capped, [0, 1 / 16, 15 / 16], atol=1e-15)
    # No draws: the sum is 0.
    none = capped_sum_distribution(np.zeros((0, 3)), np.zeros(0, int), cap=3)
    np.testing.assert_array_equal(none, [1, 0, 0, 0])


# Real case (S1): session 6a601cc5, MRN. Step 8 convolved one jittered spike at a time,
# so a test cost (lags in the window) x (spikes): 58 s for probe00_98 -> probe00_117
# (515252 and 536596 spikes, 144142 lags), over 30 minutes for the region's 702 tests.
EID_6A60 = "6a601cc5-7b79-4c75-b0e8-552246532f82"
needs_6a60 = pytest.mark.skipif(
    not SessionCache(load_data_config().cache_root).path(key_parts(EID_6A60, "bwm")).exists(),
    reason="session 6a601cc5 is not in the session cache",
)


@pytest.fixture(scope="module")
def spikes_6a60():
    return load_session(EID_6A60, "bwm").spikes


def _sequential_reference(pmfs, copies, cap):
    """Step 8's computation: one convolution per jittered spike, the tail folded into
    the last bin after each."""
    dist = np.zeros(cap + 1)
    dist[0] = 1.0
    for pmf, n in zip(pmfs, copies):
        for _ in range(n):
            dist = np.convolve(dist, pmf)
            dist[cap] = dist[cap:].sum()
            dist = dist[: cap + 1]
    return dist


@needs_6a60
def test_real_pair_distribution_matches_the_one_spike_at_a_time_computation(spikes_6a60):
    a, b = spikes_6a60["probe00_129"], spikes_6a60["probe00_98"]
    observed, pmfs, copies = window_pmfs(a, b, CFG.synaptic_window_s, CFG.jitter_s)
    assert observed == 5811
    assert pmfs.ndim == 2 and copies.shape == (pmfs.shape[0],)
    np.testing.assert_allclose(pmfs.sum(axis=1), 1.0, atol=1e-12)
    cap = observed + 1
    fast = capped_sum_distribution(pmfs, copies, cap)
    slow = _sequential_reference(pmfs, copies, cap)
    assert fast.shape == slow.shape == (cap + 1,)
    np.testing.assert_allclose(fast, slow, rtol=0, atol=P_RESOLUTION / 10)
    r = jitter_test(a, b, CFG.synaptic_window_s, CFG.jitter_s)
    assert r.observed == observed
    assert r.p_high == pytest.approx(max(slow[observed:].sum(), P_RESOLUTION), abs=P_RESOLUTION)
    assert r.p_low == pytest.approx(slow[: observed + 1].sum(), abs=P_RESOLUTION)


@needs_6a60
def test_real_high_rate_pair_is_tested_in_seconds_with_the_right_moments(spikes_6a60):
    a, b = spikes_6a60["probe00_98"], spikes_6a60["probe00_117"]
    start = time.perf_counter()
    r = jitter_test(a, b, CFG.synaptic_window_s, CFG.jitter_s)
    elapsed = time.perf_counter() - start
    assert r.observed == 144142
    assert elapsed < 5.0, f"{elapsed:.1f} s (step 8 took 58 s)"
    # Uncapped, the sum's mean and variance are the windows' summed.
    _, pmfs, copies = window_pmfs(a, b, CFG.synaptic_window_s, CFG.jitter_s)
    values = np.arange(pmfs.shape[1])
    means = pmfs @ values
    variances = pmfs @ values**2 - means**2
    support = int(copies @ np.array([np.flatnonzero(p).max() for p in pmfs]))
    full = capped_sum_distribution(pmfs, copies, cap=support)
    x = np.arange(full.size)
    assert full.sum() == pytest.approx(1.0, abs=P_RESOLUTION)
    assert full @ x == pytest.approx(copies @ means, abs=0.01)  # a wrong draw moves it by >= 1
    assert full @ (x - full @ x) ** 2 == pytest.approx(copies @ variances, rel=1e-6)
    assert r.expected == pytest.approx(copies @ means, rel=1e-12)


def test_a_pair_with_no_possible_lag_in_the_window_tests_as_nothing_seen():
    # b's spikes are seconds from any of a's, so no jitter can put a lag in 1-4 ms.
    a, b = np.array([1.0, 2.0]), np.array([5.0, 9.0])
    observed, pmfs, copies = window_pmfs(a, b, CFG.synaptic_window_s, CFG.jitter_s)
    assert observed == 0 and pmfs.shape[0] == copies.size == 0
    r = jitter_test(a, b, CFG.synaptic_window_s, CFG.jitter_s)
    assert (r.observed, r.expected, r.p_high, r.p_low) == (0, 0.0, 1.0, 1.0)
