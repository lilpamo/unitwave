"""Sliding refractory-period test: is a unit's contamination below a limit, confidently?

A port of ibllib's `brainbox.metrics.single_units.slidingRP_viol` (MIT licence,
Copyright (c) International Brain Laboratory; github.com/int-brain-lab/ibllib), the
older of IBL's two implementations. The newer one (the `slidingRP` package) is GPL and
is not used here. The phylib autocorrelogram it relies on (BSD-3-Clause, cortex-lab)
is computed directly with sorted searches instead of phylib's shift loop.

For each candidate refractory period RP in 1.25-10 ms, the count of spike pairs closer
than about RP is compared with the largest count a unit with `contamination` (a
fraction of its rate) would produce with probability `alpha` (a Poisson quantile). The
unit passes if its count is at or below that bound at any RP. Units with a low firing
rate fail by design: they can't show confidently low contamination.

As in ibllib: spike times are truncated to 20 kHz samples, correlogram bins are
0.25 ms, pairs are counted once (i < j) including zero lag, and the firing rate is the
autocorrelogram's mean from 0.5 to 1 s.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Fixed parts of IBL's algorithm, not tunables.
_RATE_HZ = 20000
_BIN_S = 0.00025
_BIN_SAMPLES = int(_RATE_HZ * _BIN_S)  # 5
_TEST_BINS = (5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40)
_ACG_BINS = 4001  # one-sided autocorrelogram of a 2 s window: lags 0 to 1 s
_RATE_FROM_BIN = _ACG_BINS // 2  # rate from the second half (0.5 to 1 s)


def pairs_closer_than(samples: np.ndarray, gap: int) -> int:
    """Number of spike pairs (i < j) with samples[j] - samples[i] < gap. samples: sorted."""
    samples = np.asarray(samples)
    assert samples.ndim == 1
    after = np.searchsorted(samples, samples + gap, side="left")
    return int((after - np.arange(1, samples.size + 1)).sum())


def max_acceptable_violations(
    rate_hz: float, rp_s: float, duration_s: float, contaminated_rate_hz: float, alpha: float
) -> float:
    """The largest violation count consistent with the contaminated rate, at quantile alpha.

    -1 when that quantile is 0: then even zero violations are unsurprising, and no count
    can pass at this refractory period.
    """
    expected = contaminated_rate_hz * rp_s * 2 * rate_hz * duration_s
    bound = stats.poisson.ppf(alpha, expected)
    if bound == 0 and stats.poisson.pmf(0, expected) > 0:
        return -1
    return bound


def _prepare(times: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """(samples, duration_s, rate_hz) as IBL computes them; None for fewer than two
    distinct spike times, which fail."""
    times = np.asarray(times, np.float64)
    assert times.ndim == 1
    if times.size == 0 or not times[-1] > times[0]:
        return None
    duration = times[-1] - times[0]
    samples = (times * _RATE_HZ).astype(np.int64)
    n = samples.size
    # Pairs in autocorrelogram bins _RATE_FROM_BIN .. _ACG_BINS - 1, each bin a rate in
    # Hz; IBL divides their sum by the bin count minus one.
    in_window = pairs_closer_than(samples, _ACG_BINS * _BIN_SAMPLES) - pairs_closer_than(
        samples, _RATE_FROM_BIN * _BIN_SAMPLES
    )
    return samples, duration, in_window / n / _BIN_S / _RATE_FROM_BIN


def sliding_rp_pass(times: np.ndarray, contamination: float, alpha: float) -> bool:
    """Whether a unit's spike train, (n_spikes,) seconds sorted, passes the test."""
    prepared = _prepare(times)
    if prepared is None:
        return False
    samples, duration, rate = prepared
    for index in _TEST_BINS:
        # Cumulative autocorrelogram through bin `index`, tested at RP = index bins.
        violations = pairs_closer_than(samples, (index + 1) * _BIN_SAMPLES)
        rp = index * _BIN_S + 1e-6
        if violations <= max_acceptable_violations(rate, rp, duration, rate * contamination, alpha):
            return True
    return False


@dataclass(frozen=True)
class RefractoryDetails:
    """The test at every refractory period, for display. rp_s, violations and
    max_acceptable are (n_rp,); a period passes where violations <= max_acceptable
    (a bound of -1 means no count can pass there). why: set when there is no table."""

    rp_s: np.ndarray
    violations: np.ndarray
    max_acceptable: np.ndarray
    rate_hz: float
    duration_s: float
    contamination: float
    alpha: float
    why: str = ""

    @property
    def passed(self) -> bool:
        return bool((self.violations <= self.max_acceptable).any())

    @property
    def first_pass_rp_s(self) -> float | None:
        ok = np.flatnonzero(self.violations <= self.max_acceptable)
        return float(self.rp_s[ok[0]]) if ok.size else None


def sliding_rp_details(times: np.ndarray, contamination: float, alpha: float) -> RefractoryDetails:
    """sliding_rp_pass's test at every refractory period, not stopping at the first
    pass. Its verdict (`passed`) is sliding_rp_pass's; tests check that on real units."""
    prepared = _prepare(times)
    if prepared is None:
        empty = np.empty(0)
        why = "fewer than two distinct spike times"
        return RefractoryDetails(empty, empty, empty, np.nan, 0.0, contamination, alpha, why)
    samples, duration, rate = prepared
    rp = np.array([index * _BIN_S + 1e-6 for index in _TEST_BINS])
    violations = np.array(
        [pairs_closer_than(samples, (index + 1) * _BIN_SAMPLES) for index in _TEST_BINS], float
    )
    bound = np.array(
        [max_acceptable_violations(rate, r, duration, rate * contamination, alpha) for r in rp]
    )
    return RefractoryDetails(rp, violations, bound, rate, duration, contamination, alpha)
