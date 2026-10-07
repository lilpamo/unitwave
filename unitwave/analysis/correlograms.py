"""Cross-correlograms and putative monosynaptic connections.

- **Cross-correlogram:** counts of spike pairs by lag = t_b - t_a (b after a is a
  positive lag), in bins centred on multiples of `bin_s`, each [c - b/2, c + b/2).
- **Interval jitter** (Amarasingham et al. 2012, J Neurophysiol 107:517): time is cut
  into fixed windows [k*D, (k+1)*D) and each spike of b is resampled uniformly within
  its own window. That keeps everything slower than D (shared rate modulation, the
  number of spikes in each window) and breaks timing finer than D.
- **Jitter-corrected correlogram:** observed minus the expected correlogram under
  jitter. The expectation is exact: a spike of b in window [s, s + D) has its lag
  from a spike of a uniform on [s - t_a, s - t_a + D).
- **Connection test (exact, so no seed):** the statistic is the count of pairs with a
  lag in the synaptic window [l1, l2). Under jitter it is a sum of independent
  per-spike counts, one per spike of b, each a function of its uniform position in
  its window. Their exact distribution comes from convolution, giving
  p_high = P(X >= observed) and p_low = P(X <= observed). Exact up to float
  rounding: p below P_RESOLUTION (1e-9) is reported as P_RESOLUTION.
  - **Direction:** "a -> b" tests b's spikes in [l1, l2) after a's; "b -> a" is the
    same with the units swapped. Both are tested for every pair, and BH runs across
    all of them.
  - **Excitatory only:** the label is a putative excitatory connection, one-sided
    on p_high. Inhibition is not labelled: the jitter expectation is the true
    correlogram smoothed over the jitter windows, so a sharp peak at +2 ms raises
    the expectation at the opposite lags too. A real a -> b peak then makes a
    significant "trough" b -> a (p below 1e-9 in the tests), which a two-sided label
    would call inhibition.
- **Close pairs:** units on one probe whose sites are within `close_um` are flagged.
  Sorting misses near-simultaneous spikes on nearby channels, which makes a false
  dip at zero lag. Units without a site position can't be flagged, and say so.

Limitation: jitter makes every pattern slower than D part of the null. A connection
whose effect is spread over more than D is not what this tests.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.fft import irfft, next_fast_len, rfft

from unitwave.analysis.responsiveness import benjamini_hochberg, load_response_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "correlograms.yaml"
# Float rounding in the FFT convolutions: against one-spike-at-a-time convolution on
# real pairs (session 6a601cc5), errors in p were at most 2.6e-12, growing with the
# number of spikes. Smaller p are not resolved; 1e-9 leaves a 400x margin and sits far
# below any BH threshold (0.05 / 870 tests ~ 6e-5).
P_RESOLUTION = 1e-9
_KEYS = ("window_s", "bin_s", "jitter_s", "synaptic_window_s", "max_units", "close_um")


@dataclass(frozen=True)
class CorrelogramConfig:
    window_s: float
    bin_s: float
    jitter_s: float
    synaptic_window_s: tuple[float, float]
    max_units: int
    close_um: float

    @property
    def null(self) -> str:
        return f"interval jitter, {self.jitter_s * 1000:g} ms windows, exact"


def load_correlogram_config(path: str | os.PathLike = DEFAULT_CONFIG) -> CorrelogramConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - set(_KEYS)), sorted(set(_KEYS) - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return CorrelogramConfig(
        float(raw["window_s"]),
        float(raw["bin_s"]),
        float(raw["jitter_s"]),
        tuple(float(x) for x in raw["synaptic_window_s"]),
        int(raw["max_units"]),
        float(raw["close_um"]),
    )


def _edges(window_s: float, bin_s: float) -> np.ndarray:
    """(2k + 2,) edges of bins centred on -k..k bins, k = window_s / bin_s."""
    k = round(window_s / bin_s)
    if k < 1 or abs(window_s / bin_s - k) > 1e-6:
        raise ValueError(f"{window_s} s is not a whole number of {bin_s} s bins")
    return (np.arange(-k, k + 2) - 0.5) * bin_s


def _pair_lags(a: np.ndarray, b: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Every b - a in [lo, hi), over all pairs. a, b: (n,) sorted."""
    first = np.searchsorted(b, a + lo, side="left")
    last = np.searchsorted(b, a + hi, side="left")
    n = last - first
    if n.sum() == 0:
        return np.empty(0)
    starts = np.repeat(first - np.cumsum(n) + n, n)
    index = starts + np.arange(n.sum())
    return b[index] - np.repeat(a, n)


def cross_correlogram(a, b, window_s: float, bin_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(lags (2k + 1,), counts (2k + 1,)) of every pair's b - a."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    edges = _edges(window_s, bin_s)
    counts, _ = np.histogram(_pair_lags(a, b, edges[0], edges[-1]), edges)
    return (edges[:-1] + edges[1:]) / 2, counts


def jitter_expected_ccg(a, b, window_s: float, bin_s: float, jitter_s: float) -> np.ndarray:
    """(2k + 1,) expected cross-correlogram when b's spikes are interval-jittered."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    edges = _edges(window_s, bin_s)
    starts = np.floor(b / jitter_s) * jitter_s  # sorted, as b is
    d = np.sort(_pair_lags(a, starts, edges[0] - jitter_s, edges[-1]))  # window start - t_a
    # Each d spreads one pair uniformly over [d, d + jitter_s). The expected number of
    # pairs with lag below e is #(d <= e - D) + sum over e - D < d < e of (e - d) / D,
    # from prefix sums, so memory stays linear in the number of pairs.
    prefix = np.concatenate([[0.0], np.cumsum(d)])
    whole = np.searchsorted(d, edges - jitter_s, side="right")
    below = np.searchsorted(d, edges, side="left")
    partial = (below - whole) * edges - (prefix[below] - prefix[whole])
    return np.diff(whole + partial / jitter_s)


@dataclass(frozen=True)
class JitterTest:
    """observed: the count; expected: its mean under jitter; p_high = P(X >= observed),
    p_low = P(X <= observed), both exact under the jitter null (to P_RESOLUTION)."""

    observed: int
    expected: float
    p_high: float
    p_low: float


def _count_at(a: np.ndarray, t: np.ndarray, l1: float, l2: float) -> np.ndarray:
    """(len(t),) number of a's spikes with t - t_a in [l1, l2)."""
    return np.searchsorted(a, t - l1, side="right") - np.searchsorted(a, t - l2, side="right")


def window_pmfs(a, b, lag_window, jitter_s: float) -> tuple[int, np.ndarray, np.ndarray]:
    """The test's ingredients: the observed count of b - a lags in lag_window, and for
    each jitter window holding b's spikes where the count can be nonzero, the pmf of
    one jittered spike's count, pmfs (n_windows, n_values), and how many of b's spikes
    the window holds, copies (n_windows,). Windows where the count is 0 everywhere add
    nothing and are left out."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    l1, l2 = lag_window
    observed = int(_count_at(a, b, l1, l2).sum())
    k_of_b = np.floor(b / jitter_s).astype(np.int64)
    windows, m = np.unique(k_of_b, return_counts=True)  # windows holding b's spikes
    # Within a window the count is piecewise constant, changing at t_a + l1 and t_a + l2.
    cuts = np.concatenate([a + l1, a + l2])
    lo, hi = windows * jitter_s, (windows + 1) * jitter_s

    def window_of(t: np.ndarray) -> np.ndarray:
        """Index into `windows` of the window holding each t, or -1. Exact comparisons
        with the boundaries (floor(t / D) misplaces t near a boundary late in a session)."""
        i = np.searchsorted(lo, t, side="right") - 1
        inside = (i >= 0) & (t < hi[np.maximum(i, 0)])
        return np.where(inside, i, -1)

    cuts = cuts[window_of(cuts) >= 0]
    points = np.unique(np.concatenate([lo, hi, cuts]))
    seg_start, seg_len = points[:-1], np.diff(points)
    seg_index = window_of(seg_start)
    keep = (seg_index >= 0) & (seg_len > 0)
    seg_start, seg_len, seg_index = seg_start[keep], seg_len[keep], seg_index[keep]
    value = _count_at(a, seg_start, l1, l2)
    nonzero = np.unique(seg_index[value > 0])
    mine = np.isin(seg_index, nonzero)
    # Probability of each count = the length of window where it holds / jitter_s.
    pmfs = np.zeros((nonzero.size, int(value.max(initial=0)) + 1))
    row = np.searchsorted(nonzero, seg_index[mine])
    np.add.at(pmfs, (row, value[mine]), seg_len[mine] / jitter_s)
    pmfs[:, 0] += 1.0 - pmfs.sum(axis=1)  # rounding: lengths sum to the window
    copies = m[nonzero]
    assert pmfs.shape == (copies.size, pmfs.shape[1])
    return observed, pmfs, copies


def _fold(rows: np.ndarray, cap: int) -> np.ndarray:
    """(n, width) -> (n, min(width, cap + 1)), the mass at cap and above in column cap."""
    if rows.shape[1] > cap + 1:
        rows[:, cap] = rows[:, cap:].sum(axis=1)
        rows = rows[:, : cap + 1]
    return rows


def capped_sum_distribution(pmfs: np.ndarray, copies: np.ndarray, cap: int) -> np.ndarray:
    """(cap + 1,) distribution of a sum of independent counts, copies[i] of them drawn
    from pmfs[i] (pmfs: (n_windows, n_values)); the last bin holds P(sum >= cap).

    Exact up to float rounding: the draws are convolved in pairs, level by level, with
    batched FFTs (about n log n work for n draws, where convolving them one at a time
    costs n x cap). Mass at cap and above is folded into the last bin after each level,
    which is exact because counts are never negative. Rounding leaves small errors of
    either sign in each bin (not clipped: clipping biases sums upward), so
    probabilities below P_RESOLUTION are not resolved.
    """
    pmfs, copies = np.asarray(pmfs, np.float64), np.asarray(copies, np.int64)
    assert pmfs.ndim == 2 and copies.shape == (pmfs.shape[0],)
    rows = _fold(np.repeat(pmfs, copies, axis=0), cap)  # (n_draws, width)
    if rows.shape[0] == 0:
        rows = np.zeros((1, 1))
        rows[0, 0] = 1.0
    while rows.shape[0] > 1:
        if rows.shape[0] % 2:  # an odd one out is paired with a sum of nothing
            nothing = np.zeros((1, rows.shape[1]))
            nothing[0, 0] = 1.0
            rows = np.vstack([rows, nothing])
        width = 2 * rows.shape[1] - 1
        size = next_fast_len(width, real=True)
        merged = irfft(rfft(rows[0::2], size) * rfft(rows[1::2], size), size)[:, :width]
        rows = _fold(merged, cap)
    dist = np.zeros(cap + 1)
    dist[: rows.shape[1]] = rows[0]
    return dist


def jitter_test(a, b, lag_window, jitter_s: float) -> JitterTest:
    """Exact interval-jitter test of the count of b - a lags in lag_window. p below
    P_RESOLUTION is reported as P_RESOLUTION (conservative)."""
    observed, pmfs, copies = window_pmfs(a, b, lag_window, jitter_s)
    # NumPy reductions, not BLAS products: BLAS can sum in a different order on another
    # thread, and the same pair must give the same number every time.
    expected = float((copies * (pmfs * np.arange(pmfs.shape[1])).sum(axis=1)).sum())
    # Capped at observed + 1: the last bin holds P(X >= observed + 1). Both p come from
    # the last two bins, not sums over many bins, which would add up their rounding.
    dist = capped_sum_distribution(pmfs, copies, observed + 1)
    p_high = float(np.clip(dist[observed] + dist[observed + 1], P_RESOLUTION, 1.0))
    p_low = float(np.clip(1.0 - dist[observed + 1], P_RESOLUTION, 1.0))
    return JitterTest(observed, expected, p_high, p_low)


def close_pairs(units: pd.DataFrame, pairs, close_um: float) -> tuple[list, list[str]]:
    """Per pair: True/False whether the units' sites are within close_um on one probe,
    or None with the reason when a position is missing. units: probe, depth_um,
    lateral_um, indexed by unit id."""
    flags, why = [], []
    for u, v in pairs:
        if units.at[u, "probe"] != units.at[v, "probe"]:
            flags.append(False)
            why.append("")
            continue
        missing = [w for w in (u, v) if pd.isna(units.at[w, "depth_um"])]
        if missing:
            flags.append(None)
            why.append(f"no site position for {', '.join(missing)}")
            continue
        lateral = [
            0.0 if pd.isna(units.at[w, "lateral_um"]) else units.at[w, "lateral_um"] for w in (u, v)
        ]
        distance = np.hypot(
            units.at[u, "depth_um"] - units.at[v, "depth_um"], lateral[0] - lateral[1]
        )
        flags.append(bool(distance <= close_um))
        why.append("")
    return flags, why


def connections(
    spikes, unit_ids, units: pd.DataFrame, cfg: CorrelogramConfig, alpha: float | None = None
) -> pd.DataFrame:
    """(2 * n_pairs, ...) one row per direction of every pair among unit_ids: pre,
    post, observed, expected, p (one-sided, excess), q, connected (putative
    excitatory), close, close_why, n_tests, null, window_s. q is Benjamini-Hochberg
    across all rows."""
    unit_ids = list(unit_ids)
    if len(unit_ids) > cfg.max_units:
        raise ValueError(
            f"{len(unit_ids)} units make {len(unit_ids) * (len(unit_ids) - 1)} tests; test "
            f"at most {cfg.max_units} units at once (configs/correlograms.yaml): narrow the "
            "shown units by probe or region"
        )
    if len(unit_ids) < 2:
        raise ValueError("needs at least 2 units")
    alpha = load_response_config().alpha if alpha is None else alpha
    pairs = list(combinations(unit_ids, 2))
    close, why = close_pairs(units, pairs, cfg.close_um)
    directed = [
        (pre, post, c, w)
        for (u, v), c, w in zip(pairs, close, why)
        for pre, post in ((u, v), (v, u))
    ]

    def test(d):
        return jitter_test(spikes[d[0]], spikes[d[1]], cfg.synaptic_window_s, cfg.jitter_s)

    # Each test is exact and independent of the others, so running them on threads
    # changes only how long they take (S1: 702 tests in one region of 6a601cc5).
    with ThreadPoolExecutor() as pool:
        results = list(pool.map(test, directed))
    rows = [
        (pre, post, r.observed, r.expected, r.p_high, c, w)
        for (pre, post, c, w), r in zip(directed, results)
    ]
    out = pd.DataFrame(
        rows, columns=["pre", "post", "observed", "expected", "p", "close", "close_why"]
    )
    out["q"] = benjamini_hochberg(out["p"].to_numpy())
    out["connected"] = out["q"] < alpha
    out["n_tests"] = len(out)
    out["null"] = cfg.null
    out["window_s"] = [tuple(cfg.synaptic_window_s)] * len(out)
    return out
