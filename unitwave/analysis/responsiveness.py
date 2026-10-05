"""Is a unit's rate around an event different from its baseline? A test against a shift null.

Statistic, per unit: the mean over trials of (rate in the response window - rate in the
baseline window), in Hz. The test is two-sided.

Null: the spike train circularly shifted against the event times. This keeps the
spike train's own timing (autocorrelation, slow drift) and the events' spacing, and
breaks only their alignment. The circle is the span the event windows cover. Spike
times and window edges sit on a `grid_s` grid, and every grid shift at least
`min_shift_s` from zero (either way round) is one null draw. All shifts are computed
at once by FFT cross-correlation, so the test is deterministic: no random draws, no
seed.

Refused presentations (the user's decision, 2026-10-05; docs/NEGATIVE_RESULTS.md):
- **Periodic:** when more than `max_realigned_fraction` of the null's shifts line the
  events up with themselves (self-overlap above `max_self_overlap`: other event onsets
  within the windows' span of each shifted event, on average), the null can't separate
  a response from the stimulus cycle. Keeping those shifts leaves it without power;
  dropping them leaves too few distinct draws and it invents responses. Refused.
- **Back to back:** when more than `max_close_fraction` of events have another within the
  windows' span, each baseline falls in the previous response. Refused.
Both are decided from event times alone; the null itself is unchanged.

p = (1 + #{allowed shifts with |statistic| >= |observed|}) / (1 + #allowed shifts)

Benjamini-Hochberg q-values are computed across the units tested in one call; a unit
is `responsive` when q < alpha. The number of units tested is part of every result.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import fft
from scipy.stats import false_discovery_control

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "analysis.yaml"
_KEYS = {
    "baseline_window",
    "response_window",
    "grid_s",
    "min_shift_s",
    "alpha",
    "max_self_overlap",
    "max_realigned_fraction",
    "max_close_fraction",
}
_CHUNK = 8  # units per FFT batch


@dataclass(frozen=True)
class ResponseConfig:
    baseline_window: tuple[float, float]
    response_window: tuple[float, float]
    grid_s: float
    min_shift_s: float
    alpha: float
    max_self_overlap: float
    max_realigned_fraction: float
    max_close_fraction: float


def load_response_config(path: str | os.PathLike = DEFAULT_CONFIG) -> ResponseConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return ResponseConfig(
        baseline_window=tuple(float(t) for t in raw["baseline_window"]),
        response_window=tuple(float(t) for t in raw["response_window"]),
        grid_s=float(raw["grid_s"]),
        min_shift_s=float(raw["min_shift_s"]),
        alpha=float(raw["alpha"]),
        max_self_overlap=float(raw["max_self_overlap"]),
        max_realigned_fraction=float(raw["max_realigned_fraction"]),
        max_close_fraction=float(raw["max_close_fraction"]),
    )


def _grid_bins(window: tuple[float, float], grid_s: float, name: str) -> tuple[int, int]:
    """(start, stop) of a window in grid steps; refuses edges off the grid."""
    t0, t1 = window
    if not t1 > t0:
        raise ValueError(f"{name} window must be (start, stop) with stop > start, got {window}")
    bins = tuple(round(t / grid_s) for t in window)
    if any(abs(b * grid_s - t) > 1e-9 for b, t in zip(bins, window)):
        raise ValueError(f"{name} window {window} is not on the {grid_s} s grid")
    return bins


@dataclass(frozen=True)
class _Circle:
    """The events on the grid circle, shared by every unit tested against them."""

    c0: float  # the circle's start, seconds
    n_grid: int
    bins: np.ndarray  # (n_trials,) each event's grid step
    kernel_fft: np.ndarray  # conj(rfft(kernel)), (n_grid // 2 + 1,)
    allowed: np.ndarray  # (n_grid,) bool, shifts that count as null draws


def _circle(events: np.ndarray, cfg: ResponseConfig) -> _Circle:
    b0, b1 = _grid_bins(cfg.baseline_window, cfg.grid_s, "baseline")
    r0, r1 = _grid_bins(cfg.response_window, cfg.grid_s, "response")
    if b0 < r1 and r0 < b1:
        raise ValueError(f"baseline {cfg.baseline_window} and response windows overlap")
    assert events.ndim == 1, f"events must be (n_trials,), got {events.shape}"
    events = events[np.isfinite(events)]
    if events.size == 0:
        raise ValueError("no trials with a time for this event")

    lo, hi = min(b0, r0), max(b1, r1)
    c0 = float(events.min()) + lo * cfg.grid_s  # the circle covers every window
    bins = np.round((events - c0) / cfg.grid_s).astype(np.int64)
    # Extended by a few steps of real time to a length the FFT handles quickly.
    n_grid = fft.next_fast_len(int(bins.max()) + hi, real=True)
    min_shift = round(cfg.min_shift_s / cfg.grid_s)
    if n_grid <= 2 * min_shift:
        raise ValueError(
            f"events span {n_grid * cfg.grid_s:.1f} s, too short for shifts of at least "
            f"{cfg.min_shift_s} s each way"
        )
    # Kernel: +1/(n L_r) over each response window, -1/(n L_b) over each baseline.
    kernel = np.zeros(n_grid)
    for start, stop in ((r0, r1), (b0, b1)):
        weight = (1 if start == r0 else -1) / (events.size * (stop - start) * cfg.grid_s)
        for offset in range(start, stop):
            np.add.at(kernel, bins + offset, weight)
    shift = np.arange(n_grid)
    allowed = np.minimum(shift, n_grid - shift) >= min_shift
    _refuse_periodic(bins, n_grid, hi - lo, allowed, cfg)
    return _Circle(
        c0=c0,
        n_grid=n_grid,
        bins=bins,
        kernel_fft=np.conj(fft.rfft(kernel)),
        allowed=allowed,
    )


def _refuse_periodic(bins: np.ndarray, n_grid: int, span: int, allowed, cfg) -> None:
    """Refuse presentations the shift null can't test: back to back (closer than the
    windows' `span` grid steps), or periodic (too many shifts line them up again)."""
    gaps = np.diff(np.sort(bins))
    if not gaps.size:
        return
    apart = f"{np.median(gaps) * cfg.grid_s:.3g} s apart"
    nearest = np.minimum(np.r_[gaps, np.inf], np.r_[np.inf, gaps])
    if np.mean(nearest <= span) > cfg.max_close_fraction:
        raise ValueError(
            f"the presentations come {apart}, closer than the test's windows "
            f"({cfg.baseline_window[0]:g} to {cfg.response_window[1]:g} s around each): each "
            "baseline falls in the previous presentation's response, so response can't be "
            "compared with baseline. Compare conditions with the selectivity test instead"
        )
    events = np.zeros(n_grid)
    np.add.at(events, bins, 1.0)
    box = np.zeros(n_grid)
    box[: span + 1] = 1.0
    box[n_grid - span :] = 1.0
    ef = fft.rfft(events)
    near = fft.irfft(ef * fft.rfft(box), n=n_grid)  # onsets within the span of each step
    overlap = fft.irfft(np.conj(ef) * fft.rfft(near), n=n_grid) / bins.size  # (n_grid,)
    realigned = float(np.mean(overlap[allowed] > cfg.max_self_overlap))
    if realigned > cfg.max_realigned_fraction:
        raise ValueError(
            f"the presentations are periodic (about {apart}): {realigned:.0%} of the null's "
            "shifts line them up with themselves, so shifting the spike train can't separate "
            "a response from the stimulus cycle, and a count here would not be a finding "
            "(docs/NEGATIVE_RESULTS.md). Compare conditions with the selectivity test instead"
        )


def _counts(spikes, unit_ids, circle: _Circle, grid_s: float) -> np.ndarray:
    """(n_units, n_grid) spike counts per grid step on the circle."""
    x = np.zeros((len(unit_ids), circle.n_grid))
    for i, u in enumerate(unit_ids):
        k = np.floor((np.asarray(spikes[u], np.float64) - circle.c0) / grid_s).astype(np.int64)
        x[i] = np.bincount(k[(k >= 0) & (k < circle.n_grid)], minlength=circle.n_grid)
    return x


def _shifted(x: np.ndarray, circle: _Circle) -> np.ndarray:
    """(n_units, n_grid): stats[:, d] = sum_m kernel[m] x[:, m + d], by FFT."""
    xf = fft.rfft(x, axis=1, workers=-1)
    return fft.irfft(xf * circle.kernel_fft, n=circle.n_grid, axis=1, workers=-1)


def circular_statistics(spikes, unit_ids, events: np.ndarray, cfg: ResponseConfig):
    """The statistic at every circular shift, for each unit. For small inputs and tests.

    Returns (stats, allowed, (x, bins, c0)):
    - stats: (n_units, n_grid), the statistic with events moved d grid steps later;
      stats[:, 0] is the observed value;
    - allowed: (n_grid,) bool, the shifts that count as null draws;
    - x: (n_units, n_grid) spike counts per grid step on the circle;
    - bins: (n_trials,) each event's grid step;
    - c0: the circle's start in seconds.
    """
    circle = _circle(np.asarray(events, np.float64), cfg)
    x = _counts(spikes, list(unit_ids), circle, cfg.grid_s)
    stats = _shifted(x, circle)
    assert stats.shape == x.shape == (len(unit_ids), circle.n_grid)
    return stats, circle.allowed, (x, circle.bins, circle.c0)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """(n_tests,) Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, np.float64)
    assert p.ndim == 1
    return false_discovery_control(p, method="bh") if p.size else p.copy()


def responsiveness(spikes, unit_ids, events: np.ndarray, cfg: ResponseConfig) -> pd.DataFrame:
    """(n_units, 8) per unit: statistic_hz, p, q, responsive, n_shifts, n_tests, n_trials,
    n_excluded. Indexed by unit id; q is Benjamini-Hochberg across these units.
    """
    unit_ids = list(unit_ids)
    events = np.asarray(events, np.float64)
    circle = _circle(events, cfg)
    observed = np.empty(len(unit_ids))
    exceed = np.empty(len(unit_ids), np.int64)
    # A few units at a time: each unit's null is n_grid values (millions in a session).
    for a in range(0, len(unit_ids), _CHUNK):
        stats = _shifted(_counts(spikes, unit_ids[a : a + _CHUNK], circle, cfg.grid_s), circle)
        obs = stats[:, 0]
        # The FFT's rounding noise must not decide ties with the observed value.
        tol = 1e-9 * np.maximum(1.0, np.abs(obs))
        null = np.abs(stats[:, circle.allowed])  # (n_chunk, n_shifts)
        observed[a : a + _CHUNK] = obs
        exceed[a : a + _CHUNK] = (null >= (np.abs(obs) - tol)[:, None]).sum(axis=1)
    n_shifts = int(circle.allowed.sum())
    p = (1 + exceed) / (1 + n_shifts)
    q = benjamini_hochberg(p)
    n_trials = int(np.isfinite(events).sum())
    return pd.DataFrame(
        {
            "statistic_hz": observed,
            "p": p,
            "q": q,
            "responsive": q < cfg.alpha,
            "n_shifts": n_shifts,
            "n_tests": len(unit_ids),
            "n_trials": n_trials,
            "n_excluded": int(events.size - n_trials),
        },
        index=pd.Index(unit_ids, name="unit_id"),
    )
