"""Event-aligned rasters and PSTHs.

Conventions:
- window = (t_start, t_stop) seconds relative to the event; bins are half-open
  [edge_i, edge_i+1) and must tile the window exactly.
- Rates are in Hz: counts / bin_width.
- Trials whose event time is NaN are excluded and counted (n_excluded), never filled.
- The PSTH is the mean over trials; SEM = std over trials (ddof=1) / sqrt(n_trials),
  NaN when there is a single trial.
- Baseline subtraction is per trial: each trial's mean rate in the baseline window is
  subtracted from that trial's bins before averaging.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PSTH:
    """bin_centers, mean, sem: (n_bins,). mean and sem in Hz."""

    bin_centers: np.ndarray
    mean: np.ndarray
    sem: np.ndarray
    n_trials: int
    n_excluded: int


def bin_edges(window: tuple[float, float], bin_width: float) -> np.ndarray:
    """(n_bins + 1,) edges relative to the event."""
    t0, t1 = (float(t) for t in window)
    if not t1 > t0:
        raise ValueError(f"window must be (start, stop) with stop > start, got {window}")
    if not bin_width > 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")
    n = (t1 - t0) / bin_width
    n_bins = round(n)
    if n_bins < 1 or abs(n - n_bins) > 1e-6:
        raise ValueError(
            f"window {window} is not a whole number of bins of {bin_width} s ({n:.4g} bins)"
        )
    return t0 + bin_width * np.arange(n_bins + 1)


def _valid(events: np.ndarray) -> np.ndarray:
    events = np.asarray(events, np.float64)
    assert events.ndim == 1, f"events must be (n_trials,), got {events.shape}"
    return events[np.isfinite(events)]


def trial_counts(
    spikes: np.ndarray, events: np.ndarray, window: tuple[float, float], bin_width: float
) -> np.ndarray:
    """(n_trials, n_bins) spike counts. spikes: (n_spikes,) sorted; events: (n_trials,) finite."""
    spikes = np.asarray(spikes, np.float64)
    events = np.asarray(events, np.float64)
    assert spikes.ndim == 1 and events.ndim == 1
    edges = bin_edges(window, bin_width)
    absolute = events[:, None] + edges[None, :]  # (n_trials, n_bins + 1)
    idx = np.searchsorted(spikes, absolute.ravel(), side="left").reshape(absolute.shape)
    counts = np.diff(idx, axis=1)
    assert counts.shape == (events.size, edges.size - 1)
    return counts


def psth(
    spikes: np.ndarray,
    events: np.ndarray,
    window: tuple[float, float],
    bin_width: float,
    baseline: tuple[float, float] | None = None,
) -> PSTH:
    """Mean ± SEM rate across trials. spikes: (n_spikes,) sorted; events: (n_trials,)."""
    valid = _valid(events)
    n_excluded = int(np.asarray(events).size - valid.size)
    if valid.size == 0:
        raise ValueError("no trials with a time for this event")
    rates = trial_counts(spikes, valid, window, bin_width) / bin_width  # (n_trials, n_bins)
    if baseline is not None:
        b0, b1 = (float(t) for t in baseline)
        if not b1 > b0:
            raise ValueError(f"baseline must be (start, stop) with stop > start, got {baseline}")
        base = trial_counts(spikes, valid, (b0, b1), b1 - b0)[:, 0] / (b1 - b0)  # (n_trials,)
        rates = rates - base[:, None]
    n = rates.shape[0]
    sem = rates.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(rates.shape[1], np.nan)
    edges = bin_edges(window, bin_width)
    result = PSTH((edges[:-1] + edges[1:]) / 2, rates.mean(axis=0), sem, n, n_excluded)
    assert result.mean.shape == result.sem.shape == result.bin_centers.shape
    return result


def raster(
    spikes: np.ndarray, events: np.ndarray, window: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """(trial_index, relative_time), both (n_spikes_in_window,), for trials with an event time.

    trial_index counts only valid trials, in their original order.
    """
    spikes = np.asarray(spikes, np.float64)
    valid = _valid(events)
    t0, t1 = window
    lo = np.searchsorted(spikes, valid + t0, side="left")
    hi = np.searchsorted(spikes, valid + t1, side="left")
    trial = np.repeat(np.arange(valid.size), hi - lo)
    rel = np.concatenate([spikes[a:b] - e for a, b, e in zip(lo, hi, valid)] or [np.empty(0)])
    assert trial.shape == rel.shape
    return trial, rel


def population_psth(
    spikes: Mapping[str, np.ndarray],
    unit_ids: Sequence[str],
    events: np.ndarray,
    window: tuple[float, float],
    bin_width: float,
    baseline: tuple[float, float] | None = None,
) -> np.ndarray:
    """(n_units, n_bins) trial-mean rates in Hz, rows in unit_ids order."""
    rows = [psth(spikes[u], events, window, bin_width, baseline).mean for u in unit_ids]
    out = np.vstack(rows) if rows else np.empty((0, bin_edges(window, bin_width).size - 1))
    assert out.shape[0] == len(unit_ids)
    return out


def peak_order(matrix: np.ndarray) -> np.ndarray:
    """(n_units,) row order by the bin of each row's maximum (earliest first; ties keep order)."""
    assert matrix.ndim == 2
    return np.argsort(np.argmax(matrix, axis=1), kind="stable")


def selection_average(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM across units of (n_units, n_bins) PSTHs: two (n_bins,) arrays."""
    assert matrix.ndim == 2 and matrix.shape[0] > 0
    n = matrix.shape[0]
    sem = matrix.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(matrix.shape[1], np.nan)
    return matrix.mean(axis=0), sem


def scale_rows_for_display(matrix: np.ndarray) -> np.ndarray:
    """(n_units, n_bins) each row divided by its max |value|; for heatmap colour only."""
    assert matrix.ndim == 2
    peak = np.abs(matrix).max(axis=1, keepdims=True)
    return np.divide(matrix, peak, out=np.zeros_like(matrix, dtype=np.float64), where=peak > 0)


def alternate_halves(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trials with an event time, split alternately: (1st, 3rd, ...) and (2nd, 4th, ...).

    For sorting a heatmap on one half and showing the other, so the order is not fitted
    to the data it displays.
    """
    valid = _valid(events)
    return valid[0::2], valid[1::2]
