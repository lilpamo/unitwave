"""Unit quality panel: what the QC already decided about a unit, shown.

Descriptive: it labels nothing new. The verdicts come from qc/ (the unit table's
`qc_passed` and `qc_reason`, the sliding refractory-period test) and, for IBL
sessions in the ONE cache, IBL's own label criteria.

- **ISI histogram:** intervals between consecutive spikes, 0 to `isi_max_s`,
  half-open bins; longer intervals counted, not dropped.
- **Autocorrelogram:** lags between ordered pairs of different spikes (i != j), in
  bins centred on multiples of `acg_bin_s` up to `acg_window_s`, each [c - b/2,
  c + b/2): the cross-correlogram's binning (analysis.correlograms). Two spikes at
  one time are a pair at lag 0.
- **Presence ratio:** IBL's definition (brainbox quick_unit_metrics): bins of
  `presence_window_s` from the probe's first spike, np.arange(start, end + w/2, w) of
  them, and the fraction with a spike. On d23a44ef probe00 it equals IBL's stored
  value for every cluster when start and end are the first and last spike of all
  674 clusters; in Studio they are those of the probe's loaded units.
- **Rate across the session:** spike counts in bins of `rate_bin_s` from the probe's
  first spike, the last bin as long as what remains, divided by bin length.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from unitwave.analysis.correlograms import cross_correlogram
from unitwave.data.cluster_files import Waveform
from unitwave.qc.refractory import RefractoryDetails, sliding_rp_details

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "unit_quality.yaml"
_KEYS = ("isi_max_s", "isi_bin_s", "acg_window_s", "acg_bin_s", "rate_bin_s", "presence_window_s")


@dataclass(frozen=True)
class QualityConfig:
    isi_max_s: float
    isi_bin_s: float
    acg_window_s: float
    acg_bin_s: float
    rate_bin_s: float
    presence_window_s: float


def load_quality_config(path: str | os.PathLike = DEFAULT_CONFIG) -> QualityConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - set(_KEYS)), sorted(set(_KEYS) - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return QualityConfig(*(float(raw[k]) for k in _KEYS))


def _n_bins(span: float, width: float) -> int:
    n = round(span / width)
    if n < 1 or abs(span / width - n) > 1e-6:
        raise ValueError(f"{span} s is not a whole number of {width} s bins")
    return n


def isi_histogram(
    times: np.ndarray, max_s: float, bin_s: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """(edges (n_bins + 1,), counts (n_bins,), intervals at or beyond max_s)."""
    isi = np.diff(np.asarray(times, np.float64))
    edges = bin_s * np.arange(_n_bins(max_s, bin_s) + 1)
    index = np.floor(isi / bin_s + 1e-9).astype(np.int64)
    inside = index < edges.size - 1
    counts = np.bincount(index[inside], minlength=edges.size - 1)
    return edges, counts, int((~inside).sum())


def autocorrelogram(
    times: np.ndarray, window_s: float, bin_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """(lags (2k + 1,), counts (2k + 1,)), k = window_s / bin_s: the cross-correlogram
    of the train with itself (analysis.correlograms), less each spike's pair with
    itself at lag 0."""
    t = np.asarray(times, np.float64)
    lags, counts = cross_correlogram(t, t, window_s, bin_s)
    counts[counts.size // 2] -= t.size
    return lags, counts


def presence_ratio(
    times: np.ndarray, start: float, end: float, window_s: float
) -> tuple[float, np.ndarray]:
    """(fraction of IBL's presence bins with a spike, (n_bins,) spike counts per bin)."""
    t = np.asarray(times, np.float64)
    n = np.arange(start, end + window_s / 2, window_s).size
    counts = np.bincount(np.floor((t - start) / window_s).astype(np.int64), minlength=n)[:n]
    return float((counts > 0).sum() / n), counts


def rate_over_session(
    times: np.ndarray, start: float, end: float, bin_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """(bin centres (n_bins,), rate in Hz (n_bins,)); the last bin ends at `end`,
    closed, and its rate is over its own length."""
    edges = np.arange(start, end, bin_s)
    if end - edges[-1] < 1e-6:  # a rounding sliver, not a bin
        edges = edges[:-1]
    edges = np.append(edges, end)
    counts, _ = np.histogram(np.asarray(times, np.float64), edges)
    return (edges[:-1] + edges[1:]) / 2, counts / np.diff(edges)


@dataclass(frozen=True)
class UnitQuality:
    """Everything the quality panel shows for one unit. Times in seconds."""

    unit: str
    n_spikes: int
    span_s: tuple[float, float]  # the probe's first and last spike
    isi: tuple[np.ndarray, np.ndarray, int]
    acg: tuple[np.ndarray, np.ndarray]
    refractory: RefractoryDetails
    presence: tuple[float, np.ndarray]
    rate: tuple[np.ndarray, np.ndarray]
    qc_passed: bool
    qc_reasons: list[str]
    waveform: Waveform | None
    waveform_missing: str
    ibl_criteria: dict | None
    ibl_missing: str


def unit_quality(
    spikes,
    unit: str,
    probe_units,
    qc_row,
    cfg: QualityConfig,
    contamination: float,
    alpha: float,
    waveform: Waveform | None = None,
    waveform_missing: str = "",
    ibl_criteria: dict | None = None,
    ibl_missing: str = "",
) -> UnitQuality:
    """The panel for `unit`. probe_units: the unit ids on its probe (for the span);
    qc_row: its unit-table row (qc_passed, qc_reason); contamination and alpha: the
    sliding RP test's settings."""
    t = np.asarray(spikes[unit], np.float64)
    firsts = [spikes[u][0] for u in probe_units if len(spikes[u])]
    lasts = [spikes[u][-1] for u in probe_units if len(spikes[u])]
    if not firsts:
        raise ValueError(f"no spikes on {unit}'s probe")
    start, end = float(min(firsts)), float(max(lasts))
    reason = str(qc_row["qc_reason"] or "")
    return UnitQuality(
        unit=unit,
        n_spikes=int(t.size),
        span_s=(start, end),
        isi=isi_histogram(t, cfg.isi_max_s, cfg.isi_bin_s),
        acg=autocorrelogram(t, cfg.acg_window_s, cfg.acg_bin_s),
        refractory=sliding_rp_details(t, contamination, alpha),
        presence=presence_ratio(t, start, end, cfg.presence_window_s),
        rate=rate_over_session(t, start, end, cfg.rate_bin_s),
        qc_passed=bool(qc_row["qc_passed"]),
        qc_reasons=[r.strip() for r in reason.split(";") if r.strip()],
        waveform=waveform,
        waveform_missing=waveform_missing,
        ibl_criteria=ibl_criteria,
        ibl_missing=ibl_missing,
    )
