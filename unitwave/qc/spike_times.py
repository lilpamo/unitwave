"""Unit QC from spike times only, for any source (step 4).

Some sources have no quality labels (a Phy folder without label files, Steinmetz et
al. 2019's NWB files). This rule needs only spike times and the task period:
- **task-period firing rate** (qc.units.task_firing_rates) at least `min_firing_rate_hz`;
- **IBL's sliding refractory-period test** (qc/refractory.py): contamination at most
  `refractory_contamination`, shown with confidence 1 - `refractory_alpha`. Low-rate
  units fail by design;
- **presence ratio** at least `min_presence_ratio`: the share of whole
  `presence_window_s` bins across the task period, from the first trial's start, with
  at least one spike. The last part of the task shorter than a bin isn't binned. A task
  shorter than one bin leaves it undefined, and the unit fails, saying so.

Studio doesn't read spike amplitudes from any source (Steinmetz 2019's NWB files store
them), so amplitude-based metrics are declared unavailable (UNAVAILABLE), never
estimated.

Sources with their own labels keep their rule (the IBL label, the Phy group); this
verdict is shown beside theirs, and `agreement` says how far the two agree.
Thresholds: configs/qc_spikes.yaml.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.data.session import Session
from unitwave.qc.phy import REFRACTORY, refractory_passes
from unitwave.qc.units import TASK_RATE, task_firing_rates, task_period

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qc_spikes.yaml"
PRESENCE = "presence_ratio"
_KEYS = (
    "min_firing_rate_hz",
    "refractory_contamination",
    "refractory_alpha",
    "presence_window_s",
    "min_presence_ratio",
)
# Metrics that need spike amplitudes, which Studio doesn't read from any source (some,
# like Steinmetz 2019's NWB files, store them), and why they are absent.
UNAVAILABLE = {
    "amplitude_cutoff": "needs spike amplitudes, which Studio doesn't read",
    "amplitude_median": "needs spike amplitudes, which Studio doesn't read",
}
# Criteria a reason can name, by its opening words, across this rule and the sources'
# own rules (qc.units, qc.phy), for `agreement`.
_CRITERIA = (
    ("task firing rate", "task firing rate"),
    ("refractory violations", "refractory violations"),
    ("presence ratio", "presence ratio"),
    ("label", "label"),
    ("location", "location"),
    ("located in", "location"),
    ("group", "group"),
)


@dataclass(frozen=True)
class SpikeQC:
    min_firing_rate_hz: float
    refractory_contamination: float
    refractory_alpha: float
    presence_window_s: float
    min_presence_ratio: float

    def hash(self) -> str:
        """sha256 of the thresholds; changes whenever any threshold does."""
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_spike_qc_config(path: str | os.PathLike = DEFAULT_CONFIG) -> SpikeQC:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - set(_KEYS)), sorted(set(_KEYS) - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return SpikeQC(*(float(raw[k]) for k in _KEYS))


def presence_ratio(spikes: np.ndarray, t0: float, t1: float, window: float) -> float:
    """Share of the whole `window` bins from t0 within [t0, t1] holding a spike, each
    bin [start, start + window); NaN when [t0, t1] is shorter than one bin.
    spikes: (n_spikes,) seconds."""
    n_bins = int(np.floor((t1 - t0) / window + 1e-9))
    if n_bins < 1:
        return np.nan
    spikes = np.asarray(spikes, np.float64)
    assert spikes.ndim == 1
    inside = spikes[(spikes >= t0) & (spikes < t0 + n_bins * window)]
    occupied = np.unique(np.floor((inside - t0) / window).astype(np.int64))
    return float(occupied.size / n_bins)


def spike_time_metrics(session: Session, cfg: SpikeQC) -> pd.DataFrame:
    """(n_units, 3): task_firing_rate (Hz), sliding_rp_pass, presence_ratio."""
    t0, t1 = task_period(session.trials)
    presence = [
        presence_ratio(session.spikes[u], t0, t1, cfg.presence_window_s)
        for u in session.units.index
    ]
    table = pd.DataFrame(
        {
            TASK_RATE: task_firing_rates(session),
            REFRACTORY: refractory_passes(session, cfg),
            PRESENCE: pd.Series(presence, session.units.index, dtype=np.float64),
        }
    )
    return table


def spike_unit_qc(metrics: pd.DataFrame, cfg: SpikeQC) -> pd.DataFrame:
    """(n_units, 2): `passed` and `reason` ("" when passed, else every failed criterion).
    metrics: spike_time_metrics' columns."""
    for column in (TASK_RATE, REFRACTORY, PRESENCE):
        if column not in metrics:
            raise ValueError(f"units have no {column} column")
    window = f"{cfg.presence_window_s:g} s"
    reasons = []
    for rate, refractory_ok, presence in zip(
        metrics[TASK_RATE].to_numpy(np.float64),
        metrics[REFRACTORY],
        metrics[PRESENCE].to_numpy(np.float64),
    ):
        why = []
        if np.isnan(rate):
            why.append("task firing rate missing")
        elif rate < cfg.min_firing_rate_hz:
            why.append(f"task firing rate {rate:.3g} Hz < {cfg.min_firing_rate_hz:g} Hz")
        if not refractory_ok:
            why.append(
                f"refractory violations: contamination below {cfg.refractory_contamination:g} "
                f"not shown at {1 - cfg.refractory_alpha:g} confidence"
            )
        if np.isnan(presence):
            why.append(f"presence ratio undefined: the task is shorter than one {window} bin")
        elif presence < cfg.min_presence_ratio:
            why.append(
                f"presence ratio {presence:.2g} < {cfg.min_presence_ratio:g} "
                f"({window} bins over the task)"
            )
        reasons.append("; ".join(why))
    result = pd.DataFrame({"reason": reasons}, index=metrics.index)
    result.insert(0, "passed", result["reason"] == "")
    return result


def _criterion(reason: str) -> str:
    for opening, name in _CRITERIA:
        if reason.startswith(opening):
            return name
    return reason


def _failed(reasons: pd.Series) -> dict[str, int]:
    """criterion -> units failing it, counting each unit once per criterion."""
    counts: dict[str, int] = {}
    for text in reasons:
        for name in dict.fromkeys(_criterion(r) for r in str(text).split("; ") if r):
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def agreement(table: pd.DataFrame) -> dict:
    """How far the source's own rule (qc_passed) and the spike-time rule
    (spike_qc_passed) agree on a units table: counts of each combination, the share
    agreeing, and, where they disagree, which criteria the failing rule named."""
    source = table["qc_passed"].astype(bool).to_numpy()
    spikes = table["spike_qc_passed"].astype(bool).to_numpy()
    n = len(table)
    both, neither = int((source & spikes).sum()), int((~source & ~spikes).sum())
    return {
        "n_units": n,
        "both": both,
        "source_only": int((source & ~spikes).sum()),
        "spikes_only": int((~source & spikes).sum()),
        "neither": neither,
        "agree": (both + neither) / n if n else np.nan,
        "source_only_fail": _failed(table.loc[source & ~spikes, "spike_qc_reason"]),
        "spikes_only_fail": _failed(table.loc[~source & spikes, "qc_reason"]),
    }
