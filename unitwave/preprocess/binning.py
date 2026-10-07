"""Spike times -> (n_units, n_bins) counts on a grid anchored at t = 0.  [VERSIONED, R6]

Bin k covers [k * w, (k + 1) * w) on the session clock, so every backend's bins for a
session line up by index. A spike's bin is floor(t * rate) with an integer rate, which
avoids floor(t / w) misplacing spikes (0.06 / 0.02 = 2.999...).

PREPROC_VERSION is bumped by hand whenever this code's output changes for the same
input. The fingerprint hashes it with the bin width and the unit-QC thresholds; caches
and split files record the fingerprint, so any change invalidates them.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from unitwave.data.session import Session
from unitwave.qc.rules import apply_qc
from unitwave.qc.units import DEFAULT_CONFIG as DEFAULT_QC_CONFIG
from unitwave.qc.units import UnitQC, load_qc_config

# 1: first version. 2: unit QC's firing-rate floor applies to the task-period rate.
PREPROC_VERSION = 2
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "preprocess.yaml"
_KEYS = {"bin_ms"}
_MAX_COUNT = np.iinfo(np.uint16).max


def _check_bin_ms(bin_ms: int) -> int:
    if not isinstance(bin_ms, (int, np.integer)) or bin_ms <= 0 or 1000 % bin_ms:
        raise ValueError(
            f"bin_ms must be a positive whole number of ms that divides 1000, got {bin_ms}"
        )
    return int(bin_ms)


@dataclass(frozen=True)
class PreprocConfig:
    """bin_ms, and qc: the unit QC rule binning keeps units by. IBL's UnitQC
    (configs/qc.yaml) by default; a Phy, spike-time or NWB rule for other sources (step
    8a). Every rule has a hash, so the fingerprint follows the rule used."""

    bin_ms: int
    qc: UnitQC

    def __post_init__(self) -> None:
        _check_bin_ms(self.bin_ms)

    def fingerprint(self) -> str:
        payload = {"preproc_version": PREPROC_VERSION, "bin_ms": self.bin_ms, "qc": self.qc.hash()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_preproc_config(
    path: str | os.PathLike = DEFAULT_CONFIG, qc_path: str | os.PathLike = DEFAULT_QC_CONFIG
) -> PreprocConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return PreprocConfig(bin_ms=raw["bin_ms"], qc=load_qc_config(qc_path))


@dataclass(frozen=True)
class BinnedSpikes:
    """counts: (n_units, n_bins) uint16; row i is unit_ids[i]; column j is bin first_bin + j.

    fingerprint is the PreprocConfig fingerprint when produced by `preprocess_session`,
    None when binned directly without unit QC.
    """

    eid: str
    unit_ids: tuple[str, ...]
    counts: np.ndarray
    bin_ms: int
    first_bin: int
    fingerprint: str | None = None

    @property
    def n_bins(self) -> int:
        return self.counts.shape[1]

    def bin_start_times(self) -> np.ndarray:
        """(n_bins,) seconds."""
        rate = 1000 // self.bin_ms
        return (self.first_bin + np.arange(self.n_bins, dtype=np.int64)) / rate


def bin_spikes(session: Session, bin_ms: int) -> BinnedSpikes:
    """Bin every unit in the session, in its units-table order. No QC is applied."""
    rate = 1000 // _check_bin_ms(bin_ms)
    t_start, t_end = session.time_bounds
    first_bin = int(np.floor(t_start * rate))
    n_bins = int(np.floor(t_end * rate)) - first_bin + 1

    unit_ids = tuple(session.units.index)
    counts = np.zeros((len(unit_ids), n_bins), dtype=np.uint16)
    for row, unit_id in enumerate(unit_ids):
        idx = np.floor(session.spikes[unit_id] * rate).astype(np.int64) - first_bin
        per_bin = np.bincount(idx, minlength=n_bins)
        if per_bin.size and per_bin.max() > _MAX_COUNT:
            raise ValueError(f"{unit_id}: {per_bin.max()} spikes in one bin overflows uint16")
        counts[row] = per_bin
    assert counts.shape == (len(unit_ids), n_bins)
    return BinnedSpikes(session.eid, unit_ids, counts, int(bin_ms), first_bin)


def preprocess_session(session: Session, config: PreprocConfig) -> BinnedSpikes:
    """Unit QC (the session's own rule: config.qc; IBL's by default), then binning; the
    result carries the config's fingerprint, which hashes that rule."""
    binned = bin_spikes(apply_qc(session, config.qc), config.bin_ms)
    return BinnedSpikes(
        binned.eid,
        binned.unit_ids,
        binned.counts,
        binned.bin_ms,
        binned.first_bin,
        fingerprint=config.fingerprint(),
    )
