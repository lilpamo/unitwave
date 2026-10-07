"""Normalisation of binned counts, with statistics fit on training data only.  [R3, R6]

`fit_normalizer` takes a Split, never a bare session, so nothing can be normalised
without declaring one (docs/SPLITS_AND_LEAKAGE.md). It reads only the training data:
the train block of each within_session session, or the train partition's sessions.
Calibration and test data are never read.

The mode follows from the split kind:
- within_session -> "per_unit": each unit's mean and std over its train block.
- leave_one_block_out -> "per_unit" per fold: over that fold's training intervals.
- any other kind -> "pooled": one mean and std over every unit and bin of the train
  partition, applied to every unit. Test units in those splits never appear in
  training; normalising train units per unit but test units pooled would itself shift
  the model's inputs between train and test.

A std is never below that of a Poisson unit firing at the QC minimum rate,
sqrt(min_firing_rate_hz * bin_s), so a unit nearly silent in training does not blow
up. Sums are exact integer sums, so the statistics do not depend on session order.

The Normalizer serialises (with a hash) into the model artifact. Changing what this
module outputs for the same input bumps PREPROC_VERSION.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields

import numpy as np

from unitwave.preprocess.binning import BinnedSpikes, PreprocConfig
from unitwave.splits.registry import Split, bin_range

MODES = ("per_unit", "pooled")


@dataclass(frozen=True)
class Normalizer:
    """z = (counts - mean) / std, per unit.

    pooled: [mean, std] in "pooled" mode, else None.
    units: in "per_unit" mode, eid -> {"unit_ids", "mean", "std"} (lists, one entry per
      unit, in BinnedSpikes order); else empty.
    """

    mode: str
    split_hash: str
    preproc_fingerprint: str
    min_std: float
    pooled: list | None
    units: dict

    @property
    def hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {**asdict(self), "hash": self.hash}

    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
        data = dict(data)
        stored = data.pop("hash", None)
        expected = {f.name for f in fields(cls)}
        if set(data) != expected:
            raise ValueError(f"normalizer fields {sorted(data)}, expected {sorted(expected)}")
        normalizer = cls(**data)
        if normalizer.hash != stored:
            raise ValueError("normalizer content does not match its hash; it was edited")
        return normalizer

    def stats(self, eid: str, unit_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(n_units,) mean and std for these units of this session."""
        n = len(unit_ids)
        if self.mode == "pooled":
            return np.full(n, self.pooled[0]), np.full(n, self.pooled[1])
        record = self.units.get(eid)
        if record is None:
            raise ValueError(f"{eid}: no per-unit statistics; it has no training data")
        if list(unit_ids) != record["unit_ids"]:
            raise ValueError(f"{eid}: units differ from the ones the statistics were fit on")
        return np.asarray(record["mean"]), np.asarray(record["std"])

    def transform(self, binned: BinnedSpikes) -> np.ndarray:
        """(n_units, n_bins) float32 z-scores of binned.counts."""
        if binned.fingerprint != self.preproc_fingerprint:
            raise ValueError(
                f"{binned.eid}: binned with fingerprint {binned.fingerprint}, the "
                f"statistics with {self.preproc_fingerprint}"
            )
        mean, std = self.stats(binned.eid, binned.unit_ids)
        z = binned.counts.astype(np.float32)
        z -= mean.astype(np.float32)[:, None]
        z /= std.astype(np.float32)[:, None]
        assert z.shape == binned.counts.shape
        return z

    def inverse_transform(self, z: np.ndarray, eid: str, unit_ids: Sequence[str]) -> np.ndarray:
        """(n_units, n_bins) float32 counts from z-scores."""
        if z.ndim != 2 or z.shape[0] != len(unit_ids):
            raise ValueError(f"expected shape ({len(unit_ids)}, n_bins), got {z.shape}")
        mean, std = self.stats(eid, unit_ids)
        counts = z.astype(np.float32)  # a copy
        counts *= std.astype(np.float32)[:, None]
        counts += mean.astype(np.float32)[:, None]
        return counts


def _sums(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-row sums of counts and of squared counts, (n_units,) int64 each.

    Row by row, so a whole session is never held as int64.
    """
    s1 = np.zeros(counts.shape[0], dtype=np.int64)
    s2 = np.zeros(counts.shape[0], dtype=np.int64)
    for i, row in enumerate(counts):
        r = row.astype(np.int64)
        s1[i], s2[i] = r.sum(), r @ r
    return s1, s2


def _mean_std(s1, s2, n: int, min_std: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(s1, dtype=np.float64) / n
    var = np.maximum(np.asarray(s2, dtype=np.float64) / n - mean**2, 0.0)
    return mean, np.maximum(np.sqrt(var), min_std)


def _train_block(split: Split, binned: BinnedSpikes) -> np.ndarray:
    """(n_units, n_train_bins) counts of the session's within_session train block."""
    first, last = bin_range(split.sessions[binned.eid]["blocks"]["train"], split.preproc["bin_ms"])
    start, stop = first - binned.first_bin, last - binned.first_bin + 1
    if start < 0 or stop > binned.n_bins:
        raise ValueError(
            f"{binned.eid}: train block bins {first}..{last} are outside the binned data"
        )
    return binned.counts[:, start:stop]


def _fold_train_counts(split: Split, binned: BinnedSpikes, fold: int) -> np.ndarray:
    """(n_units, n_train_bins): a leave_one_block_out fold's training intervals, joined."""
    parts = []
    for block in split.sessions[binned.eid]["folds"][fold]["blocks"]["train"]:
        first, last = bin_range(block, split.preproc["bin_ms"])
        start, stop = first - binned.first_bin, last - binned.first_bin + 1
        if start < 0 or stop > binned.n_bins:
            raise ValueError(
                f"{binned.eid}: training bins {first}..{last} are outside the binned data"
            )
        parts.append(binned.counts[:, start:stop])
    return np.concatenate(parts, axis=1)


def fit_normalizer(
    split: Split,
    binned: Mapping[str, BinnedSpikes],
    preproc: PreprocConfig,
    *,
    fold: int | None = None,
) -> Normalizer:
    """Fit on the split's training data. binned may hold any sessions; only train is read.

    binned: eid -> preprocess_session output with the split's preprocessing fingerprint.
    fold: required for leave_one_block_out (and only there): per-unit statistics over
    that fold's training intervals, for every session that has the fold. The
    Normalizer's split_hash is then "<split hash>#fold<k>".
    """
    if (fold is None) != (split.kind != "leave_one_block_out"):
        raise ValueError("fold is required for leave_one_block_out splits and only for them")
    fingerprint = preproc.fingerprint()
    if split.preproc["fingerprint"] != fingerprint:
        raise ValueError(
            f"split was built for preprocessing fingerprint {split.preproc['fingerprint']}, "
            f"this config's is {fingerprint}"
        )
    train = sorted(split.partitions["train"])
    missing = [eid for eid in train if eid not in binned]
    if missing:
        raise ValueError(f"binned data lacks {len(missing)} training sessions, e.g. {missing[:3]}")
    for eid in train:
        if binned[eid].eid != eid or binned[eid].fingerprint != fingerprint:
            raise ValueError(f"{eid}: binned data is not this session under this fingerprint")
    min_std = math.sqrt(preproc.qc.min_firing_rate_hz * preproc.bin_ms / 1000)

    if split.kind == "leave_one_block_out":
        units = {}
        for eid in train:
            if fold >= len(split.sessions[eid]["folds"]):
                continue
            counts = _fold_train_counts(split, binned[eid], fold)
            mean, std = _mean_std(*_sums(counts), counts.shape[1], min_std)
            units[eid] = {
                "unit_ids": list(binned[eid].unit_ids),
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        return Normalizer("per_unit", f"{split.hash}#fold{fold}", fingerprint, min_std, None, units)

    if split.kind == "within_session":
        units = {}
        for eid in train:
            counts = _train_block(split, binned[eid])
            mean, std = _mean_std(*_sums(counts), counts.shape[1], min_std)
            units[eid] = {
                "unit_ids": list(binned[eid].unit_ids),
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        return Normalizer("per_unit", split.hash, fingerprint, min_std, None, units)

    s1 = s2 = n = 0
    for eid in train:
        a, b = _sums(binned[eid].counts)
        s1, s2, n = s1 + int(a.sum()), s2 + int(b.sum()), n + binned[eid].counts.size
    mean, std = _mean_std(s1, s2, n, min_std)
    return Normalizer("pooled", split.hash, fingerprint, min_std, [float(mean), float(std)], {})
