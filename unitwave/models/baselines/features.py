"""Baseline inputs and training-only cross-validation folds.  [R2]

chunk_features: each unit's summed normalised activity in n_chunks equal chunks of a
sample's window, oldest chunk first. n_chunks = 1 is the window count per unit.

gapped_folds: n_folds contiguous folds in time; a training sample whose window comes
within gap_bins of the validation fold's windows is dropped for that fold, the split
rule (max(context, 2 s)) applied inside the training data.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from unitwave.evaluation.contract import SessionData
from unitwave.splits.registry import MIN_GAP_S

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "baselines.yaml"


@dataclass(frozen=True)
class CVConfig:
    """lambdas: per-sample L2 penalties on standardised features, ascending."""

    n_folds: int
    lambdas: tuple[float, ...]


@dataclass(frozen=True)
class BaselineConfig:
    per_bin_context_bins: int
    per_bin_chunks: int
    trial_chunks: int
    cv: CVConfig
    task_penalty_ratios: tuple[float, ...] = (1.0,)
    n_jobs: int = 1


def load_baseline_config(path: str | os.PathLike = DEFAULT_CONFIG) -> BaselineConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    expected = {"per_bin", "trial", "cv", "with_task", "n_jobs"}
    if set(raw) != expected:
        raise ValueError(f"{path}: keys {sorted(raw)}, expected {sorted(expected)}")
    w = raw["with_task"]
    ratios = np.logspace(w["log10_ratio_min"], w["log10_ratio_max"], w["n_ratios"])
    cv = raw["cv"]
    lambdas = np.logspace(cv["log10_lambda_min"], cv["log10_lambda_max"], cv["n_lambdas"])
    return BaselineConfig(
        per_bin_context_bins=raw["per_bin"]["context_bins"],
        per_bin_chunks=raw["per_bin"]["n_chunks"],
        trial_chunks=raw["trial"]["n_chunks"],
        cv=CVConfig(n_folds=cv["n_folds"], lambdas=tuple(float(v) for v in lambdas)),
        task_penalty_ratios=tuple(float(r) for r in ratios),
        n_jobs=int(raw["n_jobs"]),
    )


def chunk_features(data: SessionData, n_chunks: int) -> np.ndarray:
    """(n, n_units * n_chunks) float32, unit-major: column u * n_chunks + k is unit u's
    sum over chunk k of its window (k = 0 is the oldest)."""
    context = data.context_bins
    if context % n_chunks:
        raise ValueError(f"{n_chunks} chunks do not divide a {context}-bin window")
    size = context // n_chunks
    starts = data.ends - context + 1 - data.first_bin  # column of each window's first bin
    if len(starts) == 0:
        return np.zeros((0, data.z.shape[0] * n_chunks), dtype=np.float32)
    if starts.min() < 0 or data.ends.max() - data.first_bin >= data.z.shape[1]:
        raise ValueError(f"{data.eid}: a window reaches outside the binned data")
    lo, hi = int(starts.min()), int(data.ends.max() - data.first_bin) + 1
    # Cumulative sums over just the bins the windows use, in float64 for exact differences.
    cumsum = np.zeros((data.z.shape[0], hi - lo + 1))
    np.cumsum(data.z[:, lo:hi], axis=1, dtype=np.float64, out=cumsum[:, 1:])
    features = np.empty((len(starts), data.z.shape[0], n_chunks), dtype=np.float32)
    for k in range(n_chunks):
        a = starts - lo + k * size
        features[:, :, k] = (cumsum[:, a + size] - cumsum[:, a]).T
    return features.reshape(len(starts), -1)


def gap_bins(context_bins: int, bin_ms: int) -> int:
    return max(context_bins, math.ceil(MIN_GAP_S * 1000 / bin_ms))


def gapped_folds(
    ends: np.ndarray, context_bins: int, gap: int, n_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, validation) sample indices for each of n_folds contiguous folds in time."""
    order = np.argsort(ends, kind="stable")
    starts = ends - context_bins + 1
    folds = []
    for chunk in np.array_split(order, n_folds):
        if not len(chunk):
            continue
        v_start, v_end = starts[chunk].min(), ends[chunk].max()
        clear = (ends < v_start - gap) | (starts > v_end + gap)
        folds.append((np.flatnonzero(clear), np.sort(chunk)))
    return folds
