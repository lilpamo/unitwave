"""Decoding metrics, computed per session and summarised as a distribution.  [CLAUDE.md §5]

Classification (binary): balanced accuracy, AUROC, F1, log loss and expected
calibration error, from p = P(class 1). The hard prediction is class 1 when p >= 0.5.
Regression: R², Pearson correlation, MAE and RMSE.

Nothing here pools samples across sessions: `per_session` returns one row per session,
and `summarise` describes those rows (median, quartiles, range). R² pooled over
sessions with different means or variances rewards knowing each session's mean, not
decoding (see tests/test_metrics.py::test_no_pooled_r2_across_sessions).

A metric that is undefined for a session (AUROC with one class present, R² of a
constant target) is NaN, never an error and never a default value.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn import metrics as skm

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "evaluation.yaml"
CLASSIFICATION = ("balanced_accuracy", "auroc", "f1", "log_loss", "ece")
REGRESSION = ("r2", "correlation", "mae", "rmse")
_KEYS = {"ece_bins"}


@dataclass(frozen=True)
class EvalConfig:
    ece_bins: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.ece_bins, bool)
            or not isinstance(self.ece_bins, int)
            or self.ece_bins < 1
        ):
            raise ValueError(f"ece_bins must be a positive integer, got {self.ece_bins!r}")


def load_eval_config(path: str | os.PathLike = DEFAULT_CONFIG) -> EvalConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return EvalConfig(ece_bins=raw["ece_bins"])


def _check_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"expected two (n,) arrays of the same shape, got {a.shape} and {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("targets and predictions must be finite; drop undefined samples first")
    return a, b


def _check_binary(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, p = _check_pair(y, p)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("binary targets must be 0 or 1")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    return y.astype(np.int64), p


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """Top-label ECE: sum over bins of (bin share) * |accuracy - mean confidence|.

    The predicted class is 1 when p >= 0.5; its confidence is p or 1 - p. Bins are
    [k / n_bins, (k + 1) / n_bins), the last one closed.
    """
    y, p = _check_binary(y, p)
    predicted = (p >= 0.5).astype(np.int64)
    confidence = np.where(predicted == 1, p, 1.0 - p)
    correct = (predicted == y).astype(np.float64)
    bins = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, p: np.ndarray, config: EvalConfig) -> dict:
    """y: (n,) 0/1 targets. p: (n,) P(class 1)."""
    y, p = _check_binary(y, p)
    predicted = (p >= 0.5).astype(np.int64)
    both = np.unique(y).size == 2
    return {
        "balanced_accuracy": skm.balanced_accuracy_score(y, predicted) if both else np.nan,
        "auroc": skm.roc_auc_score(y, p) if both else np.nan,
        "f1": skm.f1_score(y, predicted, zero_division=np.nan),
        "log_loss": skm.log_loss(y, p, labels=[0, 1]),
        "ece": expected_calibration_error(y, p, config.ece_bins),
    }


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> dict:
    """y, y_hat: (n,) targets and predictions."""
    y, y_hat = _check_pair(y, y_hat)
    constant_target = np.ptp(y) == 0
    return {
        "r2": np.nan if constant_target else skm.r2_score(y, y_hat),
        "correlation": (
            np.nan if constant_target or np.ptp(y_hat) == 0 else np.corrcoef(y, y_hat)[0, 1]
        ),
        "mae": skm.mean_absolute_error(y, y_hat),
        "rmse": float(np.sqrt(skm.mean_squared_error(y, y_hat))),
    }


def per_session(
    kind: str,
    y: np.ndarray,
    prediction: np.ndarray,
    sessions: np.ndarray,
    config: EvalConfig,
) -> pd.DataFrame:
    """One row per session (sorted eid index): each metric and n_samples.

    kind: "classification" (prediction = P(class 1)) or "regression".
    y, prediction, sessions: (n,), sessions holding each sample's eid.
    """
    if kind not in ("classification", "regression"):
        raise ValueError(f"kind must be classification or regression, got {kind!r}")
    sessions = np.asarray(sessions)
    if sessions.shape != np.shape(y):
        raise ValueError(f"sessions has shape {sessions.shape}, targets {np.shape(y)}")
    rows = {}
    for eid in sorted(set(sessions.tolist())):
        mask = sessions == eid
        values = (
            classification_metrics(y[mask], prediction[mask], config)
            if kind == "classification"
            else regression_metrics(y[mask], prediction[mask])
        )
        rows[eid] = {**values, "n_samples": int(mask.sum())}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "eid"
    return table


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Per metric: n_sessions where defined, median, q25, q75, min, max across sessions."""
    rows = {}
    for column in table.columns:
        if column.startswith("n_") or not pd.api.types.is_numeric_dtype(table[column]):
            continue
        values = table[column].to_numpy(np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            rows[column] = {
                "n_sessions": 0,
                **dict.fromkeys(("median", "q25", "q75", "min", "max"), np.nan),
            }
            continue
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        rows[column] = {
            "n_sessions": int(values.size),
            "median": median,
            "q25": q25,
            "q75": q75,
            "min": values.min(),
            "max": values.max(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
