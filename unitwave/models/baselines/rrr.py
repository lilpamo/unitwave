"""Multi-session reduced-rank regression (the contract's baseline_rrr row).

Each session s decodes with weights over (unit, time chunk) of rank r:

    y_hat = sum over u, t of X_s[u, t] * (U_s V^T)[u, t] + b_s

U_s (n_units_s, r) belongs to the session; V (n_chunks, r), the temporal filters, is
shared by all sessions. Fitted in two steps rather than by alternating least squares,
which would cost hours of cross-validation on per-bin targets:

1. per session, ridge on all (unit, chunk) features with the session's own gapped-CV
   lambda (linear.RidgeDecoder); for classification, on the 0/1 labels;
2. V = the top-r right singular vectors of every session's (n_units_s, n_chunks)
   weight matrix, stacked; then each session's U_s is refit on X_s V: ridge for
   regression, logistic for classification, with the session's step-1 lambda.

The rank is chosen from `ranks` by the same gapped folds (V re-estimated per fold),
by the mean over sessions of each session's validation loss: MSE / var(y) for
regression, log loss for classification. Like the per-session baselines, it has no
U_s for a session it wasn't trained on, and says so.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from unitwave.evaluation.contract import SessionData
from unitwave.models.baselines.features import (
    CVConfig,
    chunk_features,
    gap_bins,
    gapped_folds,
)
from unitwave.models.baselines.linear import GramSums, RidgeDecoder, fit_ridge

DEFAULT_RANKS = (1, 2, 3)


@dataclass(frozen=True)
class _Step1:
    lam: float
    weights: np.ndarray  # (n_units, n_chunks), all training samples
    fold_weights: tuple  # one (n_units, n_chunks) per fold
    folds: tuple


def _shared_filters(weights: Sequence[np.ndarray], rank: int) -> np.ndarray:
    """(n_chunks, rank) orthonormal: the top right singular vectors of the stacked weights."""
    _, _, vt = np.linalg.svd(np.vstack(weights), full_matrices=False)
    return vt[:rank].T


class _RRR:
    kind = ""

    def __init__(self, n_chunks: int, cv: CVConfig, ranks: Sequence[int] = DEFAULT_RANKS):
        self.n_chunks, self.cv = n_chunks, cv
        self.ranks = tuple(r for r in ranks if r < n_chunks)
        if not self.ranks:
            raise ValueError(f"no rank in {tuple(ranks)} is below the {n_chunks} time chunks")
        self.rank: int | None = None
        self.filters: np.ndarray | None = None  # V, (n_chunks, rank)
        self.sessions: dict[str, tuple[np.ndarray, float]] = {}  # eid -> (coef, intercept)

    def _inputs(self, data: SessionData):
        x = chunk_features(data, self.n_chunks)
        return x, x.reshape(len(x), -1, self.n_chunks)

    def fit(self, train: Sequence[SessionData], *, seed: int) -> None:
        step1: dict[str, _Step1] = {}
        for data in train:  # pass 1: per-session full-rank ridge
            x, _ = self._inputs(data)
            y = np.asarray(data.y, dtype=np.float64)
            gap = gap_bins(data.context_bins, data.bin_ms)
            folds = gapped_folds(data.ends, data.context_bins, gap, self.cv.n_folds)
            ridge = RidgeDecoder(self.n_chunks, self.cv)._fit_session(data.eid, x, y, folds, seed)
            total = GramSums.of(x, y)
            fold_weights = []
            for train_idx, _ in folds:
                held_out = np.setdiff1d(np.arange(len(x)), train_idx)
                coef, _ = (total - GramSums.of(x[held_out], y[held_out])).solve([ridge.lam])[0]
                fold_weights.append(coef.reshape(-1, self.n_chunks))
            weights = ridge.coef.reshape(-1, self.n_chunks)
            step1[data.eid] = _Step1(ridge.lam, weights, tuple(fold_weights), tuple(folds))

        n_folds = min(len(s.folds) for s in step1.values())
        filters = {
            (k, r): _shared_filters([s.fold_weights[k] for s in step1.values()], r)
            for k in range(n_folds)
            for r in self.ranks
        }
        full = {r: _shared_filters([s.weights for s in step1.values()], r) for r in self.ranks}

        scores = {r: [] for r in self.ranks}
        finals = {}
        for data in train:  # pass 2: per-session unit weights in each shared subspace
            s = step1[data.eid]
            _, x3 = self._inputs(data)
            y = np.asarray(data.y, dtype=np.float64)
            for r in self.ranks:
                losses = []
                for k in range(n_folds):
                    train_idx, val_idx = s.folds[k]
                    if not self._usable(y[train_idx]):
                        continue
                    z = (x3 @ filters[(k, r)]).reshape(len(x3), -1)
                    coef, b = self._fit_reduced(z[train_idx], y[train_idx], s.lam, seed)
                    losses.append(self._loss(y[val_idx], z[val_idx] @ coef + b))
                scores[r].append(np.nanmean(losses) if losses else np.nan)
                z = (x3 @ full[r]).reshape(len(x3), -1)
                finals[(data.eid, r)] = self._fit_reduced(z, y, s.lam, seed)

        mean_scores = {r: np.nanmean(scores[r]) for r in self.ranks}
        if not any(np.isfinite(v) for v in mean_scores.values()):
            raise ValueError("no session gave a usable validation loss for any rank")
        self.rank = min(
            self.ranks, key=lambda r: np.inf if np.isnan(mean_scores[r]) else mean_scores[r]
        )
        self.filters = full[self.rank]
        self.sessions = {eid: finals[(eid, self.rank)] for eid in step1}

    def _linear(self, data: SessionData) -> np.ndarray:
        if data.eid not in self.sessions:
            raise ValueError(
                f"{data.eid}: reduced-rank regression has no unit weights for a session it "
                "wasn't trained on (cross-session baselines are a Phase 4 decision)"
            )
        coef, intercept = self.sessions[data.eid]
        _, x3 = self._inputs(data)
        z = (x3.astype(np.float64) @ self.filters).reshape(len(x3), -1)
        return z @ coef + intercept

    def _usable(self, y_train: np.ndarray) -> bool:
        return True

    def _fit_reduced(self, z, y, lam, seed) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def _loss(self, y, linear) -> float:
        raise NotImplementedError


class RRRRegression(_RRR):
    kind = "regression"

    def _fit_reduced(self, z, y, lam, seed):
        return fit_ridge(z, y, lam)

    def _loss(self, y, linear):
        variance = np.var(y)
        return float(np.mean((y - linear) ** 2) / variance) if variance > 0 else np.nan

    def predict(self, data: SessionData) -> np.ndarray:
        return self._linear(data)


class RRRClassification(_RRR):
    kind = "classification"

    def _usable(self, y_train):
        return np.unique(y_train).size == 2  # a fold whose training part holds one class

    def _fit_reduced(self, z, y, lam, seed):
        if np.unique(y).size < 2:
            raise ValueError("a session's training labels have one class")
        mean = z.mean(axis=0)
        std = z.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        model = LogisticRegression(C=1.0 / (2.0 * len(z) * lam), max_iter=2000, random_state=seed)
        model.fit((z - mean) / std, y)
        coef = model.coef_[0].astype(np.float64) / std
        return coef, float(model.intercept_[0]) - mean @ coef

    def _loss(self, y, linear):
        return float(log_loss(y, expit(linear), labels=[0, 1]))

    def predict(self, data: SessionData) -> np.ndarray:
        return expit(self._linear(data))
