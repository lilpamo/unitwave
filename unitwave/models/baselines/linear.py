"""Per-session ridge and logistic baselines (the contract's baseline_ridge row).

Each session gets its own model, fit on that session's training samples only.
Features are standardised with the training samples' mean and std, and the penalty is
per sample (mean loss + lambda ||w||²), with lambda chosen by gapped blocked
cross-validation inside the training samples (features.gapped_folds). A lambda at
either end of the grid is flagged (SessionModel.at_grid_edge), not silently kept.

Per-unit weights don't transfer to other sessions' units, so predicting a session
that wasn't in training raises: cross-session baselines are a Phase 4 decision.

Ridge solves every lambda and fold from one Gram matrix per session. Logistic walks
the lambda path from strong to weak regularisation, warm-started, and stops a fold's
path once its validation loss has risen twice in a row past its minimum; a lambda a
fold never reached can't be selected.

Penalties can be weighted per feature (lambda * sum w_j beta_j² on standardised
features). SpikesAndTaskRidge / SpikesAndTaskLogistic (the contract's model_with_task
row) use that to give the spike features and the task features separate penalties:
the task-to-spike ratio is chosen with lambda by the same CV. Sessions can be fit in
parallel (n_jobs); features are built in the parent and handed to workers lazily.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
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

_ROWS_PER_BLOCK = 8192
_PATIENCE = 2


def _std(variance: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.maximum(variance, 0.0))
    return np.where(std > 0, std, 1.0)  # a constant feature stays unscaled (weight 0)


@dataclass(frozen=True)
class GramSums:
    """Sufficient statistics of (X, y) for ridge: n, sum x, sum xx^T, sum xy, sum y."""

    n: int
    sx: np.ndarray
    sxx: np.ndarray
    sxy: np.ndarray
    sy: float

    @classmethod
    def of(cls, x: np.ndarray, y: np.ndarray) -> "GramSums":
        p = x.shape[1]
        sxx, sxy = np.zeros((p, p)), np.zeros(p)
        for i in range(0, len(x), _ROWS_PER_BLOCK):
            block = x[i : i + _ROWS_PER_BLOCK].astype(np.float64)
            sxx += block.T @ block
            sxy += block.T @ y[i : i + _ROWS_PER_BLOCK]
        return cls(len(x), x.sum(axis=0, dtype=np.float64), sxx, sxy, float(y.sum()))

    def __sub__(self, other: "GramSums") -> "GramSums":
        return GramSums(
            self.n - other.n,
            self.sx - other.sx,
            self.sxx - other.sxx,
            self.sxy - other.sxy,
            self.sy - other.sy,
        )

    def solve(self, lambdas: Sequence[float], weights=None) -> list[tuple[np.ndarray, float]]:
        """(coef, intercept) on the raw feature scale, for each per-sample lambda.

        weights: optional (p,) per-feature penalty weights on the standardised scale.
        """
        mean_x, mean_y = self.sx / self.n, self.sy / self.n
        gram = self.sxx - self.n * np.outer(mean_x, mean_x)
        cross = self.sxy - self.n * mean_x * mean_y
        # Rescaling a standardised column by 1/sqrt(w) turns penalty w * beta² into beta².
        std = _std(np.diag(gram) / self.n) * (1.0 if weights is None else np.sqrt(weights))
        eigvals, eigvecs = np.linalg.eigh(gram / np.outer(std, std))
        projected = eigvecs.T @ (cross / std)
        out = []
        for lam in lambdas:
            coef = eigvecs @ (projected / (np.maximum(eigvals, 0.0) + self.n * lam)) / std
            out.append((coef, mean_y - mean_x @ coef))
        return out


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """(coef, intercept) of ridge on standardised x with per-sample penalty lam."""
    return GramSums.of(x, np.asarray(y, dtype=np.float64)).solve([lam])[0]


@dataclass(frozen=True)
class SessionModel:
    lam: float
    coef: np.ndarray  # on the raw feature scale
    intercept: float
    validation_loss: tuple[float, ...]  # mean over folds, one per lambda (inf: unreached)
    at_grid_edge: bool
    task_penalty_ratio: float | None = None  # SpikesAndTask only


class _PerSession:
    kind = ""

    def __init__(self, n_chunks: int, cv: CVConfig, *, n_jobs: int = 1):
        self.n_chunks, self.cv, self.n_jobs = n_chunks, cv, n_jobs
        self.models: dict[str, SessionModel] = {}

    def _inputs(self, train: Sequence[SessionData]):
        for data in train:
            gap = gap_bins(data.context_bins, data.bin_ms)
            folds = gapped_folds(data.ends, data.context_bins, gap, self.cv.n_folds)
            yield data.eid, self._features(data), np.asarray(data.y, dtype=np.float64), folds

    def fit(self, train: Sequence[SessionData], *, seed: int) -> None:
        if self.n_jobs == 1:
            for eid, x, y, folds in self._inputs(train):
                self.models[eid] = self._fit_session(eid, x, y, folds, seed)
            return
        jobs = (
            (eid, delayed(self._fit_session)(eid, x, y, folds, seed))
            for eid, x, y, folds in self._inputs(train)
        )
        eids = []

        def calls():
            for eid, call in jobs:
                eids.append(eid)
                yield call

        fitted = Parallel(n_jobs=self.n_jobs, pre_dispatch="2*n_jobs")(calls())
        self.models.update(zip(eids, fitted))

    def _model(self, data: SessionData) -> SessionModel:
        if data.eid not in self.models:
            raise ValueError(
                f"{data.eid}: a per-session baseline has no model for a session it wasn't "
                "trained on (cross-session baselines are a Phase 4 decision)"
            )
        return self.models[data.eid]

    def _select(self, losses: np.ndarray) -> tuple[int, bool]:
        best = int(np.argmin(losses))
        return best, best in (0, len(self.cv.lambdas) - 1)

    def _linear(self, data: SessionData) -> np.ndarray:
        model = self._model(data)
        x = self._features(data).astype(np.float64)
        return x @ model.coef + model.intercept

    def _features(self, data: SessionData) -> np.ndarray:
        return chunk_features(data, self.n_chunks)

    def _fit_session(self, eid, x, y, folds, seed, weights=None) -> SessionModel:
        raise NotImplementedError


class RidgeDecoder(_PerSession):
    kind = "regression"

    def _fit_session(self, eid, x, y, folds, seed, weights=None) -> SessionModel:
        total = GramSums.of(x, y)
        losses = np.zeros(len(self.cv.lambdas))
        for train_idx, val_idx in folds:
            held_out = np.setdiff1d(np.arange(len(x)), train_idx)
            fold = total - GramSums.of(x[held_out], y[held_out])
            xv, yv = x[val_idx].astype(np.float64), y[val_idx]
            losses += [
                np.mean((yv - xv @ c - b) ** 2) for c, b in fold.solve(self.cv.lambdas, weights)
            ]
        losses /= len(folds)
        best, edge = self._select(losses)
        coef, intercept = total.solve([self.cv.lambdas[best]], weights)[0]
        return SessionModel(self.cv.lambdas[best], coef, intercept, tuple(losses), edge)

    def predict(self, data: SessionData) -> np.ndarray:
        return self._linear(data)


class LogisticDecoder(_PerSession):
    kind = "classification"

    @staticmethod
    def _standardise(x: np.ndarray, weights=None):
        """Standardise, then rescale by 1/sqrt(w) so the penalty weights become uniform."""
        mean = x.mean(axis=0, dtype=np.float64)
        std = _std(x.var(axis=0, dtype=np.float64)) * (1.0 if weights is None else np.sqrt(weights))
        return (x - mean) / std, mean, std

    def _fold_losses(self, x, y, train_idx, val_idx, seed, weights=None) -> np.ndarray:
        """Validation log loss per lambda, strongest regularisation first; inf if unreached."""
        xt, mean, std = self._standardise(x[train_idx], weights)
        xv = (x[val_idx] - mean) / std
        losses = np.full(len(self.cv.lambdas), np.inf)
        model = LogisticRegression(max_iter=2000, warm_start=True, random_state=seed)
        best, rises = np.inf, 0
        for i in reversed(range(len(self.cv.lambdas))):
            # sklearn minimises C * sum(log loss) + ||w||² / 2: per-sample lambda <-> C.
            model.set_params(C=1.0 / (2.0 * len(train_idx) * self.cv.lambdas[i]))
            model.fit(xt, y[train_idx])
            losses[i] = log_loss(y[val_idx], model.predict_proba(xv)[:, 1], labels=[0, 1])
            best, rises = (losses[i], 0) if losses[i] < best else (best, rises + 1)
            if rises >= _PATIENCE:
                break
        return losses

    def _fit_session(self, eid, x, y, folds, seed, weights=None) -> SessionModel:
        if np.unique(y).size < 2:
            raise ValueError(f"{eid}: training labels have one class; logistic can't be fit")
        per_fold = [
            self._fold_losses(x, y, tr, va, seed, weights)
            for tr, va in folds
            if np.unique(y[tr]).size == 2  # a fold whose training part holds one class
        ]
        if not per_fold:
            raise ValueError(f"{eid}: no cross-validation fold has both classes in training")
        losses = np.mean(per_fold, axis=0)
        best, edge = self._select(losses)
        lam = self.cv.lambdas[best]
        xs, mean, std = self._standardise(x, weights)
        final = LogisticRegression(C=1.0 / (2.0 * len(x) * lam), max_iter=2000, random_state=seed)
        final.fit(xs, y)
        coef = final.coef_[0].astype(np.float64) / std
        intercept = float(final.intercept_[0]) - mean @ coef
        return SessionModel(lam, coef, intercept, tuple(losses), edge)

    def predict(self, data: SessionData) -> np.ndarray:
        # expit is overflow-safe.
        return expit(self._linear(data))


class _TaskFeatures:
    """Fits the task-variable features instead of spikes: the null_trialstruct row.

    Reads data.task_features only; the contract passes this row's data with z = None.
    """

    def __init__(self, cv: CVConfig, *, n_jobs: int = 1):
        super().__init__(n_chunks=1, cv=cv, n_jobs=n_jobs)

    def _features(self, data: SessionData) -> np.ndarray:
        return np.asarray(data.task_features, dtype=np.float32)


class TrialStructureRidge(_TaskFeatures, RidgeDecoder):
    pass


class TrialStructureLogistic(_TaskFeatures, LogisticDecoder):
    pass


class _SpikesAndTask:
    """The model's spike features plus the null_trialstruct features: model_with_task.

    The spike block has penalty weight 1 and the task block weight `ratio`, with the
    ratio chosen from `ratios` jointly with lambda by the same gapped CV (the lowest
    mean validation loss over every (ratio, lambda)). Separate penalties stop ~100-400
    spike features from drowning ~30 task features under one shared lambda.
    """

    def __init__(self, n_chunks: int, cv: CVConfig, ratios, *, n_jobs: int = 1):
        super().__init__(n_chunks=n_chunks, cv=cv, n_jobs=n_jobs)
        self.ratios = tuple(float(r) for r in ratios)

    def _features(self, data: SessionData) -> np.ndarray:
        spikes = chunk_features(data, self.n_chunks)
        return np.hstack([spikes, np.asarray(data.task_features, dtype=np.float32)])

    def _fit_session(self, eid, x, y, folds, seed, weights=None) -> SessionModel:
        n_task = self._n_task
        best = None
        for ratio in self.ratios:
            w = np.concatenate([np.ones(x.shape[1] - n_task), np.full(n_task, ratio)])
            model = super()._fit_session(eid, x, y, folds, seed, w)
            if best is None or min(model.validation_loss) < min(best.validation_loss):
                best = dataclasses.replace(model, task_penalty_ratio=ratio)
        return best

    def fit(self, train: Sequence[SessionData], *, seed: int) -> None:
        first = train[0] if len(train) else None
        self._n_task = 0 if first is None else first.task_features.shape[1]
        super().fit(train, seed=seed)


class SpikesAndTaskRidge(_SpikesAndTask, RidgeDecoder):
    pass


class SpikesAndTaskLogistic(_SpikesAndTask, LogisticDecoder):
    pass
