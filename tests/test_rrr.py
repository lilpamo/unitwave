import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score, roc_auc_score

from unitwave.evaluation.contract import SessionData
from unitwave.models.baselines.features import CVConfig, chunk_features
from unitwave.models.baselines.rrr import RRRClassification, RRRRegression

CV = CVConfig(n_folds=5, lambdas=tuple(np.logspace(-4, 4, 17)))
N_BINS, CONTEXT, CHUNKS = 3000, 50, 5
UNITS = {"a": 5, "b": 8, "c": 6, "d": 7}


def _session(eid, v, noise=0.3, seed=0, classify=False):
    """Session whose target uses weights u_s v^T over (unit, chunk): v is shared."""
    rng = np.random.default_rng([sorted(UNITS).index(eid), seed])
    n_units = UNITS[eid]
    z = rng.normal(size=(n_units, N_BINS)).astype(np.float32)
    ends = np.arange(CONTEXT - 1, N_BINS)
    units = pd.DataFrame(index=[f"u{i}" for i in range(n_units)])
    probe = SessionData(
        eid, "train", z, 0, 20, units, ends, CONTEXT, np.zeros(len(ends)), np.zeros((len(ends), 1))
    )
    x3 = chunk_features(probe, CHUNKS).reshape(len(ends), n_units, CHUNKS).astype(np.float64)
    u = rng.normal(size=(n_units, v.shape[1]))
    y = np.einsum("nut,ut->n", x3, u @ v.T) + noise * rng.normal(size=len(ends))
    if classify:
        y = (y > 0).astype(np.float64)
    return dataclasses.replace(probe, y=y)


def _halves(d, cut=2200):
    train = dataclasses.replace(
        d, ends=d.ends[:cut], y=d.y[:cut], task_features=d.task_features[:cut]
    )
    test = dataclasses.replace(
        d, partition="test", ends=d.ends[cut:], y=d.y[cut:], task_features=d.task_features[cut:]
    )
    return train, test


def _shared(rank, seed=0):
    v, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(CHUNKS, rank)))
    return v


def test_recovers_shared_temporal_filters_and_rank_one():
    v = _shared(1)
    halves = [_halves(_session(e, v)) for e in UNITS]
    model = RRRRegression(CHUNKS, CV)
    model.fit([train for train, _ in halves], seed=0)
    assert model.rank == 1
    assert abs(float(model.filters[:, 0] @ v[:, 0])) > 0.99
    for _, test in halves:
        assert r2_score(test.y, model.predict(test)) > 0.8


def test_picks_rank_two_when_the_shared_filters_are_two():
    v = _shared(2, seed=1)
    model = RRRRegression(CHUNKS, CV)
    model.fit([_halves(_session(e, v, noise=0.1))[0] for e in UNITS], seed=0)
    assert model.rank == 2
    # The recovered subspace is the planted one.
    np.testing.assert_allclose(np.linalg.svd(model.filters.T @ v)[1], [1, 1], atol=0.02)


def test_filters_are_shared_and_orthonormal():
    model = RRRRegression(CHUNKS, CV)
    model.fit([_halves(_session(e, _shared(1)))[0] for e in UNITS], seed=0)
    np.testing.assert_allclose(model.filters.T @ model.filters, np.eye(model.rank), atol=1e-10)
    assert set(model.sessions) == set(UNITS)


def test_classification_variant():
    v = _shared(1)
    halves = [_halves(_session(e, v, classify=True)) for e in UNITS]
    model = RRRClassification(CHUNKS, CV)
    model.fit([train for train, _ in halves], seed=0)
    for _, test in halves:
        p = model.predict(test)
        assert np.all((p >= 0) & (p <= 1)) and roc_auc_score(test.y, p) > 0.9


def test_unseen_session_is_refused():
    model = RRRRegression(CHUNKS, CV)
    model.fit([_halves(_session(e, _shared(1)))[0] for e in ("a", "b")], seed=0)
    with pytest.raises(ValueError, match="Phase 4"):
        model.predict(_session("c", _shared(1)))


def test_ranks_must_be_reduced():
    with pytest.raises(ValueError, match="below"):
        RRRRegression(1, CV)
