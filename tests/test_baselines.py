import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score

from unitwave.evaluation.contract import SessionData
from unitwave.models.baselines.features import (
    BaselineConfig,
    CVConfig,
    chunk_features,
    gap_bins,
    gapped_folds,
    load_baseline_config,
)
from unitwave.models.baselines.linear import (
    GramSums,
    LogisticDecoder,
    RidgeDecoder,
    SpikesAndTaskLogistic,
    SpikesAndTaskRidge,
    fit_ridge,
)

CV = CVConfig(n_folds=5, lambdas=tuple(np.logspace(-4, 4, 17)))
N_UNITS, N_BINS, CONTEXT = 6, 4000, 50


def _data(eid="e", y=None, ends=None, seed=0, partition="train") -> SessionData:
    z = np.random.default_rng(seed).normal(size=(N_UNITS, N_BINS)).astype(np.float32)
    ends = np.arange(CONTEXT - 1, N_BINS) if ends is None else ends
    y = np.zeros(len(ends)) if y is None else y
    units = pd.DataFrame(index=[f"u{i}" for i in range(N_UNITS)])
    return SessionData(
        eid, partition, z, 100, 20, units, ends + 100, CONTEXT, y, np.zeros((len(ends), 1))
    )


def _split(data: SessionData, y: np.ndarray, cut: int):
    train = dataclasses.replace(
        data, ends=data.ends[:cut], y=y[:cut], task_features=data.task_features[:cut]
    )
    test = dataclasses.replace(
        data,
        partition="test",
        ends=data.ends[cut:],
        y=y[cut:],
        task_features=data.task_features[cut:],
    )
    return train, test


def test_chunk_features_are_window_sums():
    d = _data()
    x = chunk_features(d, 5)
    assert x.shape == (len(d.ends), N_UNITS * 5) and x.dtype == np.float32
    for i in (0, 500, len(d.ends) - 1):
        start = d.ends[i] - CONTEXT + 1 - d.first_bin
        window = d.z[:, start : start + CONTEXT].astype(np.float64)
        expected = window.reshape(N_UNITS, 5, 10).sum(axis=2).reshape(-1)
        np.testing.assert_allclose(x[i], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        chunk_features(d, 1)[:, 0], x[:, :5].sum(axis=1), rtol=1e-5, atol=1e-4
    )


def test_chunks_must_divide_the_window():
    with pytest.raises(ValueError, match="divide"):
        chunk_features(_data(), 3)


def test_gapped_folds_keep_training_windows_clear_of_validation():
    ends = np.arange(49, 4000)
    folds = gapped_folds(ends, 50, 100, 5)
    assert len(folds) == 5
    assert np.array_equal(np.sort(np.concatenate([v for _, v in folds])), np.arange(len(ends)))
    for train, val in folds:
        assert np.all(np.diff(val) == 1)  # contiguous in time
        v_start, v_end = ends[val].min() - 49, ends[val].max()
        t_start, t_end = ends[train] - 49, ends[train]
        gap = np.where(t_end < v_start, v_start - t_end - 1, t_start - v_end - 1)
        assert gap.min() >= 100


def test_gap_is_the_split_rule():
    assert gap_bins(50, 20) == 100 and gap_bins(300, 20) == 300


def test_ridge_matches_sklearn_on_standardised_features():
    rng = np.random.default_rng(1)
    x = (rng.normal(size=(500, 8)) * [1, 2, 5, 10, 0.1, 1, 3, 7]).astype(np.float32)
    y = x @ rng.normal(size=8) + 3.0 + rng.normal(size=500)
    lam = 0.05
    coef, intercept = fit_ridge(x, y, lam)
    xd = x.astype(np.float64)
    mean, std = xd.mean(axis=0), xd.std(axis=0)
    # Per-sample lambda on standardised features is sklearn's alpha = n * lambda there.
    reference = Ridge(alpha=len(x) * lam).fit((xd - mean) / std, y)
    np.testing.assert_allclose(coef, reference.coef_ / std, rtol=1e-6)
    assert intercept == pytest.approx(reference.intercept_ - mean @ (reference.coef_ / std))


def test_ridge_recovers_a_planted_signal_on_later_data():
    d = _data()
    x = chunk_features(d, 5).astype(np.float64)
    y = x @ np.random.default_rng(2).normal(size=x.shape[1]) + 0.5 * np.random.default_rng(
        3
    ).normal(size=len(x))
    train, test = _split(d, y, 3000)
    decoder = RidgeDecoder(5, CV)
    decoder.fit([train], seed=0)
    assert r2_score(test.y, decoder.predict(test)) > 0.9
    assert decoder.kind == "regression"


def test_cross_validation_regularises_noise_more_than_signal():
    d = _data()
    x = chunk_features(d, 5).astype(np.float64)
    signal = x @ np.random.default_rng(2).normal(size=x.shape[1])
    noise = np.random.default_rng(4).normal(size=len(x))
    chosen = {}
    for name, y in (("signal", signal + 0.5 * noise), ("noise", noise)):
        decoder = RidgeDecoder(5, CV)
        decoder.fit([dataclasses.replace(d, y=y)], seed=0)
        chosen[name] = decoder.models["e"]
    assert chosen["noise"].lam > 100 * chosen["signal"].lam


def test_a_choice_at_the_end_of_the_grid_is_flagged():
    d = _data()
    noise = np.random.default_rng(4).normal(size=len(d.ends))
    decoder = RidgeDecoder(5, CVConfig(n_folds=5, lambdas=(1e-4, 1e-3)))
    decoder.fit([dataclasses.replace(d, y=noise)], seed=0)
    assert decoder.models["e"].lam == 1e-3 and decoder.models["e"].at_grid_edge


def test_logistic_finds_a_planted_signal_and_returns_probabilities():
    d = _data(ends=np.arange(49, 4000, 5))
    x = chunk_features(d, 5).astype(np.float64)
    score = x @ np.random.default_rng(5).normal(size=x.shape[1])
    y = (score + np.random.default_rng(6).normal(size=len(x)) > 0).astype(float)
    train, test = _split(dataclasses.replace(d, y=y), y, 600)
    decoder = LogisticDecoder(5, CV)
    decoder.fit([train], seed=0)
    p = decoder.predict(test)
    assert np.all((p >= 0) & (p <= 1)) and roc_auc_score(test.y, p) > 0.9
    assert decoder.kind == "classification"


def test_per_session_models_refuse_unseen_sessions():
    d = _data()
    decoder = RidgeDecoder(5, CV)
    decoder.fit(
        [dataclasses.replace(d, y=np.random.default_rng(0).normal(size=len(d.ends)))], seed=0
    )
    with pytest.raises(ValueError, match="Phase 4"):
        decoder.predict(dataclasses.replace(d, eid="other"))


def test_logistic_refuses_one_class():
    with pytest.raises(ValueError, match="one class"):
        LogisticDecoder(5, CV).fit([_data(y=np.ones(N_BINS - CONTEXT + 1))], seed=0)


def test_default_config():
    cfg = load_baseline_config()
    assert isinstance(cfg, BaselineConfig)
    assert (cfg.per_bin_context_bins, cfg.per_bin_chunks, cfg.trial_chunks) == (50, 5, 1)
    assert cfg.cv.n_folds == 5 and len(cfg.cv.lambdas) == 17
    assert cfg.cv.lambdas[0] == pytest.approx(1e-4) and cfg.cv.lambdas[-1] == pytest.approx(1e4)
    np.testing.assert_allclose(cfg.task_penalty_ratios, [0.01, 0.1, 1.0, 10.0, 100.0])
    assert cfg.n_jobs == 6


def test_penalty_weights_match_sklearn_on_rescaled_features():
    rng = np.random.default_rng(7)
    x = (rng.normal(size=(400, 6)) * [1, 3, 0.5, 2, 1, 4]).astype(np.float32)
    y = x @ rng.normal(size=6) + rng.normal(size=400)
    w = np.array([1, 1, 1, 10, 10, 10.0])
    coef, intercept = GramSums.of(x, y).solve([0.02], w)[0]
    xd = x.astype(np.float64)
    mean, std = xd.mean(axis=0), xd.std(axis=0)
    scale = std * np.sqrt(w)
    reference = Ridge(alpha=len(x) * 0.02).fit((xd - mean) / scale, y)
    np.testing.assert_allclose(coef, reference.coef_ / scale, rtol=1e-6)
    assert intercept == pytest.approx(reference.intercept_ - mean @ (reference.coef_ / scale))


def _with_task(y_from, seed=0):
    d = _data(seed=seed)
    task = np.random.default_rng(seed + 100).normal(size=(len(d.ends), 3))
    spikes = chunk_features(d, 5).astype(np.float64)
    noise = 0.3 * np.random.default_rng(seed + 200).normal(size=len(d.ends))
    y = (
        task @ [1.0, -1.0, 0.5]
        if y_from == "task"
        else spikes @ np.random.default_rng(3).normal(size=spikes.shape[1])
    ) + noise
    return dataclasses.replace(d, y=y, task_features=task)


def test_spikes_and_task_learns_from_whichever_block_carries_the_signal():
    for source in ("task", "spikes"):
        train, test = _split(_with_task(source), _with_task(source).y, 3000)
        decoder = SpikesAndTaskRidge(5, CV, ratios=(0.01, 1.0, 100.0))
        decoder.fit([train], seed=0)
        assert r2_score(test.y, decoder.predict(test)) > 0.8
        assert decoder.models["e"].task_penalty_ratio in (0.01, 1.0, 100.0)
    # When only the task block carries signal, it is penalised no more than the spikes.
    decoder = SpikesAndTaskRidge(5, CV, ratios=(0.01, 1.0, 100.0))
    decoder.fit([_split(_with_task("task"), _with_task("task").y, 3000)[0]], seed=0)
    assert decoder.models["e"].task_penalty_ratio <= 1.0


def test_parallel_fit_matches_sequential():
    sessions = [
        dataclasses.replace(
            _data(eid=e, seed=i), y=np.random.default_rng(i).normal(size=N_BINS - CONTEXT + 1)
        )
        for i, e in enumerate("abc")
    ]
    one, many = RidgeDecoder(5, CV), RidgeDecoder(5, CV, n_jobs=2)
    one.fit(sessions, seed=0)
    many.fit(sessions, seed=0)
    for e in "abc":
        np.testing.assert_array_equal(one.models[e].coef, many.models[e].coef)
        assert one.models[e].lam == many.models[e].lam


def test_spikes_and_task_logistic_runs():
    d = _with_task("task")
    y = (d.y > 0).astype(float)
    train, test = _split(dataclasses.replace(d, y=y), y, 3000)
    decoder = SpikesAndTaskLogistic(5, CV, ratios=(0.1, 10.0))
    decoder.fit(
        [
            dataclasses.replace(
                train, ends=train.ends[::5], y=train.y[::5], task_features=train.task_features[::5]
            )
        ],
        seed=0,
    )
    p = decoder.predict(test)
    assert roc_auc_score(test.y, p) > 0.9
