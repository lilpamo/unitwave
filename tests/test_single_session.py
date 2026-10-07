"""Single-session verdicts (S3). Predictions here are test inputs, never shown as data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from unitwave.evaluation.single_session import (
    paired_bootstrap,
    rank_test,
    session_tests,
    weighted_auroc,
)


def test_weighted_auroc_equals_sklearn_on_the_repeated_samples():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    scores = np.round(rng.random(60), 1)  # many ties
    weights = rng.integers(0, 4, size=(5, 60))  # (n_resamples, n_samples)
    got = weighted_auroc(y, scores, weights)
    assert got.shape == (5,)
    for b in range(5):
        keep = np.repeat(np.arange(60), weights[b])
        assert got[b] == pytest.approx(roc_auc_score(y[keep], scores[keep]), abs=1e-12)
    # A resample holding one class has no AUROC.
    one_class = np.where(y == 1, 1, 0)[None, :]
    assert np.isnan(weighted_auroc(y, scores, one_class))[0]


def test_rank_p_value_by_hand_and_undefined_without_draws():
    draws = np.r_[np.full(99, 0.5), 0.85]
    assert rank_test(0.8, draws) == (pytest.approx(2 / 101), 100)
    assert rank_test(0.8, np.r_[draws, np.nan]) == (pytest.approx(2 / 101), 100)
    p, n = rank_test(0.8, np.array([np.nan, np.nan]))
    assert np.isnan(p) and n == 0


def _session(rng, n_trials=120, per_trial=3, signal=(1.0, 1.0)):
    """Trials of per_trial samples sharing a trial effect; two predictors of y."""
    groups = np.repeat(np.arange(n_trials), per_trial)
    y = np.repeat(rng.integers(0, 2, n_trials), per_trial).astype(float)
    shared = np.repeat(rng.normal(size=n_trials), per_trial)  # trial-level noise
    a = signal[0] * y + shared + rng.normal(size=y.size)
    b = signal[1] * y + shared + rng.normal(size=y.size)
    return y, a, b, groups


def test_a_better_predictor_is_detected():
    y, a, b, groups = _session(np.random.default_rng(1), signal=(3.0, 0.0))
    p, n = paired_bootstrap(y, a, b, groups, np.zeros_like(groups), n_bootstrap=500, seed=0)
    assert n == 500 and p < 0.01


def test_identical_predictions_never_beat():
    y, a, _, groups = _session(np.random.default_rng(2))
    p, _ = paired_bootstrap(y, a, a, groups, np.zeros_like(groups), n_bootstrap=200, seed=0)
    assert p == 1.0


def test_bootstrap_p_is_calibrated_for_equally_good_predictors():
    # Two predictors with the same true AUROC: P(p < 0.05) must not exceed 0.05 by
    # more than sampling error, with trial-level correlation inside each session.
    rng = np.random.default_rng(3)
    ps = []
    for _ in range(300):
        y, a, b, groups = _session(rng, n_trials=60)
        ps.append(
            paired_bootstrap(y, a, b, groups, np.zeros_like(groups), n_bootstrap=300, seed=1)[0]
        )
    ps = np.array(ps)
    se = np.sqrt(0.05 * 0.95 / ps.size)
    assert (ps < 0.05).mean() <= 0.05 + 2 * se


def test_folds_are_resampled_within_themselves_and_their_aurocs_averaged():
    # Fold 0 perfectly separable around a low offset, fold 1 around a high one: pooled
    # they would look worse; per fold, a is perfect and b is noise, every resample.
    rng = np.random.default_rng(4)
    y = np.tile([0.0, 1.0], 40)
    folds = np.repeat([0, 1], 40)
    a = y + 10 * folds
    b = rng.random(80)
    groups = np.arange(80)
    p, n = paired_bootstrap(y, a, b, groups, folds, n_bootstrap=200, seed=0)
    assert p < 0.01 and n == 200


def _result(scores, shuffle, predictions, primary="auroc"):
    from unitwave.evaluation.contract import ContractResult

    per = {
        row: pd.DataFrame({primary: [v], "n_samples": [1]}, index=pd.Index(["e"], name="eid"))
        for row, v in scores.items()
    }
    return ContractResult(
        kind="classification",
        primary=primary,
        per_session=per,
        shuffle=pd.DataFrame([shuffle], index=pd.Index(["e"], name="eid")),
        verdicts=(),
        seed=0,
        n_shifts=len(shuffle),
        split_hash="h",
        ceiling_split_hash=None,
        ceiling_is_model=True,
        predictions=predictions,
    )


def _predictions(y, scores_by_row, ends):
    return {
        row: pd.DataFrame({"eid": "e", "fold": 0, "end": ends, "y": y, "prediction": s})
        for row, s in scores_by_row.items()
    }


def test_session_tests_cover_every_comparison_with_bh_and_the_gate():
    rng = np.random.default_rng(5)
    y, strong, weak, _ = _session(rng, n_trials=200, per_trial=1, signal=(3.0, 0.3))
    ends = np.arange(y.size) * 10

    def p(s):
        return 1 / (1 + np.exp(-s))

    preds = _predictions(
        y,
        {
            "model": p(strong),
            "model_with_task": p(strong),
            "baseline_ridge": p(strong),
            "null_trialstruct": p(weak),
        },
        ends,
    )
    auroc = {r: roc_auc_score(y, d["prediction"]) for r, d in preds.items()}
    result = _result({**auroc, "null_shuffle": 0.5}, np.r_[rng.uniform(0.45, 0.55, 100)], preds)
    tests = {
        (t.subject, t.row): t
        for t in session_tests(result, lambda eid, e: e // 10, n_bootstrap=300, alpha=0.05, seed=0)
    }
    assert set(tests) == {
        ("model", "null_trialstruct"),
        ("model_with_task", "null_trialstruct"),
        ("model", "null_shuffle"),
        ("model", "baseline_ridge"),
    }
    gate = tests[("model_with_task", "null_trialstruct")]
    assert gate.is_gate and gate.beats and gate.q >= gate.p
    assert tests[("model", "null_shuffle")].beats
    assert tests[("model", "null_shuffle")].p == pytest.approx(1 / 101)
    # The same decoder in both rows: never beats, and says why.
    ridge = tests[("model", "baseline_ridge")]
    assert not ridge.beats and ridge.p == 1.0 and "same predictions" in ridge.note
    # BH over the defined p-values.
    ps = np.array([t.p for t in tests.values()])
    order = np.argsort(ps)
    adjusted = np.minimum.accumulate((ps[order] * len(ps) / np.arange(1, len(ps) + 1))[::-1])[::-1]
    assert np.allclose(np.sort([t.q for t in tests.values()]), np.minimum(adjusted, 1.0))


def test_a_null_without_draws_is_undefined_and_not_counted_in_bh():
    y = np.tile([0.0, 1.0], 50)
    ends = np.arange(100)
    preds = _predictions(
        y,
        {
            r: y * 0.8 + 0.1
            for r in ("model", "model_with_task", "baseline_ridge", "null_trialstruct")
        },
        ends,
    )
    result = _result({r: 1.0 for r in preds} | {"null_shuffle": np.nan}, [np.nan] * 5, preds)
    tests = {
        (t.subject, t.row): t
        for t in session_tests(result, lambda eid, e: e, n_bootstrap=100, alpha=0.05, seed=0)
    }
    shuffle = tests[("model", "null_shuffle")]
    assert np.isnan(shuffle.p) and np.isnan(shuffle.q) and not shuffle.beats
    assert "no valid shift" in shuffle.note
