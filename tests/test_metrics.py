import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from unitwave.evaluation.metrics import (
    EvalConfig,
    classification_metrics,
    expected_calibration_error,
    load_eval_config,
    per_session,
    regression_metrics,
    summarise,
)

CFG = EvalConfig(ece_bins=10)


def _binary(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, n), 0.01, 0.99)
    return y, p


def test_classification_metrics_match_sklearn():
    y, p = _binary()
    m = classification_metrics(y, p, CFG)
    pred = (p >= 0.5).astype(int)
    assert m["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(y, pred))
    assert m["auroc"] == pytest.approx(skm.roc_auc_score(y, p))
    assert m["f1"] == pytest.approx(skm.f1_score(y, pred))
    assert m["log_loss"] == pytest.approx(skm.log_loss(y, p))
    assert 0 <= m["ece"] <= 1


def test_ece_is_top_label_confidence_in_equal_width_bins():
    # Predicted classes 1, 1, 0, 1 with confidences 0.9, 0.9, 0.8, 0.6; correct: no, yes...
    y, p = np.array([1, 0, 0, 1]), np.array([0.9, 0.9, 0.2, 0.6])
    # Bin 0.9: accuracy 0.5 vs 0.9, weight 2/4; bin 0.8: 1 vs 0.8, 1/4; bin 0.6: 1 vs 0.6, 1/4.
    expected = 0.5 * 0.4 + 0.25 * 0.2 + 0.25 * 0.4
    assert expected_calibration_error(y, p, 10) == pytest.approx(expected)


def test_ece_is_zero_when_confidence_matches_accuracy():
    # Confidence 0.75 everywhere and 3 of every 4 predictions right.
    y = np.tile([1, 1, 1, 0], 25)
    assert expected_calibration_error(y, np.full(100, 0.75), 10) == pytest.approx(0.0)


def test_regression_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 2, 300)
    yhat = 0.7 * y + rng.normal(0, 1, 300)
    m = regression_metrics(y, yhat)
    assert m["r2"] == pytest.approx(skm.r2_score(y, yhat))
    assert m["correlation"] == pytest.approx(np.corrcoef(y, yhat)[0, 1])
    assert m["mae"] == pytest.approx(skm.mean_absolute_error(y, yhat))
    assert m["rmse"] == pytest.approx(np.sqrt(skm.mean_squared_error(y, yhat)))


def test_undefined_metrics_are_nan_not_errors():
    one_class = classification_metrics(np.ones(20, int), np.full(20, 0.7), CFG)
    assert np.isnan(one_class["auroc"]) and np.isnan(one_class["balanced_accuracy"])
    assert np.isfinite(one_class["log_loss"]) and np.isfinite(one_class["ece"])
    flat = regression_metrics(np.full(10, 3.0), np.arange(10.0))
    assert np.isnan(flat["r2"]) and np.isnan(flat["correlation"])
    assert np.isnan(regression_metrics(np.arange(10.0), np.full(10, 3.0))["correlation"])


@pytest.mark.parametrize(
    "y, p, match",
    [
        ([0, 1, np.nan], [0.1, 0.9, 0.5], "finite"),
        ([0, 1, 2], [0.1, 0.9, 0.5], "0 or 1"),
        ([0, 1, 1], [0.1, 1.2, 0.5], "probabilities"),
        ([0, 1], [0.1, 0.9, 0.5], "shape"),
    ],
)
def test_invalid_classification_input_raises(y, p, match):
    with pytest.raises(ValueError, match=match):
        classification_metrics(np.array(y), np.array(p), CFG)


def test_no_pooled_r2_across_sessions():
    """CLAUDE.md §5: pooled R² across sessions with different means is misleading."""
    rng = np.random.default_rng(2)
    y = np.concatenate([rng.normal(0, 1, 500), 10 + rng.normal(0, 1, 500)])
    # A decoder that only knows each session's mean: it explains nothing within a session.
    yhat = np.repeat([0.0, 10.0], 500)
    sessions = np.repeat(["a", "b"], 500)
    assert skm.r2_score(y, yhat) > 0.9  # what pooling would report
    table = per_session("regression", y, yhat, sessions, CFG)
    assert list(table.index) == ["a", "b"]
    assert (table["r2"].abs() < 0.01).all()
    summary = summarise(table)
    assert abs(summary.loc["r2", "median"]) < 0.01
    assert "pooled" not in " ".join(map(str, [*summary.index, *summary.columns]))


def test_per_session_groups_and_counts():
    y, p = _binary(300)
    sessions = np.repeat(["s1", "s2", "s3"], 100)
    table = per_session("classification", y, p, sessions, CFG)
    assert table.shape[0] == 3 and (table["n_samples"] == 100).all()
    for s in ("s1", "s2", "s3"):
        mask = sessions == s
        assert table.loc[s, "auroc"] == pytest.approx(skm.roc_auc_score(y[mask], p[mask]))


def test_summary_is_a_distribution_over_defined_sessions():
    table = pd.DataFrame({"auroc": [0.6, 0.7, 0.8, np.nan], "n_samples": [10, 10, 10, 10]})
    summary = summarise(table)
    row = summary.loc["auroc"]
    assert row["n_sessions"] == 3 and row["median"] == pytest.approx(0.7)
    assert row["q25"] == pytest.approx(0.65) and row["q75"] == pytest.approx(0.75)
    assert row["min"] == pytest.approx(0.6) and row["max"] == pytest.approx(0.8)
    assert "n_samples" not in summary.index


def test_default_config():
    assert load_eval_config() == CFG
