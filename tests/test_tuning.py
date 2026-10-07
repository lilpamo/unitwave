"""Tuning curves and selectivity. Spike trains here are test inputs, never shown as data."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from unitwave.analysis.responsiveness import ResponseConfig
from unitwave.analysis.tuning import (
    SelectivityConfig,
    auroc,
    load_selectivity_config,
    selectivity,
    stratified_permutations,
    tuning_curve,
    window_rates,
)
from unitwave.evaluation.nulls import generate_pseudo_blocks

WINDOWS = ResponseConfig((-0.2, 0.0), (0.0, 0.5), 0.001, 10.0, 0.05, 0.5, 0.25, 0.5)
SEL = SelectivityConfig(n_permutations=2000, n_pseudo_sessions=2000, seed=0, min_trials=5)


def test_default_config():
    assert load_selectivity_config() == SelectivityConfig(
        n_permutations=10000, n_pseudo_sessions=10000, seed=0, min_trials=10
    )


def test_window_rates_by_hand():
    # Window [0, 0.5) after events at 1, 2, 3, 4 s: 2, 0, 1, 3 spikes -> 4, 0, 2, 6 Hz.
    spikes = np.array([1.1, 1.3, 3.2, 4.0, 4.1, 4.49, 4.5])
    rates = window_rates(spikes, np.array([1.0, 2.0, 3.0, 4.0, np.nan]), (0.0, 0.5))
    np.testing.assert_allclose(rates, [4, 0, 2, 6, np.nan])


def test_tuning_curve_by_hand():
    # Levels left, right, left, right: left rates 4 and 2 (mean 3, SEM |4-2|/2 = 1);
    # right rates 0 and 6 (mean 3, SEM 3).
    trials = pd.DataFrame(
        {
            "contrastLeft": [0.5, np.nan, 1.0, np.nan],
            "contrastRight": [np.nan, 0.5, np.nan, 1.0],
            "stimOn_times": [1.0, 2.0, 3.0, 4.0],
        }
    )
    spikes = np.array([1.1, 1.3, 3.2, 4.0, 4.1, 4.49])
    curve = tuning_curve(spikes, trials, "stim_on", "side", (0.0, 0.5))
    assert list(curve.index) == ["left", "right"]
    np.testing.assert_allclose(curve["mean_hz"], [3, 3])
    np.testing.assert_allclose(curve["sem_hz"], [1, 3])
    assert curve["n"].tolist() == [2, 2]


def test_auroc_by_hand():
    # b beats a in 3 of 4 pairs, ties count half: (1 + 1 + 1 + 0.5) / 4 = 0.875.
    assert auroc(np.array([1.0, 2.0]), np.array([2.0, 3.0])) == pytest.approx(0.875)


def test_stratified_permutation_keeps_each_strata_label_counts():
    labels = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    strata = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    perms = stratified_permutations(labels, strata, 500, np.random.default_rng(0))
    assert perms.shape == (500, 10)
    for s in np.unique(strata):
        in_s = strata == s
        assert (perms[:, in_s].sum(axis=1) == labels[in_s].sum()).all()
    assert len({tuple(p) for p in perms}) > 1  # it does permute
    again = stratified_permutations(labels, strata, 500, np.random.default_rng(0))
    np.testing.assert_array_equal(perms, again)  # seeded (R7)


def _task(rng, n_trials=300):
    """Trials with IBL-like structure: side, contrasts, 80% correct choices."""
    side = rng.choice([-1, 1], n_trials)
    level = rng.choice([0.0, 0.0625, 0.125, 0.25, 1.0], n_trials)
    correct = rng.random(n_trials) < 0.8
    choice = np.where(correct, -side, side).astype(float)  # correct: right stim -> -1
    onset = 2.0 + np.arange(n_trials) * 3.0
    return pd.DataFrame(
        {
            "contrastLeft": np.where(side < 0, level, np.nan),
            "contrastRight": np.where(side > 0, level, np.nan),
            "choice": choice,
            "feedbackType": np.where(correct, 1.0, -1.0),
            "probabilityLeft": generate_pseudo_blocks(n_trials, seed=int(rng.integers(1e9))),
            "stimOn_times": onset,
        }
    )


def _poisson(rng, rate_of_trial, trials):
    """Spikes in each trial's [-1, 1) s around stimulus onset at that trial's rate."""
    parts = []
    for t, rate in zip(trials["stimOn_times"], rate_of_trial):
        n = rng.poisson(rate * 2.0)
        parts.append(t - 1.0 + 2.0 * rng.random(n))
    return np.sort(np.concatenate(parts))


def test_null_units_give_uniform_p_for_every_condition():
    rng = np.random.default_rng(1)
    trials = _task(rng)
    units = {f"u{i}": _poisson(rng, np.full(len(trials), 8.0), trials) for i in range(150)}
    for name in ("side", "choice", "outcome", "block"):
        result = selectivity(units, list(units), trials, "stim_on", name, WINDOWS, SEL)
        assert kstest(result["p"], "uniform").pvalue > 0.01, name


def test_benjamini_hochberg_controls_false_discoveries_on_null_data():
    # 40 all-null datasets of 40 units: any rejection is a false discovery, so it
    # should happen in about alpha of the datasets (expect ~2).
    rng = np.random.default_rng(2)
    with_any = 0
    for _ in range(40):
        trials = _task(rng, 200)
        units = {f"u{i}": _poisson(rng, np.full(200, 8.0), trials) for i in range(40)}
        result = selectivity(units, list(units), trials, "stim_on", "choice", WINDOWS, SEL)
        with_any += bool(result["selective"].any())
    assert with_any <= 5


def test_choice_is_permuted_within_contrast_because_choice_follows_the_stimulus():
    # Units driven by the stimulus side only. Choice is 80% determined by side, so a
    # plain shuffle calls them choice-selective; the stratified null does not.
    rng = np.random.default_rng(3)
    trials = _task(rng)
    side = np.where(trials["contrastRight"].notna(), 1, -1)
    rates = np.where(side > 0, 14.0, 6.0)
    units = {f"u{i}": _poisson(rng, rates, trials) for i in range(60)}
    fair = selectivity(units, list(units), trials, "stim_on", "choice", WINDOWS, SEL)
    plain = selectivity(
        units, list(units), trials, "stim_on", "choice", WINDOWS, SEL, stratify=False
    )
    assert fair["null"].iloc[0] == "choice permuted within signed contrast"
    assert (plain["p"] < 0.05).mean() > 0.9
    assert (fair["p"] < 0.05).mean() < 0.2


def test_block_uses_pseudo_sessions_because_blocks_are_autocorrelated():
    # Units whose rate drifts slowly over the session, with no block dependence. Block
    # labels are autocorrelated, so a plain permutation calls many of them selective.
    # The pseudo-session null is valid on average over block sequences (the real one is
    # a draw from the same generator), not for every single sequence: one atypical
    # sequence can flag more, so this averages over 12 sessions.
    rng = np.random.default_rng(4)
    sel = dataclasses.replace(SEL, n_pseudo_sessions=1000, n_permutations=1000)
    fair, plain = [], []
    for _ in range(12):
        trials = _task(rng, 400)
        units = {}
        for i in range(20):
            drift = np.convolve(rng.normal(0, 1, 400 + 99), np.ones(100) / 10, mode="valid")
            units[f"u{i}"] = _poisson(rng, np.clip(8 + 3 * drift, 0.5, None), trials)
        f = selectivity(units, list(units), trials, "stim_on", "block", WINDOWS, sel)
        g = selectivity(units, list(units), trials, "stim_on", "block", WINDOWS, sel, False)
        fair.append((f["p"] < 0.05).mean())
        plain.append((g["p"] < 0.05).mean())
    assert f["null"].iloc[0] == "pseudo-sessions from IBL's block generator"
    assert f["window"].iloc[0] == "baseline"
    assert np.mean(fair) <= 0.08
    assert np.mean(plain) > 3 * max(np.mean(fair), 0.02)


def test_selectivity_reports_counts_and_refuses_what_it_cannot_test():
    rng = np.random.default_rng(5)
    trials = _task(rng, 120)
    units = {"u": _poisson(rng, np.full(120, 8.0), trials)}
    row = selectivity(units, ["u"], trials, "stim_on", "outcome", WINDOWS, SEL).iloc[0]
    assert row["n_a"] + row["n_b"] == 120 and row["n_tests"] == 1
    with pytest.raises(ValueError, match="tuning curve"):
        selectivity(units, ["u"], trials, "stim_on", "contrast", WINDOWS, SEL)
    few = dataclasses.replace(SEL, min_trials=1000)
    with pytest.raises(ValueError, match="at least 1000"):
        selectivity(units, ["u"], trials, "stim_on", "side", WINDOWS, few)
    with pytest.raises(ValueError, match="choice"):
        selectivity(units, ["u"], trials.drop(columns="choice"), "stim_on", "side", WINDOWS, SEL)


def test_a_trial_mask_restricts_trials_but_keeps_the_block_structure():
    rng = np.random.default_rng(6)
    trials = _task(rng, 400)
    units = {"u": _poisson(rng, np.full(400, 8.0), trials)}
    mask = np.arange(400) % 2 == 0
    full = selectivity(units, ["u"], trials, "stim_on", "block", WINDOWS, SEL).iloc[0]
    half = selectivity(units, ["u"], trials, "stim_on", "block", WINDOWS, SEL, trial_mask=mask)
    assert half.iloc[0]["n_a"] + half.iloc[0]["n_b"] < full["n_a"] + full["n_b"]
    assert half.iloc[0]["null"] == "pseudo-sessions from IBL's block generator"
