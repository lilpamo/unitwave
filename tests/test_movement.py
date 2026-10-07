"""Movement controls. Spike trains and wheels here are test inputs, never shown as data."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from unitwave.analysis.movement import (
    MovementConfig,
    load_movement_config,
    movement_free,
    movement_locking,
    reaction_times,
    wheel_speed_psth,
)
from unitwave.analysis.responsiveness import ResponseConfig, responsiveness
from unitwave.data.session import TimeSeries

CFG = MovementConfig(pre_window=(-0.2, 0.0), post_window=(0.0, 0.2), n_permutations=2000, seed=0)
RESP = ResponseConfig((-0.2, 0.0), (0.0, 0.3), 0.005, 10.0, 0.05, 0.5, 0.25, 0.5)


def test_default_config():
    assert load_movement_config() == MovementConfig((-0.2, 0.0), (0.0, 0.2), 10000, 0)


def test_reaction_times_and_movement_free_trials_by_hand():
    trials = pd.DataFrame(
        {
            "stimOn_times": [1.0, 2.0, 3.0, 4.0, 5.0],
            "firstMovement_times": [1.1, 2.5, np.nan, 4.3, 5.31],
        }
    )
    np.testing.assert_allclose(reaction_times(trials), [0.1, 0.5, np.nan, 0.3, 0.31])
    # Movement after the response window's end (0.3 s): trials 1 and 4. Exactly at the end
    # is not after it; a trial without a movement time is not known to be movement-free.
    assert movement_free(trials, 0.3).tolist() == [False, True, False, False, True]
    with pytest.raises(ValueError, match="firstMovement_times"):
        movement_free(trials.drop(columns="firstMovement_times"), 0.3)


def test_wheel_speed_psth_by_hand():
    # Position 0 until 2 s, then 3 rad/s: speed 0 before each event at 2 s, 3 after.
    t = np.arange(0, 4, 0.01)
    wheel = TimeSeries(t, np.where(t < 2, 0.0, 3.0 * (t - 2)))
    p = wheel_speed_psth(wheel, np.array([2.0, 2.0]), (-0.5, 0.5), 0.1)
    np.testing.assert_allclose(p.mean[:4], 0, atol=1e-9)
    np.testing.assert_allclose(p.mean[-4:], 3, atol=1e-9)
    assert p.n_trials == 2
    with pytest.raises(ValueError, match="outside the wheel"):
        wheel_speed_psth(wheel, np.array([3.9]), (-0.5, 0.5), 0.1)


def _task(rng, n=300):
    """Irregularly spaced trials (2.5-4 s apart), IBL-like contrasts, every RT known."""
    level = rng.choice([0.0625, 0.125, 0.25, 1.0], n)
    side = rng.choice([-1, 1], n)
    stim = 2.0 + np.cumsum(rng.uniform(2.5, 4.0, n))
    rt = rng.uniform(0.1, 0.8, n)
    return pd.DataFrame(
        {
            "contrastLeft": np.where(side < 0, level, np.nan),
            "contrastRight": np.where(side > 0, level, np.nan),
            "stimOn_times": stim,
            "firstMovement_times": stim + rt,
        }
    )


def _spikes(rng, trials, lock=None, base=5.0, burst=40.0):
    """Poisson at `base` Hz over the whole session, plus a 100 ms burst after each trial's
    stimulus ('stim') or first movement ('move')."""
    end = float(trials["stimOn_times"].iloc[-1]) + 5.0
    parts = [end * rng.random(rng.poisson(base * end))]
    if lock:
        for s, m in zip(trials["stimOn_times"], trials["firstMovement_times"]):
            t0 = s if lock == "stim" else m
            parts.append(t0 + 0.1 * rng.random(rng.poisson(burst * 0.1)))
    return np.sort(np.concatenate(parts))


def test_a_movement_locked_unit_is_labelled_and_a_stimulus_locked_one_is_not():
    rng = np.random.default_rng(1)
    trials = _task(rng)
    units = {
        "move": _spikes(rng, trials, "move"),
        "stim": _spikes(rng, trials, "stim"),
        "none": _spikes(rng, trials),
    }
    result = movement_locking(units, list(units), trials, CFG)
    assert result.loc["move", "p"] < 0.01
    assert result.loc["stim", "p"] > 0.05
    assert result.loc["move", "n_trials"] == 300
    assert result["null"].iloc[0] == "reaction times permuted within signed contrast"


def test_movement_free_trials_separate_stimulus_from_movement():
    # Bursts locked to movement make the unit 'respond' at stimulus onset over all trials,
    # because movement follows within ~0.1-0.8 s; on trials whose movement comes after the
    # response window, the stimulus-aligned response is gone.
    rng = np.random.default_rng(2)
    trials = _task(rng, 400)
    units = {f"u{i}": _spikes(rng, trials, "move") for i in range(20)}
    stim = trials["stimOn_times"].to_numpy()
    everything = responsiveness(units, list(units), stim, RESP)
    free = movement_free(trials, RESP.response_window[1])
    only_free = responsiveness(units, list(units), stim[free], RESP)
    assert everything["responsive"].mean() > 0.9
    assert only_free["responsive"].mean() < 0.2
    assert only_free["n_trials"].iloc[0] == int(free.sum())


def test_the_locking_null_is_calibrated_on_units_without_locking():
    rng = np.random.default_rng(3)
    trials = _task(rng)
    units = {f"u{i}": _spikes(rng, trials, "stim") for i in range(150)}  # stimulus-locked only
    result = movement_locking(units, list(units), trials, CFG)
    assert kstest(result["p"], "uniform").pvalue > 0.01
    assert result["locked"].sum() <= 3


def test_locking_is_deterministic_and_needs_movement_times():
    rng = np.random.default_rng(4)
    trials = _task(rng, 120)
    units = {"u": _spikes(rng, trials, "move")}
    a = movement_locking(units, ["u"], trials, CFG)
    assert a.equals(movement_locking(units, ["u"], trials, CFG))
    with pytest.raises(ValueError, match="firstMovement_times"):
        movement_locking(units, ["u"], trials.drop(columns="firstMovement_times"), CFG)


def test_reaction_time_splits_early_and_late_at_the_median():
    from unitwave.analysis.conditions import available_conditions, condition

    trials = pd.DataFrame(
        {
            "stimOn_times": [1.0, 2.0, 3.0, 4.0, 5.0],
            "firstMovement_times": [1.1, 2.3, np.nan, 4.2, 5.4],
        }
    )
    # RTs 0.1, 0.3, -, 0.2, 0.4: median 0.25; the trial without a movement is excluded.
    assert "reaction_time" in available_conditions(trials)
    c = condition(trials, "reaction_time")
    np.testing.assert_array_equal(c.values, [0, 1, np.nan, 0, 1])
    assert c.names == ("early (< 250 ms)", "late (≥ 250 ms)")
    assert c.n_excluded == 1 and "first movement" in c.excluded
