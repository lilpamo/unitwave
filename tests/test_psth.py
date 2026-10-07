import numpy as np
import pandas as pd
import pytest

from unitwave.analysis.events import EVENTS, event_times
from unitwave.analysis.psth import (
    bin_edges,
    peak_order,
    population_psth,
    psth,
    raster,
    selection_average,
    trial_counts,
)

# Hand-computed example (worked out on paper, see test_psth_matches_hand_computed_example).
SPIKES = np.array([0.05, 0.12, 0.15, 1.02, 1.31])
EVENTS_T = np.array([0.1, 1.0])
WINDOW = (-0.1, 0.4)
BIN = 0.1


def test_bin_edges_are_exact_and_reject_a_window_that_bins_do_not_tile():
    np.testing.assert_allclose(bin_edges((-0.1, 0.4), 0.1), [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="whole number of bins"):
        bin_edges((0.0, 0.25), 0.1)
    with pytest.raises(ValueError, match="window"):
        bin_edges((0.2, 0.1), 0.1)


def test_trial_counts_matches_hand_computed_example():
    # Trial 1 (event 0.1): relative spikes -0.05, 0.02, 0.05 fall in bins 0, 1, 1.
    # Trial 2 (event 1.0): relative spikes 0.02, 0.31 fall in bins 1, 4.
    counts = trial_counts(SPIKES, EVENTS_T, WINDOW, BIN)
    np.testing.assert_array_equal(counts, [[1, 2, 0, 0, 0], [0, 1, 0, 0, 1]])


def test_bins_are_half_open():
    # A spike exactly on an edge belongs to the bin that starts there.
    counts = trial_counts(np.array([0.0, 0.25, 0.5]), np.array([0.0]), (0.0, 0.5), 0.25)
    np.testing.assert_array_equal(counts, [[1, 1]])


def test_psth_matches_hand_computed_example():
    # Rates (Hz) = counts / 0.1: trial 1 [10,20,0,0,0], trial 2 [0,10,0,0,10].
    # Mean [5,15,0,0,5]; SEM of two values a, b is |a - b| / 2: [5,5,0,0,5].
    p = psth(SPIKES, EVENTS_T, WINDOW, BIN)
    np.testing.assert_allclose(p.bin_centers, [-0.05, 0.05, 0.15, 0.25, 0.35])
    np.testing.assert_allclose(p.mean, [5, 15, 0, 0, 5])
    np.testing.assert_allclose(p.sem, [5, 5, 0, 0, 5])
    assert p.n_trials == 2 and p.n_excluded == 0


def test_psth_baseline_subtraction_matches_hand_computed_example():
    # Baseline [-0.1, 0) rate per trial: 10 Hz and 0 Hz, subtracted trial by trial.
    # Trial 1 [0,10,-10,-10,-10], trial 2 [0,10,0,0,10]: mean [0,10,-5,-5,0], SEM [0,0,5,5,10].
    p = psth(SPIKES, EVENTS_T, WINDOW, BIN, baseline=(-0.1, 0.0))
    np.testing.assert_allclose(p.mean, [0, 10, -5, -5, 0])
    np.testing.assert_allclose(p.sem, [0, 0, 5, 5, 10])


def test_missing_events_are_excluded_and_counted_not_filled():
    p = psth(SPIKES, np.array([0.1, np.nan, 1.0]), WINDOW, BIN)
    np.testing.assert_allclose(p.mean, [5, 15, 0, 0, 5])
    assert p.n_trials == 2 and p.n_excluded == 1


def test_sem_is_nan_with_one_trial_and_psth_refuses_zero_trials():
    p = psth(SPIKES, np.array([0.1]), WINDOW, BIN)
    assert np.all(np.isnan(p.sem))
    with pytest.raises(ValueError, match="no trials"):
        psth(SPIKES, np.array([np.nan]), WINDOW, BIN)


def test_raster_returns_relative_times_inside_the_window():
    trial, rel = raster(SPIKES, EVENTS_T, WINDOW)
    np.testing.assert_array_equal(trial, [0, 0, 0, 1, 1])
    np.testing.assert_allclose(rel, [-0.05, 0.02, 0.05, 0.02, 0.31])


def test_population_psth_rows_equal_single_unit_psths_and_peak_order():
    spikes = {"a": SPIKES, "b": np.array([0.33, 1.34])}
    pop = population_psth(spikes, ["b", "a"], EVENTS_T, WINDOW, BIN)
    assert pop.shape == (2, 5)
    np.testing.assert_allclose(pop[1], psth(SPIKES, EVENTS_T, WINDOW, BIN).mean)
    # Unit "b" peaks in bin 3 (mean [0,0,0,5,5], first maximum), unit "a" in bin 1: a first.
    np.testing.assert_array_equal(peak_order(pop), [1, 0])
    mean, sem = selection_average(pop)
    np.testing.assert_allclose(mean, pop.mean(axis=0))
    np.testing.assert_allclose(sem, pop.std(axis=0, ddof=1) / np.sqrt(2))


def test_event_times_split_feedback_by_outcome_and_keep_missing_as_nan():
    trials = pd.DataFrame(
        {
            "stimOn_times": [1.0, 2.0, 3.0],
            "firstMovement_times": [1.2, np.nan, 3.2],
            "feedback_times": [1.5, 2.5, 3.5],
            "feedbackType": [1.0, -1.0, 1.0],
        }
    )
    assert set(EVENTS) == {"stim_on", "first_movement", "feedback_reward", "feedback_error"}
    np.testing.assert_array_equal(event_times(trials, "feedback_reward"), [1.5, 3.5])
    np.testing.assert_array_equal(event_times(trials, "feedback_error"), [2.5])
    np.testing.assert_array_equal(event_times(trials, "first_movement"), [1.2, np.nan, 3.2])
    with pytest.raises(ValueError, match="unknown event"):
        event_times(trials, "lick")
