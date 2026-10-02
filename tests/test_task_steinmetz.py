"""Step 5: the Steinmetz et al. 2019 task definition, and the derivations it adds.
Trials tables here are test inputs."""

import numpy as np
import pandas as pd

from unitwave.analysis.conditions import condition, strata_values
from unitwave.analysis.tasks import derive, load_task

NAN = np.nan


def test_difference_and_combination_by_hand():
    t = pd.DataFrame({"a": [0.5, 0.0, 1.0, NAN, 0.5], "b": [0.25, 0.0, 1.0, 0.5, 0.25]})
    values, names = derive(t, "difference", ["a", "b"])
    np.testing.assert_array_equal(values, [0.25, 0.0, 0.0, NAN, 0.25])
    assert names is None
    codes, _ = derive(t, "combination", ["a", "b"])
    # Distinct (a, b) pairs, numbered in sorted order: (0, 0), (0.5, 0.25), (1, 1).
    np.testing.assert_array_equal(codes, [1.0, 0.0, 2.0, NAN, 1.0])


def test_the_steinmetz_definition_reads_its_columns():
    task = load_task("steinmetz")
    t = pd.DataFrame(
        {
            "intervals_0": [0.0, 5.0, 10.0, 15.0],
            "intervals_1": [4.0, 9.0, 14.0, 19.0],
            "visual_stimulus_time": [1.0, 6.0, 11.0, 16.0],
            "visual_stimulus_left_contrast": [0.0, 0.25, 1.0, 0.5],
            "visual_stimulus_right_contrast": [0.0, 1.0, 0.0, 0.5],
            "response_choice": [0.0, -1.0, 1.0, 1.0],
            "feedback_type": [1.0, 1.0, 1.0, -1.0],
        }
    )
    task.require(t)
    diff = condition(t, "contrast_difference", task)
    assert diff.names == ("-100%", "+0%", "+75%")
    assert condition(t, "contrast_left", task).names == ("0%", "25%", "50%", "100%")
    choice = condition(t, "choice", task)
    assert choice.names == ("right (-1)", "no-go (0)", "left (+1)")
    assert task.comparisons["choice"].levels == (-1.0, 1.0)
    assert task.comparisons["choice"].permute_within == "contrasts"
    np.testing.assert_array_equal(strata_values(t, "contrasts", task), [0.0, 1.0, 3.0, 2.0])
    assert task.movement is None  # no movement onset in its trials
    assert task.traces == ("pupil_area",) and task.wheel is None
