"""Task definitions (step 3): IBL is one built-in definition; others are refused when
malformed. Trials tables here are test inputs, never shown as data."""

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from unitwave.analysis.tasks import (
    DEFAULT_TASK,
    DERIVATIONS,
    TASKS_DIR,
    derive,
    list_tasks,
    load_task,
    task_from_dict,
)

IBL_RAW = yaml.safe_load((TASKS_DIR / "ibl.yaml").read_text())


def test_ibl_is_a_built_in_definition_with_the_names_the_app_always_used():
    task = load_task("ibl")
    assert DEFAULT_TASK == "ibl" and "ibl" in [t["name"] for t in list_tasks()]
    assert {k: (e.label, e.column, e.when) for k, e in task.events.items()} == {
        "stim_on": ("Stimulus onset", "stimOn_times", None),
        "first_movement": ("First movement", "firstMovement_times", None),
        "feedback_reward": ("Feedback: reward", "feedback_times", ("feedbackType", 1.0)),
        "feedback_error": ("Feedback: error", "feedback_times", ("feedbackType", -1.0)),
    }
    assert {k: c.label for k, c in task.conditions.items()} == {
        "side": "Stimulus side",
        "contrast": "Signed contrast",
        "choice": "Choice",
        "outcome": "Outcome",
        "block": "Block",
        "reaction_time": "Reaction time",
    }
    assert {k: c.levels for k, c in task.comparisons.items()} == {
        "side": (-1.0, 1.0),
        "choice": (-1.0, 1.0),
        "outcome": (-1.0, 1.0),
        "block": (0.2, 0.8),
    }
    assert task.comparisons["block"].pseudo_sessions == "ibl_blocks"
    assert task.comparisons["block"].window == "baseline"
    assert task.comparisons["choice"].permute_within == "signed_contrast"
    assert list(task.trial_filters) == [
        "bwm_include",
        "exclude_nogo",
        "contrasts",
        "blocks",
        "outcomes",
    ]
    assert task.movement.stimulus == "stim_on" and task.movement.movement == "first_movement"


def _broken(change):
    raw = copy.deepcopy(IBL_RAW)
    change(raw)
    return raw


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: r.update(colour="blue"), "unknown keys"),
        (lambda r: r.pop("events"), "missing keys"),
        (lambda r: r["events"]["stim_on"].pop("column"), "stim_on"),
        (lambda r: r["events"]["feedback_reward"].update(when={"column": "x"}), "when"),
        (
            lambda r: r["conditions"]["choice"].update(
                derive={"name": "stimulus_side", "columns": ["a", "b"]}
            ),
            "either a column or a derivation",
        ),
        (
            lambda r: r["conditions"]["side"]["derive"].update(name="astrology"),
            "unknown derivation 'astrology'",
        ),
        (
            lambda r: r["conditions"]["side"]["derive"].update(columns=["contrastLeft"]),
            "takes 2 columns",
        ),
        (
            lambda r: r["conditions"]["block"].update(type="fuzzy"),
            "categorical, ordinal or continuous",
        ),
        (lambda r: r["comparisons"]["side"].update(levels=[-1, 0, 1]), "two levels"),
        (
            lambda r: r["comparisons"].update(
                colour={"levels": [0, 1], "null_model": {"permute_within": "choice"}}
            ),
            "no condition 'colour'",
        ),
        (
            lambda r: r["comparisons"]["side"].update(null_model={"permute_within": "mood"}),
            "no strata 'mood'",
        ),
        (
            lambda r: r["comparisons"]["block"].update(null_model={"pseudo_sessions": "my_blocks"}),
            "no block generator 'my_blocks'",
        ),
        (lambda r: r["comparisons"]["side"].update(window="later"), "response or baseline"),
        (lambda r: r["movement"].update(stimulus="go"), "no event 'go'"),
        (
            lambda r: r["trial_filters"]["blocks"].update(kind="maybe"),
            "flag, exclude_values or select",
        ),
    ],
)
def test_malformed_definitions_are_refused_in_plain_language(change, message):
    with pytest.raises(ValueError, match=message):
        task_from_dict(_broken(change))


def test_missing_required_columns_are_refused_naming_them():
    task = load_task("ibl")
    trials = pd.DataFrame({"intervals_0": [0.0], "stimOn_times": [0.5]})
    with pytest.raises(ValueError, match="intervals_1") as err:
        task.require(trials)
    assert "'ibl'" in str(err.value)
    # Optional features say which columns they lack.
    missing = task.unavailable(pd.DataFrame({"intervals_0": [0.0], "intervals_1": [1.0]}))
    assert missing["conditions"]["choice"] == ["choice"]
    assert missing["events"]["stim_on"] == ["stimOn_times"]


def test_derivations_by_hand():
    trials = pd.DataFrame(
        {
            "contrastLeft": [1.0, np.nan, 0.0, np.nan, 0.5],
            "contrastRight": [np.nan, 0.25, np.nan, 0.0, 0.5],  # last: both set
            "rt_end": [1.2, 1.5, np.nan, 2.0, 1.1],
            "rt_start": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    assert set(DERIVATIONS) >= {
        "signed_contrast",
        "stimulus_side",
        "signed_contrast_zero_by_side",
        "absolute_contrast",
        "median_split_difference",
    }
    sides = ["contrastLeft", "contrastRight"]
    values, _ = derive(trials, "signed_contrast", sides)
    np.testing.assert_array_equal(values, [-1.0, 0.25, 0.0, 0.0, np.nan])
    values, _ = derive(trials, "stimulus_side", sides)
    np.testing.assert_array_equal(values, [-1.0, 1.0, -1.0, 1.0, np.nan])
    values, _ = derive(trials, "absolute_contrast", sides)
    np.testing.assert_array_equal(values, [1.0, 0.25, 0.0, 0.0, np.nan])
    values, _ = derive(trials, "signed_contrast_zero_by_side", sides)
    assert values[2] < 0 < values[3] and values[2] != values[3]  # -0% and +0% differ
    values, names = derive(trials, "median_split_difference", ["rt_end", "rt_start"])
    np.testing.assert_array_equal(values, [0.0, 1.0, np.nan, 1.0, 0.0])  # median 0.35 s
    assert names == {0.0: "early (< 350 ms)", 1.0: "late (≥ 350 ms)"}


def test_a_definition_can_be_read_from_any_yaml_file(tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump({**IBL_RAW, "name": "mine", "label": "Mine"}))
    assert load_task(path).name == "mine"
    with pytest.raises(ValueError, match="no task definition 'nope'"):
        load_task("nope")


def test_ibl_trial_filter_keys_are_byte_identical_to_before_definitions():
    """Cached results and saved sets key on these strings; they must not change."""
    from unitwave.analysis.conditions import TrialFilter

    f = TrialFilter(bwm_include=True, contrasts=(1.0, 0.25))
    assert f.key() == (
        '{"blocks": [], "bwm_include": true, "contrasts": [0.25, 1.0], '
        '"exclude_nogo": false, "outcomes": []}'
    )
    assert TrialFilter().key() == (
        '{"blocks": [], "bwm_include": false, "contrasts": [], '
        '"exclude_nogo": false, "outcomes": []}'
    )
