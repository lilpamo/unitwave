"""Step 6: orientation and direction selectivity, a circular comparison. The statistic is
the vector-sum index |sum_k r_k exp(2 pi i theta_k / period)| / sum_k r_k over
presentations; its null permutes the angle labels within declared strata; p is
one-sided; Benjamini-Hochberg runs across the units tested. Spike trains are test inputs."""

import numpy as np
import pandas as pd
import pytest
import yaml

from unitwave.analysis.responsiveness import ResponseConfig
from unitwave.analysis.tasks import task_from_dict
from unitwave.analysis.tuning import SelectivityConfig, circular_index, circular_selectivity

WINDOWS = ResponseConfig((-0.2, 0.0), (0.0, 0.3), 0.001, 0.5, 0.05)
SEL = SelectivityConfig(n_permutations=2000, n_pseudo_sessions=10, seed=0, min_trials=5)
TASK = {
    "name": "gratings_test",
    "label": "Gratings test",
    "required_columns": ["intervals_0", "intervals_1"],
    "events": {"stim_on": {"label": "Stimulus onset", "column": "intervals_0"}},
    "conditions": {
        "direction": {
            "label": "Direction",
            "type": "ordinal",
            "column": "orientation",
            "level_format": "{v:g}°",
            "excluded": "blank presentations",
        },
        "orientation": {
            "label": "Orientation",
            "type": "ordinal",
            "derive": {"name": "orientation_of_direction", "columns": ["orientation"]},
            "level_format": "{v:g}°",
            "excluded": "blank presentations",
        },
        "temporal_frequency": {
            "label": "Temporal frequency",
            "type": "ordinal",
            "column": "temporal_frequency",
            "level_format": "{v:g} Hz",
            "excluded": "blank presentations",
        },
    },
    "strata": {"temporal_frequency": {"condition": "temporal_frequency"}},
    "comparisons": {
        "direction": {
            "circular": {"period": 360},
            "null_model": {"permute_within": "temporal_frequency"},
            "window": [0.0, 1.0],
        },
        "orientation": {
            "circular": {"period": 180},
            "null_model": {"permute_within": "temporal_frequency"},
            "window": [0.0, 1.0],
        },
    },
}


def test_the_index_by_hand():
    rates = np.array([[2.0, 0.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    angles = np.array([0.0, 90.0, 180.0, 270.0])
    index, preferred = circular_index(rates, angles, 180.0)
    # Period 180: 0 and 180 point the same way, 90 and 270 the opposite: |2 + 2| / 4 = 1.
    assert index[0] == pytest.approx(1.0) and preferred[0] == pytest.approx(0.0)
    assert index[1] == pytest.approx(0.0, abs=1e-12)  # flat
    assert np.isnan(index[2]) and np.isnan(preferred[2])  # silent: undefined, not 0
    index, _ = circular_index(rates[:1], angles, 360.0)
    assert index[0] == pytest.approx(0.0, abs=1e-12)  # 0 and 180 cancel over 360


def _session(rng):
    directions = np.tile(np.repeat(np.arange(0.0, 360.0, 45.0), 6), 2)
    tf = np.repeat([1.0, 4.0], 48)
    starts = 1.0 + 2.5 * np.arange(directions.size)
    trials = pd.DataFrame(
        {
            "intervals_0": starts,
            "intervals_1": starts + 2.0,
            "orientation": directions,
            "temporal_frequency": tf,
        }
    )
    trials.loc[[5, 50], ["orientation", "temporal_frequency"]] = np.nan  # blanks

    def unit(rate_of):
        out = []
        for t, d in zip(starts, directions):
            out.append(t + rng.uniform(0.0, 1.0, rng.poisson(rate_of(d))))
        return np.sort(np.concatenate(out))

    spikes = {
        "tuned": unit(lambda d: 2.0 + 18.0 * np.exp(np.cos(np.radians(d - 90.0)) * 3 - 3)),
        "flat": unit(lambda d: 8.0),
    }
    return trials, spikes


def test_a_tuned_unit_is_selective_and_a_flat_one_is_not():
    rng = np.random.default_rng(1)
    trials, spikes = _session(rng)
    task = task_from_dict(TASK)
    t = circular_selectivity(
        spikes, ["tuned", "flat"], trials, "stim_on", "direction", WINDOWS, SEL, task=task
    )
    assert t.loc["tuned", "selective"] and not t.loc["flat", "selective"]
    assert t.loc["tuned", "preferred_deg"] == pytest.approx(90.0, abs=15.0)
    assert t["n_trials"].iloc[0] == 94  # 96 presentations, 2 blanks excluded
    assert t["n_tests"].iloc[0] == 2 and t["n_null"].iloc[0] == 2000
    assert t["null"].iloc[0] == "direction permuted within temporal frequency"
    assert t["window"].iloc[0] == "0 to 1 s"
    again = circular_selectivity(
        spikes, ["tuned", "flat"], trials, "stim_on", "direction", WINDOWS, SEL, task=task
    )
    pd.testing.assert_frame_equal(t, again)  # seeded


def test_orientation_folds_opposite_directions_together():
    trials = pd.DataFrame({"orientation": [0.0, 90.0, 180.0, 270.0, 315.0, np.nan]})
    from unitwave.analysis.tasks import derive

    values, _ = derive(trials, "orientation_of_direction", ["orientation"])
    np.testing.assert_array_equal(values, [0.0, 90.0, 0.0, 90.0, 135.0, np.nan])


def test_circular_comparisons_are_validated():
    bad = yaml.safe_load(yaml.safe_dump(TASK))
    bad["comparisons"]["direction"]["levels"] = [0, 90]
    with pytest.raises(ValueError, match="either two levels or circular"):
        task_from_dict(bad)
    bad = yaml.safe_load(yaml.safe_dump(TASK))
    bad["comparisons"]["direction"]["circular"] = {"period": 0}
    with pytest.raises(ValueError, match="period"):
        task_from_dict(bad)
    bad = yaml.safe_load(yaml.safe_dump(TASK))
    bad["comparisons"]["direction"]["window"] = [1.0, 0.5]
    with pytest.raises(ValueError, match="window"):
        task_from_dict(bad)
