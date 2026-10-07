import numpy as np
import pandas as pd
import pytest

from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
    TimeSeries,
)


def _units(ids=("u0", "u1")):
    df = pd.DataFrame({f: [1.0] * len(ids) for f in UNIT_FIELDS}, index=list(ids))
    df.index.name = "unit_id"
    return df


def _trials():
    return pd.DataFrame({f: [1.0, 2.0] for f in TRIAL_FIELDS})


def _capabilities(present_extra=(), missing_extra=None):
    present = {f"trials.{f}" for f in TRIAL_FIELDS}
    present |= {f"units.{f}" for f in UNIT_FIELDS}
    present |= {"behaviour.wheel", *present_extra}
    missing = {
        f"behaviour.{f}": "not in fixture"
        for f in BEHAVIOUR_FIELDS
        if f"behaviour.{f}" not in present
    }
    missing.update(missing_extra or {})
    return Capabilities(present=frozenset(present - set(missing)), missing=missing)


def make_session(**overrides):
    kwargs = dict(
        eid="fixture-eid",
        time_bounds=(0.0, 10.0),
        spikes={"u0": np.array([0.1, 0.2, 0.2]), "u1": np.array([5.0])},
        units=_units(),
        trials=_trials(),
        behaviour={"wheel": TimeSeries(np.array([0.0, 1.0]), np.array([0.0, 0.5]))},
        available=_capabilities(),
    )
    kwargs.update(overrides)
    return Session(**kwargs)


def test_valid_session_constructs():
    s = make_session()
    assert s.n_units == 2
    assert s.n_trials == 2


def test_spike_times_must_be_sorted():
    with pytest.raises(ValueError, match="sorted"):
        make_session(spikes={"u0": np.array([0.2, 0.1]), "u1": np.array([5.0])})


def test_spike_times_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        make_session(spikes={"u0": np.array([0.1, np.nan]), "u1": np.array([5.0])})


def test_spike_units_must_match_units_table():
    with pytest.raises(ValueError, match="unit ids"):
        make_session(spikes={"u0": np.array([0.1])})


def test_units_index_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        make_session(units=_units(ids=("u0", "u0")), spikes={"u0": np.array([0.1])})


def test_spikes_outside_time_bounds_rejected():
    with pytest.raises(ValueError, match="time_bounds"):
        make_session(spikes={"u0": np.array([0.1]), "u1": np.array([11.0])})


def test_trial_times_outside_time_bounds_rejected():
    trials = _trials()
    trials.loc[0, "stimOn_times"] = 42.0
    with pytest.raises(ValueError, match="stimOn_times"):
        make_session(trials=trials)


def test_trial_times_may_contain_nan():
    trials = _trials()
    trials.loc[0, "firstMovement_times"] = np.nan
    make_session(trials=trials)


def test_millisecond_time_bounds_rejected():
    # 61 min recorded in ms by mistake: 3.67e6 "seconds" is ~42 days.
    with pytest.raises(ValueError, match="seconds"):
        make_session(time_bounds=(0.0, 3_668_938.0))


def test_time_bounds_must_be_increasing():
    with pytest.raises(ValueError, match="time_bounds"):
        make_session(time_bounds=(10.0, 0.0))


def test_behaviour_outside_time_bounds_rejected():
    wheel = TimeSeries(np.array([0.0, 12.0]), np.array([0.0, 0.5]))
    with pytest.raises(ValueError, match="wheel"):
        make_session(behaviour={"wheel": wheel})


def test_every_canonical_field_must_be_declared():
    caps = _capabilities()
    undeclared = Capabilities(
        present=caps.present - {"trials.stimOff_times"}, missing=dict(caps.missing)
    )
    with pytest.raises(ValueError, match="trials.stimOff_times"):
        make_session(available=undeclared)


def test_field_cannot_be_both_present_and_missing():
    with pytest.raises(ValueError, match="both"):
        Capabilities(present=frozenset({"behaviour.wheel"}), missing={"behaviour.wheel": "x"})


def test_missing_field_must_have_a_reason():
    with pytest.raises(ValueError, match="reason"):
        Capabilities(present=frozenset(), missing={"behaviour.pose": ""})


def test_declared_present_but_absent_column_rejected():
    with pytest.raises(ValueError, match="stimOff_times"):
        make_session(trials=_trials().drop(columns=["stimOff_times"]))


def test_declared_missing_but_column_present_rejected():
    caps = _capabilities(missing_extra={"trials.stimOff_times": "not in source"})
    with pytest.raises(ValueError, match="stimOff_times"):
        make_session(available=caps)


def test_missing_field_is_absent_not_defaulted():
    caps = _capabilities(missing_extra={"trials.stimOff_times": "not in source"})
    s = make_session(trials=_trials().drop(columns=["stimOff_times"]), available=caps)
    assert "stimOff_times" not in s.trials.columns
    assert s.available.missing["trials.stimOff_times"] == "not in source"


def test_all_nan_present_field_rejected():
    trials = _trials()
    trials["stimOff_times"] = np.nan
    with pytest.raises(ValueError, match="all-NaN"):
        make_session(trials=trials)


def test_timeseries_length_mismatch_rejected():
    with pytest.raises(ValueError, match="n_samples"):
        TimeSeries(np.array([0.0, 1.0]), np.array([0.0]))


def test_timeseries_timestamps_must_be_sorted():
    with pytest.raises(ValueError, match="sorted"):
        TimeSeries(np.array([1.0, 0.0]), np.array([0.0, 0.5]))


def test_timeseries_accepts_named_multichannel_data():
    names = ("nose_x", "nose_y", "paw_x", "paw_y")
    ts = TimeSeries(np.array([0.0, 1.0, 2.0]), np.zeros((3, 4)), channel_names=names)
    assert ts.data.shape == (3, 4)
    assert ts.channel_names == names


def test_multichannel_timeseries_requires_channel_names():
    with pytest.raises(ValueError, match="channel_names"):
        TimeSeries(np.array([0.0, 1.0]), np.zeros((2, 3)))


def test_channel_names_must_match_columns():
    with pytest.raises(ValueError, match="channel_names"):
        TimeSeries(np.array([0.0, 1.0]), np.zeros((2, 3)), channel_names=("a", "b"))


def test_channel_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        TimeSeries(np.array([0.0, 1.0]), np.zeros((2, 2)), channel_names=("a", "a"))


def test_single_channel_timeseries_takes_no_channel_names():
    with pytest.raises(ValueError, match="channel_names"):
        TimeSeries(np.array([0.0, 1.0]), np.zeros(2), channel_names=("a",))


def test_cameras_have_separate_keys_and_sampling_rates():
    left = TimeSeries(np.array([0.0, 0.5, 1.0]), np.array([1.0, 2.0, 3.0]))
    body = TimeSeries(np.array([0.1, 1.1]), np.array([4.0, 5.0]))
    behaviour = {
        "wheel": TimeSeries(np.array([0.0, 1.0]), np.array([0.0, 0.5])),
        "motion_energy_left": left,
        "motion_energy_body": body,
    }
    caps = _capabilities(
        present_extra={"behaviour.motion_energy_left", "behaviour.motion_energy_body"}
    )
    s = make_session(behaviour=behaviour, available=caps)
    assert s.behaviour["motion_energy_left"].timestamps.shape == (3,)
    assert s.behaviour["motion_energy_body"].timestamps.shape == (2,)


def test_single_camera_agnostic_key_is_not_canonical():
    caps = _capabilities(missing_extra={"behaviour.motion_energy": "old single key"})
    with pytest.raises(ValueError, match="unknown"):
        make_session(available=caps)


def test_pupil_has_no_body_camera():
    caps = _capabilities(missing_extra={"behaviour.pupil_body": "body camera sees no pupil"})
    with pytest.raises(ValueError, match="unknown"):
        make_session(available=caps)
