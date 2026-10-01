from pathlib import Path

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
from unitwave.env import env
from unitwave.preprocess.binning import BinnedSpikes, PreprocConfig
from unitwave.qc.units import UnitQC
from unitwave.targets.bins import movement_state_from_epochs, wheel_velocity
from unitwave.targets.config import TargetConfig, TrialWindow, load_target_config
from unitwave.targets.trials import block, choice, movement_onsets, stimulus_side

PREPROC = PreprocConfig(bin_ms=20, qc=UnitQC(1.0, ("void", "root"), 0.1))
FP = PREPROC.fingerprint()
CFG = TargetConfig(
    windows={
        "block": TrialWindow("stimOn_times", -0.4, -0.1),
        "choice": TrialWindow("firstMovement_times", -0.1, 0.0),
        "stimulus_side": TrialWindow("stimOn_times", 0.0, 0.1),
    }
)


def _session(wheel=None, trials=None) -> Session:
    units = pd.DataFrame({f: [1.0] for f in UNIT_FIELDS}, index=pd.Index(["u0"], name="unit_id"))
    units["acronym"] = "CA1"
    if trials is None:
        trials = _trials()
    behaviour = {} if wheel is None else {"wheel": wheel}
    missing = {f"behaviour.{f}": "fixture" for f in BEHAVIOUR_FIELDS if f not in behaviour}
    present = (
        {f"trials.{f}" for f in TRIAL_FIELDS}
        | {f"units.{f}" for f in UNIT_FIELDS}
        | {f"behaviour.{f}" for f in behaviour}
    )
    return Session(
        eid="e",
        time_bounds=(0.0, 20.0),
        spikes={"u0": np.array([1.0])},
        units=units,
        trials=trials,
        behaviour=behaviour,
        available=Capabilities(present=frozenset(present), missing=missing),
    )


def _trials() -> pd.DataFrame:
    # Four trials, 4 s apart. Trial 1 is excluded by bwm_include; trial 2 is in the
    # unbiased block; trial 3's movement starts exactly on a bin edge (13.5 s).
    n = 4
    start = 1.0 + 4.0 * np.arange(n)
    t = pd.DataFrame({f: np.nan for f in TRIAL_FIELDS}, index=range(n))
    t["intervals_0"], t["intervals_1"] = start, start + 3.0
    t["stimOn_times"] = start + 0.5 + 0.0071
    t["goCue_times"] = t["stimOn_times"]
    t["firstMovement_times"] = [1.8133, 5.9, 9.7777, 13.5]
    t["response_times"] = t["firstMovement_times"] + 0.3
    t["feedback_times"] = t["response_times"] + 0.01
    t["stimOff_times"] = t["feedback_times"] + 1.0
    t["choice"] = [1.0, -1.0, -1.0, 1.0]
    t["feedbackType"] = 1.0
    t["contrastLeft"] = [1.0, np.nan, 0.25, np.nan]
    t["contrastRight"] = [np.nan, 1.0, np.nan, 0.0]
    t["probabilityLeft"] = [0.8, 0.2, 0.5, 0.2]
    t["bwm_include"] = [True, False, True, True]
    return t


def _binned(n_bins=1000) -> BinnedSpikes:
    return BinnedSpikes("e", ("u0",), np.zeros((1, n_bins), np.uint16), 20, 0, FP)


def _wheel(position) -> TimeSeries:
    # The release's 100 Hz grid, offset from the bin grid like the real one (3.1 ms).
    t = 0.0031 + np.arange(1999) / 100
    return TimeSeries(t, position(t))


def test_wheel_velocity_is_displacement_per_bin():
    v = wheel_velocity(_session(_wheel(lambda t: 2.0 * t)), _binned(), CFG)
    assert v.values.shape == (1000,) and v.values.dtype == np.float32
    assert v.name == "wheel_velocity" and v.preproc_fingerprint == FP
    inside = ~np.isnan(v.values)
    np.testing.assert_allclose(v.values[inside], 2.0, rtol=1e-5)
    # Bin 0 starts before the first wheel sample, and the last bins end after the last.
    assert np.isnan(v.values[0]) and np.isnan(v.values[-1])
    assert inside.sum() == 998


def test_wheel_velocity_uses_only_the_bin_itself():
    # A step between 10.0 and 10.01 s moves only bin 500 (10.00-10.02 s).
    step = _wheel(lambda t: np.where(t < 10.005, 0.0, 0.02))
    v = wheel_velocity(_session(step), _binned(), CFG).values
    assert v[500] == pytest.approx(0.02 / 0.02, rel=1e-5)
    assert np.all(v[1:500] == 0) and np.all(v[501:-1] == 0)


def test_missing_wheel_raises_with_its_reason():
    with pytest.raises(ValueError, match="fixture"):
        wheel_velocity(_session(), _binned(), CFG)


def test_movement_state_labels_only_bins_wholly_inside_an_epoch():
    moves = np.array([[1.005, 1.1]])  # bins 51-54 wholly inside; 50 and 55 straddle
    quiet = np.array([[0.4, 1.005]])  # bins 20-49 wholly inside
    s = movement_state_from_epochs(_binned(), moves, quiet, CFG).values
    assert np.all(s[51:55] == 1) and np.all(s[20:50] == 0)
    assert np.isnan(s[50]) and np.isnan(s[55]) and np.isnan(s[19])
    assert np.isnan(s[100:]).all()


def test_overlapping_epochs_raise():
    with pytest.raises(ValueError, match="overlap"):
        movement_state_from_epochs(_binned(), np.array([[1.0, 2.0]]), np.array([[1.5, 3.0]]), CFG)


def test_choice_uses_included_trials_and_the_window_before_movement():
    t = choice(_session(), _binned(), CFG)
    assert t.context_bins == 5 and t.name == "choice"
    # Trial 1 is not bwm_include; trial 2 is included (choice is defined in any block).
    assert t.table["trial"].tolist() == [0, 2, 3]
    assert t.table["label"].tolist() == [1, 0, 1]  # clockwise (+1) -> 1
    # The window ends in the last bin that finishes by the first movement.
    assert t.table["end_bin"].tolist() == [89, 487, 674]
    ends_s = (t.table["end_bin"] + 1) * 0.02
    assert np.all(ends_s <= _trials().loc[t.table["trial"], "firstMovement_times"].to_numpy())


def test_block_drops_the_unbiased_block_and_ends_before_the_stimulus():
    t = block(_session(), _binned(), CFG)
    assert t.context_bins == 15
    assert t.table["trial"].tolist() == [0, 3]
    assert t.table["label"].tolist() == [1, 0]  # probabilityLeft 0.8 -> 1
    stim = _trials().loc[t.table["trial"], "stimOn_times"].to_numpy()
    ends_s = (t.table["end_bin"].to_numpy() + 1) * 0.02
    assert np.all(ends_s <= stim - 0.1 + 1e-9) and np.all(ends_s > stim - 0.1 - 0.02)


def test_stimulus_side_drops_zero_contrast_and_counts_it():
    t = stimulus_side(_session(), _binned(), CFG)
    assert t.context_bins == 5 and t.name == "stimulus_side"
    # Trial 1 is not bwm_include; trial 3's stimulus is 0% contrast, so it has no side
    # to decode and is excluded, and counted.
    assert t.table["trial"].tolist() == [0, 2]
    assert t.table["label"].tolist() == [1, 1]  # stimulus on the left -> 1
    assert t.excluded == {"0% contrast": 1}
    # The window is the 100 ms after stimulus onset: it ends in the last bin finishing
    # by stimOn + 0.1 s.
    assert t.table["end_bin"].tolist() == [79, 479]
    stim = _trials().loc[t.table["trial"], "stimOn_times"].to_numpy()
    ends_s = (t.table["end_bin"].to_numpy() + 1) * 0.02
    assert np.all(ends_s <= stim + 0.1 + 1e-9) and np.all(ends_s > stim + 0.1 - 0.02)
    # Other targets exclude nothing beyond their definition.
    assert choice(_session(), _binned(), CFG).excluded == {}


def test_stimulus_side_refuses_a_trial_with_both_or_neither_side():
    trials = _trials()
    trials.loc[0, "contrastRight"] = 0.5  # both sides set
    with pytest.raises(ValueError, match="one side"):
        stimulus_side(_session(trials=trials), _binned(), CFG)


def test_movement_onsets_are_event_times_of_included_trials():
    t = movement_onsets(_session(), _binned(), CFG)
    assert t.context_bins is None
    assert t.table["trial"].tolist() == [0, 2, 3]
    assert t.table["label"].tolist() == [1.8133, 9.7777, 13.5]
    assert t.table["end_bin"].tolist() == [90, 488, 675]  # the bin containing the onset


def test_trial_targets_need_bwm_include():
    with pytest.raises(ValueError, match="bwm_include"):
        choice(_session(trials=_trials().drop(columns="bwm_include")), _binned(), CFG)


def test_windows_must_be_whole_bins_and_named_events():
    with pytest.raises(ValueError, match="whole"):
        choice(
            _session(),
            _binned(),
            TargetConfig({**CFG.windows, "choice": TrialWindow("firstMovement_times", -0.05, 0.0)}),
        )
    with pytest.raises(ValueError, match="anchor"):
        TrialWindow("stimOn", -0.4, -0.1)
    with pytest.raises(ValueError, match="start"):
        TrialWindow("stimOn_times", -0.1, -0.4)


def test_grid_session_must_match():
    other = BinnedSpikes("other", ("u0",), np.zeros((1, 1000), np.uint16), 20, 0, FP)
    with pytest.raises(ValueError, match="other"):
        choice(_session(), other, CFG)


def test_default_config_and_fingerprint():
    cfg = load_target_config()
    assert cfg == CFG
    assert (
        cfg.fingerprint()
        != TargetConfig(
            {**CFG.windows, "block": TrialWindow("stimOn_times", -0.5, -0.1)}
        ).fingerprint()
    )


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EPHYS = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
BEHAVIOUR = DATA_ROOT / "bwm_compressed/bwm_behavior/2.0.0"
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"


@pytest.fixture(scope="module")
def real():
    from unitwave.data.backends.bwm_compressed import load_session_bwm
    from unitwave.preprocess.binning import preprocess_session

    session = load_session_bwm(EID, EPHYS, BEHAVIOUR)
    return session, preprocess_session(session, PREPROC)


needs_bwm = pytest.mark.skipif(
    not (EPHYS.exists() and BEHAVIOUR.exists()), reason="BWM releases not available"
)


@needs_bwm
def test_real_wheel_velocity_agrees_with_ibl_movement_epochs(real):
    from unitwave.targets.bins import movement_state

    session, binned = real
    v = wheel_velocity(session, binned, CFG).values
    state = movement_state(session, binned, BEHAVIOUR, CFG).values
    assert v.shape == state.shape == (binned.n_bins,)
    speed = np.abs(v)
    moving, quiet = speed[state == 1], speed[state == 0]
    # IBL's detector and our velocity are independent constructions from one wheel.
    assert np.nanmean(moving) > 20 * np.nanmean(quiet)
    assert (state == 1).sum() > 0 and (state == 0).sum() > (state == 1).sum()


@needs_bwm
def test_real_trial_targets_match_the_release_counts(real):
    session, binned = real
    trials = session.trials
    included = trials["bwm_include"].to_numpy(bool)
    c = choice(session, binned, CFG)
    assert len(c.table) == included.sum()
    b = block(session, binned, CFG)
    assert len(b.table) == (included & (trials["probabilityLeft"] != 0.5).to_numpy()).sum()
    s = stimulus_side(session, binned, CFG)
    contrast = trials["contrastLeft"].fillna(trials["contrastRight"]).to_numpy(float)
    assert len(s.table) == (included & (contrast > 0)).sum()
    assert s.excluded == {"0% contrast": int((included & (contrast == 0)).sum())}
    left = trials["contrastLeft"].notna().to_numpy()
    assert s.table["label"].tolist() == left[s.table["trial"]].astype(int).tolist()
    for t in (c, b, s):
        start = t.table["end_bin"] - t.context_bins + 1
        assert (start >= binned.first_bin).all()
        assert (t.table["end_bin"] < binned.first_bin + binned.n_bins).all()
