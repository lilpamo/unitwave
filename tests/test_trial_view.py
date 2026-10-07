"""Single-trial view. Spike trains, trials and wheels here are test inputs, never shown
as data."""

import numpy as np
import pandas as pd
import pytest

from unitwave.analysis.movement import binned_wheel_speed, wheel_speed_psth
from unitwave.analysis.trial_view import (
    TrialViewConfig,
    load_trial_view_config,
    position,
    row_order,
    step,
    trial_events,
    trial_header,
    trial_view,
    trial_window,
)
from unitwave.data.session import CANONICAL_FIELDS, Capabilities, Session, TimeSeries

CFG = TrialViewConfig(pre_pad_s=0.5, post_pad_s=0.5, speed_bin_s=0.02, n_trials=3, max_trials=9)

# Trial 0: rewarded right 25% trial. Trial 1: an error with no first movement recorded.
TRIALS = pd.DataFrame(
    {
        "intervals_0": [1.0, 4.0],
        "intervals_1": [3.0, 6.0],
        "stimOn_times": [1.5, 4.5],
        "goCue_times": [1.5, 4.5],
        "firstMovement_times": [1.8, np.nan],
        "response_times": [2.0, 5.2],
        "feedback_times": [2.0, 5.2],
        "stimOff_times": [2.5, 5.6],
        "choice": [-1.0, 1.0],
        "feedbackType": [1.0, -1.0],
        "contrastLeft": [np.nan, 1.0],
        "contrastRight": [0.25, np.nan],
        "probabilityLeft": [0.5, 0.8],
    }
)
SPIKES = {
    "a": np.array([0.4, 0.5, 0.6, 1.2, 2.9, 3.49, 3.5, 3.6, 5.0]),
    "b": np.array([1.0, 6.4, 6.5]),
}


def _session(trials=TRIALS, behaviour=None, missing=None):
    t = np.arange(0.0, 10.0, 0.01)
    wheel = TimeSeries(t, np.where(t < 1.7, 0.0, 2.0 * (t - 1.7)))  # 2 rad/s after 1.7 s
    behaviour = {"wheel": wheel} if behaviour is None else behaviour
    units = pd.DataFrame(
        {"probe_name": ["probe00", "probe00"], "depths": [100.0, 300.0]},
        index=pd.Index(["a", "b"], name="unit_id"),
    )
    reasons = missing or {"behaviour.pupil_left": "leftCamera has no pupil in this fixture"}
    present = (
        {f"trials.{c}" for c in trials}
        | {f"units.{c}" for c in units}
        | {f"behaviour.{b}" for b in behaviour}
    ) & CANONICAL_FIELDS
    missing = {f: reasons.get(f, "not in this fixture") for f in CANONICAL_FIELDS - present}
    return Session(
        eid="fixture",
        time_bounds=(0.0, 10.0),
        spikes=SPIKES,
        units=units,
        trials=trials,
        behaviour=behaviour,
        available=Capabilities(frozenset(present), missing),
    )


def _units():
    """The unit table's shape for the two units: probe and depth from the tip."""
    return pd.DataFrame(
        {"probe": ["probe00", "probe00"], "depth_um": [100.0, 300.0]},
        index=pd.Index(["a", "b"], name="unit_id"),
    )


def test_default_config():
    cfg = load_trial_view_config()
    assert cfg.pre_pad_s >= 0 and cfg.post_pad_s >= 0
    assert 1 <= cfg.n_trials <= cfg.max_trials


def test_the_raster_holds_exactly_the_spikes_in_the_padded_window():
    v = trial_view(_session(), _units(), 0, cfg=CFG)
    # Window [0.5, 3.5): 0.4 is before it, 3.5 is its (open) end. Zero is the trial start.
    assert v.window.zero_s == 1.0 and (v.window.start_rel_s, v.window.stop_rel_s) == (-0.5, 2.5)
    a = v.spikes[v.rows.index("a")]
    np.testing.assert_allclose(a, [-0.5, -0.4, 0.2, 1.9, 2.49])
    np.testing.assert_allclose(v.spikes[v.rows.index("b")], [0.0])
    # Aligned to stimulus onset (1.5 s), the same spikes shift by 0.5 s.
    s = trial_view(_session(), _units(), 0, cfg=CFG, align="stimOn_times")
    np.testing.assert_allclose(s.spikes[s.rows.index("a")], [-1.0, -0.9, -0.3, 1.4, 1.99])
    assert s.window.align == "stimOn_times"


def test_neighbouring_trials_share_one_window_with_boundaries():
    v = trial_view(_session(), _units(), 0, cfg=CFG, n_trials=2)
    assert v.window.trials == (0, 1)
    assert (v.window.start_rel_s, v.window.stop_rel_s) == (-0.5, 5.5)
    assert [(b["trial"], b["start_rel_s"], b["stop_rel_s"]) for b in v.boundaries] == [
        (0, 0.0, 2.0),
        (1, 3.0, 5.0),
    ]
    np.testing.assert_allclose(v.spikes[v.rows.index("b")], [0.0, 5.4])
    with pytest.raises(ValueError, match="at most 9"):
        trial_view(_session(), _units(), 0, cfg=CFG, n_trials=10)


def test_rows_are_grouped_by_probe_then_ordered_by_depth():
    units = pd.DataFrame(
        {
            "probe": ["probe01", "probe00", "probe00", "probe01", "probe00"],
            "depth_um": [50.0, 200.0, np.nan, 900.0, 3000.0],
        },
        index=["p1_deep", "p0_mid", "p0_nodepth", "p1_top", "p0_top"],
    )
    # Within a probe the most superficial unit (largest distance from the tip) is on top;
    # a unit without a depth goes last in its probe.
    assert row_order(units) == ["p0_top", "p0_mid", "p0_nodepth", "p1_top", "p1_deep"]


def test_a_missing_event_is_listed_and_never_drawn():
    w = trial_window(TRIALS, 1, 1, "trial_start", 0.5, 0.5)
    drawn, not_recorded, absent = trial_events(TRIALS, w)
    assert not_recorded == [{"trial": 1, "event": "firstMovement_times", "label": "first movement"}]
    assert "firstMovement_times" not in {e["event"] for e in drawn}
    assert absent == []
    feedback = [e for e in drawn if e["event"] == "feedback_times"]
    assert [(e["label"], e["kind"]) for e in feedback] == [("feedback: error", "error")]
    # A column the table lacks is not "not recorded on this trial": it is absent.
    _, _, absent = trial_events(TRIALS.drop(columns="goCue_times"), w)
    assert absent == ["goCue_times"]
    # Aligning to an event this trial lacks is refused, never guessed.
    with pytest.raises(ValueError, match="first movement is not recorded on trial 1"):
        trial_window(TRIALS, 1, 1, "firstMovement_times", 0.5, 0.5)


def test_the_header_names_the_trial_and_its_conditions():
    keep = np.array([True, False])
    h = trial_header(TRIALS, 0, keep)
    assert h["trial"] == 0 and h["numbering"] == "0-based (row of the trials table)"
    assert h["side"] == "right" and h["signed_contrast"] == 0.25
    assert h["choice"] == "right (-1)" and h["outcome"] == "reward" and h["block"] == 0.5
    assert h["reaction_time_s"] == pytest.approx(0.3)
    assert h["bwm_include"] is None and h["passes_filter"] is True
    h1 = trial_header(TRIALS, 1, keep)
    assert h1["reaction_time_s"] is None and h1["passes_filter"] is False


def test_single_trial_wheel_speed_is_that_trials_row_of_the_psth_computation():
    session = _session()
    v = trial_view(session, _units(), 0, cfg=CFG, align="stimOn_times")
    lo, hi = v.wheel["speed_window_s"]
    assert (lo, hi) == pytest.approx((-1.0, 2.0))
    stim = TRIALS["stimOn_times"].to_numpy()
    rows = binned_wheel_speed(session.behaviour["wheel"], stim, (lo, hi), CFG.speed_bin_s)
    np.testing.assert_array_equal(v.wheel["speed"], rows[0])
    one = wheel_speed_psth(session.behaviour["wheel"], stim[:1], (lo, hi), CFG.speed_bin_s)
    np.testing.assert_allclose(v.wheel["speed"], one.mean)
    # 0 rad/s until the wheel turns at 1.7 s (0.2 s after stimulus onset), then 2 rad/s.
    np.testing.assert_allclose(v.wheel["speed"][v.wheel["speed_t_s"] < 0.15], 0, atol=1e-9)
    np.testing.assert_allclose(v.wheel["speed"][v.wheel["speed_t_s"] > 0.25], 2, atol=1e-9)


def test_missing_behaviour_says_why():
    v = trial_view(_session(), _units(), 0, cfg=CFG, traces=("pupil_left",))
    assert v.traces == {}
    assert v.traces_missing == {"pupil_left": "leftCamera has no pupil in this fixture"}
    nowheel = trial_view(
        _session(behaviour={}, missing={"behaviour.wheel": "no wheel here"}), _units(), 0, cfg=CFG
    )
    assert nowheel.wheel is None and nowheel.wheel_missing == "no wheel here"


def test_next_skips_trials_failing_the_filter_unless_all_trials_are_chosen():
    keep = np.array([True, False, True, True, False])
    assert step(keep, 0, +1, filtered=True) == 2
    assert step(keep, 0, +1, filtered=False) == 1
    assert step(keep, 3, +1, filtered=True) is None  # no later trial passes
    assert step(keep, 3, +1, filtered=False) == 4
    assert step(keep, 2, -1, filtered=True) == 0
    assert position(keep, 2, filtered=True) == (2, 3)
    assert position(keep, 1, filtered=True) == (None, 3)  # this trial fails the filter
    assert position(keep, 1, filtered=False) == (2, 5)
