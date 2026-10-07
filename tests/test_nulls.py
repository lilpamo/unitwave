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
)
from unitwave.env import env
from unitwave.evaluation.nulls import (
    NullConfig,
    bin_trialstruct_features,
    draw_shifts,
    load_null_config,
    shift_bin_target,
    shift_trial_target,
    trial_trialstruct_features,
)
from unitwave.preprocess.binning import BinnedSpikes, PreprocConfig
from unitwave.qc.units import UnitQC
from unitwave.targets.bins import BinTarget
from unitwave.targets.trials import TrialTarget

FP = PreprocConfig(bin_ms=20, qc=UnitQC(1.0, ("void", "root"), 0.1)).fingerprint()
CFG = NullConfig(
    n_shifts=20,
    min_shift_s=30.0,
    min_shift_trials=100,
    time_bin_s=0.1,
    max_time_s=3.0,
    history_trials=10,
)
N_TRIALS = 12


def _trials() -> pd.DataFrame:
    # Trial i starts at 10 + 5i s; stimulus 0.5 s later, go cue with it, movement 0.4 s
    # after that, feedback 0.5 s after the stimulus.
    start = 10.0 + 5.0 * np.arange(N_TRIALS)
    t = pd.DataFrame(index=range(N_TRIALS))
    t["intervals_0"], t["intervals_1"] = start, start + 4.0
    t["stimOn_times"] = t["goCue_times"] = start + 0.5
    t["firstMovement_times"] = start + 0.9
    t["response_times"] = start + 1.0
    t["feedback_times"] = start + 1.0
    t["stimOff_times"] = start + 2.0
    t["choice"] = np.tile([1.0, -1.0, 1.0, 0.0], 3)
    t["feedbackType"] = np.tile([1.0, -1.0, 1.0, -1.0], 3)
    t["contrastLeft"] = np.tile([1.0, np.nan, 0.25, np.nan], 3)
    t["contrastRight"] = np.tile([np.nan, 0.5, np.nan, 0.0], 3)
    t["probabilityLeft"] = [0.5] * 4 + [0.8] * 4 + [0.2] * 4
    return t[list(TRIAL_FIELDS)]


def _session(trials=None) -> Session:
    units = pd.DataFrame({f: [1.0] for f in UNIT_FIELDS}, index=pd.Index(["u0"], name="unit_id"))
    return Session(
        eid="e",
        time_bounds=(0.0, 80.0),
        spikes={"u0": np.array([1.0])},
        units=units,
        trials=_trials() if trials is None else trials,
        behaviour={},
        available=Capabilities(
            present=frozenset(
                {f"trials.{f}" for f in TRIAL_FIELDS} | {f"units.{f}" for f in UNIT_FIELDS}
            ),
            missing={f"behaviour.{f}": "fixture" for f in BEHAVIOUR_FIELDS},
        ),
    )


def _binned(counts=None) -> BinnedSpikes:
    counts = np.zeros((1, 4000), np.uint16) if counts is None else counts
    return BinnedSpikes("e", ("u0",), counts, 20, 0, FP)


def test_shifts_respect_the_minimum_and_the_seed():
    shifts = draw_shifts(10_000, 1500, 20, seed=0)
    assert len(shifts) == 20 and len(set(shifts.tolist())) == 20
    assert shifts.min() >= 1500 and shifts.max() <= 10_000 - 1500
    assert np.array_equal(shifts, draw_shifts(10_000, 1500, 20, seed=0))
    assert not np.array_equal(shifts, draw_shifts(10_000, 1500, 20, seed=1))


def test_too_short_for_the_minimum_gets_no_shifts():
    assert draw_shifts(199, 100, 20, seed=0).size == 0
    assert draw_shifts(200, 100, 20, seed=0).tolist() == [100]


def test_seed_is_required():
    with pytest.raises(TypeError):
        draw_shifts(10_000, 1500, 20)


def test_bin_shift_rotates_within_the_task_period_only():
    values = np.arange(4000, dtype=np.float32)
    target = BinTarget("e", "wheel_velocity", values, 20, 0, FP, "t")
    shifted = shift_bin_target(target, _session(), 7)
    # Task period [10, 69] s = bins 500..3449; outside it the shifted target is undefined.
    assert np.isnan(shifted.values[:500]).all() and np.isnan(shifted.values[3450:]).all()
    inside = shifted.values[500:3450]
    np.testing.assert_array_equal(inside, np.roll(values[500:3450], -7))
    assert shifted.name == "wheel_velocity:shift7"


def test_circular_shift_preserves_autocorrelation():
    rng = np.random.default_rng(3)
    x = np.convolve(rng.normal(size=4000), np.ones(25) / 25, mode="same").astype(np.float32)
    target = BinTarget("e", "wheel_velocity", x, 20, 0, FP, "t")
    inside = shift_bin_target(target, _session(), 1500).values[500:3450].astype(np.float64)
    original = x[500:3450].astype(np.float64)

    def acf(v, lag):
        v = v - v.mean()
        return float((v[:-lag] * v[lag:]).sum() / (v * v).sum())

    for lag in (1, 10, 25):
        assert acf(inside, lag) == pytest.approx(acf(original, lag), abs=0.02)
    assert abs(np.corrcoef(inside, original)[0, 1]) < 0.1


def test_trial_shift_rotates_labels_but_keeps_each_window():
    table = pd.DataFrame(
        {"trial": range(6), "label": [0, 0, 1, 1, 1, 0], "end_bin": range(100, 106)}
    )
    target = TrialTarget("e", "block", table, 15, 20, FP, "t")
    shifted = shift_trial_target(target, 2)
    assert shifted.table["label"].tolist() == [1, 1, 1, 0, 0, 0]
    assert shifted.table["end_bin"].tolist() == list(range(100, 106))
    assert shifted.table["trial"].tolist() == list(range(6))
    with pytest.raises(ValueError, match="shift"):
        shift_trial_target(target, 6)


def test_bin_features_use_task_timing_not_spikes_or_behaviour():
    features, names = bin_trialstruct_features(_session(), _binned(), CFG)
    assert features.shape == (4000, len(names)) and features.dtype == np.float32
    assert len(names) == 3 * 31 + 2
    # Spikes are never read.
    counts = np.random.default_rng(0).integers(0, 5, (1, 4000)).astype(np.uint16)
    np.testing.assert_array_equal(
        bin_trialstruct_features(_session(), _binned(counts), CFG)[0], features
    )
    # Behaviour-timed events are never read.
    moved = _trials()
    for col in ("firstMovement_times", "response_times", "feedback_times", "choice"):
        moved[col] = moved[col].to_numpy()[::-1]
    np.testing.assert_array_equal(
        bin_trialstruct_features(_session(moved), _binned(), CFG)[0], features
    )
    # Stimulus timing is.
    shifted = _trials()
    shifted["stimOn_times"] = shifted["stimOn_times"] + 0.3
    assert not np.array_equal(
        bin_trialstruct_features(_session(shifted), _binned(), CFG)[0], features
    )


def test_bin_features_encode_time_since_each_task_event():
    features, names = bin_trialstruct_features(_session(), _binned(), CFG)
    col = {n: i for i, n in enumerate(names)}
    # Bin 530 starts at 10.6 s: 0.6 s after trial 0 starts, 0.1 s after its stimulus.
    row = features[530]
    assert row[col["since_intervals_0_0.6s"]] == 1 and row[col["since_stimOn_times_0.1s"]] == 1
    assert row[col["signed_contrast"]] == pytest.approx(-1.0)
    assert row[col["block_prior"]] == pytest.approx(0.5)
    # Before the first trial every event is "later" (none yet) and the prior is 0.5.
    assert features[100, col["since_stimOn_times_later"]] == 1
    assert features[100, col["block_prior"]] == pytest.approx(0.5)


def _trial_target(name, trials=range(N_TRIALS)):
    table = pd.DataFrame({"trial": list(trials), "label": 0, "end_bin": 0})
    return TrialTarget("e", name, table, 5, 20, FP, "t")


def test_trial_history_uses_only_previous_trials():
    features, names = trial_trialstruct_features(_session(), _trial_target("choice"), CFG)
    col = {n: i for i, n in enumerate(names)}
    assert features.shape == (N_TRIALS, 3 * 10 + 2)
    # Trial 5's previous trial is trial 4: stimulus on the left (-1), choice +1, rewarded.
    assert features[5, col["side_lag1"]] == -1 and features[5, col["choice_lag1"]] == 1
    assert features[5, col["reward_lag1"]] == 1
    # No history before the first trial.
    assert (features[0, [col[f"side_lag{k}"] for k in range(1, 11)]] == 0).all()
    # Changing a later trial never changes an earlier row.
    later = _trials()
    later.loc[8:, "choice"] = -later.loc[8:, "choice"]
    changed = trial_trialstruct_features(_session(later), _trial_target("choice"), CFG)[0]
    np.testing.assert_array_equal(changed[:9], features[:9])


def test_choice_null_sees_the_current_stimulus_and_block_prior():
    features, names = trial_trialstruct_features(_session(), _trial_target("choice"), CFG)
    col = {n: i for i, n in enumerate(names)}
    assert features[1, col["signed_contrast"]] == pytest.approx(0.5)
    assert features[5, col["block_prior"]] == pytest.approx(0.8)


def test_block_null_never_sees_the_current_trial_or_its_label():
    features, names = trial_trialstruct_features(_session(), _trial_target("block"), CFG)
    assert "block_prior" not in names and "signed_contrast" not in names
    # Trial 5's own block, stimulus, choice and reward don't reach its row.
    current = _trials()
    current.loc[5, "probabilityLeft"] = 0.2
    current.loc[5, ["contrastLeft", "contrastRight"]] = [np.nan, 1.0]
    current.loc[5, ["choice", "feedbackType"]] = [1.0, 1.0]
    changed = trial_trialstruct_features(_session(current), _trial_target("block", [5]), CFG)[0]
    np.testing.assert_array_equal(changed[0], features[5])


def test_stimulus_side_null_sees_the_block_prior_but_never_the_current_stimulus():
    # S3: within a biased block the stimulus is on the prior's side 80% of the time, so
    # the prior is the task's own prediction of the side; spikes must beat it.
    features, names = trial_trialstruct_features(_session(), _trial_target("stimulus_side"), CFG)
    assert names[-1] == "block_prior" and "signed_contrast" not in names
    assert features.shape == (N_TRIALS, 3 * 10 + 1)
    col = {n: i for i, n in enumerate(names)}
    assert features[5, col["block_prior"]] == pytest.approx(0.8)
    # Trial 5's own stimulus, choice and reward don't reach its row.
    current = _trials()
    current.loc[5, ["contrastLeft", "contrastRight"]] = [np.nan, 1.0]
    current.loc[5, ["choice", "feedbackType"]] = [1.0, -1.0]
    changed = trial_trialstruct_features(
        _session(current), _trial_target("stimulus_side", [5]), CFG
    )[0]
    np.testing.assert_array_equal(changed[0], features[5])


def test_rows_follow_the_targets_trials():
    features, _ = trial_trialstruct_features(_session(), _trial_target("choice", [3, 7]), CFG)
    full, _ = trial_trialstruct_features(_session(), _trial_target("choice"), CFG)
    np.testing.assert_array_equal(features, full[[3, 7]])


def test_targets_without_a_defined_null_raise():
    with pytest.raises(ValueError, match="movement_onset"):
        trial_trialstruct_features(_session(), _trial_target("movement_onset"), CFG)


def test_default_config():
    assert load_null_config() == CFG


def test_pseudo_blocks_follow_the_ibl_protocol():
    from unitwave.evaluation.nulls import generate_pseudo_blocks

    p = generate_pseudo_blocks(600, seed=3)
    assert p.shape == (600,) and (p[:90] == 0.5).all()
    assert set(np.unique(p[90:])) <= {0.2, 0.8}
    change = np.flatnonzero(np.diff(p[90:]) != 0) + 1
    lengths = np.diff(np.concatenate([[0], change]))
    assert ((lengths >= 20) & (lengths < 100)).all()
    assert np.array_equal(p, generate_pseudo_blocks(600, seed=3))
    assert not np.array_equal(p, generate_pseudo_blocks(600, seed=4))
    assert (generate_pseudo_blocks(50, seed=0) == 0.5).all()


_TRIALS = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser() / (
    "bwm_compressed/bwm_ephys/1.2.1/metadata/trials.parquet"
)


@pytest.mark.skipif(not _TRIALS.exists(), reason="BWM release not available")
def test_pseudo_block_lengths_match_the_release():
    from scipy.stats import ks_2samp

    from unitwave.evaluation.nulls import generate_pseudo_blocks

    def complete_block_lengths(prior):
        prior = prior[prior != 0.5]
        change = np.flatnonzero(np.diff(prior) != 0) + 1
        lengths = np.diff(np.concatenate([[0], change, [len(prior)]]))
        return lengths[:-1]  # the last block is cut short by the session's end

    trials = pd.read_parquet(_TRIALS, columns=["eid", "trial_id", "probabilityLeft"])
    # BWM ephys sessions replay a few pre-generated block sequences (18 distinct openings
    # across 459 sessions), so each distinct sequence counts once, not once per session.
    sequences = {
        tuple(complete_block_lengths(t.sort_values("trial_id").probabilityLeft.to_numpy()))
        for _, t in trials.groupby("eid")
    }
    real = np.concatenate([np.array(seq) for seq in sequences])
    fake = np.concatenate(
        [complete_block_lengths(generate_pseudo_blocks(700, seed=s)) for s in range(400)]
    )
    assert len(real) > 150
    assert abs(np.mean(real) - np.mean(fake)) < 4.0
    assert ks_2samp(real, fake).pvalue > 0.05
