import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unitwave.data.manifest import Manifest, manifest_versions
from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
    TimeSeries,
)
from unitwave.env import env
from unitwave.evaluation.contract import evaluate
from unitwave.evaluation.data import SplitData
from unitwave.evaluation.nulls import load_null_config, shift_bin_target, shift_trial_target
from unitwave.models.baselines.features import load_baseline_config
from unitwave.models.baselines.linear import (
    LogisticDecoder,
    RidgeDecoder,
    SpikesAndTaskLogistic,
    SpikesAndTaskRidge,
    TrialStructureLogistic,
    TrialStructureRidge,
)
from unitwave.models.baselines.rrr import RRRClassification, RRRRegression
from unitwave.preprocess.binning import load_preproc_config
from unitwave.splits.registry import bin_range, within_session

PREPROC = load_preproc_config()
N_TRIALS, DURATION = 40, 200.0
EIDS = ("e0", "e1", "e2", "e3", "e4")


def _trials(seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = 10.0 + 4.5 * np.arange(N_TRIALS)
    t = pd.DataFrame(index=range(N_TRIALS))
    t["intervals_0"], t["intervals_1"] = start, start + 4.0
    t["stimOn_times"] = t["goCue_times"] = start + 0.5
    t.loc[0, ["stimOn_times", "goCue_times"]] = start[0] + 0.2  # block window starts pre-trial
    t["firstMovement_times"] = t["stimOn_times"] + 0.3
    t["response_times"] = t["firstMovement_times"] + 0.2
    t["feedback_times"] = t["response_times"] + 0.01
    t["stimOff_times"] = t["feedback_times"] + 1.0
    t["choice"] = rng.choice([-1.0, 1.0], N_TRIALS)
    t["feedbackType"] = rng.choice([-1.0, 1.0], N_TRIALS)
    right = rng.random(N_TRIALS) < 0.5
    t["contrastLeft"] = np.where(right, np.nan, 0.5)
    t["contrastRight"] = np.where(right, 0.5, np.nan)
    t["probabilityLeft"] = np.repeat([0.8, 0.2, 0.8, 0.2], 10)
    t = t[list(TRIAL_FIELDS)]
    t["bwm_include"] = np.arange(N_TRIALS) % 7 != 3
    return t


def _session(eid) -> Session:
    seed = EIDS.index(eid)
    rng = np.random.default_rng(seed)
    ids = [f"p_{i}" for i in range(4)]
    units = pd.DataFrame({f: [1.0] * 4 for f in UNIT_FIELDS}, index=pd.Index(ids, name="unit_id"))
    units["acronym"] = "CA1"
    spikes = {u: np.sort(rng.uniform(0.0, DURATION, 3000)) for u in ids}
    t = 0.0031 + np.arange(int((DURATION - 0.01) * 100)) / 100
    wheel = TimeSeries(t, np.sin(t / 3.0 + seed))
    present = (
        {f"trials.{f}" for f in TRIAL_FIELDS}
        | {f"units.{f}" for f in UNIT_FIELDS}
        | {"behaviour.wheel"}
    )
    return Session(
        eid=eid,
        time_bounds=(0.0, DURATION),
        spikes=spikes,
        units=units,
        trials=_trials(seed),
        behaviour={"wheel": wheel},
        available=Capabilities(
            present=frozenset(present),
            missing={f"behaviour.{f}": "fixture" for f in BEHAVIOUR_FIELDS if f != "wheel"},
        ),
    )


def _manifest() -> Manifest:
    rows = [{"eid": e, "subject": f"s{i}", "lab": "lab"} for i, e in enumerate(EIDS)]
    return Manifest(
        sessions=pd.DataFrame(rows), insertions=pd.DataFrame(), provenance=manifest_versions()
    )


SESSIONS = {e: _session(e) for e in EIDS}
SPLIT = within_session(
    _manifest(), {e: SESSIONS[e].trials for e in EIDS}, PREPROC, train_fraction=0.8, gap_s=2.0
)


def _provider(target, **kwargs):
    if target in ("wheel_velocity", "movement_state"):
        kwargs.setdefault("context_bins", 50)
    return SplitData(SPLIT, target, load=SESSIONS.__getitem__, **kwargs)


def test_per_bin_samples_stay_in_their_block_and_the_task_period():
    provider = _provider("wheel_velocity")
    for partition in ("train", "test"):
        d = provider.data("e0", partition)
        first, last = bin_range(SPLIT.sessions["e0"]["blocks"][partition], 20)
        assert d.ends.min() - 49 >= first and d.ends.max() <= last
        assert np.all(np.isfinite(d.y)) and d.z.dtype == np.float32
        target = provider._prepared["e0"].target
        np.testing.assert_array_equal(d.y, target.values[d.ends - d.first_bin])
        assert d.task_features.shape == (len(d.ends), 95)
    train, test = provider.data("e0", "train"), provider.data("e0", "test")
    assert test.ends.min() - 49 - train.ends.max() - 1 >= 100  # the split's gap survives


def test_z_is_normalised_on_the_train_block_only():
    provider = _provider("wheel_velocity")
    d = provider.data("e1", "train")
    first, last = bin_range(SPLIT.sessions["e1"]["blocks"]["train"], 20)
    block = d.z[:, first - d.first_bin : last - d.first_bin + 1].astype(np.float64)
    np.testing.assert_allclose(block.mean(axis=1), 0.0, atol=1e-4)


def test_shifted_bin_targets_come_from_the_null_rotation():
    provider = _provider("wheel_velocity")
    shifts = provider.shifts("e0", 5, seed=0)
    assert len(shifts) == 5 and shifts.min() >= 1500
    d = provider.data("e0", "test", shift=int(shifts[0]))
    p = provider._prepared["e0"]
    rotated = shift_bin_target(p.target, p.trials, int(shifts[0])).values
    np.testing.assert_array_equal(d.y, rotated[d.ends - d.first_bin])


def test_choice_rows_are_the_partitions_included_trials():
    provider = _provider("choice")
    assert provider.context_bins == 5
    d = provider.data("e2", "test")
    listed = set(SPLIT.sessions["e2"]["trials"]["test"])
    table = provider._prepared["e2"].target.table
    expected = table[table["trial"].isin(listed)]
    np.testing.assert_array_equal(d.ends, expected["end_bin"])
    np.testing.assert_array_equal(d.y, expected["label"])
    assert provider.dropped[("e2", "test")] == 0


def test_block_windows_reaching_before_the_block_are_dropped_and_counted():
    provider = _provider("block")
    assert provider.context_bins == 15
    d = provider.data("e0", "train")
    assert provider.dropped[("e0", "train")] == 1  # trial 0: window starts before its block
    first, _ = bin_range(SPLIT.sessions["e0"]["blocks"]["train"], 20)
    assert (d.ends - 14 >= first).all()


def test_trial_shifts_rotate_over_the_sessions_usable_trials():
    assert len(_provider("choice").shifts("e0", 5, seed=0)) == 0  # too few trials for 100
    nulls = dataclasses.replace(load_null_config(), min_shift_trials=5)
    provider = _provider("choice", nulls=nulls)
    shift = int(provider.shifts("e0", 5, seed=0)[0])
    d = provider.data("e0", "train", shift=shift)
    rotated = shift_trial_target(provider._prepared["e0"].target, shift).table
    listed = SPLIT.sessions["e0"]["trials"]["train"]
    np.testing.assert_array_equal(d.y, rotated[rotated["trial"].isin(listed)]["label"])


def test_train_stride_thins_training_only():
    dense, sparse = _provider("wheel_velocity"), _provider("wheel_velocity", train_stride=5)
    assert (sparse.data("e0", "train").ends % 5 == 0).all()
    assert len(sparse.data("e0", "train").ends) < len(dense.data("e0", "train").ends) / 4
    np.testing.assert_array_equal(sparse.data("e0", "test").ends, dense.data("e0", "test").ends)


def test_context_rules():
    with pytest.raises(ValueError, match="context_bins"):
        SplitData(SPLIT, "wheel_velocity", load=SESSIONS.__getitem__)
    with pytest.raises(ValueError, match="window"):
        SplitData(SPLIT, "choice", context_bins=10, load=SESSIONS.__getitem__)


def test_a_unit_subset_decodes_only_those_units_and_counts_qc_failures():
    # S3: Studio decodes from the shown units. A shown unit that fails unit QC is not
    # binned for decoding; it is left out and counted, never silently used.
    sessions = dict(SESSIONS)
    failing = SESSIONS["e0"].units.copy()
    failing.loc["p_3", "acronym"] = "root"  # QC excludes root
    sessions["e0"] = dataclasses.replace(SESSIONS["e0"], units=failing)
    provider = SplitData(
        SPLIT, "choice", load=sessions.__getitem__, unit_ids={"e0": ["p_1", "p_3"]}
    )
    d = provider.data("e0", "test")
    assert d.z.shape[0] == 1 and list(d.units.index) == ["p_1"]
    assert provider.units_excluded == {"e0": ["p_3"]}
    # Sessions not named keep every QC-passing unit.
    assert provider.data("e1", "test").z.shape[0] == 4
    with pytest.raises(ValueError, match="no unit"):
        SplitData(SPLIT, "choice", load=sessions.__getitem__, unit_ids={"e0": ["p_3"]})
    with pytest.raises(ValueError, match="not in"):
        SplitData(SPLIT, "choice", load=sessions.__getitem__, unit_ids={"e0": ["zz"]})


def test_samples_are_grouped_into_their_trials():
    # Trial targets: each sample is its own trial (from the target table, even when a
    # window ends before its trial starts). Per-bin targets: the trial whose start is
    # the last at or before the bin.
    choice_provider = _provider("choice")
    d = choice_provider.data("e0", "test")
    table = choice_provider._prepared["e0"].target.table.set_index("end_bin")
    np.testing.assert_array_equal(
        choice_provider.sample_trials("e0", d.ends), table.loc[d.ends, "trial"].to_numpy()
    )
    wheel = _provider("wheel_velocity")
    d = wheel.data("e0", "test")
    starts = np.floor(SESSIONS["e0"].trials["intervals_0"].to_numpy() * 50).astype(int)
    expected = np.searchsorted(starts, d.ends, side="right") - 1
    np.testing.assert_array_equal(wheel.sample_trials("e0", d.ends), expected)


def test_the_guard_runs():
    stale = dataclasses.replace(SPLIT, preproc={"fingerprint": "0" * 64, "bin_ms": 20})
    with pytest.raises(ValueError, match="preprocessing"):
        SplitData(stale, "choice", load=SESSIONS.__getitem__)


@pytest.mark.parametrize("target", ["wheel_velocity", "choice"])
def test_the_contract_runs_end_to_end_on_the_provider(target):
    cfg = load_baseline_config()
    provider = _provider(target)
    if target == "wheel_velocity":
        chunks = cfg.per_bin_chunks
        rows = {
            "model": lambda: RidgeDecoder(chunks, cfg.cv),
            "model_with_task": lambda: SpikesAndTaskRidge(chunks, cfg.cv, ratios=(0.1, 10.0)),
            "baseline_ridge": lambda: RidgeDecoder(chunks, cfg.cv),
            "baseline_rrr": lambda: RRRRegression(chunks, cfg.cv),
            "trialstruct": lambda: TrialStructureRidge(cfg.cv),
        }
    else:
        rows = {
            "model": lambda: LogisticDecoder(cfg.trial_chunks, cfg.cv),
            "model_with_task": lambda: SpikesAndTaskLogistic(
                cfg.trial_chunks, cfg.cv, ratios=(0.1, 10.0)
            ),
            "baseline_ridge": lambda: LogisticDecoder(cfg.trial_chunks, cfg.cv),
            "baseline_rrr": lambda: RRRClassification(provider.context_bins, cfg.cv),
            "trialstruct": lambda: TrialStructureLogistic(cfg.cv),
        }
    result = evaluate(provider, ceiling=None, seed=0, n_shifts=5, **rows)
    assert tuple(result.summary().index)[-3:] == ("model", "model_with_task", "ceiling_within")
    assert result.ceiling_is_model and len(result.per_session["model"]) == len(EIDS)


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EPHYS = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
BEHAVIOUR = DATA_ROOT / "bwm_compressed/bwm_behavior/2.0.0"
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"


@pytest.mark.skipif(
    not (EPHYS.exists() and BEHAVIOUR.exists()), reason="BWM releases not available"
)
def test_real_session_samples():
    from unitwave.data.backends.bwm_compressed import load_session_bwm
    from unitwave.data.manifest import build_manifest

    session = load_session_bwm(EID, EPHYS, BEHAVIOUR)
    split = within_session(
        build_manifest(EPHYS, BEHAVIOUR),
        {EID: session.trials},
        PREPROC,
        train_fraction=0.8,
        gap_s=2.0,
    )
    for target in ("wheel_velocity", "movement_state", "choice", "block"):
        provider = SplitData(
            split,
            target,
            context_bins=50 if target in ("wheel_velocity", "movement_state") else None,
            load=lambda eid: session,
            behaviour_root=BEHAVIOUR,
        )
        for partition in ("train", "test"):
            d = provider.data(EID, partition)
            assert len(d.ends) > 0 and np.all(np.isfinite(d.y))
            assert d.z.shape[0] == 390 and len(d.units) == 390


LOBO = __import__("unitwave.splits.registry", fromlist=["x"]).leave_one_block_out(
    _manifest(), {e: SESSIONS[e].trials for e in EIDS}, PREPROC, gap_s=2.0
)


def test_leave_one_block_out_folds_serve_the_held_out_pair():
    provider = SplitData(LOBO, "block", load=SESSIONS.__getitem__)
    folds = provider.folds()  # 4 blocks of 10 trials: two pairs
    assert len(folds) == 2 and provider.normalizer is None and len(provider.normalizers) == 2
    table = provider._prepared["e0"].target.table
    for k, fold in enumerate(folds):
        record = LOBO.sessions["e0"]["folds"][k]
        test = fold.data("e0", "test")
        expected = table[table["trial"].isin(record["trials"]["test"])]
        first, last = bin_range(record["blocks"]["test"], 20)
        inside = (expected["end_bin"] - 14 >= first) & (expected["end_bin"] <= last)
        np.testing.assert_array_equal(test.ends, expected["end_bin"][inside])
        train = fold.data("e0", "train")
        assert not set(train.ends) & set(test.ends) and len(train.ends) > 0
        # The fold's normaliser uses its training intervals only.
        assert provider.normalizers[k].split_hash.endswith(f"#fold{k}")


def test_leave_one_block_out_is_refused_for_per_bin_targets_and_pseudo_for_choice():
    with pytest.raises(ValueError, match="trial targets only"):
        SplitData(LOBO, "wheel_velocity", context_bins=50, load=SESSIONS.__getitem__)
    choice = SplitData(LOBO, "choice", load=SESSIONS.__getitem__)
    assert not choice.pseudo_sessions
    with pytest.raises(ValueError, match="no pseudo-sessions"):
        choice.folds()[0].data("e0", "train", pseudo=1)
    with pytest.raises(ValueError, match="through folds"):
        choice.data("e0", "train")


def test_the_contract_scores_leave_one_block_out_folds_separately():
    cfg = load_baseline_config()
    provider = SplitData(LOBO, "block", load=SESSIONS.__getitem__)
    provider.pseudo_sessions = False  # 40-trial fixtures are all inside IBL's 90 unbiased trials
    rows = {
        "model": lambda: LogisticDecoder(cfg.trial_chunks, cfg.cv),
        "model_with_task": lambda: SpikesAndTaskLogistic(
            cfg.trial_chunks, cfg.cv, ratios=(0.1, 10.0)
        ),
        "baseline_ridge": lambda: LogisticDecoder(cfg.trial_chunks, cfg.cv),
        "baseline_rrr": lambda: RRRClassification(provider.context_bins, cfg.cv),
        "trialstruct": lambda: TrialStructureLogistic(cfg.cv),
    }
    result = evaluate(provider, ceiling=None, seed=0, n_shifts=5, **rows)
    assert result.n_folds == 2 and result.ceiling_is_model
    # Every usable block trial whose window fits its fold is tested once, each fold scored
    # on its own; both folds hold both labels, so both are scored.
    assert (result.per_session["model"]["n_folds"] == 2).all()
    n = result.per_session["model"]["n_samples"]
    usable = {e: len(provider._prepared[e].target.table) for e in EIDS}
    dropped = {
        e: sum(v for (s, k), v in provider.dropped.items() if s == e and k.startswith("test"))
        for e in EIDS
    }
    assert all(n[e] == usable[e] - dropped[e] for e in EIDS)


@pytest.mark.skipif(
    not (EPHYS.exists() and BEHAVIOUR.exists()), reason="BWM releases not available"
)
def test_real_pseudo_sessions_replace_block_labels_on_the_same_trials():
    from unitwave.data.backends.bwm_compressed import load_session_bwm
    from unitwave.data.manifest import build_manifest
    from unitwave.evaluation.nulls import generate_pseudo_blocks
    from unitwave.splits.registry import leave_one_block_out

    session = load_session_bwm(EID, EPHYS, BEHAVIOUR)
    split = leave_one_block_out(
        build_manifest(EPHYS, BEHAVIOUR), {EID: session.trials}, PREPROC, gap_s=2.0
    )
    provider = SplitData(split, "block", load=lambda eid: session)
    assert provider.pseudo_sessions and len(provider.folds()) == len(split.sessions[EID]["folds"])
    fold = provider.folds()[1]
    real, fake = fold.data(EID, "test"), fold.data(EID, "test", pseudo=7)
    np.testing.assert_array_equal(real.ends, fake.ends)
    prior = generate_pseudo_blocks(len(session.trials), seed=7)
    trials = provider._prepared[EID].target.table.set_index("end_bin").loc[fake.ends, "trial"]
    np.testing.assert_array_equal(fake.y, (prior[trials.to_numpy()] == 0.8).astype(float))
