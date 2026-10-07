import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unitwave.data.manifest import Manifest
from unitwave.env import env
from unitwave.preprocess.binning import BinnedSpikes, PreprocConfig
from unitwave.preprocess.normalize import Normalizer, fit_normalizer
from unitwave.qc.units import UnitQC
from unitwave.splits.registry import held_out_groups, within_session

PREPROC = PreprocConfig(bin_ms=20, qc=UnitQC(1.0, ("void", "root"), 0.1))
FP = PREPROC.fingerprint()
PROVENANCE = {"manifest_version": 1, "sources": {"bwm_ephys": "1.2.1", "bwm_behavior": "2.0.0"}}
# A Poisson unit at the QC minimum rate, 0.1 Hz, has std sqrt(0.1 * 0.02) per 20 ms bin.
FLOOR = math.sqrt(0.1 * 0.02)
N_BINS = 5000  # 100 s


def _manifest() -> Manifest:
    rows = [
        {"eid": f"e{s}{k}", "subject": f"s{s}", "lab": f"lab{s % 2}"}
        for s in range(6)
        for k in range(2)
    ]
    return Manifest(
        sessions=pd.DataFrame(rows), insertions=pd.DataFrame(), provenance=dict(PROVENANCE)
    )


def _binned(eid: str, seed: int, n_units=4) -> BinnedSpikes:
    rates = np.array([0.02, 0.1, 0.3, 1.0])[:n_units, None]
    counts = np.random.default_rng(seed).poisson(rates, (n_units, N_BINS)).astype(np.uint16)
    return BinnedSpikes(eid, tuple(f"u{i}" for i in range(n_units)), counts, 20, 0, FP)


def _all_binned() -> dict:
    return {eid: _binned(eid, i) for i, eid in enumerate(_manifest().sessions["eid"])}


def _animals():
    return held_out_groups(_manifest(), PREPROC, by="subject", n_test=2, n_calibration=1, seed=0)


def _within():
    # Trials every 4 s from 10 s; the train block is [10, 73] s = bins 500..3650.
    starts = 10.0 + 4.0 * np.arange(20)
    trials = {"e00": pd.DataFrame({"intervals_0": starts, "intervals_1": starts + 3.0})}
    return within_session(_manifest(), trials, PREPROC, train_fraction=0.8, gap_s=2.0)


def _mutate(b: BinnedSpikes, columns=slice(None)) -> BinnedSpikes:
    counts = b.counts.copy()
    counts[:, columns] = 50
    return dataclasses.replace(b, counts=counts)


def test_normalization_fit_on_train_only():
    split, binned = _animals(), _all_binned()
    before = fit_normalizer(split, binned, PREPROC)
    for eid in split.partitions["test"] + split.partitions["calibration"]:
        binned[eid] = _mutate(binned[eid])
    assert fit_normalizer(split, binned, PREPROC).hash == before.hash
    # Sanity: the statistics do depend on training data.
    train_eid = split.partitions["train"][0]
    binned[train_eid] = _mutate(binned[train_eid])
    assert fit_normalizer(split, binned, PREPROC).hash != before.hash


def test_within_session_fits_on_the_train_block_only():
    split, binned = _within(), {"e00": _binned("e00", 0)}
    before = fit_normalizer(split, binned, PREPROC)
    after_test = fit_normalizer(split, {"e00": _mutate(binned["e00"], slice(3651, None))}, PREPROC)
    assert after_test.hash == before.hash
    after_train = fit_normalizer(split, {"e00": _mutate(binned["e00"], slice(3650, 3651))}, PREPROC)
    assert after_train.hash != before.hash


def test_within_session_is_per_unit():
    split, binned = _within(), {"e00": _binned("e00", 0)}
    norm = fit_normalizer(split, binned, PREPROC)
    assert norm.mode == "per_unit"
    z = norm.transform(binned["e00"])
    assert z.shape == (4, N_BINS) and z.dtype == np.float32
    train = z[:, 500:3651].astype(np.float64)
    np.testing.assert_allclose(train.mean(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(train[1:].std(axis=1), 1.0, atol=1e-5)


def test_cross_session_splits_are_pooled_for_every_unit():
    split, binned = _animals(), _all_binned()
    norm = fit_normalizer(split, binned, PREPROC)
    assert norm.mode == "pooled"
    train = np.concatenate([norm.transform(binned[e]) for e in split.partitions["train"]], axis=1)
    assert abs(train.astype(np.float64).mean()) < 1e-5
    assert abs(train.astype(np.float64).std() - 1.0) < 1e-5
    # Test units, never seen in training, get the same affine map.
    test_eid = split.partitions["test"][0]
    mean, std = norm.pooled
    expected = (binned[test_eid].counts - mean) / std
    np.testing.assert_allclose(norm.transform(binned[test_eid]), expected, rtol=1e-6, atol=1e-5)


def test_std_is_floored_at_the_qc_minimum_rate():
    b = _binned("e00", 0)
    counts = b.counts.copy()
    counts[0, :3651] = 0  # silent throughout the train block
    norm = fit_normalizer(_within(), {"e00": dataclasses.replace(b, counts=counts)}, PREPROC)
    assert norm.min_std == pytest.approx(FLOOR)
    assert norm.units["e00"]["std"][0] == pytest.approx(FLOOR)


def test_fitting_does_not_depend_on_session_order():
    split, binned = _animals(), _all_binned()
    reordered = dict(reversed(list(binned.items())))
    assert (
        fit_normalizer(split, reordered, PREPROC).hash
        == fit_normalizer(split, binned, PREPROC).hash
    )


def test_missing_training_session_raises():
    split, binned = _animals(), _all_binned()
    del binned[split.partitions["train"][0]]
    with pytest.raises(ValueError, match="training"):
        fit_normalizer(split, binned, PREPROC)


def test_binned_from_other_preprocessing_is_refused():
    split, binned = _animals(), _all_binned()
    eid = split.partitions["train"][0]
    binned[eid] = dataclasses.replace(binned[eid], fingerprint=None)
    with pytest.raises(ValueError, match="fingerprint"):
        fit_normalizer(split, binned, PREPROC)
    other = PreprocConfig(bin_ms=10, qc=PREPROC.qc)
    with pytest.raises(ValueError, match="fingerprint"):
        fit_normalizer(split, _all_binned(), other)


def test_per_unit_transform_refuses_unknown_sessions_and_units():
    norm = fit_normalizer(_within(), {"e00": _binned("e00", 0)}, PREPROC)
    with pytest.raises(ValueError, match="e01"):
        norm.transform(_binned("e01", 1))
    with pytest.raises(ValueError, match="units"):
        norm.transform(_binned("e00", 0, n_units=3))


def test_serialisation_round_trips_and_detects_edits():
    norm = fit_normalizer(_within(), {"e00": _binned("e00", 0)}, PREPROC)
    data = norm.to_dict()
    assert Normalizer.from_dict(data) == norm
    data["units"]["e00"]["mean"][0] += 1.0
    with pytest.raises(ValueError, match="hash"):
        Normalizer.from_dict(data)


def test_inverse_transform_recovers_counts():
    split, binned = _animals(), _all_binned()
    norm = fit_normalizer(split, binned, PREPROC)
    b = binned[split.partitions["test"][0]]
    back = norm.inverse_transform(norm.transform(b), b.eid, b.unit_ids)
    np.testing.assert_allclose(back, b.counts, atol=1e-4)


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EPHYS = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
BEHAVIOUR = DATA_ROOT / "bwm_compressed/bwm_behavior/2.0.0"
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"


@pytest.mark.skipif(
    not (EPHYS.exists() and BEHAVIOUR.exists()), reason="BWM releases not available"
)
def test_real_session_round_trips_through_normalisation():
    from unitwave.data.backends.bwm_compressed import load_session_bwm
    from unitwave.data.manifest import build_manifest
    from unitwave.preprocess.binning import preprocess_session
    from unitwave.splits.guards import assert_split_valid

    session = load_session_bwm(EID, EPHYS)
    split = within_session(
        build_manifest(EPHYS, BEHAVIOUR),
        {EID: session.trials},
        PREPROC,
        train_fraction=0.8,
        gap_s=2.0,
    )
    assert_split_valid(split, context_bins=50, manifest_provenance=split.manifest)
    binned = preprocess_session(session, PREPROC)
    norm = fit_normalizer(split, {EID: binned}, PREPROC)
    z = norm.transform(binned)
    back = norm.inverse_transform(z, EID, binned.unit_ids)
    # float32 keeps ~7 significant digits; counts are at most a few tens.
    np.testing.assert_allclose(back, binned.counts, atol=1e-3)
    assert np.array_equal(np.rint(back).astype(np.uint16), binned.counts)


def test_leave_one_block_out_fits_each_fold_on_its_training_intervals_only():
    from unitwave.splits.registry import leave_one_block_out

    prior = [0.5] * 5 + [p for b in range(5) for p in [[0.8, 0.2][b % 2]] * 4]
    starts = 1.0 + 3.5 * np.arange(len(prior))
    trials = {
        "e00": pd.DataFrame(
            {"intervals_0": starts, "intervals_1": starts + 3.0, "probabilityLeft": prior}
        )
    }
    split = leave_one_block_out(_manifest(), trials, PREPROC, gap_s=2.0)
    b = _binned("e00", 0)
    for k, fold in enumerate(split.sessions["e00"]["folds"]):
        norm = fit_normalizer(split, {"e00": b}, PREPROC, fold=k)
        assert norm.mode == "per_unit" and norm.split_hash.endswith(f"#fold{k}")
        # Overwriting the fold's test block (and the gap) never changes its statistics.
        first = int(np.floor(fold["blocks"]["test"][0] * 50))
        last = int(np.floor(fold["blocks"]["test"][1] * 50))
        counts = b.counts.copy()
        counts[:, first : last + 1] = 50
        again = fit_normalizer(
            split, {"e00": dataclasses.replace(b, counts=counts)}, PREPROC, fold=k
        )
        assert again.hash == norm.hash
    with pytest.raises(ValueError, match="fold is required"):
        fit_normalizer(split, {"e00": b}, PREPROC)
