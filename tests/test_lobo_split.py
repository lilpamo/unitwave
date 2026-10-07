import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest

from unitwave.data.manifest import Manifest, manifest_versions
from unitwave.preprocess.binning import load_preproc_config
from unitwave.splits.guards import assert_split_valid
from unitwave.splits.registry import leave_one_block_out, load_split, save_split

PREPROC = load_preproc_config()
N_UNBIASED, BLOCK, N_BLOCKS = 20, 10, 5


def _manifest(eids=("e0",)) -> Manifest:
    rows = [{"eid": e, "subject": "s", "lab": "lab"} for e in eids]
    return Manifest(
        sessions=pd.DataFrame(rows), insertions=pd.DataFrame(), provenance=manifest_versions()
    )


def _trials(n_blocks=N_BLOCKS) -> pd.DataFrame:
    # Trial i spans [10 + 4i, 13 + 4i] s: 1 s between trials, less than the 2 s gap.
    prior = [0.5] * N_UNBIASED + [p for b in range(n_blocks) for p in [[0.8, 0.2][b % 2]] * BLOCK]
    start = 10.0 + 4.0 * np.arange(len(prior))
    return pd.DataFrame(
        {"intervals_0": start, "intervals_1": start + 3.0, "probabilityLeft": prior}
    )


def _split(n_blocks=N_BLOCKS):
    return leave_one_block_out(_manifest(), {"e0": _trials(n_blocks)}, PREPROC, gap_s=2.0)


def _test_blocks(fold):
    """Which blocks (0 = first biased block) a fold's test trials cover."""
    return sorted({(i - N_UNBIASED) // BLOCK for i in fold["trials"]["test"]})


def test_one_fold_per_adjacent_pair_and_the_odd_block_joins_the_last_pair():
    split = _split()  # 5 blocks: (0, 1), then (2, 3, 4)
    folds = split.sessions["e0"]["folds"]
    assert split.kind == "leave_one_block_out" and split.params["blocks_per_fold"] == 2
    assert [_test_blocks(f) for f in folds] == [[0, 1], [2, 3, 4]]
    assert folds[0]["trials"]["test"] == list(range(N_UNBIASED, N_UNBIASED + 2 * BLOCK))
    assert_split_valid(split, context_bins=15)
    even = _split(n_blocks=6)
    assert [_test_blocks(f) for f in even.sessions["e0"]["folds"]] == [[0, 1], [2, 3], [4, 5]]
    assert_split_valid(even, context_bins=15)


def test_every_test_set_holds_both_labels():
    for n_blocks in (4, 5, 6, 7):
        record = _split(n_blocks).sessions["e0"]
        for fold in record["folds"]:
            for partition in ("train", "test"):
                labels = {record["trial_prior"][i] for i in fold["trials"][partition]}
                assert {0.2, 0.8} <= labels


def test_trials_within_the_gap_train_in_no_fold():
    fold = _split().sessions["e0"]["folds"][0]  # test trials 20-39
    train = fold["trials"]["train"]
    # Trials 19 and 40 end / start 1 s from the test pair: inside the 2 s gap.
    assert 19 not in train and 40 not in train
    assert 18 in train and 41 in train
    assert len(fold["blocks"]["train"]) == 2  # before and after


def test_every_biased_trial_is_tested_exactly_once():
    folds = _split().sessions["e0"]["folds"]
    tested = [i for fold in folds for i in fold["trials"]["test"]]
    assert sorted(tested) == list(range(N_UNBIASED, N_UNBIASED + N_BLOCKS * BLOCK))


def test_guard_refuses_a_context_longer_than_the_gap():
    with pytest.raises(ValueError, match="gap"):
        assert_split_valid(_split(), context_bins=300)


def test_guard_catches_a_gap_trial_moved_into_training():
    split = _split()
    sessions = copy.deepcopy(split.sessions)
    sessions["e0"]["folds"][0]["trials"]["train"].append(40)
    with pytest.raises(ValueError, match="trial 40"):
        assert_split_valid(dataclasses.replace(split, sessions=sessions), context_bins=15)


def test_guard_catches_a_trial_tested_twice():
    split = _split()
    sessions = copy.deepcopy(split.sessions)
    sessions["e0"]["folds"][1]["trials"]["test"].insert(0, 39)
    sessions["e0"]["folds"][1]["blocks"]["test"][0] = sessions["e0"]["trial_intervals"][39][0]
    with pytest.raises(ValueError, match="more than one fold|shares bins|gap"):
        assert_split_valid(dataclasses.replace(split, sessions=sessions), context_bins=15)


def test_guard_rejects_a_test_set_that_holds_one_label():
    # The first version held out single blocks; a fold like that must never pass again.
    split = _split()
    sessions = copy.deepcopy(split.sessions)
    fold = sessions["e0"]["folds"][0]
    fold["trials"]["test"] = fold["trials"]["test"][:BLOCK]  # block 0 only: all 0.8
    with pytest.raises(ValueError, match="only block label"):
        assert_split_valid(dataclasses.replace(split, sessions=sessions), context_bins=15)


def test_guard_rejects_a_split_without_trial_labels():
    split = _split()
    sessions = copy.deepcopy(split.sessions)
    del sessions["e0"]["trial_prior"]
    with pytest.raises(ValueError, match="predates the block-pair fix"):
        assert_split_valid(dataclasses.replace(split, sessions=sessions), context_bins=15)


def test_too_few_blocks_is_refused_not_dropped():
    with pytest.raises(ValueError, match="3 biased blocks"):
        _split(n_blocks=3)


def test_lobo_split_round_trips(tmp_path):
    split = _split()
    assert load_split(save_split(split, tmp_path / "lobo.json")) == split
