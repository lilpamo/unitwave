import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unitwave.data.manifest import Manifest, manifest_versions
from unitwave.env import env
from unitwave.preprocess.binning import PreprocConfig
from unitwave.qc.units import UnitQC
from unitwave.splits.guards import assert_split_valid
from unitwave.splits.registry import (
    Split,
    held_out_groups,
    held_out_session,
    load_split,
    save_split,
    within_session,
)

PREPROC = PreprocConfig(bin_ms=20, qc=UnitQC(1.0, ("void", "root"), 0.1))
# This code's manifest provenance (version and release versions), as a real split records it.
PROVENANCE = manifest_versions()
EXPECT = {"manifest_provenance": PROVENANCE, "preproc_fingerprint": PREPROC.fingerprint()}


def _manifest() -> Manifest:
    # 4 labs x 3 animals x 2 sessions.
    rows = [
        {"eid": f"e{lab}{subj}{k}", "subject": f"L{lab}S{subj}", "lab": f"lab{lab}"}
        for lab in range(4)
        for subj in range(3)
        for k in range(2)
    ]
    return Manifest(
        sessions=pd.DataFrame(rows), insertions=pd.DataFrame(), provenance=dict(PROVENANCE)
    )


def _trials(eids, n=20) -> dict:
    # Trial i spans [10 + 4i, 13 + 4i] s.
    starts = 10.0 + 4.0 * np.arange(n)
    return {eid: pd.DataFrame({"intervals_0": starts, "intervals_1": starts + 3.0}) for eid in eids}


def _within(gap_s=2.0) -> Split:
    return within_session(_manifest(), _trials(["e000"]), PREPROC, train_fraction=0.8, gap_s=gap_s)


def _animals(n_calibration=0, seed=0) -> Split:
    return held_out_groups(
        _manifest(), PREPROC, by="subject", n_test=3, n_calibration=n_calibration, seed=seed
    )


def _subjects(split: Split, partition: str) -> set:
    return {split.sessions[e]["subject"] for e in split.partitions[partition]}


def test_seed_is_required():
    with pytest.raises(TypeError):
        held_out_groups(_manifest(), PREPROC, by="subject", n_test=3, n_calibration=0)
    with pytest.raises(TypeError):
        held_out_session(_manifest(), PREPROC)


def test_no_subject_leakage():
    split = _animals(n_calibration=2)
    assert split.kind == "held_out_animal"
    train, calib, test = (_subjects(split, p) for p in ("train", "calibration", "test"))
    assert len(test) == 3 and len(calib) == 2 and len(train) == 7
    assert not (train & test) and not (train & calib) and not (calib & test)
    # Every session of a test animal is in test.
    assert len(split.partitions["test"]) == 6
    assert_split_valid(split, context_bins=50, **EXPECT)


def test_group_split_is_deterministic_per_seed():
    assert _animals(seed=0).hash == _animals(seed=0).hash
    assert _animals(seed=0).partitions["test"] != _animals(seed=1).partitions["test"]


def test_held_out_lab_keeps_labs_disjoint():
    split = held_out_groups(_manifest(), PREPROC, by="lab", n_test=1, n_calibration=1, seed=0)
    labs = {p: {split.sessions[e]["lab"] for e in eids} for p, eids in split.partitions.items()}
    assert split.kind == "held_out_lab"
    assert len(labs["test"]) == 1 and len(labs["calibration"]) == 1
    assert not (labs["train"] & labs["test"])
    assert_split_valid(split, context_bins=50, **EXPECT)


def test_requesting_more_groups_than_exist_raises():
    with pytest.raises(ValueError, match="groups"):
        held_out_groups(_manifest(), PREPROC, by="lab", n_test=3, n_calibration=1, seed=0)


def test_held_out_session_tests_animals_seen_in_training():
    split = held_out_session(_manifest(), PREPROC, seed=0)
    assert len(split.partitions["test"]) == 12
    assert _subjects(split, "test") <= _subjects(split, "train")
    assert not set(split.partitions["test"]) & set(split.partitions["train"])
    assert_split_valid(split, context_bins=50, **EXPECT)


def test_within_session_blocks_trials_with_a_gap():
    s = _within().sessions["e000"]
    assert s["blocks"]["train"] == [10.0, 73.0]
    assert s["blocks"]["test"] == [78.0, 89.0]
    assert s["trials"]["train"] == list(range(16))
    # Trial 16 starts at 74 s, inside the 2 s gap after 73 s: it is in neither block.
    assert s["trials"]["test"] == [17, 18, 19]
    assert_split_valid(_within(), context_bins=50, **EXPECT)


def test_no_temporal_leakage():
    split = _within()
    rate = 1000 // split.preproc["bin_ms"]
    train, test = (split.sessions["e000"]["blocks"][p] for p in ("train", "test"))
    train_bins = set(range(int(train[0] * rate), int(train[1] * rate) + 1))
    test_bins = set(range(int(test[0] * rate), int(test[1] * rate) + 1))
    assert not train_bins & test_bins
    assert min(test_bins) - max(train_bins) - 1 >= 50
    # The gap is 5 s (249 empty bins). A 300-bin context would reach back into train.
    with pytest.raises(ValueError, match="gap"):
        assert_split_valid(split, context_bins=300, **EXPECT)


def test_gap_below_two_seconds_is_refused():
    with pytest.raises(ValueError, match="2.0 s"):
        _within(gap_s=1.0)


def test_guard_catches_a_trial_outside_its_block():
    split = _within()
    bad = {**split.sessions["e000"], "trials": {"train": list(range(17)), "test": [17, 18, 19]}}
    with pytest.raises(ValueError, match="trial 16"):
        assert_split_valid(
            dataclasses.replace(split, sessions={"e000": bad}), context_bins=50, **EXPECT
        )


def test_guard_catches_a_trial_in_two_partitions():
    split = _within()
    bad = {**split.sessions["e000"], "trials": {"train": [0, 1], "test": [1, 17]}}
    with pytest.raises(ValueError, match="both"):
        assert_split_valid(
            dataclasses.replace(split, sessions={"e000": bad}), context_bins=50, **EXPECT
        )


def test_guard_catches_overlapping_blocks():
    split = _within()
    blocks = {"train": [10.0, 80.0], "test": [78.0, 89.0]}
    bad = {**split.sessions["e000"], "blocks": blocks}
    with pytest.raises(ValueError, match="share bins"):
        assert_split_valid(
            dataclasses.replace(split, sessions={"e000": bad}), context_bins=50, **EXPECT
        )


def test_guard_catches_a_subject_in_two_partitions():
    split = _animals()
    # Move one session of a test animal into train; its other session stays in test.
    moved = split.partitions["test"][0]
    partitions = {
        "train": sorted([*split.partitions["train"], moved]),
        "test": split.partitions["test"][1:],
    }
    with pytest.raises(ValueError, match="subject"):
        assert_split_valid(
            dataclasses.replace(split, partitions=partitions), context_bins=50, **EXPECT
        )


def test_guard_catches_calibration_sharing_an_animal_with_test():
    split = _animals(n_calibration=2)
    moved = split.partitions["test"][0]
    partitions = {
        **split.partitions,
        "calibration": sorted([*split.partitions["calibration"], moved]),
        "test": split.partitions["test"][1:],
    }
    with pytest.raises(ValueError, match="calibration and test"):
        assert_split_valid(
            dataclasses.replace(split, partitions=partitions), context_bins=50, **EXPECT
        )


def test_guard_catches_a_session_in_two_partitions():
    split = held_out_session(_manifest(), PREPROC, seed=0)
    train = sorted(split.partitions["train"] + split.partitions["test"][:1])
    with pytest.raises(ValueError, match="both"):
        assert_split_valid(
            dataclasses.replace(split, partitions={**split.partitions, "train": train}),
            context_bins=50,
            **EXPECT,
        )


def test_guard_catches_an_empty_partition():
    split = _animals()
    with pytest.raises(ValueError, match="empty"):
        assert_split_valid(
            dataclasses.replace(split, partitions={**split.partitions, "test": []}),
            context_bins=50,
            **EXPECT,
        )


def test_guard_catches_version_mismatches():
    split = _animals()
    with pytest.raises(ValueError, match="preprocessing"):
        assert_split_valid(
            split, context_bins=50, manifest_provenance=PROVENANCE, preproc_fingerprint="0" * 64
        )
    with pytest.raises(ValueError, match="manifest"):
        assert_split_valid(
            split,
            context_bins=50,
            manifest_provenance={
                **PROVENANCE,
                "manifest_version": PROVENANCE["manifest_version"] + 1,
            },
            preproc_fingerprint=PREPROC.fingerprint(),
        )


def test_guard_defaults_to_the_current_code_and_configs():
    # The fixture provenance and preprocessing are this code's and configs/*.yaml's.
    assert_split_valid(_animals(), context_bins=50)


def test_save_and_load_round_trip(tmp_path):
    split = _animals(n_calibration=2)
    loaded = load_split(save_split(split, tmp_path / "split.json"))
    assert loaded == split and loaded.hash == split.hash
    within = _within()
    assert load_split(save_split(within, tmp_path / "within.json")) == within


def test_edited_split_file_is_refused(tmp_path):
    path = save_split(held_out_session(_manifest(), PREPROC, seed=0), tmp_path / "split.json")
    content = json.loads(path.read_text())
    content["partitions"]["test"] = content["partitions"]["test"][1:]
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="hash"):
        load_split(path)


def test_a_saved_split_is_never_replaced(tmp_path):
    path = save_split(_animals(seed=0), tmp_path / "split.json")
    save_split(_animals(seed=0), path)  # the same split again is a no-op
    with pytest.raises(FileExistsError):
        save_split(_animals(seed=1), path)


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EPHYS = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
BEHAVIOUR = DATA_ROOT / "bwm_compressed/bwm_behavior/2.0.0"
needs_bwm = pytest.mark.skipif(
    not (EPHYS.exists() and BEHAVIOUR.exists()), reason="BWM releases not available"
)


@pytest.fixture(scope="module")
def manifest():
    from unitwave.data.manifest import build_manifest

    return build_manifest(EPHYS, BEHAVIOUR)


@needs_bwm
def test_real_held_out_animal_split_has_enough_test_sessions(manifest):
    split = held_out_groups(manifest, PREPROC, by="subject", n_test=20, n_calibration=15, seed=0)
    # Phase 7 needs at least 20 held-out sessions for a session-level correlation.
    assert len(split.partitions["test"]) >= 20
    assert sum(len(eids) for eids in split.partitions.values()) == len(manifest.sessions)
    assert_split_valid(split, context_bins=50, manifest_provenance=manifest.provenance)


@needs_bwm
def test_real_within_session_split_validates(manifest):
    eids = list(manifest.sessions["eid"][:5])
    table = pd.read_parquet(EPHYS / "metadata/trials.parquet", filters=[("eid", "in", eids)])
    # Session.trials order, as the BWM backend builds it.
    trials = {
        eid: t.sort_values("trial_id").reset_index(drop=True) for eid, t in table.groupby("eid")
    }
    split = within_session(manifest, trials, PREPROC, train_fraction=0.8, gap_s=2.0)
    assert_split_valid(split, context_bins=50, manifest_provenance=manifest.provenance)
    for eid in eids:
        s = split.sessions[eid]
        assert len(s["trials"]["train"]) + len(s["trials"]["test"]) >= 0.95 * len(trials[eid])
