import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from unitwave.data.manifest import Manifest
from unitwave.env import env
from unitwave.preprocess.binning import PreprocConfig
from unitwave.qc.units import UnitQC
from unitwave.splits.guards import assert_split_valid
from unitwave.splits.registry import held_out_config, load_split, save_split

PREPROC = PreprocConfig(bin_ms=20, qc=UnitQC(1.0, ("void", "root"), 0.1))
PROVENANCE = {"manifest_version": 1, "sources": {"bwm_ephys": "1.2.1", "bwm_behavior": "2.0.0"}}
EXPECT = {"manifest_provenance": PROVENANCE, "preproc_fingerprint": PREPROC.fingerprint()}
# Lab A's session a{k} has k QC-passing units and lab B's b{k} has 100 + k, k = 1..20.
# Within each lab the 10th percentile is the 2.9th-smallest count and the 25th the
# 5.75th, so a1-a2 and b1-b2 are test, a6-a20 and b6-b20 train, the rest dropped.
COUNTS = {
    **{f"a{k}": ("A", k) for k in range(1, 21)},
    **{f"b{k}": ("B", 100 + k) for k in range(1, 21)},
}


def _manifest(counts=COUNTS) -> Manifest:
    rows = [
        {"eid": e, "subject": f"s{i % 7}", "lab": lab}
        for i, (e, (lab, _)) in enumerate(counts.items())
    ]
    return Manifest(
        sessions=pd.DataFrame(rows), insertions=pd.DataFrame(), provenance=dict(PROVENANCE)
    )


def _units(counts=COUNTS) -> pd.DataFrame:
    rows = [
        {"eid": eid, "label": 1.0, "acronym": "CA1", "task_firing_rate": 5.0}
        for eid, (_, n) in counts.items()
        for _ in range(n)
    ]
    # Units failing QC don't count: 30 silent units in a1 leave it with 1.
    rows += [{"eid": "a1", "label": 1.0, "acronym": "CA1", "task_firing_rate": 0.0}] * 30
    return pd.DataFrame(rows)


def _split():
    return held_out_config(_manifest(), _units(), PREPROC)


def test_percentiles_are_within_each_lab():
    split = _split()
    assert split.kind == "held_out_config"
    assert split.partitions["test"] == ["a1", "a2", "b1", "b2"]
    train = [f"{lab}{k}" for lab in "ab" for k in range(6, 21)]
    assert split.partitions["train"] == sorted(train)
    # b1 (101 units) is test while a20 (20 units) trains: "few" is relative to the lab.
    assert_split_valid(split, context_bins=50, **EXPECT)


def test_each_labs_cutoffs_and_every_sessions_count_are_recorded():
    split = _split()
    assert split.params["percentiles_within"] == "lab"
    assert split.params["test_percentile"] == 10 and split.params["train_percentile"] == 25
    a, b = split.params["cutoffs"]["A"], split.params["cutoffs"]["B"]
    assert a["n_sessions"] == 20
    assert a["test_below_units"] == pytest.approx(2.9) and a["train_from_units"] == pytest.approx(
        5.75
    )
    assert (a["test_max_units"], a["train_min_units"]) == (2, 6)
    assert (b["test_max_units"], b["train_min_units"]) == (102, 106)
    assert {e: s["n_units"] for e, s in split.sessions.items()} == {
        e: n for e, (_, n) in COUNTS.items()
    }


def test_integer_percentile_cutoffs_are_strict_for_test():
    # 11 sessions with 1..11 units: the 10th percentile is exactly 2, so only a1 is test.
    counts = {f"a{k}": ("A", k) for k in range(1, 12)}
    split = held_out_config(_manifest(counts), _units(counts), PREPROC)
    assert split.params["cutoffs"]["A"]["test_below_units"] == 2.0
    assert split.params["cutoffs"]["A"]["test_max_units"] == 1
    assert split.partitions["test"] == ["a1"]


def test_guard_catches_a_moved_session():
    split = _split()
    for partition, eid in (("test", "a3"), ("train", "b5")):
        moved = {**split.partitions, partition: sorted([*split.partitions[partition], eid])}
        with pytest.raises(ValueError, match=eid):
            assert_split_valid(
                dataclasses.replace(split, partitions=moved), context_bins=50, **EXPECT
            )


def test_guard_catches_edited_counts_labs_or_cutoffs():
    split = _split()
    edits = [
        # A count change that moves lab A's percentiles.
        {**split.sessions, "a2": {**split.sessions["a2"], "n_units": 1}},
        # A session reassigned to the other lab.
        {**split.sessions, "b20": {**split.sessions["b20"], "lab": "A"}},
    ]
    for sessions in edits:
        with pytest.raises(ValueError, match="cutoff"):
            assert_split_valid(
                dataclasses.replace(split, sessions=sessions), context_bins=50, **EXPECT
            )
    cutoffs = {
        **split.params["cutoffs"],
        "B": {**split.params["cutoffs"]["B"], "train_from_units": 3.0},
    }
    with pytest.raises(ValueError, match="cutoff"):
        assert_split_valid(
            dataclasses.replace(split, params={**split.params, "cutoffs": cutoffs}),
            context_bins=50,
            **EXPECT,
        )
    # Raising a train session's count keeps the percentiles, so the split stays valid.
    sessions = {**split.sessions, "a20": {**split.sessions["a20"], "n_units": 500}}
    assert_split_valid(dataclasses.replace(split, sessions=sessions), context_bins=50, **EXPECT)


def test_units_must_cover_every_manifest_session():
    units = _units()
    with pytest.raises(ValueError, match="a7"):
        held_out_config(_manifest(), units[units["eid"] != "a7"], PREPROC)


def test_config_split_round_trips(tmp_path):
    split = _split()
    assert load_split(save_split(split, tmp_path / "config.json")) == split


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EPHYS = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
BEHAVIOUR = DATA_ROOT / "bwm_compressed/bwm_behavior/2.0.0"
DERIVED = DATA_ROOT / "derived"


@pytest.mark.skipif(
    not (EPHYS.exists() and BEHAVIOUR.exists() and (DERIVED / "bwm_ephys-1.2.1").exists()),
    reason="BWM releases and task-rate table not available",
)
def test_real_config_split_spreads_over_every_lab():
    from unitwave.data.manifest import build_manifest
    from unitwave.qc.task_rates import release_units

    manifest = build_manifest(EPHYS, BEHAVIOUR)
    split = held_out_config(manifest, release_units(EPHYS, DERIVED), PREPROC)
    assert_split_valid(split, context_bins=50, manifest_provenance=manifest.provenance)
    assert len(split.partitions["test"]) == 50 and len(split.partitions["train"]) == 346
    labs = pd.Series([split.sessions[e]["lab"] for e in split.partitions["test"]])
    assert labs.nunique() == 12
    # No lab's share of the test set exceeds its share of all sessions by 2 points.
    all_labs = pd.Series([s["lab"] for s in split.sessions.values()])
    excess = labs.value_counts(normalize=True) - all_labs.value_counts(normalize=True)
    assert excess.max() < 0.02
