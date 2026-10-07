import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from unitwave.data.manifest import Manifest
from unitwave.env import env
from unitwave.preprocess.binning import PreprocConfig
from unitwave.qc.units import UnitQC
from unitwave.splits.guards import assert_split_valid
from unitwave.splits.registry import held_out_region, load_split, save_split

PREPROC = PreprocConfig(bin_ms=20, qc=UnitQC(1.0, ("void", "root"), 0.1))
PROVENANCE = {"manifest_version": 1, "sources": {"bwm_ephys": "1.2.1", "bwm_behavior": "2.0.0"}}
EXPECT = {"manifest_provenance": PROVENANCE, "preproc_fingerprint": PREPROC.fingerprint()}

# eid -> [(acronym, beryl_acronym or None, n_units, label)]
SESSIONS = {
    "eTest": [("CP", "CP", 3, 1.0), ("MOs5", "MOs", 7, 1.0)],  # 30% in CP
    "eLow": [("CP", "CP", 1, 1.0), ("MOs5", "MOs", 9, 1.0)],  # 10%: neither
    "eClean": [("MOs5", "MOs", 10, 1.0)],
    "eParent": [("MOs5", "MOs", 9, 1.0), ("STR", None, 1, 1.0)],  # STR could be CP
    "eTract": [("MOs5", "MOs", 9, 1.0), ("arb", None, 1, 1.0)],  # a fibre tract is not
    "eFailedQC": [("CP", "CP", 5, 0.5), ("MOs5", "MOs", 5, 1.0)],  # CP units fail QC
    "eDenominator": [("CP", "CP", 2, 1.0), ("arb", None, 8, 1.0)],  # 2 of 10 is 20%
}


def _manifest(eids=SESSIONS) -> Manifest:
    rows = [{"eid": e, "subject": f"s{i}", "lab": f"lab{i % 2}"} for i, e in enumerate(eids)]
    return Manifest(
        sessions=pd.DataFrame(rows), insertions=pd.DataFrame(), provenance=dict(PROVENANCE)
    )


def _units(sessions=SESSIONS) -> pd.DataFrame:
    rows = [
        {
            "eid": eid,
            "acronym": acr,
            "beryl_acronym": beryl,
            "label": label,
            "task_firing_rate": 5.0,
        }
        for eid, groups in sessions.items()
        for acr, beryl, n, label in groups
        for _ in range(n)
    ]
    return pd.DataFrame(rows)


def _cp():
    return held_out_region(_manifest(), _units(), PREPROC, region="CP")


def test_sessions_are_assigned_by_their_qc_passing_units():
    split = _cp()
    assert split.kind == "held_out_region"
    assert split.partitions == {
        "train": ["eClean", "eFailedQC", "eTract"],
        "test": ["eDenominator", "eTest"],
    }
    # Dropped sessions keep a record, so the split file shows why they are out.
    assert split.sessions["eParent"]["n_possibly_in_region"] == 1
    assert split.sessions["eLow"]["n_in_region"] == 1
    assert split.sessions["eFailedQC"]["n_units"] == 5
    assert_split_valid(split, context_bins=50, **EXPECT)


def test_guard_catches_a_train_session_with_units_in_the_region():
    split = _cp()
    for moved in ("eLow", "eParent"):
        partitions = {**split.partitions, "train": sorted([*split.partitions["train"], moved])}
        with pytest.raises(ValueError, match="CP"):
            assert_split_valid(
                dataclasses.replace(split, partitions=partitions), context_bins=50, **EXPECT
            )


def test_guard_catches_a_test_session_below_the_threshold():
    split = _cp()
    partitions = {**split.partitions, "test": sorted([*split.partitions["test"], "eLow"])}
    with pytest.raises(ValueError, match="20%"):
        assert_split_valid(
            dataclasses.replace(split, partitions=partitions), context_bins=50, **EXPECT
        )


def test_exactly_twenty_percent_is_test():
    # 0.2 * 35 is 7.000000000000001 in floating point; 7 of 35 must still count.
    sessions = {
        "eEdge": [("CP", "CP", 7, 1.0), ("MOs5", "MOs", 28, 1.0)],
        "eClean": [("MOs5", "MOs", 10, 1.0)],
    }
    split = held_out_region(_manifest(sessions), _units(sessions), PREPROC, region="CP")
    assert split.partitions["test"] == ["eEdge"]


def test_region_without_train_sessions_raises():
    sessions = {"eTest": SESSIONS["eTest"], "eParent": SESSIONS["eParent"]}
    with pytest.raises(ValueError, match="free of"):
        held_out_region(_manifest(sessions), _units(sessions), PREPROC, region="CP")


def test_region_must_be_a_beryl_region():
    # STR is an Allen region but coarser than Beryl's parcellation.
    with pytest.raises(ValueError, match="Beryl"):
        held_out_region(_manifest(), _units(), PREPROC, region="STR")


def test_region_without_test_sessions_raises():
    with pytest.raises(ValueError, match="no session"):
        held_out_region(_manifest(), _units(), PREPROC, region="PO")


def test_units_must_cover_every_manifest_session():
    units = _units()
    with pytest.raises(ValueError, match="eClean"):
        held_out_region(_manifest(), units[units["eid"] != "eClean"], PREPROC, region="CP")


def test_region_split_round_trips(tmp_path):
    split = _cp()
    assert load_split(save_split(split, tmp_path / "cp.json")) == split


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EPHYS = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
BEHAVIOUR = DATA_ROOT / "bwm_compressed/bwm_behavior/2.0.0"
DERIVED = DATA_ROOT / "derived"


@pytest.mark.skipif(
    not (EPHYS.exists() and BEHAVIOUR.exists() and (DERIVED / "bwm_ephys-1.2.1").exists()),
    reason="BWM releases and task-rate table not available",
)
def test_real_cp_split():
    from unitwave.data.manifest import build_manifest
    from unitwave.qc.task_rates import release_units

    manifest = build_manifest(EPHYS, BEHAVIOUR)
    units = release_units(EPHYS, DERIVED)
    split = held_out_region(manifest, units, PREPROC, region="CP")
    assert_split_valid(split, context_bins=50, manifest_provenance=manifest.provenance)
    assert len(split.partitions["test"]) == 43
    assert len(split.partitions["train"]) == 358
    # Without the parent-label rule, 30 more sessions would train with possible CP units.
    parent_only = [
        e
        for e, s in split.sessions.items()
        if s["n_in_region"] == 0 and s["n_possibly_in_region"] > 0
    ]
    assert len(parent_only) == 30
