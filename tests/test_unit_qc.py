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
from unitwave.qc.units import UnitQC, apply_unit_qc, load_qc_config, unit_qc

QC = UnitQC(min_label=1.0, exclude_regions=("void", "root"), min_firing_rate_hz=0.1)


def _units(**columns) -> pd.DataFrame:
    base = {
        "label": [1.0, 1.0, 0.6667, 1.0],
        "task_firing_rate": [5.0, 0.05, 5.0, 5.0],
        "firing_rate": [5.0, 5.0, 5.0, 5.0],
        "acronym": ["CA1", "CA1", "CA1", "root"],
    }
    base.update(columns)
    return pd.DataFrame(base, index=pd.Index(["u0", "u1", "u2", "u3"], name="unit_id"))


def test_default_config_matches_the_signed_off_choices():
    assert load_qc_config() == QC


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "qc.yaml"
    path.write_text(
        "min_label: 1.0\nexclude_regions: [void]\nmin_firing_rate_hz: 0.1\nrpv_max: 0.1\n"
    )
    with pytest.raises(ValueError, match="rpv_max"):
        load_qc_config(path)


def test_hash_is_stable_and_changes_with_any_threshold():
    assert QC.hash() == UnitQC(1.0, ("void", "root"), 0.1).hash()
    assert QC.hash() != UnitQC(1.0, ("void", "root"), 1.0).hash()
    assert QC.hash() != UnitQC(1.0, ("void",), 0.1).hash()


def test_each_criterion_excludes_with_a_reason():
    result = unit_qc(_units(), QC)
    assert result["passed"].tolist() == [True, False, False, False]
    assert result.loc["u0", "reason"] == ""
    assert "firing rate" in result.loc["u1", "reason"]
    assert "label" in result.loc["u2", "reason"]
    assert "root" in result.loc["u3", "reason"]


def test_all_failing_criteria_are_reported():
    result = unit_qc(_units(label=[0.0, 1.0, 1.0, 1.0], task_firing_rate=[0.01, 5, 5, 5]), QC)
    assert "label" in result.loc["u0", "reason"] and "firing rate" in result.loc["u0", "reason"]


def test_missing_label_or_rate_fails_rather_than_passes():
    units = _units(label=[np.nan, 1.0, 1.0, 1.0], task_firing_rate=[5.0, np.nan, 5, 5])
    result = unit_qc(units, QC)
    assert not result.loc["u0", "passed"] and "label missing" in result.loc["u0", "reason"]
    assert not result.loc["u1", "passed"] and "firing rate missing" in result.loc["u1", "reason"]


def test_the_rate_criterion_reads_the_task_period_rate_only():
    # u0 fires rarely over the whole recording but enough during the task, u1 the reverse.
    result = unit_qc(_units(firing_rate=[0.01, 5.0, 5.0, 5.0]), QC)
    assert result["passed"].tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="task_firing_rate"):
        unit_qc(_units().drop(columns="task_firing_rate"), QC)


def test_location_from_atlas_id_when_there_is_no_acronym():
    units = _units().drop(columns="acronym").assign(atlas_id=[382, 382, 382, 997])
    assert unit_qc(units, QC)["passed"].tolist() == [True, False, False, False]
    units = units.assign(atlas_id=[0, 382, 382, 382])
    assert "void" in unit_qc(units, QC).loc["u0", "reason"]


def test_location_from_nwb_names_when_there_is_no_acronym_or_atlas_id():
    units = (
        _units()
        .drop(columns="acronym")
        .assign(location=["Field CA1", "Field CA1", "Field CA1", "root"])
    )
    assert unit_qc(units, QC)["passed"].tolist() == [True, False, False, False]


def test_no_location_raises_instead_of_skipping_the_criterion():
    with pytest.raises(ValueError, match="location"):
        unit_qc(_units().drop(columns="acronym"), QC)


def test_missing_location_value_fails_the_unit():
    result = unit_qc(_units(acronym=[None, "CA1", "CA1", "CA1"]), QC)
    assert not result.loc["u0", "passed"] and "location missing" in result.loc["u0", "reason"]


def _session(units: pd.DataFrame) -> Session:
    table = units.drop(columns="task_firing_rate")
    for field in UNIT_FIELDS:
        if field not in table:
            table[field] = 1.0
    present = {f"trials.{f}" for f in TRIAL_FIELDS} | {f"units.{f}" for f in UNIT_FIELDS}
    trials = pd.DataFrame({f: [1.0] for f in TRIAL_FIELDS})
    trials["intervals_0"], trials["intervals_1"] = 0.5, 9.5  # a 9 s task period
    # One spike each, inside the task except u1's: task rates 1/9, 0, 1/9, 1/9 Hz.
    spikes = dict(zip(table.index, ([1.0], [9.8], [3.0], [4.0])))
    return Session(
        eid="e",
        time_bounds=(0.0, 10.0),
        spikes={u: np.array(t) for u, t in spikes.items()},
        units=table,
        trials=trials,
        behaviour={},
        available=Capabilities(
            present=frozenset(present),
            missing={f"behaviour.{f}": "fixture" for f in BEHAVIOUR_FIELDS},
        ),
    )


def test_apply_keeps_only_passing_units_and_their_spikes():
    out = apply_unit_qc(_session(_units()), QC)
    assert list(out.units.index) == ["u0"]
    assert list(out.spikes) == ["u0"]
    np.testing.assert_array_equal(out.spikes["u0"], [1.0])
    assert out.n_trials == 1
    # The rate QC used, computed from the session's own spikes, is kept.
    assert out.units.loc["u0", "task_firing_rate"] == pytest.approx(1 / 9)
    assert out.units.loc["u0", "firing_rate"] == 5.0


def test_apply_refuses_to_return_a_session_with_no_units():
    with pytest.raises(ValueError, match="no units pass"):
        apply_unit_qc(_session(_units(label=[0.0] * 4)), QC)


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
BWM = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
NWB = (
    DATA_ROOT
    / "dandi/000409/sub-DY-016"
    / f"sub-DY-016_ses-{EID}_desc-processed_behavior+ecephys.nwb"
)
ONE_CACHE = DATA_ROOT / "one"
DERIVED = DATA_ROOT / "derived"


@pytest.mark.skipif(
    not (BWM.exists() and (DERIVED / "bwm_ephys-1.2.1").exists()),
    reason="BWM release and task-rate table needed",
)
def test_real_bwm_floor_removes_exactly_the_sub_0_1_hz_task_rate_units():
    from unitwave.qc.task_rates import release_units

    units = release_units(BWM, DERIVED)
    result = unit_qc(units, QC)
    # 83 under the whole-recording rate (PREPROC_VERSION 1); 1,402 over the task period.
    assert int((~result["passed"]).sum()) == 1402
    assert (units.loc[~result["passed"].to_numpy(), "task_firing_rate"] < 0.1).all()


@pytest.mark.skipif(
    not (BWM.exists() and NWB.exists() and (ONE_CACHE / "danlab").exists()),
    reason="all three backends' data are needed",
)
def test_real_three_backends_keep_the_same_units():
    from unitwave.data.backends.bwm_compressed import load_session_bwm
    from unitwave.data.backends.dandi_nwb import load_session_nwb
    from unitwave.data.backends.one_backend import load_session_one, make_one

    kept = [
        set(apply_unit_qc(s, QC).units.index)
        for s in (
            load_session_bwm(EID, BWM),
            load_session_nwb(NWB),
            load_session_one(EID, make_one(ONE_CACHE)),
        )
    ]
    assert kept[0] == kept[1] == kept[2]
    # BWM's 398 good units minus 8 under 0.1 Hz during the task: probe00_27 (0.0965 Hz
    # over the recording, 0.0013 Hz in the task), probe01_280, and 6 units that fire
    # mostly after the task (see DECISIONS.md). 397 under PREPROC_VERSION 1.
    assert len(kept[0]) == 390 and "probe00_27" not in kept[0]
    assert "probe01_953" not in kept[0] and "probe00_514" in kept[0]
