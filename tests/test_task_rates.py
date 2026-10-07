import json
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
from unitwave.qc import task_rates
from unitwave.qc.task_rates import load_task_rates, release_units, table_dir
from unitwave.qc.units import in_task, task_firing_rates, task_period


def _trials(starts, stops) -> pd.DataFrame:
    t = pd.DataFrame({f: [1.0] * len(starts) for f in TRIAL_FIELDS})
    t["intervals_0"], t["intervals_1"] = starts, stops
    return t


def test_task_period_is_first_start_to_last_end():
    assert task_period(_trials([5.0, 2.0, 9.0], [6.0, 3.0, 9.5])) == (2.0, 9.5)


def test_task_period_needs_every_interval():
    with pytest.raises(ValueError, match="intervals"):
        task_period(_trials([2.0, np.nan], [3.0, 4.0]))
    with pytest.raises(ValueError, match="intervals"):
        task_period(_trials([], []))


def test_in_task_includes_both_ends():
    mask = in_task(np.array([0.999, 1.0, 2.0, 3.0, 3.001]), 1.0, 3.0)
    assert mask.tolist() == [False, True, True, True, False]


def test_session_rates_count_only_task_spikes():
    units = pd.DataFrame({f: [1.0, 1.0] for f in UNIT_FIELDS}, index=pd.Index(["a", "b"]))
    session = Session(
        eid="e",
        time_bounds=(0.0, 10.0),
        spikes={"a": np.array([0.5, 2.0, 3.0, 9.0]), "b": np.array([9.5])},
        units=units,
        trials=_trials([1.0, 4.0], [3.5, 5.0]),
        behaviour={},
        available=Capabilities(
            present=frozenset(
                {f"trials.{f}" for f in TRIAL_FIELDS} | {f"units.{f}" for f in UNIT_FIELDS}
            ),
            missing={f"behaviour.{f}": "fixture" for f in BEHAVIOUR_FIELDS},
        ),
    )
    rates = task_firing_rates(session)
    assert rates.name == "task_firing_rate"
    assert rates["a"] == pytest.approx(2 / 4.0) and rates["b"] == 0.0


def _write_table(root: Path, rows: pd.DataFrame, **provenance) -> None:
    out = table_dir(root)
    out.mkdir(parents=True)
    rows.to_parquet(out / "task_rates-v1.parquet")
    content = {**task_rates._provenance(len(rows)), **provenance}
    (out / "task_rates-v1.provenance.json").write_text(json.dumps(content))


def _rows(n=2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pid": ["p"] * n,
            "eid": ["e"] * n,
            "cluster_id": list(range(n)),
            "n_task_spikes": [10] * n,
            "task_start": 0.0,
            "task_end": 10.0,
            "task_firing_rate": [1.0] * n,
        }
    )


def test_missing_table_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_task_rates"):
        load_task_rates(tmp_path)


def test_table_from_other_code_or_release_is_refused(tmp_path):
    _write_table(tmp_path, _rows(), task_rates_version=0)
    with pytest.raises(ValueError, match="provenance"):
        load_task_rates(tmp_path)


def test_release_units_refuses_a_table_missing_units(tmp_path):
    ephys = tmp_path / "ephys"
    (ephys / "metadata").mkdir(parents=True)
    (ephys / "manifest.json").write_text(
        json.dumps({"dataset_name": "bwm_ephys", "dataset_version": "1.2.1"})
    )
    units = pd.DataFrame({"pid": ["p"] * 3, "eid": ["e"] * 3, "cluster_id": [0, 1, 2]})
    units.to_parquet(ephys / "metadata/units.parquet")
    _write_table(tmp_path / "derived", _rows(2))
    with pytest.raises(ValueError, match="different units"):
        release_units(ephys, tmp_path / "derived")


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EPHYS = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
DERIVED = DATA_ROOT / "derived"
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"


@pytest.fixture(scope="module")
def bwm_session():
    from unitwave.data.backends.bwm_compressed import load_session_bwm

    return load_session_bwm(EID, EPHYS)


def _by_unit(session, table: pd.DataFrame) -> pd.Series:
    names = dict(zip(zip(session.units["pid"], session.units["cluster_id"]), session.units.index))
    keys = zip(table["pid"], table["cluster_id"])
    return pd.Series(table["task_firing_rate"].to_numpy(), index=[names[k] for k in keys])


@pytest.mark.skipif(not EPHYS.exists(), reason="BWM release not available")
def test_real_probe_rates_equal_the_session_path(bwm_session):
    """The table's builder and apply_unit_qc's session path give identical rates."""
    session_rates = task_firing_rates(bwm_session)
    period = task_period(bwm_session.trials)
    for pid in bwm_session.units["pid"].unique():
        table = task_rates.probe_task_rates(EPHYS, pid, EID, period)
        rates = _by_unit(bwm_session, table)
        assert (rates == session_rates[rates.index]).all()


@pytest.mark.skipif(
    not (EPHYS.exists() and (DERIVED / "bwm_ephys-1.2.1").exists()),
    reason="BWM release and task-rate table needed",
)
def test_real_table_covers_the_release_and_matches_sessions(bwm_session):
    units = release_units(EPHYS, DERIVED)
    assert len(units) == 75_395
    table = load_task_rates(DERIVED)
    rates = _by_unit(bwm_session, table[table["eid"] == EID])
    session_rates = task_firing_rates(bwm_session)
    assert (rates.sort_index() == session_rates.sort_index()).all()
