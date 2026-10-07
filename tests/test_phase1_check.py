import numpy as np
import pandas as pd
import pytest

from unitwave.cli.phase1_check import run_check, select_sessions
from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
)


def _sessions(n_labs=4, subjects_per_lab=5, sessions_per_subject=2) -> pd.DataFrame:
    rows = []
    for lab in range(n_labs):
        for subj in range(subjects_per_lab):
            for k in range(sessions_per_subject):
                rows.append(
                    {
                        "eid": f"e{lab}{subj}{k}",
                        "subject": f"L{lab}S{subj}",
                        "lab": f"lab{lab}",
                        "date": f"2020-01-0{k + 1}",
                        "session_number": 1,
                        "n_good_units": 2,
                        "n_trials": 3,
                    }
                )
    return pd.DataFrame(rows)


def _session(eid: str, n_units: int = 2, n_trials: int = 3) -> Session:
    ids = [f"p_{i}" for i in range(n_units)]
    units = pd.DataFrame(
        {f: [1.0] * n_units for f in UNIT_FIELDS}, index=pd.Index(ids, name="unit_id")
    )
    present = {f"trials.{f}" for f in TRIAL_FIELDS} | {f"units.{f}" for f in UNIT_FIELDS}
    return Session(
        eid=eid,
        time_bounds=(0.0, 10.0),
        spikes={u: np.array([1.0]) for u in ids},
        units=units,
        trials=pd.DataFrame({f: [1.0] * n_trials for f in TRIAL_FIELDS}),
        behaviour={},
        available=Capabilities(
            present=frozenset(present),
            missing={f"behaviour.{f}": "not in fixture" for f in BEHAVIOUR_FIELDS},
        ),
    )


def test_selection_takes_one_session_per_subject_across_labs():
    chosen = select_sessions(_sessions(), n=12, min_subjects=10, min_labs=3)
    assert len(chosen) == 12
    assert chosen["subject"].is_unique
    assert chosen["lab"].nunique() == 4
    assert (chosen["date"] == "2020-01-01").all()


def test_selection_is_deterministic():
    table = _sessions()
    first = select_sessions(table, n=12, min_subjects=10, min_labs=3)
    shuffled = select_sessions(
        table.sample(frac=1, random_state=0), n=12, min_subjects=10, min_labs=3
    )
    assert list(first["eid"]) == list(shuffled["eid"])


def test_selection_refuses_too_few_subjects():
    with pytest.raises(ValueError, match="subjects"):
        select_sessions(_sessions(n_labs=2, subjects_per_lab=2), n=4, min_subjects=10, min_labs=1)


def test_selection_refuses_too_few_labs():
    with pytest.raises(ValueError, match="labs"):
        select_sessions(_sessions(n_labs=2), n=10, min_subjects=10, min_labs=3)


def test_check_passes_when_fast_and_counts_match():
    chosen = select_sessions(_sessions(), n=12, min_subjects=10, min_labs=3)
    report = run_check(chosen, load=_session, budget_s=5.0, min_subjects=10, min_labs=3)
    assert report["passed"] is True
    assert report["summary"]["n_sessions"] == 12
    assert all(r["counts_match"] for r in report["sessions"])


def test_check_fails_on_count_mismatch():
    chosen = select_sessions(_sessions(), n=12, min_subjects=10, min_labs=3)
    bad = chosen["eid"].iloc[3]

    def load(eid):
        return _session(eid, n_units=1 if eid == bad else 2)

    report = run_check(chosen, load=load, budget_s=5.0, min_subjects=10, min_labs=3)
    assert report["passed"] is False
    assert [r["eid"] for r in report["sessions"] if not r["counts_match"]] == [bad]


def test_check_fails_when_a_cached_read_is_over_budget():
    chosen = select_sessions(_sessions(), n=12, min_subjects=10, min_labs=3)
    report = run_check(chosen, load=_session, budget_s=0.0, min_subjects=10, min_labs=3)
    assert report["passed"] is False
    assert "over budget" in " ".join(report["failures"])
