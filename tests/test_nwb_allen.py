"""Step 6: the Allen Brain Observatory Visual Coding session (DANDI 000021,
sub-707296975 ses-721123822) read with its layout and three passive tasks, as checked by
hand. Used only when the file is downloaded."""

import json
from pathlib import Path

import pytest

from unitwave.analysis.tasks import load_task
from unitwave.data.load import load_data_config
from unitwave.nwb.intake import load_layout, read_nwb
from unitwave.studio.project import Source, load_source
from unitwave.studio.server import Studio, default_trials

ALLEN = Path.home() / "data/neurodecoder/dandi/000021/sub-707296975/sub-707296975_ses-721123822.nwb"
pytestmark = pytest.mark.skipif(not ALLEN.exists(), reason="the Allen session isn't downloaded")
Q = {"event": "stim_on", "t0": "-0.5", "t1": "2.5", "bin": "0.02", "theme": "light"}


def _studio(task: str) -> Studio:
    source = Source(kind="nwb", file=str(ALLEN), layout="allen_visual_coding", task=task)
    session, qc = load_source(source)
    view = {"trials": default_trials(session.trials, None, load_task(task)).to_dict()}
    return Studio(session, qc, load_data_config().data_root / "atlas", source, view=view)


def test_the_session_reads_as_checked_by_hand():
    intake = read_nwb(ALLEN, load_layout("allen_visual_coding"), task="allen_drifting_gratings")
    s, r = intake.session, intake.report
    assert s.n_units == 1603 and len(s.trials) == 628
    assert s.units["probe_name"].value_counts().sort_index().tolist() == [
        184,
        218,
        332,
        284,
        299,
        286,
    ]
    assert r["regions"]["allen"] is True and r["regions"]["no_location"] == 13
    assert s.units["acronym"].nunique() == 19
    assert int(s.trials["orientation"].isna().sum()) == 30  # blanks
    assert "repeats its y" in r["positions"]["refused"]
    assert "running_speed" in s.behaviour and len(r["trials"]["invalid_times"]) == 5


def test_drifting_gratings_in_studio():
    studio = _studio("allen_drifting_gratings")
    s = studio.session_json({})
    assert s["n_units_passing"] == 447
    assert s["comparisons"]["orientation"] == {
        "kind": "circular",
        "period": 180.0,
        "window": "0 to 2 s",
    }
    assert s["movement"]["stimulus"] is None
    sel = studio.selectivity_json({**Q, "split": "orientation"})
    assert sel["kind"] == "circular" and sel["n_trials"] == 598
    assert sel["null"] == "orientation permuted within temporal frequency"
    assert 0 < sel["n_selective"] <= sel["n_tests"] == 447


def test_static_gratings_exclude_invalid_data_by_default():
    studio = _studio("allen_static_gratings")
    summary = studio._trial_summary({"tf": json.dumps(studio.view["trials"])})
    assert summary["excluded"] == {"during invalid data": 328} and summary["n_kept"] == 5672


def test_flashes_compare_dark_and_light():
    studio = _studio("allen_flashes")
    assert studio.session_json({})["comparisons"]["color"] == ["dark", "light"]
    sel = studio.selectivity_json({**Q, "split": "color"})
    assert sel["null"] == "color permuted across trials (no strata declared)"
    assert sel["n_a"] == 75 and sel["n_b"] == 75 and sel["window"] == "0 to 0.25 s"
