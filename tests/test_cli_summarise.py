"""The region-summary CLI (S5): a session set in, a logged run out."""

import dataclasses
import json

import pandas as pd
import pytest

from unitwave.analysis.summary import LabelSpec, load_summary_config
from unitwave.cli.summarise import run
from unitwave.data.manifest import MANIFEST_VERSION
from unitwave.studio.sets import save_set

EIDS = ["3a3ea015-b5f4-4e8b-b189-9364d1fc7435", "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"]
TRIAL_FILTER = {"bwm_include": True, "exclude_nogo": True}


@pytest.fixture(scope="module")
def two_session_set(tmp_path_factory):
    from unitwave.data.load import load_session

    try:
        for eid in EIDS:
            load_session(eid, "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"BWM sessions not available: {e}")
    folder = tmp_path_factory.mktemp("sets")
    return save_set(folder, "two sessions", EIDS, MANIFEST_VERSION, {}, TRIAL_FILTER)


def test_a_run_logs_everything_and_re_running_is_identical(two_session_set, tmp_path):
    cfg = dataclasses.replace(load_summary_config(), min_sessions=2)
    label = LabelSpec("responsive", "stim_on", "")
    first = run(two_session_set, label, cfg=cfg, runs_dir=tmp_path / "a")
    second = run(two_session_set, label, cfg=cfg, runs_dir=tmp_path / "b")
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["git"]["sha"]) == 40
    assert manifest["set"]["eids"] == EIDS and len(manifest["set"]["hash"]) == 64
    assert manifest["set"]["trial_filter"] == TRIAL_FILTER
    assert manifest["label"] == {"kind": "responsive", "event": "stim_on", "split": ""}
    assert manifest["level"] == "Beryl" and manifest["summary_config"]["min_sessions"] == 2
    assert {"summary", "qc", "analysis", "selectivity", "movement"} <= set(manifest["configs"])
    assert manifest["sessions_used"] == EIDS and manifest["sessions_failed"] == {}
    for name in ("units", "regions", "sessions"):
        a, b = (pd.read_parquet(run_dir / f"{name}.parquet") for run_dir in (first, second))
        pd.testing.assert_frame_equal(a, b)
    units = pd.read_parquet(first / "units.parquet")
    assert set(units["eid"]) == set(EIDS)
    regions = pd.read_parquet(first / "regions.parquet")
    assert regions["n_sessions"].max() <= 2 and "refused" in regions
    for name in ("flatmap", "spread"):
        assert (first / f"{name}.svg").stat().st_size > 0 and (first / f"{name}.pdf").exists()


def test_a_session_that_cannot_be_labelled_is_left_out_and_counted(two_session_set, tmp_path):
    from unitwave.data.load import load_session

    def load(eid):
        if eid == EIDS[0]:
            raise ValueError("simulated: this session has no trials file")
        return load_session(eid, "bwm")

    cfg = dataclasses.replace(load_summary_config(), min_sessions=2)
    out = run(two_session_set, LabelSpec("locked"), cfg=cfg, runs_dir=tmp_path, load=load)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sessions_used"] == [EIDS[1]]
    assert manifest["sessions_failed"] == {EIDS[0]: "simulated: this session has no trials file"}
