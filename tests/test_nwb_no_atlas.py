"""Step 7: a dataset without the Allen mouse atlas (Neural Latents Benchmark MC_Maze_Small,
DANDI 000140: macaque M1 and PMd, Utah arrays). A small file is written here with its
layout: units linked to electrodes by per-array channel numbers, no depths or
coordinates, and two-channel hand velocity. Test inputs only, except the last test,
which reads the downloaded file."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, TimeSeries

from unitwave.nwb.intake import layout_from_dict, load_layout, read_nwb

LAYOUT = {
    "name": "utah_test",
    "label": "Utah test",
    "task": "nlb_mc_maze",
    "units": {
        "probe": "single",
        "location": {"refused": {"reason": "the unit-to-electrode links are per-array channels"}},
    },
    "behaviour": {
        "hand_speed": {"path": "processing/behavior/hand_vel", "combine": "norm"},
        "hand_x": "processing/behavior/hand_x",
    },
}


def _write(path: Path) -> Path:
    nwb = NWBFile("hand-built", "utah-test", datetime(2026, 1, 1, tzinfo=UTC))
    for name in ("PMd", "M1"):
        group = nwb.create_electrode_group(
            f"electrode_group_{name}", "array", name, nwb.create_device(name)
        )
        for _ in range(2):
            nwb.add_electrode(location=name, group=group)
    for k in range(3):
        nwb.add_unit(spike_times=np.linspace(0.5, 19.5, 30) + 0.01 * k, electrodes=[k % 2])
    nwb.add_trial_column("go_cue_time", "go cue")
    nwb.add_trial_column("move_onset_time", "movement onset")
    for i in range(4):
        nwb.add_trial(
            start_time=4.0 * i,
            stop_time=4.0 * i + 3.0,
            go_cue_time=4.0 * i + 1.0,
            move_onset_time=4.0 * i + 1.3,
        )
    module = nwb.create_processing_module("behavior", "behaviour")
    t = np.arange(2000) / 100.0
    vel = np.column_stack([np.full(2000, 3.0), np.full(2000, 4.0)])  # mm/s
    module.add(TimeSeries(name="hand_vel", data=vel, unit="m/s", timestamps=t, conversion=0.001))
    module.add(TimeSeries(name="hand_x", data=np.zeros(2000), unit="m", timestamps=t))
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return path


def test_a_two_channel_series_can_be_declared_as_its_magnitude(tmp_path):
    intake = read_nwb(_write(tmp_path / "f.nwb"), layout_from_dict(LAYOUT))
    speed = intake.session.behaviour["hand_speed"]
    np.testing.assert_allclose(speed.data, 0.005)  # |(3, 4)| mm/s = 5 mm/s = 0.005 m/s
    assert "magnitude of 2 channels" in intake.report["behaviour"]["hand_speed"]["loaded"]
    assert "hand_x" in intake.session.behaviour


def test_refused_areas_and_a_single_group(tmp_path):
    intake = read_nwb(_write(tmp_path / "f.nwb"), layout_from_dict(LAYOUT))
    s, r = intake.session, intake.report
    assert set(s.units["probe_name"]) == {"units"}
    assert "units.acronym" in s.available.missing and "location" not in s.units
    assert r["regions"] == {"refused": "the unit-to-electrode links are per-array channels"}
    assert "units.depths" in s.available.missing


def test_malformed_behaviour_and_location_entries_are_refused():
    bad = {**LAYOUT, "behaviour": {"x": {"path": "a/b", "combine": "mean"}}}
    with pytest.raises(ValueError, match="combine"):
        layout_from_dict(bad)
    bad = {**LAYOUT, "units": {"location": {"refused": {}}}}
    with pytest.raises(ValueError, match="reason"):
        layout_from_dict(bad)


MC_MAZE = (
    Path.home()
    / "data/neurodecoder/dandi/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
)


@pytest.mark.skipif(not MC_MAZE.exists(), reason="MC_Maze_Small isn't downloaded")
def test_mc_maze_small_reads_as_checked_by_hand():
    intake = read_nwb(MC_MAZE, load_layout("nlb_mc_maze"), task="nlb_mc_maze")
    s, r = intake.session, intake.report
    assert s.n_units == 142 and len(s.trials) == 100
    assert "per-array" in r["regions"]["refused"]
    assert set(s.units["probe_name"]) == {"units"}
    assert {"hand_speed"} <= set(s.behaviour)
    assert s.trials["move_onset_time"].notna().all()
    assert s.available.missing["units.depths"] == "Utah arrays have no depth along a shank"


@pytest.mark.skipif(not MC_MAZE.exists(), reason="MC_Maze_Small isn't downloaded")
def test_mc_maze_small_in_studio_works_without_an_atlas():
    from unitwave.data.load import load_data_config
    from unitwave.studio.project import Source, load_source
    from unitwave.studio.server import Studio

    source = Source(kind="nwb", file=str(MC_MAZE), layout="nlb_mc_maze", task="nlb_mc_maze")
    session, qc = load_source(source)
    studio = Studio(session, qc, load_data_config().data_root / "atlas", source)
    s = studio.session_json({})
    # Atlas-only features say why they are off.
    assert not studio.has_regions and "per-array" in s["missing"]["units.acronym"]
    assert "Allen mouse atlas doesn't apply" in studio.geometry_json({})["missing"]
    assert studio.units_json({"all": "1"})["tree"] == []
    # Everything else works, movement controls included.
    assert s["qc"]["rule"] == "spike times" and s["n_units_total"] == 142
    assert s["movement"]["first_movement"] is True
    locking = studio.locking_json({"all": "1"})
    assert locking["null"] == "reaction times permuted within maze" and locking["n_trials"] == 100
    q = {"event": "go_cue", "t0": "-0.5", "t1": "1.0", "bin": "0.02", "all": "1"}
    assert studio.test_json({**q, "movement_free": "1"})["n_trials"] == 69
    sel = studio.selectivity_json({**q, "event": "target_on", "split": "targets"})
    assert (sel["n_a"], sel["n_b"]) == (66, 34)
