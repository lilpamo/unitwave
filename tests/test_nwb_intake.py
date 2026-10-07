"""Step 5: general NWB intake. A small NWB file is written here with pynwb, carrying
the quirks found in DANDI 000017 on purpose: units.electrodes counted per probe, a
sampling period stored as a rate, and a curation column. Test inputs only; never shown
as data."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import PupilTracking

from unitwave.nwb.intake import GENERIC, layout_from_dict, load_layout, read_nwb

LAYOUT = {
    "name": "test_layout",
    "label": "Test layout",
    "task": "steinmetz",
    "units": {
        "probe": "electrode_group",
        "depth": {"column": "cluster_depths"},
        "location": {"peak_channel": {"column": "peak_channel", "first": 1}},
        "quality": {
            "column": "phy_annotations",
            "pass_at_least": 2,
            "names": {1: "MUA", 2: "good", 3: "unsorted"},
        },
    },
    "behaviour": {
        "pupil_area": "processing/behavior/PupilTracking/eye_area",
        "wheel": "acquisition/wheel_position",
        "lick_piezo": "acquisition/lick_piezo",
        "nose": "processing/behavior/nose",
    },
    "positions": {"refused": "the test's coordinates are not checked"},
}
SPIKES = [np.linspace(1.0, 99.0, 50) + 0.01 * k for k in range(4)]


def _write(path: Path, *, peaks=(1, 3, 5, 7), names=None, units=True, trials=True) -> Path:
    nwb = NWBFile("hand-built", "intake-test", datetime(2026, 1, 1, tzinfo=UTC))
    names = names or ["CA1", "CA1", "DG", "root", "VISp", "VISp", "LGd", "root"]
    groups = []
    for g in ("ProbeA", "ProbeB"):
        device = nwb.create_device(g)
        groups.append(nwb.create_electrode_group(g, "a probe", "brain", device))
    for row in range(8):
        nwb.add_electrode(location=names[row], group=groups[row // 4])
    if units:
        nwb.add_unit_column("peak_channel", "1-based electrode row of the peak")
        nwb.add_unit_column("cluster_depths", "um from the tip")
        nwb.add_unit_column("phy_annotations", "1 MUA, 2 good, 3 unsorted")
        for k in range(4):
            nwb.add_unit(
                spike_times=SPIKES[k],
                # Per-probe rows, as DANDI 000017 stores them: ProbeB's point into ProbeA.
                electrodes=[peaks[k] - 1 - 4 * (k // 2)],
                electrode_group=groups[k // 2],
                peak_channel=peaks[k],
                cluster_depths=[100.0, 300.0, 200.0, 400.0][k],
                phy_annotations=[2, 1, 3, 2][k],
            )
    if trials:
        nwb.add_trial_column("visual_stimulus_time", "stimulus onset")
        nwb.add_trial_column("feedback_type", "-1 error, +1 reward")
        nwb.add_trial_column("included", "engaged")
        for i in range(6):
            t = 10.0 + 12.0 * i
            nwb.add_trial(
                start_time=t,
                stop_time=t + 10.0,
                visual_stimulus_time=t + 2.0,
                feedback_type=[1, -1][i % 2],
                included=i != 5,
            )
    module = nwb.create_processing_module("behavior", "behaviour")
    pupil = PupilTracking(name="PupilTracking")
    pupil.add_timeseries(
        TimeSeries(
            name="eye_area",
            data=np.arange(1000.0),
            unit="arb. unit",
            timestamps=np.linspace(0.5, 99.5, 1000),
        )
    )
    module.add(pupil)
    module.add(TimeSeries(name="nose", data=np.zeros((10, 2)), unit="px", rate=0.1))
    # The sampling period stored as the rate (1 / 2500 s), as in DANDI 000017.
    nwb.add_acquisition(
        TimeSeries(
            name="wheel_position",
            data=np.arange(5000.0),
            unit="mm",
            rate=0.0004,
            starting_time=1.0,
            conversion=0.135,
        )
    )
    nwb.add_acquisition(
        TimeSeries(name="lick_piezo", data=np.ones(4000), unit="V", rate=50.0, conversion=2.0)
    )
    nwb.add_acquisition(TimeSeries(name="unmapped", data=np.zeros(10), unit="V", rate=1.0))
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return path


def test_a_declared_layout_maps_each_field(tmp_path):
    intake = read_nwb(_write(tmp_path / "f.nwb"), layout_from_dict(LAYOUT))
    s = intake.session
    assert list(s.units.index) == ["ProbeA_0", "ProbeA_1", "ProbeB_2", "ProbeB_3"]
    np.testing.assert_array_equal(s.spikes["ProbeB_3"], SPIKES[3])
    assert s.units["probe_name"].tolist() == ["ProbeA", "ProbeA", "ProbeB", "ProbeB"]
    assert s.units["depths"].tolist() == [100.0, 300.0, 200.0, 400.0]
    # Regions from the declared peak channel (row = value - 1, across probes).
    assert s.units["acronym"].tolist() == ["CA1", "DG", "VISp", "LGd"]
    assert s.units["quality"].tolist() == [2, 1, 3, 2]
    t = s.trials
    assert t["intervals_0"].tolist()[:2] == [10.0, 22.0] and len(t) == 6
    assert t["visual_stimulus_time"].iloc[0] == 12.0
    assert t["included"].tolist() == [1.0] * 5 + [0.0]
    assert t["feedback_type"].dtype == np.float64
    pupil = s.behaviour["pupil_area"]
    assert pupil.timestamps[0] == 0.5 and pupil.data[-1] == 999.0
    lick = s.behaviour["lick_piezo"]  # rate-sampled: timestamps from start and rate
    assert lick.timestamps[1] == pytest.approx(0.02) and lick.data[0] == 2.0  # conversion
    assert "wheel" not in s.behaviour
    assert s.available.missing["behaviour.wheel"].startswith("acquisition/wheel_position: ")
    assert "units.acronym" in s.available.present and "units.x" in s.available.missing


def test_the_capability_report_says_what_was_read_and_why_not(tmp_path):
    r = read_nwb(_write(tmp_path / "f.nwb"), layout_from_dict(LAYOUT)).report
    assert r["layout"] == "test_layout" and r["units"]["n"] == 4
    assert (
        r["units"]["quality"] == "phy_annotations: passes at 2 or more (MUA 1, good 2, unsorted 3)"
    )
    assert r["regions"]["source"] == "electrodes.location at peak_channel - 1"
    assert r["regions"]["allen"] is True
    assert r["positions"] == {"refused": "the test's coordinates are not checked"}
    assert r["trials"]["n"] == 6 and "visual_stimulus_time" in r["trials"]["columns"]
    wheel = r["behaviour"]["wheel"]["refused"]
    assert "0.0004 Hz × 5,000 samples spans 12,500,000 s" in wheel
    assert "longer than 24 h" in wheel and "sampling period" in wheel
    assert r["behaviour"]["pupil_area"]["loaded"].startswith(
        "processing/behavior/PupilTracking/eye_area"
    )
    # 10 samples at 0.1 Hz is a plausible 100 s, but two columns: one channel only.
    assert "2 channels" in r["behaviour"]["nose"]["refused"]
    assert r["not_read"] == ["acquisition/unmapped"]


def test_generic_reading_refuses_contradictory_electrode_links(tmp_path):
    intake = read_nwb(_write(tmp_path / "f.nwb"), GENERIC)
    s, r = intake.session, intake.report
    assert "units.acronym" in s.available.missing
    why = s.available.missing["units.acronym"]
    assert "2 units" in why and "another probe" in why
    assert r["regions"]["refused"] == why
    assert s.behaviour == {} and r["behaviour"] == {}
    assert "quality" not in s.units and r["units"]["quality"] is None


def test_a_peak_channel_off_its_own_probe_refuses_regions(tmp_path):
    path = _write(tmp_path / "f.nwb", peaks=(1, 6, 5, 7))  # unit 1 (ProbeA) -> ProbeB's row
    s = read_nwb(path, layout_from_dict(LAYOUT)).session
    why = s.available.missing["units.acronym"]
    assert "1 unit" in why and "peak_channel" in why and "ProbeA_1" in why


def test_names_that_are_not_allen_acronyms_are_kept_but_not_used_as_regions(tmp_path):
    names = ["hippocampus"] * 4 + ["V1"] * 4
    s = read_nwb(_write(tmp_path / "f.nwb", names=names), layout_from_dict(LAYOUT)).session
    assert s.units["location"].tolist() == ["hippocampus", "hippocampus", "V1", "V1"]
    assert "not Allen CCF 2017 acronyms" in s.available.missing["units.acronym"]


def test_files_and_layouts_outside_the_subset_are_refused(tmp_path):
    with pytest.raises(ValueError, match="no spike-sorted units"):
        read_nwb(_write(tmp_path / "a.nwb", units=False), GENERIC)
    with pytest.raises(ValueError, match="no trials table"):
        read_nwb(_write(tmp_path / "b.nwb", trials=False), GENERIC)
    with pytest.raises(ValueError, match="unknown keys"):
        layout_from_dict({**LAYOUT, "colour": "blue"})
    bad = {**LAYOUT, "units": {**LAYOUT["units"], "location": {"guess": {}}}}
    with pytest.raises(ValueError, match="location"):
        layout_from_dict(bad)
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump(LAYOUT))
    assert load_layout(path).name == "test_layout"
    with pytest.raises(ValueError, match="no NWB layout 'nope'"):
        load_layout("nope")


RICHARDS = (
    Path.home() / "data/neurodecoder/dandi/000017/sub-Richards/sub-Richards_ses-20171031T120000.nwb"
)


@pytest.mark.skipif(not RICHARDS.exists(), reason="the Steinmetz Richards file isn't downloaded")
def test_the_steinmetz_file_reads_as_checked_by_hand():
    intake = read_nwb(RICHARDS, load_layout("steinmetz_2019"))
    s, r = intake.session, intake.report
    assert s.n_units == 778 and len(s.trials) == 260
    regions = s.units["acronym"].value_counts().to_dict()
    assert regions == {
        "ORB": 217,
        "MOs": 122,
        "SCs": 114,
        "RSP": 98,
        "SCm": 75,
        "PAG": 67,
        "MRN": 39,
        "root": 33,
        "OLF": 13,
    }
    assert int((s.units["quality"] >= 2).sum()) == 522
    assert set(s.units["probe_name"]) == {"Probe1", "Probe2"}
    for name in ("wheel", "lick_piezo", "face_motion_energy"):
        assert "sampling period" in r["behaviour"][name]["refused"]
    assert "pupil_area" in s.behaviour
    assert "disagree" in r["positions"]["refused"]
