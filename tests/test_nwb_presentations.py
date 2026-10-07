"""Step 6: NWB files without a trials table, whose stimulus presentations play the role
of trials (Allen Brain Observatory Visual Coding, DANDI 000021). A small file is written
here with the layouts found there: presentations tables per stimulus, numbers stored as
text, units linked to electrodes by electrode id, empty locations, and invalid times.
Test inputs only; never shown as data."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile
from pynwb.epoch import TimeIntervals

from unitwave.nwb.intake import layout_from_dict, read_nwb

LAYOUT = {
    "name": "presentations_test",
    "label": "Presentations test",
    "task": "t_gratings",
    "trials": {
        "by_task": {"t_gratings": "gratings_presentations", "t_flashes": "flashes_presentations"},
        "mark_invalid_times": True,
    },
    "units": {
        "probe": "electrode_group",
        "depth": {"electrodes_column": "probe_vertical_position"},
        "location": {"electrode_id": {"column": "peak_channel_id"}},
        "quality": {"column": "quality", "pass_values": ["good"]},
    },
}
IDS = [850001, 850002, 850003, 850004]  # electrode ids, not rows


def _write(path: Path) -> Path:
    nwb = NWBFile("hand-built", "presentations-test", datetime(2026, 1, 1, tzinfo=UTC))
    groups = []
    for g in ("probeA", "probeB"):
        groups.append(nwb.create_electrode_group(g, "a probe", "brain", nwb.create_device(g)))
    nwb.add_electrode_column("probe_vertical_position", "um along the probe from its tip")
    # Allen's files hold "" for electrodes outside the brain; pynwb now refuses to write an
    # empty location, so a blank stands in for it (both are read as missing).
    for k, (loc, depth) in enumerate(zip(["VISp", " ", "LGd", "CA1"], [20.0, 40.0, 60.0, 80.0])):
        nwb.add_electrode(
            location=loc, group=groups[k // 2], probe_vertical_position=depth, id=IDS[k]
        )
    nwb.add_unit_column("peak_channel_id", "electrode id of the peak")
    nwb.add_unit_column("quality", "good or noise")
    for k, peak in enumerate([IDS[0], IDS[1], IDS[2], IDS[3]]):
        nwb.add_unit(
            spike_times=np.linspace(1.0, 99.0, 40) + 0.01 * k,
            electrode_group=groups[k // 2],
            peak_channel_id=peak,
            quality=["good", "good", "noise", "good"][k],
        )
    gratings = TimeIntervals(name="gratings_presentations", description="gratings")
    gratings.add_column("orientation", "degrees; NaN for blank")
    gratings.add_column("spatial_frequency", "cycles/degree, as text")
    gratings.add_column("size", "degrees, as text pairs")
    for i in range(6):
        t = 10.0 + 10.0 * i
        gratings.add_interval(
            start_time=t,
            stop_time=t + 2.0,
            orientation=[0.0, 90.0, np.nan, 0.0, 90.0, 0.0][i],
            spatial_frequency=["0.04", "0.04", "N/A", "0.08", "0.08", "0.04"][i],
            size="[250.0, 250.0]",
        )
    nwb.add_time_intervals(gratings)
    flashes = TimeIntervals(name="flashes_presentations", description="flashes")
    flashes.add_column("color", "-1 dark, 1 light, as text")
    for i in range(4):
        flashes.add_interval(start_time=80.0 + i, stop_time=80.25 + i, color=["-1.0", "1.0"][i % 2])
    nwb.add_time_intervals(flashes)
    # probeB's data are invalid over part of the third presentation.
    nwb.add_invalid_time_interval(start_time=31.0, stop_time=31.5, tags=["probeB"])
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return path


def test_a_tasks_presentations_play_the_role_of_trials(tmp_path):
    intake = read_nwb(_write(tmp_path / "f.nwb"), layout_from_dict(LAYOUT), task="t_gratings")
    t, r = intake.session.trials, intake.report
    assert r["trials"]["source"] == "intervals/gratings_presentations"
    assert t["intervals_0"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    np.testing.assert_array_equal(t["orientation"], [0.0, 90.0, np.nan, 0.0, 90.0, 0.0])
    # Numbers stored as text are read as numbers; N/A is missing, not zero.
    np.testing.assert_array_equal(t["spatial_frequency"], [0.04, 0.04, np.nan, 0.08, 0.08, 0.04])
    assert r["trials"]["converted"] == {
        "spatial_frequency": "numbers stored as text; 'N/A' read as missing"
    }
    assert t["size"].iloc[0] == "[250.0, 250.0]"  # not numbers: kept as text
    flashes = read_nwb(tmp_path / "f.nwb", layout_from_dict(LAYOUT), task="t_flashes").session
    assert flashes.trials["color"].tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_presentations_overlapping_invalid_times_are_marked(tmp_path):
    intake = read_nwb(_write(tmp_path / "f.nwb"), layout_from_dict(LAYOUT), task="t_gratings")
    assert intake.session.trials["invalid_overlap"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert intake.report["trials"]["invalid_times"] == ["31.0–31.5 s (probeB)"]


def test_units_link_to_electrodes_by_id_and_empty_locations_are_missing(tmp_path):
    intake = read_nwb(_write(tmp_path / "f.nwb"), layout_from_dict(LAYOUT), task="t_gratings")
    u, r = intake.session.units, intake.report
    assert u["acronym"].isna().tolist() == [False, True, False, False]
    assert u["acronym"].dropna().tolist() == ["VISp", "LGd", "CA1"]
    assert u["depths"].tolist() == [20.0, 40.0, 60.0, 80.0]
    assert r["regions"]["source"] == "electrodes.location at peak_channel_id (an electrode id)"
    assert r["regions"]["no_location"] == 1
    assert u["quality"].tolist() == ["good", "good", "noise", "good"]


def test_a_task_without_presentations_or_trials_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no trials table"):
        read_nwb(_write(tmp_path / "f.nwb"), layout_from_dict(LAYOUT), task="other")
    bad = {**LAYOUT, "trials": {"by_task": {"t_gratings": "missing_presentations"}}}
    with pytest.raises(ValueError, match="missing_presentations"):
        read_nwb(tmp_path / "f.nwb", layout_from_dict(bad), task="t_gratings")


def test_a_units_probe_can_come_from_its_electrode(tmp_path):
    # Allen's units have no electrode_group: the probe is the peak electrode's group.
    layout = {**LAYOUT, "units": {**LAYOUT["units"], "probe": "location_electrode"}}
    s = read_nwb(_write(tmp_path / "f.nwb"), layout_from_dict(layout), task="t_gratings").session
    assert s.units["probe_name"].tolist() == ["probeA", "probeA", "probeB", "probeB"]
    assert s.units["acronym"].dropna().tolist() == ["VISp", "LGd", "CA1"]
