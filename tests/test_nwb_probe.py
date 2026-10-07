from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries

from unitwave.env import env
from unitwave.nwb.probe import main, probe


def _write(path: Path, *, units=True, trials=True, behaviour=True) -> Path:
    nwb = NWBFile("synthetic", "probe-test", datetime(2026, 1, 1, tzinfo=timezone.utc))
    if units:
        device = nwb.create_device("probe0")
        group = nwb.create_electrode_group("shank0", "s", "CA1", device)
        for i in range(4):
            nwb.add_electrode(location=["CA1", "CA1", "DG", "root"][i], group=group)
        nwb.add_unit_column("quality", "curation label")
        rng = np.random.default_rng(0)
        for i in range(3):
            nwb.add_unit(
                spike_times=np.sort(rng.uniform(0, 100, 50)),
                electrodes=[i],
                quality=["good", "mua", "good"][i],
            )
    if trials:
        nwb.add_trial_column("choice", "animal's choice")
        for i in range(5):
            nwb.add_trial(start_time=10.0 * i, stop_time=10.0 * i + 5, choice=float(i % 2))
    if behaviour:
        module = nwb.create_processing_module("behavior", "behaviour")
        module.add(TimeSeries(name="wheel_position", data=np.zeros(1000), unit="rad", rate=100.0))
        position = Position(name="pose")
        position.add_spatial_series(
            SpatialSeries(
                name="nose_dlc",
                data=np.zeros((600, 2)),
                reference_frame="camera",
                timestamps=np.arange(600) / 60.0,
            )
        )
        module.add(position)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return path


def test_complete_file_is_decodable(tmp_path):
    r = probe(_write(tmp_path / "full.nwb"))
    assert r["opened"] and r["errors"] == {}
    units = r["units"]
    assert units["n_units"] == 3 and units["n_spikes"] == 150
    assert units["qc_columns"]["quality"]["top"] == {"good": 2, "mua": 1}
    assert units["location"]["source"] == "electrodes.location via units.electrodes"
    assert units["location"]["top"] == {"CA1": 2, "DG": 1}
    assert r["intervals"]["trials"]["n"] == 5 and "choice" in r["intervals"]["trials"]["columns"]
    wheel = r["series"]["processing/behavior/wheel_position"]
    assert wheel["rate_hz"] == 100.0 and wheel["duration_s"] == 10.0
    pose = r["series"]["processing/behavior/pose/nose_dlc"]
    assert pose["rate_hz"] == pytest.approx(60.0) and pose["shape"] == [600, 2]
    assert r["behaviour"] == {
        "wheel": ["processing/behavior/wheel_position"],
        "pose": ["processing/behavior/pose", "processing/behavior/pose/nose_dlc"][1:],
    }
    assert r["usable"]["tier"].startswith("decodable")


def test_file_without_units_is_classified_not_crashed(tmp_path):
    r = probe(_write(tmp_path / "no_units.nwb", units=False))
    assert r["opened"] and r["units"] == {"present": False}
    assert r["usable"]["tier"] == "no spike-sorted units"


def test_spikes_only(tmp_path):
    r = probe(_write(tmp_path / "spikes.nwb", trials=False, behaviour=False))
    assert r["usable"]["tier"] == "spikes only" and r["usable"]["behaviour"] == []


def test_unreadable_file_is_a_result_not_an_exception(tmp_path):
    bad = tmp_path / "bad.nwb"
    bad.write_bytes(b"not an hdf5 file")
    r = probe(bad)
    assert r["opened"] is False and "open" in r["errors"]


def test_cli_writes_json(tmp_path, capsys):
    out = tmp_path / "reports.json"
    assert (
        main(
            [str(_write(tmp_path / "full.nwb")), str(tmp_path / "missing.nwb"), "--json", str(out)]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "decodable" in text and "unreadable" in text
    assert out.exists()


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
NWB = (
    DATA_ROOT
    / "dandi/000409/sub-DY-016"
    / f"sub-DY-016_ses-{EID}_desc-processed_behavior+ecephys.nwb"
)


@pytest.mark.skipif(not NWB.exists(), reason="local 000409 file not available")
def test_real_ibl_file():
    r = probe(NWB)
    assert r["opened"] and r["usable"]["spikes"] and r["usable"]["trials"]
    assert {"wheel", "pose", "pupil"} <= set(r["usable"]["behaviour"])
    assert {"ibl_quality_score", "kilosort2_label"} <= set(r["units"]["qc_columns"])
