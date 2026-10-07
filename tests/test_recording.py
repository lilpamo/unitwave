"""Step 14a: a recording (several Phy probes, events, rig behaviour) described by a
recording.yaml (the user's choices, 2026-10-05). Every probe's spikes, and behaviour
recorded on a probe's clock, move onto the shared events clock, each by its own sync
fit; a probe without pulses is taken to be on the events clock already. Spikes beyond
the pulses are dropped and counted. Files here are test inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from phy_folder import write_phy_folder

from unitwave.data.backends.recording import load_recording

RATE = 30000.0
CLOCKS = {"imec0": (1.5, 20.0), "imec1": (-0.7, -15.0)}  # offset s, drift ppm vs events
STIM = 20.0 + 5.0 * np.arange(40) + np.linspace(0, 1.3, 40)  # events clock


def _on_probe(name, t):
    offset, drift = CLOCKS[name]
    return offset + (1 + drift * 1e-6) * np.asarray(t)


def _write(root: Path, extra_spike_s=None, sync=True, behaviour=True) -> Path:
    """Two probes; on each, cluster 1 fires 10 ms after every stimulus (planted on the
    events clock) and cluster 2 fires evenly. Pulses every second on every clock."""
    root.mkdir(parents=True, exist_ok=True)
    pulses = np.arange(5.0, 240.0, 1.0)
    for name in CLOCKS:
        locked = STIM + 0.010
        even = np.arange(1.0, 235.0, 0.1)  # after every probe's clock starts
        t_events = np.r_[locked, even, [] if extra_spike_s is None else extra_spike_s]
        clusters = np.r_[
            np.ones(locked.size, int),
            np.full(even.size + (0 if extra_spike_s is None else len(extra_spike_s)), 2),
        ]
        t_probe = _on_probe(name, t_events) if sync else t_events
        order = np.argsort(t_probe)
        write_phy_folder(
            root / name,
            np.round(t_probe[order] * RATE),
            clusters[order],
            ks_label={1: "good", 2: "good"},
        )
        np.savetxt(root / f"{name}.txt", _on_probe(name, pulses))
    np.savetxt(root / "nidq.txt", pulses)
    pd.DataFrame(
        {"intervals_0": STIM - 0.5, "intervals_1": STIM + 3.0, "stimOn_times": STIM}
    ).to_csv(root / "events.csv", index=False)
    recording = {
        "events": "events.csv",
        "probes": [
            {"name": n, "folder": n} | ({"sync": f"{n}.txt"} if sync else {}) for n in CLOCKS
        ],
    }
    if sync:
        recording["events_sync"] = "nidq.txt"
    if behaviour:
        (root / "behaviour").mkdir(exist_ok=True)
        t = np.arange(0.0, 230.0, 0.01)
        pd.DataFrame({"time": t, "position": np.sin(t)}).to_csv(
            root / "behaviour" / "wheel.csv", index=False
        )
        np.save(root / "behaviour" / "pupil.times.npy", _on_probe("imec1", t) if sync else t)
        np.save(root / "behaviour" / "pupil.values.npy", np.cos(t))
        recording["behaviour"] = {
            "wheel": {"file": "behaviour/wheel.csv", "clock": "events"},
            "pupil_left": {"file": "behaviour/pupil.times.npy", "clock": "imec1"},
        }
    (root / "recording.yaml").write_text(yaml.safe_dump(recording))
    return root / "recording.yaml"


def test_every_probe_s_spikes_land_on_the_events_clock(tmp_path):
    session, report = load_recording(_write(tmp_path))
    for name in CLOCKS:
        locked = session.spikes[f"{name}_1"]
        np.testing.assert_allclose(locked, STIM + 0.010, atol=5e-5)  # within a sample
    assert sorted(session.units["probe_name"].unique()) == ["imec0", "imec1"]
    np.testing.assert_allclose(session.trials["stimOn_times"], STIM)  # events untouched
    probes = {p["name"]: p for p in report["probes"]}
    assert probes["imec0"]["clock"].startswith(
        "235 sync pulses · offset 1500.000 ms · drift 20.0 ppm"
    )
    assert probes["imec1"]["dropped_spikes"] == 0


def test_behaviour_is_read_and_moved_onto_the_events_clock(tmp_path):
    session, report = load_recording(_write(tmp_path))
    wheel, pupil = session.behaviour["wheel"], session.behaviour["pupil_left"]
    np.testing.assert_allclose(wheel.timestamps[:3], [0.0, 0.01, 0.02])
    np.testing.assert_allclose(
        pupil.timestamps[:3], [0.0, 0.01, 0.02], atol=1e-9
    )  # from imec1's clock
    np.testing.assert_allclose(pupil.data[:3], np.cos([0.0, 0.01, 0.02]))
    assert report["behaviour"]["pupil_left"] == {
        "file": "behaviour/pupil.times.npy",
        "clock": "imec1",
        "samples": 23000,
    }
    assert "behaviour.wheel" in session.available.present


def test_spikes_beyond_the_pulses_are_dropped_and_counted(tmp_path):
    session, report = load_recording(_write(tmp_path, extra_spike_s=[300.0, 301.0]))
    probes = {p["name"]: p for p in report["probes"]}
    assert probes["imec0"]["dropped_spikes"] == 2 and probes["imec1"]["dropped_spikes"] == 2
    assert session.spikes["imec0_2"].max() < 250.0


def test_probes_without_pulses_are_taken_to_be_on_the_events_clock(tmp_path):
    session, report = load_recording(_write(tmp_path, sync=False, behaviour=False))
    np.testing.assert_allclose(session.spikes["imec0_1"], STIM + 0.010, atol=5e-5)
    assert all(p["clock"] is None for p in report["probes"])


def test_a_malformed_recording_is_refused_saying_where(tmp_path):
    path = _write(tmp_path)

    def broken(change):
        raw = yaml.safe_load(path.read_text())
        change(raw)
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw))
        return bad

    cases = [
        (lambda r: r.update(colour="red"), "unknown keys \\['colour'\\]"),
        (lambda r: r["probes"][1].update(name="imec0"), "probe name imec0 is used twice"),
        (lambda r: r["probes"][0].update(folder="../elsewhere"), "outside the recording's folder"),
        (lambda r: r["probes"][0].update(folder="nothere"), "no nothere"),
        (lambda r: r.pop("events_sync"), "probes have sync pulses but the events' clock has none"),
        (lambda r: r["behaviour"]["wheel"].update(clock="imec9"), "clock imec9: events or one of"),
        (lambda r: r.update(probes=[]), "at least one probe"),
    ]
    for change, message in cases:
        with pytest.raises(ValueError, match=message):
            load_recording(broken(change))
    pd.DataFrame({"t": [0.0, 1.0], "position": [0, 1]}).to_csv(
        tmp_path / "behaviour" / "wheel.csv", index=False
    )
    with pytest.raises(ValueError, match="wheel.csv: needs a time column"):
        load_recording(path)


def test_opening_a_recording_folder_from_the_homepage(tmp_path):
    import dataclasses

    from test_catalog import _manifest
    from test_studio_app import dataclasses_replace_root

    from unitwave.analysis.catalog import load_catalog_config
    from unitwave.data.load import load_data_config
    from unitwave.studio.server import App

    root = tmp_path / "phy"
    _write(root / "mouse1" / "2026-10-01")
    app = App(dataclasses_replace_root(load_data_config(), tmp_path), manifest=_manifest())
    app.catalog_cfg = dataclasses.replace(load_catalog_config(), phy_root=str(root))
    app.open({"kind": "phy", "path": "mouse1/2026-10-01"})
    studio = app.studio
    assert studio.source.kind == "recording" and studio.probes == ["imec0", "imec1"]
    report = studio.session_json({})["source"]["report"]
    assert report["kind"] == "recording" and len(report["probes"]) == 2
    wheel = studio.wheel_data(
        {"event": "stim_on", "t0": "-0.5", "t1": "1.0", "bin": "0.02", "all": "1"}
    )
    assert wheel["psth"].n_trials == 40  # the rig's wheel, on the events clock
    studio.save({"event": "stim_on"})
    hashes = json.loads(studio.project_path.read_text())["files"]
    assert {
        "recording.yaml",
        "events.csv",
        "imec0/spike_times.npy",
        "behaviour/pupil.values.npy",
    } <= set(hashes)


def test_studio_starts_on_a_recording_from_the_command_line(tmp_path):
    from unitwave.studio.server import build_app

    path = _write(tmp_path / "rec")
    app, first = build_app(
        ["--recording", str(path), "--project", str(tmp_path / "p.unitwave.json")]
    )
    assert first == "/session" and app.studio.source.kind == "recording"
    assert sorted(app.studio.session.units["probe_name"].unique()) == ["imec0", "imec1"]
