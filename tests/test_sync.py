"""Step 13a: aligning a Phy folder's events to the probe's clock from sync pulses (the
user's choices, 2026-10-05: CatGT edge files and IBL's sync files; a line, offset plus
drift, refused above 1 ms; different pulse counts refused). Pulses here are test
inputs, except the last test, which reads IBL's own sync files for d23a44ef."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from phy_folder import write_phy_folder

from unitwave.data.backends.phy import load_session_phy
from unitwave.data.sync import fit_clock, load_sync_config, read_pulses

CFG = load_sync_config()


def _pulses(n=3600, offset=2.345, drift_ppm=25.0, jitter_s=20e-6, seed=0):
    """(events clock, probe clock) rising edges of a 1 Hz square wave."""
    rng = np.random.default_rng(seed)
    events = 10.0 + np.arange(n) + rng.normal(0, jitter_s, n)
    probe = offset + (1 + drift_ppm * 1e-6) * events + rng.normal(0, jitter_s, n)
    return events, probe


def test_config_is_the_signed_off_tolerance():
    assert CFG.tolerance_ms == 1.0


def test_a_planted_offset_and_drift_are_recovered():
    events, probe = _pulses()
    fit = fit_clock(probe, events, CFG)
    assert fit.offset_s == pytest.approx(2.345, abs=1e-5)
    assert fit.drift_ppm == pytest.approx(25.0, abs=0.05)
    assert fit.max_residual_ms < 0.2 and fit.rms_residual_ms < 0.05
    assert fit.n_pulses == 3600
    t = np.array([100.0, 2000.0, np.nan])
    mapped = fit.to_probe(t)
    np.testing.assert_allclose(mapped[:2], 2.345 + (1 + 25e-6) * t[:2], atol=1e-5)
    assert np.isnan(mapped[2])  # missing stays missing


def test_a_fit_worse_than_the_tolerance_is_refused():
    events, probe = _pulses()
    probe[1800:] += 0.003  # a 3 ms jump halfway: no line fits both halves
    with pytest.raises(ValueError, match=r"largest residual \d+\.\d+ ms, above the 1 ms tolerance"):
        fit_clock(probe, events, CFG)


def test_different_pulse_counts_are_refused_with_the_counts():
    events, probe = _pulses()
    with pytest.raises(ValueError, match="3599 pulses on the probe's clock and 3600"):
        fit_clock(probe[1:], events, CFG)


def test_events_far_outside_the_pulses_are_refused():
    events, probe = _pulses()
    fit = fit_clock(probe, events, CFG)
    fit.to_probe(np.array([events[0] - 5.0]))  # just before the first pulse: fine
    with pytest.raises(ValueError, match="2 event times lie more than 10 s outside"):
        fit.to_probe(np.array([-5.0, 5000.0, 20.0]))


def test_catgt_edge_files_are_read(tmp_path):
    path = tmp_path / "run_g0_tcat.imec0.ap.xd_384_6_500.txt"
    path.write_text("0.5\n1.5\n2.5\n")
    np.testing.assert_array_equal(read_pulses(path), [0.5, 1.5, 2.5])
    path.write_text("0.5\n2.5\n1.5\n")
    with pytest.raises(ValueError, match="not increasing"):
        read_pulses(path)


def _ibl_triplet(folder: Path, suffix: str, channels: dict[int, np.ndarray]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    times, chans, pols = [], [], []
    for ch, rising in channels.items():
        for t in rising:  # each pulse: up, then down 0.5 s later
            times += [t, t + 0.5]
            chans += [ch, ch]
            pols += [1, -1]
    order = np.argsort(times, kind="stable")
    np.save(folder / f"_spikeglx_sync.times{suffix}.npy", np.asarray(times)[order])
    np.save(folder / f"_spikeglx_sync.channels{suffix}.npy", np.asarray(chans)[order])
    np.save(folder / f"_spikeglx_sync.polarities{suffix}.npy", np.asarray(pols)[order])
    return folder / f"_spikeglx_sync.times{suffix}.npy"


def test_ibl_sync_files_give_rising_edges_of_the_sync_channel(tmp_path):
    one = _ibl_triplet(tmp_path / "probe00", ".probe00", {6: np.arange(5.0)})
    np.testing.assert_array_equal(read_pulses(one), np.arange(5.0))  # its only channel
    many = _ibl_triplet(tmp_path / "nidq", "", {3: np.arange(5.0), 7: np.array([0.2, 3.3])})
    with pytest.raises(ValueError, match=r"channels \[3, 7\]: name the sync channel"):
        read_pulses(many)
    (tmp_path / "nidq" / "_spikeglx_ephysData_g0_t0.nidq.meta").write_text("syncNiChan=3\n")
    np.testing.assert_array_equal(read_pulses(many), np.arange(5.0))  # from the meta file
    np.testing.assert_array_equal(read_pulses(many, channel=7), [0.2, 3.3])


def test_a_phy_folder_s_events_are_moved_onto_the_probe_clock(tmp_path):
    rate = 30000.0
    spike_s = np.arange(0.01, 200.0, 0.05)
    folder = write_phy_folder(
        tmp_path / "imec0", np.round(spike_s * rate), np.zeros(spike_s.size, int)
    )
    events_clock, probe_clock = _pulses(n=180, offset=3.2, drift_ppm=40.0, jitter_s=0.0)
    fit = fit_clock(probe_clock, events_clock, CFG)
    true_starts = 20.0 + 5.0 * np.arange(30)  # on the probe's clock
    on_events_clock = (true_starts - 3.2) / (1 + 40e-6)
    pd.DataFrame(
        {
            "intervals_0": on_events_clock,
            "intervals_1": on_events_clock + 3.0,
            "stimOn_times": on_events_clock + 0.5,
            "contrastLeft": np.full(30, 0.25),  # not a time: never moved
        }
    ).to_csv(tmp_path / "events.csv", index=False)
    s = load_session_phy(folder, tmp_path / "events.csv", clock=fit)
    np.testing.assert_allclose(s.trials["intervals_0"], true_starts, atol=1e-6)
    # 0.5 s on the events' clock is 0.5 * (1 + 40 ppm) s on the probe's.
    np.testing.assert_allclose(s.trials["stimOn_times"], true_starts + 0.5 * (1 + 40e-6), atol=1e-6)
    assert (s.trials["contrastLeft"] == 0.25).all()
    unsynced = load_session_phy(folder, tmp_path / "events.csv")  # today's rule: as written
    np.testing.assert_allclose(unsynced.trials["intervals_0"], on_events_clock)


SYNC = Path.home() / "data/neurodecoder/one/danlab/Subjects/DY_016/2020-09-12/001/raw_ephys_data"


@pytest.mark.skipif(
    not (SYNC / "_spikeglx_sync.times.npy").exists(), reason="d23a44ef sync files not downloaded"
)
def test_ibl_s_own_probe_alignment_is_reproduced_on_d23a44ef():
    probe = read_pulses(SYNC / "probe00" / "_spikeglx_sync.times.probe00.npy")
    nidq = read_pulses(SYNC / "_spikeglx_sync.times.npy")  # channel 3, from the .nidq.meta
    fit = fit_clock(probe, nidq, CFG)
    assert fit.n_pulses == 3669 and fit.max_residual_ms < 0.1
    # IBL's own map: probe time against NIDQ time every 20 s.
    ibl = np.load(SYNC / "probe00" / "_spikeglx_ephysData_g0_t0.imec0.sync.npy")
    inside = (ibl[:, 1] >= nidq.min()) & (ibl[:, 1] <= nidq.max())
    assert np.abs(fit.to_probe(ibl[inside, 1]) - ibl[inside, 0]).max() < 1e-4  # within 0.1 ms


def test_opening_a_phy_folder_with_sync_pulses_from_the_homepage(tmp_path):
    import dataclasses
    import json

    from test_catalog import _manifest
    from test_studio_app import dataclasses_replace_root

    from unitwave.analysis.catalog import load_catalog_config
    from unitwave.data.load import load_data_config
    from unitwave.studio.project import open_project
    from unitwave.studio.server import App

    root = tmp_path / "phy"
    spike_s = np.arange(0.01, 200.0, 0.05)
    folder = write_phy_folder(
        root / "m1" / "probe00", np.round(spike_s * 30000.0), np.zeros(spike_s.size, int)
    )
    events_clock, probe_clock = _pulses(n=180, offset=3.2, drift_ppm=40.0, jitter_s=0.0)
    np.savetxt(root / "m1" / "probe.txt", probe_clock)
    np.savetxt(root / "m1" / "nidq.txt", events_clock)
    starts = (20.0 + 5.0 * np.arange(30) - 3.2) / (1 + 40e-6)
    pd.DataFrame({"intervals_0": starts, "intervals_1": starts + 3.0}).to_csv(
        folder / "events.csv", index=False
    )
    app = App(dataclasses_replace_root(load_data_config(), tmp_path), manifest=_manifest())
    app.catalog_cfg = dataclasses.replace(load_catalog_config(), phy_root=str(root))
    with pytest.raises(ValueError, match="on both clocks"):
        app.open({"kind": "phy", "path": "m1/probe00", "sync_probe": "m1/probe.txt"})
    with pytest.raises(ValueError, match="outside the Phy folder root"):
        app.open(
            {
                "kind": "phy",
                "path": "m1/probe00",
                "sync_probe": "../x.txt",
                "sync_events": "m1/nidq.txt",
            }
        )
    with pytest.raises(ValueError, match="isn't a sync pulse file"):
        app.open(
            {
                "kind": "phy",
                "path": "m1/probe00",
                "sync_probe": "m1/probe00/events.csv",
                "sync_events": "m1/nidq.txt",
            }
        )
    app.open(
        {
            "kind": "phy",
            "path": "m1/probe00",
            "sync_probe": "m1/probe.txt",
            "sync_events": "m1/nidq.txt",
        }
    )
    studio = app.studio
    np.testing.assert_allclose(
        studio.session.trials["intervals_0"], 20.0 + 5.0 * np.arange(30), atol=1e-6
    )
    report = studio.session_json({})["source"]["report"]
    assert report["clock"].startswith("180 sync pulses · offset 3200.000 ms · drift 40.0 ppm")
    # The project records the pulse files; a changed one is reported on reopening.
    studio.save({"event": "stim_on"})
    hashes = json.loads(studio.project_path.read_text())["files"]
    assert {"sync_probe/probe.txt", "sync_events/nidq.txt"} <= set(hashes)
    np.savetxt(root / "m1" / "probe.txt", probe_clock + 1e-4)  # 0.1 ms later, still a line
    _, _, _, warnings = open_project(studio.project_path)
    assert "sync_probe/probe.txt changed since the project was saved" in warnings
