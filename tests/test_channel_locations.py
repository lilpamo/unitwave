"""Step 13b: channel locations for a Phy folder. Histology-aligned channel positions (the
IBL alignment GUI's channel_locations.json, or a CSV of channel, CCF position and
acronym) give each unit the region and position of its peak channel, so Phy units get
the region tree, the brain view and region summaries. A unit's channel is matched by
its position on the probe (lateral, axial) where the file gives one, never guessed.
Files here are test inputs, except the round trip on IBL's own d23a44ef sorting."""

import json

import numpy as np
import pandas as pd
import pytest
from phy_folder import write_phy_folder

from unitwave.analysis.atlas import ccf_um
from unitwave.data.backends.phy import load_session_phy
from unitwave.data.channel_locations import read_channel_locations

# Four channels of a two-column probe: (lateral, axial) µm.
POSITIONS = np.array([[11.0, 20.0], [43.0, 20.0], [11.0, 40.0], [43.0, 40.0]])
ACRONYMS = ["CA1", "CA1", "DG", "VISp"]
IDS = [382, 382, 726, 385]  # Allen ids of those acronyms


def _json(path, positions=POSITIONS, acronyms=ACRONYMS, ids=IDS):
    data = {
        f"channel_{i}": {
            "x": -1300.0 - i,  # µm from bregma, IBL's axes (ML, AP, DV)
            "y": -2200.0,
            "z": -4000.0 + 20 * i,
            "axial": float(positions[i, 1]),
            "lateral": float(positions[i, 0]),
            "brain_region_id": ids[i],
            "brain_region": acronyms[i],
        }
        for i in range(len(positions))
    }
    data["origin"] = {"probe00": "hand-built"}
    path.write_text(json.dumps(data))
    return path


def test_the_alignment_gui_file_is_read_in_ibl_coordinates(tmp_path):
    t = read_channel_locations(_json(tmp_path / "channel_locations.json"))
    assert t["acronym"].tolist() == ACRONYMS
    np.testing.assert_allclose(t[["x", "y", "z"]].iloc[0], [-1300e-6, -2200e-6, -4000e-6])
    np.testing.assert_allclose(t[["lateral_um", "axial_um"]], POSITIONS)


def test_a_region_name_that_disagrees_with_its_id_is_refused(tmp_path):
    with pytest.raises(
        ValueError, match="channel_2: brain_region 'CA3' but brain_region_id 726 is DG"
    ):
        read_channel_locations(_json(tmp_path / "c.json", acronyms=["CA1", "CA1", "CA3", "VISp"]))


def test_a_csv_in_ccf_coordinates_is_read(tmp_path):
    xyz = np.array([[-1300e-6, -2200e-6, -4000e-6], [-1250e-6, -2100e-6, -3500e-6]])
    ccf = ccf_um(xyz)  # (ap, dv, ml) µm
    pd.DataFrame(
        {
            "channel": [7, 8],
            "ccf_ap_um": ccf[:, 0],
            "ccf_dv_um": ccf[:, 1],
            "ccf_ml_um": ccf[:, 2],
            "acronym": ["CA1", "DG"],
        }
    ).to_csv(tmp_path / "locations.csv", index=False)
    t = read_channel_locations(tmp_path / "locations.csv")
    assert t.index.tolist() == [7, 8] and t["acronym"].tolist() == ["CA1", "DG"]
    np.testing.assert_allclose(t[["x", "y", "z"]].to_numpy(), xyz, atol=1e-12)
    pd.DataFrame({"channel": [1], "x": [0.0], "acronym": ["CA1"]}).to_csv(
        tmp_path / "bad.csv", index=False
    )
    with pytest.raises(ValueError, match="ccf_ap_um"):
        read_channel_locations(tmp_path / "bad.csv")
    pd.DataFrame(
        {
            "channel": [1],
            "ccf_ap_um": [1.0],
            "ccf_dv_um": [1.0],
            "ccf_ml_um": [1.0],
            "acronym": ["CA9"],
        }
    ).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="not Allen CCF 2017 acronyms"):
        read_channel_locations(tmp_path / "bad.csv")


def _phy(tmp_path, positions=POSITIONS):
    """Three units: unit 0 peaks on channel 1, unit 1 on channel 2, unit 2 on channel 3."""
    times = np.arange(0.01, 100.0, 0.05)
    clusters = np.arange(times.size) % 3
    templates = np.zeros((3, 5, len(positions)), np.float32)
    for unit, channel in enumerate((1, 2, 3)):
        templates[unit, 2, channel] = -1.0
    folder = write_phy_folder(
        tmp_path / "probe00",
        np.round(times * 30000),
        clusters,
        templates=templates,
        spike_templates=clusters,
        channel_positions=positions,
    )
    starts = 2.0 + 4.0 * np.arange(20) + np.random.default_rng(0).uniform(0, 1, 20)
    pd.DataFrame({"intervals_0": starts, "intervals_1": starts + 3.0}).to_csv(
        tmp_path / "events.csv", index=False
    )
    return folder


def test_units_take_their_peak_channel_s_region_and_position(tmp_path):
    folder = _phy(tmp_path)
    locations = _json(tmp_path / "channel_locations.json")
    s = load_session_phy(folder, tmp_path / "events.csv", locations=locations)
    assert s.units["acronym"].tolist() == ["CA1", "DG", "VISp"]
    np.testing.assert_allclose(s.units["x"], [-1301e-6, -1302e-6, -1303e-6])
    assert {"units.acronym", "units.x", "units.y", "units.z"} <= s.available.present
    plain = load_session_phy(folder, tmp_path / "events.csv")  # as before: no regions
    assert "units.acronym" in plain.available.missing


def test_a_unit_whose_channel_isn_t_in_the_file_is_refused(tmp_path):
    folder = _phy(tmp_path)
    moved = POSITIONS.copy()
    moved[2] = [11.0, 60.0]  # the file's channel 2 sits elsewhere on the probe
    locations = _json(tmp_path / "channel_locations.json", positions=moved)
    with pytest.raises(
        ValueError, match=r"1 unit's peak channel \(lateral 11, axial 40 µm\) is not in"
    ):
        load_session_phy(folder, tmp_path / "events.csv", locations=locations)


def test_locations_need_the_templates_to_find_each_unit_s_channel(tmp_path):
    folder = write_phy_folder(tmp_path / "p", np.arange(100) * 3000, np.zeros(100, int))
    pd.DataFrame({"intervals_0": [0.5], "intervals_1": [2.0]}).to_csv(
        tmp_path / "e.csv", index=False
    )
    with pytest.raises(
        ValueError,
        match="locations need templates.npy, spike_templates.npy and channel_positions.npy",
    ):
        load_session_phy(folder, tmp_path / "e.csv", locations=_json(tmp_path / "c.json"))


# Real data: IBL's own sorting and alignment for d23a44ef probe00, written as a Phy folder
# and a channel_locations.json, against the BWM backend's units.
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"


def test_d23a44ef_units_get_the_bwm_backend_s_regions_and_positions(tmp_path):
    from unitwave.data.load import load_data_config, load_session

    alf = (
        load_data_config().one_cache_root
        / "danlab/Subjects/DY_016/2020-09-12/001/alf/probe00/pykilosort/#2024-05-06#"
    )
    if not alf.exists():
        pytest.skip("ONE cache for d23a44ef not available")
    from iblatlas.regions import BrainRegions

    coords = np.load(alf / "channels.localCoordinates.npy")  # (384, 2) lateral, axial
    mlapdv = np.load(alf / "channels.mlapdv.npy")  # (384, 3) µm
    ids = np.load(alf / "channels.brainLocationIds_ccf_2017.npy")
    acronyms = BrainRegions().id2acronym(ids)
    data = {
        f"channel_{i}": {
            "x": float(mlapdv[i, 0]),
            "y": float(mlapdv[i, 1]),
            "z": float(mlapdv[i, 2]),
            "axial": float(coords[i, 1]),
            "lateral": float(coords[i, 0]),
            "brain_region_id": int(ids[i]),
            "brain_region": str(acronyms[i]),
        }
        for i in range(len(ids))
    }
    (tmp_path / "channel_locations.json").write_text(json.dumps(data))
    # Templates peaking on each cluster's own channel, as Kilosort's do.
    peak = np.load(alf / "clusters.channels.npy")
    templates = np.zeros((peak.size, 3, len(ids)), np.float32)
    templates[np.arange(peak.size), 1, peak] = -1.0
    times = np.load(alf / "spikes.times.npy")
    clusters = np.load(alf / "spikes.clusters.npy")
    folder = write_phy_folder(
        tmp_path / "probe00",
        np.round(times * 30000.0),
        clusters,
        templates=templates,
        spike_templates=clusters,
        channel_positions=coords,
    )
    trials = load_session(EID, "bwm").trials
    pd.DataFrame(
        {"intervals_0": trials["intervals_0"], "intervals_1": trials["intervals_1"]}
    ).to_csv(tmp_path / "events.csv", index=False)
    s = load_session_phy(
        folder, tmp_path / "events.csv", locations=tmp_path / "channel_locations.json"
    )
    bwm = load_session(EID, "bwm").units
    bwm = bwm[bwm["probe_name"] == "probe00"].set_index("cluster_id")
    ours = s.units.set_index("cluster_id").loc[bwm.index]
    assert len(bwm) > 100
    assert ours["acronym"].tolist() == bwm["acronym"].tolist()
    np.testing.assert_allclose(
        ours[["x", "y", "z"]].to_numpy(), bwm[["x", "y", "z"]].to_numpy(), atol=1e-9
    )


def test_opening_a_phy_folder_with_locations_gives_regions_and_the_brain_view(tmp_path):
    import dataclasses

    from test_catalog import _manifest
    from test_studio_app import dataclasses_replace_root

    from unitwave.analysis.catalog import load_catalog_config
    from unitwave.data.load import load_data_config
    from unitwave.studio.server import App

    root = tmp_path / "phy"
    folder = _phy(root / "m1")
    (root / "m1" / "events.csv").rename(folder / "events.csv")
    _json(root / "m1" / "channel_locations.json")
    app = App(dataclasses_replace_root(load_data_config(), tmp_path), manifest=_manifest())
    app.catalog_cfg = dataclasses.replace(load_catalog_config(), phy_root=str(root))
    with pytest.raises(ValueError, match="isn't a channel locations file"):
        app.open({"kind": "phy", "path": "m1/probe00", "locations": "m1/probe00/params.py"})
    app.open({"kind": "phy", "path": "m1/probe00", "locations": "m1/channel_locations.json"})
    studio = app.studio
    assert studio.has_regions and studio.has_positions
    tree = studio.units_json({"all": "1"})["tree"]
    assert tree and {u["region_level"] for u in studio.units_json({"all": "1"})["units"]} == {
        "CA1",
        "DG",
        "VISp",
    }
    geometry = studio.geometry_json({"all": "1"})
    assert len(geometry["units"]) == 3 and len(geometry["tracks"]) == 1
    report = studio.session_json({})["source"]["report"]
    assert report == {"kind": "phy", "clock": None, "regions": "channel_locations.json"}
    studio.save({"event": "stim_on"})
    hashes = json.loads(studio.project_path.read_text())["files"]
    assert "locations/channel_locations.json" in hashes
