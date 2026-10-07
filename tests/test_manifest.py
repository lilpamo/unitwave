import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unitwave.data.manifest import build_manifest, read_manifest, write_manifest
from unitwave.env import env

# Two sessions: "s1" with probes p1 (2 good units of 3 clusters) and p2 (1 of 1),
# and "s2" with probe p3 (1 of 2). s1 has wheel + left/body pose, s2 wheel only.


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _fake_release(tmp_path: Path, **overrides) -> tuple[Path, Path]:
    ephys, behaviour = tmp_path / "bwm_ephys", tmp_path / "bwm_behavior"
    ephys.mkdir()
    behaviour.mkdir()
    (ephys / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "bwm_ephys",
                "dataset_version": overrides.get("ephys_version", "1.2.1"),
            }
        )
    )
    (behaviour / "manifest.json").write_text(
        json.dumps({"dataset_name": "bwm_behavior", "dataset_version": "2.0.0"})
    )

    sessions = pd.DataFrame(
        {
            "eid": ["s1", "s2"],
            "subject": ["mouseA", "mouseB"],
            "date": ["2020-01-01", "2020-01-02"],
            "session_number": [1, 1],
            "lab": ["labX", "labY"],
            "n_trials": [3, 2],
            "n_included_trials": [2, 2],
            "n_insertions": [2, 1],
            "n_good_units": overrides.get("n_good_units", [3, 1]),
        }
    )
    _write(ephys / "metadata/sessions.parquet", sessions)
    _write(
        ephys / "metadata/insertions.parquet",
        pd.DataFrame(
            {
                "pid": ["p1", "p2", "p3"],
                "eid": ["s1", "s1", "s2"],
                "probe_name": ["probe00", "probe01", "probe00"],
            }
        ),
    )
    # p3's cluster 1 has label 1 but, like BWM's out-of-brain units, is not a good unit.
    _write(
        ephys / "clusters.pqt",
        pd.DataFrame(
            {
                "pid": ["p1", "p1", "p1", "p2", "p3", "p3"],
                "eid": ["s1", "s1", "s1", "s1", "s2", "s2"],
                "cluster_id": [0, 1, 2, 0, 0, 1],
                "label": [1.0, 1.0, 0.33, 1.0, 1.0, 1.0],
            }
        ),
    )
    _write(
        ephys / "metadata/units.parquet",
        pd.DataFrame(
            {
                "pid": ["p1", "p1", "p2", "p3"],
                "eid": ["s1", "s1", "s1", "s2"],
                "cluster_id": overrides.get("unit_cluster_ids", [0, 1, 0, 0]),
                "acronym": ["CA1", "PO", "CA1", "LP"],
                "beryl_acronym": ["CA1", "PO", "CA1", "LP"],
            }
        ),
    )
    _write(
        ephys / "metadata/trials.parquet",
        pd.DataFrame(
            {"eid": ["s1", "s1", "s1", "s2", "s2"], "bwm_include": [True, True, False, True, True]}
        ),
    )
    # Probe p1 runs from its tip at (100, -2000, -3000) um up to (100, -2000, -1000) um.
    _write(
        ephys / "metadata/channels.parquet",
        pd.DataFrame(
            {
                "pid": ["p1", "p1", "p2", "p3"],
                "localCoordinates_y": [20.0, 3000.0, 20.0, 20.0],
                "mlapdv_x": [100.0, 100.0, 0.0, 0.0],
                "mlapdv_y": [-2000.0, -2000.0, 0.0, 0.0],
                "mlapdv_z": [-3000.0, -1000.0, -500.0, -500.0],
            }
        ),
    )
    _write(
        behaviour / "metadata/wheel_availability.parquet",
        pd.DataFrame({"eid": ["s1", "s2"], "wheel_present": [True, True]}),
    )
    _write(
        behaviour / "metadata/pose_availability.parquet",
        pd.DataFrame(
            {
                "eid": ["s1", "s1", "s2"],
                "camera": ["leftCamera", "bodyCamera", ""],
                "pose_present": [True, True, False],
            }
        ),
    )
    # s1's session shard: the left camera has motion energy and a skipped pupil; the body
    # camera has motion energy. s2 has no shard.
    shard = behaviour / "sessions" / "s1.zip"
    shard.parent.mkdir()
    meta = {
        "arrays": {"leftCamera.features": {}, "bodyCamera.features": {}},
        "cameras": {
            "leftCamera": {
                "columns": ["nose_tip_x", "whiskerMotionEnergy", "pupilDiameter_raw"],
                "skipped_sources": ["pupilDiameter_raw"],
            },
            "bodyCamera": {"columns": ["bodyMotionEnergy"]},
        },
    }
    with zipfile.ZipFile(shard, "w") as z:
        z.writestr("meta.json", json.dumps(meta))
    return ephys, behaviour


def test_session_counts(tmp_path):
    m = build_manifest(*_fake_release(tmp_path))
    s = m.sessions.set_index("eid")
    assert s.loc["s1", "n_probes"] == 2 and s.loc["s2", "n_probes"] == 1
    assert s.loc["s1", "n_units"] == 4 and s.loc["s1", "n_good_units"] == 3
    assert s.loc["s2", "n_units"] == 2 and s.loc["s2", "n_good_units"] == 1
    assert s.loc["s1", "n_label1_units"] == 3 and s.loc["s2", "n_label1_units"] == 2
    assert s.loc["s1", "n_trials"] == 3 and s.loc["s1", "n_included_trials"] == 2


def test_good_units_must_have_label_1(tmp_path):
    # Unit table lists p1's cluster 2, whose label is 0.33.
    with pytest.raises(ValueError, match="label"):
        build_manifest(*_fake_release(tmp_path, unit_cluster_ids=[0, 2, 0, 0]))


def test_region_list_is_sorted_and_unique(tmp_path):
    s = build_manifest(*_fake_release(tmp_path)).sessions.set_index("eid")
    assert list(s.loc["s1", "regions"]) == ["CA1", "PO"]
    assert list(s.loc["s2", "regions"]) == ["LP"]


def test_modalities_use_canonical_names(tmp_path):
    s = build_manifest(*_fake_release(tmp_path)).sessions.set_index("eid")
    assert list(s.loc["s1", "modalities"]) == [
        "motion_energy_body",
        "motion_energy_left",
        "pose_body",
        "pose_left",
        "wheel",
    ]
    assert list(s.loc["s2", "modalities"]) == ["wheel"]


def test_motion_energy_and_pupil_follow_the_backends_rule(tmp_path):
    # Present only if the camera has features and the column, and it wasn't skipped.
    m = build_manifest(*_fake_release(tmp_path))
    mods = dict(zip(m.sessions["eid"], m.sessions["modalities"]))
    assert "motion_energy_left" in mods["s1"] and "motion_energy_body" in mods["s1"]
    assert "pupil_left" not in mods["s1"]  # skipped by the bwm_behavior build
    assert not [x for x in mods["s2"] if x.startswith(("motion_energy", "pupil"))]


def test_region_units_count_good_units_per_allen_region(tmp_path):
    m = build_manifest(*_fake_release(tmp_path))
    table = m.region_units.set_index(["pid", "acronym"])["n_good_units"]
    assert table.to_dict() == {
        ("p1", "CA1"): 1,
        ("p1", "PO"): 1,
        ("p2", "CA1"): 1,
        ("p3", "LP"): 1,
    }
    assert m.provenance["manifest_version"] == 2


def test_insertion_tip_and_top_in_meters(tmp_path):
    ins = build_manifest(*_fake_release(tmp_path)).insertions.set_index("pid")
    tip = ins.loc["p1", ["tip_x", "tip_y", "tip_z"]].to_numpy(float)
    top = ins.loc["p1", ["top_x", "top_y", "top_z"]].to_numpy(float)
    np.testing.assert_allclose(tip, [100e-6, -2000e-6, -3000e-6])
    np.testing.assert_allclose(top, [100e-6, -2000e-6, -1000e-6])
    assert ins.loc["p1", "n_units"] == 3 and ins.loc["p1", "n_good_units"] == 2


def test_disagreement_with_release_counts_raises(tmp_path):
    with pytest.raises(ValueError, match="n_good_units"):
        build_manifest(*_fake_release(tmp_path, n_good_units=[3, 2]))


def test_wrong_release_version_raises(tmp_path):
    with pytest.raises(ValueError, match="9.9.9"):
        build_manifest(*_fake_release(tmp_path, ephys_version="9.9.9"))


def test_write_and_read_round_trip(tmp_path):
    m = build_manifest(*_fake_release(tmp_path))
    write_manifest(m, tmp_path / "manifest")
    back = read_manifest(tmp_path / "manifest")
    assert back.provenance == m.provenance
    for original, loaded in [(m.sessions, back.sessions), (m.insertions, back.insertions)]:
        assert list(original.columns) == list(loaded.columns)
        assert len(original) == len(loaded)
    s = back.sessions.set_index("eid")
    assert list(s.loc["s1", "regions"]) == ["CA1", "PO"]
    pd.testing.assert_frame_equal(back.region_units, m.region_units)


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EPHYS = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
BEHAVIOUR = DATA_ROOT / "bwm_compressed/bwm_behavior/2.0.0"


@pytest.fixture(scope="module")
def real():
    return build_manifest(EPHYS, BEHAVIOUR)


needs_bwm = pytest.mark.skipif(
    not (EPHYS.exists() and BEHAVIOUR.exists()), reason="BWM releases not available"
)


@needs_bwm
def test_real_totals_match_the_verified_release_counts(real):
    s = real.sessions
    assert len(s) == 459 and s["eid"].is_unique
    assert s["subject"].nunique() == 139 and s["lab"].nunique() == 12
    assert s["n_probes"].sum() == 699 == len(real.insertions)
    assert s["n_good_units"].sum() == 75_395
    # 313 label-1 clusters are not good units: 307 in void/root, 6 unexplained.
    assert s["n_label1_units"].sum() == 75_708
    assert s["n_units"].sum() == 621_733
    assert s["n_trials"].sum() == 295_920


@needs_bwm
def test_real_row_for_the_reference_session(real):
    row = real.sessions.set_index("eid").loc["d23a44ef-1402-4ed7-97f5-47e9a7a504d9"]
    assert (
        row["subject"],
        row["n_probes"],
        row["n_units"],
        row["n_good_units"],
        row["n_trials"],
    ) == (
        "DY_016",
        2,
        1961,
        398,
        410,
    )
    assert "wheel" in list(row["modalities"])


@needs_bwm
def test_real_probe_tips_are_deeper_than_their_tops(real):
    ins = real.insertions
    assert (ins["tip_z"] < ins["top_z"]).all()


@needs_bwm
def test_real_sessions_without_pose(real):
    has_pose = real.sessions["modalities"].map(lambda m: any(x.startswith("pose_") for x in m))
    assert int((~has_pose).sum()) == 15


@needs_bwm
def test_real_modalities_match_what_the_backend_loads(real):
    from unitwave.data.backends.bwm_compressed import load_session_bwm

    eid = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
    listed = set(real.sessions.set_index("eid").loc[eid, "modalities"])
    loaded = set(load_session_bwm(eid, EPHYS, behaviour_root=BEHAVIOUR).behaviour)
    assert listed == loaded
