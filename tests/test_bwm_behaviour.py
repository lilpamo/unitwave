import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from numcodecs import Blosc

from unitwave.data.backends.bwm_compressed import _behaviour_from_shard, _read_behaviour_shard
from unitwave.env import env

EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
LEFT = [
    "paw_l_x",
    "paw_l_y",
    "paw_l_likelihood",
    "whiskerMotionEnergy",
    "pupilDiameter_raw",
    "pupilDiameter_smooth",
]


def _blob(zf: zipfile.ZipFile, entry: str, arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    zf.writestr(entry, Blosc(cname="zstd", clevel=7).encode(arr))
    return {"dtype": arr.dtype.str, "shape": list(arr.shape), "nbytes": int(arr.nbytes)}


def _write_shard(path: Path, *, skipped=(), with_body=False) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        arrays = {
            "wheel.timestamps": {
                "encoding": {
                    "kind": "timestamp_delta_ticks",
                    "start": 1.0,
                    "tick_s": 0.001,
                    "ticks_entry": "arrays/wheel.t.blosc",
                    "ticks_spec": _blob(
                        zf, "arrays/wheel.t.blosc", np.array([0, 10, 20], dtype="<u4")
                    ),
                }
            },
            "wheel.position": {
                "encoding": {
                    "kind": "scaled_numeric",
                    "precision": 0.001,
                    "entry": "arrays/wheel.p.blosc",
                    "payload_spec": _blob(
                        zf, "arrays/wheel.p.blosc", np.array([0, 5, -3], dtype="<i4")
                    ),
                }
            },
            "leftCamera.timestamps": {
                "encoding": {
                    "kind": "timestamp_fixed_rate",
                    "start": 2.0,
                    "rate_hz": 60.0,
                    "count": 3,
                }
            },
            "leftCamera.features": {
                "encoding": {
                    "kind": "feature_matrix",
                    "shape": [3, len(LEFT)],
                    "columns": LEFT,
                    "groups": [
                        {
                            "kind": "delta_scaled",
                            "columns": [0, 1],
                            "precision": 0.5,
                            "first_entry": "arrays/l.first.blosc",
                            "first_spec": _blob(
                                zf, "arrays/l.first.blosc", np.array([[20, 40]], dtype="<i4")
                            ),
                            "delta_entry": "arrays/l.delta.blosc",
                            "delta_spec": _blob(
                                zf, "arrays/l.delta.blosc", np.array([[1, -2], [3, 0]], dtype="<i2")
                            ),
                        },
                        {
                            "kind": "likelihood",
                            "columns": [2],
                            "bits": 8,
                            "entry": "arrays/l.lik.blosc",
                            "payload_spec": _blob(
                                zf, "arrays/l.lik.blosc", np.array([[255], [0], [51]], dtype="u1")
                            ),
                        },
                        {
                            "kind": "scaled",
                            "columns": [3, 4, 5],
                            "precision": 0.05,
                            "entry": "arrays/l.scaled.blosc",
                            "payload_spec": _blob(
                                zf,
                                "arrays/l.scaled.blosc",
                                np.array(
                                    [[10, 60, 61], [20, -32768, 62], [30, 64, 63]], dtype="<i2"
                                ),
                            ),
                        },
                    ],
                }
            },
        }
        cameras = {
            "leftCamera": {
                "columns": LEFT,
                "skipped_sources": list(skipped),
                "tracker": "lightningPose",
            }
        }
        if with_body:
            arrays["bodyCamera.timestamps"] = {
                "encoding": {
                    "kind": "timestamp_fixed_rate",
                    "start": 2.0,
                    "rate_hz": 30.0,
                    "count": 2,
                }
            }
            arrays["bodyCamera.features"] = {
                "encoding": {
                    "kind": "feature_matrix",
                    "shape": [2, 1],
                    "columns": ["bodyMotionEnergy"],
                    "groups": [
                        {
                            "kind": "scaled",
                            "columns": [0],
                            "precision": 0.05,
                            "entry": "arrays/b.blosc",
                            "payload_spec": _blob(
                                zf, "arrays/b.blosc", np.array([[4], [8]], dtype="<i2")
                            ),
                        }
                    ],
                }
            }
            cameras["bodyCamera"] = {
                "columns": ["bodyMotionEnergy"],
                "skipped_sources": [],
                "tracker": "lightningPose",
            }
        meta = {
            "format": "ibl_ai_agent_behavior_session_shard_v2",
            "eid": EID,
            "arrays": arrays,
            "cameras": cameras,
            "wheel": {"present": True, "fs": 100.0},
        }
        zf.writestr(f"{EID}/meta.json", json.dumps(meta))
    return path


def test_decodes_every_encoding_kind(tmp_path):
    arrays, _ = _read_behaviour_shard(_write_shard(tmp_path / "s.zip"))
    np.testing.assert_allclose(arrays["wheel.timestamps"], [1.0, 1.01, 1.02])
    np.testing.assert_allclose(arrays["wheel.position"], [0.0, 0.005, -0.003], atol=1e-7)
    np.testing.assert_allclose(arrays["leftCamera.timestamps"], [2.0, 2.0 + 1 / 60, 2.0 + 2 / 60])
    feat = arrays["leftCamera.features"]
    np.testing.assert_allclose(feat[:, 0], [10.0, 10.5, 12.0])
    np.testing.assert_allclose(feat[:, 1], [20.0, 19.0, 19.0])
    np.testing.assert_allclose(feat[:, 2], [1.0, 0.0, 0.2], atol=1e-6)
    np.testing.assert_allclose(feat[:, 3], [0.5, 1.0, 1.5], atol=1e-6)
    assert np.isnan(feat[1, 4])


def test_unknown_encoding_kind_rejected(tmp_path):
    path = _write_shard(tmp_path / "s.zip")
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read(f"{EID}/meta.json"))
        blobs = {n: zf.read(n) for n in zf.namelist() if not n.endswith("meta.json")}
    meta["arrays"]["wheel.position"]["encoding"]["kind"] = "mystery"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in blobs.items():
            zf.writestr(name, data)
        zf.writestr(f"{EID}/meta.json", json.dumps(meta))
    with pytest.raises(ValueError, match="mystery"):
        _read_behaviour_shard(path)


def test_maps_cameras_to_canonical_keys(tmp_path):
    behaviour, missing = _behaviour_from_shard(_write_shard(tmp_path / "s.zip", with_body=True))
    assert behaviour["pose_left"].channel_names == ("paw_l_x", "paw_l_y", "paw_l_likelihood")
    np.testing.assert_allclose(behaviour["motion_energy_left"].data, [0.5, 1.0, 1.5], atol=1e-6)
    assert np.isnan(behaviour["pupil_left"].data[1])
    np.testing.assert_allclose(behaviour["motion_energy_body"].data, [0.2, 0.4], atol=1e-6)
    assert "pose_body" not in behaviour and missing["pose_body"]
    assert missing["pupil_right"] and missing["pose_right"] and missing["lick"]


def test_wheel_is_position_only(tmp_path):
    behaviour, _ = _behaviour_from_shard(_write_shard(tmp_path / "s.zip"))
    assert behaviour["wheel"].data.shape == (3,)
    assert behaviour["wheel"].channel_names is None


def test_pupil_uses_the_raw_not_the_smoothed_diameter(tmp_path):
    behaviour, _ = _behaviour_from_shard(_write_shard(tmp_path / "s.zip"))
    np.testing.assert_allclose(behaviour["pupil_left"].data[[0, 2]], [3.0, 3.2], atol=1e-6)


def test_skipped_sources_are_declared_missing(tmp_path):
    behaviour, missing = _behaviour_from_shard(
        _write_shard(tmp_path / "s.zip", skipped=["whiskerMotionEnergy"])
    )
    assert "motion_energy_left" not in behaviour
    assert "skipped" in missing["motion_energy_left"]


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
BWM_EPHYS = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
BWM_BEHAVIOUR = DATA_ROOT / "bwm_compressed/bwm_behavior/2.0.0"
NWB_PATH = (
    DATA_ROOT
    / "dandi/000409/sub-DY-016"
    / f"sub-DY-016_ses-{EID}_desc-processed_behavior+ecephys.nwb"
)
needs_real = pytest.mark.skipif(
    not (BWM_EPHYS.exists() and BWM_BEHAVIOUR.exists() and NWB_PATH.exists()),
    reason="BWM releases and the NWB file are needed",
)


@pytest.fixture(scope="module")
def sessions():
    from unitwave.data.backends.bwm_compressed import load_session_bwm
    from unitwave.data.backends.dandi_nwb import load_session_nwb

    return load_session_bwm(EID, BWM_EPHYS, behaviour_root=BWM_BEHAVIOUR), load_session_nwb(
        NWB_PATH
    )


def _nearest(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    return np.where(np.abs(times[idx - 1] - targets) <= np.abs(times[idx] - targets), idx - 1, idx)


@needs_real
def test_real_wheel_is_raw_position_interpolated_to_100_hz(sessions):
    bwm, nwb = sessions
    w, raw = bwm.behaviour["wheel"], nwb.behaviour["wheel"]
    assert w.timestamps.size == 366_885
    np.testing.assert_allclose(np.diff(w.timestamps), 0.01, atol=1e-9)
    residual = w.data - np.interp(w.timestamps, raw.timestamps, raw.data)
    # BWM rounds positions to 0.001 rad, so the residual is at most half a step.
    assert np.abs(residual).max() <= 0.0005 + 3e-5


@needs_real
@pytest.mark.parametrize(
    "camera, pairs, half_frame_s",
    [
        (
            "right",
            {"nose_tip": "nose_tip", "paw_r": "right_paw", "tongue_end_l": "left_tongue_end"},
            1 / 300,
        ),
        ("body", {"tail_start": "tail_start"}, 1 / 60),
    ],
)
def test_real_pose_matches_nwb_within_quantisation(sessions, camera, pairs, half_frame_s):
    bwm, nwb = sessions
    b, n = bwm.behaviour[f"pose_{camera}"], nwb.behaviour[f"pose_{camera}"]
    near = _nearest(n.timestamps, b.timestamps)
    assert np.abs(n.timestamps[near] - b.timestamps).max() <= half_frame_s + 1e-4
    for ours, theirs in pairs.items():
        for suffix, tolerance in (("x", 0.25), ("y", 0.25), ("likelihood", 0.5 / 255)):
            bv = b.data[:, b.channel_names.index(f"{ours}_{suffix}")].astype(np.float64)
            nv = n.data[near, n.channel_names.index(f"{theirs}_{suffix}")]
            ok = np.isfinite(bv) & np.isfinite(nv)
            assert np.abs(bv[ok] - nv[ok]).max() <= tolerance + 1e-4, f"{camera} {ours}_{suffix}"


@needs_real
def test_real_left_camera_frames_are_within_half_a_frame(sessions):
    # Left-camera pose values differ from NWB's at the matching frames (see DECISIONS.md);
    # only the frame timing is checked here.
    bwm, nwb = sessions
    b, n = bwm.behaviour["pose_left"], nwb.behaviour["pose_left"]
    near = _nearest(n.timestamps, b.timestamps)
    assert np.abs(n.timestamps[near] - b.timestamps).max() <= 1 / 120 + 1e-4


@needs_real
def test_real_capabilities(sessions):
    bwm, _ = sessions
    present = bwm.available.present
    for key in [
        "wheel",
        "pose_left",
        "pose_right",
        "pose_body",
        "motion_energy_left",
        "motion_energy_body",
        "pupil_left",
    ]:
        assert f"behaviour.{key}" in present
    assert bwm.available.missing["behaviour.pupil_right"]
