from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from unitwave.data.backends import dandi_nwb
from unitwave.data.backends.dandi_nwb import (
    _is_url,
    _keypoint_name,
    _map_choice,
    _map_feedback,
    _pose,
    _split_contrast,
    dandi_asset_url,
    load_session_nwb,
)
from unitwave.env import env

EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
NWB_PATH = (
    DATA_ROOT
    / "dandi/000409/sub-DY-016"
    / f"sub-DY-016_ses-{EID}_desc-processed_behavior+ecephys.nwb"
)

needs_nwb = pytest.mark.skipif(not NWB_PATH.exists(), reason=f"real NWB file not at {NWB_PATH}")
needs_network = pytest.mark.skipif(
    env("NETWORK_TESTS") != "1",
    reason="set UNITWAVE_NETWORK_TESTS=1 to stream from DANDI",
)
# Same blob (same SHA-256) as the local file, in DANDI 000409 version 0.260309.1324.
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/d0f/ba3/d0fba38f-324e-4bec-b111-95365c2617b1"


def test_choice_mapping():
    got = _map_choice(pd.Series(["clockwise", "counter_clockwise", "clockwise"]))
    np.testing.assert_array_equal(got, [1.0, -1.0, 1.0])


def test_choice_mapping_rejects_unknown_values():
    with pytest.raises(ValueError, match="no_go"):
        _map_choice(pd.Series(["clockwise", "no_go"]))


def test_feedback_mapping():
    np.testing.assert_array_equal(_map_feedback(pd.Series([True, False])), [1.0, -1.0])


def test_feedback_mapping_rejects_non_bool():
    with pytest.raises(ValueError, match="bool"):
        _map_feedback(pd.Series([1.0, np.nan]))


def test_contrast_split():
    left, right = _split_contrast(pd.Series(["left", "right"]), pd.Series([6.25, 100.0]))
    np.testing.assert_array_equal(left, [0.0625, np.nan])
    np.testing.assert_array_equal(right, [np.nan, 1.0])


def test_contrast_split_rejects_unknown_side():
    with pytest.raises(ValueError, match="side"):
        _split_contrast(pd.Series(["centre"]), pd.Series([25.0]))


def test_contrast_split_rejects_fractions():
    # NWB stores percent; a 0-1 value here means the source changed convention.
    with pytest.raises(ValueError, match="percent"):
        _split_contrast(pd.Series(["left", "right"]), pd.Series([0.25, 0.5]))


def test_urls_are_streamed_and_paths_are_local():
    assert _is_url(S3_URL)
    assert not _is_url(str(NWB_PATH))
    assert not _is_url(NWB_PATH)


def _fake_dandi_api(asset_results, content_urls, seen):
    def get_json(url):
        seen.append(url)
        if "/assets/?" in url:
            return {"count": len(asset_results), "results": asset_results}
        return {"contentUrl": content_urls}

    return get_json


def test_resolver_returns_the_s3_url_from_the_pinned_version(monkeypatch):
    seen = []
    urls = ["https://api.dandiarchive.org/api/assets/a1/download/", S3_URL]
    monkeypatch.setattr(dandi_nwb, "_get_json", _fake_dandi_api([{"asset_id": "a1"}], urls, seen))
    assert dandi_asset_url(EID) == S3_URL
    assert all(f"/versions/{dandi_nwb.DANDISET_VERSION}/" in u for u in seen)
    assert not any("/draft/" in u for u in seen)


def test_resolver_rejects_zero_or_many_matches(monkeypatch):
    for results in ([], [{"asset_id": "a1"}, {"asset_id": "a2"}]):
        monkeypatch.setattr(dandi_nwb, "_get_json", _fake_dandi_api(results, [S3_URL], []))
        with pytest.raises(ValueError, match="expected 1"):
            dandi_asset_url(EID)


def test_keypoint_names_are_snake_case():
    assert _keypoint_name("PoseEstimationSeriesRightPupilBottom") == "right_pupil_bottom"
    assert _keypoint_name("PoseEstimationSeriesTailStart") == "tail_start"


def _keypoint(timestamps, n):
    return SimpleNamespace(
        timestamps=np.asarray(timestamps, dtype=float),
        data=np.arange(2 * n, dtype=float).reshape(n, 2),
        confidence=np.full(n, 0.9),
    )


def test_pose_stacks_keypoints_with_named_columns():
    series = {
        "PoseEstimationSeriesNoseTip": _keypoint([0.0, 1.0], 2),
        "PoseEstimationSeriesLeftPaw": _keypoint([0.0, 1.0], 2),
    }
    ts = _pose(SimpleNamespace(name="LeftCamera", pose_estimation_series=series))
    assert ts.channel_names == (
        "left_paw_x",
        "left_paw_y",
        "left_paw_likelihood",
        "nose_tip_x",
        "nose_tip_y",
        "nose_tip_likelihood",
    )
    assert ts.data.shape == (2, 6)


def test_pose_rejects_keypoints_on_different_clocks():
    series = {
        "PoseEstimationSeriesNoseTip": _keypoint([0.0, 1.0], 2),
        "PoseEstimationSeriesLeftPaw": _keypoint([0.0, 2.0], 2),
    }
    with pytest.raises(ValueError, match="timestamps"):
        _pose(SimpleNamespace(name="LeftCamera", pose_estimation_series=series))


@pytest.fixture(scope="module")
def session():
    return load_session_nwb(NWB_PATH)


# Expected counts below were checked to be identical to ONE for this session
# (spike sorting revision 2024-05-06, trials revision 2025-03-03).


@needs_nwb
def test_eid_read_from_file(session):
    assert session.eid == EID


@needs_nwb
def test_all_units_and_spikes(session):
    assert session.n_units == 1961
    assert session.units["probe_name"].value_counts().to_dict() == {"probe01": 1287, "probe00": 674}
    assert sum(len(t) for t in session.spikes.values()) == 61_981_600


@needs_nwb
def test_unit_ids_encode_probe_and_cluster(session):
    u = session.units
    assert u.index[0] == "probe00_0"
    assert (u.index == u["probe_name"] + "_" + u["cluster_id"].astype(str)).all()
    assert u["cluster_uuid"].is_unique


@needs_nwb
def test_unit_quality_fields(session):
    u = session.units
    assert set(np.round(u["label"].unique(), 4)) <= {0.0, 0.3333, 0.6667, 1.0}
    assert int((u["label"] == 1.0).sum()) == 398
    assert u["depths"].notna().all()
    assert u["firing_rate"].notna().all()


@needs_nwb
def test_trials_use_canonical_encodings(session):
    t = session.trials
    assert len(t) == 410
    assert t["choice"].value_counts().to_dict() == {-1.0: 292, 1.0: 118}
    assert t["feedbackType"].value_counts().to_dict() == {1.0: 304, -1.0: 106}
    assert (t["contrastLeft"].isna() ^ t["contrastRight"].isna()).all()
    shown = pd.concat([t["contrastLeft"], t["contrastRight"]]).dropna()
    assert set(shown.unique()) <= {0.0, 0.0625, 0.125, 0.25, 1.0}
    assert set(t["probabilityLeft"].unique()) == {0.2, 0.5, 0.8}


@needs_nwb
def test_wheel_is_raw_position(session):
    wheel = session.behaviour["wheel"]
    assert wheel.timestamps.shape == (755_552,)
    assert wheel.data.shape == (755_552,)


@needs_nwb
def test_time_bounds_span_all_loaded_data(session):
    t0, t1 = session.time_bounds
    assert t0 == pytest.approx(0.0008, abs=1e-3)
    assert t1 == pytest.approx(3668.940, abs=1e-3)


@needs_nwb
def test_motion_energy_has_one_series_per_camera(session):
    sizes = {
        cam: session.behaviour[f"motion_energy_{cam}"].timestamps.size
        for cam in ("body", "left", "right")
    }
    assert sizes == {"body": 109_866, "left": 218_896, "right": 548_105}


@needs_nwb
def test_pupil_is_the_raw_series(session):
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(NWB_PATH), "r", load_namespaces=True) as io:
        raw = io.read().processing["pupil"]["LeftPupilDiameter"].data[:]
    np.testing.assert_array_equal(session.behaviour["pupil_left"].data, raw)
    assert "pupil_body" not in session.behaviour


@needs_nwb
def test_pose_columns_are_named_per_keypoint(session):
    body = session.behaviour["pose_body"]
    assert body.channel_names == ("tail_start_x", "tail_start_y", "tail_start_likelihood")
    assert body.data.shape == (109_866, 3)
    assert session.behaviour["pose_left"].data.shape == (218_896, 18)
    right = session.behaviour["pose_right"]
    assert right.data.shape == (548_105, 33)
    assert "nose_tip_x" in right.channel_names


@needs_nwb
def test_camera_signals_share_their_camera_clock(session):
    b = session.behaviour
    for cam in ("left", "right"):
        clock = b[f"motion_energy_{cam}"].timestamps
        np.testing.assert_array_equal(b[f"pose_{cam}"].timestamps, clock)
        np.testing.assert_array_equal(b[f"pupil_{cam}"].timestamps, clock)


@needs_network
def test_resolver_finds_the_verified_asset():
    assert dandi_asset_url(EID) == S3_URL


@needs_network
@needs_nwb
def test_streamed_load_equals_local_load(session):
    remote = load_session_nwb(dandi_asset_url(EID))
    assert remote.eid == session.eid
    assert remote.time_bounds == session.time_bounds
    pd.testing.assert_frame_equal(remote.units, session.units)
    pd.testing.assert_frame_equal(remote.trials, session.trials)
    assert remote.spikes.keys() == session.spikes.keys()
    for unit_id, times in session.spikes.items():
        np.testing.assert_array_equal(remote.spikes[unit_id], times)
    assert remote.behaviour.keys() == session.behaviour.keys()
    for name, series in session.behaviour.items():
        np.testing.assert_array_equal(remote.behaviour[name].timestamps, series.timestamps)
        np.testing.assert_array_equal(remote.behaviour[name].data, series.data)
    assert remote.available.present == session.available.present
    assert dict(remote.available.missing) == dict(session.available.missing)


@needs_nwb
def test_capabilities_are_explicit(session):
    caps = session.available
    for field in [
        "trials.stimOn_times",
        "trials.choice",
        "units.label",
        "units.depths",
        "behaviour.wheel",
        "behaviour.pose_left",
        "behaviour.motion_energy_body",
        "behaviour.pupil_right",
    ]:
        assert field in caps.present
    for field in ["units.acronym", "units.x", "behaviour.lick"]:
        assert caps.missing[field]
