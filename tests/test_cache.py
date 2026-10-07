from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unitwave.data import cache as cache_module
from unitwave.data.cache import SessionCache, cache_key
from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
    TimeSeries,
)
from unitwave.env import env

KEY_PARTS = {
    "eid": "fixture-eid",
    "backend": "fixture",
    "source": {"dataset": "fixture", "version": "1"},
    "loader_version": 1,
}


def _session():
    units = pd.DataFrame(
        {
            "probe_name": ["probe00", "probe01", "probe01"],
            "acronym": ["CA1", "PO", "LP"],
            "x": [0.001, 0.002, np.nan],
            "y": [0.0, 0.1, 0.2],
            "z": [0.0, 0.0, 0.0],
            "depths": [20.0, 40.0, 60.0],
            "label": [1.0, 1.0, 1.0],
            "firing_rate": np.array([5.5, 7.25, 0.0], dtype=np.float32),
            "cluster_uuid": ["a", "b", "c"],
        },
        index=pd.Index(["probe00_0", "probe01_3", "probe01_9"], name="unit_id"),
    )
    trials = pd.DataFrame({f: [1.0, 2.0] for f in TRIAL_FIELDS})
    trials["stimOff_times"] = [1.5, np.nan]
    trials["bwm_include"] = [True, False]
    pose = TimeSeries(
        np.array([0.0, 0.5, 1.0]),
        np.array([[1.0, 2.0, 0.9], [np.nan, 2.5, 0.1], [1.5, 3.0, 0.8]]),
        channel_names=("nose_tip_x", "nose_tip_y", "nose_tip_likelihood"),
    )
    behaviour = {
        "wheel": TimeSeries(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.25, np.nan])),
        "pose_left": pose,
    }
    present = {f"trials.{f}" for f in TRIAL_FIELDS} | {f"units.{f}" for f in UNIT_FIELDS}
    present |= {"behaviour.wheel", "behaviour.pose_left"}
    missing = {f"behaviour.{f}": "not in fixture" for f in BEHAVIOUR_FIELDS}
    del missing["behaviour.wheel"], missing["behaviour.pose_left"]
    return Session(
        eid="fixture-eid",
        time_bounds=(0.0, 10.0),
        spikes={
            "probe00_0": np.array([0.1, 0.2, 3.0]),
            "probe01_3": np.array([], dtype=np.float64),
            "probe01_9": np.array([9.5]),
        },
        units=units,
        trials=trials,
        behaviour=behaviour,
        available=Capabilities(present=frozenset(present), missing=missing),
    )


def assert_sessions_identical(a: Session, b: Session):
    assert a.eid == b.eid
    assert a.time_bounds == b.time_bounds
    assert list(a.spikes) == list(b.spikes)
    for unit_id, times in a.spikes.items():
        assert b.spikes[unit_id].dtype == times.dtype
        np.testing.assert_array_equal(b.spikes[unit_id], times)
    pd.testing.assert_frame_equal(a.units, b.units)
    pd.testing.assert_frame_equal(a.trials, b.trials)
    assert list(a.behaviour) == list(b.behaviour)
    for name, series in a.behaviour.items():
        other = b.behaviour[name]
        assert other.channel_names == series.channel_names
        assert other.data.dtype == series.data.dtype
        np.testing.assert_array_equal(other.timestamps, series.timestamps)
        np.testing.assert_array_equal(other.data, series.data)
    assert a.available.present == b.available.present
    assert dict(a.available.missing) == dict(b.available.missing)


def test_key_ignores_dict_order_but_not_content():
    reordered = dict(reversed(list(KEY_PARTS.items())))
    assert cache_key(KEY_PARTS) == cache_key(reordered)
    assert cache_key(KEY_PARTS) != cache_key({**KEY_PARTS, "eid": "other"})
    assert cache_key(KEY_PARTS) != cache_key({**KEY_PARTS, "loader_version": 2})
    assert cache_key(KEY_PARTS) != cache_key(
        {**KEY_PARTS, "source": {"dataset": "fixture", "version": "2"}}
    )


def test_key_requires_the_identifying_parts():
    with pytest.raises(ValueError, match="loader_version"):
        cache_key({k: v for k, v in KEY_PARTS.items() if k != "loader_version"})


def test_key_includes_the_cache_format_version(monkeypatch):
    before = cache_key(KEY_PARTS)
    monkeypatch.setattr(cache_module, "CACHE_FORMAT_VERSION", cache_module.CACHE_FORMAT_VERSION + 1)
    assert cache_key(KEY_PARTS) != before


def test_round_trip_is_identical(tmp_path):
    cache = SessionCache(tmp_path)
    original = _session()
    cache.put(KEY_PARTS, original)
    assert_sessions_identical(original, cache.get(KEY_PARTS))


def test_miss_returns_none(tmp_path):
    assert SessionCache(tmp_path).get(KEY_PARTS) is None


def test_get_or_load_calls_the_loader_once(tmp_path):
    cache = SessionCache(tmp_path)
    calls = []

    def loader():
        calls.append(1)
        return _session()

    first = cache.get_or_load(KEY_PARTS, loader)
    second = cache.get_or_load(KEY_PARTS, loader)
    assert len(calls) == 1
    assert_sessions_identical(first, second)


def test_loader_must_return_the_keyed_session(tmp_path):
    with pytest.raises(ValueError, match="eid"):
        SessionCache(tmp_path).get_or_load({**KEY_PARTS, "eid": "other-eid"}, _session)


def test_failed_write_leaves_no_entry(tmp_path, monkeypatch):
    cache = SessionCache(tmp_path)
    real_save = np.save
    calls = []

    def flaky_save(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(np, "save", flaky_save)
    with pytest.raises(OSError):
        cache.put(KEY_PARTS, _session())
    monkeypatch.undo()
    assert cache.get(KEY_PARTS) is None
    assert not any(p.name.startswith(".tmp") for p in tmp_path.rglob("*"))


def test_entry_without_meta_is_a_miss(tmp_path):
    cache = SessionCache(tmp_path)
    cache.put(KEY_PARTS, _session())
    (cache.path(KEY_PARTS) / "meta.json").unlink()
    assert cache.get(KEY_PARTS) is None


def test_entry_records_what_it_holds(tmp_path):
    cache = SessionCache(tmp_path)
    cache.put(KEY_PARTS, _session())
    assert cache.describe(KEY_PARTS)["key_parts"] == KEY_PARTS


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
BWM_ROOT = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
NWB_PATH = (
    DATA_ROOT
    / "dandi/000409/sub-DY-016"
    / f"sub-DY-016_ses-{EID}_desc-processed_behavior+ecephys.nwb"
)


@pytest.mark.skipif(not BWM_ROOT.exists(), reason=f"BWM data not at {BWM_ROOT}")
def test_real_bwm_session_round_trips(tmp_path):
    from unitwave.data.backends.bwm_compressed import load_session_bwm

    original = load_session_bwm(EID, BWM_ROOT)
    cache = SessionCache(tmp_path)
    parts = {**KEY_PARTS, "eid": EID, "backend": "bwm_compressed"}
    cache.put(parts, original)
    assert_sessions_identical(original, cache.get(parts))


@pytest.mark.skipif(not NWB_PATH.exists(), reason=f"NWB file not at {NWB_PATH}")
def test_real_nwb_session_round_trips(tmp_path):
    from unitwave.data.backends.dandi_nwb import load_session_nwb

    original = load_session_nwb(NWB_PATH)
    cache = SessionCache(tmp_path)
    parts = {**KEY_PARTS, "eid": EID, "backend": "dandi_nwb"}
    cache.put(parts, original)
    assert_sessions_identical(original, cache.get(parts))
