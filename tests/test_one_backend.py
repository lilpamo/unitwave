from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unitwave.data.backends.one_backend import (
    SORTER_REVISION,
    TRIALS_REVISION,
    _probe_collections,
    load_session_one,
)
from unitwave.data.session import TRIAL_FIELDS
from unitwave.env import env

EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"


class FakeAlf(dict):
    """ONE returns an AlfBunch: a dict whose keys are also attributes."""

    __getattr__ = dict.__getitem__


class FakeOne:
    """Records the (object, collection, revision) asked for, and serves canned objects."""

    def __init__(self, probes=("probe00", "probe01"), objects=None, missing=()):
        self.probes = probes
        self.objects = objects or {}
        self.missing = set(missing)
        self.asked = []

    def list_datasets(self, eid, filename=None, collection=None, **kwargs):
        if collection and "probe" in collection:
            return [f"alf/{p}/pykilosort/#{SORTER_REVISION}#/spikes.times.npy" for p in self.probes]
        return []

    def list_collections(self, eid, collection=None, **kwargs):
        return [f"alf/{p}/pykilosort" for p in self.probes]

    def load_object(self, eid, obj, collection=None, revision=None, **kwargs):
        self.asked.append((obj, collection, revision))
        if obj in self.missing:
            raise FileNotFoundError(f"no {obj}")
        probe = None if collection is None else collection.split("/")[1]
        return self.objects[(obj, probe)]


def _probe_objects(probe: str, n_units: int, n_spikes: int, first_cluster: int = 0):
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0.5, 9.5, n_spikes))
    clusters = np.arange(n_spikes) % n_units
    cluster_ids = np.arange(first_cluster, first_cluster + n_units)
    return {
        ("spikes", probe): FakeAlf(times=times, clusters=clusters),
        ("clusters", probe): FakeAlf(
            channels=np.arange(n_units),
            depths=np.linspace(20.0, 100.0, n_units),
            metrics=pd.DataFrame(
                {
                    "cluster_id": cluster_ids,
                    "label": [1.0] * n_units,
                    "firing_rate": np.linspace(1.0, 5.0, n_units),
                }
            ),
            uuids=pd.Series([f"{probe}-uuid-{i}" for i in range(n_units)]),
        ),
        ("channels", probe): FakeAlf(
            mlapdv=np.tile(np.array([[100.0, -2000.0, -3000.0]]), (n_units, 1)),
            brainLocationIds_ccf_2017=np.full(n_units, 697),
        ),
    }


def _trials(n=3):
    return FakeAlf(
        intervals=np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float) + 0.9]),
        stimOn_times=np.arange(n, dtype=float) + 0.1,
        stimOff_times=np.arange(n, dtype=float) + 0.8,
        goCue_times=np.arange(n, dtype=float) + 0.15,
        firstMovement_times=np.arange(n, dtype=float) + 0.3,
        response_times=np.arange(n, dtype=float) + 0.5,
        feedback_times=np.arange(n, dtype=float) + 0.55,
        choice=np.array([-1.0, 1.0, -1.0][:n]),
        feedbackType=np.array([1.0, -1.0, 1.0][:n]),
        contrastLeft=np.array([0.25, np.nan, 1.0][:n]),
        contrastRight=np.array([np.nan, 0.5, np.nan][:n]),
        probabilityLeft=np.array([0.8, 0.8, 0.2][:n]),
        rewardVolume=np.array([1.5, 0.0, 1.5][:n]),
    )


def _fake_one(**overrides):
    objects = {
        **_probe_objects("probe00", 3, 30),
        **_probe_objects("probe01", 2, 20, first_cluster=0),
    }
    objects[("trials", None)] = _trials()
    objects[("wheel", None)] = FakeAlf(
        timestamps=np.array([0.0, 1.0, 2.0]), position=np.array([0.0, 0.1, 0.2])
    )
    objects.update(overrides.pop("objects", {}))
    return FakeOne(objects=objects, **overrides)


def test_probe_collections_are_found_and_sorted():
    assert _probe_collections(FakeOne(probes=("probe01", "probe00")), EID) == [
        ("probe00", "alf/probe00/pykilosort"),
        ("probe01", "alf/probe01/pykilosort"),
    ]


def test_no_probes_raises():
    with pytest.raises(ValueError, match="no spike-sorted probes"):
        _probe_collections(FakeOne(probes=()), EID)


def test_revisions_are_pinned_on_every_request():
    one = _fake_one()
    load_session_one(EID, one)
    spikes = [a for a in one.asked if a[0] in ("spikes", "clusters", "channels")]
    assert spikes and all(rev == SORTER_REVISION for _, _, rev in spikes)
    assert ("trials", None, TRIALS_REVISION) in one.asked


def test_units_are_keyed_by_probe_and_cluster():
    s = load_session_one(EID, _fake_one())
    assert s.n_units == 5
    assert list(s.units.index[:2]) == ["probe00_0", "probe00_1"]
    assert s.units["probe_name"].value_counts().to_dict() == {"probe00": 3, "probe01": 2}
    assert s.units["cluster_uuid"].is_unique


def test_unit_coordinates_are_metres_from_channel_mlapdv():
    s = load_session_one(EID, _fake_one())
    np.testing.assert_allclose(
        s.units.loc["probe00_0", ["x", "y", "z"]].to_numpy(float), [100e-6, -2000e-6, -3000e-6]
    )


def test_spikes_are_split_per_unit_in_time_order():
    s = load_session_one(EID, _fake_one())
    assert sum(len(t) for t in s.spikes.values()) == 50
    for times in s.spikes.values():
        assert (np.diff(times) >= 0).all()


def test_trials_use_canonical_names():
    s = load_session_one(EID, _fake_one())
    assert all(f in s.trials for f in TRIAL_FIELDS)
    np.testing.assert_allclose(s.trials["intervals_0"].to_numpy(), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(s.trials["intervals_1"].to_numpy(), [0.9, 1.9, 2.9])


def test_wheel_is_raw_position(session=None):
    s = load_session_one(EID, _fake_one())
    assert s.behaviour["wheel"].data.shape == (3,)
    assert s.available.present >= {"behaviour.wheel"}


def test_missing_wheel_is_declared_not_fatal():
    s = load_session_one(EID, _fake_one(missing={"wheel"}))
    assert "wheel" not in s.behaviour
    assert s.available.missing["behaviour.wheel"]


def test_missing_trials_raises():
    with pytest.raises(FileNotFoundError):
        load_session_one(EID, _fake_one(missing={"trials"}))


def test_spike_cluster_index_out_of_range_raises():
    objects = {("spikes", "probe00"): FakeAlf(times=np.array([1.0]), clusters=np.array([99]))}
    with pytest.raises(ValueError, match="cluster"):
        load_session_one(EID, _fake_one(objects=objects))


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
ONE_CACHE = DATA_ROOT / "one"
NWB_PATH = (
    DATA_ROOT
    / "dandi/000409/sub-DY-016"
    / f"sub-DY-016_ses-{EID}_desc-processed_behavior+ecephys.nwb"
)
needs_one_cache = pytest.mark.skipif(
    not (ONE_CACHE / "danlab").exists(), reason=f"ONE cache with the session not at {ONE_CACHE}"
)
needs_nwb = pytest.mark.skipif(not NWB_PATH.exists(), reason=f"NWB file not at {NWB_PATH}")


@pytest.fixture(scope="module")
def real_one_session():
    from unitwave.data.backends.one_backend import make_one

    return load_session_one(EID, make_one(ONE_CACHE))


@pytest.fixture(scope="module")
def nwb_session():
    from unitwave.data.backends.dandi_nwb import load_session_nwb

    return load_session_nwb(NWB_PATH)


@needs_one_cache
def test_real_counts(real_one_session):
    s = real_one_session
    assert s.eid == EID
    assert s.n_units == 1961
    assert s.units["probe_name"].value_counts().to_dict() == {"probe01": 1287, "probe00": 674}
    assert sum(len(t) for t in s.spikes.values()) == 61_981_600
    assert s.n_trials == 410


@needs_one_cache
@needs_nwb
def test_real_spike_times_match_nwb_exactly(real_one_session, nwb_session):
    assert set(real_one_session.spikes) == set(nwb_session.spikes)
    for unit_id, times in real_one_session.spikes.items():
        np.testing.assert_array_equal(times, nwb_session.spikes[unit_id], err_msg=unit_id)


@needs_one_cache
@needs_nwb
def test_real_trials_match_nwb_exactly(real_one_session, nwb_session):
    for field in TRIAL_FIELDS:
        pd.testing.assert_series_equal(
            real_one_session.trials[field], nwb_session.trials[field], check_names=False
        )


@needs_one_cache
@needs_nwb
def test_real_unit_metadata_agrees_with_nwb(real_one_session, nwb_session):
    one_u, nwb_u = real_one_session.units, nwb_session.units
    pd.testing.assert_series_equal(one_u["cluster_uuid"], nwb_u["cluster_uuid"], check_names=False)
    pd.testing.assert_series_equal(one_u["label"], nwb_u["label"], check_names=False)
    np.testing.assert_allclose(one_u["depths"], nwb_u["depths"])


@needs_one_cache
@needs_nwb
def test_real_wheel_matches_nwb(real_one_session, nwb_session):
    one_w, nwb_w = real_one_session.behaviour["wheel"], nwb_session.behaviour["wheel"]
    np.testing.assert_array_equal(one_w.timestamps, nwb_w.timestamps)
    np.testing.assert_array_equal(one_w.data, nwb_w.data)
