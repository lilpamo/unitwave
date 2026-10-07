import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numcodecs import Blosc

from unitwave.data.backends.bwm_compressed import (
    _decode_spike_shard,
    _split_by_unit,
    load_session_bwm,
)
from unitwave.data.backends.dandi_nwb import load_session_nwb
from unitwave.data.session import TRIAL_FIELDS
from unitwave.env import env

EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
BWM_ROOT = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
NWB_PATH = (
    DATA_ROOT
    / "dandi/000409/sub-DY-016"
    / f"sub-DY-016_ses-{EID}_desc-processed_behavior+ecephys.nwb"
)

needs_bwm = pytest.mark.skipif(not BWM_ROOT.exists(), reason=f"BWM data not at {BWM_ROOT}")
needs_nwb = pytest.mark.skipif(not NWB_PATH.exists(), reason=f"NWB file not at {NWB_PATH}")


def _write_shard(path: Path, arrays: dict, **meta_overrides) -> Path:
    path.mkdir(parents=True)
    specs = {}
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        (path / f"{name}.blosc").write_bytes(Blosc(cname="zstd", clevel=7).encode(arr))
        specs[name] = {"entry": f"{name}.blosc", "dtype": arr.dtype.str, "shape": list(arr.shape)}
    meta = {
        "arrays": specs,
        "format": "ibl_agent_spike_shard_v2",
        "time_encoding": "delta_int_ticks",
        "cluster_encoding": "dense_local_indices",
        "time_quantization_us": 100,
        "time_origin_ticks": 0,
    }
    meta.update(meta_overrides)
    (path / "meta.json").write_text(json.dumps(meta))
    return path


def _shard_arrays():
    # Absolute ticks 10, 12, 15, 40 (x 100 us) for units 7, 3, 7, 3.
    return {
        "spike_times_delta_ticks": np.array([10, 2, 3, 25], dtype="<u2"),
        "spike_clusters": np.array([1, 0, 1, 0], dtype="<i4"),
        "cluster_ids": np.array([3, 7], dtype="<i4"),
        "cluster_spike_counts": np.array([2, 2], dtype="<i4"),
    }


def test_decodes_delta_ticks_to_seconds(tmp_path):
    times, ids, local = _decode_spike_shard(_write_shard(tmp_path / "s", _shard_arrays()))
    np.testing.assert_allclose(times, [0.0010, 0.0012, 0.0015, 0.0040])
    np.testing.assert_array_equal(ids, [3, 7])
    np.testing.assert_array_equal(local, [1, 0, 1, 0])


def test_time_origin_ticks_are_added(tmp_path):
    shard = _write_shard(tmp_path / "s", _shard_arrays(), time_origin_ticks=1000)
    times, _, _ = _decode_spike_shard(shard)
    assert times[0] == pytest.approx(0.1010)


def test_spikes_split_by_unit_keep_time_order(tmp_path):
    times, ids, local = _decode_spike_shard(_write_shard(tmp_path / "s", _shard_arrays()))
    per_unit = _split_by_unit(times, local, len(ids))
    np.testing.assert_allclose(per_unit[0], [0.0012, 0.0040])
    np.testing.assert_allclose(per_unit[1], [0.0010, 0.0015])


def test_spike_counts_must_match_cluster_assignments(tmp_path):
    arrays = _shard_arrays()
    arrays["cluster_spike_counts"] = np.array([1, 3], dtype="<i4")
    with pytest.raises(ValueError, match="cluster_spike_counts"):
        _decode_spike_shard(_write_shard(tmp_path / "s", arrays))


def test_unknown_shard_format_rejected(tmp_path):
    shard = _write_shard(tmp_path / "s", _shard_arrays(), time_encoding="float_seconds")
    with pytest.raises(ValueError, match="time_encoding"):
        _decode_spike_shard(shard)


@pytest.fixture(scope="module")
def session():
    return load_session_bwm(EID, BWM_ROOT)


@needs_bwm
def test_eid_and_good_units_only(session):
    assert session.eid == EID
    assert session.n_units == 398
    assert session.units["probe_name"].value_counts().to_dict() == {"probe01": 284, "probe00": 114}
    assert (session.units["label"] == 1.0).all()
    assert (
        session.units.index
        == session.units["probe_name"] + "_" + session.units["cluster_id"].astype(str)
    ).all()


@needs_bwm
def test_spike_totals_per_probe(session):
    probe = session.units["probe_name"]
    totals = {
        p: sum(len(session.spikes[u]) for u in probe.index[probe == p])
        for p in ("probe00", "probe01")
    }
    assert totals == {"probe00": 2_955_937, "probe01": 22_932_105}


@needs_bwm
def test_trials_have_every_canonical_field(session):
    assert session.n_trials == 410
    assert all(f"trials.{f}" in session.available.present for f in TRIAL_FIELDS)
    assert session.trials["bwm_include"].sum() == 290


@needs_bwm
def test_behaviour_declared_missing_with_reason(session):
    assert session.behaviour == {}
    assert session.available.missing["behaviour.wheel"]


def test_wrong_dataset_version_rejected(tmp_path):
    fake = tmp_path / "bwm_ephys"
    fake.mkdir()
    (fake / "manifest.json").write_text(
        json.dumps({"dataset_name": "bwm_ephys", "dataset_version": "9.9.9"})
    )
    with pytest.raises(ValueError, match="9.9.9"):
        load_session_bwm(EID, fake)


@pytest.fixture(scope="module")
def nwb_session():
    return load_session_nwb(NWB_PATH)


@needs_bwm
@needs_nwb
def test_same_units_and_spike_counts_as_nwb(session, nwb_session):
    good = nwb_session.units.index[nwb_session.units["label"] == 1.0]
    assert set(session.units.index) == set(good)
    for unit_id in session.units.index:
        assert len(session.spikes[unit_id]) == len(nwb_session.spikes[unit_id]), unit_id


@needs_bwm
@needs_nwb
def test_spike_times_within_half_a_tick_of_nwb(session, nwb_session):
    # BWM stores spike times rounded to 100 us ticks, so each is within 50 us.
    worst = max(
        float(np.max(np.abs(session.spikes[u] - nwb_session.spikes[u])))
        for u in session.units.index
        if len(session.spikes[u])
    )
    assert worst <= 50e-6 + 1e-9


@needs_bwm
@needs_nwb
def test_trials_match_nwb(session, nwb_session):
    bwm, nwb = session.trials, nwb_session.trials
    for field in TRIAL_FIELDS:
        a, b = bwm[field].to_numpy(float), nwb[field].to_numpy(float)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=field)
        if field == "firstMovement_times":
            # Both sources put onsets on a 1 kHz grid, offset by a constant phase: BWM
            # uses the Brain Wide Map paper's frozen trials table, NWB matches ONE's
            # 2025-03-03 revision. They differ by that phase plus a few whole samples.
            diff = a - b
            assert np.ptp(np.mod(diff, 1e-3)) < 1e-8
            assert np.abs(diff).max() < 2e-3
        else:
            pd.testing.assert_series_equal(bwm[field], nwb[field], check_names=False)
