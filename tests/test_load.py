import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unitwave.data import load as load_module
from unitwave.data.backends import bwm_compressed, dandi_nwb
from unitwave.data.load import (
    Backend,
    DataConfig,
    key_parts,
    load_data_config,
    load_session,
    nwb_source,
)
from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
)
from unitwave.env import env

EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
NWB_NAME = f"sub-DY-016_ses-{EID}_desc-processed_behavior+ecephys.nwb"


def _config(tmp_path: Path, data_root: Path | None = None) -> DataConfig:
    root = data_root or tmp_path
    return DataConfig(
        data_root=root,
        bwm_ephys_root=root / "bwm_compressed/bwm_ephys/1.2.1",
        bwm_behavior_root=root / "bwm_compressed/bwm_behavior/2.0.0",
        nwb_dir=root / "dandi/000409",
        one_cache_root=root / "one",
        cache_root=tmp_path / "cache",
    )


def _tiny_session(eid: str) -> Session:
    units = pd.DataFrame({f: [1.0] for f in UNIT_FIELDS}, index=pd.Index(["p_0"], name="unit_id"))
    present = {f"trials.{f}" for f in TRIAL_FIELDS} | {f"units.{f}" for f in UNIT_FIELDS}
    return Session(
        eid=eid,
        time_bounds=(0.0, 10.0),
        spikes={"p_0": np.array([1.0, 2.0])},
        units=units,
        trials=pd.DataFrame({f: [1.0] for f in TRIAL_FIELDS}),
        behaviour={},
        available=Capabilities(
            present=frozenset(present),
            missing={f"behaviour.{f}": "not in fixture" for f in BEHAVIOUR_FIELDS},
        ),
    )


def test_default_config_resolves_paths_under_the_data_root(monkeypatch):
    monkeypatch.delenv("UNITWAVE_DATA_ROOT", raising=False)
    monkeypatch.delenv("NEURODECODER_DATA_ROOT", raising=False)
    cfg = load_data_config()
    assert cfg.data_root == Path("~/data/neurodecoder").expanduser()
    assert cfg.bwm_ephys_root == cfg.data_root / "bwm_compressed/bwm_ephys/1.2.1"
    assert cfg.bwm_behavior_root == cfg.data_root / "bwm_compressed/bwm_behavior/2.0.0"
    assert cfg.one_cache_root == cfg.data_root / "one"
    assert cfg.nwb_dir == cfg.data_root / "dandi/000409"
    assert cfg.cache_root == cfg.data_root / "cache"


def test_env_var_overrides_the_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITWAVE_DATA_ROOT", str(tmp_path))
    cfg = load_data_config()
    assert cfg.data_root == tmp_path
    assert cfg.cache_root == tmp_path / "cache"


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("data_root: /x\nbwm_ephys: a\nnwb_000409: b\ncache: c\ntypo_key: d\n")
    with pytest.raises(ValueError, match="typo_key"):
        load_data_config(path)


def test_unknown_backend_lists_the_available_ones(tmp_path):
    with pytest.raises(ValueError, match="bwm.*nwb.*one"):
        load_session(EID, "spikeglx", config=_config(tmp_path))


def test_key_parts_pin_source_and_loader_versions():
    assert key_parts(EID, "bwm") == {
        "eid": EID,
        "backend": "bwm",
        "source": {"dataset": "bwm_ephys", "version": "1.2.1", "behaviour_version": "2.0.0"},
        "loader_version": bwm_compressed.LOADER_VERSION,
    }
    assert key_parts(EID, "nwb")["source"] == {"dandiset": "000409", "version": "0.260309.1324"}
    assert key_parts(EID, "nwb")["loader_version"] == dandi_nwb.LOADER_VERSION
    one_source = key_parts(EID, "one")["source"]
    assert one_source["sorter_revision"] == "2024-05-06"
    assert one_source["trials_revision"] == "2025-03-03"


def test_second_call_is_served_from_the_cache(monkeypatch, tmp_path):
    calls = []

    def fake_load(eid, config):
        calls.append(eid)
        return _tiny_session(eid)

    monkeypatch.setitem(load_module.BACKENDS, "fake", Backend({"dataset": "fake"}, 1, fake_load))
    cfg = _config(tmp_path)
    first = load_session(EID, "fake", config=cfg)
    second = load_session(EID, "fake", config=cfg)
    assert calls == [EID]
    np.testing.assert_array_equal(first.spikes["p_0"], second.spikes["p_0"])

    load_session(EID, "fake", config=cfg, use_cache=False)
    assert calls == [EID, EID]


def test_nwb_source_prefers_a_local_copy(tmp_path):
    cfg = _config(tmp_path)
    local = cfg.nwb_dir / "sub-DY-016" / NWB_NAME
    local.parent.mkdir(parents=True)
    local.touch()
    assert nwb_source(EID, cfg) == local


def test_nwb_source_streams_when_there_is_no_local_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(load_module.dandi_nwb, "dandi_asset_url", lambda eid: f"https://s3/{eid}")
    assert nwb_source(EID, _config(tmp_path)) == f"https://s3/{EID}"


def test_nwb_source_rejects_ambiguous_local_copies(tmp_path):
    cfg = _config(tmp_path)
    for subject in ("sub-A", "sub-B"):
        path = cfg.nwb_dir / subject / f"{subject}_ses-{EID}_desc-processed_behavior+ecephys.nwb"
        path.parent.mkdir(parents=True)
        path.touch()
    with pytest.raises(ValueError, match="2 local"):
        nwb_source(EID, cfg)


REAL_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
HAS_BWM = (REAL_ROOT / "bwm_compressed/bwm_ephys/1.2.1").exists()
HAS_NWB = (REAL_ROOT / "dandi/000409/sub-DY-016" / NWB_NAME).exists()


def _assert_same(a: Session, b: Session) -> None:
    assert a.eid == b.eid and a.time_bounds == b.time_bounds
    pd.testing.assert_frame_equal(a.units, b.units)
    pd.testing.assert_frame_equal(a.trials, b.trials)
    assert list(a.spikes) == list(b.spikes)
    for unit_id, times in a.spikes.items():
        np.testing.assert_array_equal(times, b.spikes[unit_id])
    assert list(a.behaviour) == list(b.behaviour)
    for name, series in a.behaviour.items():
        np.testing.assert_array_equal(series.data, b.behaviour[name].data)
    assert a.available.present == b.available.present


@pytest.mark.skipif(not HAS_BWM, reason="BWM data not available")
def test_bwm_through_the_entry_point_equals_the_backend(monkeypatch, tmp_path):
    cfg = _config(tmp_path, data_root=REAL_ROOT)
    direct = bwm_compressed.load_session_bwm(
        EID, cfg.bwm_ephys_root, behaviour_root=cfg.bwm_behavior_root
    )
    calls = []
    original = load_module.BACKENDS["bwm"]

    def counting_load(eid, config):
        calls.append(eid)
        return original.load(eid, config)

    monkeypatch.setitem(
        load_module.BACKENDS, "bwm", dataclasses.replace(original, load=counting_load)
    )
    _assert_same(direct, load_session(EID, "bwm", config=cfg))
    _assert_same(direct, load_session(EID, "bwm", config=cfg))
    assert calls == [EID]


@pytest.mark.skipif(not HAS_NWB, reason="local NWB file not available")
def test_nwb_through_the_entry_point_equals_the_backend(tmp_path):
    cfg = _config(tmp_path, data_root=REAL_ROOT)
    direct = dandi_nwb.load_session_nwb(cfg.nwb_dir / "sub-DY-016" / NWB_NAME)
    _assert_same(direct, load_session(EID, "nwb", config=cfg))
