"""`load_session(eid, backend)`: one entry point over the data backends, via the cache."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from unitwave.data.backends import bwm_compressed, dandi_nwb, one_backend
from unitwave.data.cache import SessionCache
from unitwave.data.session import Session
from unitwave.env import env

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"
_CONFIG_KEYS = {
    "data_root",
    "bwm_ephys",
    "bwm_behavior",
    "nwb_000409",
    "one_cache",
    "cache",
    "derived",
}


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    bwm_ephys_root: Path
    bwm_behavior_root: Path
    nwb_dir: Path
    one_cache_root: Path
    cache_root: Path
    # Tables derived from a release, e.g. qc/task_rates.py's; None when not configured.
    derived_root: Path | None = None


def load_data_config(path: str | os.PathLike = DEFAULT_CONFIG) -> DataConfig:
    """Read configs/data.yaml; UNITWAVE_DATA_ROOT (formerly NEURODECODER_DATA_ROOT, still
    read) overrides its data_root."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _CONFIG_KEYS), sorted(_CONFIG_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    root = Path(env("DATA_ROOT", raw["data_root"])).expanduser()
    return DataConfig(
        data_root=root,
        bwm_ephys_root=root / raw["bwm_ephys"],
        bwm_behavior_root=root / raw["bwm_behavior"],
        nwb_dir=root / raw["nwb_000409"],
        one_cache_root=root / raw["one_cache"],
        cache_root=root / raw["cache"],
        derived_root=root / raw["derived"],
    )


@dataclass(frozen=True)
class Backend:
    """source and loader_version identify what a backend returns, for the cache key."""

    source: dict
    loader_version: int
    load: Callable[[str, DataConfig], Session]


def nwb_source(eid: str, config: DataConfig) -> Path | str:
    """A local copy of the session's DANDI 000409 file if there is one, else its S3 URL."""
    pattern = f"sub-*/sub-*_ses-{eid}_desc-processed_behavior+ecephys.nwb"
    matches = sorted(config.nwb_dir.glob(pattern))
    if len(matches) > 1:
        raise ValueError(f"found {len(matches)} local copies of eid {eid}: {matches}")
    return matches[0] if matches else dandi_nwb.dandi_asset_url(eid)


def _load_bwm(eid: str, config: DataConfig) -> Session:
    return bwm_compressed.load_session_bwm(
        eid, config.bwm_ephys_root, behaviour_root=config.bwm_behavior_root
    )


def _load_nwb(eid: str, config: DataConfig) -> Session:
    # Local copies and streamed files share a cache key: both are the pinned version's
    # asset (downloads are SHA-256 checked, and streamed == local is tested).
    return dandi_nwb.load_session_nwb(nwb_source(eid, config))


def _load_one(eid: str, config: DataConfig) -> Session:
    return one_backend.load_session_one(eid, one_backend.make_one(config.one_cache_root))


BACKENDS = {
    "bwm": Backend(
        source={
            "dataset": bwm_compressed.DATASET_NAME,
            "version": bwm_compressed.DATASET_VERSION,
            "behaviour_version": bwm_compressed.BEHAVIOUR_DATASET_VERSION,
        },
        loader_version=bwm_compressed.LOADER_VERSION,
        load=_load_bwm,
    ),
    "nwb": Backend(
        source={"dandiset": dandi_nwb.DANDISET, "version": dandi_nwb.DANDISET_VERSION},
        loader_version=dandi_nwb.LOADER_VERSION,
        load=_load_nwb,
    ),
    "one": Backend(
        source={
            "database": one_backend.PUBLIC_ALYX,
            "sorter_revision": one_backend.SORTER_REVISION,
            "trials_revision": one_backend.TRIALS_REVISION,
        },
        loader_version=one_backend.LOADER_VERSION,
        load=_load_one,
    ),
}


def _backend(name: str) -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[name]


def key_parts(eid: str, backend: str) -> dict:
    spec = _backend(backend)
    return {
        "eid": eid,
        "backend": backend,
        "source": dict(spec.source),
        "loader_version": spec.loader_version,
    }


def load_session(
    eid: str,
    backend: str = "bwm",
    *,
    config: DataConfig | None = None,
    use_cache: bool = True,
) -> Session:
    """Load one session from a backend ("bwm" or "nwb"), served from the cache when present."""
    config = config or load_data_config()
    spec = _backend(backend)

    def loader() -> Session:
        return spec.load(eid, config)

    if not use_cache:
        return loader()
    return SessionCache(config.cache_root).get_or_load(key_parts(eid, backend), loader)
