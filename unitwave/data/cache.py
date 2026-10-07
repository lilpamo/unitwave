"""Content-addressed on-disk cache of loaded Sessions.

An entry's key is a hash of everything that determines its contents: the eid, the
backend, the source data version, the loader version and this cache's own format
version. Entries are written to a temporary directory and renamed into place, so a
crashed write never leaves a half-written entry that looks valid. Arrays are .npy
(no pickling), tables are parquet, and meta.json records what the entry holds.
"""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from unitwave.data.session import Capabilities, Session, TimeSeries

# Bump when the on-disk layout changes; every existing entry then becomes a miss.
CACHE_FORMAT_VERSION = 1
REQUIRED_KEY_PARTS = ("eid", "backend", "source", "loader_version")


def cache_key(key_parts: Mapping) -> str:
    """sha256 of the key parts plus CACHE_FORMAT_VERSION; independent of dict order."""
    missing = [k for k in REQUIRED_KEY_PARTS if k not in key_parts]
    if missing:
        raise ValueError(f"cache key is missing required parts: {missing}")
    payload = {"cache_format_version": CACHE_FORMAT_VERSION, **key_parts}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _save(path: Path, array: np.ndarray) -> None:
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)


def _load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _write_session(directory: Path, session: Session, key_parts: Mapping) -> None:
    unit_ids = list(session.spikes)
    lengths = [len(session.spikes[u]) for u in unit_ids]
    flat = (
        np.concatenate([session.spikes[u] for u in unit_ids])
        if unit_ids
        else np.empty(0, dtype=np.float64)
    )
    _save(directory / "spike_times.npy", flat)
    _save(
        directory / "spike_offsets.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    )
    session.units.to_parquet(directory / "units.parquet")
    session.trials.to_parquet(directory / "trials.parquet")

    (directory / "behaviour").mkdir()
    behaviour = []
    for name, series in session.behaviour.items():
        _save(directory / "behaviour" / f"{name}.timestamps.npy", series.timestamps)
        _save(directory / "behaviour" / f"{name}.data.npy", series.data)
        names = None if series.channel_names is None else list(series.channel_names)
        behaviour.append({"name": name, "channel_names": names})

    meta = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "key_parts": dict(key_parts),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eid": session.eid,
        "time_bounds": list(session.time_bounds),
        "spike_unit_ids": unit_ids,
        "behaviour": behaviour,
        "capabilities": {
            "present": sorted(session.available.present),
            "missing": dict(session.available.missing),
        },
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))


def _read_session(directory: Path, meta: dict) -> Session:
    if meta["cache_format_version"] != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"{directory}: cache format {meta['cache_format_version']} is not {CACHE_FORMAT_VERSION}"
        )
    flat = _load(directory / "spike_times.npy")
    offsets = _load(directory / "spike_offsets.npy")
    spikes = {
        unit_id: flat[offsets[i] : offsets[i + 1]]
        for i, unit_id in enumerate(meta["spike_unit_ids"])
    }
    behaviour = {}
    for entry in meta["behaviour"]:
        name, names = entry["name"], entry["channel_names"]
        behaviour[name] = TimeSeries(
            _load(directory / "behaviour" / f"{name}.timestamps.npy"),
            _load(directory / "behaviour" / f"{name}.data.npy"),
            channel_names=None if names is None else tuple(names),
        )
    caps = meta["capabilities"]
    return Session(
        eid=meta["eid"],
        time_bounds=tuple(meta["time_bounds"]),
        spikes=spikes,
        units=pd.read_parquet(directory / "units.parquet"),
        trials=pd.read_parquet(directory / "trials.parquet"),
        behaviour=behaviour,
        available=Capabilities(present=frozenset(caps["present"]), missing=caps["missing"]),
    )


class SessionCache:
    """Sessions stored under `root`, one directory per key."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def path(self, key_parts: Mapping) -> Path:
        key = cache_key(key_parts)
        return self.root / key[:2] / key

    def describe(self, key_parts: Mapping) -> dict | None:
        meta = self.path(key_parts) / "meta.json"
        return json.loads(meta.read_text()) if meta.exists() else None

    def get(self, key_parts: Mapping) -> Session | None:
        meta = self.describe(key_parts)
        return None if meta is None else _read_session(self.path(key_parts), meta)

    def put(self, key_parts: Mapping, session: Session) -> Path:
        if session.eid != key_parts["eid"]:
            raise ValueError(f"session eid {session.eid} does not match key eid {key_parts['eid']}")
        entry = self.path(key_parts)
        entry.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=entry.parent))
        try:
            _write_session(tmp, session, key_parts)
            if (entry / "meta.json").exists():
                shutil.rmtree(tmp)
                return entry
            if entry.exists():
                shutil.rmtree(entry)
            os.rename(tmp, entry)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return entry

    def get_or_load(self, key_parts: Mapping, loader: Callable[[], Session]) -> Session:
        cached = self.get(key_parts)
        if cached is not None:
            return cached
        session = loader()
        self.put(key_parts, session)
        return session
