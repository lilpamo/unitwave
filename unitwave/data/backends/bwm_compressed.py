"""ibl-ai-agent's compressed Brain Wide Map datasets (`bwm_ephys`, `bwm_behavior`) -> Session.

Checked against the DANDI NWB backend for session d23a44ef-1402-4ed7-97f5-47e9a7a504d9
(see docs/DECISIONS.md): the same good units, identical spike counts, spike times
within 50 us (they are stored rounded to 100 us ticks), and identical trials except
firstMovement_times, which comes from a different trials revision.

Behaviour comes from `bwm_behavior`, which stores resampled, quantised signals: the
wheel is the raw position linearly interpolated onto a 100 Hz grid and rounded to
0.001 rad, and each camera keeps the nearest real frame to a uniform 60 or 30 Hz grid,
with the grid times standing in for the frame times (within half a frame).
"""

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numcodecs import Blosc

from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    CAMERA_SIGNALS,
    TRIAL_FIELDS,
    TRIAL_TIME_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
    TimeSeries,
)

DATASET_NAME = "bwm_ephys"
# Pinned: a new release can change counts or encodings, like ONE and DANDI revisions.
DATASET_VERSION = "1.2.1"
BEHAVIOUR_DATASET_NAME = "bwm_behavior"
BEHAVIOUR_DATASET_VERSION = "2.0.0"
# Bump when this module's mapping into a Session changes; it invalidates cached sessions.
LOADER_VERSION = 2

_SHARD_FORMAT = {
    "format": "ibl_agent_spike_shard_v2",
    "time_encoding": "delta_int_ticks",
    "cluster_encoding": "dense_local_indices",
}
_UNIT_EXTRAS = [
    "pid",
    "cluster_id",
    "beryl_acronym",
    "atlas_id",
    "axial_um",
    "lateral_um",
    "spike_count",
]
_NO_BEHAVIOUR_REASON = "no bwm_behavior release was given to the loader"
_BEHAVIOUR_SHARD_FORMAT = "ibl_ai_agent_behavior_session_shard_v2"
_CAMERAS = {"left": "leftCamera", "right": "rightCamera", "body": "bodyCamera"}
_MOTION_ENERGY = {
    "left": "whiskerMotionEnergy",
    "right": "whiskerMotionEnergy",
    "body": "bodyMotionEnergy",
}
# The raw diameter, not `pupilDiameter_smooth`: smoothing is a preprocessing choice.
_PUPIL = "pupilDiameter_raw"
_POSE_SUFFIXES = ("_x", "_y", "_likelihood")


def _read_array(shard_dir: Path, meta: dict, name: str) -> np.ndarray:
    spec = meta["arrays"][name]
    dtype = np.dtype(spec["dtype"])
    if int(np.prod(spec["shape"])) == 0:
        return np.empty(spec["shape"], dtype=dtype)
    raw = Blosc().decode((shard_dir / spec["entry"]).read_bytes())
    return np.frombuffer(raw, dtype=dtype).reshape(spec["shape"])


def _decode_spike_shard(shard_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one insertion's spikes.

    Returns spike times in seconds (n_spikes,), the shard's cluster ids (n_units,),
    and each spike's index into those ids (n_spikes,).
    """
    shard_dir = Path(shard_dir)
    meta = json.loads((shard_dir / "meta.json").read_text())
    for key, expected in _SHARD_FORMAT.items():
        if meta.get(key) != expected:
            raise ValueError(f"{shard_dir}: {key}={meta.get(key)!r}, expected {expected!r}")

    deltas = _read_array(shard_dir, meta, "spike_times_delta_ticks").astype(np.int64)
    ticks = np.cumsum(deltas) + int(meta["time_origin_ticks"])
    times = ticks * int(meta["time_quantization_us"]) / 1_000_000.0

    cluster_ids = _read_array(shard_dir, meta, "cluster_ids").astype(np.int64)
    local = _read_array(shard_dir, meta, "spike_clusters").astype(np.int64)
    counts = _read_array(shard_dir, meta, "cluster_spike_counts")
    if local.shape != times.shape:
        raise ValueError(f"{shard_dir}: spike_clusters and spike times differ in length")
    if not np.array_equal(np.bincount(local, minlength=len(cluster_ids)), counts):
        raise ValueError(f"{shard_dir}: spike_clusters disagree with cluster_spike_counts")
    return times, cluster_ids, local


def _split_by_unit(times: np.ndarray, local: np.ndarray, n_units: int) -> list[np.ndarray]:
    """Group (n_spikes,) times by unit index; a stable sort keeps each unit in time order."""
    # NumPy's stable sort is a linear-time radix sort for <=16-bit integers, ~10x
    # faster than for int64 on a 23M-spike probe; the resulting order is identical.
    fits_16_bits = n_units <= np.iinfo(np.uint16).max
    order = np.argsort(local.astype(np.uint16) if fits_16_bits else local, kind="stable")
    bounds = np.searchsorted(local[order], np.arange(n_units + 1))
    sorted_times = times[order]
    return [sorted_times[bounds[i] : bounds[i + 1]] for i in range(n_units)]


def _check_release(root: Path, name: str, version: str) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    found = (manifest.get("dataset_name"), manifest.get("dataset_version"))
    if found != (name, version):
        raise ValueError(f"{root}: found {found}, this backend reads {name} {version}")


def _check_version(root: Path) -> None:
    _check_release(root, DATASET_NAME, DATASET_VERSION)


def _units_and_spikes(root: Path, eid: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    units = pd.read_parquet(root / "metadata/units.parquet", filters=[("eid", "==", eid)])
    if units.empty:
        raise ValueError(f"eid {eid} has no units in {root}")
    units = units.sort_values(["probe_name", "cluster_id"])
    units.index = pd.Index(
        units["probe_name"] + "_" + units["cluster_id"].astype(str), name="unit_id"
    )

    spikes = {}
    for pid, probe_units in units.groupby("pid", sort=False):
        times, cluster_ids, local = _decode_spike_shard(root / "spikes" / pid)
        table_ids = probe_units["cluster_id"].to_numpy(np.int64)
        if set(cluster_ids) != set(table_ids):
            raise ValueError(f"insertion {pid}: units table and spike shard list different units")
        per_unit = _split_by_unit(times, local, len(cluster_ids))
        probe_name = probe_units["probe_name"].iloc[0]
        for cluster_id, unit_times in zip(cluster_ids, per_unit):
            spikes[f"{probe_name}_{cluster_id}"] = unit_times

    counts = pd.Series({u: len(t) for u, t in spikes.items()})
    if not (counts[units.index].to_numpy() == units["spike_count"].to_numpy()).all():
        raise ValueError(f"eid {eid}: units table spike_count disagrees with the spike shards")

    table = units[[*UNIT_FIELDS, *_UNIT_EXTRAS]].copy()
    numeric = [f for f in UNIT_FIELDS if f not in ("probe_name", "acronym")]
    table[numeric] = table[numeric].astype(np.float64)
    return table, spikes


def _trials(root: Path, eid: str) -> pd.DataFrame:
    trials = pd.read_parquet(root / "metadata/trials.parquet", filters=[("eid", "==", eid)])
    trials = trials.sort_values("trial_id").reset_index(drop=True)
    keep = [f for f in TRIAL_FIELDS if f in trials] + ["trial_id", "bwm_include"]
    return trials[keep]


def _blosc(zf: zipfile.ZipFile, entry: str, spec: dict) -> np.ndarray:
    dtype, shape = np.dtype(spec["dtype"]), tuple(spec["shape"])
    if int(np.prod(shape)) == 0:
        return np.empty(shape, dtype=dtype)
    return np.frombuffer(Blosc().decode(zf.read(entry)), dtype=dtype).reshape(shape)


def _scaled(encoded: np.ndarray, precision: float) -> np.ndarray:
    """Integers times their precision; the integer type's minimum marks NaN."""
    decoded = encoded.astype(np.float64) * precision
    if np.issubdtype(encoded.dtype, np.integer):
        decoded[encoded == np.iinfo(encoded.dtype).min] = np.nan
    return decoded


def _unpack_uint4(packed: np.ndarray, shape: tuple) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8).ravel()
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2], out[1::2] = packed >> 4, packed & 0x0F
    return out[: int(np.prod(shape))].reshape(shape)


def _feature_group(zf: zipfile.ZipFile, group: dict) -> np.ndarray:
    kind = group["kind"]
    if kind == "delta_scaled":
        first = _blosc(zf, group["first_entry"], group["first_spec"]).astype(np.int64)
        deltas = _blosc(zf, group["delta_entry"], group["delta_spec"]).astype(np.int64)
        absolute = np.concatenate([first, first + np.cumsum(deltas, axis=0)], axis=0)
        return absolute.astype(np.float64) * group["precision"]
    if kind == "scaled":
        return _scaled(_blosc(zf, group["entry"], group["payload_spec"]), group["precision"])
    if kind == "likelihood":
        bits, spec = int(group["bits"]), group["payload_spec"]
        if bits == 4:
            levels = _unpack_uint4(
                _blosc(zf, group["entry"], spec["packed_uint4"]), tuple(spec["shape"])
            )
        else:
            levels = _blosc(zf, group["entry"], spec)
        decoded = levels.astype(np.float64) / (2**bits - 1)
        if "nan_mask_entry" in group:
            mask = np.unpackbits(_blosc(zf, group["nan_mask_entry"], group["nan_mask_spec"]))
            decoded[mask[: decoded.size].astype(bool).reshape(decoded.shape)] = np.nan
        return decoded
    if kind == "float16":
        return _blosc(zf, group["entry"], group["payload_spec"]).astype(np.float64)
    raise ValueError(f"unsupported feature group kind {kind!r}")


def _decode_behaviour_array(zf: zipfile.ZipFile, name: str, spec: dict) -> np.ndarray:
    """Decode one shard array, following ibl-ai-agent's bwm_behavior_compression decoder."""
    enc = spec["encoding"]
    kind = enc["kind"]
    if kind == "timestamp_fixed_rate":
        return enc["start"] + np.arange(int(enc["count"]), dtype=np.float64) / enc["rate_hz"]
    if kind == "timestamp_delta_ticks":
        ticks = _blosc(zf, enc["ticks_entry"], enc["ticks_spec"]).astype(np.float64)
        return enc["start"] + ticks * enc["tick_s"]
    if kind == "scaled_numeric":
        return _scaled(_blosc(zf, enc["entry"], enc["payload_spec"]), enc["precision"])
    if kind == "feature_matrix":
        out = np.empty(tuple(enc["shape"]), dtype=np.float64)
        for group in enc["groups"]:
            out[:, group["columns"]] = _feature_group(zf, group).reshape(len(out), -1)
        return out
    raise ValueError(f"{name}: unsupported encoding kind {kind!r}")


def _read_behaviour_shard(path: str | os.PathLike) -> tuple[dict[str, np.ndarray], dict]:
    """Decode every array in a bwm_behavior session shard; returns (arrays, meta)."""
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read(next(n for n in zf.namelist() if n.endswith("meta.json"))))
        if meta.get("format") != _BEHAVIOUR_SHARD_FORMAT:
            raise ValueError(
                f"{path}: format {meta.get('format')!r}, expected {_BEHAVIOUR_SHARD_FORMAT}"
            )
        arrays = {
            name: _decode_behaviour_array(zf, name, spec) for name, spec in meta["arrays"].items()
        }
    return arrays, meta


def _behaviour_from_shard(path: str | os.PathLike) -> tuple[dict[str, TimeSeries], dict[str, str]]:
    """Map a session shard to canonical behaviour keys; returns (series, missing reasons)."""
    arrays, meta = _read_behaviour_shard(path)
    behaviour, missing = {}, {"lick": "not in bwm_behavior"}

    if "wheel.timestamps" in arrays and "wheel.position" in arrays:
        # Position only: the stored velocity bakes in IBL's smoothing filter.
        behaviour["wheel"] = TimeSeries(arrays["wheel.timestamps"], arrays["wheel.position"])
    else:
        missing["wheel"] = "no wheel in this bwm_behavior session"

    for camera, name in _CAMERAS.items():
        signals = [s for s, cameras in CAMERA_SIGNALS.items() if camera in cameras]
        info = meta.get("cameras", {}).get(name)
        if info is None or f"{name}.features" not in arrays:
            for signal in signals:
                missing[f"{signal}_{camera}"] = f"no {name} in this bwm_behavior session"
            continue
        times, features = arrays[f"{name}.timestamps"], arrays[f"{name}.features"]
        columns, skipped = list(info["columns"]), set(info.get("skipped_sources", []))

        pose = [c for c in columns if c.endswith(_POSE_SUFFIXES)]
        if pose:
            behaviour[f"pose_{camera}"] = TimeSeries(
                times, features[:, [columns.index(c) for c in pose]], channel_names=tuple(pose)
            )
        else:
            missing[f"pose_{camera}"] = f"{name} has no pose keypoints in bwm_behavior"

        for signal, column in (("motion_energy", _MOTION_ENERGY[camera]), ("pupil", _PUPIL)):
            if signal not in signals:
                continue
            key = f"{signal}_{camera}"
            if column in skipped:
                missing[key] = (
                    f"{column} skipped by the bwm_behavior build: timestamps did not match {name}"
                )
            elif column in columns:
                behaviour[key] = TimeSeries(times, features[:, columns.index(column)])
            else:
                missing[key] = f"{name} has no {column} in bwm_behavior"
    return behaviour, missing


def _time_bounds(
    spikes: dict[str, np.ndarray], trials: pd.DataFrame, behaviour: dict[str, TimeSeries]
) -> tuple[float, float]:
    firsts = [t[0] for t in spikes.values() if t.size]
    lasts = [t[-1] for t in spikes.values() if t.size]
    trial_times = trials[[f for f in TRIAL_TIME_FIELDS if f in trials]].to_numpy(np.float64)
    trial_times = trial_times[np.isfinite(trial_times)]
    if trial_times.size:
        firsts.append(trial_times.min())
        lasts.append(trial_times.max())
    for series in behaviour.values():
        if series.timestamps.size:
            firsts.append(series.timestamps[0])
            lasts.append(series.timestamps[-1])
    return float(min(firsts)), float(max(lasts))


def _declare_missing(table: pd.DataFrame, group: str, fields, missing: dict) -> pd.DataFrame:
    """Record canonical fields that are absent or all-NaN as missing, and drop the latter."""
    for field in fields:
        if field not in table:
            missing[f"{group}.{field}"] = f"not in the bwm_ephys {group} table"
        elif table[field].isna().all():
            missing[f"{group}.{field}"] = f"all values are NaN in the bwm_ephys {group} table"
    return table.drop(columns=[f for f in fields if f"{group}.{f}" in missing and f in table])


def load_session_bwm(
    eid: str, root: str | os.PathLike, behaviour_root: str | os.PathLike | None = None
) -> Session:
    """Read one session from extracted `bwm_ephys` (and optionally `bwm_behavior`) releases.

    root: the bwm_ephys release directory, e.g. .../bwm_compressed/bwm_ephys/1.2.1.
    behaviour_root: the bwm_behavior release directory; without it behaviour is declared
    missing. Units are IBL good units only (label == 1). Spike times are seconds, rounded
    to the dataset's 100 us ticks. Trials keep the release's `bwm_include` flag as an
    extra column. Pose keeps the tracker's raw estimates and likelihood, unthresholded;
    nothing is filtered here.
    """
    root = Path(root)
    _check_version(root)
    units, spikes = _units_and_spikes(root, eid)
    trials = _trials(root, eid)

    if behaviour_root is None:
        behaviour, missing_behaviour = {}, {f: _NO_BEHAVIOUR_REASON for f in BEHAVIOUR_FIELDS}
    else:
        behaviour_root = Path(behaviour_root)
        _check_release(behaviour_root, BEHAVIOUR_DATASET_NAME, BEHAVIOUR_DATASET_VERSION)
        shard = behaviour_root / "sessions" / f"{eid}.zip"
        if shard.exists():
            behaviour, missing_behaviour = _behaviour_from_shard(shard)
        else:
            behaviour = {}
            missing_behaviour = {f: "no session shard in bwm_behavior" for f in BEHAVIOUR_FIELDS}

    missing = {f"behaviour.{f}": why for f, why in missing_behaviour.items()}
    trials = _declare_missing(trials, "trials", TRIAL_FIELDS, missing)
    units = _declare_missing(units, "units", UNIT_FIELDS, missing)

    present = {f"trials.{f}" for f in TRIAL_FIELDS if f in trials}
    present |= {f"units.{f}" for f in UNIT_FIELDS if f in units}
    present |= {f"behaviour.{f}" for f in BEHAVIOUR_FIELDS if f in behaviour}

    return Session(
        eid=eid,
        time_bounds=_time_bounds(spikes, trials, behaviour),
        spikes=spikes,
        units=units,
        trials=trials,
        behaviour=behaviour,
        available=Capabilities(present=frozenset(present), missing=missing),
    )
