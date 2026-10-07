"""DANDI 000409 (IBL Brain Wide Map) processed NWB files -> Session.

Every mapping to IBL's ALF conventions below was checked value by value against
ONE for session d23a44ef-1402-4ed7-97f5-47e9a7a504d9 (see docs/DECISIONS.md).
Values outside the checked encodings raise instead of being guessed.

Files are read from a local path or streamed from an https URL with remfile, which
fetches only the byte ranges that are read; nothing is downloaded in bulk.
"""

import contextlib
import json
import os
import re
import urllib.parse
import urllib.request

import h5py
import numpy as np
import pandas as pd
import remfile
from pynwb import NWBHDF5IO

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

DANDI_API = "https://api.dandiarchive.org/api"
DANDISET = "000409"
# A published, immutable version: the draft can change under us, like ONE revisions.
DANDISET_VERSION = "0.260309.1324"
# Bump when this module's mapping into a Session changes; it invalidates cached sessions.
LOADER_VERSION = 1

_TRIAL_TIMES = {
    "intervals_0": "start_time",
    "intervals_1": "stop_time",
    "stimOn_times": "gabor_stimulus_onset_time",
    "stimOff_times": "gabor_stimulus_offset_time",
    "goCue_times": "auditory_cue_time",
    "firstMovement_times": "wheel_movement_onset_time",
    "response_times": "choice_registration_time",
    "feedback_times": "feedback_time",
}
_TRIAL_EXTRAS = {
    "quiescencePeriod": "quiescence_period",
    "rewardVolume": "reward_volume_uL",
    "block_index": "block_index",
}
_CHOICE = {"clockwise": 1.0, "counter_clockwise": -1.0}
_SIDES = ("left", "right")
_CONTRAST_PERCENT = (0.0, 6.25, 12.5, 25.0, 50.0, 100.0)

_UNIT_COLUMNS = {
    "firing_rate": "firing_rate",
    "label": "ibl_quality_score",
    "depths": "distance_from_probe_tip_um",
}
_COORDS_REASON = (
    "NWB electrode coordinates are Allen CCF um; the canonical (BWM) convention is not verified"
)
_MISSING_UNITS = {
    "acronym": "NWB stores full Allen region names (kept as 'location'); "
    "name-to-acronym mapping not implemented",
    "x": _COORDS_REASON,
    "y": _COORDS_REASON,
    "z": _COORDS_REASON,
}
# Canonical behaviour key -> (processing module, data interface) in the NWB file.
# Pupil is the raw diameter, not the `...Smoothed` series: smoothing is preprocessing.
_NWB_CAMERA = {"left": "Left", "right": "Right", "body": "Body"}
_NWB_SIGNAL = {
    "motion_energy": ("motion_energy", "{cam}CameraMotionEnergy"),
    "pupil": ("pupil", "{cam}PupilDiameter"),
    "pose": ("pose_estimation", "{cam}Camera"),
}
_BEHAVIOUR_SOURCES = {"wheel": ("wheel", "WheelPosition")}
for _signal, _cameras in CAMERA_SIGNALS.items():
    _module, _name = _NWB_SIGNAL[_signal]
    for _camera in _cameras:
        _BEHAVIOUR_SOURCES[f"{_signal}_{_camera}"] = (
            _module,
            _name.format(cam=_NWB_CAMERA[_camera]),
        )
_LICK_REASON = "lick times are events, not a sampled signal; not loaded"


def _map_choice(values: pd.Series) -> np.ndarray:
    unknown = sorted(set(values) - set(_CHOICE))
    if unknown:
        raise ValueError(f"unexpected mouse_wheel_choice values: {unknown}")
    return values.map(_CHOICE).to_numpy(dtype=np.float64)


def _map_feedback(values: pd.Series) -> np.ndarray:
    if values.dtype != bool:
        raise ValueError(f"is_mouse_rewarded must be bool, got {values.dtype}")
    return np.where(values.to_numpy(), 1.0, -1.0)


def _split_contrast(side: pd.Series, contrast_percent: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (contrastLeft, contrastRight), each (n_trials,), as fractions with NaN off-side."""
    unknown = sorted(set(side) - set(_SIDES))
    if unknown:
        raise ValueError(f"unexpected gabor_stimulus_side values: {unknown}")
    pct = contrast_percent.to_numpy(dtype=np.float64)
    odd = sorted(set(pct[~np.isin(pct, _CONTRAST_PERCENT)]))
    if odd:
        raise ValueError(
            f"gabor_stimulus_contrast values {odd} are not IBL percent contrasts {_CONTRAST_PERCENT}"
        )
    frac = pct / 100.0
    is_left = (side == "left").to_numpy()
    return np.where(is_left, frac, np.nan), np.where(is_left, np.nan, frac)


def _trials(nwb) -> pd.DataFrame:
    src = nwb.trials.to_dataframe().reset_index(drop=True)
    out = pd.DataFrame(
        {ours: src[theirs].to_numpy(np.float64) for ours, theirs in _TRIAL_TIMES.items()}
    )
    out["choice"] = _map_choice(src["mouse_wheel_choice"])
    out["feedbackType"] = _map_feedback(src["is_mouse_rewarded"])
    out["contrastLeft"], out["contrastRight"] = _split_contrast(
        src["gabor_stimulus_side"], src["gabor_stimulus_contrast"]
    )
    out["probabilityLeft"] = src["probability_left"].to_numpy(np.float64)
    for ours, theirs in _TRIAL_EXTRAS.items():
        out[ours] = src[theirs].to_numpy()
    return out


def _as_str(values) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def _units_and_spikes(nwb) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    units = nwb.units
    names = _as_str(units["unit_name"].data[:])
    probe_from_name, cluster_id = zip(*(n.rsplit("_", 1) for n in names))
    probe_col = np.char.lower(_as_str(units["probe_name"].data[:]))
    if not np.array_equal(np.array(probe_from_name), probe_col):
        raise ValueError("unit_name prefixes disagree with the probe_name column")

    table = pd.DataFrame(
        {
            "probe_name": probe_col,
            "cluster_id": np.array(cluster_id, dtype=np.int64),
            "cluster_uuid": _as_str(units["cluster_uuid"].data[:]),
        },
        index=pd.Index(names, name="unit_id"),
    )
    for ours, theirs in _UNIT_COLUMNS.items():
        table[ours] = np.asarray(units[theirs].data[:], dtype=np.float64)
    electrode_rows = np.asarray(units["max_electrode"].data[:])
    table["location"] = _as_str(nwb.electrodes["location"].data[:])[electrode_rows]

    # Spike times are stored flat with cumulative end offsets, one per unit.
    flat = np.asarray(units["spike_times"].target.data[:], dtype=np.float64)
    ends = np.asarray(units["spike_times"].data[:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    spikes = {uid: flat[s:e] for uid, s, e in zip(names, starts, ends)}
    return table, spikes


def _keypoint_name(series_name: str) -> str:
    """'PoseEstimationSeriesRightPupilBottom' -> 'right_pupil_bottom'."""
    name = series_name.removeprefix("PoseEstimationSeries")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _pose(camera) -> TimeSeries:
    """One camera's keypoints as (n_samples, n_channels): x, y (px) and likelihood per keypoint.

    Likelihood is kept unthresholded; filtering low-confidence points is preprocessing.
    """
    series = camera.pose_estimation_series
    names = sorted(series, key=_keypoint_name)
    timestamps = np.asarray(series[names[0]].timestamps[:], dtype=np.float64)
    columns, channel_names = [], []
    for name in names:
        s = series[name]
        if not np.array_equal(np.asarray(s.timestamps[:], dtype=np.float64), timestamps):
            raise ValueError(f"{camera.name}: keypoint {name} has its own timestamps; cannot stack")
        xy = np.asarray(s.data[:], dtype=np.float64)
        keypoint = _keypoint_name(name)
        columns += [xy[:, 0], xy[:, 1]]
        channel_names += [f"{keypoint}_x", f"{keypoint}_y"]
        if s.confidence is not None:
            columns.append(np.asarray(s.confidence[:], dtype=np.float64))
            channel_names.append(f"{keypoint}_likelihood")
    return TimeSeries(timestamps, np.column_stack(columns), channel_names=tuple(channel_names))


def _behaviour(nwb) -> tuple[dict[str, TimeSeries], dict[str, str]]:
    loaded, missing = {}, {"lick": _LICK_REASON}
    for key, (module, name) in _BEHAVIOUR_SOURCES.items():
        if module not in nwb.processing or name not in nwb.processing[module].data_interfaces:
            missing[key] = f"no {module}/{name} in this NWB file"
            continue
        obj = nwb.processing[module][name]
        if key.startswith("pose_"):
            loaded[key] = _pose(obj)
            continue
        if obj.timestamps is None:
            raise ValueError(f"{module}/{name} is rate-sampled, without timestamps; not supported")
        loaded[key] = TimeSeries(obj.timestamps[:], obj.data[:])
    return loaded, missing


def _time_bounds(spikes, trials: pd.DataFrame, behaviour: dict[str, TimeSeries]) -> tuple:
    firsts = [t[0] for t in spikes.values() if t.size]
    lasts = [t[-1] for t in spikes.values() if t.size]
    trial_times = trials[[f for f in TRIAL_TIME_FIELDS if f in trials]].to_numpy(np.float64)
    trial_times = trial_times[np.isfinite(trial_times)]
    firsts += [trial_times.min()] if trial_times.size else []
    lasts += [trial_times.max()] if trial_times.size else []
    for series in behaviour.values():
        if series.timestamps.size:
            firsts.append(series.timestamps[0])
            lasts.append(series.timestamps[-1])
    return float(min(firsts)), float(max(lasts))


def _drop_all_nan(table: pd.DataFrame, group: str, fields, missing: dict) -> pd.DataFrame:
    empty = [f for f in fields if f in table and table[f].isna().all()]
    for f in empty:
        missing[f"{group}.{f}"] = "all values are NaN in the NWB file"
    return table.drop(columns=empty)


def _is_url(source) -> bool:
    return isinstance(source, str) and source.startswith(("https://", "http://"))


@contextlib.contextmanager
def _open_nwb(source):
    """Yield the NWBFile at a local path, or streamed from an http(s) URL."""
    with contextlib.ExitStack() as stack:
        if _is_url(source):
            remote = stack.enter_context(contextlib.closing(remfile.File(source)))
            h5 = stack.enter_context(h5py.File(remote, "r"))
            io = stack.enter_context(NWBHDF5IO(file=h5, mode="r", load_namespaces=True))
        else:
            io = stack.enter_context(NWBHDF5IO(str(source), "r", load_namespaces=True))
        yield io.read()


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def dandi_asset_url(eid: str, *, dandiset=DANDISET, version=DANDISET_VERSION) -> str:
    """S3 URL of one IBL session's `desc-processed` NWB file in a pinned DANDI version."""
    base = f"{DANDI_API}/dandisets/{dandiset}/versions/{version}/assets"
    pattern = f"sub-*/sub-*_ses-{eid}_desc-processed_behavior+ecephys.nwb"
    found = _get_json(f"{base}/?" + urllib.parse.urlencode({"glob": pattern}))
    if found["count"] != 1:
        raise ValueError(
            f"expected 1 desc-processed asset for eid {eid} in {dandiset}@{version}, "
            f"found {found['count']}"
        )
    meta = _get_json(f"{base}/{found['results'][0]['asset_id']}/")
    s3 = [u for u in meta.get("contentUrl", []) if "s3.amazonaws.com" in u]
    if len(s3) != 1:
        raise ValueError(f"asset for eid {eid} has {len(s3)} S3 content URLs, expected 1")
    return s3[0]


def load_session_nwb(source: str | os.PathLike) -> Session:
    """Read one DANDI 000409 `desc-processed` NWB file into a Session.

    source: a local path, or an https URL (e.g. from `dandi_asset_url`) to stream.
    Spike times and all other times are seconds on the session clock. The wheel is
    the raw encoder position (radians) and pupil the raw diameter; smoothing belongs
    to preprocessing. Video signals get one entry per camera, each on its own clock.
    """
    with _open_nwb(source) as nwb:
        if not nwb.session_id:
            raise ValueError(f"{source}: NWB session_id (the IBL eid) is missing")
        units, spikes = _units_and_spikes(nwb)
        trials = _trials(nwb)
        behaviour, missing_behaviour = _behaviour(nwb)
        eid = str(nwb.session_id)

    missing = {f"units.{f}": why for f, why in _MISSING_UNITS.items()}
    missing.update({f"behaviour.{f}": why for f, why in missing_behaviour.items()})
    trials = _drop_all_nan(trials, "trials", TRIAL_FIELDS, missing)
    units = _drop_all_nan(units, "units", UNIT_FIELDS, missing)

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
