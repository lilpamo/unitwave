"""Backend-independent session container returned by every data backend.

All times are seconds on the session clock. Every canonical field is either
present in the data or declared missing with a reason; nothing is defaulted.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

TRIAL_FIELDS = (
    "intervals_0",
    "intervals_1",
    "stimOn_times",
    "stimOff_times",
    "goCue_times",
    "firstMovement_times",
    "response_times",
    "feedback_times",
    "choice",
    "feedbackType",
    "contrastLeft",
    "contrastRight",
    "probabilityLeft",
)
TRIAL_TIME_FIELDS = tuple(
    f for f in TRIAL_FIELDS if f.endswith("_times") or f.startswith("intervals_")
)
UNIT_FIELDS = ("probe_name", "acronym", "x", "y", "z", "depths", "label", "firing_rate")
# IBL films each session with three cameras at different rates, so video-derived
# signals get one key per camera. The body camera does not see the pupil.
CAMERA_SIGNALS = {
    "motion_energy": ("left", "right", "body"),
    "pupil": ("left", "right"),
    "pose": ("left", "right", "body"),
}
BEHAVIOUR_FIELDS = ("wheel", "lick") + tuple(
    f"{signal}_{camera}" for signal, cameras in CAMERA_SIGNALS.items() for camera in cameras
)
CANONICAL_FIELDS = frozenset(
    [f"trials.{f}" for f in TRIAL_FIELDS]
    + [f"units.{f}" for f in UNIT_FIELDS]
    + [f"behaviour.{f}" for f in BEHAVIOUR_FIELDS]
)

# A real recording is hours long. Anything longer almost certainly means times in ms.
MAX_SESSION_SECONDS = 24 * 3600.0


def _check_times(name: str, t: np.ndarray) -> None:
    if t.ndim != 1:
        raise ValueError(f"{name}: expected shape (n,), got {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{name}: times must be finite")
    if t.size > 1 and np.any(np.diff(t) < 0):
        raise ValueError(f"{name}: times must be sorted (non-decreasing)")


@dataclass(frozen=True)
class TimeSeries:
    """A sampled behavioural signal.

    timestamps: (n_samples,) seconds, sorted.
    data: (n_samples,) or (n_samples, n_channels).
    channel_names: one unique name per column, required when data is 2-D.
    """

    timestamps: np.ndarray
    data: np.ndarray
    channel_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        ts = np.asarray(self.timestamps, dtype=np.float64)
        _check_times("timestamps", ts)
        data = np.asarray(self.data)
        if data.ndim not in (1, 2) or data.shape[0] != ts.shape[0]:
            raise ValueError(
                f"data must be (n_samples,) or (n_samples, n_channels) with "
                f"n_samples={ts.shape[0]}, got {data.shape}"
            )
        names = None if self.channel_names is None else tuple(self.channel_names)
        if data.ndim == 1 and names is not None:
            raise ValueError("channel_names given for single-channel (n_samples,) data")
        if data.ndim == 2 and (names is None or len(names) != data.shape[1]):
            raise ValueError(
                f"(n_samples, n_channels) data with n_channels={data.shape[1]} needs "
                f"that many channel_names, got {names}"
            )
        if names is not None and len(set(names)) != len(names):
            raise ValueError(f"channel_names must be unique, got {names}")
        object.__setattr__(self, "timestamps", ts)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "channel_names", names)


@dataclass(frozen=True)
class Capabilities:
    """Which canonical fields a session has, keyed like 'trials.stimOn_times'.

    present: fields whose data is in the session.
    missing: field -> reason it is absent.
    """

    present: frozenset[str]
    missing: Mapping[str, str]

    def __post_init__(self) -> None:
        both = set(self.present) & set(self.missing)
        if both:
            raise ValueError(f"fields both present and missing: {sorted(both)}")
        no_reason = [k for k, why in self.missing.items() if not str(why).strip()]
        if no_reason:
            raise ValueError(f"missing fields need a reason: {sorted(no_reason)}")
        object.__setattr__(self, "present", frozenset(self.present))
        object.__setattr__(self, "missing", MappingProxyType(dict(self.missing)))


@dataclass(frozen=True)
class Session:
    """One recording session.

    eid: session id.
    time_bounds: (t_start, t_end) seconds, the recording's extent on the session clock.
    spikes: unit_id -> (n_spikes,) spike times in seconds, sorted.
    units: (n_units, ...) indexed by unit_id, the same ids as spikes.
    trials: (n_trials, ...) with canonical column names from TRIAL_FIELDS.
    behaviour: name -> TimeSeries, names from BEHAVIOUR_FIELDS.
    available: every canonical field, present or missing with a reason.
    """

    eid: str
    time_bounds: tuple[float, float]
    spikes: Mapping[str, np.ndarray]
    units: pd.DataFrame
    trials: pd.DataFrame
    behaviour: Mapping[str, TimeSeries]
    available: Capabilities

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def __post_init__(self) -> None:
        self._check_bounds()
        self._check_spikes_and_units()
        self._check_trials()
        self._check_behaviour()
        self._check_capabilities()

    def _check_bounds(self) -> None:
        t0, t1 = (float(t) for t in self.time_bounds)
        if not (np.isfinite(t0) and np.isfinite(t1) and t0 < t1):
            raise ValueError(f"time_bounds must be finite and increasing, got {self.time_bounds}")
        if t1 - t0 > MAX_SESSION_SECONDS:
            raise ValueError(
                f"time_bounds span {t1 - t0:.0f} s (> 24 h); are times in seconds, not ms?"
            )
        object.__setattr__(self, "time_bounds", (t0, t1))

    def _within_bounds(self, t: np.ndarray) -> bool:
        t0, t1 = self.time_bounds
        return bool(np.all((t >= t0) & (t <= t1)))

    def _check_spikes_and_units(self) -> None:
        if not self.units.index.is_unique:
            raise ValueError("units index (unit_id) must be unique")
        if set(self.spikes) != set(self.units.index):
            only_spikes = sorted(set(self.spikes) - set(self.units.index))[:5]
            only_units = sorted(set(self.units.index) - set(self.spikes))[:5]
            raise ValueError(
                f"unit ids differ between spikes and units: only in spikes {only_spikes}, "
                f"only in units {only_units}"
            )
        spikes = {}
        for unit_id, times in self.spikes.items():
            t = np.asarray(times, dtype=np.float64)
            _check_times(f"spikes of unit {unit_id!r}", t)
            if t.size and not self._within_bounds(t[[0, -1]]):
                raise ValueError(
                    f"spikes of unit {unit_id!r} fall outside time_bounds {self.time_bounds}"
                )
            spikes[unit_id] = t
        object.__setattr__(self, "spikes", MappingProxyType(spikes))

    def _check_trials(self) -> None:
        for field in TRIAL_TIME_FIELDS:
            if field not in self.trials.columns:
                continue
            t = self.trials[field].to_numpy(dtype=np.float64)
            t = t[np.isfinite(t)]
            if not self._within_bounds(t):
                raise ValueError(
                    f"trials.{field} has values outside time_bounds {self.time_bounds}"
                )

    def _check_behaviour(self) -> None:
        for name, series in self.behaviour.items():
            if not isinstance(series, TimeSeries):
                raise ValueError(f"behaviour {name!r} must be a TimeSeries")
            ts = series.timestamps
            if ts.size and not self._within_bounds(ts[[0, -1]]):
                raise ValueError(
                    f"behaviour {name!r} timestamps fall outside time_bounds {self.time_bounds}"
                )
        object.__setattr__(self, "behaviour", MappingProxyType(dict(self.behaviour)))

    def _check_capabilities(self) -> None:
        declared = set(self.available.present) | set(self.available.missing)
        unknown = sorted(declared - CANONICAL_FIELDS)
        if unknown:
            raise ValueError(f"unknown capability fields: {unknown}")
        undeclared = sorted(CANONICAL_FIELDS - declared)
        if undeclared:
            raise ValueError(f"canonical fields neither present nor missing: {undeclared}")

        for key in sorted(CANONICAL_FIELDS):
            group, name = key.split(".", 1)
            in_data, values = self._lookup(group, name)
            if key in self.available.present:
                if not in_data:
                    raise ValueError(f"{key} declared present but not in data")
                if values is not None and bool(pd.isna(values).all()):
                    raise ValueError(f"{key} is all-NaN; declare it missing instead")
            elif in_data:
                raise ValueError(f"{key} declared missing but present in data")

    def _lookup(self, group: str, name: str) -> tuple[bool, pd.Series | None]:
        if group == "behaviour":
            return name in self.behaviour, None
        table = self.trials if group == "trials" else self.units
        if name not in table.columns:
            return False, None
        return True, table[name]
