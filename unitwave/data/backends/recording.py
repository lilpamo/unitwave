"""A recording: one session's Phy probes, its events and its rig's behaviour, described
by a `recording.yaml` in the session's folder (step 14a; the user's choices, 2026-10-05).

    events: events.csv                    # trial events, on the events clock
    events_sync: nidq.xd_8_3_500.txt      # sync pulses on the events clock
    probes:
      - name: imec0                       # optional; default the folder's name
        folder: imec0/kilosort4           # a Kilosort/Phy folder
        sync: imec0.ap.xd_384_6_500.txt   # the same pulses on this probe's clock
        locations: imec0/channel_locations.json   # optional (data.channel_locations)
    behaviour:                            # optional; names as the task uses them
      wheel: {file: behaviour/wheel.csv, clock: events}
      pupil_left: {file: behaviour/pupil.times.npy, clock: imec0, column: area}

- **One clock:** every probe's spikes move onto the shared events clock by that probe's
  own fit (data.sync, the inverse line). A probe without `sync` is taken to be on the
  events clock already, and the report says so. Spikes more than the sync config's
  `max_outside_s` beyond the pulses aren't on a checked clock: they are dropped and
  counted per probe.
- **Behaviour files:** a CSV with a `time` column (seconds) and value columns (`column`
  picks one; otherwise every value column is a channel), or `<name>.times.npy` with its
  `<name>.values.npy`. Times must increase. `clock` is `events` or a probe's name.
- **Paths** are relative to the file's folder and must stay inside it. Anything
  malformed is refused, saying where.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.analysis.tasks import TaskDefinition
from unitwave.data.backends.phy import _events, read_probe
from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
    TimeSeries,
)
from unitwave.data.sync import fit_clock, load_sync_config, read_pulses

RECORDING_FILE = "recording.yaml"
_KEYS = {"events", "events_sync", "probes", "behaviour"}
_PROBE_KEYS = {"name", "folder", "sync", "locations"}
_BEHAVIOUR_KEYS = {"file", "clock", "column"}
_TIME_COLUMNS = ("time", "times", "t_s", "timestamps")


def _inside(base: Path, rel, where: str) -> Path:
    """rel resolved under base (symlinks followed); refused outside it or missing."""
    if not isinstance(rel, str) or not rel:
        raise ValueError(f"{where}: expected a path relative to the recording's folder")
    path = (base / rel).resolve()
    if path != base and base not in path.parents:
        raise ValueError(f"{where}: {rel} is outside the recording's folder")
    if not path.exists():
        raise ValueError(f"{where}: no {rel} in {base}")
    return path


def _keys(where: str, raw, allowed: set, required: set) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping")  # noqa: TRY004 - a refusal, in words
    unknown, missing = sorted(set(raw) - allowed), sorted(required - set(raw))
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}")


def read_recording(path: str | os.PathLike) -> dict:
    """The recording file, validated, with every path resolved: {events, events_sync,
    probes: [{name, folder, sync, locations}], behaviour: {name: {file, clock, column}}}."""
    path = Path(path).resolve()
    base = path.parent
    raw = yaml.safe_load(path.read_text()) or {}
    where = path.name
    _keys(where, raw, _KEYS, {"events", "probes"})
    probes, names = [], set()
    if not isinstance(raw["probes"], list) or not raw["probes"]:
        raise ValueError(f"{where}: give at least one probe")
    for i, p in enumerate(raw["probes"]):
        here = f"{where}, probe {i + 1}"
        _keys(here, p, _PROBE_KEYS, {"folder"})
        folder = _inside(base, p["folder"], here)
        name = str(p.get("name") or folder.name)
        if name in names:
            raise ValueError(f"{where}: probe name {name} is used twice")
        names.add(name)
        probes.append(
            {
                "name": name,
                "folder": folder,
                "sync": _inside(base, p["sync"], here) if p.get("sync") else None,
                "locations": _inside(base, p["locations"], here) if p.get("locations") else None,
            }
        )
    events_sync = _inside(base, raw["events_sync"], where) if raw.get("events_sync") else None
    if any(p["sync"] for p in probes) and events_sync is None:
        raise ValueError(
            f"{where}: probes have sync pulses but the events' clock has none (events_sync)"
        )
    behaviour = {}
    for name, b in (raw.get("behaviour") or {}).items():
        here = f"{where}, behaviour {name}"
        _keys(here, b, _BEHAVIOUR_KEYS, {"file", "clock"})
        if b["clock"] != "events" and b["clock"] not in names:
            raise ValueError(f"{here}: clock {b['clock']}: events or one of {sorted(names)}")
        behaviour[str(name)] = {
            "file": _inside(base, b["file"], here),
            "rel": b["file"],
            "clock": b["clock"],
            "column": b.get("column"),
        }
    return {
        "events": _inside(base, raw["events"], where),
        "events_sync": events_sync,
        "probes": probes,
        "behaviour": behaviour,
    }


def _read_series(path: Path, column: str | None) -> tuple[np.ndarray, np.ndarray, tuple | None]:
    """(times, values, channel names or None) from a behaviour CSV or .times.npy pair."""
    if path.name.endswith(".times.npy"):
        times = np.load(path).astype(np.float64).ravel()
        values = np.load(path.with_name(path.name.replace(".times.npy", ".values.npy")))
        names = None
    elif path.suffix == ".csv":
        table = pd.read_csv(path)
        time_col = next((c for c in _TIME_COLUMNS if c in table), None)
        if time_col is None:
            raise ValueError(
                f"{path.name}: needs a time column ({', '.join(_TIME_COLUMNS)}), in seconds"
            )
        times = table[time_col].to_numpy(np.float64)
        rest = [c for c in table if c != time_col]
        if column is not None:
            if column not in rest:
                raise ValueError(f"{path.name}: no column {column!r}; it has {rest}")
            rest = [column]
        if not rest:
            raise ValueError(f"{path.name}: no value column beside {time_col}")
        values = table[rest].to_numpy(np.float64)
        names = tuple(rest) if len(rest) > 1 else None
        values = values if names else values[:, 0]
    else:
        raise ValueError(f"{path.name}: give a .csv with a time column, or <name>.times.npy")
    if times.ndim != 1 or len(values) != times.size or times.size < 2:
        raise ValueError(
            f"{path.name}: times and values differ in length, or hold fewer than 2 samples"
        )
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path.name}: times must be finite and increasing")
    return times, np.asarray(values, np.float64), names


def load_recording(
    path: str | os.PathLike, task: TaskDefinition | None = None
) -> tuple[Session, dict]:
    """(Session on the events clock, report): every probe's units, the events and the
    behaviour; the report gives each probe's clock fit (or none), its dropped spikes and
    its locations, and each behaviour signal's file, clock and samples."""
    path = Path(path).resolve()
    rec = read_recording(path)
    cfg = load_sync_config()
    events_pulses = read_pulses(rec["events_sync"]) if rec["events_sync"] else None
    spikes, tables, missing_units, fits, report_probes = {}, [], {}, {}, []
    for p in rec["probes"]:
        probe_spikes, units, missing = read_probe(p["folder"], p["name"], p["locations"])
        fit, dropped = None, 0
        if p["sync"] is not None:
            fit = fit_clock(read_pulses(p["sync"]), events_pulses, cfg)
            for unit, t in probe_spikes.items():
                mapped, inside = fit.to_events(t)
                dropped += int((~inside).sum())
                probe_spikes[unit] = mapped[inside]
            units["n_spikes"] = [probe_spikes[u].size for u in units.index]
        fits[p["name"]] = fit
        spikes |= probe_spikes
        tables.append(units)
        missing_units |= missing  # the same reasons on every probe
        report_probes.append(
            {
                "name": p["name"],
                "folder": str(p["folder"].relative_to(path.parent)),
                "units": len(units),
                "clock": None if fit is None else fit.describe(),
                "dropped_spikes": dropped,
                "regions": None if p["locations"] is None else p["locations"].name,
            }
        )
    units = pd.concat(tables)
    # A field is known only if every probe knows it (e.g. regions from every probe's file).
    for field in ("acronym", "x", "y", "z", "depths"):
        if field in units and units[field].isna().any():
            units = units.drop(columns=field)
            missing_units[f"units.{field}"] = f"not known for every probe of {path.name}"
        elif field in units:
            missing_units.pop(f"units.{field}", None)
    nonempty = [t for t in spikes.values() if t.size]
    if not nonempty:
        raise ValueError(f"{path.name}: no spikes on the events clock")
    span = (min(float(t[0]) for t in nonempty), max(float(t[-1]) for t in nonempty))
    trials = _events(rec["events"], span, task, None)
    behaviour, report_behaviour = {}, {}
    for name, b in rec["behaviour"].items():
        times, values, channels = _read_series(b["file"], b["column"])
        if b["clock"] != "events" and fits[b["clock"]] is not None:
            times, inside = fits[b["clock"]].to_events(times)
            times, values = times[inside], values[inside]
        behaviour[name] = TimeSeries(times, values, channels)
        report_behaviour[name] = {"file": b["rel"], "clock": b["clock"], "samples": int(times.size)}
    first = min([span[0], *(float(s.timestamps[0]) for s in behaviour.values())])
    last = max([span[1], *(float(s.timestamps[-1]) for s in behaviour.values())])
    missing = dict(missing_units)
    missing |= {
        f"trials.{f}": "not in the events file, or empty there"
        for f in TRIAL_FIELDS
        if f not in trials
    }
    missing |= {
        f"behaviour.{f}": f"not in {path.name}" for f in BEHAVIOUR_FIELDS if f not in behaviour
    }
    present = {f"units.{f}" for f in UNIT_FIELDS if f in units}
    present |= {f"trials.{f}" for f in TRIAL_FIELDS if f in trials}
    present |= {f"behaviour.{f}" for f in behaviour if f in BEHAVIOUR_FIELDS}
    session = Session(
        eid=f"recording:{path}",
        time_bounds=(min(first, 0.0), last),
        spikes=spikes,
        units=units,
        trials=trials,
        behaviour=behaviour,
        available=Capabilities(present=frozenset(present), missing=missing),
    )
    return session, {
        "kind": "recording",
        "file": path.name,
        "probes": report_probes,
        "behaviour": report_behaviour,
    }
