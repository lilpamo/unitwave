"""Read a declared subset of any NWB file into a Session, with a capability report.

The subset Studio reads (step 5):
- **units:** spike times (required), the unit's probe (its electrode group), and, when
  a layout declares them, a depth column, a quality column with its passing rule, and
  where the unit's region comes from;
- **trials:** the trials table (required: the task period comes from it). start_time
  and stop_time become intervals_0 and intervals_1; other scalar columns keep the
  file's names, for the task definition to read. Booleans become 0/1;
- **behaviour:** the time series a layout maps by path, single-channel only;
- **regions:** location names, used as regions only when every one is an Allen CCF
  acronym; otherwise kept as `location` and the regions are unavailable, saying why.

Anything outside it is refused with the reason, never guessed:
- a unit whose electrodes lie on another probe than its own group;
- a declared peak channel off the unit's own probe;
- a rate-sampled series whose rate makes it longer than 24 h (DANDI 000017 stores
  sampling periods as rates);
- a series with several channels.
Positions are never read: no NWB coordinate convention has been verified against
Studio's, so the 3D view says why. Series in the file that no layout maps are listed
as not read.

A layout (configs/nwb/*.yaml) declares a dataset's specifics. Without one (GENERIC),
regions come from units.location or units.electrodes (checked), there is no quality
column (unit QC is then spike times only) and no behaviour.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.analysis.atlas import _check as check_allen
from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    MAX_SESSION_SECONDS,
    TRIAL_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
    TimeSeries,
)
from unitwave.nwb.probe import _all_series, open_nwb

LAYOUTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "nwb"
# Session eid -> the capability report of the file it was read from, for Studio.
REPORTS: dict[str, dict] = {}
LOCATION_KINDS = ("peak_channel", "electrodes", "units_column")
_POSITIONS = "positions aren't read from NWB files: no coordinate convention is verified"


# ---------- layouts: what a dataset's NWB files hold, declared ----------
@dataclass(frozen=True)
class Quality:
    column: str
    pass_at_least: float | None
    pass_values: tuple | None
    names: dict

    def describe(self) -> str:
        rule = (
            f"passes at {self.pass_at_least:g} or more"
            if self.pass_at_least is not None
            else f"passes at {', '.join(map(str, self.pass_values))}"
        )
        named = ", ".join(f"{n} {v:g}" for v, n in sorted(self.names.items()))
        return f"{self.column}: {rule}" + (f" ({named})" if named else "")

    def passes(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if self.pass_at_least is not None:
            return np.asarray(values, np.float64) >= self.pass_at_least
        return np.isin(values, self.pass_values)


@dataclass(frozen=True)
class NwbLayout:
    name: str
    label: str
    task: str | None
    depth_column: str | None
    location: tuple | None  # (kind, options) or None: chosen from the file (GENERIC)
    quality: Quality | None
    behaviour: dict
    positions_reason: str
    raw: dict = field(repr=False, compare=False, default_factory=dict)


GENERIC = NwbLayout("generic", "Generic NWB (no layout)", None, None, None, None, {}, _POSITIONS)


def _need(where: str, raw, keys: set, optional: set = frozenset()) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping")  # noqa: TRY004 - a refusal, in words
    unknown, missing = sorted(set(raw) - keys - set(optional)), sorted(keys - set(raw))
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}")


def layout_from_dict(raw: dict) -> NwbLayout:
    """A validated layout; anything malformed is refused, saying where and why."""
    _need("NWB layout", raw, {"name", "label", "units"}, {"task", "behaviour", "positions"})
    units = raw["units"]
    _need("units", units, set(), {"probe", "depth", "location", "quality"})
    if units.get("probe", "electrode_group") != "electrode_group":
        raise ValueError("units probe: only electrode_group is read")
    depth = None
    if "depth" in units:
        _need("units depth", units["depth"], {"column"})
        depth = str(units["depth"]["column"])
    location = None
    if "location" in units:
        loc = units["location"]
        if not isinstance(loc, dict) or len(loc) != 1 or next(iter(loc)) not in LOCATION_KINDS:
            raise ValueError(f"units location: give one of {', '.join(LOCATION_KINDS)}")
        kind, options = next(iter(loc.items()))
        options = options or {}
        if kind == "peak_channel":
            _need("units location peak_channel", options, {"column", "first"})
        elif kind == "units_column":
            _need("units location units_column", options, {"column"})
        location = (kind, dict(options))
    quality = None
    if "quality" in units:
        q = units["quality"]
        _need("units quality", q, {"column"}, {"pass_at_least", "pass_values", "names"})
        if ("pass_at_least" in q) == ("pass_values" in q):
            raise ValueError("units quality: give either pass_at_least or pass_values")
        quality = Quality(
            column=str(q["column"]),
            pass_at_least=float(q["pass_at_least"]) if "pass_at_least" in q else None,
            pass_values=tuple(q["pass_values"]) if "pass_values" in q else None,
            names={float(k): str(v) for k, v in (q.get("names") or {}).items()},
        )
    behaviour = raw.get("behaviour") or {}
    if not isinstance(behaviour, dict) or not all(isinstance(p, str) for p in behaviour.values()):
        raise ValueError("behaviour: map each name to a path in the file")
    positions = raw.get("positions") or {"refused": _POSITIONS}
    _need("positions", positions, {"refused"})
    return NwbLayout(
        name=str(raw["name"]),
        label=str(raw["label"]),
        task=raw.get("task"),
        depth_column=depth,
        location=location,
        quality=quality,
        behaviour={str(k): str(v) for k, v in behaviour.items()},
        positions_reason=str(positions["refused"]),
        raw=raw,
    )


def list_layouts() -> list[dict]:
    """The built-in layouts: name, label, task and file."""
    rows = []
    for path in sorted(LAYOUTS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        rows.append(
            {
                "name": raw.get("name", path.stem),
                "label": raw.get("label", ""),
                "task": raw.get("task"),
                "file": path.name,
            }
        )
    return rows


def layout_path(name: str | os.PathLike) -> Path:
    path = Path(name)
    if path.suffix not in (".yaml", ".yml"):
        path = LAYOUTS_DIR / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(t["name"] for t in list_layouts())
        raise ValueError(f"no NWB layout {str(name)!r} (built in: {known})")
    return path.resolve()


@lru_cache(maxsize=16)
def _load(path: str, mtime: float) -> NwbLayout:
    return layout_from_dict(yaml.safe_load(Path(path).read_text()) or {})


def load_layout(name: str | os.PathLike | None) -> NwbLayout:
    """A built-in layout by name, one read from a YAML path, or GENERIC for None/"generic"."""
    if name in (None, "", GENERIC.name):
        return GENERIC
    path = layout_path(name)
    return _load(str(path), path.stat().st_mtime)


# ---------- reading ----------
@dataclass(frozen=True)
class Intake:
    """The Session, and the capability report: what was read, refused or not read, and why."""

    session: Session
    report: dict


def _strings(values) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def _ragged_first(column) -> tuple[np.ndarray, np.ndarray]:
    """(flat values, start of each row) of a ragged units column."""
    ends = np.asarray(column.data[:], dtype=np.int64)
    return np.asarray(column.target.data[:]), np.concatenate([[0], ends[:-1]])


def _electrode_groups(nwb) -> np.ndarray:
    table = nwb.electrodes
    if "group_name" in table.colnames:
        return _strings(table["group_name"].data[:])
    return _strings([g.name for g in table["group"].data[:]])


def _locations(nwb, units, ids, probes, layout: NwbLayout) -> tuple[np.ndarray | None, str, str]:
    """(per-unit location names or None, where they came from, why not if None)."""
    kind, options = layout.location or (None, {})
    names = units.colnames
    if kind is None:  # GENERIC: the file's standard columns
        if "location" in names:
            kind, options = "units_column", {"column": "location"}
        elif "electrodes" in names and nwb.electrodes is not None:
            kind = "electrodes"
        else:
            return None, "", "the units have no location and no electrodes"
    if kind == "units_column":
        column = options["column"]
        if column not in names:
            return None, "", f"the units have no {column} column"
        return _strings(units[column].data[:]), f"units.{column}", ""
    if nwb.electrodes is None or "location" not in nwb.electrodes.colnames:
        return None, "", "the file has no electrodes table with a location column"
    where = _strings(nwb.electrodes["location"].data[:])
    groups = _electrode_groups(nwb)
    if kind == "electrodes":
        flat, starts = _ragged_first(units["electrodes"])
        rows = flat[starts].astype(np.int64)
        source, how = "electrodes.location via units.electrodes", "units.electrodes points to"
    else:
        column = options["column"]
        if column not in names:
            return None, "", f"the units have no {column} column"
        rows = np.asarray(units[column].data[:]).ravel().astype(np.int64) - int(options["first"])
        source = f"electrodes.location at {column} - {int(options['first'])}"
        how = f"{column} - {int(options['first'])} gives"
    inside = (rows >= 0) & (rows < len(where))
    own = inside & (groups[np.clip(rows, 0, len(where) - 1)] == probes)
    if not own.all():
        bad = [ids[i] for i in np.flatnonzero(~own)]
        n = len(bad)
        return (
            None,
            source,
            (
                f"{how} electrodes of another probe for {n} unit{'' if n == 1 else 's'} "
                f"({', '.join(bad[:5])}{', …' if n > 5 else ''}): regions aren't read"
            ),
        )
    return where[rows], source, ""


def _series_at(nwb, path: str):
    parts = path.split("/")
    try:
        if parts[0] == "acquisition" and len(parts) == 2:
            return nwb.acquisition[parts[1]]
        if parts[0] == "processing" and len(parts) in (3, 4):
            obj = nwb.processing[parts[1]][parts[2]]
            return obj[parts[3]] if len(parts) == 4 else obj
    except KeyError:
        return None
    return None


def _behaviour(nwb, layout: NwbLayout) -> tuple[dict, dict, dict]:
    """(loaded TimeSeries, report entry per name, reason per name not loaded)."""
    loaded, report, why_not = {}, {}, {}
    for name, path in layout.behaviour.items():
        obj = _series_at(nwb, path)
        shape = tuple(getattr(getattr(obj, "data", None), "shape", ()) or ())
        n = shape[0] if shape else 0
        rate = getattr(obj, "rate", None)
        if obj is None or not shape:
            why_not[name] = f"{path}: not in this file"
        elif len(shape) > 1 and shape[1] > 1:
            why_not[name] = f"{path}: {shape[1]} channels; only single-channel series are read"
        elif obj.timestamps is None and not (rate is not None and rate > 0):
            why_not[name] = f"{path}: neither timestamps nor a rate"
        elif obj.timestamps is None and n / rate > MAX_SESSION_SECONDS:
            why_not[name] = (
                f"{path}: {rate:g} Hz × {n:,} samples spans {n / rate:,.0f} s, longer than "
                "24 h: the stored rate may be a sampling period. Refused, not corrected"
            )
        else:
            if obj.timestamps is not None:
                ts = np.asarray(obj.timestamps[:], dtype=np.float64)
            else:
                ts = float(obj.starting_time or 0.0) + np.arange(n) / float(rate)
            data = np.asarray(obj.data[:], dtype=np.float64).ravel()
            values = data * float(obj.conversion) + float(getattr(obj, "offset", 0.0))
            loaded[name] = TimeSeries(ts, values)
            how = "timestamps" if obj.timestamps is not None else f"{rate:g} Hz"
            report[name] = {"loaded": f"{path} ({n:,} samples, {how}, {obj.unit})"}
        if name in why_not:
            report[name] = {"refused": why_not[name]}
    return loaded, report, why_not


def _trials(nwb) -> tuple[pd.DataFrame, list[str]]:
    """The trials table, scalar columns only; (table, columns skipped)."""
    table, skipped = {}, []
    for name in nwb.trials.colnames:
        column = nwb.trials[name]
        if hasattr(column, "target"):  # ragged: not a value per trial
            skipped.append(name)
            continue
        values = np.asarray(column.data[:])
        if values.ndim != 1:
            skipped.append(name)
        elif values.dtype == bool or np.issubdtype(values.dtype, np.number):
            table[name] = values.astype(np.float64)
        else:
            table[name] = _strings(values)
    trials = pd.DataFrame(table).rename(
        columns={"start_time": "intervals_0", "stop_time": "intervals_1"}
    )
    return trials, skipped


def _time_bounds(spikes: dict, trials: pd.DataFrame, behaviour: dict) -> tuple[float, float]:
    firsts = [t[0] for t in spikes.values() if t.size]
    lasts = [t[-1] for t in spikes.values() if t.size]
    intervals = trials[["intervals_0", "intervals_1"]].to_numpy(np.float64)
    firsts.append(np.nanmin(intervals))
    lasts.append(np.nanmax(intervals))
    for series in behaviour.values():
        firsts.append(series.timestamps[0])
        lasts.append(series.timestamps[-1])
    return float(min(firsts)), float(max(lasts))


def read_nwb(path: str | os.PathLike, layout: NwbLayout = GENERIC) -> Intake:
    """The Session of one NWB file under a layout, and its capability report."""
    path = Path(path)
    with open_nwb(str(path)) as nwb:
        units = nwb.units
        if units is None or "spike_times" not in units.colnames:
            raise ValueError(
                f"{path.name}: no spike-sorted units (no units table with spike times). "
                "Studio needs them"
            )
        if nwb.trials is None:
            raise ValueError(
                f"{path.name}: no trials table. Studio needs each trial's start and end "
                "(the task period)"
            )
        flat, starts = _ragged_first(units["spike_times"])
        ends = np.append(starts[1:], flat.size)
        if "electrode_group" in units.colnames:
            probes = _strings([g.name for g in units["electrode_group"].data[:]])
        else:
            probes = np.full(len(units), "units", dtype=object)
        ids = [f"{p}_{i}" for p, i in zip(probes, units.id[:])]
        spikes = {u: np.sort(flat[a:b].astype(np.float64)) for u, a, b in zip(ids, starts, ends)}
        table = pd.DataFrame({"probe_name": probes}, index=pd.Index(ids, name="unit_id"))
        missing = {}
        if layout.depth_column:
            table["depths"] = np.asarray(units[layout.depth_column].data[:], np.float64)
        else:
            missing["units.depths"] = f"the {layout.label} layout declares no depth column"
        if layout.quality is not None:
            table["quality"] = np.asarray(units[layout.quality.column].data[:]).ravel()
        locations, source, why = _locations(nwb, units, ids, probes, layout)
        trials, skipped = _trials(nwb)
        behaviour, behaviour_report, behaviour_why = _behaviour(nwb, layout)
        mapped = set(layout.behaviour.values())
        not_read = sorted(p for p in _all_series(nwb) if p not in mapped)

    regions: dict = {}
    if locations is None:
        missing["units.acronym"] = why
        regions = {"refused": why}
    else:
        table["location"] = locations
        try:
            check_allen(locations)
        except ValueError as e:
            missing["units.acronym"] = f"{e}; the names are kept as 'location'"
            regions = {"source": source, "allen": False, "refused": missing["units.acronym"]}
        else:
            table["acronym"] = locations
            regions = {"source": source, "allen": True}
    for f in ("x", "y", "z"):
        missing[f"units.{f}"] = layout.positions_reason
    missing["units.label"] = "IBL's numeric QC label; a declared NWB quality column is 'quality'"
    missing["units.firing_rate"] = (
        "the recording's length isn't stored; QC uses the task-period rate"
    )
    for f in TRIAL_FIELDS:
        if f not in trials:
            missing[f"trials.{f}"] = "not in this file's trials (the task definition reads its own)"
    for f in BEHAVIOUR_FIELDS:
        if f not in behaviour:
            missing[f"behaviour.{f}"] = behaviour_why.get(
                f, f"not mapped by the {layout.label} layout"
            )
    present = {f"units.{f}" for f in UNIT_FIELDS if f in table}
    present |= {f"trials.{f}" for f in TRIAL_FIELDS if f in trials}
    present |= {f"behaviour.{f}" for f in BEHAVIOUR_FIELDS if f in behaviour}
    session = Session(
        eid=f"nwb:{path.resolve()}",
        time_bounds=_time_bounds(spikes, trials, behaviour),
        spikes=spikes,
        units=table,
        trials=trials,
        behaviour=behaviour,
        available=Capabilities(present=frozenset(present), missing=missing),
    )
    report = {
        "file": str(path),
        "layout": layout.name,
        "units": {
            "n": len(table),
            "probes": sorted(set(probes)),
            "depth": layout.depth_column,
            "quality": layout.quality.describe() if layout.quality else None,
        },
        "regions": regions,
        "positions": {"refused": layout.positions_reason},
        "trials": {"n": len(trials), "columns": list(trials.columns), "not_read": skipped},
        "behaviour": behaviour_report,
        "not_read": not_read,
    }
    return Intake(session, report)
