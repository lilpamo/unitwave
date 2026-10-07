"""Probe an arbitrary NWB file: report what's in it, and what our pipeline could use.

    python -m unitwave.nwb.probe <path or https URL> [...] [--json report.json]

Thin by design (ROADMAP Phase 3): no normalisation, no model, no Session. Files are
opened locally or streamed with remfile, and only metadata plus small samples are
read (never a whole spike_times or raw-data array). Each section is probed on its own;
anything that can't be read is recorded under "errors", never raised.

Reported: units (count, columns, spike count, a QC-label column, per-unit location),
electrodes (count, a region column), trials and other intervals, every time series in
acquisition and processing (shape, unit, rate, duration), and how names map onto the
pipeline's behaviour fields. `usable` summarises what the pipeline could take.
"""

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import remfile
from pynwb import NWBHDF5IO

# A units column whose name matches this carries a QC label or score (e.g. quality,
# KSLabel, IBL's ibl_quality_score and kilosort2_label). Location columns, exactly.
QC_PATTERN = r"label|quality"
LOCATION_COLUMNS = ("location", "brain_area", "brain_region", "acronym", "region", "area")
# Our behaviour fields (data.session) and name fragments that point to them.
BEHAVIOUR_PATTERNS = {
    "wheel": r"wheel",
    "lick": r"lick",
    "pupil": r"pupil",
    "pose": r"pose|dlc|keypoint|lightning",
    "motion_energy": r"motion.?energy|whisker",
}
_SAMPLE = 1000


@contextlib.contextmanager
def open_nwb(source):
    """Yield the NWBFile at a local path, or streamed from an http(s) URL."""
    with contextlib.ExitStack() as stack:
        if isinstance(source, str) and source.startswith(("https://", "http://")):
            remote = stack.enter_context(contextlib.closing(remfile.File(source)))
            h5 = stack.enter_context(h5py.File(remote, "r"))
            io = stack.enter_context(NWBHDF5IO(file=h5, mode="r", load_namespaces=True))
        else:
            io = stack.enter_context(NWBHDF5IO(str(source), "r", load_namespaces=True))
        yield io.read()


def _error(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:200]}"


def _values(column, limit=None) -> np.ndarray:
    data = column.data if hasattr(column, "data") else column
    return np.asarray(data[:limit] if limit else data[:])


def _find(names, candidates) -> str | None:
    lowered = {n.lower(): n for n in names}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None


def _distinct(values: np.ndarray) -> dict:
    values = np.array(
        [v.decode() if isinstance(v, bytes) else v for v in values.ravel()], dtype=object
    )
    names, counts = np.unique(values.astype(str), return_counts=True)
    order = np.argsort(-counts)
    return {
        "n_distinct": int(len(names)),
        "top": {str(names[i]): int(counts[i]) for i in order[:8]},
    }


def _units(nwb) -> dict:
    units = nwb.units
    if units is None:
        return {"present": False}
    out = {"present": True, "n_units": len(units), "columns": list(units.colnames)}
    if "spike_times" in units.colnames:
        column = units["spike_times"]
        # A ragged column comes back as its index (one entry per unit); the flat spike
        # times are the index's target.
        data = column.target.data if hasattr(column, "target") else column.data
        out["n_spikes"] = int(len(data))
        head, tail = _values(data, _SAMPLE), np.asarray(data[-_SAMPLE:])
        out["spike_time_sample_range_s"] = (
            [float(head.min()), float(tail.max())] if len(head) else None
        )
    qc = [c for c in units.colnames if re.search(QC_PATTERN, c, flags=re.IGNORECASE)]
    if qc:
        out["qc_columns"] = {c: _distinct(_values(units[c])) for c in qc}
    location = _find(units.colnames, LOCATION_COLUMNS)
    if location:
        out["location"] = {"source": f"units.{location}", **_distinct(_values(units[location]))}
    elif "electrodes" in units.colnames and nwb.electrodes is not None:
        column = _find(nwb.electrodes.colnames, LOCATION_COLUMNS)
        if column:
            per_unit = [row[column].iloc[0] for row in units["electrodes"][:]]
            out["location"] = {
                "source": f"electrodes.{column} via units.electrodes",
                **_distinct(np.array(per_unit)),
            }
    return out


def _electrodes(nwb) -> dict:
    table = nwb.electrodes
    if table is None:
        return {"present": False}
    out = {"present": True, "n": len(table), "columns": list(table.colnames)}
    column = _find(table.colnames, LOCATION_COLUMNS)
    if column:
        out["location"] = {"column": column, **_distinct(_values(table[column]))}
    return out


def _intervals(nwb) -> dict:
    out = {}
    tables = dict(nwb.intervals) if nwb.intervals else {}
    if nwb.trials is not None:
        tables["trials"] = nwb.trials
    for name, table in tables.items():
        entry = {"n": len(table), "columns": list(table.colnames)}
        if len(table) and "start_time" in table.colnames:
            entry["range_s"] = [
                float(_values(table["start_time"])[0]),
                float(_values(table["stop_time"])[-1]),
            ]
        out[name] = entry
    return out


def _series(obj, path: str, found: dict) -> None:
    """Record every TimeSeries under obj (containers are walked one level down)."""
    if hasattr(obj, "data") and hasattr(obj, "rate"):
        data = obj.data
        entry = {
            "type": type(obj).__name__,
            "shape": list(getattr(data, "shape", ()) or ()),
            "unit": getattr(obj, "unit", None),
        }
        if obj.rate is not None:
            entry["rate_hz"] = float(obj.rate)
            if entry["shape"]:
                entry["duration_s"] = entry["shape"][0] / float(obj.rate)
        elif obj.timestamps is not None:
            ts = obj.timestamps
            head = _values(ts, _SAMPLE)
            if len(head) > 1:
                entry["rate_hz"] = float(1.0 / np.median(np.diff(head)))
                entry["duration_s"] = float(np.asarray(ts[-1:])[0] - head[0])
        found[path] = entry
        return
    children = getattr(obj, "children", None) or []
    for child in children:
        name = getattr(child, "name", "?")
        if hasattr(child, "data") or getattr(child, "children", None):
            try:
                _series(child, f"{path}/{name}", found)
            except Exception as e:  # noqa: BLE001 - recorded, never raised
                found[f"{path}/{name}"] = {"error": _error(e)}


def _all_series(nwb) -> dict:
    found: dict = {}
    for name, obj in nwb.acquisition.items():
        _series(obj, f"acquisition/{name}", found)
    for module_name, module in nwb.processing.items():
        for name, obj in module.data_interfaces.items():
            _series(obj, f"processing/{module_name}/{name}", found)
    return found


def _behaviour_map(series: dict) -> dict:
    mapped: dict = {}
    for path in series:
        for field, pattern in BEHAVIOUR_PATTERNS.items():
            if re.search(pattern, path, flags=re.IGNORECASE):
                mapped.setdefault(field, []).append(path)
    return mapped


def probe(source) -> dict:
    """What the NWB file at `source` holds; sections that fail are listed under errors."""
    report: dict = {"source": str(source), "errors": {}}
    try:
        with open_nwb(source) as nwb:
            report["opened"] = True
            report["file"] = {
                "identifier": nwb.identifier,
                "session_description": (nwb.session_description or "")[:200],
                "session_start_time": str(nwb.session_start_time),
            }
            sections = {
                "units": _units,
                "electrodes": _electrodes,
                "intervals": _intervals,
                "series": _all_series,
            }
            for name, reader in sections.items():
                try:
                    report[name] = reader(nwb)
                except Exception as e:  # noqa: BLE001 - recorded, never raised
                    report["errors"][name] = _error(e)
    except Exception as e:  # noqa: BLE001 - an unreadable file is a result, not a crash
        report["opened"] = False
        report["errors"]["open"] = _error(e)
        return report
    report["behaviour"] = _behaviour_map(report.get("series", {}))
    report["usable"] = _usable(report)
    return report


def _usable(report: dict) -> dict:
    units = report.get("units", {})
    spikes = bool(units.get("present") and units.get("n_spikes"))
    usable = {
        "spikes": spikes,
        "unit_qc_label": bool(units.get("qc_columns")),
        "unit_location": bool(units.get("location")),
        "trials": "trials" in report.get("intervals", {}),
        "behaviour": sorted(report.get("behaviour", {})),
    }
    if not spikes:
        tier = "no spike-sorted units"
    elif usable["unit_location"] and usable["trials"] and usable["behaviour"]:
        tier = "decodable: spikes, locations, trials and behaviour"
    elif usable["trials"] or usable["behaviour"]:
        tier = "partial: spikes with some task or behaviour data"
    else:
        tier = "spikes only"
    usable["tier"] = tier
    return usable


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report what NWB files contain.")
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--json", type=Path, help="write every report to this file")
    args = parser.parse_args(argv)
    reports = [probe(s) for s in args.sources]
    for r in reports:
        usable = r.get("usable", {})
        print(f"{r['source']}\n  {usable.get('tier', 'unreadable')}; errors: {sorted(r['errors'])}")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
