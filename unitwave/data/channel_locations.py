"""Histology-aligned channel locations for a Phy folder (step 13b; docs/DECISIONS.md,
"Step 13b").

Two files are read (`read_channel_locations`), both giving each channel a region and a
position, returned in IBL's convention (metres from bregma: x = ML right +, y = AP
anterior +, z = DV dorsal +), as the BWM backend's units are:
- **The IBL alignment GUI's `channel_locations.json`:** one `channel_N` entry per channel
  with x, y, z (µm, IBL's axes), its position on the probe (lateral, axial µm) and its
  region (brain_region, and brain_region_id, which must name the same region).
- **A CSV:** channel, ccf_ap_um, ccf_dv_um, ccf_ml_um (Allen CCF µm) and acronym;
  optionally lateral_um and axial_um.
Every acronym must be an Allen CCF 2017 one.

A unit takes its peak channel's region and position (`assign_units`). Its channel is
matched by its position on the probe (lateral, axial) when the file gives positions, so
the file's channel numbering doesn't matter; otherwise by channel number (Phy's
channel_map.npy, else the row). A unit whose channel isn't in the file is refused.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM

from unitwave.analysis.atlas import region_info

_KEY = re.compile(r"^channel_(\d+)$")
_CSV = ("channel", "ccf_ap_um", "ccf_dv_um", "ccf_ml_um", "acronym")
_COLUMNS = ["lateral_um", "axial_um", "x", "y", "z", "acronym"]


def _from_ccf(ap, dv, ml) -> np.ndarray:
    """(n, 3) IBL xyz metres from Allen CCF µm: the inverse of analysis.atlas.ccf_um."""
    ml0, ap0, dv0 = ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"]
    return np.column_stack(
        [(np.asarray(ml) - ml0) / 1e6, (ap0 - np.asarray(ap)) / 1e6, (dv0 - np.asarray(dv)) / 1e6]
    )


def _read_json(path: Path) -> pd.DataFrame:
    from iblatlas.regions import BrainRegions

    raw = json.loads(path.read_text())
    rows, ids = {}, BrainRegions()
    for key, entry in raw.items():
        match = _KEY.match(key)
        if match is None:
            continue  # e.g. "origin": where the alignment came from
        missing = [k for k in ("x", "y", "z", "brain_region") if k not in entry]
        if missing:
            raise ValueError(f"{path.name}, {key}: missing {missing}")
        acronym = str(entry["brain_region"])
        if "brain_region_id" in entry:
            named = str(ids.id2acronym([int(entry["brain_region_id"])])[0])
            if named != acronym:
                raise ValueError(
                    f"{path.name}, {key}: brain_region {acronym!r} but brain_region_id "
                    f"{int(entry['brain_region_id'])} is {named}"
                )
        rows[int(match.group(1))] = {
            "lateral_um": float(entry.get("lateral", np.nan)),
            "axial_um": float(entry.get("axial", np.nan)),
            "x": float(entry["x"]) / 1e6,
            "y": float(entry["y"]) / 1e6,
            "z": float(entry["z"]) / 1e6,
            "acronym": acronym,
        }
    if not rows:
        raise ValueError(f"{path.name}: no channel_N entries")
    return pd.DataFrame.from_dict(rows, orient="index")[_COLUMNS].sort_index()


def _read_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = [c for c in _CSV if c not in raw]
    if missing:
        raise ValueError(f"{path.name}: needs columns {list(_CSV)}; missing {missing}")
    if raw["channel"].duplicated().any():
        raise ValueError(f"{path.name}: a channel is listed twice")
    xyz = _from_ccf(raw["ccf_ap_um"], raw["ccf_dv_um"], raw["ccf_ml_um"])
    table = pd.DataFrame(
        {
            "lateral_um": raw.get("lateral_um", np.nan),
            "axial_um": raw.get("axial_um", np.nan),
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "acronym": raw["acronym"].astype(str),
        }
    )
    table.index = raw["channel"].astype(int).to_numpy()
    return table.sort_index()


def read_channel_locations(path: str | os.PathLike) -> pd.DataFrame:
    """(n_channels, 6) lateral_um, axial_um (NaN when not given), x, y, z (IBL metres),
    acronym; indexed by the file's channel number."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"no channel locations file {path}")
    if path.suffix == ".json":
        table = _read_json(path)
    elif path.suffix == ".csv":
        table = _read_csv(path)
    else:
        raise ValueError(f"{path.name}: give the alignment GUI's .json or a .csv")
    region_info(table["acronym"].unique())  # refuses any acronym outside Allen CCF 2017
    if not np.isfinite(table[["x", "y", "z"]].to_numpy()).all():
        raise ValueError(f"{path.name}: every channel needs a finite position")
    return table


def _where(key, by_position: bool) -> str:
    """A channel in words, for a refusal."""
    return f"lateral {key[0]:g}, axial {key[1]:g} µm" if by_position else f"channel {key}"


def assign_units(
    table: pd.DataFrame,
    positions: np.ndarray,
    unit_rows: np.ndarray,
    channel_map: np.ndarray | None,
    source: str,
) -> pd.DataFrame:
    """(n_units, 4) acronym, x, y, z of each unit's peak channel.

    positions: (n_channels, 2) channel_positions.npy (lateral, axial µm); unit_rows:
    (n_units,) each unit's peak channel, a row of positions; channel_map: (n_channels,)
    each row's channel number, or None (the row is the number)."""
    positions = np.asarray(positions, np.float64)
    unit_rows = np.asarray(unit_rows)
    assert positions.ndim == 2 and positions.shape[1] == 2 and unit_rows.ndim == 1
    by_position = table[["lateral_um", "axial_um"]].notna().all(axis=1).all()
    if by_position:
        keys = list(zip(table["lateral_um"].round(1), table["axial_um"].round(1)))
        if len(set(keys)) != len(keys):
            raise ValueError(f"{source}: two channels share a position on the probe")
        index = dict(zip(keys, range(len(table))))
        wanted = [(round(lat, 1), round(ax, 1)) for lat, ax in positions[unit_rows]]
    else:
        numbers = channel_map if channel_map is not None else np.arange(len(positions))
        index = {int(c): i for i, c in enumerate(table.index)}
        wanted = [int(c) for c in numbers[unit_rows]]
    found = [index.get(k) for k in wanted]
    lost = [k for k, f in zip(wanted, found) if f is None]
    if lost:
        one = len(lost) == 1
        units, channels, verb = (
            ("unit's", "channel", "is") if one else ("units'", "channels", "are")
        )
        first = "" if one else ", first"
        raise ValueError(
            f"{len(lost)} {units} peak {channels} ({_where(lost[0], by_position)}{first}) "
            f"{verb} not in {source}: is it this probe's alignment?"
        )
    rows = table.iloc[found]
    return pd.DataFrame(
        {k: rows[k].to_numpy() for k in ("acronym", "x", "y", "z")},
    )
