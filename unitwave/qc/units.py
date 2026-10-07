"""Unit QC: which units are usable, with a reason for every exclusion.

A unit passes when its IBL QC label is at least `min_label`, it is not located in an
excluded region, and it fires at least `min_firing_rate_hz` during the task: from the
first trial's start to the last trial's end, both included. Recordings run before
and after the task, and some units fire almost only then (docs/DECISIONS.md). The
whole-recording `firing_rate` stays in the units table as a reported column; QC
reads `task_firing_rate`. A missing label, rate or location fails the unit rather
than passing it. Thresholds live in configs/qc.yaml.

`task_period` and `in_task` are the one definition of the task period. A session
gets its rates from its own spikes (`task_firing_rates`); the release-wide table in
qc/task_rates.py uses the same two functions.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.data.session import Capabilities, Session

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qc.yaml"
TASK_RATE = "task_firing_rate"
_KEYS = {"min_label", "exclude_regions", "min_firing_rate_hz"}
# IBL/Allen atlas ids for the two non-regions, for backends that give ids, not acronyms.
_ATLAS_ID_NAMES = {0: "void", 997: "root"}


@dataclass(frozen=True)
class UnitQC:
    min_label: float
    exclude_regions: tuple[str, ...]
    min_firing_rate_hz: float

    def hash(self) -> str:
        """sha256 of the thresholds; changes whenever any threshold does."""
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_qc_config(path: str | os.PathLike = DEFAULT_CONFIG) -> UnitQC:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return UnitQC(
        min_label=float(raw["min_label"]),
        exclude_regions=tuple(str(r) for r in raw["exclude_regions"]),
        min_firing_rate_hz=float(raw["min_firing_rate_hz"]),
    )


def task_period(trials: pd.DataFrame) -> tuple[float, float]:
    """(first trial's intervals_0, last trial's intervals_1), seconds."""
    start = trials["intervals_0"].to_numpy(np.float64)
    stop = trials["intervals_1"].to_numpy(np.float64)
    if not start.size or not (np.all(np.isfinite(start)) and np.all(np.isfinite(stop))):
        raise ValueError("the task period needs every trial's intervals_0 and intervals_1")
    t0, t1 = float(start.min()), float(stop.max())
    if not t1 > t0:
        raise ValueError(f"empty task period [{t0}, {t1}]")
    return t0, t1


def in_task(times: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """(n_spikes,) bool: inside the task period [t0, t1], both ends included."""
    return (times >= t0) & (times <= t1)


def task_firing_rates(session: Session) -> pd.Series:
    """(n_units,) Hz: each unit's spikes inside the task period / its length."""
    t0, t1 = task_period(session.trials)
    counts = [np.count_nonzero(in_task(session.spikes[u], t0, t1)) for u in session.units.index]
    return pd.Series(np.asarray(counts, np.int64) / (t1 - t0), session.units.index, name=TASK_RATE)


def _regions(units: pd.DataFrame) -> pd.Series:
    """Each unit's region name, from whichever location field its backend provides.

    BWM gives `acronym`, ONE gives Allen `atlas_id`s, and NWB gives full region names in
    `location`; the non-regions void and root are spelled the same in all three.
    """
    if "acronym" in units:
        return units["acronym"]
    if "atlas_id" in units:
        ids = units["atlas_id"]
        return ids.map(lambda i: None if pd.isna(i) else _ATLAS_ID_NAMES.get(int(i), str(int(i))))
    if "location" in units:
        return units["location"]
    raise ValueError("units have no location field (acronym, atlas_id or location)")


def unit_qc(units: pd.DataFrame, qc: UnitQC) -> pd.DataFrame:
    """(n_units, 2): `passed` and `reason` ("" when passed, else every failed criterion).

    units needs `task_firing_rate`: from `task_firing_rates` for a session, or from
    qc.task_rates.release_units for the whole release.
    """
    if TASK_RATE not in units:
        raise ValueError(f"units have no {TASK_RATE} column; QC uses the task-period rate")
    label = units["label"].to_numpy(np.float64)
    rate = units[TASK_RATE].to_numpy(np.float64)
    region = _regions(units)
    excluded = {r.lower() for r in qc.exclude_regions}

    reasons = []
    for lab, fr, reg in zip(label, rate, region):
        why = []
        if np.isnan(lab):
            why.append("label missing")
        elif lab < qc.min_label:
            why.append(f"label {lab:.4g} < {qc.min_label:g}")
        if reg is None or (isinstance(reg, float) and np.isnan(reg)):
            why.append("location missing")
        elif str(reg).lower() in excluded:
            why.append(f"located in {reg}")
        if np.isnan(fr):
            why.append("task firing rate missing")
        elif fr < qc.min_firing_rate_hz:
            why.append(f"task firing rate {fr:.3g} Hz < {qc.min_firing_rate_hz:g} Hz")
        reasons.append("; ".join(why))
    result = pd.DataFrame({"reason": reasons}, index=units.index)
    result.insert(0, "passed", result["reason"] == "")
    return result


def apply_unit_qc(session: Session, qc: UnitQC) -> Session:
    """The same session with only the units that pass QC (and their spikes).

    Its units table gains the `task_firing_rate` column QC used.
    """
    units = session.units.assign(**{TASK_RATE: task_firing_rates(session)})
    passed = unit_qc(units, qc)["passed"]
    keep = passed.index[passed.to_numpy()]
    if len(keep) == 0:
        raise ValueError(f"{session.eid}: no units pass QC")
    return Session(
        eid=session.eid,
        time_bounds=session.time_bounds,
        spikes={u: session.spikes[u] for u in keep},
        units=units.loc[keep],
        trials=session.trials,
        behaviour=dict(session.behaviour),
        available=Capabilities(
            present=session.available.present, missing=dict(session.available.missing)
        ),
    )
