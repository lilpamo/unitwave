"""Unit QC for NWB files whose layout declares a quality column, with a reason for
every exclusion.

A unit passes when its value in the layout's quality column passes the layout's rule
(Steinmetz 2019: phy_annotations at 2 or more, the file's own guidance) and it fires at
least `min_firing_rate_hz` during the task (qc.units' task period). A missing value or
rate fails the unit. The rate threshold lives in configs/qc_nwb.yaml; the column and
its rule in the layout (configs/nwb/*.yaml).
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.nwb.intake import Quality
from unitwave.qc.units import TASK_RATE

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qc_nwb.yaml"
_KEYS = {"min_firing_rate_hz"}


@dataclass(frozen=True)
class NwbUnitQC:
    quality: Quality
    min_firing_rate_hz: float

    def hash(self) -> str:
        """sha256 of the rule and threshold; changes whenever either does."""
        text = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @property
    def rule(self) -> str:
        return f"NWB quality ({self.quality.column})"


def load_nwb_qc_config(quality: Quality, path: str | os.PathLike = DEFAULT_CONFIG) -> NwbUnitQC:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return NwbUnitQC(quality, float(raw["min_firing_rate_hz"]))


def quality_names(values, quality: Quality) -> pd.Series:
    """Each unit's quality value, named by the layout where it names it."""
    values = pd.Series(values)
    return values.map(lambda v: quality.names.get(float(v), str(v)) if pd.notna(v) else None)


def nwb_unit_qc(units: pd.DataFrame, qc: NwbUnitQC) -> pd.DataFrame:
    """(n_units, 2): `passed` and `reason` ("" when passed, else every failed criterion).

    units needs `quality` (the layout's column, as nwb.intake reads it) and
    `task_firing_rate` (qc.units.task_firing_rates).
    """
    for column in ("quality", TASK_RATE):
        if column not in units:
            raise ValueError(f"units have no {column} column")
    q = qc.quality
    values = units["quality"].to_numpy()
    passes = q.passes(values)
    reasons = []
    for value, ok, rate in zip(values, passes, units[TASK_RATE].to_numpy(np.float64)):
        why = []
        if pd.isna(value):
            why.append(f"{q.column} missing")
        elif not ok:
            name = q.names.get(float(value))
            shown = f"{value:g}" if isinstance(value, float | int | np.number) else str(value)
            rule = (
                f"< {q.pass_at_least:g}"
                if q.pass_at_least is not None
                else f"not in {list(q.pass_values)}"
            )
            why.append(f"{q.column} {shown}{f' ({name})' if name else ''} {rule}")
        if np.isnan(rate):
            why.append("task firing rate missing")
        elif rate < qc.min_firing_rate_hz:
            why.append(f"task firing rate {rate:.3g} Hz < {qc.min_firing_rate_hz:g} Hz")
        reasons.append("; ".join(why))
    result = pd.DataFrame({"reason": reasons}, index=units.index)
    result.insert(0, "passed", result["reason"] == "")
    return result
