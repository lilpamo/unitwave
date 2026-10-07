"""Unit QC for Phy/Kilosort data, with a reason for every exclusion.

Phy folders have no IBL QC label and no brain region, so qc.units.unit_qc would fail
every unit. A Phy unit passes when:
- its group (from cluster_group.tsv or cluster_KSLabel.tsv, see data/backends/phy.py)
  is one of `groups`;
- it passes IBL's sliding refractory-period test (qc/refractory.py): contamination at
  most `refractory_contamination` of its rate, shown with confidence 1 - `refractory_alpha`;
- it fires at least `min_firing_rate_hz` during the task (qc.units' task period).
A missing group or rate fails the unit. Thresholds live in configs/qc_phy.yaml.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.data.session import Session
from unitwave.qc.refractory import sliding_rp_pass
from unitwave.qc.units import TASK_RATE

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qc_phy.yaml"
_KEYS = {"groups", "min_firing_rate_hz", "refractory_contamination", "refractory_alpha"}
REFRACTORY = "sliding_rp_pass"


@dataclass(frozen=True)
class PhyUnitQC:
    groups: tuple[str, ...]
    min_firing_rate_hz: float
    refractory_contamination: float
    refractory_alpha: float

    def hash(self) -> str:
        """sha256 of the thresholds; changes whenever any threshold does."""
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_phy_qc_config(path: str | os.PathLike = DEFAULT_CONFIG) -> PhyUnitQC:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return PhyUnitQC(
        groups=tuple(str(g) for g in raw["groups"]),
        min_firing_rate_hz=float(raw["min_firing_rate_hz"]),
        refractory_contamination=float(raw["refractory_contamination"]),
        refractory_alpha=float(raw["refractory_alpha"]),
    )


def refractory_passes(session: Session, qc: PhyUnitQC) -> pd.Series:
    """(n_units,) bool: each unit's result on the sliding refractory-period test."""
    passes = [
        sliding_rp_pass(session.spikes[u], qc.refractory_contamination, qc.refractory_alpha)
        for u in session.units.index
    ]
    return pd.Series(passes, session.units.index, name=REFRACTORY)


def phy_unit_qc(units: pd.DataFrame, qc: PhyUnitQC) -> pd.DataFrame:
    """(n_units, 2): `passed` and `reason` ("" when passed, else every failed criterion).

    units needs `phy_group`, `task_firing_rate` (qc.units.task_firing_rates) and
    `sliding_rp_pass` (refractory_passes).
    """
    for column in ("phy_group", TASK_RATE, REFRACTORY):
        if column not in units:
            raise ValueError(f"units have no {column} column")
    reasons = []
    rows = zip(units["phy_group"], units[REFRACTORY], units[TASK_RATE].to_numpy(np.float64))
    for group, refractory_ok, rate in rows:
        why = []
        if not isinstance(group, str):
            why.append("group missing")
        elif group not in qc.groups:
            why.append(f"group {group} not in {list(qc.groups)}")
        if not refractory_ok:
            why.append(
                f"refractory violations: contamination below {qc.refractory_contamination:g} "
                f"not shown at {1 - qc.refractory_alpha:g} confidence"
            )
        if np.isnan(rate):
            why.append("task firing rate missing")
        elif rate < qc.min_firing_rate_hz:
            why.append(f"task firing rate {rate:.3g} Hz < {qc.min_firing_rate_hz:g} Hz")
        reasons.append("; ".join(why))
    result = pd.DataFrame({"reason": reasons}, index=units.index)
    result.insert(0, "passed", result["reason"] == "")
    return result
