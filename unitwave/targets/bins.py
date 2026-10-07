"""Per-bin targets on the spike-bin grid: wheel velocity and movement state.  [R6]

Both come out as (n_bins,) float32 aligned with a BinnedSpikes (bin k covers
[k * w, (k + 1) * w)), NaN wherever the target is undefined. Built from the BWM
release (docs/DECISIONS.md, "Targets").

- wheel_velocity: (position at the bin's end - position at its start) / bin width,
  in rad/s, with the sign of the wheel position. Positions are linearly interpolated
  at bin edges and never extrapolated. No filter: it is the exact mean velocity over
  the bin and uses no sample from after it.
- movement_state: 1 for a bin wholly inside a wheel movement, 0 for a bin wholly
  inside a quiescent period, NaN otherwise. The epochs are IBL's detector output
  (ibllib extract_wheel_moves, quiescence >= 0.2 s) as shipped in bwm_behavior.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from unitwave.data.backends.bwm_compressed import (
    BEHAVIOUR_DATASET_NAME,
    BEHAVIOUR_DATASET_VERSION,
)
from unitwave.data.session import Session
from unitwave.preprocess.binning import BinnedSpikes
from unitwave.targets.config import TargetConfig


@dataclass(frozen=True)
class BinTarget:
    """values: (n_bins,) float32; values[j] is bin first_bin + j, NaN where undefined."""

    eid: str
    name: str
    values: np.ndarray
    bin_ms: int
    first_bin: int
    preproc_fingerprint: str | None
    targets_fingerprint: str


def check_grid(session_eid: str, binned: BinnedSpikes) -> int:
    """Raise unless binned is this session's; return bins per second."""
    if binned.eid != session_eid:
        raise ValueError(f"binned data is for {binned.eid}, the session is {session_eid}")
    return 1000 // binned.bin_ms


def _target(binned: BinnedSpikes, name: str, values: np.ndarray, config) -> BinTarget:
    values = values.astype(np.float32)
    assert values.shape == (binned.n_bins,)
    return BinTarget(
        binned.eid,
        name,
        values,
        binned.bin_ms,
        binned.first_bin,
        binned.fingerprint,
        config.fingerprint(),
    )


def wheel_velocity(session: Session, binned: BinnedSpikes, config: TargetConfig) -> BinTarget:
    rate = check_grid(session.eid, binned)
    if "behaviour.wheel" not in session.available.present:
        reason = session.available.missing.get("behaviour.wheel", "not in the session")
        raise ValueError(f"{session.eid}: wheel is missing: {reason}")
    wheel = session.behaviour["wheel"]
    if wheel.data.ndim != 1:
        raise ValueError(f"{session.eid}: expected (n_samples,) wheel positions")
    edges = (binned.first_bin + np.arange(binned.n_bins + 1)) / rate
    position = np.interp(edges, wheel.timestamps, wheel.data.astype(np.float64))
    position[(edges < wheel.timestamps[0]) | (edges > wheel.timestamps[-1])] = np.nan
    return _target(binned, "wheel_velocity", np.diff(position) * rate, config)


def _check_release(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    found = (manifest.get("dataset_name"), manifest.get("dataset_version"))
    if found != (BEHAVIOUR_DATASET_NAME, BEHAVIOUR_DATASET_VERSION):
        raise ValueError(f"{root}: found {found}, expected bwm_behavior 2.0.0")


def load_movement_epochs(
    behaviour_root: str | os.PathLike, eid: str
) -> tuple[np.ndarray, np.ndarray]:
    """(n_moves, 2) movement and (n_quiet, 2) quiescence [t_start, t_end], seconds."""
    root = Path(behaviour_root)
    _check_release(root)
    epochs = []
    for name in ("movement_state_epochs", "quiescence_state_epochs"):
        table = pd.read_parquet(
            root / "features" / f"{name}.parquet",
            columns=["t_start", "t_end"],
            filters=[("eid", "==", eid)],
        )
        if table.empty:
            raise ValueError(f"{eid}: no {name} in {root}")
        epochs.append(table.sort_values("t_start").to_numpy(np.float64))
    return epochs[0], epochs[1]


def movement_state_from_epochs(
    binned: BinnedSpikes, moves: np.ndarray, quiescent: np.ndarray, config: TargetConfig
) -> BinTarget:
    rate = 1000 // binned.bin_ms
    labelled = np.concatenate([moves, quiescent])
    labelled = labelled[np.argsort(labelled[:, 0], kind="stable")]
    if np.any(labelled[:, 1] < labelled[:, 0]) or np.any(labelled[1:, 0] < labelled[:-1, 1]):
        raise ValueError(f"{binned.eid}: movement and quiescence epochs overlap")
    state = np.full(binned.n_bins, np.nan)
    for value, epochs in ((1.0, moves), (0.0, quiescent)):
        # Bin k is wholly inside [t0, t1] when k >= t0 * rate and k + 1 <= t1 * rate.
        first = np.ceil(epochs[:, 0] * rate).astype(np.int64) - binned.first_bin
        last = np.floor(epochs[:, 1] * rate).astype(np.int64) - 1 - binned.first_bin
        for a, b in zip(np.clip(first, 0, None), np.clip(last, None, binned.n_bins - 1)):
            if a <= b:
                state[a : b + 1] = value
    return _target(binned, "movement_state", state, config)


def movement_state(
    session: Session,
    binned: BinnedSpikes,
    behaviour_root: str | os.PathLike,
    config: TargetConfig,
) -> BinTarget:
    check_grid(session.eid, binned)
    moves, quiescent = load_movement_epochs(behaviour_root, session.eid)
    return movement_state_from_epochs(binned, moves, quiescent, config)
