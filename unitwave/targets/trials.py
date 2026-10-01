"""Trial-level targets: choice, block, stimulus side and movement onset.  [R6]

Only trials the BWM release marks `bwm_include` are used (the BWM paper's rule:
reaction time 0.08-2 s, a choice made, no missing events), and only those whose
label is defined:
- choice: 1 for clockwise (choice == +1), 0 for counter-clockwise (-1).
- block: 1 for the left block (probabilityLeft == 0.8), 0 for the right (0.2); the
  unbiased 0.5 block has no label.
- stimulus side: 1 for a stimulus on the left (contrastLeft set), 0 for the right.
  A 0% contrast trial shows nothing, so it has no side to decode: it is excluded and
  counted in `excluded` (S3, docs/DECISIONS.md).
- movement onset: the event time firstMovement_times itself.

choice, block and stimulus side come with a decoding window from configs/targets.yaml: context_bins
bins ending at `end_bin`, the last bin that finishes by anchor + stop_s, so a window
never reaches past its stop. Movement onset has no window; end_bin is the bin that
contains the onset. Pass end_bin to preprocess.windows `extract` as `ends`.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from unitwave.data.session import Session
from unitwave.preprocess.binning import BinnedSpikes
from unitwave.targets.bins import check_grid
from unitwave.targets.config import TargetConfig


@dataclass(frozen=True)
class TrialTarget:
    """table: one row per usable trial: `trial` (row in Session.trials), `label`, `end_bin`.

    context_bins: window length in bins, or None for an event target without a window.
    excluded: why -> count, for included trials left out beyond the label's definition.
    """

    eid: str
    name: str
    table: pd.DataFrame
    context_bins: int | None
    bin_ms: int
    preproc_fingerprint: str | None
    targets_fingerprint: str
    excluded: dict = field(default_factory=dict)


def _included(session: Session) -> np.ndarray:
    if "bwm_include" not in session.trials:
        raise ValueError(
            f"{session.eid}: trial targets need the BWM release's bwm_include flag, which "
            "this session's trials lack (load it with the bwm backend)"
        )
    return session.trials["bwm_include"].to_numpy(bool)


def _anchor_times(session: Session, rows: np.ndarray, anchor: str) -> np.ndarray:
    times = session.trials[anchor].to_numpy(np.float64)[rows]
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{session.eid}: an included trial has no {anchor}")
    return times


def _result(
    session, binned, name, rows, labels, end_bins, context, config, excluded=None
) -> TrialTarget:
    first, last = binned.first_bin, binned.first_bin + binned.n_bins - 1
    starts = end_bins - (context or 1) + 1
    if len(end_bins) and (starts.min() < first or end_bins.max() > last):
        raise ValueError(f"{session.eid}: a {name} window falls outside the binned data")
    table = pd.DataFrame({"trial": rows, "label": labels, "end_bin": end_bins})
    return TrialTarget(
        session.eid,
        name,
        table,
        context,
        binned.bin_ms,
        binned.fingerprint,
        config.fingerprint(),
        dict(excluded or {}),
    )


def _windowed(session, binned, config, name, defined, labels_of, excluded=None) -> TrialTarget:
    rate = check_grid(session.eid, binned)
    window = config.windows[name]
    context = window.n_bins(binned.bin_ms)
    rows = np.flatnonzero(_included(session) & defined)
    times = _anchor_times(session, rows, window.anchor)
    # Bin k finishes at (k + 1) / rate, so the last one finishing by t is floor(t * rate) - 1.
    end_bins = np.floor((times + window.stop_s) * rate).astype(np.int64) - 1
    return _result(
        session, binned, name, rows, labels_of(rows), end_bins, context, config, excluded
    )


def choice(session: Session, binned: BinnedSpikes, config: TargetConfig) -> TrialTarget:
    values = session.trials["choice"].to_numpy(np.float64)
    return _windowed(
        session,
        binned,
        config,
        "choice",
        np.isin(values, (-1.0, 1.0)),
        lambda rows: (values[rows] == 1.0).astype(np.int64),
    )


def block(session: Session, binned: BinnedSpikes, config: TargetConfig) -> TrialTarget:
    values = session.trials["probabilityLeft"].to_numpy(np.float64)
    return _windowed(
        session,
        binned,
        config,
        "block",
        np.isin(values, (0.2, 0.8)),
        lambda rows: (values[rows] == 0.8).astype(np.int64),
    )


def stimulus_side(session: Session, binned: BinnedSpikes, config: TargetConfig) -> TrialTarget:
    left = session.trials["contrastLeft"].to_numpy(np.float64)
    right = session.trials["contrastRight"].to_numpy(np.float64)
    included = _included(session)
    sides = np.isfinite(left).astype(int) + np.isfinite(right).astype(int)
    if np.any(included & (sides != 1)):
        raise ValueError(f"{session.eid}: an included trial has a contrast on not exactly one side")
    contrast = np.where(np.isfinite(left), left, right)
    zero = included & (contrast == 0)
    return _windowed(
        session,
        binned,
        config,
        "stimulus_side",
        contrast > 0,
        lambda rows: np.isfinite(left[rows]).astype(np.int64),
        {"0% contrast": int(zero.sum())},
    )


def movement_onsets(session: Session, binned: BinnedSpikes, config: TargetConfig) -> TrialTarget:
    rate = check_grid(session.eid, binned)
    rows = np.flatnonzero(_included(session))
    times = _anchor_times(session, rows, "firstMovement_times")
    end_bins = np.floor(times * rate).astype(np.int64)
    return _result(session, binned, "movement_onset", rows, times, end_bins, None, config)
