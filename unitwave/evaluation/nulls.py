"""Inputs for the two null rows of the evaluation contract.  [CLAUDE.md §5, R4, R7]

null_shuffle: the target circularly shifted within its session, which breaks the
neural-behaviour alignment and keeps the target's autocorrelation. Per-bin targets
rotate within the task period (first trial's start to last trial's end) and are
undefined outside it; trial-level labels rotate over the target's trials while each
trial's window stays where it was. Shifts are drawn with an explicit seed, never
shorter than configs/nulls.yaml's minimum; a session too short for it has no shifts.

null_trialstruct: features from task variables only, never spikes:
- per-bin targets: one-hot time since each trial start, stimulus onset and go cue,
  plus the current trial's signed contrast and block prior. Behaviour-timed events
  (first movement, response, feedback) are never read.
- choice: the current trial's signed contrast and block prior, plus the previous
  trials' stimulus side, choice and reward.
- block: the previous trials' stimulus side, choice and reward only. Block is decoded
  from a pre-stimulus window, so the current stimulus is not available to it either.
- stimulus side (S3): the current trial's block prior, the task's own prediction of
  the side, plus the previous trials' stimulus side, choice and reward. Never the
  current stimulus, which is the target.

null_pseudosession (block only; adopted after the first table): labels from
pseudo-sessions drawn from the task's own block generator, a seeded port of IBL's
brainbox generate_pseudo_blocks (the Brain Wide Map paper's null).

Fitting these features is the baseline models' job; this module only builds them.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.data.session import Session
from unitwave.preprocess.binning import BinnedSpikes
from unitwave.qc.units import task_period
from unitwave.targets.bins import BinTarget
from unitwave.targets.trials import TrialTarget

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "nulls.yaml"
TASK_EVENTS = ("intervals_0", "stimOn_times", "goCue_times")
# Lets a time that is an exact multiple of time_bin_s land in its own bin despite
# floating-point representation (0.6 / 0.1 == 5.999...).
_EPS = 1e-9


@dataclass(frozen=True)
class NullConfig:
    n_shifts: int
    min_shift_s: float
    min_shift_trials: int
    time_bin_s: float
    max_time_s: float
    history_trials: int
    n_pseudo_sessions: int = 100

    def __post_init__(self) -> None:
        for name in ("n_shifts", "min_shift_trials", "history_trials"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not 0 < self.time_bin_s < self.max_time_s:
            raise ValueError("need 0 < time_bin_s < max_time_s")
        n = self.max_time_s / self.time_bin_s
        if abs(n - round(n)) > 1e-9:
            raise ValueError("max_time_s must be a whole number of time_bin_s steps")

    @property
    def n_time_bins(self) -> int:
        return round(self.max_time_s / self.time_bin_s)


def load_null_config(path: str | os.PathLike = DEFAULT_CONFIG) -> NullConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sections = {"null_shuffle", "null_trialstruct", "null_pseudosession"}
    if set(raw) != sections:
        raise ValueError(f"{path}: keys {sorted(raw)}, expected {sorted(sections)}")
    shuffle, trialstruct = raw["null_shuffle"], raw["null_trialstruct"]
    expected = {
        "null_shuffle": {"n_shifts", "min_shift_s", "min_shift_trials"},
        "null_trialstruct": {"time_bin_s", "max_time_s", "history_trials"},
        "null_pseudosession": {"n_sessions"},
    }
    for section, keys in expected.items():
        if set(raw[section]) != keys:
            raise ValueError(
                f"{path}: {section} keys {sorted(raw[section])}, expected {sorted(keys)}"
            )
    return NullConfig(
        n_shifts=shuffle["n_shifts"],
        min_shift_s=float(shuffle["min_shift_s"]),
        min_shift_trials=shuffle["min_shift_trials"],
        time_bin_s=float(trialstruct["time_bin_s"]),
        max_time_s=float(trialstruct["max_time_s"]),
        history_trials=trialstruct["history_trials"],
        n_pseudo_sessions=raw["null_pseudosession"]["n_sessions"],
    )


# IBL's biased-block protocol: block lengths ~ exponential(60), redrawn until strictly
# between 20 and 100 trials, sides alternating, after 90 unbiased trials.
IBL_BLOCK_FACTOR, IBL_BLOCK_MIN, IBL_BLOCK_MAX, IBL_UNBIASED_TRIALS = 60.0, 20, 100, 90


def generate_pseudo_blocks(n_trials: int, *, seed: int) -> np.ndarray:
    """(n_trials,) probabilityLeft of one pseudo-session: 0.5 for the first 90 trials,
    then alternating 0.2 / 0.8 blocks with IBL's length distribution.

    A port of brainbox.task.closed_loop.generate_pseudo_blocks (ibllib 4.0.1) with a
    seeded generator: the same distribution, not the same random stream (brainbox
    uses numpy's global RNG and draws an extra integer on every loop).
    """
    rng = np.random.default_rng(seed)
    first = min(IBL_UNBIASED_TRIALS, n_trials)
    blocks: list[float] = []
    while len(blocks) < n_trials - first:
        length = rng.exponential(IBL_BLOCK_FACTOR)
        while length <= IBL_BLOCK_MIN or length >= IBL_BLOCK_MAX:
            length = rng.exponential(IBL_BLOCK_FACTOR)
        if not blocks:
            side = 0.2 if rng.integers(2) == 0 else 0.8
        else:
            side = 0.8 if blocks[-1] == 0.2 else 0.2
        blocks += [side] * int(length)
    return np.array([0.5] * first + blocks[: n_trials - first])


def draw_shifts(length: int, min_shift: int, n_shifts: int, *, seed: int) -> np.ndarray:
    """Up to n_shifts distinct shifts, uniform over [min_shift, length - min_shift].

    Empty when length < 2 * min_shift: the sequence is too short for a valid shift.
    """
    candidates = np.arange(min_shift, length - min_shift + 1, dtype=np.int64)
    if candidates.size == 0:
        return candidates
    rng = np.random.default_rng(seed)
    return rng.choice(candidates, size=min(n_shifts, candidates.size), replace=False)


def task_bins(session: Session, first_bin: int, n_bins: int, bin_ms: int) -> slice:
    """The bins wholly inside the session's task period, as a slice of a (n_bins,) array."""
    rate = 1000 // bin_ms
    t0, t1 = task_period(session.trials)
    start = max(int(np.ceil(t0 * rate)) - first_bin, 0)
    stop = min(int(np.floor(t1 * rate)) - first_bin, n_bins)
    if stop <= start:
        raise ValueError(f"{session.eid}: no bin lies wholly inside the task period")
    return slice(start, stop)


def shift_bin_target(target: BinTarget, session: Session, shift: int) -> BinTarget:
    """The target rotated by `shift` bins within the task period; NaN outside it."""
    if target.eid != session.eid:
        raise ValueError(f"target is for {target.eid}, the session is {session.eid}")
    span = task_bins(session, target.first_bin, len(target.values), target.bin_ms)
    inside = target.values[span]
    if not 0 < shift < len(inside):
        raise ValueError(f"shift {shift} must be in (0, {len(inside)})")
    values = np.full_like(target.values, np.nan)
    values[span] = np.roll(inside, -shift)
    return BinTarget(
        target.eid,
        f"{target.name}:shift{shift}",
        values,
        target.bin_ms,
        target.first_bin,
        target.preproc_fingerprint,
        target.targets_fingerprint,
    )


def shift_trial_target(target: TrialTarget, shift: int) -> TrialTarget:
    """Labels rotated by `shift` trials over the target's rows; trials and windows stay."""
    table = target.table.sort_values("trial").reset_index(drop=True)
    if not 0 < shift < len(table):
        raise ValueError(f"shift {shift} must be in (0, {len(table)})")
    table = table.assign(label=np.roll(table["label"].to_numpy(), -shift))
    return TrialTarget(
        target.eid,
        f"{target.name}:shift{shift}",
        table,
        target.context_bins,
        target.bin_ms,
        target.preproc_fingerprint,
        target.targets_fingerprint,
    )


def _signed_contrast(trials: pd.DataFrame) -> np.ndarray:
    right = np.nan_to_num(trials["contrastRight"].to_numpy(np.float64))
    left = np.nan_to_num(trials["contrastLeft"].to_numpy(np.float64))
    return right - left


def bin_trialstruct_features(
    session: Session, binned: BinnedSpikes, config: NullConfig
) -> tuple[np.ndarray, list[str]]:
    """(n_bins, n_features) float32 task-variable features on binned's grid, and names.

    Only binned's grid (first_bin, n_bins, bin_ms) is used, never its counts.
    """
    if binned.eid != session.eid:
        raise ValueError(f"binned data is for {binned.eid}, the session is {session.eid}")
    rate = 1000 // binned.bin_ms
    times = (binned.first_bin + np.arange(binned.n_bins)) / rate
    n_time = config.n_time_bins
    trials = session.trials
    blocks, names = [], []
    for event in TASK_EVENTS:
        onsets = np.sort(trials[event].to_numpy(np.float64))
        onsets = onsets[np.isfinite(onsets)]
        latest = np.searchsorted(onsets, times, side="right") - 1
        since = np.where(latest >= 0, times - onsets[np.clip(latest, 0, None)], np.inf)
        index = np.minimum(np.floor(since / config.time_bin_s + _EPS), n_time).astype(np.int64)
        one_hot = np.zeros((binned.n_bins, n_time + 1), dtype=np.float32)
        one_hot[np.arange(binned.n_bins), index] = 1.0
        blocks.append(one_hot)
        names += [f"since_{event}_{k * config.time_bin_s:.1f}s" for k in range(n_time)]
        names.append(f"since_{event}_later")

    starts = trials["intervals_0"].to_numpy(np.float64)
    current = np.searchsorted(starts, times, side="right") - 1
    started = current >= 0
    contrast = np.where(started, _signed_contrast(trials)[np.clip(current, 0, None)], 0.0)
    prior = trials["probabilityLeft"].to_numpy(np.float64)
    prior = np.where(started, prior[np.clip(current, 0, None)], 0.5)
    blocks.append(np.column_stack([contrast, prior]).astype(np.float32))
    names += ["signed_contrast", "block_prior"]
    features = np.hstack(blocks)
    assert features.shape == (binned.n_bins, len(names))
    return features, names


def _history(trials: pd.DataFrame, n_lags: int) -> tuple[np.ndarray, list[str]]:
    """(n_trials, 3 * n_lags): each trial's previous trials' side, choice and reward.

    Side is +1 when the stimulus was on the right (contrastRight given), -1 on the left,
    whatever its contrast. Choice is -1/0/+1 and reward the feedbackType. Lags before
    the first trial are 0.
    """
    side = np.where(np.isfinite(trials["contrastRight"].to_numpy(np.float64)), 1.0, -1.0)
    choice = np.nan_to_num(trials["choice"].to_numpy(np.float64))
    reward = np.nan_to_num(trials["feedbackType"].to_numpy(np.float64))
    n = len(trials)
    columns, names = [], []
    for kind, values in (("side", side), ("choice", choice), ("reward", reward)):
        for lag in range(1, n_lags + 1):
            shifted = np.zeros(n)
            shifted[lag:] = values[: n - lag] if lag < n else []
            columns.append(shifted)
            names.append(f"{kind}_lag{lag}")
    return np.column_stack(columns), names


def trial_trialstruct_features(
    session: Session, target: TrialTarget, config: NullConfig
) -> tuple[np.ndarray, list[str]]:
    """(n_rows, n_features) float32, one row per row of target.table, and names."""
    if target.eid != session.eid:
        raise ValueError(f"target is for {target.eid}, the session is {session.eid}")
    name = target.name.split(":")[0]
    if name not in ("choice", "block", "stimulus_side"):
        raise ValueError(f"no trial-structure null is defined for {name}")
    trials = session.trials
    history, names = _history(trials, config.history_trials)
    blocks = [history]
    prior = trials["probabilityLeft"].to_numpy(np.float64)
    if name == "choice":
        blocks.append(np.column_stack([_signed_contrast(trials), prior]))
        names = [*names, "signed_contrast", "block_prior"]
    elif name == "stimulus_side":
        # The block prior is the task's own prediction of the side (S3); the current
        # stimulus is the target, so it is never a feature.
        blocks.append(prior[:, None])
        names = [*names, "block_prior"]
    rows = target.table["trial"].to_numpy(np.int64)
    features = np.hstack(blocks)[rows].astype(np.float32)
    assert features.shape == (len(rows), len(names))
    return features, names
