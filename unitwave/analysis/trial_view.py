"""Single-trial view: every shown unit's spikes in one trial (or a few consecutive
trials), with the task events and behaviour on the same time axis.

Descriptive only: no statistic, no null, no label.

Conventions:
- **Trials** are numbered by their 0-based row in the session's trials table.
- **Window:** from the first shown trial's start (intervals_0) minus a pre-pad to the
  last shown trial's end (intervals_1) plus a post-pad, half-open [start, stop).
- **Zero:** the current trial's start, or one of its events. An event this trial
  lacks is refused as a zero, never guessed.
- **Rows:** grouped by probe, then by depth; the most superficial unit (largest
  distance from the tip) on top, units without a depth last in their probe.
- **Events:** each of the task definition's trial-view event columns the trials table
  has. An event a shown trial has no time for is listed as not recorded on that trial
  and not drawn. A column the table lacks is listed as absent from the session.
- **Header:** the trial's level of each of the task's conditions (None where the
  table has no value; an excluded trial says why), its reaction time when the task
  declares movement events, and its flag filters. For IBL, the fields the page has
  always had (side, signed contrast, choice, outcome, block, bwm_include) too.
- **Neighbouring trials** are consecutive rows of the trials table, around the
  current trial, whether or not they pass the trial filters (each says whether).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.analysis.conditions import (
    LEVEL_NAMES,
    available_conditions,
    condition,
    signed_contrast,
    stimulus_side,
)
from unitwave.analysis.movement import binned_wheel_speed, reaction_times
from unitwave.analysis.tasks import DEFAULT_TASK, TaskDefinition, load_task
from unitwave.data.session import Session, TimeSeries

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "trial_view.yaml"
_KEYS = {"pre_pad_s", "post_pad_s", "speed_bin_s", "n_trials", "max_trials"}
NUMBERING = "0-based (row of the trials table)"
TRIAL_START = "trial_start"
# The IBL definition's trials column -> label, in task order, and its optional
# single-channel behaviour traces, in the order they are offered.
EVENT_COLUMNS = dict(load_task().trial_view_events)
TRACES = load_task().traces


def _task(task: TaskDefinition | None) -> TaskDefinition:
    return task if task is not None else load_task()


@dataclass(frozen=True)
class TrialViewConfig:
    pre_pad_s: float
    post_pad_s: float
    speed_bin_s: float
    n_trials: int
    max_trials: int


def load_trial_view_config(path: str | os.PathLike = DEFAULT_CONFIG) -> TrialViewConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    cfg = TrialViewConfig(
        float(raw["pre_pad_s"]),
        float(raw["post_pad_s"]),
        float(raw["speed_bin_s"]),
        int(raw["n_trials"]),
        int(raw["max_trials"]),
    )
    if not 1 <= cfg.n_trials <= cfg.max_trials:
        raise ValueError(f"{path}: n_trials must be between 1 and max_trials")
    return cfg


def alignment_label(align: str, task: TaskDefinition | None = None) -> str:
    return "trial start" if align == TRIAL_START else _task(task).trial_view_events[align]


def row_order(units: pd.DataFrame) -> list[str]:
    """Unit ids grouped by probe (sorted by name), then by depth, most superficial first.

    units: (n_units, ...) indexed by unit id, with `probe` and `depth_um` columns.
    """
    depth = units["depth_um"].to_numpy(np.float64)
    order = sorted(
        range(len(units)),
        key=lambda i: (
            units["probe"].iloc[i],
            np.isnan(depth[i]),
            -depth[i] if np.isfinite(depth[i]) else 0.0,
            str(units.index[i]),
        ),
    )
    return [units.index[i] for i in order]


@dataclass(frozen=True)
class Window:
    """trials: the shown trials, consecutive; trial: the current one. Times in seconds:
    zero_s on the session clock, start/stop relative to it."""

    trials: tuple[int, ...]
    trial: int
    align: str
    zero_s: float
    start_rel_s: float
    stop_rel_s: float

    @property
    def start_s(self) -> float:
        return self.zero_s + self.start_rel_s

    @property
    def stop_s(self) -> float:
        return self.zero_s + self.stop_rel_s


def _value(trials: pd.DataFrame, column: str, k: int) -> float:
    return float(trials[column].iloc[k]) if column in trials else np.nan


def trial_window(
    trials: pd.DataFrame,
    trial: int,
    n_trials: int,
    align: str,
    pre: float,
    post: float,
    task: TaskDefinition | None = None,
) -> Window:
    """The plotted window for `n_trials` consecutive trials around `trial`."""
    columns = _task(task).trial_view_events
    n_total = len(trials)
    if not 0 <= trial < n_total:
        raise ValueError(f"no trial {trial}: this session has trials 0 to {n_total - 1}")
    if not 1 <= n_trials <= n_total:
        raise ValueError(f"cannot show {n_trials} trials of a {n_total}-trial session")
    if pre < 0 or post < 0:
        raise ValueError("pads must be zero or positive")
    if align != TRIAL_START and align not in columns:
        raise ValueError(f"unknown alignment {align!r}; choose {[TRIAL_START, *columns]}")
    if align != TRIAL_START and align not in trials:
        raise ValueError(f"the trials table has no {align} column to align on")
    first = int(np.clip(trial - (n_trials - 1) // 2, 0, n_total - n_trials))
    shown = tuple(range(first, first + n_trials))
    start, stop = _value(trials, "intervals_0", first), _value(trials, "intervals_1", shown[-1])
    if not (np.isfinite(start) and np.isfinite(stop)):
        raise ValueError("the trials table has no start or end time for these trials")
    zero = _value(trials, "intervals_0" if align == TRIAL_START else align, trial)
    if not np.isfinite(zero):
        raise ValueError(
            f"{alignment_label(align, task)} is not recorded on trial {trial}: "
            "align to the trial start or another event"
        )
    return Window(shown, trial, align, zero, start - pre - zero, stop + post - zero)


def trial_spikes(spikes, unit_ids, window: Window) -> list[np.ndarray]:
    """Per unit, (n_spikes_in_window,) times relative to the window's zero."""
    out = []
    for unit in unit_ids:
        s = np.asarray(spikes[unit], np.float64)
        lo, hi = np.searchsorted(s, [window.start_s, window.stop_s], side="left")
        out.append(s[lo:hi] - window.zero_s)
    return out


def _kind(trials: pd.DataFrame, task: TaskDefinition, column: str, k: int) -> str | None:
    """Which of the task's split events this time is on trial k (IBL feedback: reward or
    error), named by the condition reading the split column; None if none applies."""
    for e in task.events.values():
        if e.column == column and e.when and e.when[0] in trials:
            v = _value(trials, e.when[0], k)
            if v == e.when[1]:
                names = {}
                for c in task.conditions.values():
                    if c.column == e.when[0]:
                        names.update(c.level_names)
                return names.get(v, f"{v:g}")
    return None


def trial_events(
    trials: pd.DataFrame, window: Window, task: TaskDefinition | None = None
) -> tuple[list[dict], list[dict], list[str]]:
    """(drawn, not recorded, absent) events of the shown trials.

    drawn: {trial, event (column), label, time_s (relative to zero), kind, slot}; kind
    names the split event this time is on (IBL feedback: "reward" or "error"), else
    None; slot is the column's place in the task's event order.
    not recorded: {trial, event, label} for a trial with no time for an event.
    absent: event columns this trials table does not have.
    """
    task = _task(task)
    columns = task.trial_view_events
    drawn, not_recorded = [], []
    for k in window.trials:
        for slot, (column, label) in enumerate(columns.items()):
            if column not in trials:
                continue
            t = _value(trials, column, k)
            if not np.isfinite(t):
                not_recorded.append({"trial": k, "event": column, "label": label})
                continue
            kind = _kind(trials, task, column, k)
            label = f"{label}: {kind}" if kind else label
            drawn.append(
                {
                    "trial": k,
                    "event": column,
                    "label": label,
                    "time_s": t - window.zero_s,
                    "kind": kind,
                    "slot": slot,
                }
            )
    absent = [c for c in columns if c not in trials]
    return drawn, not_recorded, absent


def _named(trials: pd.DataFrame, column: str, k: int, names: dict) -> str | None:
    v = _value(trials, column, k)
    return None if np.isnan(v) else names.get(v, f"{v:g}")


def trial_header(
    trials: pd.DataFrame, trial: int, keep: np.ndarray, task: TaskDefinition | None = None
) -> dict:
    """The trial's conditions; None where the table has no value for it."""
    task = _task(task)
    k = trial
    levels = []
    for name, label in available_conditions(trials, task).items():
        c = condition(trials, name, task)
        v = c.values[k]
        level = None if np.isnan(v) else c.names[c.levels.index(float(v))]
        excluded = task.conditions[name].excluded if level is None else None
        levels.append({"name": name, "label": label, "level": level, "excluded": excluded})
    rt = np.nan
    if task.movement is not None and all(
        col in trials
        for e in (task.movement.stimulus, task.movement.movement)
        for col in task.events[e].columns
    ):
        rt = reaction_times(trials, task)[k]
    flags = {
        f.column: (None if pd.isna(trials[f.column].iloc[k]) else bool(trials[f.column].iloc[k]))
        for f in task.trial_filters.values()
        if f.kind == "flag" and f.column in trials
    }
    header = {
        "trial": k,
        "numbering": NUMBERING,
        "task": task.label,
        "conditions": levels,
        "reaction_time_s": None if np.isnan(rt) else float(rt),
        "flags": flags,
        "passes_filter": bool(keep[k]),
    }
    if task.name == DEFAULT_TASK:  # IBL's own fields, as the page has always had them
        header.update(_ibl_header(trials, k))
    return header


def _ibl_header(trials: pd.DataFrame, k: int) -> dict:
    has_contrast = "contrastLeft" in trials and "contrastRight" in trials
    side = stimulus_side(trials)[k] if has_contrast else np.nan
    contrast = signed_contrast(trials)[k] if has_contrast else np.nan
    bwm = trials["bwm_include"].iloc[k] if "bwm_include" in trials else None
    block = _value(trials, "probabilityLeft", k)
    return {
        "side": None if np.isnan(side) else LEVEL_NAMES["side"][side],
        "signed_contrast": None if np.isnan(contrast) else float(contrast),
        "choice": _named(trials, "choice", k, {**LEVEL_NAMES["choice"], 0.0: "no-go (0)"}),
        "outcome": _named(trials, "feedbackType", k, LEVEL_NAMES["outcome"]),
        "block": None if np.isnan(block) else block,
        "bwm_include": None if bwm is None or pd.isna(bwm) else bool(bwm),
    }


def step(keep: np.ndarray, trial: int, direction: int, filtered: bool) -> int | None:
    """The next (direction +1) or previous (-1) trial, among those passing the filter
    when `filtered`, else among all trials; None past the ends."""
    keep = np.asarray(keep, bool)
    candidates = np.flatnonzero(keep) if filtered else np.arange(keep.size)
    later = candidates[candidates > trial] if direction > 0 else candidates[candidates < trial]
    if later.size == 0:
        return None
    return int(later[0] if direction > 0 else later[-1])


def position(keep: np.ndarray, trial: int, filtered: bool) -> tuple[int | None, int]:
    """(1-based place of the trial among the stepped-through trials, how many there are).
    The place is None when the trial fails the filter it is stepped through by."""
    keep = np.asarray(keep, bool)
    if not filtered:
        return trial + 1, keep.size
    if not keep[trial]:
        return None, int(keep.sum())
    return int(keep[: trial + 1].sum()), int(keep.sum())


def trial_wheel(wheel: TimeSeries, window: Window, bin_width: float) -> dict:
    """Wheel position (raw samples, rad, relative to its value at zero) and speed (rad/s,
    binned_wheel_speed on bins tiling the window from the zero)."""
    t = np.asarray(wheel.timestamps, np.float64)
    pos = np.asarray(wheel.data, np.float64)
    lo, hi = np.searchsorted(t, [window.start_s, window.stop_s], side="left")
    at_zero = float(np.interp(window.zero_s, t, pos))
    first = bin_width * np.floor(window.start_rel_s / bin_width + 1e-9)
    last = bin_width * np.ceil(window.stop_rel_s / bin_width - 1e-9)
    speed = binned_wheel_speed(wheel, np.array([window.zero_s]), (first, last), bin_width)[0]
    edges = first + bin_width * np.arange(speed.size + 1)
    return {
        "position_t_s": t[lo:hi] - window.zero_s,
        "position_rad": pos[lo:hi] - at_zero,
        "speed_t_s": (edges[:-1] + edges[1:]) / 2,
        "speed": speed,
        "speed_window_s": (float(first), float(last)),
        "bin_width_s": bin_width,
    }


def trial_trace(series: TimeSeries, window: Window) -> tuple[np.ndarray, np.ndarray]:
    """(n_samples_in_window,) times relative to zero and raw values (NaN kept)."""
    t = np.asarray(series.timestamps, np.float64)
    lo, hi = np.searchsorted(t, [window.start_s, window.stop_s], side="left")
    return t[lo:hi] - window.zero_s, np.asarray(series.data, np.float64)[lo:hi]


@dataclass(frozen=True)
class TrialView:
    """Everything the single-trial figure plots. rows, probes, spikes: (n_rows,), top
    row first; spikes[i] is (n_spikes,) seconds relative to window.zero_s."""

    window: Window
    rows: list[str]
    probes: list[str]
    spikes: list[np.ndarray]
    events: list[dict]
    not_recorded: list[dict]
    absent_events: list[str]
    header: dict
    boundaries: list[dict]
    wheel: dict | None
    wheel_missing: str | None
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    traces_missing: dict[str, str] = field(default_factory=dict)
    align_label: str = "trial start"  # what the zero is, in the task's words


def _missing(session: Session, name: str) -> str:
    return session.available.missing.get(f"behaviour.{name}", "not in this session")


def trial_view(
    session: Session,
    units: pd.DataFrame,
    trial: int,
    *,
    cfg: TrialViewConfig,
    n_trials: int = 1,
    align: str = TRIAL_START,
    pre: float | None = None,
    post: float | None = None,
    keep: np.ndarray | None = None,
    traces=(),
    task: TaskDefinition | None = None,
) -> TrialView:
    """The single-trial view of `units` ((n_units, ...) with probe and depth_um, the
    units to show) around `trial`. keep: (n_trials,) bool, the trials passing the
    current trial filters (None: all)."""
    if n_trials > cfg.max_trials:
        raise ValueError(f"at most {cfg.max_trials} trials at once (configs/trial_view.yaml)")
    trials = session.trials
    keep = np.ones(len(trials), bool) if keep is None else np.asarray(keep, bool)
    assert keep.shape == (len(trials),)
    pre = cfg.pre_pad_s if pre is None else pre
    post = cfg.post_pad_s if post is None else post
    task = _task(task)
    window = trial_window(trials, trial, n_trials, align, pre, post, task)
    rows = row_order(units)
    drawn, not_recorded, absent = trial_events(trials, window, task)
    boundaries = [
        {
            "trial": k,
            "start_rel_s": _value(trials, "intervals_0", k) - window.zero_s,
            "stop_rel_s": _value(trials, "intervals_1", k) - window.zero_s,
            "passes_filter": bool(keep[k]),
        }
        for k in window.trials
    ]
    wheel = session.behaviour.get("wheel")
    unknown = sorted(set(traces) - set(task.traces))
    if unknown:
        raise ValueError(f"unknown behaviour traces {unknown}; choose from {list(task.traces)}")
    return TrialView(
        window=window,
        rows=rows,
        probes=units.loc[rows, "probe"].tolist(),
        spikes=trial_spikes(session.spikes, rows, window),
        events=drawn,
        not_recorded=not_recorded,
        absent_events=absent,
        header=trial_header(trials, trial, keep, task),
        boundaries=boundaries,
        wheel=None if wheel is None else trial_wheel(wheel, window, cfg.speed_bin_s),
        wheel_missing=None if wheel is not None else _missing(session, "wheel"),
        traces={
            n: trial_trace(session.behaviour[n], window) for n in traces if n in session.behaviour
        },
        traces_missing={n: _missing(session, n) for n in traces if n not in session.behaviour},
        align_label=alignment_label(align, task),
    )
