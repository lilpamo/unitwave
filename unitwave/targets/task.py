"""Trial targets declared by a task definition, and their no-spikes baseline (step 8a).

A task's decoding section (analysis.tasks.Decoding) names each target's condition and
its two levels (the second is label 1), the window around an event it is decoded
from, and the trial filters it uses. IBL's own targets (targets/trials.py) are
unchanged; this is for every other task.

- **Trials:** the task's named trial filters first, then the condition's two levels;
  a trial without the window's event is left out. Each exclusion is counted by reason.
- **Window:** like IBL's targets: context bins ending at the last bin that finishes by
  event + stop_s, so a window never reaches past its stop.
- **No-spikes baseline (null_trialstruct):** the task's trialstruct variables on the
  current trial, except the target's own columns and those only known after its
  window (`after`), and every trialstruct variable on the previous `history_trials`
  trials (0 before the first trial and for missing values), as IBL's baseline does
  with its own variables.
"""

import numpy as np
import pandas as pd

from unitwave.analysis.conditions import TrialFilter, apply_trial_filter, condition
from unitwave.analysis.events import trial_event_times
from unitwave.analysis.tasks import TaskDefinition
from unitwave.data.session import Session
from unitwave.preprocess.binning import BinnedSpikes
from unitwave.targets.bins import check_grid
from unitwave.targets.trials import TrialTarget

PREFIX = "task:"


def target_name(name: str) -> str:
    """'task:choice' and its shifted copies ('task:choice:shift5') -> 'choice'."""
    return name.removeprefix(PREFIX).split(":")[0]


def window_bins(task: TaskDefinition, name: str, bin_ms: int) -> int:
    spec = task.decoding.targets[name]
    n = (spec.stop_s - spec.start_s) * 1000 / bin_ms
    if abs(n - round(n)) > 1e-9:
        raise ValueError(
            f"{name}: a {spec.stop_s - spec.start_s:g} s window is not a whole number of "
            f"{bin_ms} ms bins"
        )
    return round(n)


def task_trial_target(
    session: Session, binned: BinnedSpikes, task: TaskDefinition, name: str
) -> TrialTarget:
    """table: one row per usable trial: `trial` (row in Session.trials), `label` (1 for
    the second level), `end_bin`; excluded: why -> count."""
    if task.decoding is None or name not in task.decoding.targets:
        raise ValueError(f"the {task.label} definition declares no decoding target {name!r}")
    spec = task.decoding.targets[name]
    rate = check_grid(session.eid, binned)
    trials = session.trials
    keep = np.ones(len(trials), dtype=bool)
    excluded: dict[str, int] = {}

    def drop(mask: np.ndarray, why: str) -> None:
        nonlocal keep
        n = int((keep & ~mask).sum())
        if n:
            excluded[why] = excluded.get(why, 0) + n
        keep &= mask

    for f in spec.trial_filters:
        sel = apply_trial_filter(trials, TrialFilter(task, **{f: True}))
        drop(sel.mask, task.trial_filters[f].reason)
    cond = condition(trials, spec.condition, task)
    names = dict(zip(cond.levels, cond.names))
    a, b = (names.get(v, f"{v:g}") for v in spec.levels)
    drop(
        np.isin(cond.values, spec.levels),
        f"{task.conditions[spec.condition].label.lower()} not {a} or {b}",
    )
    times = trial_event_times(trials, spec.event, task)
    drop(np.isfinite(times), f"no {task.events[spec.event].label.lower()} time")
    rows = np.flatnonzero(keep)
    labels = (cond.values[rows] == spec.levels[1]).astype(np.int64)
    # Bin k finishes at (k + 1) / rate, so the last one finishing by t is floor(t * rate) - 1.
    end_bins = np.floor((times[rows] + spec.stop_s) * rate).astype(np.int64) - 1
    context = window_bins(task, name, binned.bin_ms)
    first, last = binned.first_bin, binned.first_bin + binned.n_bins - 1
    if len(end_bins) and (end_bins.min() - context + 1 < first or end_bins.max() > last):
        raise ValueError(f"{session.eid}: a {name} window falls outside the binned data")
    table = pd.DataFrame({"trial": rows, "label": labels, "end_bin": end_bins})
    return TrialTarget(
        session.eid,
        f"{PREFIX}{name}",
        table,
        context,
        binned.bin_ms,
        binned.fingerprint,
        task.fingerprint,
        excluded,
    )


def task_trialstruct_features(
    trials: pd.DataFrame, target: TrialTarget, task: TaskDefinition, *, history_trials: int
) -> tuple[np.ndarray, list[str]]:
    """(n_rows, n_features) float32, one row per row of target.table, and names."""
    name = target_name(target.name)
    spec = task.decoding.targets[name]
    own = set(task.conditions[spec.condition].columns)
    now = [c for c in task.trialstruct if c not in own and c not in spec.after]
    missing = [c for c in task.trialstruct if c not in trials]
    if missing:
        raise ValueError(f"the trials table lacks trialstruct columns {missing}")
    n = len(trials)
    columns, names = [], []
    for c in now:
        columns.append(np.nan_to_num(trials[c].to_numpy(np.float64)))
        names.append(c)
    for c in task.trialstruct:
        values = np.nan_to_num(trials[c].to_numpy(np.float64))
        for lag in range(1, history_trials + 1):
            shifted = np.zeros(n)
            shifted[lag:] = values[: n - lag] if lag < n else []
            columns.append(shifted)
            names.append(f"{c}_lag{lag}")
    rows = target.table["trial"].to_numpy(np.int64)
    features = np.column_stack(columns)[rows].astype(np.float32)
    assert features.shape == (len(rows), len(names))
    return features, names
