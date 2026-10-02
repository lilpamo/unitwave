"""Task events to align on, as the task definition declares them (IBL by default).

An event is a time column, optionally limited to trials where another column equals a
value: IBL's feedback is split by outcome (reward: feedbackType 1, error: -1), so the
two are never averaged together. A trial with no time for an event keeps NaN; psth()
excludes and counts it rather than filling it.
"""

import numpy as np
import pandas as pd

from unitwave.analysis.tasks import TaskDefinition, load_task


def _task(task: TaskDefinition | None) -> TaskDefinition:
    return task if task is not None else load_task()


# The IBL definition's events, as (label, column, value of the `when` column or None).
EVENTS = {
    name: (e.label, e.column, e.when[1] if e.when else None)
    for name, e in load_task().events.items()
}


def _event(trials: pd.DataFrame, event: str, task: TaskDefinition | None):
    task = _task(task)
    if event not in task.events:
        raise ValueError(f"unknown event {event!r}; available: {sorted(task.events)}")
    e = task.events[event]
    for column in e.columns:
        if column not in trials:
            raise ValueError(f"trials have no {column} column, needed for {event}")
    return e


def event_times(trials: pd.DataFrame, event: str, task: TaskDefinition | None = None) -> np.ndarray:
    """(n_selected_trials,) seconds, trial order kept; NaN where the trial has no time."""
    e = _event(trials, event, task)
    times = trials[e.column].to_numpy(np.float64)
    if e.when is not None:
        column, value = e.when
        times = times[trials[column].to_numpy(np.float64) == value]
    assert times.ndim == 1
    return times


def available_events(trials: pd.DataFrame, task: TaskDefinition | None = None) -> dict[str, str]:
    """event name -> label, for the events this trials table has the columns for."""
    return {
        name: e.label
        for name, e in _task(task).events.items()
        if all(column in trials for column in e.columns)
    }


def trial_event_times(
    trials: pd.DataFrame, event: str, task: TaskDefinition | None = None
) -> np.ndarray:
    """(n_trials,) seconds, one per trial row: NaN where the trial has no such event,
    including feedback of the other outcome. For splitting trials by condition."""
    e = _event(trials, event, task)
    times = trials[e.column].to_numpy(np.float64).copy()
    if e.when is not None:
        column, value = e.when
        times[trials[column].to_numpy(np.float64) != value] = np.nan
    return times
