"""Task conditions to split trials by, as the task definition declares them.

Each condition reads a column (with values that mean "missing" turned into NaN) or a
named derivation of columns (unitwave/analysis/tasks.py). IBL's conventions, checked on
d23a44ef (docs/DECISIONS.md), are its built-in definition (configs/tasks/ibl.yaml):
- **Contrast:** each trial sets exactly one of contrastLeft / contrastRight; the side
  without a stimulus is NaN. Zero contrast is 0 on the stimulus side. Signed contrast
  is contrastRight - contrastLeft with NaN read as 0: right positive, left negative,
  and 0% for both zero-contrast sides. A trial with both or neither side set breaks
  the convention and is excluded and counted, never guessed.
- **Choice:** -1 or +1, and 0 for no-go, which is excluded and counted. On correct
  trials a right stimulus has choice -1 and a left one +1, so the levels are named
  "right (-1)" and "left (+1)".
- **Outcome:** feedbackType, -1 error, +1 reward.
- **Block:** probabilityLeft, 0.2 / 0.5 / 0.8 (0.5 is the unbiased opening block).
- **Reaction time:** first movement - stimulus onset, split at this table's median
  into early and late. Trials without a first movement are excluded and counted.

Only conditions whose columns the table has are offered, so Phy sessions get what
their events CSV provides.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from unitwave.analysis.events import trial_event_times
from unitwave.analysis.tasks import TaskDefinition, derive, load_task


def _task(task: TaskDefinition | None) -> TaskDefinition:
    return task if task is not None else load_task()


_IBL = load_task()
# Views of the IBL definition: name -> (label, trials columns it needs), and
# condition -> value -> name (shared with the single-trial header).
CONDITIONS = {name: (c.label, c.columns) for name, c in _IBL.conditions.items()}
LEVEL_NAMES = {name: c.level_names for name, c in _IBL.conditions.items() if c.level_names}


@dataclass(frozen=True)
class Condition:
    """values: (n_trials,) each trial's level, NaN where excluded (reason in `excluded`)."""

    name: str
    values: np.ndarray
    levels: tuple[float, ...]
    names: tuple[str, ...]
    n_excluded: int
    excluded: str


@dataclass(frozen=True)
class Part:
    """One level's event times, (n_level_trials,), NaN kept where the event is missing."""

    level: float
    name: str
    times: np.ndarray


def available_conditions(
    trials: pd.DataFrame, task: TaskDefinition | None = None
) -> dict[str, str]:
    """name -> label, for the conditions this trials table has the columns for."""
    return {
        name: c.label
        for name, c in _task(task).conditions.items()
        if all(col in trials for col in c.columns)
    }


def _need(trials: pd.DataFrame, name: str, task: TaskDefinition | None = None) -> None:
    conditions = _task(task).conditions
    if name not in conditions:
        raise ValueError(f"unknown condition {name!r}; available: {sorted(conditions)}")
    missing = [c for c in dict.fromkeys(conditions[name].columns) if c not in trials]
    if missing:
        raise ValueError(f"trials have no {missing} column(s), needed for {name}")


def _ibl_derived(trials: pd.DataFrame, condition_name: str, derivation: str) -> np.ndarray:
    _need(trials, condition_name)
    return derive(trials, derivation, _IBL.conditions[condition_name].columns)[0]


def signed_contrast(trials: pd.DataFrame) -> np.ndarray:
    """(n_trials,) IBL's contrastRight - contrastLeft, NaN read as 0; NaN if not one
    side set."""
    return _ibl_derived(trials, "contrast", "signed_contrast")


def stimulus_side(trials: pd.DataFrame) -> np.ndarray:
    """(n_trials,) IBL's +1 right, -1 left; NaN if not exactly one side set."""
    return _ibl_derived(trials, "side", "stimulus_side")


def contrast_strata(trials: pd.DataFrame) -> np.ndarray:
    """(n_trials,) signed contrast with zero split by side (-0% and +0% differ).

    For stratified nulls: at 0% contrast the rewarded side still differs, and choice
    still tracks it, so merging the two zeros would leave choice confounded with side.
    """
    return strata_values(trials, "signed_contrast")


def strata_values(
    trials: pd.DataFrame, name: str, task: TaskDefinition | None = None
) -> np.ndarray:
    """(n_trials,) the stratum of each trial, for a null that permutes within strata;
    NaN where it is undefined."""
    task = _task(task)
    if name not in task.strata:
        raise ValueError(f"unknown strata {name!r}; available: {sorted(task.strata)}")
    s = task.strata[name]
    if s.condition is not None:
        return condition(trials, s.condition, task).values
    missing = [c for c in s.derivation.columns if c not in trials]
    if missing:
        raise ValueError(f"trials have no {missing} column(s), needed for {name} strata")
    return derive(trials, s.derivation.name, s.derivation.columns)[0]


def _percent(value: float) -> str:
    text = f"{abs(value) * 100:g}%"
    return text if value == 0 else ("+" if value > 0 else "-") + text


def _level_name(spec, derived: dict | None, v: float) -> str:
    if derived is not None:
        return derived.get(v, f"{v:g}")
    if spec.level_format == "percent":
        return _percent(v)
    if spec.level_format:
        return spec.level_format.format(v=v)
    return spec.level_names.get(v, f"{v:g}")


def condition(trials: pd.DataFrame, name: str, task: TaskDefinition | None = None) -> Condition:
    task = _task(task)
    _need(trials, name, task)
    spec = task.conditions[name]
    if spec.column is not None:
        values, derived = trials[spec.column].to_numpy(np.float64).copy(), None
    else:
        values, derived = derive(trials, spec.derivation.name, spec.derivation.columns)
    if spec.missing_values:
        values[np.isin(values, spec.missing_values)] = np.nan
    levels = tuple(float(v) for v in np.unique(values[np.isfinite(values)]))
    names = tuple(_level_name(spec, derived, v) for v in levels)
    n_excluded = int(np.isnan(values).sum())
    assert values.shape == (len(trials),)
    return Condition(name, values, levels, names, n_excluded, spec.excluded if n_excluded else "")


def split_event_times(
    trials: pd.DataFrame, event: str, cond: Condition, task: TaskDefinition | None = None
) -> list[Part]:
    """One Part per level: the event time of every trial at that level."""
    times = trial_event_times(trials, event, task)
    return [
        Part(level, name, times[cond.values == level])
        for level, name in zip(cond.levels, cond.names)
    ]


# ---------- trial filters: which trials an analysis uses ----------

# The IBL definition's filters: name -> (label, trials columns it needs).
TRIAL_FILTERS = {name: (f.label, f.columns) for name, f in _IBL.trial_filters.items()}


class TrialFilter:
    """Which trials to keep, one value per filter the task declares. Empty tuples and
    False keep everything. Built with keywords, e.g. TrialFilter(bwm_include=True), or
    TrialFilter(task, **values) for another task's filters.

    For IBL: bwm_include keeps only the release's bwm_include trials (on d23a44ef that
    is exactly a reaction time, first movement - stimulus onset, within [0.08, 2] s);
    exclude_nogo drops choice 0; contrasts: absolute contrasts to keep (0 is 0%);
    blocks: probabilityLeft values; outcomes: feedbackType values (-1 error, 1 reward).
    """

    __slots__ = ("_values", "task")

    def __init__(self, task: TaskDefinition | None = None, /, **values):
        task = _task(task)
        unknown = sorted(set(values) - set(task.trial_filters))
        if unknown:
            raise TypeError(f"unknown trial filters {unknown}; known: {list(task.trial_filters)}")
        object.__setattr__(self, "task", task)
        object.__setattr__(
            self,
            "_values",
            {
                name: values.get(name, () if spec.kind == "select" else False)
                for name, spec in task.trial_filters.items()
            },
        )

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_values")[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AttributeError("a TrialFilter is frozen")

    def __eq__(self, other):
        if not isinstance(other, TrialFilter):
            return NotImplemented
        return self.task.name == other.task.name and self._values == other._values

    def __hash__(self):
        return hash((self.task.name, self.key()))

    def __repr__(self):
        values = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"TrialFilter({values})"

    def normalised(self) -> "TrialFilter":
        def value(name, v):
            if self.task.trial_filters[name].kind == "select":
                return tuple(sorted({float(x) for x in v}))
            return bool(v)

        return TrialFilter(self.task, **{k: value(k, v) for k, v in self._values.items()})

    def to_dict(self) -> dict:
        n = self.normalised()
        return {k: list(v) if isinstance(v, tuple) else v for k, v in n._values.items()}

    @classmethod
    def from_dict(cls, raw: dict, task: TaskDefinition | None = None) -> "TrialFilter":
        task = _task(task)
        unknown = sorted(set(raw) - set(task.trial_filters))
        if unknown:
            raise ValueError(
                f"unknown trial filters {unknown}; known: {sorted(task.trial_filters)}"
            )
        return cls(task, **raw).normalised()

    def key(self) -> str:
        """One canonical string per filter, for caches and records."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def active(self) -> list[str]:
        n = self.normalised()
        return [name for name, v in n._values.items() if v]


@dataclass(frozen=True)
class TrialSelection:
    """mask: (n_trials,) kept trials. excluded: reason -> trials failing it (a trial
    failing several filters is counted under each, and once in n_excluded). failed:
    reason -> (n_trials,) bool, which trials failed it (for the Trials workspace)."""

    mask: np.ndarray
    n_total: int
    n_kept: int
    n_excluded: int
    excluded: dict[str, int]
    failed: dict[str, np.ndarray] = field(default_factory=dict)


def available_trial_filters(
    trials: pd.DataFrame, task: TaskDefinition | None = None
) -> dict[str, dict]:
    """name -> {label, kind, available, reason}: offered only when the table has the
    columns."""
    out = {}
    for name, spec in _task(task).trial_filters.items():
        missing = [c for c in dict.fromkeys(spec.columns) if c not in trials]
        out[name] = {
            "label": spec.label,
            "kind": spec.kind,
            "available": not missing,
            "reason": f"trials have no {', '.join(missing)} column" if missing else "",
        }
    return out


def filter_levels(
    trials: pd.DataFrame, name: str, task: TaskDefinition | None = None
) -> list[dict]:
    """A select filter's choices in this table: [{value, name}], sorted by value."""
    spec = _task(task).trial_filters[name]
    values = _filter_values(trials, spec)
    out = []
    for v in sorted({float(v) for v in values[np.isfinite(values)]}):
        if spec.level_names:
            text = spec.level_names.get(v, f"{v:g}")
        elif spec.level_format == "percent":
            text = f"{v * 100:g}%"
        elif spec.level_format:
            text = spec.level_format.format(v=v)
        else:
            text = f"{v:g}"
        out.append({"value": v, "name": text})
    return out


def _filter_values(trials: pd.DataFrame, spec) -> np.ndarray:
    if spec.column is not None:
        return trials[spec.column].to_numpy(np.float64)
    return derive(trials, spec.derivation.name, spec.derivation.columns)[0]


def apply_trial_filter(trials: pd.DataFrame, f: TrialFilter) -> TrialSelection:
    """Which trials the filter keeps, with every exclusion counted by reason."""
    f = f.normalised()
    offered = available_trial_filters(trials, f.task)
    for name in f.active():
        if not offered[name]["available"]:
            raise ValueError(f"cannot filter by {name}: {offered[name]['reason']}")
    n = len(trials)
    reasons: dict[str, np.ndarray] = {}
    for name in f.active():
        spec, chosen = f.task.trial_filters[name], getattr(f, name)
        if spec.kind == "flag":
            reasons[spec.reason] = ~trials[spec.column].fillna(False).to_numpy(bool)
        elif spec.kind == "exclude_values":
            reasons[spec.reason] = np.isin(_filter_values(trials, spec), spec.values)
        else:
            values = _filter_values(trials, spec)
            if spec.undefined_reason:
                undefined = np.isnan(values)
                reasons[spec.undefined_reason] = undefined
                reasons[spec.reason] = ~undefined & ~np.isin(values, chosen)
            else:
                reasons[spec.reason] = ~np.isin(values, chosen)
    dropped = np.zeros(n, bool)
    for failed in reasons.values():
        dropped |= failed
    excluded = {r: int(v.sum()) for r, v in reasons.items() if v.any()}
    failed = {r: v for r, v in reasons.items() if v.any()}
    return TrialSelection(~dropped, n, int((~dropped).sum()), int(dropped.sum()), excluded, failed)
