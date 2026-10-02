"""Task definitions (step 3): what a task's trials table means, declared per task.

A definition is a YAML file in configs/tasks/ (or any path). It declares:
- events: time columns to align on, each optionally limited to trials where another
  column equals a value (IBL's reward and error feedback);
- conditions: columns, or named derivations of columns, to split trials by, with type
  (categorical, ordinal or continuous), level names and the reason trials are
  excluded;
- strata: what nulls permute within (a condition, or a derivation);
- comparisons: the two-level comparisons offered for selectivity, each with its null:
  a permutation within declared strata, a plain permutation, or pseudo-sessions from
  a known block generator (only IBL's exists). The key is `null_model`: a bare `null`
  key reads as YAML's null;
- movement: the stimulus and movement events, and the strata reaction times are
  permuted within;
- behaviour: the session's time series the task uses;
- trialstruct: what the no-spikes decoding baseline may use (step 8);
- trial_filters: the filters offered (keep flagged trials, drop values, or keep chosen
  levels of a column or derivation, named by `level_names` or `level_format`).

Level formats: "percent" is the value × 100 with a % sign (a condition's also carries
the sign: "+25%"); any other format is a Python format string of `v`, e.g.
"p(left) {v:g}".

Derivations are code (DERIVATIONS below), each tested by hand, so a definition declares
structure, never formulas. IBL is one built-in definition (configs/tasks/ibl.yaml),
read exactly as the app always read IBL's trials (tests/test_task_golden.py).
`required_columns` must all be present, or the definition is refused naming them; any
other feature whose columns are missing is unavailable, and says which.
"""

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TASKS_DIR = Path(__file__).resolve().parents[2] / "configs" / "tasks"
DEFAULT_TASK = "ibl"
TYPES = ("categorical", "ordinal", "continuous")
FILTER_KINDS = ("flag", "exclude_values", "select")
GENERATORS = ("ibl_blocks",)  # pseudo-session block generators known (evaluation/nulls.py)
_REQUIRED = {"name", "label", "required_columns", "events", "conditions"}
_OPTIONAL = {
    "trial_view_events",
    "strata",
    "comparisons",
    "movement",
    "behaviour",
    "trialstruct",
    "trial_filters",
}


# ---------- derivations: named functions of trials columns ----------
def _one_side(trials, columns):
    left = trials[columns[0]].to_numpy(np.float64)
    right = trials[columns[1]].to_numpy(np.float64)
    return left, right, np.isnan(left) != np.isnan(right)


def _signed_contrast(trials, columns):
    """right - left with NaN read as 0; NaN unless exactly one side is set."""
    left, right, valid = _one_side(trials, columns)
    out = np.nan_to_num(right) - np.nan_to_num(left) + 0.0  # + 0.0 turns -0.0 into 0.0
    out[~valid] = np.nan
    return out, None


def _stimulus_side(trials, columns):
    """+1 right, -1 left; NaN unless exactly one side is set."""
    _, right, valid = _one_side(trials, columns)
    return np.where(valid, np.where(np.isnan(right), -1.0, 1.0), np.nan), None


def _signed_contrast_zero_by_side(trials, columns):
    """Signed contrast with zero split by side (-0% and +0% differ), for strata: at 0%
    the rewarded side still differs, and choice still tracks it."""
    return _signed_contrast(trials, columns)[0] + 1e-9 * _stimulus_side(trials, columns)[0], None


def _absolute_contrast(trials, columns):
    return np.abs(_signed_contrast(trials, columns)[0]), None


def _median_split_difference(trials, columns):
    """first - second column, split at its median: 0 early, 1 late; NaN where either is
    missing. Level names carry the median."""
    d = trials[columns[0]].to_numpy(np.float64) - trials[columns[1]].to_numpy(np.float64)
    median = float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan
    values = np.where(np.isfinite(d), (d >= median).astype(np.float64), np.nan)
    ms = f"{median * 1000:.0f} ms"
    return values, {0.0: f"early (< {ms})", 1.0: f"late (≥ {ms})"}


# name -> (number of columns, function(trials, columns) -> (values, level names or None))
DERIVATIONS: dict[str, tuple[int, Callable]] = {
    "signed_contrast": (2, _signed_contrast),
    "stimulus_side": (2, _stimulus_side),
    "signed_contrast_zero_by_side": (2, _signed_contrast_zero_by_side),
    "absolute_contrast": (2, _absolute_contrast),
    "median_split_difference": (2, _median_split_difference),
}


def derive(trials: pd.DataFrame, name: str, columns) -> tuple[np.ndarray, dict | None]:
    """(n_trials,) values of a named derivation, and its level names if it names them."""
    values, names = DERIVATIONS[name][1](trials, list(columns))
    assert values.shape == (len(trials),)
    return values, names


# ---------- the definition ----------
@dataclass(frozen=True)
class Event:
    label: str
    column: str
    when: tuple[str, float] | None = None  # (column, value it must equal)

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.column,) + ((self.when[0],) if self.when else ())


@dataclass(frozen=True)
class Derivation:
    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    type: str
    column: str | None
    derivation: Derivation | None
    missing_values: tuple[float, ...]
    level_names: dict
    level_format: str | None
    excluded: str

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.column,) if self.column else self.derivation.columns


@dataclass(frozen=True)
class StrataSpec:
    condition: str | None
    derivation: Derivation | None


@dataclass(frozen=True)
class Comparison:
    levels: tuple[float, float]
    permute_within: str | None  # strata name, or None
    pseudo_sessions: str | None  # generator name, or None
    window: str  # "response" or "baseline"


@dataclass(frozen=True)
class Movement:
    stimulus: str
    movement: str
    strata: str


@dataclass(frozen=True)
class FilterSpec:
    label: str
    kind: str
    column: str | None
    derivation: Derivation | None
    values: tuple[float, ...]
    reason: str
    undefined_reason: str
    level_names: dict = field(default_factory=dict)
    level_format: str | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.column,) if self.column else self.derivation.columns


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    label: str
    required_columns: tuple[str, ...]
    events: dict
    trial_view_events: dict
    conditions: dict
    strata: dict
    comparisons: dict
    movement: Movement | None
    wheel: str | None
    traces: tuple[str, ...]
    trialstruct: tuple[str, ...]
    trial_filters: dict
    raw: dict = field(repr=False, compare=False, default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()

    @property
    def time_columns(self) -> tuple[str, ...]:
        """Trials columns holding times, in seconds on the session clock."""
        events = [e.column for e in self.events.values()]
        return tuple(dict.fromkeys([*events, *self.trial_view_events]))

    @property
    def columns(self) -> tuple[str, ...]:
        """Every trials column the definition reads."""
        cols = [*self.required_columns, *self.time_columns]
        cols += [c for e in self.events.values() for c in e.columns]
        cols += [c for spec in self.conditions.values() for c in spec.columns]
        cols += [c for name in self.strata for c in self.strata_columns(name)]
        cols += [c for f in self.trial_filters.values() for c in f.columns]
        cols += list(self.trialstruct)
        return tuple(dict.fromkeys(cols))

    def strata_columns(self, name: str) -> tuple[str, ...]:
        s = self.strata[name]
        return self.conditions[s.condition].columns if s.condition else s.derivation.columns

    def require(self, trials: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in trials]
        if missing:
            s = "" if len(missing) == 1 else "s"
            raise ValueError(
                f"The trials table has no {', '.join(missing)} column{s}, which the task "
                f"definition {self.label!r} ({self.name!r}) needs. Choose another task "
                f"definition, or add the column{s} to the events file."
            )

    def unavailable(self, trials: pd.DataFrame) -> dict:
        """feature kind -> name -> the columns it needs and the table lacks (only those
        lacking some)."""

        def lacking(columns):
            return [c for c in dict.fromkeys(columns) if c not in trials]

        out = {
            "events": {k: lacking(e.columns) for k, e in self.events.items()},
            "conditions": {k: lacking(c.columns) for k, c in self.conditions.items()},
            "trial_filters": {k: lacking(f.columns) for k, f in self.trial_filters.items()},
        }
        return {kind: {k: v for k, v in d.items() if v} for kind, d in out.items()}


def _need(where: str, raw, keys: set, optional: set = frozenset()) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping")  # noqa: TRY004 - a refusal, in words
    unknown, missing = sorted(set(raw) - keys - set(optional)), sorted(keys - set(raw))
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}")


def _derivation(where: str, raw) -> Derivation:
    _need(f"{where} derivation", raw, {"name", "columns"})
    if raw["name"] not in DERIVATIONS:
        raise ValueError(
            f"{where}: unknown derivation {raw['name']!r} (known: {', '.join(sorted(DERIVATIONS))})"
        )
    n = DERIVATIONS[raw["name"]][0]
    if len(raw["columns"]) != n:
        raise ValueError(
            f"{where}: derivation {raw['name']} takes {n} columns, got {len(raw['columns'])}"
        )
    return Derivation(raw["name"], tuple(str(c) for c in raw["columns"]))


def _column_or_derivation(where: str, raw) -> tuple[str | None, Derivation | None]:
    if ("column" in raw) == ("derive" in raw):
        raise ValueError(f"{where}: give either a column or a derivation")
    if "column" in raw:
        return str(raw["column"]), None
    return None, _derivation(where, raw["derive"])


def task_from_dict(raw: dict) -> TaskDefinition:
    """A validated definition; anything malformed is refused, saying where and why."""
    _need("task definition", raw, _REQUIRED, _OPTIONAL)
    events = {}
    for name, e in raw["events"].items():
        if (
            not isinstance(e, dict)
            or not {"label", "column"} <= set(e)
            or set(e)
            - {
                "label",
                "column",
                "when",
            }
        ):
            raise ValueError(f"event {name}: needs a label and a column (and optionally when)")
        when = None
        if "when" in e:
            w = e["when"]
            if not isinstance(w, dict) or set(w) != {"column", "equals"}:
                raise ValueError(f"event {name}: `when` needs a column and a value to equal")
            when = (str(w["column"]), float(w["equals"]))
        events[name] = Event(str(e["label"]), str(e["column"]), when)

    conditions = {}
    for name, c in raw["conditions"].items():
        where = f"condition {name}"
        _need(
            where,
            c,
            {"label", "type", "excluded"},
            {"column", "derive", "missing_values", "level_names", "level_format"},
        )
        if c["type"] not in TYPES:
            raise ValueError(f"{where}: type must be categorical, ordinal or continuous")
        column, derivation = _column_or_derivation(where, c)
        conditions[name] = ConditionSpec(
            label=str(c["label"]),
            type=c["type"],
            column=column,
            derivation=derivation,
            missing_values=tuple(float(v) for v in c.get("missing_values", [])),
            level_names={float(k): str(v) for k, v in (c.get("level_names") or {}).items()},
            level_format=c.get("level_format"),
            excluded=str(c["excluded"]),
        )

    strata = {}
    for name, s in (raw.get("strata") or {}).items():
        where = f"strata {name}"
        if not isinstance(s, dict) or len(s) != 1 or not set(s) <= {"condition", "derive"}:
            raise ValueError(f"{where}: give either a condition or a derivation")
        if "condition" in s:
            if s["condition"] not in conditions:
                raise ValueError(f"{where}: no condition {s['condition']!r}")
            strata[name] = StrataSpec(s["condition"], None)
        else:
            strata[name] = StrataSpec(None, _derivation(where, s["derive"]))

    comparisons = {}
    for name, c in (raw.get("comparisons") or {}).items():
        where = f"comparison {name}"
        _need(where, c, {"levels", "null_model"}, {"window"})
        if name not in conditions:
            raise ValueError(f"{where}: no condition {name!r} to compare")
        if len(c["levels"]) != 2:
            raise ValueError(f"{where}: needs two levels, got {len(c['levels'])}")
        null = c["null_model"]
        within = generator = None
        if null == {"permute": "all"}:
            pass
        elif isinstance(null, dict) and set(null) == {"permute_within"}:
            within = null["permute_within"]
            if within not in strata:
                raise ValueError(f"{where}: no strata {within!r} to permute within")
        elif isinstance(null, dict) and set(null) == {"pseudo_sessions"}:
            generator = null["pseudo_sessions"]
            if generator not in GENERATORS:
                raise ValueError(
                    f"{where}: no block generator {generator!r} (known: {', '.join(GENERATORS)})"
                )
        else:
            raise ValueError(
                f"{where}: null_model is permute_within (strata), permute: all, or pseudo_sessions"
            )
        window = c.get("window", "response")
        if window not in ("response", "baseline"):
            raise ValueError(f"{where}: window must be response or baseline")
        a, b = (float(v) for v in c["levels"])
        comparisons[name] = Comparison((a, b), within, generator, window)

    movement = None
    if raw.get("movement"):
        m = raw["movement"]
        _need("movement", m, {"stimulus", "movement", "strata"})
        for key in ("stimulus", "movement"):
            if m[key] not in events:
                raise ValueError(f"movement: no event {m[key]!r} for the {key}")
        if m["strata"] not in strata:
            raise ValueError(f"movement: no strata {m['strata']!r}")
        movement = Movement(m["stimulus"], m["movement"], m["strata"])

    behaviour = raw.get("behaviour") or {}
    _need("behaviour", behaviour, set(), {"wheel", "traces"})

    filters = {}
    for name, f in (raw.get("trial_filters") or {}).items():
        where = f"trial filter {name}"
        _need(
            where,
            f,
            {"label", "kind", "reason"},
            {"column", "derive", "values", "undefined_reason", "level_names", "level_format"},
        )
        if f["kind"] not in FILTER_KINDS:
            raise ValueError(f"{where}: kind must be flag, exclude_values or select")
        column, derivation = _column_or_derivation(where, f)
        filters[name] = FilterSpec(
            label=str(f["label"]),
            kind=f["kind"],
            column=column,
            derivation=derivation,
            values=tuple(float(v) for v in f.get("values", [])),
            reason=str(f["reason"]),
            undefined_reason=str(f.get("undefined_reason", "")),
            level_names={float(k): str(v) for k, v in (f.get("level_names") or {}).items()},
            level_format=f.get("level_format"),
        )

    return TaskDefinition(
        name=str(raw["name"]),
        label=str(raw["label"]),
        required_columns=tuple(str(c) for c in raw["required_columns"]),
        events=events,
        trial_view_events={str(k): str(v) for k, v in (raw.get("trial_view_events") or {}).items()},
        conditions=conditions,
        strata=strata,
        comparisons=comparisons,
        movement=movement,
        wheel=behaviour.get("wheel"),
        traces=tuple(behaviour.get("traces") or ()),
        trialstruct=tuple(raw.get("trialstruct") or ()),
        trial_filters=filters,
        raw=raw,
    )


def list_tasks() -> list[dict]:
    """The built-in definitions: name, label and file."""
    rows = []
    for path in sorted(TASKS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        rows.append(
            {"name": raw.get("name", path.stem), "label": raw.get("label", ""), "file": path.name}
        )
    return rows


@lru_cache(maxsize=16)
def _load(path: str, mtime: float) -> TaskDefinition:
    return task_from_dict(yaml.safe_load(Path(path).read_text()) or {})


def task_path(name: str | os.PathLike = DEFAULT_TASK) -> Path:
    """The YAML file of a built-in definition (by name) or of a definition's path."""
    path = Path(name)
    if path.suffix not in (".yaml", ".yml"):
        path = TASKS_DIR / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(t["name"] for t in list_tasks())
        raise ValueError(f"no task definition {str(name)!r} (built in: {known})")
    return path.resolve()


def load_task(name: str | os.PathLike = DEFAULT_TASK) -> TaskDefinition:
    """A built-in definition by name, or one read from a YAML file's path."""
    path = task_path(name)
    return _load(str(path), path.stat().st_mtime)
