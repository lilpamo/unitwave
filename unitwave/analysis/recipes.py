"""Guided recipes (step 9): one scientific question, answered by a sequence of Studio's
own analyses. A recipe is a YAML file under configs/recipes/; its wording was approved
by the user (docs/proposals/recipe_wording.md).

- **What a recipe declares (`needs`):** what it requires from the task and the data:
  - `stimulus`: `movement` (the task's movement section names it) or an event name;
  - `movement`: the task's movement events, for reaction times;
  - `comparison`: a two-level comparison on a condition, whose null permutes within
    strata (so the stimulus is held fixed);
  - `regions`: brain regions in the Allen mouse atlas.
  A step may also need a decoding target on the comparison's condition.
- **Each step runs one manual analysis (`run.kind`):** responsiveness (optionally
  on movement-free trials only), movement_locking, split_view (descriptive), selectivity,
  decoding, choose_sessions and region_summary (on the command line). `resolve` turns a
  step into the manual view's own parameters; Studio runs them through the same method.
- **Wording:** six parts per step (question, why, analysis, null_and_correction, needs, read), each
  a text and optional points with a bold lead. `{placeholders}` take the task's words
  (event and condition labels, strata, level names) and the configs' numbers, so no
  number is typed into a recipe. A recipe the session can't support still reads in
  plain words (generic fallbacks), greyed out with the reason.
- **Refused when malformed:** unknown keys, missing parts, unknown placeholders or
  analyses, repeated step ids, an analysis whose needs aren't declared.
"""

import os
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from unitwave.analysis.movement import load_movement_config
from unitwave.analysis.responsiveness import load_response_config
from unitwave.analysis.summary import load_summary_config
from unitwave.analysis.tasks import TaskDefinition
from unitwave.analysis.tuning import load_selectivity_config

RECIPES_DIR = Path(__file__).resolve().parents[2] / "configs" / "recipes"
PARTS = ("question", "why", "analysis", "null_and_correction", "needs", "read")
NEEDS = ("stimulus", "movement", "comparison", "regions")
# analysis kind -> (the recipe needs it requires, its own options)
KINDS = {
    "responsiveness": (("stimulus",), {"movement_free"}),
    "movement_locking": (("movement",), set()),
    "split_view": (("stimulus", "comparison"), set()),
    "selectivity": (("stimulus", "comparison"), set()),
    "decoding": (("comparison",), set()),
    "choose_sessions": (("regions",), set()),
    "region_summary": (("stimulus", "regions"), {"label"}),
}
_RECIPE_KEYS = {"name", "number", "title", "needs", "steps"}
_STEP_KEYS = {"id", "title", "run", *PARTS}
# Placeholder -> its plain-words fallback, for a recipe the session can't support.
TASK_WORDS = {
    "stimulus": "stimulus",
    "stimulus_onset": "stimulus onset",
    "movement": "first movement",
    "rt_strata": "stimulus",
    "choice": "choice",
    "choice_a": "one choice",
    "choice_b": "the other",
    "choice_strata": "stimulus",
    "choice_excluded": "trials without a choice",
    "decoding_window": "a window before the movement",
}


def _thousands(n: int) -> str:
    return f"{n:,}"


def _seconds(x: float) -> str:
    return f"{x:g} s"


@lru_cache(maxsize=1)
def config_numbers() -> dict[str, str]:
    """Placeholder -> a number from the configs, as the wording shows it. The wording
    reads windows as "the X s after/before the event", so each must touch the event."""
    from unitwave.analysis.decoding import load_decoding_config
    from unitwave.evaluation.nulls import load_null_config

    response, lock = load_response_config(), load_movement_config()
    for name, (start, stop), at in (
        ("response window", response.response_window, "start"),
        ("baseline window", response.baseline_window, "stop"),
        ("movement post_window", lock.post_window, "start"),
        ("movement pre_window", lock.pre_window, "stop"),
    ):
        if (start if at == "start" else stop) != 0:
            raise ValueError(f"the recipes' wording assumes the {name} {at}s at the event")
    sel, decoding, summary = (
        load_selectivity_config(),
        load_decoding_config(),
        load_summary_config(),
    )
    return {
        "response_after": _seconds(response.response_window[1]),
        "baseline_before": _seconds(-response.baseline_window[0]),
        "min_shift": _seconds(response.min_shift_s),
        "alpha": f"{response.alpha:g}",
        "lock_after": _seconds(lock.post_window[1]),
        "lock_before": _seconds(-lock.pre_window[0]),
        "lock_permutations": _thousands(lock.n_permutations),
        "lock_seed": str(lock.seed),
        "sel_permutations": _thousands(sel.n_permutations),
        "sel_seed": str(sel.seed),
        "min_trials": str(sel.min_trials),
        "train_percent": f"{decoding.train_fraction:.0%}",
        "test_percent": f"{1 - decoding.train_fraction:.0%}",
        "gap": _seconds(decoding.gap_s),
        "n_shifts": _thousands(decoding.n_shifts),
        "n_bootstrap": _thousands(decoding.n_bootstrap),
        "decoding_alpha": f"{decoding.alpha:g}",
        "history_trials": str(load_null_config().history_trials),
        "min_sessions": str(summary.min_sessions),
        "summary_alpha": f"{summary.alpha:g}",
        "level": summary.level,
    }


_NUMBER_NAMES = (
    "response_after",
    "baseline_before",
    "min_shift",
    "alpha",
    "lock_after",
    "lock_before",
    "lock_permutations",
    "lock_seed",
    "sel_permutations",
    "sel_seed",
    "min_trials",
    "train_percent",
    "test_percent",
    "gap",
    "n_shifts",
    "n_bootstrap",
    "decoding_alpha",
    "history_trials",
    "min_sessions",
    "summary_alpha",
    "level",
)
PLACEHOLDERS = frozenset(TASK_WORDS) | frozenset(_NUMBER_NAMES)


@dataclass(frozen=True)
class Step:
    id: str
    title: str
    run: dict  # {"kind": ..., options}
    parts: dict  # part -> {"text": str, "points": [{"lead", "text"}]}
    needs_decoding: bool


@dataclass(frozen=True)
class Recipe:
    name: str
    number: int
    title: str
    needs: dict
    steps: tuple[Step, ...]


def _need_keys(where: str, raw, keys: set) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping")  # noqa: TRY004 - a refusal, in words
    unknown, missing = sorted(set(raw) - keys), sorted(keys - set(raw))
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}")


def _check_words(where: str, text) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{where}: expected some text")
    names = {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}
    unknown = sorted(names - PLACEHOLDERS)
    if unknown:
        raise ValueError(f"{where}: unknown placeholders {unknown}")
    return " ".join(text.split())


def _part(where: str, raw) -> dict:
    """A part: text, or {text, points}; a point is text or {lead, text}."""
    if isinstance(raw, str):
        return {"text": _check_words(where, raw), "points": []}
    if not isinstance(raw, dict) or not set(raw) <= {"text", "points"} or "points" not in raw:
        raise ValueError(f"{where}: give text, or text and points")
    points = []
    for i, p in enumerate(raw["points"]):
        if isinstance(p, str):
            points.append({"lead": "", "text": _check_words(f"{where} point {i + 1}", p)})
        elif isinstance(p, dict) and set(p) == {"lead", "text"}:
            points.append(
                {
                    "lead": _check_words(f"{where} point {i + 1}", p["lead"]),
                    "text": _check_words(f"{where} point {i + 1}", p["text"]),
                }
            )
        else:
            raise ValueError(f"{where} point {i + 1}: a point is text, or a lead and text")
    text = _check_words(where, raw["text"]) if "text" in raw else ""
    return {"text": text, "points": points}


def recipe_from_dict(raw: dict) -> Recipe:
    """A validated recipe; anything malformed is refused, saying where and why."""
    _need_keys("recipe", raw, _RECIPE_KEYS)
    name = str(raw["name"])
    needs = raw["needs"] or {}
    unknown = sorted(set(needs) - set(NEEDS))
    if unknown:
        raise ValueError(f"recipe {name}: unknown needs {unknown}; known: {list(NEEDS)}")
    if "stimulus" in needs and not isinstance(needs["stimulus"], str):
        raise ValueError(f"recipe {name}: stimulus is 'movement' or an event name")
    if "comparison" in needs and not isinstance(needs["comparison"], str):
        raise ValueError(f"recipe {name}: comparison names a condition")
    steps, ids = [], set()
    for i, s in enumerate(raw["steps"] or []):
        where = f"recipe {name}, step {i + 1}"
        keys = set(s) if isinstance(s, dict) else set()
        _need_keys(where, s, _STEP_KEYS | ({"needs_decoding"} & keys))
        if s["id"] in ids:
            raise ValueError(f"recipe {name}: step ids repeat ({s['id']})")
        ids.add(s["id"])
        run = dict(s["run"] or {})
        kind = run.pop("kind", None)
        if kind not in KINDS:
            raise ValueError(f"{where}: unknown analysis {kind!r}; known: {sorted(KINDS)}")
        required, options = KINDS[kind]
        for need in required:
            if need not in needs:
                raise ValueError(f"{where}: {kind} needs {need!r} in the recipe's needs")
        if set(run) - options:
            raise ValueError(f"{where}: {kind} takes {sorted(options) or 'no options'}")
        steps.append(
            Step(
                id=str(s["id"]),
                title=_check_words(f"{where} title", s["title"]),
                run={"kind": kind, **run},
                parts={p: _part(f"{where} {p}", s[p]) for p in PARTS},
                needs_decoding=bool(s.get("needs_decoding", False)) or kind == "decoding",
            )
        )
    if not steps:
        raise ValueError(f"recipe {name}: no steps")
    return Recipe(
        name,
        int(raw["number"]),
        _check_words(f"recipe {name} title", raw["title"]),
        dict(needs),
        tuple(steps),
    )


@lru_cache(maxsize=16)
def _load(path: str, mtime: float) -> Recipe:
    return recipe_from_dict(yaml.safe_load(Path(path).read_text()))


def load_recipe(name: str | os.PathLike) -> Recipe:
    """A built-in recipe by name, or one read from a YAML file's path."""
    path = Path(name)
    if path.suffix not in (".yaml", ".yml"):
        path = RECIPES_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"no recipe {str(name)!r}")
    return _load(str(path.resolve()), path.stat().st_mtime)


def list_recipes() -> list[Recipe]:
    """The built-in recipes, by number."""
    return sorted((load_recipe(p) for p in RECIPES_DIR.glob("*.yaml")), key=lambda r: r.number)


def window_words(start_s: float, stop_s: float, event: str) -> str:
    """A decoding window in words, e.g. "the 100 ms before the first movement"."""
    if stop_s == 0 and start_s < 0:
        return f"the {-start_s * 1000:.0f} ms before the {event}"
    if start_s == 0 and stop_s > 0:
        return f"the {stop_s * 1000:.0f} ms after the {event}"
    return f"{start_s * 1000:+.0f} to {stop_s * 1000:+.0f} ms from the {event}"


def _stimulus_event(recipe: Recipe, task: TaskDefinition) -> str | None:
    want = recipe.needs.get("stimulus")
    if want == "movement":
        return task.movement.stimulus if task.movement is not None else None
    return want if want in task.events else None


def _level_name(spec, value: float) -> str:
    if float(value) in spec.level_names:
        return str(spec.level_names[float(value)])
    if spec.level_format and spec.level_format != "percent":
        return spec.level_format.format(v=value)
    return f"{value:g}"


def _availability(recipe: Recipe, task: TaskDefinition, regions: str | None) -> str:
    """'' when the task and data have what the recipe needs; else why not, in words."""
    needs = recipe.needs
    if needs.get("regions") and regions is not None:
        return f"No brain regions: {regions.rstrip('.')}."
    if "comparison" in needs:
        c = task.comparisons.get(needs["comparison"])
        if c is None:
            return f"The {task.label} definition has no comparison on {needs['comparison']}."
        if c.kind != "two_level" or not c.permute_within:
            return (
                f"The {task.label} definition's {needs['comparison']} comparison doesn't "
                "permute within strata, so it can't hold the stimulus fixed."
            )
    if needs.get("movement") and task.movement is None:
        return f"The {task.label} definition declares no movement events."
    if "stimulus" in needs and _stimulus_event(recipe, task) is None:
        if needs["stimulus"] == "movement":
            return f"The {task.label} definition declares no movement events."
        return f"The {task.label} definition has no {needs['stimulus']!r} event."
    return ""


def _task_words(recipe: Recipe, task: TaskDefinition, decoding: dict) -> dict[str, str]:
    """Placeholder -> the task's word for it; the plain fallback where it has none."""
    words = dict(TASK_WORDS)
    event = _stimulus_event(recipe, task)
    if event is not None:
        label = task.events[event].label.lower()
        words["stimulus_onset"] = label
        words["stimulus"] = label.removesuffix(" onset")
    if task.movement is not None:
        words["movement"] = task.events[task.movement.movement].label.lower()
        words["rt_strata"] = task.movement.strata.replace("_", " ")
    name = recipe.needs.get("comparison")
    c = task.comparisons.get(name) if name else None
    if c is not None and c.kind == "two_level":
        spec = task.conditions[name]
        words["choice"] = spec.label.lower()
        words["choice_a"], words["choice_b"] = (_level_name(spec, v) for v in c.levels)
        words["choice_excluded"] = spec.excluded
        if c.permute_within:
            words["choice_strata"] = c.permute_within.replace("_", " ")
        found = decoding.get(name)
        if isinstance(found, tuple):
            words["decoding_window"] = found[1]
    return words


def _sentence(text: str) -> str:
    """A part's text or a point's lead may open with a placeholder: capitalise it."""
    return text[:1].upper() + text[1:]


def _fill(part: dict, words: dict) -> dict:
    def point(p):
        if not p["lead"]:
            return {"lead": "", "text": _sentence(p["text"].format_map(words))}
        return {"lead": _sentence(p["lead"].format_map(words)), "text": p["text"].format_map(words)}

    return {
        "text": _sentence(part["text"].format_map(words)),
        "points": [point(p) for p in part["points"]],
    }


def resolve(recipe: Recipe, task: TaskDefinition, *, regions: str | None, decoding: dict) -> dict:
    """The recipe as one session shows it: its wording in the task's words and the
    configs' numbers, whether it (and each step) is available and why not, and each
    step's manual analysis with its parameters.

    regions: None when the session has brain regions, else why not.
    decoding: condition -> (target id, window in words) when a decoding target exists,
    or why not (text)."""
    words = {**_task_words(recipe, task, decoding), **config_numbers()}
    reason = _availability(recipe, task, regions)
    event = _stimulus_event(recipe, task)
    name = recipe.needs.get("comparison")
    steps = []
    for s in recipe.steps:
        run = dict(s.run)
        kind = run["kind"]
        if kind in ("responsiveness", "split_view", "selectivity", "region_summary"):
            run["event"] = event
        if kind == "responsiveness":
            run["movement_free"] = bool(run.get("movement_free", False))
        if kind in ("split_view", "selectivity"):
            run["split"] = name
        step_reason = reason
        if kind == "decoding" or s.needs_decoding:
            found = decoding.get(name, f"No decoding target on {name} in this session.")
            if isinstance(found, tuple):
                run["target"] = found[0]
            elif not step_reason:
                step_reason = str(found)
            run.setdefault("target", None)
        steps.append(
            {
                "id": s.id,
                "title": _sentence(s.title.format_map(words)),
                "available": not step_reason,
                "reason": step_reason,
                "run": run,
                **{p: _fill(s.parts[p], words) for p in PARTS},
            }
        )
    return {
        "name": recipe.name,
        "number": recipe.number,
        "title": _sentence(recipe.title.format_map(words)),
        "available": not reason,
        "reason": reason,
        "steps": steps,
    }
