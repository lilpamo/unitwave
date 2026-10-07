"""Step 9: guided recipes. A recipe is a YAML file under configs/recipes/: the question,
what it needs from the task and the data, and steps that each run one of Studio's
manual analyses. Its wording takes the task's words and the configs' numbers; a recipe
a session can't support is offered greyed out, with the reason. No data is read here."""

import copy

import pytest
import yaml

from unitwave.analysis.recipes import (
    RECIPES_DIR,
    list_recipes,
    load_recipe,
    recipe_from_dict,
    resolve,
)
from unitwave.analysis.tasks import load_task

NAMES = ["stimulus_beyond_movement", "choice_beyond_stimulus", "regions_differ"]
REGIONS_MISSING = "Utah arrays have no atlas regions"


def test_the_three_approved_recipes_load_in_order():
    assert [r.name for r in list_recipes()] == NAMES
    assert [len(load_recipe(n).steps) for n in NAMES] == [3, 3, 2]


def _raw(name="stimulus_beyond_movement") -> dict:
    return yaml.safe_load((RECIPES_DIR / f"{name}.yaml").read_text())


def test_a_malformed_recipe_is_refused():
    def broken(change, name="stimulus_beyond_movement"):
        raw = copy.deepcopy(_raw(name))
        change(raw)
        return raw

    cases = [
        (lambda r: r.update(colour="red"), "unknown keys \\['colour'\\]"),
        (lambda r: r["steps"][0].pop("why"), "missing keys \\['why'\\]"),
        (lambda r: r["steps"][0].update(question="Does the {stimulis} matter?"), "stimulis"),
        (lambda r: r["steps"][0]["run"].update(kind="guess"), "unknown analysis 'guess'"),
        (lambda r: r["needs"].update(weather=True), "unknown needs \\['weather'\\]"),
        (lambda r: r["steps"][1].update(id=r["steps"][0]["id"]), "step ids repeat"),
        # Locking needs the task's movement events, which a recipe must declare.
        (lambda r: r["needs"].pop("movement"), "movement_locking needs 'movement'"),
        (lambda r: r["steps"][0].update(read={"points": [{"lead": "x"}]}), "lead and text"),
    ]
    for change, message in cases:
        with pytest.raises(ValueError, match=message):
            recipe_from_dict(broken(change))
    raw = broken(
        lambda r: r["steps"].append(dict(r["steps"][0], id="z", run={"kind": "selectivity"}))
    )
    with pytest.raises(ValueError, match="selectivity needs 'comparison'"):
        recipe_from_dict(raw)


IBL_DECODING = {"choice": ("choice", "the 100 ms before the first movement")}


def _offered(name, task, regions=None, decoding=None):
    return resolve(load_recipe(name), load_task(task), regions=regions, decoding=decoding or {})


def test_each_recipe_is_offered_where_the_approved_wording_says():
    expected = {
        "stimulus_beyond_movement": {"ibl", "nlb_mc_maze"},
        "choice_beyond_stimulus": {"ibl", "steinmetz"},
        "regions_differ": {"ibl", "steinmetz", "allen_drifting_gratings"},
    }
    tasks = ("ibl", "steinmetz", "nlb_mc_maze", "allen_drifting_gratings")
    for name, where in expected.items():
        for task in tasks:
            regions = REGIONS_MISSING if task == "nlb_mc_maze" else None
            r = _offered(name, task, regions)
            assert r["available"] == (task in where), (name, task, r["reason"])
            assert bool(r["reason"]) != r["available"]


def test_refusals_say_why_in_the_tasks_words():
    r = _offered("stimulus_beyond_movement", "steinmetz")
    assert (
        "Steinmetz et al. 2019 visual discrimination definition declares no movement" in r["reason"]
    )
    r = _offered("choice_beyond_stimulus", "allen_drifting_gratings")
    assert "no comparison on choice" in r["reason"]
    r = _offered("regions_differ", "nlb_mc_maze", REGIONS_MISSING)
    assert r["reason"] == f"No brain regions: {REGIONS_MISSING}."
    # Step 3 needs a decoding target; without one, steps 1 and 2 still run.
    r = _offered(
        "choice_beyond_stimulus",
        "ibl",
        decoding={"choice": "Decoding needs a Brain Wide Map session."},
    )
    assert r["available"] and [s["available"] for s in r["steps"]] == [True, True, False]
    assert r["steps"][2]["reason"] == "Decoding needs a Brain Wide Map session."


def _text(resolved) -> str:
    """Every word a resolved recipe shows."""
    out = [resolved["title"]]
    for s in resolved["steps"]:
        out.append(s["title"])
        for part in ("question", "why", "analysis", "null_and_correction", "needs", "read"):
            out.append(s[part]["text"])
            out += [p["lead"] + " " + p["text"] for p in s[part]["points"]]
    return "\n".join(out)


def test_wording_takes_the_tasks_words_and_the_configs_numbers():
    maze = _offered("stimulus_beyond_movement", "nlb_mc_maze")
    assert maze["title"] == "Which units respond to the go cue beyond the movement that follows it?"
    text = _text(maze)
    assert "Reaction times permuted 10,000 times within maze strata" in text
    assert "rate 0 to 0.3 s after" in text and "movement onset" in text
    assert "{" not in text and "[" not in text
    ibl = _text(_offered("choice_beyond_stimulus", "ibl", decoding=IBL_DECODING))
    assert "within signed contrast strata" in ibl and "the 100 ms before the first movement" in ibl
    assert "left (+1)" in ibl and "right (-1)" in ibl
    steinmetz = _text(
        _offered(
            "choice_beyond_stimulus",
            "steinmetz",
            decoding={"choice": ("task:choice", "the 100 ms before the response")},
        )
    )
    assert "within contrasts strata" in steinmetz and "before the response" in steinmetz


def test_an_unavailable_recipe_still_reads_in_plain_words():
    text = _text(_offered("stimulus_beyond_movement", "allen_drifting_gratings"))
    assert "{" not in text and "the stimulus" in text


def test_steps_name_the_manual_analysis_they_run():
    runs = [s["run"] for s in _offered("stimulus_beyond_movement", "ibl")["steps"]]
    assert runs == [
        {"kind": "responsiveness", "event": "stim_on", "movement_free": False},
        {"kind": "movement_locking"},
        {"kind": "responsiveness", "event": "stim_on", "movement_free": True},
    ]
    runs = [
        s["run"]
        for s in _offered(
            "choice_beyond_stimulus", "steinmetz", decoding={"choice": ("task:choice", "w")}
        )["steps"]
    ]
    assert runs == [
        {"kind": "split_view", "event": "stim_on", "split": "choice"},
        {"kind": "selectivity", "event": "stim_on", "split": "choice"},
        {"kind": "decoding", "target": "task:choice"},
    ]
    runs = [s["run"] for s in _offered("regions_differ", "allen_drifting_gratings")["steps"]]
    assert runs == [
        {"kind": "choose_sessions"},
        {"kind": "region_summary", "label": "responsive", "event": "stim_on"},
    ]
