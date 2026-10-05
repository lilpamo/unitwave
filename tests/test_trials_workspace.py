"""The Trials workspace's strip: every trial at a glance, kept or filtered out (with why),
its reaction time and its level of a chosen condition, coloured as the page's splits are.
The page only draws it (/api/trials). Trials here are test inputs, except the last test,
which reads the cached IBL session."""

import json

import numpy as np
import pandas as pd
import pytest

from unitwave.analysis.conditions import TrialFilter, apply_trial_filter
from unitwave.analysis.tasks import load_task


def test_each_trial_keeps_the_reasons_it_failed():
    trials = pd.DataFrame(
        {
            "intervals_0": np.arange(6.0),
            "intervals_1": np.arange(6.0) + 0.9,
            "bwm_include": [True, False, True, False, True, True],
            "choice": [1.0, 1.0, 0.0, 0.0, -1.0, 1.0],
        }
    )
    f = TrialFilter.from_dict({"bwm_include": True, "exclude_nogo": True}, load_task("ibl"))
    sel = apply_trial_filter(trials, f)
    assert sel.mask.tolist() == [True, False, False, False, True, True]
    assert {k: v.tolist() for k, v in sel.failed.items()} == {
        "not bwm_include": [False, True, False, True, False, False],
        "no-go": [False, False, True, True, False, False],
    }
    assert sel.excluded == {"not bwm_include": 2, "no-go": 2}  # counts as before


EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"


@pytest.fixture(scope="module")
def studio():
    from unitwave.data.load import load_data_config, load_session
    from unitwave.qc.units import load_qc_config
    from unitwave.studio.project import Source
    from unitwave.studio.server import Studio

    try:
        session = load_session(EID, "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")
    return Studio(
        session,
        load_qc_config(),
        load_data_config().data_root / "atlas",
        Source(kind="ibl", eid=EID, backend="bwm"),
    )


TF = json.dumps({"bwm_include": True, "exclude_nogo": True})


def test_the_strip_lists_every_trial_with_why_it_was_filtered(studio):
    from unitwave.analysis.movement import reaction_times

    d = studio.trials_json({"tf": TF, "colour": "choice", "theme": "light"})
    trials = studio.session.trials
    assert d["n"] == len(trials) == len(d["trials"]) and d["n_kept"] == 290
    kept = [t for t in d["trials"] if t["kept"]]
    assert len(kept) == 290 and d["first_kept"] == kept[0]["i"]
    dropped = next(t for t in d["trials"] if not t["kept"])
    assert dropped["why"] and set(dropped["why"]) <= {"not bwm_include", "no-go"}
    rt = reaction_times(trials, load_task("ibl"))
    for t in d["trials"][:50]:
        expected = None if np.isnan(rt[t["i"]]) else float(rt[t["i"]])
        assert t["rt_s"] == expected
    # Coloured by choice, with the page's split colours and the task's level names.
    assert d["condition"] == {"name": "choice", "label": "Choice"}
    assert [lv["name"] for lv in d["levels"]] == ["right (-1)", "left (+1)"]
    assert [lv["colour"] for lv in d["levels"]] == studio._colours("choice", (-1.0, 1.0), "light")
    assert {t["level"] for t in d["trials"]} <= {-1.0, 1.0, None}  # no-go: no level
    assert d["rt_label"] == "first movement − stimulus onset"


def test_the_strip_s_default_colour_is_the_split_then_the_outcome(studio):
    assert studio.trials_json({"tf": TF, "split": "side"})["condition"]["name"] == "side"
    assert studio.trials_json({"tf": TF})["condition"]["name"] == "outcome"
    assert studio.trials_json({"tf": TF, "colour": "none"})["condition"] is None
    with pytest.raises(ValueError, match="unknown condition 'mood'"):
        studio.trials_json({"tf": TF, "colour": "mood"})
