"""Step 9: Studio's Recipes panel. Each step runs the manual view's own method with the
manual view's parameters, so its numbers are the manual ones; a step the session can't
run is refused with the reason. The first test uses a hand-built Phy folder; the others
read downloaded files."""

import json
from pathlib import Path

import pandas as pd
import pytest
from test_studio_server import _studio as phy_studio

from unitwave.data.load import load_data_config
from unitwave.studio.project import Source, load_source
from unitwave.studio.server import Studio

MC_MAZE = (
    Path.home()
    / "data/neurodecoder/dandi/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
)
RICHARDS = (
    Path.home() / "data/neurodecoder/dandi/000017/sub-Richards/sub-Richards_ses-20171031T120000.nwb"
)


def test_a_phy_folder_is_offered_what_it_supports_and_told_why_not(tmp_path):
    studio = phy_studio(tmp_path)
    recipes = {r["name"]: r for r in studio.recipes_json({})["recipes"]}
    assert list(recipes) == ["stimulus_beyond_movement", "choice_beyond_stimulus", "regions_differ"]
    assert (
        recipes["regions_differ"]["reason"] == "No brain regions: Phy folders have no brain region."
    )
    choice = recipes["choice_beyond_stimulus"]
    assert choice["available"] and not choice["steps"][2]["available"]
    assert "Brain Wide Map session" in choice["steps"][2]["reason"]
    with pytest.raises(ValueError, match="Brain Wide Map session"):
        studio.recipe_run({"recipe": "choice_beyond_stimulus", "step": "decode"})
    with pytest.raises(ValueError, match="No brain regions"):
        studio.recipe_run({"recipe": "regions_differ", "step": "summary"})
    with pytest.raises(ValueError, match="no step 'nope'"):
        studio.recipe_run({"recipe": "choice_beyond_stimulus", "step": "nope"})


def _nwb(path, layout, task) -> Studio:
    source = Source(kind="nwb", file=str(path), layout=layout, task=task)
    session, qc = load_source(source)
    return Studio(session, qc, load_data_config().data_root / "atlas", source)


PAGE = {"all": "0", "probe": "", "tf": "{}"}


def _only(store: dict) -> pd.DataFrame:
    assert len(store) == 1
    return next(iter(store.values()))


@pytest.mark.skipif(not MC_MAZE.exists(), reason="MC_Maze_Small isn't downloaded")
def test_recipe_1_gives_the_manual_numbers_on_mc_maze():
    steps = [
        ("respond", "test_json", {"event": "go_cue", "movement_free": "0"}, "_tests"),
        ("locked", "locking_json", {}, "_locking"),
        ("movement_free", "test_json", {"event": "go_cue", "movement_free": "1"}, "_tests"),
    ]
    for step, method, manual, store in steps:
        by_recipe, by_hand = (_nwb(MC_MAZE, "nlb_mc_maze", "nlb_mc_maze") for _ in range(2))
        out = by_recipe.recipe_run({"recipe": "stimulus_beyond_movement", "step": step, **PAGE})
        assert out["kind"] in ("responsiveness", "movement_locking")
        assert out["query"] == {**PAGE, **manual}
        assert out["result"] == getattr(by_hand, method)({**PAGE, **manual})
        pd.testing.assert_frame_equal(
            _only(getattr(by_recipe, store)), _only(getattr(by_hand, store))
        )
    assert out["result"]["movement_free"] and out["result"]["n_trials"] == 69


@pytest.fixture(scope="module")
def richards():
    if not RICHARDS.exists():
        pytest.skip("the Steinmetz Richards file isn't downloaded")
    return _nwb(RICHARDS, "steinmetz_2019", "steinmetz")


def test_recipe_2_on_steinmetz_runs_the_manual_views(richards):
    tf = json.dumps({"included": True})
    page = {**PAGE, "tf": tf}
    view = richards.recipe_run({"recipe": "choice_beyond_stimulus", "step": "split", **page})
    assert view == {"kind": "split_view", "query": {**page, "event": "stim_on", "split": "choice"}}
    out = richards.recipe_run({"recipe": "choice_beyond_stimulus", "step": "selective", **page})
    by_hand = _nwb(RICHARDS, "steinmetz_2019", "steinmetz")
    assert out["result"] == by_hand.selectivity_json(
        {**page, "event": "stim_on", "split": "choice"}
    )
    pd.testing.assert_frame_equal(_only(richards._selectivity), _only(by_hand._selectivity))
    decode = richards.recipe_run({"recipe": "choice_beyond_stimulus", "step": "decode", **page})
    assert decode == {"kind": "decoding", "query": page, "target": "task:choice"}
    steps = {r["name"]: r for r in richards.recipes_json({})["recipes"]}["choice_beyond_stimulus"]
    assert "the 100 ms before the response" in steps["steps"][2]["analysis"]["points"][0]["text"]


def test_recipe_3_gives_the_command_for_this_file(richards, tmp_path):
    out = richards.recipe_run({"recipe": "regions_differ", "step": "summary", **PAGE})
    assert out["kind"] == "region_summary"
    command = out["command"]
    assert command.startswith("python -m unitwave.cli.summarise --nwb ")
    assert str(RICHARDS) in command and "--layout steinmetz_2019" in command
    assert "--label responsive --event stim_on" in command
    assert out["label"] == "responsive to stimulus onset"
    assert all(r["task"] == "steinmetz" for r in out["runs"])  # not IBL's runs of the same label
