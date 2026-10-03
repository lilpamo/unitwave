"""Step 8b: population trajectories (and every view coloured by condition) for any task.

Trajectories are PCA on condition averages, fit on odd trials and shown on even ones,
with components named pc_k only (R5). Another task's conditions are coloured by their
type: an ordinal condition on the one-hue ramp, light to dark; a categorical one, or
an angle, on the categorical slots in order; more categories than slots aren't drawn
by colour (the tuning curve still is, its x-axis naming each level). IBL's colours are
unchanged. The last tests read downloaded sessions."""

from pathlib import Path

import pytest

from unitwave.data.load import load_data_config
from unitwave.studio.project import Source, load_source
from unitwave.studio.server import Studio
from unitwave.viz.studio_plots import THEMES, ramp_colours

MC_MAZE = (
    Path.home()
    / "data/neurodecoder/dandi/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
)
ALLEN = Path.home() / "data/neurodecoder/dandi/000021/sub-707296975/sub-707296975_ses-721123822.nwb"


def test_the_ramp_runs_light_to_dark_in_one_hue():
    light = ramp_colours(5, "light")
    assert len(set(light)) == 5
    assert light[0] != THEMES["light"]["sequential"][0]  # never the background-pale step
    luminance = [sum(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in light]
    assert luminance == sorted(luminance, reverse=True)  # light to dark on a light page
    dark = ramp_colours(5, "dark")
    lum = [sum(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in dark]
    assert lum == sorted(lum)  # dark to light on a dark page
    assert ramp_colours(1, "light") == [THEMES["light"]["sequential"][-2]]


def _studio(path, layout, task):
    source = Source(kind="nwb", file=str(path), layout=layout, task=task)
    session, qc = load_source(source)
    return Studio(session, qc, load_data_config().data_root / "atlas", source)


Q = {"t0": "-0.5", "t1": "1.0", "bin": "0.02", "theme": "light"}


@pytest.mark.skipif(not MC_MAZE.exists(), reason="MC_Maze_Small isn't downloaded")
def test_mc_maze_colours_follow_condition_types_and_refuse_too_many_categories():
    studio = _studio(MC_MAZE, "nlb_mc_maze", "nlb_mc_maze")
    cat = THEMES["light"]["categorical"]
    assert studio._colours("targets", (1.0, 3.0), "light") == cat[:2]
    # Nine mazes: more than the 8 colours that can be told apart.
    with pytest.raises(ValueError, match="9 levels.*8 colours"):
        studio.unit_data({**Q, "event": "go_cue", "unit": studio.units.index[0], "split": "maze"})
    with pytest.raises(ValueError, match="9 levels"):
        studio.trajectory_data({**Q, "event": "go_cue", "split": "maze"})
    png, _ = studio.tuning_png(
        {**Q, "event": "go_cue", "unit": studio.units.index[0], "split": "maze"}
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # the tuning curve still draws: its axis names each maze
    # Keeping 8 mazes or fewer with the trial filter draws them.
    import json

    tf = json.dumps({"mazes": [2.0, 3.0, 4.0]})
    d = studio.unit_data(
        {**Q, "event": "go_cue", "unit": studio.units.index[0], "split": "maze", "tf": tf}
    )
    assert [g.name for g in d["groups"]] == ["maze 2", "maze 3", "maze 4"]


@pytest.mark.skipif(not MC_MAZE.exists(), reason="MC_Maze_Small isn't downloaded")
def test_mc_maze_trajectories_are_cross_validated_with_pc_names_only():
    studio = _studio(MC_MAZE, "nlb_mc_maze", "nlb_mc_maze")
    d = studio.trajectory_data({**Q, "event": "move_onset", "split": "targets"})
    r = d["result"]
    assert list(r.axis_names[:3]) == ["pc_1", "pc_2", "pc_3"]
    assert list(r.names) == ["one target", "three targets"]
    assert sum(r.n_fit) + sum(r.n_show) == 100 and all(
        abs(f - s) <= 1 for f, s in zip(r.n_fit, r.n_show)
    )
    assert 0 < r.explained_held_out[0] <= 1
    assert "descriptive (no test)" in d["caption"] and "Movement onset" in d["caption"]


@pytest.mark.skipif(not ALLEN.exists(), reason="the Allen session isn't downloaded")
def test_allen_directions_are_categories_and_temporal_frequency_a_ramp():
    studio = _studio(ALLEN, "allen_visual_coding", "allen_drifting_gratings")
    directions = tuple(float(d) for d in range(0, 360, 45))
    assert studio._colours("direction", directions, "light") == THEMES["light"]["categorical"]
    assert studio._colours(
        "temporal_frequency", (1.0, 2.0, 4.0, 8.0, 15.0), "light"
    ) == ramp_colours(5, "light")
    d = studio.trajectory_data(
        {**Q, "event": "stim_on", "t1": "2.0", "split": "temporal_frequency"}
    )
    assert d["result"].axis_names[0] == "pc_1" and len(d["result"].names) == 5
