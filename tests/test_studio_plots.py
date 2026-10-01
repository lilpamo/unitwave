"""Studio figures. Arrays here are test inputs, never shown as data."""

import numpy as np
import pytest
from matplotlib.colors import to_hex

from unitwave.analysis.trajectories import Trajectories
from unitwave.viz.studio_plots import (
    THEMES,
    build_population_figure,
    build_trajectory_figure,
)


def test_heatmap_unit_axis_has_whole_unit_ticks_with_few_units():
    # Real case (S1, session 3a3ea015): with 2 units the heatmap's unit axis read
    # 0.00, 0.25, ..., 2.00.
    centers = np.linspace(-0.49, 0.99, 75)
    scaled = np.random.default_rng(0).random((2, centers.size))  # (n_units, n_bins)
    fig, _ = build_population_figure(
        scaled, centers, scaled.mean(axis=0), scaled.std(axis=0), (-0.5, 1.0), "light"
    )
    heatmap = fig.axes[0]
    fig.canvas.draw()
    ticks = [t for t in heatmap.get_yticks() if 0 <= t <= scaled.shape[0]]
    assert ticks and all(float(t).is_integer() for t in ticks), ticks


def _trajectories():
    centers = np.linspace(-0.49, 0.99, 75)
    rng = np.random.default_rng(1)
    return Trajectories(
        names=["a", "b"],
        bin_centers=centers,
        trajectories=rng.normal(size=(2, centers.size, 3)),  # (n_conditions, n_bins, k)
        components=rng.normal(size=(10, 3)),
        explained_fit=np.array([0.65, 0.17, 0.04]),
        explained_held_out=np.array([0.59, 0.17, 0.03]),
        n_fit=[108, 38],
        n_show=[107, 37],
        excluded={},
        axis_names=["pc_1", "pc_2", "pc_3"],
    )


@pytest.mark.parametrize("dims", [2, 3])
def test_trajectory_axes_carry_each_components_share_of_the_shown_variance(dims):
    # S4: the share of the shown (held-out) trials' variance is on each component's
    # axis; names stay pc_k (R5).
    fig = build_trajectory_figure(_trajectories(), ["#000000", "#555555"], dims, "light")
    main = fig.axes[0]
    labels = [main.get_xlabel(), main.get_ylabel()]
    if dims == 3:
        labels.append(main.get_zlabel())
    shares = ["59%", "17%", "3%"][: len(labels)]
    for label, name, share in zip(labels, ["pc_1", "pc_2", "pc_3"], shares):
        assert label == f"{name} ({share})"
    # What the percentage is, said once (in 3-D long axis labels collide).
    title = fig.legends[0].get_title().get_text()
    assert "(%)" in title and "shown trials' variance" in title.replace("\n", " ")


def test_trajectory_legend_title_is_readable_in_the_dark_theme():
    # The title was drawn black on the dark page; only the entries were themed.
    fig = build_trajectory_figure(_trajectories(), ["#000000", "#555555"], 2, "dark")
    title = fig.legends[0].get_title()
    assert to_hex(title.get_color()) == to_hex(THEMES["dark"]["ink2"])
    rows = [a.get_ylabel() for a in fig.axes[1:] if a.get_ylabel().startswith("pc_")]
    assert rows == ["pc_1\n59%", "pc_2\n17%", "pc_3\n3%"]
