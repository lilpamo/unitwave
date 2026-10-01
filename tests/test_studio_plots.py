"""Studio figures. Arrays here are test inputs, never shown as data."""

import numpy as np

from unitwave.viz.studio_plots import build_population_figure


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
