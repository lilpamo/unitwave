"""Region-summary figures (S5). Unit tables here are test inputs, never data."""

import numpy as np
import pandas as pd
import pytest

from unitwave.analysis.summary import SummaryConfig, region_summary
from unitwave.viz.summary_plots import build_region_flatmap, build_region_spread

CFG = SummaryConfig(level="Beryl", min_sessions=3, alpha=0.05)


def _summary():
    """VISp enriched in 5 sessions; CA1 and LP not; MOs in 2 sessions only (refused)."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(5):
        for region, rate in (("VISp", 0.8), ("CA1", 0.2), ("LP", 0.2)):
            rows += [(f"e{s}", region, bool(x)) for x in rng.random(25) < rate]
        if s < 2:
            rows += [(f"e{s}", "MOs", bool(x)) for x in rng.random(10) < 0.5]
    units = pd.DataFrame(rows, columns=["eid", "region", "labelled"])
    units.insert(1, "unit_id", [f"u{i}" for i in range(len(units))])
    return region_summary(units, CFG)


def test_a_long_refused_list_sits_below_the_axis_label():
    """18 refused regions (Steinmetz, six sessions) wrap onto several lines; none may
    overlap the x-axis label."""
    regions, sessions = _summary()
    many = pd.DataFrame(
        {"n_sessions": 2, "refused": "in 2 sessions; a verdict needs at least 3"},
        index=[f"R{i:02d}" for i in range(40)],
    )
    regions = pd.concat([regions, many])
    fig = build_region_spread(regions, sessions, "light")
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # places the axis label
    renderer = canvas.get_renderer()
    label = fig.axes[0].xaxis.label.get_window_extent(renderer)
    note = next(t for t in fig.texts if "refused" in t.get_text())
    box = note.get_window_extent(renderer)
    assert box.y1 < label.y0 and box.y0 >= 0  # below the label, inside the figure
    assert note.get_text().count("\n") >= 1 and box.x1 <= fig.bbox.x1


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_spread_shows_every_session_of_every_tested_region(theme):
    regions, sessions = _summary()
    fig = build_region_spread(regions, sessions, theme)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    tested = regions[regions["refused"] == ""]
    assert [label.split(" ")[0] for label in labels] == list(
        ((tested["observed"] - tested["expected"]) / tested["n_units"]).sort_values().index
    )
    dots = sum(len(c.get_offsets()) for c in ax.collections)
    assert dots == (sessions["region"].isin(tested.index)).sum()  # one dot per session
    claim = next(label for label in labels if label.startswith("VISp"))
    assert "5 sessions" in claim and "q =" in claim
    notes = " ".join(t.get_text() for t in fig.texts)
    assert "MOs" in notes and "refused" in notes  # a refused region is named, not drawn


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_flatmap_names_only_the_regions_with_a_claim(theme):
    regions, _ = _summary()
    fig = build_region_flatmap(regions, theme)
    names = {t.get_text() for ax in fig.axes for t in ax.texts}
    # VISp holds more labelled units than its sessions predict; CA1 and LP, fewer.
    claims = regions[regions["claim"]]
    assert claims["direction"].to_dict() == {"CA1": "fewer", "LP": "fewer", "VISp": "more"}
    assert names == set(claims.index) and "MOs" not in names
    legend = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    labels = " ".join(legend + [a.get_ylabel() for a in fig.axes])
    assert "minus expected" in labels and "too few sessions" in labels


def test_a_summary_with_no_tested_region_draws_no_data_colours():
    # Real case (S5, 9 sessions that share no region): with nothing tested, iblatlas
    # filled every region in its own atlas colours, which read as results.
    from matplotlib.colors import to_hex

    from unitwave.viz.studio_plots import THEMES

    regions, _ = _summary()
    regions = regions.assign(refused="in 2 sessions; a verdict needs at least 9")
    regions = regions.assign(claim=False, q=np.nan, p=np.nan)
    fig = build_region_flatmap(regions, "light")
    fills = {to_hex(p.get_facecolor()) for p in fig.axes[0].patches}
    t = THEMES["light"]
    assert fills <= {to_hex(t["grid"]), to_hex(t["muted"])}, fills
    assert to_hex(t["muted"]) in fills  # the refused regions are still shown as refused
