"""Region-summary figures (S5): the Swanson flatmap and the per-session spread.

Both draw only what analysis.summary computed. A region's colour (flatmap) or
position (spread) is its labelled fraction minus the fraction expected from its own
sessions. The spread shows every session separately, never only the pooled number
(§5). Regions refused for too few sessions are drawn apart (flatmap) or listed
(spread), never coloured as results.
"""

import warnings

import numpy as np
import pandas as pd
from iblatlas.plots import plot_swanson_vector
from iblatlas.regions import BrainRegions
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from unitwave.viz.studio_plots import _frame, _png, _style, _theme

_REGIONS = BrainRegions()


def _excess(regions: pd.DataFrame) -> pd.Series:
    """Labelled fraction minus the fraction expected from the region's sessions."""
    return (regions["observed"] - regions["expected"]) / regions["n_units"]


def build_region_flatmap(regions: pd.DataFrame, theme: str, title=None) -> Figure:
    """Swanson flatmap of the tested regions' excess labelled fraction (diverging,
    centred on 0). Regions refused for too few sessions are filled in the muted grey,
    regions without units in the grid colour; regions with a claim (q < alpha) are
    named."""
    t = _theme(theme)
    fig = Figure(figsize=(7.4, 4.2))
    add, squeeze = _frame(fig, title, t["ink2"])
    ax = add((0.01, 0.13, 0.86, 0.85))
    cax = add((0.89, 0.25, 0.018, 0.6))
    tested = regions[regions["refused"] == ""]
    refused = regions.index[regions["refused"] != ""].tolist()
    claims = regions.index[regions["claim"]].tolist()
    excess = _excess(tested)
    span = float(np.nanmax(np.abs(excess))) if len(tested) else 0.0
    span = span or 1.0
    cmap = LinearSegmentedColormap.from_list("summary", t["diverging"])
    with warnings.catch_warnings():  # iblatlas takes a median over regions without values
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        _draw_flatmap(ax, tested, excess, refused, claims, cmap, span, t)
    ax.set_axis_off()
    for text in ax.texts:  # iblatlas writes its labels in black
        text.set_color(t["ink"])
    _colorbar_and_legend(fig, cax, cmap, span, squeeze, t)
    return fig


def _draw_flatmap(ax, tested, excess, refused, claims, cmap, span, t) -> None:
    plot_swanson_vector(
        # With no region tested, a placeholder value on "void" (not on the map): given
        # no regions at all, iblatlas fills every region in its own atlas colours.
        acronyms=np.array(tested.index) if len(tested) else np.array(["void"]),
        values=excess.to_numpy() if len(tested) else np.zeros(1),
        ax=ax,
        br=_REGIONS,
        cmap=cmap,
        vmin=-span,
        vmax=span,
        empty_color=t["grid"],
        mask=refused or None,
        mask_color=t["muted"],
        annotate=bool(claims),
        annotate_list=np.array(claims) if claims else None,
        fontsize=6,
        edgecolor=t["axis"],
        linewidth=0.15,
    )


def _colorbar_and_legend(fig, cax, cmap, span, squeeze, t) -> None:
    bar = fig.colorbar(ScalarMappable(Normalize(-span, span), cmap), cax=cax)
    bar.set_label("labelled fraction minus expected", color=t["ink2"], size=8)
    bar.outline.set_visible(False)
    cax.tick_params(colors=t["ink2"], labelsize=7, length=2)
    fig.legend(
        handles=[
            Patch(color=t["grid"], label="no units"),
            Patch(color=t["muted"], label="too few sessions"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.01 * squeeze),
        ncol=2,
        frameon=False,
        fontsize=7,
        labelcolor=t["ink2"],
        handlelength=1.2,
        title="named: q < α (Benjamini–Hochberg across regions)",
        title_fontsize=7,
        alignment="left",
    ).get_title().set_color(t["ink2"])


def build_region_spread(
    regions: pd.DataFrame, sessions: pd.DataFrame, theme: str, title=None
) -> Figure:
    """One row per tested region, sorted by its excess: one dot per session (its
    fraction minus the fraction among that session's other units) and a tick at the
    median. Labels give sessions, units and q; refused regions are listed below."""
    t = _theme(theme)
    tested = regions[regions["refused"] == ""]
    order = _excess(tested).sort_values().index.tolist()
    rows = max(len(order), 1)
    fig = Figure(figsize=(6.4, 1.2 + 0.26 * rows))
    add, _ = _frame(fig, title, t["ink2"])
    bottom = 0.75 / (1.2 + 0.26 * rows)
    ax = add((0.42, bottom, 0.54, 1 - bottom - 0.04))
    labels = []
    for y, region in enumerate(order):
        rows_r = sessions[sessions["region"] == region]
        diff = (rows_r["fraction"] - rows_r["fraction_rest"]).to_numpy(np.float64)
        colour = t["series"] if regions.at[region, "claim"] else t["muted"]
        ax.scatter(diff, np.full(diff.size, y), s=14, color=colour, alpha=0.8, lw=0)
        ax.plot([np.nanmedian(diff)] * 2, [y - 0.36, y + 0.36], color=t["ink"], lw=2.2)
        r = regions.loc[region]
        mark = " ✓" if r["claim"] else ""
        labels.append(
            f"{region} · {r['n_sessions']} sessions · {r['n_units']} units · "
            f"q = {r['q']:.2g}{mark}"
        )
    ax.axvline(0, color=t["axis"], lw=1)
    ax.set_yticks(range(len(order)), labels)
    ax.set_ylim(-0.6, rows - 0.4)
    ax.set_xlabel(
        "fraction labelled: region minus session's other units\n(dot: a session; bar: median)"
    )
    ax.grid(axis="x", color=t["grid"], lw=0.6)
    ax.set_axisbelow(True)
    _style(ax, t)
    ax.tick_params(axis="y", labelsize=7, length=0)
    refused = regions[regions["refused"] != ""]
    if len(refused):
        listed = ", ".join(f"{name} ({row['n_sessions']})" for name, row in refused.iterrows())
        fig.text(
            0.02,
            0.01,
            f"refused, too few sessions: {listed}",
            size=7,
            color=t["ink2"],
            wrap=True,
            va="bottom",
        )
    return fig


def region_flatmap(regions: pd.DataFrame, theme: str) -> bytes:
    return _png(build_region_flatmap(regions, theme))


def region_spread(regions: pd.DataFrame, sessions: pd.DataFrame, theme: str) -> bytes:
    return _png(build_region_spread(regions, sessions, theme))
