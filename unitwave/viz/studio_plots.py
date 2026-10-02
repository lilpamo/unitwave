"""Studio figures, drawn from analysis/ outputs only. Returns PNG bytes.

Colours are the dataviz reference palette's roles, stepped separately for light and
dark: ink, gridlines, one series blue, a single-hue blue ramp for magnitude and a
blue-grey-red ramp for signed values. The event line is muted ink, not a status
colour. Figures have a transparent background and a fixed layout, so the page can
map clicks on heatmap rows back to units.

Uses matplotlib's object API (no pyplot state), so figures are independent.
"""

import io
import os
from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib import rc_context
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from unitwave.analysis.psth import PSTH

# The reference palette's categorical slots, in its fixed, CVD-validated order.
_CATEGORICAL_LIGHT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
_CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]
_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
THEMES = {
    "light": {
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": "#2a78d6",
        "sequential": _BLUE,
        "diverging": ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#c22f2f"],
        "categorical": _CATEGORICAL_LIGHT,
    },
    "dark": {
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": "#3987e5",
        # Near zero recedes into the dark surface; high values are bright.
        "sequential": _BLUE[::-1],
        "diverging": ["#6da7ec", "#256abf", "#383835", "#c22f2f", "#ec8a89"],
        "categorical": _CATEGORICAL_DARK,
    },
}
_DPI = 110


def _theme(name: str) -> dict:
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}")
    return THEMES[name]


def probe_colours(probes: list[str], theme: str) -> dict[str, str]:
    """probe -> colour by its place in the session's probe list, so filtering never
    repaints a probe. Past the eighth probe, the rest share the muted ink ("other")."""
    t = _theme(theme)
    slots = t["categorical"]
    return {p: slots[i] if i < len(slots) else t["muted"] for i, p in enumerate(probes)}


def lab_colours(labs: list[str], theme: str) -> dict[str, str]:
    """lab -> colour, in the given order (largest lab first): the eight categorical slots,
    then the muted ink shared by every other lab. Never a generated hue."""
    t = _theme(theme)
    slots = t["categorical"]
    return {lab: slots[i] if i < len(slots) else t["muted"] for i, lab in enumerate(labs)}


def _style(ax, t: dict) -> None:
    ax.set_facecolor("none")
    ax.tick_params(colors=t["ink2"], labelsize=8, length=3)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])


# Written into every figure file, so a figure says what made it.
_CREATOR = f"UnitWave Studio, with Matplotlib {matplotlib.__version__}"


def _png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, transparent=True, metadata={"Software": _CREATOR})
    return buf.getvalue()


def save_vector(fig: Figure, path: str | os.PathLike) -> None:
    """SVG or PDF by suffix, white background, text kept as editable text."""
    with rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        fig.savefig(path, facecolor="white", metadata={"Creator": _CREATOR})


def _frame(fig: Figure, title: str | None, ink2: str):
    """An add_axes that leaves room for a title line at the top when there is one."""
    squeeze = 1.0
    if title:
        fig.text(0.02, 0.985, title, va="top", ha="left", size=8, color=ink2, wrap=True)
        squeeze = 0.91

    def add(box, **kwargs):
        left, bottom, width, height = box
        return fig.add_axes((left, bottom * squeeze, width, height * squeeze), **kwargs)

    return add, squeeze


@dataclass(frozen=True)
class TraceGroup:
    """One condition's raster rows and PSTH; colour None draws the unsplit style."""

    name: str
    colour: str | None
    psth: PSTH
    trial: np.ndarray  # (n_spikes,) trial index within this group
    rel: np.ndarray  # (n_spikes,) seconds from the event


def condition_colours(name: str, levels: tuple[float, ...], theme: str) -> list[str]:
    """One colour per level. Sided conditions use the diverging pair: left blue, right
    red, graded by rank for contrast; 0% and the unbiased block get the muted ink. Choice
    takes the side it reports on correct trials. Outcome uses the categorical slots."""
    t = _theme(theme)
    ramp = LinearSegmentedColormap.from_list("sided", t["diverging"])

    def sided(position: float) -> str:  # 0 = strongest left, 1 = strongest right
        return to_hex(ramp(position))

    if name == "outcome":
        return [t["categorical"][1] if v < 0 else t["categorical"][0] for v in levels]
    if name == "side":
        return [sided(0.0) if v < 0 else sided(1.0) for v in levels]
    if name == "choice":  # -1 reports right, +1 reports left
        return [sided(1.0) if v < 0 else sided(0.0) for v in levels]
    if name == "block":  # p(left) 0.8 is a left block
        return [t["muted"] if v == 0.5 else sided(0.0 if v > 0.5 else 1.0) for v in levels]
    if name == "contrast":
        left = sorted(-v for v in levels if v < 0)
        right = sorted(v for v in levels if v > 0)
        out = []
        for v in levels:
            if v == 0:
                out.append(t["muted"])
                continue
            mags = left if v < 0 else right
            reach = 0.2 + 0.3 * (mags.index(abs(v)) + 1) / len(mags)  # never near the midpoint
            out.append(sided(0.5 - reach if v < 0 else 0.5 + reach))
        return out
    return [t["categorical"][i % len(t["categorical"])] for i in range(len(levels))]


def build_unit_figure(
    groups: list[TraceGroup], window, baseline: bool, theme: str, title=None
) -> Figure:
    """Raster (top, trials grouped by condition) and PSTH mean ± SEM (bottom), with a
    legend naming each condition and its trial count when split."""
    t = _theme(theme)
    fig = Figure(figsize=(5.6, 4.6))
    add, squeeze = _frame(fig, title, t["ink2"])
    split = len(groups) > 1
    # A split leaves a band between the panels for the legend, off the traces.
    ax_r = add((0.13, 0.56, 0.84, 0.41) if split else (0.13, 0.47, 0.84, 0.50))
    ax_p = add((0.13, 0.11, 0.84, 0.29) if split else (0.13, 0.11, 0.84, 0.31), sharex=ax_r)
    offset = 0
    for g in groups:
        ink = g.colour or t["ink"]
        ax_r.plot(g.rel, g.trial + offset, "|", color=ink, markersize=2.2, markeredgewidth=0.6)
        offset += g.psth.n_trials
        if len(groups) > 1:
            ax_r.axhline(offset - 0.5, color=t["grid"], lw=0.6)
        line = g.colour or t["series"]
        p = g.psth
        ax_p.fill_between(
            p.bin_centers, p.mean - p.sem, p.mean + p.sem, color=line, alpha=0.18, lw=0
        )
        ax_p.plot(p.bin_centers, p.mean, color=line, lw=2, label=f"{g.name} (n = {p.n_trials})")
    ax_r.set_ylim(offset - 0.5, -0.5)
    ax_r.set_ylabel("trial, by condition" if len(groups) > 1 else "trial")
    ax_r.tick_params(labelbottom=False)
    ax_p.set_ylabel("Δ rate (Hz)" if baseline else "rate (Hz)")
    ax_p.set_xlabel("time from event (s)")
    ax_p.set_xlim(*window)
    ax_p.grid(axis="y", color=t["grid"], lw=0.6)
    ax_p.set_axisbelow(True)
    if split:
        fig.legend(
            *ax_p.get_legend_handles_labels(),
            loc="upper left",
            bbox_to_anchor=(0.1, 0.545 * squeeze),
            ncol=min(5, len(groups)),
            fontsize=6.5,
            frameon=False,
            labelcolor=t["ink2"],
            handlelength=1.2,
            columnspacing=0.7,
        )
    for ax in (ax_r, ax_p):
        _style(ax, t)
        ax.axvline(0, color=t["muted"], lw=1, ls=(0, (3, 3)))
    return fig


def _box(ax) -> tuple[float, float, float, float]:
    """An axes' (left, top, right, bottom) as fractions of the image, top-down."""
    pos = ax.get_position()
    return pos.x0, 1 - pos.y1, pos.x1, 1 - pos.y0


def unit_figure(
    groups: list[TraceGroup], window, baseline: bool, theme: str
) -> tuple[bytes, tuple[float, float, float, float]]:
    """build_unit_figure as a PNG for the page, with the raster's box: raster row i
    (top-down, groups in order) spans top + (bottom - top) * [i, i + 1] / n_rows."""
    fig = build_unit_figure(groups, window, baseline, theme)
    return _png(fig), _box(fig.axes[0])


def build_tuning_figure(
    names, means, sems, ns, colours, ordinal: bool, theme: str, title=None
) -> Figure:
    """Mean ± SEM response rate per condition level, n under each level's label."""
    t = _theme(theme)
    fig = Figure(figsize=(5.6, 2.5))
    add, _ = _frame(fig, title, t["ink2"])
    ax = add((0.13, 0.26, 0.84, 0.66))
    x = np.arange(len(names))
    if ordinal:
        ax.plot(x, means, color=t["axis"], lw=1, zorder=1)
    for xi, m, e, c in zip(x, means, sems, colours):
        ax.errorbar(
            xi, m, yerr=0 if np.isnan(e) else e, fmt="o", color=c, ms=6, lw=1.5, capsize=0, zorder=2
        )
    ax.set_xticks(x, [f"{n}\nn = {k}" for n, k in zip(names, ns)])
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylabel("response rate (Hz)")
    ax.grid(axis="y", color=t["grid"], lw=0.6)
    ax.set_axisbelow(True)
    _style(ax, t)
    ax.tick_params(axis="x", labelsize=7)
    return fig


def build_wheel_figure(p: PSTH, window, theme: str, title=None) -> Figure:
    """Wheel speed mean ± SEM around the event, on the unit PSTH's time axis (same
    left edge and width, so the two line up on the page)."""
    t = _theme(theme)
    fig = Figure(figsize=(5.6, 1.6))
    add, _ = _frame(fig, title, t["ink2"])
    ax = add((0.13, 0.3, 0.84, 0.62))
    ax.fill_between(
        p.bin_centers, p.mean - p.sem, p.mean + p.sem, color=t["ink2"], alpha=0.18, lw=0
    )
    ax.plot(p.bin_centers, p.mean, color=t["ink2"], lw=1.6)
    ax.set_xlim(*window)
    ax.set_ylabel("wheel (rad/s)")
    ax.set_xlabel("time from event (s)")
    ax.grid(axis="y", color=t["grid"], lw=0.6)
    ax.set_axisbelow(True)
    _style(ax, t)
    ax.axvline(0, color=t["muted"], lw=1, ls=(0, (3, 3)))
    return fig


def wheel_figure(p: PSTH, window, theme: str) -> bytes:
    """build_wheel_figure as a PNG for the page."""
    return _png(build_wheel_figure(p, window, theme))


def tuning_figure(names, means, sems, ns, colours, ordinal: bool, theme: str) -> bytes:
    return _png(build_tuning_figure(names, means, sems, ns, colours, ordinal, theme))


def population_figure(scaled, centers, mean, sem, window, theme: str, **stripe):
    """build_population_figure as a PNG for the page, with the heatmap's box."""
    fig, box = build_population_figure(scaled, centers, mean, sem, window, theme, **stripe)
    return _png(fig), box


def build_population_figure(
    scaled: np.ndarray,
    centers: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    window,
    theme: str,
    title=None,
    row_groups: list[str] | None = None,
    group_colours: dict[str, str] | None = None,
) -> tuple[Figure, tuple[float, float, float, float]]:
    """Heatmap of row-scaled PSTHs, (n_units, n_bins) already sorted, and the selection mean.

    row_groups (n_units,) and group_colours draw a stripe beside the rows (each row's
    probe) with a legend naming every colour, so identity is never colour alone.

    Returns the figure and the heatmap's box (left, top, right, bottom) as fractions of
    the image, top-down, so row i spans top + (bottom - top) * [i, i + 1] / n_units.
    """
    t = _theme(theme)
    signed = bool((scaled < 0).any())
    cmap = LinearSegmentedColormap.from_list(
        "studio", t["diverging"] if signed else t["sequential"]
    )
    fig = Figure(figsize=(5.6, 5.6))
    add, squeeze = _frame(fig, title, t["ink2"])
    box = (0.13, 0.40, 0.72, 0.56)  # left, bottom, width, height
    ax_h = add(box)
    cax = add((0.885, 0.40, 0.022, 0.56))
    ax_m = add((0.13, 0.08, 0.72, 0.24), sharex=ax_h)
    im = ax_h.imshow(
        scaled,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-1 if signed else 0,
        vmax=1,
        extent=(window[0], window[1], scaled.shape[0], 0),
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("rate / unit's max |rate|", color=t["ink2"], size=8)
    cb.outline.set_visible(False)
    cax.tick_params(colors=t["ink2"], labelsize=7, length=2)
    ax_h.set_ylabel("unit (sorted by peak time)")
    ax_h.yaxis.set_major_locator(MaxNLocator(integer=True))  # whole units, even for 2
    ax_h.tick_params(labelbottom=False)
    if row_groups is not None:
        assert len(row_groups) == scaled.shape[0]
        ax_s = add((0.855, 0.40, 0.014, 0.56), sharey=ax_h)
        rgb = np.array([to_rgb(group_colours[g]) for g in row_groups])[:, None, :]
        ax_s.imshow(rgb, aspect="auto", interpolation="nearest", extent=(0, 1, len(rgb), 0))
        ax_s.axis("off")
        shown = [g for g in group_colours if g in set(row_groups)]
        fig.legend(
            handles=[Patch(color=group_colours[g], label=g) for g in shown],
            loc="lower left",
            bbox_to_anchor=(0.13, 0.335 * squeeze),
            ncol=len(shown),
            frameon=False,
            fontsize=7,
            labelcolor=t["ink2"],
            handlelength=1,
            handleheight=0.8,
            columnspacing=1.2,
        )
    ax_m.fill_between(centers, mean - sem, mean + sem, color=t["series"], alpha=0.2, lw=0)
    ax_m.plot(centers, mean, color=t["series"], lw=2)
    ax_m.set_ylabel("mean ± SEM (Hz)")
    ax_m.set_xlabel("time from event (s)")
    ax_m.set_xlim(*window)
    ax_m.grid(axis="y", color=t["grid"], lw=0.6)
    ax_m.set_axisbelow(True)
    for ax in (ax_h, ax_m):
        _style(ax, t)
        ax.axvline(0, color=t["muted"], lw=1, ls=(0, (3, 3)))
    left, bottom, width, height = box
    bottom, height = bottom * squeeze, height * squeeze
    return fig, (left, 1 - bottom - height, left + width, 1 - bottom)


# Event lines take the categorical slots in task order; feedback keeps one hue and
# differs by style (reward solid, error dotted), so outcome is never colour alone.
_EVENT_STYLE = {"error": (0, (1, 1.5))}
# In IBL the go cue comes within ~1 ms of stimulus onset and feedback within ~1 ms of
# the response, so those pairs overlap at any readable scale. The go cue and the
# response are drawn wider underneath, as a halo, so both lines of a pair stay
# visible at their true times.
_EVENT_WIDTH = {"goCue_times": 3.2, "response_times": 3.2}
_TRACE_LABELS = {
    "motion_energy_left": "motion energy\n(left cam)",
    "motion_energy_right": "motion energy\n(right cam)",
    "motion_energy_body": "motion energy\n(body cam)",
    "pupil_left": "pupil, raw\n(left cam)",
    "pupil_right": "pupil, raw\n(right cam)",
}


def _runs(values: list) -> list[tuple[int, int, object]]:
    """(start, stop, value) for each run of equal consecutive values."""
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((start, i, values[start]))
            start = i
    return out


def build_trial_figure(
    view,
    row_regions: list[str | None],
    region_colours: dict[str, str],
    selected: str | None,
    theme: str,
    title=None,
) -> tuple[Figure, tuple[float, float, float, float]]:
    """Single-trial population raster (analysis.trial_view.TrialView), the task events as
    lines with one legend, and behaviour on the same time axis below.

    Rows run top to bottom in view.rows order, with probe separators and region colour
    bands (row_regions, (n_rows,)) on the left. Returns the figure and the raster's box
    (left, top, right, bottom) as fractions of the image, top-down, like the heatmap.
    """
    t = _theme(theme)
    n = len(view.rows)
    panels = []
    if view.wheel is not None:
        panels += [
            ("wheel position\n(rad)", view.wheel["position_t_s"], view.wheel["position_rad"])
        ]
        panels += [("wheel speed\n(rad/s)", view.wheel["speed_t_s"], view.wheel["speed"])]
    panels += [(_TRACE_LABELS.get(k, k.replace("_", " ")), *v) for k, v in view.traces.items()]
    notes = []
    if view.not_recorded:
        by_trial: dict[int, list[str]] = {}
        for e in view.not_recorded:
            by_trial.setdefault(e["trial"], []).append(e["label"])
        notes += [f"not recorded on trial {k}: {', '.join(v)}" for k, v in by_trial.items()]
    if view.absent_events:
        notes.append(f"not in this session's trials table: {', '.join(view.absent_events)}")
    if view.wheel_missing:
        notes.append(f"no wheel: {view.wheel_missing}")
    notes += [f"no {k.replace('_', ' ')}: {why}" for k, why in view.traces_missing.items()]

    # Layout in inches, top to bottom, then converted to figure fractions.
    width, left_in, right_in = 11.0, 1.25, 0.7
    raster_h = float(np.clip(n * 0.012, 1.6, 4.5))
    heights = {
        "title": 0.3 if title else 0.08,
        "legend": 0.32,
        "notes": 0.2 * len(notes),
        "strip": 0.22,
        "raster": raster_h,
        "panels": len(panels) * 0.72,
        "bottom": 0.5,
    }
    height = sum(heights.values())
    fig = Figure(figsize=(width, height))
    if title:
        fig.text(0.01, 1 - 0.08 / height, title, va="top", ha="left", size=8, color=t["ink2"])
    x0, x1 = left_in / width, 1 - right_in / width

    def box(top_in, h_in):
        return (x0, 1 - (top_in + h_in) / height, x1 - x0, h_in / height)

    top = heights["title"] + heights["legend"] + heights["notes"]
    for i, note in enumerate(notes):
        y = 1 - (heights["title"] + heights["legend"] + 0.16 + 0.2 * i) / height
        fig.text(x0, y, note, size=7.5, color=t["ink2"], va="center")
    ax_t = fig.add_axes(box(top, heights["strip"]))
    ax_r = fig.add_axes(box(top + heights["strip"], raster_h), sharex=ax_t)
    raster_box = box(top + heights["strip"], raster_h)
    ax_b = fig.add_axes((x0 - 0.16 / width, raster_box[1], 0.1 / width, raster_box[3]), sharey=ax_r)
    axes_p = []
    y = top + heights["strip"] + raster_h + 0.12
    for _ in panels:
        axes_p.append(fig.add_axes(box(y, 0.6), sharex=ax_r))
        y += 0.72
    lo, hi = view.window.start_rel_s, view.window.stop_rel_s

    # Trial strip: each shown trial as a bar with its number; the current one in ink.
    for b in view.boundaries:
        current = b["trial"] == view.window.trial
        ax_t.axvspan(b["start_rel_s"], b["stop_rel_s"], color=t["grid"], lw=0)
        label = f"trial {b['trial']}" + ("" if b["passes_filter"] else " · fails trial filters")
        ax_t.text(
            max(b["start_rel_s"], lo) + 0.01 * (hi - lo),
            0.5,
            label,
            va="center",
            size=7.5,
            color=t["ink"] if current else t["ink2"],
            weight="bold" if current else "normal",
        )
    ax_t.set_ylim(0, 1)
    ax_t.axis("off")

    # Raster: one row per unit, row 0 on top.
    segments = [
        ((s, i + 0.12), (s, i + 0.88)) for i, spikes in enumerate(view.spikes) for s in spikes
    ]
    ax_r.add_collection(LineCollection(segments, colors=t["ink"], linewidths=0.6))
    ax_r.set_ylim(n, 0)
    ax_r.set_xlim(lo, hi)
    if selected in view.rows:
        i = view.rows.index(selected)
        ax_r.axhspan(i, i + 1, color=t["muted"], alpha=0.3, lw=0)
        ax_r.plot([hi], [i + 0.5], marker="<", color=t["ink"], ms=6, clip_on=False, zorder=5)
    for start, stop, probe in _runs(view.probes):
        if start:
            ax_r.axhline(start, color=t["axis"], lw=1.2)
        ax_r.text(
            1.02,
            1 - (start + stop) / 2 / n,
            probe,
            transform=ax_r.transAxes,
            size=7.5,
            color=t["ink2"],
            va="center",
            ha="left",
        )
    ax_r.tick_params(labelbottom=not panels, left=False, labelleft=False)
    _style(ax_r, t)
    ax_r.spines["left"].set_visible(False)

    # Region bands, labelled where a run is tall enough to read.
    none = (0.0, 0.0, 0.0, 0.0)
    rgba = np.array([to_rgb(region_colours[r]) + (1.0,) if r else none for r in row_regions])
    ax_b.imshow(rgba[:, None, :], aspect="auto", interpolation="nearest", extent=(0, 1, n, 0))
    ticks = [(a + b) / 2 for a, b, r in _runs(row_regions) if r and (b - a) >= max(1, 0.025 * n)]
    names = [r for a, b, r in _runs(row_regions) if r and (b - a) >= max(1, 0.025 * n)]
    ax_b.set_yticks(ticks, names)
    ax_b.set_xticks([])
    ax_b.tick_params(axis="y", colors=t["ink2"], labelsize=7, length=0)
    for side in ax_b.spines.values():
        side.set_visible(False)
    ax_b.set_facecolor("none")

    # Behaviour panels.
    for ax, (label, x, yv) in zip(axes_p, panels):
        ax.plot(x, yv, color=t["ink2"], lw=1.1)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", size=7.5)
        ax.grid(axis="y", color=t["grid"], lw=0.6)
        ax.set_axisbelow(True)
        _style(ax, t)
        ax.tick_params(labelbottom=ax is axes_p[-1], labelsize=7)
    last = axes_p[-1] if axes_p else ax_r
    last.set_xlabel(f"time from {view.align_label} (s)")

    # Trial boundaries and events on every axis that shares the time axis.
    def style(e):
        return {
            "color": t["categorical"][e["slot"] % len(t["categorical"])],
            "lw": _EVENT_WIDTH.get(e["event"], 1.3),
            "ls": _EVENT_STYLE.get(e["kind"], "-"),
        }

    for ax in [ax_r, *axes_p]:
        for b in view.boundaries:
            for x in (b["start_rel_s"], b["stop_rel_s"]):
                ax.axvline(x, color=t["muted"], lw=0.8)
        for e in view.events:
            ax.axvline(e["time_s"], zorder=3 if e["event"] in _EVENT_WIDTH else 4, **style(e))
    # One legend entry per event label drawn, in task order (reward before error).
    legend = {}
    for e in sorted(view.events, key=lambda e: (e["slot"], e["kind"] or "")):
        legend.setdefault(e["label"], Line2D([], [], **style(e)))
    if legend:
        fig.legend(
            list(legend.values()),
            list(legend),
            loc="center left",
            bbox_to_anchor=(x0 - 0.01, 1 - (heights["title"] + 0.16) / height),
            ncol=len(legend),
            frameon=False,
            fontsize=7.5,
            labelcolor=t["ink2"],
            handlelength=1.8,
            columnspacing=1.4,
        )
    left, bottom, w, h = raster_box
    return fig, (left, 1 - bottom - h, left + w, 1 - bottom)


def trial_figure(view, row_regions, region_colours, selected, theme) -> tuple[bytes, tuple]:
    """build_trial_figure as a PNG for the page, with the raster's box."""
    fig, raster_box = build_trial_figure(view, row_regions, region_colours, selected, theme)
    return _png(fig), raster_box


def build_quality_figure(q, theme: str, title=None) -> Figure:
    """The unit quality panel (analysis.unit_quality.UnitQuality): ISI histogram and
    autocorrelogram, the sliding refractory-period test at every period tested, the
    rate across the session, and the mean waveform when there is one. Shows QC already
    computed; labels nothing."""
    t = _theme(theme)
    fig = Figure(figsize=(5.6, 7.8))
    add, _ = _frame(fig, title, t["ink2"])
    ax_isi = add((0.13, 0.80, 0.36, 0.155))
    ax_acg = add((0.61, 0.80, 0.36, 0.155))
    ax_rp = add((0.13, 0.565, 0.84, 0.15))
    ax_rate = add((0.13, 0.345, 0.84, 0.135))
    ax_wf = add((0.13, 0.06, 0.84, 0.19))
    rd = q.refractory

    edges, counts, beyond = q.isi
    ax_isi.stairs(counts, edges * 1000, color=t["series"], fill=True, alpha=0.85)
    if rd.first_pass_rp_s is not None:
        ax_isi.axvline(rd.first_pass_rp_s * 1000, color=t["ink"], lw=1, ls=(0, (3, 2)))
    ax_isi.set_xlim(0, edges[-1] * 1000)
    ax_isi.set_xlabel("interspike interval (ms)")
    ax_isi.set_ylabel("count")
    where = (
        f"dashed: {rd.first_pass_rp_s * 1000:.2f} ms, where the RP test first passes\n"
        if rd.first_pass_rp_s is not None
        else ""
    )
    ax_isi.set_title(
        f"{where}{beyond:,} intervals ≥ {edges[-1] * 1000:g} ms not shown",
        size=6.5,
        color=t["ink2"],
    )

    lags, acg = q.acg
    width = (lags[1] - lags[0]) * 1000
    ax_acg.bar(lags * 1000, acg, width=width, color=t["series"], lw=0)
    ax_acg.set_xlim(lags[0] * 1000 - width / 2, lags[-1] * 1000 + width / 2)
    ax_acg.set_xlabel("lag (ms)")
    ax_acg.set_ylabel("spike pairs")

    if rd.rp_s.size:
        ms = rd.rp_s * 1000
        ax_rp.plot(
            ms,
            rd.max_acceptable,
            color=t["series"],
            lw=1.6,
            drawstyle="steps-mid",
            label=f"most violations allowed ({rd.contamination:.0%} contamination, "
            f"{1 - rd.alpha:.0%} confidence)",
        )
        ok = rd.violations <= rd.max_acceptable
        ax_rp.plot(ms[ok], rd.violations[ok], "o", color=t["ink"], ms=4, label="violations: pass")
        ax_rp.plot(
            ms[~ok],
            rd.violations[~ok],
            "o",
            mfc="none",
            mec=t["ink"],
            ms=4,
            label="violations: fail",
        )
        ax_rp.set_yscale("symlog", linthresh=1)  # counts span 0 to thousands
        ax_rp.set_xlabel("refractory period tested (ms)")
        ax_rp.set_ylabel("spike pairs (log)")
        ax_rp.legend(fontsize=6.5, frameon=False, labelcolor=t["ink2"], loc="best")
        verdict = (
            f"passes from {rd.first_pass_rp_s * 1000:.2f} ms"
            if rd.passed
            else "fails at every period tested"
        )
        ax_rp.set_title(f"Sliding refractory-period test: {verdict}", size=7.5, color=t["ink2"])
    else:
        ax_rp.text(
            0.5,
            0.5,
            f"No refractory test: {rd.why}",
            ha="center",
            va="center",
            transform=ax_rp.transAxes,
            size=8,
            color=t["ink2"],
        )
        ax_rp.set_axis_off()

    centres, rate = q.rate
    ax_rate.plot(centres / 60, rate, color=t["series"], lw=1.4, drawstyle="steps-mid")
    ax_rate.set_xlabel("time in session (min)")
    ax_rate.set_ylabel("rate (Hz)")
    ax_rate.set_ylim(bottom=0)
    ax_rate.set_title(
        f"presence ratio {q.presence[0]:.3f} (IBL's 10 s bins)", size=7.5, color=t["ink2"]
    )

    w = q.waveform
    if w is not None:
        ptp = np.ptp(w.samples_uv, axis=0)
        ms = np.arange(w.samples_uv.shape[0]) / w.sample_rate * 1000
        for c in np.argsort(ptp)[::-1][1:4]:
            ax_wf.plot(ms, w.samples_uv[:, c], color=t["muted"], lw=0.9)
        ax_wf.plot(ms, w.samples_uv[:, w.peak], color=t["ink"], lw=1.6)
        ax_wf.set_xlabel("time (ms)")
        ax_wf.set_ylabel(f"mean waveform ({w.unit})")
        ax_wf.set_title(
            f"peak channel {int(w.channels[w.peak])} (black) and the next 3 by amplitude (grey)"
            f" · {w.source}",
            size=7,
            color=t["ink2"],
        )
    else:
        ax_wf.text(
            0.5,
            0.5,
            f"No waveform: {q.waveform_missing}",
            ha="center",
            va="center",
            transform=ax_wf.transAxes,
            size=8,
            color=t["ink2"],
            wrap=True,
        )
        ax_wf.set_axis_off()

    for ax in (ax_isi, ax_acg, ax_rp, ax_rate, ax_wf):
        if ax.axison:
            _style(ax, t)
            ax.grid(axis="y", color=t["grid"], lw=0.6)
            ax.set_axisbelow(True)
    return fig


def quality_figure(q, theme: str) -> bytes:
    """build_quality_figure as a PNG for the page."""
    return _png(build_quality_figure(q, theme))


def build_ccg_figure(lags, observed, expected, synaptic_window, theme: str, title=None) -> Figure:
    """Cross-correlogram (counts by lag, bars) with its expectation under interval
    jitter (line) and the synaptic window shaded; below, the jitter-corrected
    correlogram (observed - expected). Lags in seconds, drawn in ms."""
    t = _theme(theme)
    fig = Figure(figsize=(5.6, 4.2))
    add, _ = _frame(fig, title, t["ink2"])
    ax_raw = add((0.13, 0.47, 0.84, 0.45))
    ax_cor = add((0.13, 0.11, 0.84, 0.26), sharex=ax_raw)
    ms = np.asarray(lags) * 1000
    width = (ms[1] - ms[0]) if ms.size > 1 else 1.0
    for ax in (ax_raw, ax_cor):
        ax.axvspan(synaptic_window[0] * 1000, synaptic_window[1] * 1000, color=t["grid"], lw=0)
    ax_raw.bar(ms, observed, width=width, color=t["series"], lw=0, label="observed")
    ax_raw.plot(ms, expected, color=t["ink"], lw=1.4, label="expected under jitter")
    ax_raw.set_ylabel("spike pairs")
    ax_raw.tick_params(labelbottom=False)
    ax_raw.legend(fontsize=7, frameon=False, labelcolor=t["ink2"], loc="upper left")
    corrected = np.asarray(observed, float) - np.asarray(expected, float)
    ax_cor.bar(ms, corrected, width=width, color=t["series"], lw=0)
    ax_cor.axhline(0, color=t["axis"], lw=1)
    ax_cor.set_ylabel("observed −\nexpected")
    ax_cor.set_xlabel("lag (ms): partner's spike − selected unit's spike")
    ax_cor.set_xlim(ms[0] - width / 2, ms[-1] + width / 2)
    for ax in (ax_raw, ax_cor):
        _style(ax, t)
        ax.grid(axis="y", color=t["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.axvline(0, color=t["muted"], lw=1, ls=(0, (3, 3)))
    return fig


def ccg_figure(lags, observed, expected, synaptic_window, theme: str) -> bytes:
    """build_ccg_figure as a PNG for the page."""
    return _png(build_ccg_figure(lags, observed, expected, synaptic_window, theme))


def build_trajectory_figure(r, colours: list[str], dims: int, theme: str, title=None) -> Figure:
    """Population trajectories (analysis.trajectories.Trajectories) on pc_1-pc_2, or
    pc_1-pc_3 in 3-D, one line per condition (open circle at the window's start, filled
    dot at the event); below, each component against time. Descriptive: axes are pc_k
    with their share of the shown (held-out) trials' variance, and nothing is labelled."""
    t = _theme(theme)
    k = r.trajectories.shape[2]
    dims = 3 if dims == 3 and k >= 3 else 2
    rows = k
    fig = Figure(figsize=(5.6, 3.6 + 0.85 * rows))
    add, squeeze = _frame(fig, title, t["ink2"])
    top = 3.6 / (3.6 + 0.85 * rows)
    box = (0.13, 1 - top + 0.06, 0.62, top - 0.1)
    if dims == 3:
        ax = add(box, projection="3d")
        ax.view_init(elev=22, azim=-58)
        ax.set_facecolor("none")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_alpha(0)
    else:
        ax = add(box)
    zero = int(np.argmin(np.abs(r.bin_centers)))
    for traj, name, colour, nf, ns in zip(r.trajectories, r.names, colours, r.n_fit, r.n_show):
        coords = [traj[:, i] for i in range(dims)]
        ax.plot(*coords, color=colour, lw=1.8, label=f"{name} ({nf}/{ns})")
        ax.plot(*[[c[0]] for c in coords], "o", mfc="none", mec=colour, ms=5)
        ax.plot(*[[c[zero]] for c in coords], "o", color=colour, ms=5)
    share = [f"{v:.0%}" for v in r.explained_held_out]
    names = [f"{n} ({v})" for n, v in zip(r.axis_names, share)]
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    if dims == 3:
        ax.set_zlabel(names[2], labelpad=8)
        ax.tick_params(colors=t["ink2"], labelsize=7)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color(t["ink2"])
            axis.label.set_size(9)
    else:
        _style(ax, t)
        ax.grid(color=t["grid"], lw=0.6)
        ax.set_axisbelow(True)
    legend = fig.legend(
        loc="upper left",
        bbox_to_anchor=(0.77, (1 - top + 0.06 + top - 0.1) * squeeze),
        fontsize=6.5,
        frameon=False,
        labelcolor=t["ink2"],
        handlelength=1.2,
        title=(
            "trials fit/shown\n○ window start · ● event\n"
            "pc_k (%): share of the\nshown trials' variance"
        ),
        title_fontsize=6.5,
    )
    legend.get_title().set_color(t["ink2"])  # labelcolor themes only the entries
    first = None
    for i in range(rows):
        bottom = (1 - top - 0.02) - (i + 1) * (1 - top - 0.08) / rows
        a = add((0.13, bottom, 0.84, (1 - top - 0.08) / rows - 0.03), sharex=first)
        first = first or a
        for traj, colour in zip(r.trajectories, colours):
            a.plot(r.bin_centers, traj[:, i], color=colour, lw=1.4)
        a.set_ylabel(f"{r.axis_names[i]}\n{share[i]}", rotation=0, ha="right", va="center")
        a.axvline(0, color=t["muted"], lw=1, ls=(0, (3, 3)))
        a.grid(axis="y", color=t["grid"], lw=0.6)
        a.set_axisbelow(True)
        _style(a, t)
        a.tick_params(labelbottom=i == rows - 1, labelsize=7)
    a.set_xlabel("time from event (s)")
    return fig


def trajectory_figure(r, colours: list[str], dims: int, theme: str) -> bytes:
    """build_trajectory_figure as a PNG for the page."""
    return _png(build_trajectory_figure(r, colours, dims, theme))
