"""Population trajectories: condition-averaged activity on its principal components.

Descriptive: no statistic, no null, no label. Axes are named pc_1, pc_2, ... (R5), and
nothing here says what a component means.

Cross-validated, the heatmap's rule from step 3:
- Each condition's trials with an event time are split alternately
  (psth.alternate_halves): the 1st, 3rd, ... trials fit, the 2nd, 4th, ... are shown.
- Everything learned comes from the fit half alone: each unit's soft-normalisation
  scale (its rate range plus `soft_norm_hz`, Churchland et al. 2012), its mean, and the
  components. The shown half is then normalised and centred with those same numbers
  and projected.
- explained_fit is each component's share of the fit half's variance;
  explained_held_out is its share of the shown half's variance, a check that the
  components describe trials they were not fit on.

Signs of components are arbitrary; each is flipped so its largest loading is
positive, so the same data always draw the same picture.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

from unitwave.analysis.psth import alternate_halves, bin_edges, population_psth

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "trajectories.yaml"
_KEYS = ("n_components", "soft_norm_hz", "smooth_s", "min_trials")


@dataclass(frozen=True)
class TrajectoryConfig:
    n_components: int
    soft_norm_hz: float
    smooth_s: float
    min_trials: int


def load_trajectory_config(path: str | os.PathLike = DEFAULT_CONFIG) -> TrajectoryConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - set(_KEYS)), sorted(set(_KEYS) - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return TrajectoryConfig(
        int(raw["n_components"]),
        float(raw["soft_norm_hz"]),
        float(raw["smooth_s"]),
        int(raw["min_trials"]),
    )


@dataclass(frozen=True)
class Trajectories:
    """names: the conditions shown, in the order given. trajectories: (n_conditions,
    n_bins, k) the shown half projected; components: (n_units, k) loadings, rows in
    unit order; explained_fit, explained_held_out: (k,); n_fit, n_show: trials per
    condition in each half; excluded: condition -> why it is not shown."""

    names: list[str]
    bin_centers: np.ndarray
    trajectories: np.ndarray
    components: np.ndarray
    explained_fit: np.ndarray
    explained_held_out: np.ndarray
    n_fit: list[int]
    n_show: list[int]
    excluded: dict[str, str]
    axis_names: list[str]


def _rates(spikes, unit_ids, events, window, bin_width, smooth_s) -> np.ndarray:
    """(n_units, n_bins) trial-mean rates (Hz), Gaussian-smoothed along time."""
    rates = population_psth(spikes, unit_ids, events, window, bin_width)
    if smooth_s > 0:
        rates = gaussian_filter1d(rates, smooth_s / bin_width, axis=1, mode="nearest")
    return rates


def trajectories(
    spikes, unit_ids, parts, window, bin_width: float, cfg: TrajectoryConfig
) -> Trajectories:
    """parts: [(condition name, (n_trials,) event times, NaN where missing)], one per
    condition (a single part for no split). Returns the cross-validated trajectories."""
    unit_ids = list(unit_ids)
    if len(unit_ids) < 2:
        raise ValueError("trajectories need at least 2 units")
    kept, excluded, fit, show, n_fit, n_show = [], {}, [], [], [], []
    for name, times in parts:
        fit_t, show_t = alternate_halves(np.asarray(times, np.float64))
        if min(fit_t.size, show_t.size) < cfg.min_trials:
            excluded[name] = (
                f"{fit_t.size} + {show_t.size} trials, fewer than {cfg.min_trials} in a half"
            )
            continue
        kept.append(name)
        n_fit.append(int(fit_t.size))
        n_show.append(int(show_t.size))
        fit.append(_rates(spikes, unit_ids, fit_t, window, bin_width, cfg.smooth_s))
        show.append(_rates(spikes, unit_ids, show_t, window, bin_width, cfg.smooth_s))
    if not kept:
        raise ValueError(
            f"no condition has at least {cfg.min_trials} trials in each half "
            "(trials alternate between fitting and showing)"
        )
    fit_a = np.stack(fit)  # (n_conditions, n_units, n_bins)
    show_a = np.stack(show)
    n_cond, n_units, n_bins = fit_a.shape
    assert n_units == len(unit_ids) and show_a.shape == fit_a.shape

    # Learned from the fit half only: scale, mean, components.
    scale = fit_a.max(axis=(0, 2)) - fit_a.min(axis=(0, 2)) + cfg.soft_norm_hz  # (n_units,)
    x_fit = (fit_a / scale[None, :, None]).transpose(1, 0, 2).reshape(n_units, -1)
    mean = x_fit.mean(axis=1, keepdims=True)  # (n_units, 1)
    x_fit = x_fit - mean
    x_show = (show_a / scale[None, :, None]).transpose(1, 0, 2).reshape(n_units, -1) - mean

    u, s, _ = np.linalg.svd(x_fit, full_matrices=False)
    k = min(cfg.n_components, u.shape[1])
    components = u[:, :k].copy()
    biggest = np.abs(components).argmax(axis=0)
    components *= np.sign(components[biggest, np.arange(k)])
    explained_fit = s[:k] ** 2 / (s**2).sum()
    projected = components.T @ x_show  # (k, n_conditions * n_bins)
    explained_held_out = (projected**2).sum(axis=1) / (x_show**2).sum()

    edges = bin_edges(window, bin_width)
    return Trajectories(
        names=kept,
        bin_centers=(edges[:-1] + edges[1:]) / 2,
        trajectories=projected.reshape(k, n_cond, n_bins).transpose(1, 2, 0),
        components=components,
        explained_fit=explained_fit,
        explained_held_out=explained_held_out,
        n_fit=n_fit,
        n_show=n_show,
        excluded=excluded,
        axis_names=[f"pc_{i + 1}" for i in range(k)],
    )
