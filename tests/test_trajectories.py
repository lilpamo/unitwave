"""Population trajectories. Spike trains here are test inputs, never shown as data."""

import numpy as np
import pytest

from unitwave.analysis.trajectories import (
    TrajectoryConfig,
    load_trajectory_config,
    trajectories,
)

CFG = TrajectoryConfig(n_components=3, soft_norm_hz=5.0, smooth_s=0.03, min_trials=5)
WINDOW, BIN = (-0.2, 0.6), 0.02


def test_default_config():
    assert load_trajectory_config() == CFG


def _latents(t):
    """(n_bins, 2) two planted time courses over the window."""
    return np.stack([np.exp(-((t - 0.15) ** 2) / 0.005), np.clip(t, 0, None) * 2], axis=1)


def _session(rng, n_units=40, n_trials=60, conditions=(1.0, -1.0)):
    """Rates are a planted rank-2 function of time, with a sign per condition; Poisson
    spikes on a 1 ms grid. Trials alternate condition, 3 s apart, so windows never
    overlap. Returns spikes, per-condition event times, and the planted loadings."""
    loadings = rng.normal(0, 8, (n_units, 2))
    base = rng.uniform(10, 20, n_units)
    dt = 0.001
    t = np.arange(WINDOW[0], WINDOW[1], dt)
    events = {c: [] for c in conditions}
    spikes = [[] for _ in range(n_units)]
    for i in range(n_trials * len(conditions)):
        c = conditions[i % len(conditions)]
        e = 5.0 + 3.0 * i
        events[c].append(e)
        rates = np.clip(base[:, None] + c * loadings @ _latents(t).T, 0, None)  # (n_units, n_t)
        hits = rng.random(rates.shape) < rates * dt
        for u in range(n_units):
            spikes[u].append(e + t[hits[u]])
    spikes = {f"u{u}": np.sort(np.concatenate(s)) for u, s in enumerate(spikes)}
    parts = [(f"cond {c:+g}", np.array(events[c])) for c in conditions]
    return spikes, parts


def test_planted_low_rank_structure_is_recovered_on_held_out_trials():
    rng = np.random.default_rng(0)
    spikes, parts = _session(rng)
    r = trajectories(spikes, list(spikes), parts, WINDOW, BIN, CFG)
    assert r.trajectories.shape == (2, 40, 3) and r.axis_names == ["pc_1", "pc_2", "pc_3"]
    # Two planted dimensions: on held-out trials pc_1 and pc_2 are strong and pc_3 is at
    # the noise floor. (Their total share also depends on Poisson noise, which falls with
    # more trials: 0.70 at 60 trials per condition, 0.91 at 300.)
    held_out = r.explained_held_out
    assert held_out[2] < 0.1 * held_out[1] and held_out[1] > 0.1
    # The held-out trajectories in (pc_1, pc_2) are a linear map of the planted latents.
    planted = np.concatenate([c * _latents(r.bin_centers) for c in (1.0, -1.0)])  # (80, 2)
    shown = r.trajectories[:, :, :2].reshape(-1, 2)
    design = np.column_stack([planted, np.ones(len(planted))])
    fit, *_ = np.linalg.lstsq(design, shown, rcond=None)
    residual = shown - design @ fit
    r2 = 1 - (residual**2).sum(axis=0) / ((shown - shown.mean(axis=0)) ** 2).sum(axis=0)
    assert (r2 > 0.9).all()


def test_held_out_trials_never_enter_the_fit():
    rng = np.random.default_rng(1)
    spikes, parts = _session(rng, n_units=12, n_trials=20)
    before = trajectories(spikes, list(spikes), parts, WINDOW, BIN, CFG)
    # Flood the held-out trials (2nd, 4th, ... of each condition) with spikes in one unit.
    held_out = np.concatenate([times[1::2] for _, times in parts])
    extra = np.concatenate([e + np.linspace(0, 0.5, 200) for e in held_out])
    changed = {**spikes, "u0": np.sort(np.concatenate([spikes["u0"], extra]))}
    after = trajectories(changed, list(changed), parts, WINDOW, BIN, CFG)
    np.testing.assert_array_equal(after.components, before.components)
    np.testing.assert_array_equal(after.explained_fit, before.explained_fit)
    assert not np.allclose(after.trajectories, before.trajectories)  # only the shown half moved


def test_signs_are_fixed_and_conditions_with_too_few_trials_are_excluded_and_counted():
    rng = np.random.default_rng(2)
    spikes, parts = _session(rng, n_units=10, n_trials=20)
    sparse = ("cond rare", parts[0][1][:7])  # 4 fit + 3 shown: fewer than 5 in a half
    r = trajectories(spikes, list(spikes), [*parts, sparse], WINDOW, BIN, CFG)
    assert r.names == ["cond +1", "cond -1"]
    assert r.excluded == {"cond rare": "4 + 3 trials, fewer than 5 in a half"}
    assert r.n_fit == [10, 10] and r.n_show == [10, 10]
    # Each component's largest loading is positive, so reruns draw the same picture.
    big = r.components[np.abs(r.components).argmax(axis=0), np.arange(r.components.shape[1])]
    assert (big > 0).all()
    with pytest.raises(ValueError, match="at least 2 units"):
        trajectories(spikes, ["u0"], parts, WINDOW, BIN, CFG)
    with pytest.raises(ValueError, match="no condition has at least 5 trials"):
        trajectories(spikes, list(spikes), [sparse], WINDOW, BIN, CFG)
