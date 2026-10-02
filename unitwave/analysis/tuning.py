"""Tuning curves and selectivity between two conditions, each unit against its own null.

Response rate: spikes in a window after the event / its length, one value per trial.

Tuning curve: the mean ± SEM (across trials) response rate at each level of a
condition, with n per level.

Selectivity: AUROC = P(rate at level b > rate at level a), ties counted half; 0.5 means
no difference. It is tested two-sided against a null, with p = (1 + #null with
|AUROC - 0.5| >= observed) / (1 + n_null). Benjamini-Hochberg runs across the units
tested together, and n_tests is reported. The comparisons and their nulls (R4) come
from the task definition (unitwave/analysis/tasks.py); IBL's, by condition:

- **choice:** labels permuted within signed-contrast strata, zero split by side.
  Choice follows the stimulus on correct trials, so a plain shuffle would call a
  stimulus-driven unit choice-selective.
- **side:** labels permuted within choice strata, the mirror case: a choice-driven
  unit is not stimulus-selective.
- **outcome:** labels permuted within signed-contrast strata. Errors concentrate at
  low contrast, so a contrast-driven unit is not outcome-selective.
- **block:** pseudo-sessions from IBL's block generator (evaluation/nulls.py). Block
  labels are autocorrelated, so any label permutation is invalid for them. Block uses
  the pre-event (baseline) window: the block prior predicts the stimulus side, so a
  post-stimulus window would count stimulus responses as block selectivity.

A task may also declare a plain permutation (`permute: all`) when nothing confounds a
comparison. Signed contrast has many levels: it gets a tuning curve, not a two-level
AUROC.
Seeds and counts come from configs/selectivity.yaml (R7). Trials are treated as
exchangeable within strata; slow drift in a unit's rate is only accounted for by the
block null.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from unitwave.analysis.conditions import (
    condition,
    split_event_times,
    strata_values,
)
from unitwave.analysis.events import trial_event_times
from unitwave.analysis.psth import trial_counts
from unitwave.analysis.responsiveness import ResponseConfig, benjamini_hochberg
from unitwave.analysis.tasks import TaskDefinition, load_task
from unitwave.evaluation.nulls import generate_pseudo_blocks

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "selectivity.yaml"
_KEYS = {"n_permutations", "n_pseudo_sessions", "seed", "min_trials"}
# The IBL definition's comparisons, condition -> (level a, level b): AUROC > 0.5 means
# a higher rate at level b.
COMPARISONS = {name: c.levels for name, c in load_task().comparisons.items()}
# Pseudo-session generators: name -> what the null is called.
_PSEUDO = {"ibl_blocks": "pseudo-sessions from IBL's block generator"}


def _task(task: TaskDefinition | None) -> TaskDefinition:
    return task if task is not None else load_task()


def null_name(name: str, task: TaskDefinition | None = None) -> str:
    """How a comparison's null is described, from the task definition."""
    c = _task(task).comparisons[name]
    if c.pseudo_sessions:
        return _PSEUDO[c.pseudo_sessions]
    if c.permute_within:
        return f"{name} permuted within {c.permute_within.replace('_', ' ')}"
    return f"{name} permuted across trials (no strata declared)"


@dataclass(frozen=True)
class SelectivityConfig:
    n_permutations: int
    n_pseudo_sessions: int
    seed: int
    min_trials: int


def load_selectivity_config(path: str | os.PathLike = DEFAULT_CONFIG) -> SelectivityConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return SelectivityConfig(**{k: int(raw[k]) for k in sorted(_KEYS)})


def window_rates(spikes: np.ndarray, events: np.ndarray, window) -> np.ndarray:
    """(n_trials,) Hz: spikes in [event + start, event + stop) / window length; NaN events
    stay NaN."""
    events = np.asarray(events, np.float64)
    assert events.ndim == 1
    width = float(window[1]) - float(window[0])
    out = np.full(events.shape, np.nan)
    valid = np.isfinite(events)
    if valid.any():
        out[valid] = trial_counts(spikes, events[valid], window, width)[:, 0] / width
    return out


def tuning_curve(
    spikes, trials: pd.DataFrame, event: str, name: str, window, task: TaskDefinition | None = None
) -> pd.DataFrame:
    """(n_levels, 4): level, mean_hz, sem_hz, n per level of the condition, indexed by name.

    SEM is across trials (ddof=1), NaN with one trial.
    """
    rows = []
    for part in split_event_times(trials, event, condition(trials, name, task), task):
        rates = window_rates(spikes, part.times, window)
        rates = rates[np.isfinite(rates)]
        n = rates.size
        sem = rates.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append((part.name, part.level, rates.mean() if n else np.nan, sem, n))
    table = pd.DataFrame(rows, columns=["name", "level", "mean_hz", "sem_hz", "n"])
    return table.set_index("name")


def auroc(rates_a: np.ndarray, rates_b: np.ndarray) -> float:
    """P(rate from b > rate from a), ties counted half."""
    ranks = rankdata(np.concatenate([rates_a, rates_b]))
    n_a, n_b = len(rates_a), len(rates_b)
    return float((ranks[n_a:].sum() - n_b * (n_b + 1) / 2) / (n_a * n_b))


def stratified_permutations(labels: np.ndarray, strata: np.ndarray, n: int, rng) -> np.ndarray:
    """(n, n_trials): labels shuffled independently within each stratum, n times."""
    labels, strata = np.asarray(labels), np.asarray(strata)
    assert labels.shape == strata.shape and labels.ndim == 1
    out = np.empty((n, labels.size), dtype=labels.dtype)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        order = rng.random((n, idx.size)).argsort(axis=1)
        out[:, idx] = labels[idx][order]
    return out


@lru_cache(maxsize=8)
def _pseudo_blocks(n_trials: int, n: int, seed: int) -> np.ndarray:
    """(n, n_trials) pseudo-session probabilityLeft sequences, seeds seed + 1 .. seed + n."""
    return np.vstack([generate_pseudo_blocks(n_trials, seed=seed + 1 + i) for i in range(n)])


def _strata(trials: pd.DataFrame, name: str, task: TaskDefinition) -> np.ndarray:
    within = task.comparisons[name].permute_within
    columns = task.strata_columns(within)
    if any(c not in trials for c in columns):
        what = f"the {columns[0]} column" if len(columns) == 1 else " and ".join(columns)
        raise ValueError(f"{name} selectivity needs {what} to stratify by")
    return strata_values(trials, within, task)


def selectivity(
    spikes,
    unit_ids,
    trials: pd.DataFrame,
    event: str,
    name: str,
    windows: ResponseConfig,
    cfg: SelectivityConfig,
    stratify: bool = True,
    trial_mask: np.ndarray | None = None,
    task: TaskDefinition | None = None,
) -> pd.DataFrame:
    """(n_units, ...) auroc, p, q, selective, n_a, n_b, n_null, n_tests, null, window.

    trial_mask: (n_trials,) trials to use (a trial filter). The full table is still
    passed, so block pseudo-sessions are generated over the whole session and then
    masked, keeping IBL's block structure intact.
    stratify=False replaces the null with a plain label permutation: only to show why
    the stratified and pseudo-session nulls are needed.
    """
    task = _task(task)
    if name not in task.comparisons:
        spec = task.conditions.get(name)
        if spec is not None and spec.type == "ordinal":
            raise ValueError(f"{spec.label.lower()} has many levels: use its tuning curve")
        raise ValueError(f"no selectivity test for {name!r}; available: {sorted(task.comparisons)}")
    unit_ids = list(unit_ids)
    comparison = task.comparisons[name]
    level_a, level_b = comparison.levels
    values = condition(trials, name, task).values
    events = trial_event_times(trials, event, task)
    keep = np.isfinite(events) & np.isin(values, (level_a, level_b))
    if trial_mask is not None:
        trial_mask = np.asarray(trial_mask, bool)
        assert trial_mask.shape == keep.shape, "trial_mask must be (n_trials,)"
        keep &= trial_mask
    strata = None
    if stratify and comparison.permute_within:
        strata = _strata(trials, name, task)
        keep &= np.isfinite(strata)
    y = (values[keep] == level_b).astype(np.int64)  # (n_keep,)
    n_b, n_a = int(y.sum()), int(y.size - y.sum())
    if min(n_a, n_b) < cfg.min_trials:
        raise ValueError(
            f"{name}: {n_a} and {n_b} trials; each condition needs at least {cfg.min_trials}"
        )

    baseline = comparison.window == "baseline"
    window = windows.baseline_window if baseline else windows.response_window
    rates = np.vstack(
        [window_rates(spikes[u], events[keep], window) for u in unit_ids]
    )  # (n_units, n_keep)
    ranks = rankdata(rates, axis=1)
    rng = np.random.default_rng(cfg.seed)
    if comparison.pseudo_sessions and stratify:  # only ibl_blocks is known (tasks.py)
        pseudo = _pseudo_blocks(len(trials), cfg.n_pseudo_sessions, cfg.seed)[:, keep]
        if not np.isin(pseudo, (level_a, level_b)).all():
            raise ValueError(
                f"{name}: some tested trials fall in the pseudo-sessions' unbiased opening "
                "block; this session's block structure differs from IBL's generator"
            )
        labels, described = (pseudo == level_b).astype(np.int64), null_name(name, task)
    elif strata is not None:
        labels = stratified_permutations(y, strata[keep], cfg.n_permutations, rng)
        described = null_name(name, task)
    else:
        labels = stratified_permutations(y, np.zeros_like(y), cfg.n_permutations, rng)
        described = null_name(name, task) if stratify else f"{name} permuted without strata"

    # AUROC from rank sums: ranks don't depend on labels, so one product gives every draw.
    nb = labels.sum(axis=1)  # (n_null,)
    na = labels.shape[1] - nb
    ok = (na > 0) & (nb > 0)
    labels, nb, na = labels[ok], nb[ok], na[ok]
    observed = (ranks @ y - n_b * (n_b + 1) / 2) / (n_a * n_b)  # (n_units,)
    null = (ranks @ labels.T - nb * (nb + 1) / 2) / (na * nb)  # (n_units, n_null)
    tol = 1e-12
    extreme = np.abs(null - 0.5) >= (np.abs(observed - 0.5) - tol)[:, None]
    p = (1 + extreme.sum(axis=1)) / (1 + null.shape[1])
    q = benjamini_hochberg(p)
    return pd.DataFrame(
        {
            "auroc": observed,
            "p": p,
            "q": q,
            "selective": q < windows.alpha,
            "n_a": n_a,
            "n_b": n_b,
            "n_null": int(null.shape[1]),
            "n_tests": len(unit_ids),
            "null": described,
            "window": comparison.window,
            "seed": cfg.seed,
        },
        index=pd.Index(unit_ids, name="unit_id"),
    )
