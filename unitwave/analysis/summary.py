"""Across-session region summaries (S5, plan step 12).  [R4, §5]

Per region, at one level (Beryl by default), across the sessions of a set: how many
units carry a label (responsive, selective or movement-locked), against the session's
other units.

- **The unit of inference is the session.** The region test conditions on each
  session: its units, its labelled units, and its units in the region. Under the
  null, region labels are permuted across a session's units, within each session, so
  the region's labelled count in session s is hypergeometric. The statistic is the
  region's labelled units summed over sessions. Its exact null distribution is the
  convolution of the sessions' hypergeometrics, so no seed is needed.
- **Why within sessions:** a session with many labelled units can't make a region
  look enriched just by contributing many of its units (Simpson's paradox). Pooling
  units across sessions does exactly that (tests/test_summary.py).
- **p and correction:** p is two-sided, twice the smaller tail, at most 1. The
  direction says whether the region holds more or fewer labelled units than expected
  from its sessions. Benjamini-Hochberg runs across the regions with a verdict.
- **Refused regions:** a region with units in fewer than `min_sessions` sessions
  gets no verdict, and the reason is given.
- **Never only pooled numbers (§5):** every region keeps its per-session rows, with
  its units, its labelled units, its fraction, and the fraction among the session's
  other units.

Units without a region at the level (None, "root", "void") count among their
session's units, but are never a region.

Per-session labels (`session_labels`) are Studio's, computed the same way:
- **Which task:** the label's task definition (IBL's by default, step 8c). Its
  events, comparisons and movement events are the ones a label may name.
- **Which units:** the QC-passing units of every probe, by the session's own rule
  (IBL's configs/qc.yaml by default; an NWB layout's quality column, or spike times).
- **Which trials:** the set's trial filter, read with the task's filters.
- **Which test:** responsiveness, selectivity (the circular test for angles) or
  movement locking, with its own null and BH across that session's units
  (configs/analysis.yaml, selectivity.yaml, movement.yaml). tests/test_summary.py and
  tests/test_summary_tasks.py check them against Studio's on real sessions.
- **No regions, no labels:** a session without brain regions is refused, with the
  reason its source gives.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import false_discovery_control, hypergeom

from unitwave.analysis.atlas import region_at_level
from unitwave.analysis.conditions import TrialFilter, apply_trial_filter
from unitwave.analysis.events import event_times
from unitwave.analysis.movement import load_movement_config, movement_locking
from unitwave.analysis.responsiveness import load_response_config, responsiveness
from unitwave.analysis.tasks import DEFAULT_TASK, TaskDefinition, load_task
from unitwave.analysis.tuning import circular_selectivity, load_selectivity_config, selectivity
from unitwave.analysis.units import unit_table
from unitwave.qc.units import load_qc_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "summary.yaml"
_KEYS = ("level", "min_sessions", "alpha")
NOT_REGIONS = ("root", "void")


@dataclass(frozen=True)
class SummaryConfig:
    level: str
    min_sessions: int
    alpha: float


def load_summary_config(path: str | os.PathLike = DEFAULT_CONFIG) -> SummaryConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - set(_KEYS)), sorted(set(_KEYS) - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return SummaryConfig(str(raw["level"]), int(raw["min_sessions"]), float(raw["alpha"]))


@dataclass(frozen=True)
class LabelSpec:
    """kind: "responsive" (to event), "selective" (for comparison split, at event) or
    "locked" (movement-locked; no event). task: a built-in task's name or a definition's
    YAML path; its events, comparisons and movement events are the ones allowed."""

    kind: str
    event: str = ""
    split: str = ""
    task: str = DEFAULT_TASK

    def __post_init__(self) -> None:
        if self.kind not in ("responsive", "selective", "locked"):
            raise ValueError(f"unknown label kind {self.kind!r}: responsive, selective or locked")
        task = self.definition()
        if self.kind != "locked" and self.event not in task.events:
            raise ValueError(f"{self.kind} needs an event, one of {sorted(task.events)}")
        if self.kind == "selective" and self.split not in task.comparisons:
            raise ValueError(
                "selective needs a condition with a comparison in the "
                f"{task.label} definition, one of {sorted(task.comparisons)}"
            )
        if self.kind == "locked" and task.movement is None:
            raise ValueError(
                f"the {task.label} definition declares no movement events, so movement "
                "locking isn't available"
            )

    def definition(self) -> TaskDefinition:
        return load_task(self.task)

    def describe(self) -> str:
        task = self.definition()
        if self.kind == "responsive":
            return f"responsive to {task.events[self.event].label.lower()}"
        if self.kind == "selective":
            return (
                f"selective for {task.conditions[self.split].label.lower()} at "
                f"{task.events[self.event].label.lower()}"
            )
        return "movement-locked"


def session_labels(
    session, label: LabelSpec, trial_filter: dict, *, level: str, qc=None
) -> pd.DataFrame:
    """(n_units, 4) eid, unit_id, region (at level), labelled: the session's QC-passing
    units, labelled by the same test Studio runs, on the set's trial filter. qc: the
    session's unit QC rule (default: IBL's, configs/qc.yaml)."""
    if "units.acronym" not in session.available.present:
        why = session.available.missing.get("units.acronym", "not in this session")
        raise ValueError(f"{session.eid}: no brain regions: {why}")
    task = label.definition()
    task.require(session.trials)
    units = unit_table(session, qc if qc is not None else load_qc_config())
    ids = list(units.index[units["qc_passed"].astype(bool)])
    if not ids:
        raise ValueError(f"{session.eid}: no unit passes QC")
    sel = apply_trial_filter(session.trials, TrialFilter.from_dict(trial_filter, task))
    trials = session.trials[sel.mask]
    response = load_response_config()
    if label.kind == "responsive":
        events = event_times(trials, label.event, task)
        table = responsiveness(session.spikes, ids, events, response)
        column = "responsive"
    elif label.kind == "selective":
        circular = task.comparisons[label.split].kind == "circular"
        table = (circular_selectivity if circular else selectivity)(
            session.spikes,
            ids,
            session.trials,
            label.event,
            label.split,
            response,
            load_selectivity_config(),
            trial_mask=sel.mask,
            task=task,
        )
        column = "selective"
    else:
        table = movement_locking(
            session.spikes, ids, trials, load_movement_config(), response.alpha, task
        )
        column = "locked"
    assert list(table.index) == ids
    regions = region_at_level(units.loc[ids, "region"].to_numpy(), level)
    return pd.DataFrame(
        {
            "eid": session.eid,
            "unit_id": ids,
            "region": pd.Series(regions, dtype=object).where(pd.notna(regions), None),
            "labelled": table[column].astype(bool).to_numpy(),
        }
    )


def _null_counts(sessions: pd.DataFrame) -> np.ndarray:
    """(max count + 1,) exact distribution of the region's labelled units summed over
    sessions, each hypergeometric. sessions: N (units), K (labelled), n (in region)."""
    dist = np.ones(1)
    for big_n, k, n in sessions[["N", "K", "n"]].itertuples(index=False):
        support = np.arange(min(n, k) + 1)
        dist = np.convolve(dist, hypergeom.pmf(support, big_n, k, n))
    return dist


def region_summary(units: pd.DataFrame, cfg: SummaryConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """units: one row per unit, eid, unit_id, region (or None) and labelled.

    Returns (regions, sessions):
    - regions, indexed by region: n_sessions, n_units, observed, expected, p_high,
      p_low, p, q, direction, claim, refused, n_tests;
    - sessions, one row per region and session: n_units, n_labelled, fraction,
      n_rest, n_labelled_rest, fraction_rest.
    """
    units = units.sort_values(["eid", "unit_id"]).reset_index(drop=True)
    labelled = units["labelled"].astype(bool)
    totals = pd.DataFrame(
        {"N": units.groupby("eid").size(), "K": labelled.groupby(units["eid"]).sum()}
    )
    is_region = units["region"].notna() & ~units["region"].isin(NOT_REGIONS)
    inside = units[is_region].assign(labelled=labelled[is_region])
    counts = inside.groupby(["region", "eid"])["labelled"].agg(n="size", x="sum").reset_index()
    counts = counts.join(totals, on="eid")
    sessions = pd.DataFrame(
        {
            "region": counts["region"],
            "eid": counts["eid"],
            "n_units": counts["n"],
            "n_labelled": counts["x"],
            "fraction": counts["x"] / counts["n"],
            "n_rest": counts["N"] - counts["n"],
            "n_labelled_rest": counts["K"] - counts["x"],
        }
    )
    rest = sessions["n_rest"].where(sessions["n_rest"] > 0)
    sessions["fraction_rest"] = sessions["n_labelled_rest"] / rest
    sessions = sessions.sort_values(["region", "eid"]).reset_index(drop=True)

    rows = {}
    for region, group in counts.groupby("region", sort=True):
        n_sessions = len(group)
        observed = int(group["x"].sum())
        expected = float((group["n"] * group["K"] / group["N"]).sum())
        row = {
            "n_sessions": n_sessions,
            "n_units": int(group["n"].sum()),
            "observed": observed,
            "expected": expected,
            "p_high": np.nan,
            "p_low": np.nan,
            "p": np.nan,
            "refused": "",
        }
        if n_sessions < cfg.min_sessions:
            word = "session" if n_sessions == 1 else "sessions"
            row["refused"] = (
                f"in {n_sessions} {word}; a verdict needs at least {cfg.min_sessions} "
                "(configs/summary.yaml)"
            )
        else:
            dist = _null_counts(group)
            row["p_high"] = float(min(1.0, dist[observed:].sum()))
            row["p_low"] = float(min(1.0, dist[: observed + 1].sum()))
            row["p"] = min(1.0, 2 * min(row["p_high"], row["p_low"]))
        rows[region] = row
    regions = pd.DataFrame.from_dict(rows, orient="index")
    regions.index.name = "region"
    tested = regions["refused"] == ""
    regions["q"] = np.nan
    if tested.any():
        regions.loc[tested, "q"] = false_discovery_control(
            regions.loc[tested, "p"].to_numpy(np.float64), method="bh"
        )
    regions["direction"] = np.where(
        regions["observed"] > regions["expected"],
        "more",
        np.where(regions["observed"] < regions["expected"], "fewer", ""),
    )
    regions["claim"] = (regions["q"] < cfg.alpha).fillna(False).astype(bool)
    regions["n_tests"] = int(tested.sum())
    return regions, sessions
