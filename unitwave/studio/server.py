"""UnitWave Studio: a local web UI over one sorted session.

Standard-library HTTP server, no web framework. It loads the session once and answers
each request by calling analysis/ (numbers), viz/ (PNGs) and data/atlas_meshes
(cached Allen meshes). The page draws what it is sent and computes nothing.

    python -m unitwave.studio.server            # the homepage: choose data
    python -m unitwave.studio.server --eid d23a44ef-1402-4ed7-97f5-47e9a7a504d9
    python -m unitwave.studio.server --phy FOLDER --events events.csv
"""

import argparse
import dataclasses
import hashlib
import json
import mimetypes
import re
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from functools import cached_property
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pandas as pd

from unitwave.analysis.atlas import (
    LEVELS,
    ROOT,
    ccf_um,
    depth_runs,
    probe_track,
    region_at_level,
    region_info,
    region_tree,
    units_in_node,
)
from unitwave.analysis.catalog import (
    SessionFilter,
    filter_sessions,
    load_catalog,
    load_catalog_config,
    probe_lines,
    region_counts,
    summary,
)
from unitwave.analysis.conditions import (
    TrialFilter,
    TrialSelection,
    apply_trial_filter,
    available_conditions,
    available_trial_filters,
    condition,
    filter_levels,
    split_event_times,
)
from unitwave.analysis.correlograms import (
    close_pairs,
    connections,
    cross_correlogram,
    jitter_expected_ccg,
    load_correlogram_config,
)
from unitwave.analysis.decoding import decode, load_decoding_config
from unitwave.analysis.events import available_events, event_times, trial_event_times
from unitwave.analysis.movement import (
    load_movement_config,
    movement_free,
    movement_locking,
    reaction_times,
    wheel_speed_psth,
)
from unitwave.analysis.psth import (
    alternate_halves,
    bin_edges,
    peak_order,
    population_psth,
    psth,
    raster,
    scale_rows_for_display,
    selection_average,
)
from unitwave.analysis.recipes import list_recipes, resolve, window_words
from unitwave.analysis.responsiveness import load_response_config, responsiveness
from unitwave.analysis.summary import LabelSpec
from unitwave.analysis.tasks import DEFAULT_TASK, TaskDefinition, list_tasks, load_task
from unitwave.analysis.trajectories import load_trajectory_config, trajectories
from unitwave.analysis.trial_view import (
    NUMBERING,
    TRIAL_START,
    alignment_label,
    load_trial_view_config,
    position,
    step,
    trial_view,
)
from unitwave.analysis.tuning import (
    circular_selectivity,
    load_selectivity_config,
    selectivity,
    tuning_curve,
)
from unitwave.analysis.unit_quality import load_quality_config, unit_quality
from unitwave.analysis.units import unit_table
from unitwave.data.atlas_meshes import mesh_path
from unitwave.data.cache import SessionCache
from unitwave.data.cluster_files import (
    IBL_RP_ALPHA,
    IBL_RP_CONTAMINATION,
    ibl_criteria,
    ibl_session_folder,
    ibl_sorting_folder,
    ibl_waveform,
    phy_waveform,
)
from unitwave.data.load import DataConfig, key_parts, load_data_config
from unitwave.data.manifest import MANIFEST_VERSION, Manifest, build_manifest
from unitwave.data.sync import load_sync_config
from unitwave.nwb.intake import REPORTS, list_layouts, load_layout
from unitwave.qc.nwb import NwbUnitQC
from unitwave.qc.phy import PhyUnitQC
from unitwave.qc.spike_times import UNAVAILABLE, SpikeQC, load_spike_qc_config
from unitwave.studio import APP_NAME
from unitwave.studio.analysis_log import log_key, make_entry, running_total, write_log
from unitwave.studio.entry import (
    complete_nwb_path,
    complete_phy_path,
    project_path,
    recent_projects,
    resolve_locations_file,
    resolve_nwb_file,
    resolve_phy_folder,
    resolve_recording,
    resolve_sync_file,
)
from unitwave.studio.export import export_view
from unitwave.studio.freshness import Freshness
from unitwave.studio.plans import plan_status, source_identity
from unitwave.studio.project import (
    DEFAULT_VIEW,
    REPO,
    SUFFIX,
    Source,
    _configs,
    load_source,
    make_project,
    open_project,
    qc_config_path,
    save_project,
    saving_path,
    session_fingerprint,
)
from unitwave.studio.sets import list_sets, read_set, save_set, set_path
from unitwave.studio.summaries import list_summaries, read_summary
from unitwave.targets.config import load_target_config
from unitwave.viz.studio_plots import (
    THEMES,
    TraceGroup,
    ccg_figure,
    condition_colours,
    lab_colours,
    population_figure,
    probe_colours,
    quality_figure,
    ramp_colours,
    trajectory_figure,
    trial_figure,
    tuning_figure,
    unit_figure,
    wheel_figure,
)
from unitwave.viz.summary_plots import region_flatmap, region_spread

HERE = Path(__file__).parent
STATIC = HERE / "static"
DEFAULT_LEVEL = "Beryl"
BRAIN_ID = 997  # Allen structure id of the whole brain ("root")
DEFAULT_EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
_MESH = re.compile(r"^/mesh/(\d+)\.obj$")
_MAX_BODY = 1 << 20
RUNS = Path(__file__).resolve().parents[2] / "runs"


def _number(value) -> float | None:
    """A float for JSON; None for NaN or a missing value."""
    return None if value is None or pd.isna(value) else float(value)


def _records(frame: pd.DataFrame) -> list[dict]:
    """JSON-ready rows; NaN becomes null."""
    return json.loads(frame.to_json(orient="records"))


class Studio:
    def __init__(
        self,
        session,
        qc,
        atlas_root: Path,
        source: Source | None = None,
        project_path: Path | None = None,
        view: dict | None = None,
        warnings: list[str] = (),
        ibl_alf: Path | None = None,
        task: TaskDefinition | None = None,
        analysis_log: list | None = None,
    ):
        self.session = session
        # The task definition the trials table is read with (analysis.tasks): the
        # source's, else IBL's. A table without its required columns is refused here.
        if task is None:
            task = load_task(source.task if source is not None else DEFAULT_TASK)
        task.require(session.trials)
        self.task = task
        # The session's alf folder in the local ONE cache (IBL only): waveforms and
        # IBL's per-criterion metrics. None when not downloaded.
        self.ibl_alf = ibl_alf
        self.quality_cfg = load_quality_config()
        self.ccg_cfg = load_correlogram_config()
        self.traj_cfg = load_trajectory_config()
        # The exact shown unit set -> connections result: BH ran over its pairs.
        self._connections: dict[tuple, pd.DataFrame] = {}
        self._ibl_criteria: dict[str, pd.DataFrame | str] = {}
        self.qc = qc
        self.source = source  # where the session came from; recorded in projects and exports
        # A project opened from a file with the old ending saves beside it under the new
        # ending (studio.project.saving_path); the opened file is never overwritten.
        self.opened_path = project_path
        self.project_path = None if project_path is None else saving_path(project_path)
        self.view = {**DEFAULT_VIEW, **(view or {})}
        self.warnings = list(warnings)
        # The spike-time rule shown beside the source's own (qc.spike_times).
        self.spike_qc = qc if isinstance(qc, SpikeQC) else load_spike_qc_config()
        self.units = unit_table(session, qc, self.spike_qc)
        self.atlas_root = atlas_root
        self.has_regions = "units.acronym" in session.available.present
        self.has_positions = all(f"units.{a}" in session.available.present for a in "xyz")
        self.response_cfg = load_response_config()
        self.selectivity_cfg = load_selectivity_config()
        self.movement_cfg = load_movement_config()
        self.trial_cfg = load_trial_view_config()
        # (all units, probe, trial filter) -> movement-locking result: BH over that set.
        self._locking: dict[tuple, pd.DataFrame] = {}
        # (event, condition, all units, probe, trial filter) -> result: BH over that set.
        self._selectivity: dict[tuple, pd.DataFrame] = {}
        self.probes = sorted(self.units["probe"].unique())
        # (event, all units, probe, trial filter) -> result: BH ran over exactly that set.
        self._tests: dict[tuple, pd.DataFrame] = {}
        # Decoding (S3): one background job at a time; the page polls decode_status.
        self.decode_cfg = load_decoding_config()
        self._decode_lock = threading.Lock()
        self._decode: dict = {"state": "idle"}
        # The analysis log (step 9b): every test computed here, written to the project
        # file at once; held-out plans decide which runs are confirmatory.
        self.analysis_log: list[dict] = list(analysis_log or [])
        self.plans_dir = load_data_config().data_root / "plans"
        self._log_lock = threading.Lock()

    def _test_key(self, q: dict) -> tuple:
        """A result belongs to one event, unit set, probe and trial filter."""
        return q.get("event", ""), q.get("all") == "1", q.get("probe", ""), self._filter(q).key()

    def _response_key(self, q: dict) -> tuple:
        """Responsiveness also depends on whether only movement-free trials were used."""
        return (*self._test_key(q), q.get("movement_free") == "1")

    def _filter(self, q: dict) -> TrialFilter:
        """The request's trial filter; no `tf` means every trial."""
        if q.get("tf"):
            return TrialFilter.from_dict(json.loads(q["tf"]), self.task)
        return TrialFilter(self.task)

    def _event_label(self, event: str) -> str:
        if event not in self.task.events:
            raise ValueError(f"unknown event {event!r}; available: {sorted(self.task.events)}")
        return self.task.events[event].label

    def _condition_label(self, name: str) -> str:
        if name not in self.task.conditions:
            raise ValueError(
                f"unknown condition {name!r}; available: {sorted(self.task.conditions)}"
            )
        return self.task.conditions[name].label

    def _colours(self, split: str, levels, theme: str, *, for_axis: bool = False) -> list[str]:
        """IBL's conditions have their own colours (sides, contrast ramp). Another task's
        (step 8b): an ordered condition takes the one-hue ramp; a categorical one, or an
        angle (a circular comparison), the categorical slots in order. More categories than
        slots can't be told apart by colour: refused, except on an axis that names each
        level (for_axis: the tuning curve), where they share one colour."""
        if self.task.name == DEFAULT_TASK:
            return condition_colours(split, levels, theme)
        spec = self.task.conditions.get(split)
        comparison = self.task.comparisons.get(split)
        circular = comparison is not None and comparison.kind == "circular"
        n = len(levels)
        if spec is not None and spec.type in ("ordinal", "continuous") and not circular:
            return ramp_colours(n, theme)
        slots = THEMES[theme]["categorical"]
        if n <= len(slots):
            return list(slots[:n])
        if for_axis:
            return [THEMES[theme]["series"]] * n
        label = spec.label if spec is not None else split
        raise ValueError(
            f"{label} has {n} levels, more than the {len(slots)} colours that can be told "
            f"apart, so its trials aren't drawn by colour: read its tuning curve, or keep "
            f"{len(slots)} levels or fewer with the trial filters"
        )

    def _level_names(self, name: str, levels) -> list[str]:
        """Names of the given levels of a condition, as this session's table has them."""
        c = condition(self.session.trials, name, self.task)
        named = dict(zip(c.levels, c.names))
        return [named.get(float(v), f"{float(v):g}") for v in levels]

    def _trials(self, q: dict) -> tuple[pd.DataFrame, TrialSelection]:
        sel = apply_trial_filter(self.session.trials, self._filter(q))
        return self.session.trials[sel.mask], sel

    @staticmethod
    def _trial_note(sel: TrialSelection) -> str:
        """For captions: how many trials the filters excluded, and why."""
        if not sel.n_excluded:
            return ""
        why = ", ".join(f"{n} {reason}" for reason, n in sel.excluded.items())
        return f" · trial filters kept {sel.n_kept} of {sel.n_total} ({why})"

    def _tested(self, q: dict) -> pd.DataFrame | None:
        return self._tests.get(self._response_key(q))

    def _level(self, q: dict) -> str:
        level = q.get("level", DEFAULT_LEVEL)
        if level not in LEVELS:
            raise ValueError(f"unknown region level {level!r}")
        return level

    def _regions(self, q: dict) -> pd.Series:
        """(n_units,) region at the requested level; None without regions."""
        if not self.has_regions:
            return pd.Series(None, self.units.index, dtype=object)
        values = region_at_level(self.units["region"].to_numpy(), self._level(q))
        return pd.Series(values, self.units.index, dtype=object)

    def _select(self, q: dict) -> list[str]:
        keep = np.ones(len(self.units), bool) if q.get("all") == "1" else self.units["qc_passed"]
        if q.get("probe"):
            if q["probe"] not in self.probes:
                raise ValueError(f"no probe {q['probe']!r} in this session; it has {self.probes}")
            keep = keep & (self.units["probe"] == q["probe"])
        if q.get("node"):
            keep = keep & units_in_node(self._regions(q).to_numpy(), q["node"])
        if q.get("responsive") == "1":
            tested = self._tested(q)
            if tested is None:
                raise ValueError("run the responsiveness test for this event first")
            keep = keep & self.units.index.isin(tested.index[tested["responsive"]])
        return list(self.units.index[np.asarray(keep, bool)])

    def _params(self, q: dict):
        window = (float(q["t0"]), float(q["t1"]))
        baseline = (float(q["b0"]), float(q["b1"])) if q.get("baseline") == "1" else None
        events = event_times(self._trials(q)[0], q["event"], self.task)
        return window, float(q["bin"]), baseline, events

    def session_json(self, q: dict) -> dict:
        missing = self.session.available.missing
        return {
            "log": running_total(self.analysis_log),
            "eid": self.session.eid,
            "n_trials": self.session.n_trials,
            "n_units_total": len(self.units),
            "n_units_passing": int(self.units["qc_passed"].sum()),
            "n_units_passing_spikes": int(self.units["spike_qc_passed"].sum()),
            "task": self._task_json(),
            "source": self._source_json(),
            "events": available_events(self.session.trials, self.task),
            "levels": list(LEVELS),
            "default_level": DEFAULT_LEVEL,
            "missing": {k: v for k, v in missing.items() if k.startswith("units.")},
            "response": dict(self.response_cfg.__dict__),
            "probes": self.probes,
            "probe_colours": {theme: probe_colours(self.probes, theme) for theme in THEMES},
            "conditions": available_conditions(self.session.trials, self.task),
            "comparisons": {
                name: self._comparison_json(name)
                for name in available_conditions(self.session.trials, self.task)
                if name in self.task.comparisons
            },
            "selectivity": dict(self.selectivity_cfg.__dict__),
            "trial_filters": available_trial_filters(self.session.trials, self.task),
            "qc": self._qc_info(),
            "trial_levels": self._trial_levels(),
            "movement": {
                "wheel": "wheel" in self.session.behaviour,
                "wheel_missing": self.session.available.missing.get("behaviour.wheel", ""),
                "first_movement": self._has_movement(),
                **self._movement_names(),
                "windows": [
                    list(self.movement_cfg.pre_window),
                    list(self.movement_cfg.post_window),
                ],
            },
            "decoding": self._decoding_info(),
            "correlograms": {
                "max_units": self.ccg_cfg.max_units,
                "window_s": self.ccg_cfg.window_s,
                "jitter_s": self.ccg_cfg.jitter_s,
                "synaptic_window_s": list(self.ccg_cfg.synaptic_window_s),
                "close_um": self.ccg_cfg.close_um,
                "null": self.ccg_cfg.null,
            },
            "trial_view": {
                "numbering": NUMBERING,
                "alignments": {
                    TRIAL_START: "Trial start",
                    **{
                        c: alignment_label(c, self.task).capitalize()
                        for c in self.task.trial_view_events
                        if c in self.session.trials
                    },
                },
                "pre_pad_s": self.trial_cfg.pre_pad_s,
                "post_pad_s": self.trial_cfg.post_pad_s,
                "n_trials": self.trial_cfg.n_trials,
                "max_trials": self.trial_cfg.max_trials,
                "traces": {
                    name: {
                        "available": name in self.session.behaviour,
                        "reason": missing.get(f"behaviour.{name}", ""),
                    }
                    for name in self.task.traces
                },
            },
            "project": {
                "path": None if self.project_path is None else str(self.project_path),
                "opened": (
                    str(self.opened_path)
                    if self.opened_path is not None and self.opened_path != self.project_path
                    else None
                ),
                "view": self.view,
                "warnings": self.warnings,
            },
        }

    def _trial_levels(self) -> dict:
        """The values the select trial filters can choose from, in this session."""
        return {
            name: [level["value"] for level in levels]
            for name, levels in self._filter_levels().items()
        }

    def _filter_levels(self) -> dict:
        """Each available select filter's choices, named: {name: [{value, name}]}."""
        offered = available_trial_filters(self.session.trials, self.task)
        return {
            name: filter_levels(self.session.trials, name, self.task)
            for name, f in self.task.trial_filters.items()
            if f.kind == "select" and offered[name]["available"]
        }

    def _has_movement(self) -> bool:
        m = self.task.movement
        return m is not None and all(
            c in self.session.trials
            for e in (m.stimulus, m.movement)
            for c in self.task.events[e].columns
        )

    def _source_json(self) -> dict:
        """Where the session came from, and, for an NWB file, what the file supports
        (nwb.intake's capability report)."""
        s = self.source
        if s is None or s.kind == "ibl":
            return {"kind": "ibl", "label": f"IBL session · {self.session.eid}"}
        if s.kind == "phy":
            return {
                "kind": "phy",
                "label": f"Phy folder · {s.folder}",
                "report": REPORTS.get(self.session.eid),  # its clock (step 13a)
            }
        if s.kind == "recording":  # step 14a
            report = REPORTS.get(self.session.eid) or {"probes": []}
            return {
                "kind": "recording",
                "label": f"Recording · {Path(s.file).parent.name} · {len(report['probes'])} probes",
                "report": report,
            }
        layout = load_layout(s.layout)
        return {
            "kind": "nwb",
            "label": f"NWB file · {Path(s.file).name} · {layout.label}",
            "report": REPORTS.get(self.session.eid),
        }

    def _comparison_json(self, name: str):
        """A two-level comparison's level names, [a, b]; a circular one's kind and period."""
        c = self.task.comparisons[name]
        if c.kind == "circular":
            return {"kind": "circular", "period": c.period, "window": c.window_label}
        return self._level_names(name, c.levels)

    def _qc_info(self) -> dict:
        """Which rule decides QC here, and the spike-time rule shown beside it."""
        qc = self.qc
        if isinstance(qc, SpikeQC):
            rule = "spike times"
        elif isinstance(qc, NwbUnitQC):
            rule = qc.rule
        else:
            rule = "Phy group" if isinstance(qc, PhyUnitQC) else "IBL label"
        s = self.spike_qc
        return {
            "rule": rule,
            "config": str(qc_config_path(qc).relative_to(REPO)),
            "spike_config": str(qc_config_path(s).relative_to(REPO)),
            "spike_rule": (
                f"task-period rate ≥ {s.min_firing_rate_hz:g} Hz; IBL's sliding refractory "
                f"test ({s.refractory_contamination:.0%} contamination, "
                f"{1 - s.refractory_alpha:.0%} confidence); presence ratio ≥ "
                f"{s.min_presence_ratio:g} in {s.presence_window_s:g} s bins over the task"
            ),
            "spike_unavailable": UNAVAILABLE,
        }

    def _movement_names(self) -> dict:
        """The task's stimulus and movement events, and the locking null, in its words."""
        m = self.task.movement
        if m is None:
            return {"stimulus": None, "stimulus_label": None, "movement_label": None, "null": None}
        return {
            "stimulus": m.stimulus,
            "stimulus_label": self.task.events[m.stimulus].label,
            "movement_label": self.task.events[m.movement].label,
            "null": f"reaction times permuted within {m.strata.replace('_', ' ')}",
        }

    def _task_json(self) -> dict:
        """The task definition as the page needs it: its names, and the choices of each
        trial filter, named."""
        t, m = self.task, self.task.movement
        levels = self._filter_levels()
        return {
            "name": t.name,
            "label": t.label,
            "conditions": {k: {"label": c.label, "type": c.type} for k, c in t.conditions.items()},
            "trial_filters": {
                k: {"label": f.label, "kind": f.kind, "levels": levels.get(k)}
                for k, f in t.trial_filters.items()
            },
            "movement": (
                None
                if m is None
                else {
                    "stimulus": t.events[m.stimulus].label,
                    "movement": t.events[m.movement].label,
                }
            ),
        }

    def save(self, view: dict) -> dict:
        """Write the project file: source, hashes, configs and this view, never results."""
        if self.source is None or self.project_path is None:
            raise ValueError("this Studio was started without a data source to save")
        project = make_project(self.source, self.session, self.qc, view)
        with self._log_lock:
            project["analysis_log"] = self.analysis_log
            save_project(project, self.project_path)
        self.view = {**DEFAULT_VIEW, **view}
        return {"path": str(self.project_path)}

    def export(self, view: dict) -> dict:
        out = export_view(self, view, RUNS)
        return {"folder": str(out), "files": sorted(p.name for p in out.iterdir())}

    # ---------- the analysis log (step 9b) ----------
    @cached_property
    def _fingerprint(self) -> str:
        return session_fingerprint(self.session)

    @cached_property
    def _identity(self) -> tuple[str, str | None]:
        """(eid, sha256 of the source's files), as held-out plans name sessions."""
        return source_identity(self.source, self.session)

    def _config_hashes(self) -> dict:
        return {name: c["sha256"] for name, c in _configs(self.qc).items()} | {
            "task": self.task.name
        }

    def _log(self, kind: str, what: str, params: dict, t: pd.DataFrame | None, **counts) -> dict:
        """Log a test that was just computed: write it to the project file at once, and
        attach the entry to its table (t.attrs) so its status shows with the result.
        params["recipe"]: the recipe step that ran it, if any (only those can be
        confirmatory)."""
        params = dict(params)
        recipe = params.pop("recipe", None)
        at = datetime.now(UTC)
        status = plan_status(
            self.plans_dir,
            eid=self._identity[0],
            sha256=self._identity[1] if recipe else None,
            step=recipe,
            trial_filter=params.get("trial_filter", {}),
            default_units=not params.get("all") and not params.get("probe"),
            at=at,
            claim=True,
        )
        with self._log_lock:
            entry = make_entry(
                self.analysis_log,
                at=at,
                kind=kind,
                what=what,
                params=params,
                recipe=recipe,
                key=log_key(self._fingerprint, kind, params, self._config_hashes()),
                status=status,
                **counts,
            )
            self.analysis_log.append(entry)
            if self.project_path is not None and self.source is not None:
                write_log(
                    self.project_path,
                    self.analysis_log,
                    lambda: make_project(self.source, self.session, self.qc, self.view),
                )
        if t is not None:
            t.attrs["log"] = entry
        return entry

    def _test_params(self, q: dict, **extra) -> dict:
        """What a logged test was run on: the units shown and the trial filter (normalised)."""
        return {
            "all": q.get("all") == "1",
            "probe": q.get("probe", ""),
            "trial_filter": self._filter(q).to_dict(),
            **extra,
            "recipe": q.get("_recipe"),
        }

    @staticmethod
    def _status(t: pd.DataFrame) -> dict:
        """A result's status from its log entry: exploratory (with why) or confirmatory."""
        entry = t.attrs.get("log") or {"status": "exploratory", "why": "not logged"}
        return {k: entry[k] for k in ("status", "why", "plan") if k in entry}

    def log_json(self, q: dict) -> dict:
        """The running total and the entries, newest first."""
        with self._log_lock:
            log = list(self.analysis_log)
        return {"total": running_total(log), "entries": log[::-1]}

    def test_json(self, q: dict) -> dict:
        """Run (or reuse) the responsiveness test on every shown unit for one event."""
        key = self._response_key(q)
        if key not in self._tests:
            ids = self._select({"all": q.get("all", "0"), "probe": q.get("probe", "")})
            if not ids:
                raise ValueError("no units to test")
            trials = self._trials(q)[0]
            if q.get("movement_free") == "1":
                m = self.task.movement
                if m is None:
                    raise ValueError(
                        f"the {self.task.label} definition declares no movement events"
                    )
                if q["event"] != m.stimulus:
                    raise ValueError(
                        "movement-free trials are defined for "
                        f"{self.task.events[m.stimulus].label.lower()} only"
                    )
                free = movement_free(trials, self.response_cfg.response_window[1], self.task)
                trials = trials[free]
            events = event_times(trials, q["event"], self.task)
            t = responsiveness(self.session.spikes, ids, events, self.response_cfg)
            free = q.get("movement_free") == "1"
            self._log(
                "responsiveness",
                f"responsive to {self._event_label(q['event']).lower()}"
                + (" (movement-free trials)" if free else ""),
                self._test_params(q, event=q["event"], movement_free=free),
                t,
                n_tests=int(t["n_tests"].iloc[0]),
                n_units=len(ids),
                n_trials=int(t["n_trials"].iloc[0]),
            )
            self._tests[key] = t
        return self._summary(self._tests[key], key[-1])

    def _summary(self, t: pd.DataFrame, movement_free_only: bool = False) -> dict:
        up = t["responsive"] & (t["statistic_hz"] > 0)
        return self._status(t) | {
            "movement_free": movement_free_only,
            "n_tests": int(t["n_tests"].iloc[0]),
            "n_responsive": int(t["responsive"].sum()),
            "n_up": int(up.sum()),
            "n_down": int((t["responsive"] & ~up).sum()),
            "n_shifts": int(t["n_shifts"].iloc[0]),
            "n_trials": int(t["n_trials"].iloc[0]),
            "n_excluded": int(t["n_excluded"].iloc[0]),
            "probes": sorted(self.units.loc[t.index, "probe"].unique()),
        }

    def units_json(self, q: dict) -> dict:
        """The units in the selected tree node; the tree counts every unit shown."""
        regions = self._regions(q)
        everywhere = self._select({**q, "node": ""})
        shown = self.units.loc[self._select(q)]
        info = region_info(regions.dropna().unique()) if self.has_regions else {}
        rows = shown.assign(
            region_level=regions[shown.index],
            colour=[info.get(r, {}).get("colour") for r in regions[shown.index]],
        )
        tree = region_tree(regions[everywhere].to_numpy()) if self.has_regions else []
        tested = self._tested(q)
        if tested is not None:
            t = tested.reindex(rows.index)  # every shown unit was tested together
            tested_here = t["responsive"].notna()
            responsive = t["responsive"].where(tested_here, False).astype(bool)
            verdict = np.select(
                [~tested_here, ~responsive, t["statistic_hz"] > 0], [None, "no", "up"], "down"
            )
            rows = rows.assign(
                resp_q=t["q"], resp_p=t["p"], resp_hz=t["statistic_hz"], resp=verdict
            )
        locking = self._locking.get(self._locking_key(q))
        if locking is not None:
            t = locking.reindex(rows.index)
            rows = rows.assign(
                lock_hz=t["statistic_hz"], lock_p=t["p"], lock_q=t["q"], locked=t["locked"]
            )
        chosen = self._selectivity.get(self._selectivity_key(q))
        if chosen is not None:
            t = chosen.reindex(rows.index)
            stat = t["auroc"] if "auroc" in t else t["index"]
            rows = rows.assign(sel_auroc=stat, sel_p=t["p"], sel_q=t["q"], sel=t["selective"])
            if "preferred_deg" in t:
                rows = rows.assign(sel_preferred=t["preferred_deg"])
        tested_pairs = self._connections.get(self._pairs_key(q))
        if tested_pairs is not None:
            hit = tested_pairs[tested_pairs["connected"]]
            to = hit.groupby("pre")["post"].apply(list)
            frm = hit.groupby("post")["pre"].apply(list)
            rows = rows.assign(
                conn_out=hit["pre"].value_counts().reindex(rows.index, fill_value=0),
                conn_in=hit["post"].value_counts().reindex(rows.index, fill_value=0),
                conn_to=[to.get(u, []) for u in rows.index],
                conn_from=[frm.get(u, []) for u in rows.index],
            )
        return {
            "level": self._level(q) if self.has_regions else None,
            "units": _records(rows.reset_index().rename(columns={"unit_id": "id"})),
            "tree": tree,
            "test": (
                None if tested is None else self._summary(tested, q.get("movement_free") == "1")
            ),
            "selectivity": None if chosen is None else self._selectivity_summary(chosen, q),
            "trials": self._trial_summary(q),
            "locking": None if locking is None else self._locking_summary(locking),
            "connections": (
                None if tested_pairs is None else self._connections_summary(tested_pairs)
            ),
        }

    def _decoding_info(self) -> dict:
        """What the page offers for decoding, and why not when it can't."""
        cfg, windows = self.decode_cfg, load_target_config().windows
        ibl = self._bwm()
        if not ibl and self.task.name != DEFAULT_TASK:
            return self._task_decoding_info()  # a task's own targets (step 8a)

        def label(target: str) -> str:
            if target == "movement_state":
                return "movement state (each 20 ms bin, from the second before it)"
            w = windows[target]
            anchor = {"stimOn_times": "stimulus onset", "firstMovement_times": "first movement"}

            def ms(t: float) -> str:
                return "0" if t == 0 else f"{'−' if t < 0 else '+'}{abs(t) * 1000:.0f}"

            span = f"{ms(w.start_s)} to {ms(w.stop_s)} ms from {anchor[w.anchor]}"
            extra = "; 0% contrast trials excluded" if target == "stimulus_side" else ""
            return f"{target.replace('_', ' ')} ({span}{extra})"

        return {
            "available": ibl,
            "why": (
                ""
                if ibl
                else (
                    "Decoding needs a Brain Wide Map session: splits come from the split "
                    "registry, built on the release's session manifest, which "
                    f"{'an NWB file' if self.source and self.source.kind == 'nwb' else 'a Phy folder'}"
                    " lacks."
                )
            ),
            "targets": [{"id": t, "label": label(t)} for t in cfg.targets],
            "train_fraction": cfg.train_fraction,
            "gap_s": cfg.gap_s,
            "leave_one_block_out": list(cfg.leave_one_block_out),
            "n_shifts": cfg.n_shifts,
            "n_bootstrap": cfg.n_bootstrap,
            "alpha": cfg.alpha,
        }

    def _bwm(self) -> bool:
        return (
            self.source is not None and self.source.kind == "ibl" and self.source.backend == "bwm"
        )

    def _task_decoding_info(self) -> dict:
        """A session outside the BWM release decodes its task's own targets, if its
        definition declares any (step 8a)."""
        cfg, d = self.decode_cfg, self.task.decoding
        base = {
            "train_fraction": cfg.train_fraction,
            "gap_s": cfg.gap_s,
            "leave_one_block_out": [],
            "n_shifts": cfg.n_shifts,
            "n_bootstrap": cfg.n_bootstrap,
            "alpha": cfg.alpha,
        }
        if d is None or not d.targets:
            why = f"The task definition ({self.task.label}) declares no decoding targets."
            return {**base, "available": False, "why": why, "targets": []}
        targets, lacking = [], {}
        trials = self.session.trials
        for name, t in d.targets.items():
            needed = [
                *self.task.conditions[t.condition].columns,
                *self.task.events[t.event].columns,
                *(c for f in t.trial_filters for c in self.task.trial_filters[f].columns),
                *self.task.trialstruct,
            ]
            missing = [c for c in dict.fromkeys(needed) if c not in trials]
            if missing:
                lacking[name] = missing
                continue
            event = self.task.events[t.event].label.lower()
            span = f"{t.start_s * 1000:+.0f} to {t.stop_s * 1000:+.0f} ms from {event}"
            targets.append({"id": f"task:{name}", "label": f"{t.label} ({span})"})
        if not targets:
            why = "; ".join(
                f"{name} needs {', '.join(cols)}, which this session's trials lack"
                for name, cols in lacking.items()
            )
            return {
                **base,
                "available": False,
                "why": f"No decoding target fits: {why}.",
                "targets": [],
            }
        return {**base, "available": True, "why": "", "targets": targets}

    def decode_start(self, body: dict, manifest) -> dict:
        """Start decoding body["target"] from the units shown by body["query"], in the
        background (analysis.decoding). One run at a time."""
        target = body.get("target")
        if self.task.name != DEFAULT_TASK and not self._bwm():  # a task's own targets
            info = self._task_decoding_info()
            offered = [t["id"] for t in info["targets"]]
            if not info["available"]:
                raise ValueError(info["why"])
            if target not in offered:
                raise ValueError(f"{target!r} is not a decoding target here: one of {offered}")
        elif not self._bwm():
            raise ValueError(
                "decoding needs a Brain Wide Map session: splits come from the split "
                "registry, which is built on the release's session manifest, and a Phy "
                "folder has none yet"
            )
        elif target not in self.decode_cfg.targets:
            raise ValueError(
                f"{target!r} is not a Studio decoding target: one of {list(self.decode_cfg.targets)}"
            )
        q = {k: str(v) for k, v in (body.get("query") or {}).items()}
        recipe = body.get("recipe")
        if recipe is not None:  # a recipe step may only start the target it names
            name, _, step_id = str(recipe).partition("/")
            step = next(
                (
                    s
                    for r in self._recipes()
                    if r["name"] == name
                    for s in r["steps"]
                    if s["id"] == step_id
                ),
                None,
            )
            if step is None or step["run"].get("target") != target:
                raise ValueError(f"{recipe!r} is not a recipe step decoding {target!r}")
        q["_recipe"] = recipe
        if q.get("responsive") == "1":
            raise ValueError(
                "decoding can't use 'Responsive only': responsiveness was tested on every "
                "trial, the decoder's test trials included, so units chosen by it would let "
                "test data pick the units. Untick it to decode"
            )
        shown = list(self._select(q))
        with self._decode_lock:
            if self._decode["state"] == "running":
                raise ValueError("a decoding run is already running; wait for it to finish")
            self._decode = {
                "state": "running",
                "target": target,
                "stage": "starting",
                "done": 0,
                "total": 1,
                "started": time.time(),
                "n_units_shown": len(shown),
            }
        threading.Thread(
            target=self._decode_job, args=(target, shown, manifest, q), daemon=True
        ).start()
        return self.decode_status({})

    def _decode_job(self, target: str, shown: list, manifest, q: dict | None = None) -> None:
        def progress(stage: str, done: int, total: int) -> None:
            with self._decode_lock:
                self._decode.update(stage=stage, done=done, total=total)

        try:
            own = {} if self._bwm() else {"session": self.session, "qc": self.qc, "task": self.task}
            run = decode(
                self.session.eid,
                target,
                unit_ids=shown,
                cfg=self.decode_cfg,
                manifest=manifest if self._bwm() else None,
                load=lambda eid: self.session,
                progress=progress,
                **own,
            )
            summary = run.summary()
            units = "\n".join(shown).encode()
            entry = self._log(
                "decoding",
                f"decoding {target}",
                {
                    "target": target,
                    "units_sha256": hashlib.sha256(units).hexdigest(),
                    "all": (q or {}).get("all") == "1",
                    "probe": (q or {}).get("probe", ""),
                    "recipe": (q or {}).get("_recipe"),
                },
                None,
                n_tests=len(summary["tests"]),
                n_units=len(shown),
                n_trials=None,
            )
            summary |= {k: entry[k] for k in ("status", "why", "plan") if k in entry}
            update = {"state": "done", "summary": summary}
        except ValueError as e:  # a plain refusal, e.g. too few biased blocks
            update = {"state": "error", "error": str(e)}
        except Exception as e:  # noqa: BLE001 - a bug: plain words for the page, details in the log
            traceback.print_exc(file=sys.stderr)
            update = {"state": "error", "error": f"decoding failed ({type(e).__name__}: {e})"}
        with self._decode_lock:
            self._decode.update(update, finished=time.time())

    def decode_status(self, q: dict) -> dict:
        """The decoding job: idle, running (stage, done of total), done (summary) or error."""
        with self._decode_lock:
            status = dict(self._decode)
        if "started" in status:
            status["elapsed_s"] = round(status.get("finished", time.time()) - status["started"], 1)
        return status

    # ---------- recipes (step 9) ----------
    def _decoding_words(self, condition: str) -> tuple[str, str] | str:
        """(target id, its window in words) for a decoding target on `condition` that this
        session can run; else why not."""
        if self._bwm():
            if condition not in self.decode_cfg.targets:
                return f"IBL decoding has no target on {condition}."
            w = load_target_config().windows[condition]
            anchor = {"stimOn_times": "stimulus onset", "firstMovement_times": "first movement"}
            return condition, window_words(w.start_s, w.stop_s, anchor[w.anchor])
        info = self._decoding_info()
        if not info["available"]:
            return info["why"]
        d = self.task.decoding
        for name, t in (d.targets if d is not None else {}).items():
            if t.condition == condition and f"task:{name}" in {x["id"] for x in info["targets"]}:
                label = self.task.events[t.event].label.lower()
                return f"task:{name}", window_words(t.start_s, t.stop_s, label)
        return f"The {self.task.label} definition declares no decoding target on {condition}."

    def _recipes(self) -> list[dict]:
        regions = None
        if not self.has_regions:
            regions = self.session.available.missing.get("units.acronym", "not in this session")
        conditions = {r.needs["comparison"] for r in list_recipes() if "comparison" in r.needs}
        decoding = {c: self._decoding_words(c) for c in conditions}
        return [resolve(r, self.task, regions=regions, decoding=decoding) for r in list_recipes()]

    def recipes_json(self, q: dict) -> dict:
        """The recipes, in this session's task words, each available or greyed out with why."""
        return {"recipes": self._recipes()}

    def recipe_run(self, q: dict) -> dict:
        """Run one recipe step: the manual view's own method with the manual view's
        parameters (the page's unit set and trial filter, the step's event, split or
        movement-free trials). Decoding returns its target for the page's decoding pane
        to start; a split view returns what the toolbar should show; a region summary
        returns its command line, which is run outside the page."""
        recipe = next((r for r in self._recipes() if r["name"] == q.get("recipe")), None)
        if recipe is None:
            raise ValueError(f"no recipe {q.get('recipe')!r}")
        step = next((s for s in recipe["steps"] if s["id"] == q.get("step")), None)
        if step is None:
            raise ValueError(f"no step {q.get('step')!r} in recipe {recipe['name']}")
        if not step["available"]:
            raise ValueError(step["reason"])
        run, page = (
            step["run"],
            {k: q.get(k, d) for k, d in (("all", "0"), ("probe", ""), ("tf", "{}"))},
        )
        kind = run["kind"]
        step_id = {"_recipe": f"{recipe['name']}/{step['id']}"}
        if kind == "responsiveness":
            query = {
                **page,
                "event": run["event"],
                "movement_free": "1" if run["movement_free"] else "0",
            }
            return {"kind": kind, "query": query, "result": self.test_json(query | step_id)}
        if kind == "movement_locking":
            return {"kind": kind, "query": page, "result": self.locking_json(page | step_id)}
        if kind == "selectivity":
            query = {**page, "event": run["event"], "split": run["split"]}
            return {"kind": kind, "query": query, "result": self.selectivity_json(query | step_id)}
        if kind == "split_view":
            return {"kind": kind, "query": {**page, "event": run["event"], "split": run["split"]}}
        if kind == "decoding":
            return {
                "kind": kind,
                "query": page,
                "target": run["target"],
                "recipe": step_id["_recipe"],
            }
        label = LabelSpec("responsive", run["event"], task=self.task.name)
        if kind == "choose_sessions":
            return {"kind": kind, "command": self._summary_command(label, page)}
        return {
            "kind": kind,
            "command": self._summary_command(label, page),
            "label": label.describe(),
            "runs": [
                r
                for r in list_summaries(RUNS)
                if r["label"] == label.describe() and r["task"] == self.task.name
            ],
        }

    def _summary_command(self, label: LabelSpec, page: dict) -> str:
        """The region-summary command for sessions like this one (cli.summarise)."""
        tail = f"--label {label.kind} --event {label.event}"
        if self.source is not None and self.source.kind == "nwb":
            tf = json.dumps(TrialFilter.from_dict(json.loads(page["tf"]), self.task).to_dict())
            return (
                f'python -m unitwave.cli.summarise --nwb "{self.source.file}" OTHER.nwb … '
                f"--layout {self.source.layout} --name NAME --trial-filter '{tf}' {tail}"
            )
        sets = load_data_config().data_root / "sets"
        return f'python -m unitwave.cli.summarise "{sets}/NAME.unitwave-set.json" {tail}'

    def _pairs_key(self, q: dict) -> tuple:
        """A connections result belongs to exactly the units shown when it ran."""
        return tuple(self._select(q))

    def connections_json(self, q: dict) -> dict:
        """Run (or reuse) the connection test on every pair of shown units."""
        key = self._pairs_key(q)
        if key not in self._connections:
            t = connections(
                self.session.spikes, list(key), self.units, self.ccg_cfg, self.response_cfg.alpha
            )
            units = "\n".join(key).encode()
            self._log(
                "connections",
                "connected pairs among the shown units",
                {"units_sha256": hashlib.sha256(units).hexdigest(), "recipe": q.get("_recipe")},
                t,
                n_tests=int(t["n_tests"].iloc[0]) if len(t) else 0,
                n_units=len(key),
                n_trials=None,
            )
            self._connections[key] = t
        return self._connections_summary(self._connections[key])

    def _connections_summary(self, t: pd.DataFrame) -> dict:
        hit = t[t["connected"]]
        units = sorted(set(t["pre"]))
        return self._status(t) | {
            "n_units": len(units),
            "n_pairs": len(t) // 2,
            "n_tests": int(t["n_tests"].iloc[0]),
            "n_connected": len(hit),
            "n_connected_close": int(hit["close"].eq(True).sum()),
            "null": t["null"].iloc[0],
            "window_s": list(self.ccg_cfg.synaptic_window_s),
            "alpha": self.response_cfg.alpha,
            "probes": sorted(self.units.loc[units, "probe"].unique()),
        }

    def pair_data(self, q: dict) -> dict:
        """The selected unit's cross-correlogram with a partner, its jitter expectation,
        and the connection tests for the pair when they were run."""
        unit, partner = q.get("unit"), q.get("partner")
        if not partner:
            raise ValueError("choose a partner unit")
        for u in (unit, partner):
            if u not in self.units.index:
                raise ValueError(f"no unit {u!r} in this session")
        if unit == partner:
            raise ValueError("choose a partner other than the selected unit")
        cfg = self.ccg_cfg
        a, b = self.session.spikes[unit], self.session.spikes[partner]
        lags, observed = cross_correlogram(a, b, cfg.window_s, cfg.bin_s)
        expected = jitter_expected_ccg(a, b, cfg.window_s, cfg.bin_s, cfg.jitter_s)
        (close,), (close_why,) = close_pairs(self.units, [(unit, partner)], cfg.close_um)
        tested = self._connections.get(self._pairs_key(q))
        tests = []
        if tested is not None:
            mine = tested[
                ((tested["pre"] == unit) & (tested["post"] == partner))
                | ((tested["pre"] == partner) & (tested["post"] == unit))
            ]
            tests = _records(mine[["pre", "post", "observed", "expected", "p", "q", "connected"]])
        l1, l2 = (x * 1000 for x in cfg.synaptic_window_s)
        caption = (
            f"{unit} → {partner} · cross-correlogram over the whole session · "
            f"{a.size:,} and {b.size:,} spikes · line: expected under {cfg.null} · "
            f"shaded: the synaptic window tested, {l1:g} to {l2:g} ms"
        )
        return {
            "lags": lags,
            "observed": observed,
            "expected": expected,
            "close": close,
            "close_why": close_why,
            "tests": tests,
            "caption": caption,
        }

    def pair_json(self, q: dict) -> dict:
        d = self.pair_data(q)
        return {k: d[k] for k in ("close", "close_why", "tests", "caption")} | {
            "close_um": self.ccg_cfg.close_um
        }

    def pair_png(self, q: dict) -> tuple[bytes, dict]:
        d = self.pair_data(q)
        png = ccg_figure(
            d["lags"],
            d["observed"],
            d["expected"],
            self.ccg_cfg.synaptic_window_s,
            q.get("theme", "light"),
        )
        return png, {"X-Caption": quote(d["caption"])}

    def _locking_key(self, q: dict) -> tuple:
        """Locking is to each trial's own first movement: no event in the key."""
        return q.get("all") == "1", q.get("probe", ""), self._filter(q).key()

    def locking_json(self, q: dict) -> dict:
        """Run (or reuse) the movement-locking test on every shown unit."""
        key = self._locking_key(q)
        if key not in self._locking:
            ids = self._select({"all": q.get("all", "0"), "probe": q.get("probe", "")})
            if not ids:
                raise ValueError("no units to test")
            trials = self._trials(q)[0]
            t = movement_locking(
                self.session.spikes,
                ids,
                trials,
                self.movement_cfg,
                self.response_cfg.alpha,
                self.task,
            )
            self._log(
                "movement_locking",
                "movement-locked",
                self._test_params(q),
                t,
                n_tests=int(t["n_tests"].iloc[0]),
                n_units=len(ids),
                n_trials=int(t["n_trials"].iloc[0]),
            )
            self._locking[key] = t
        return self._locking_summary(self._locking[key])

    def _locking_summary(self, t: pd.DataFrame) -> dict:
        return self._status(t) | {
            "n_tests": int(t["n_tests"].iloc[0]),
            "n_locked": int(t["locked"].sum()),
            "n_trials": int(t["n_trials"].iloc[0]),
            "n_null": int(t["n_null"].iloc[0]),
            "null": t["null"].iloc[0],
            "seed": int(t["seed"].iloc[0]),
            "windows": [list(self.movement_cfg.pre_window), list(self.movement_cfg.post_window)],
            "probes": sorted(self.units.loc[t.index, "probe"].unique()),
        }

    def wheel_png(self, q: dict) -> tuple[bytes, dict]:
        """Wheel speed around the event, on the same trials and bins as the PSTH."""
        d = self.wheel_data(q)
        png = wheel_figure(d["psth"], d["window"], q.get("theme", "light"))
        return png, {"X-Caption": quote(d["caption"])}

    def wheel_data(self, q: dict) -> dict:
        """Wheel speed PSTH (rad/s) on the page's event, window, bins and trials."""
        if "wheel" not in self.session.behaviour:
            why = self.session.available.missing.get("behaviour.wheel", "no wheel in this session")
            raise ValueError(f"No wheel: {why}")
        window, bin_width, _, events = self._params(q)
        p = wheel_speed_psth(self.session.behaviour["wheel"], events, window, bin_width)
        caption = (
            f"Wheel speed · {self._event_label(q['event'])} · n = {p.n_trials} trials"
            + (f", {p.n_excluded} without this event excluded" if p.n_excluded else "")
            + self._trial_note(self._trials(q)[1])
        )
        return {"psth": p, "window": window, "caption": caption}

    def _trial_summary(self, q: dict) -> dict:
        _, sel = self._trials(q)
        return {
            "filter": self._filter(q).to_dict(),
            "n_total": sel.n_total,
            "n_kept": sel.n_kept,
            "n_excluded": sel.n_excluded,
            "excluded": sel.excluded,
        }

    def _selectivity_key(self, q: dict) -> tuple:
        return (*self._test_key(q), q.get("split", ""))

    def selectivity_json(self, q: dict) -> dict:
        """Run (or reuse) the selectivity test for the split condition on the shown units."""
        if not q.get("split"):
            raise ValueError("choose a condition to split by first")
        key = self._selectivity_key(q)
        if key not in self._selectivity:
            ids = self._select({"all": q.get("all", "0"), "probe": q.get("probe", "")})
            if not ids:
                raise ValueError("no units to test")
            # The full table and a mask: block pseudo-sessions span the whole session.
            comparison = self.task.comparisons.get(q["split"])
            circular = comparison is not None and comparison.kind == "circular"
            t = (circular_selectivity if circular else selectivity)(
                self.session.spikes,
                ids,
                self.session.trials,
                q["event"],
                q["split"],
                self.response_cfg,
                self.selectivity_cfg,
                trial_mask=self._trials(q)[1].mask,
                task=self.task,
            )
            n_trials = t["n_trials"].iloc[0] if circular else t["n_a"].iloc[0] + t["n_b"].iloc[0]
            self._log(
                "selectivity",
                f"selective for {self._condition_label(q['split']).lower()} at "
                f"{self._event_label(q['event']).lower()}",
                self._test_params(q, event=q["event"], split=q["split"]),
                t,
                n_tests=int(t["n_tests"].iloc[0]),
                n_units=len(ids),
                n_trials=int(n_trials),
            )
            self._selectivity[key] = t
        return self._selectivity_summary(self._selectivity[key], q)

    def _selectivity_summary(self, t: pd.DataFrame, q: dict) -> dict:
        if "index" in t:  # circular
            return self._status(t) | {
                "kind": "circular",
                "condition": self._condition_label(q["split"]),
                "period": float(t["period_deg"].iloc[0]),
                "n_tests": int(t["n_tests"].iloc[0]),
                "n_selective": int(t["selective"].sum()),
                "n_silent": int(t["index"].isna().sum()),
                "n_trials": int(t["n_trials"].iloc[0]),
                "null": t["null"].iloc[0],
                "n_null": int(t["n_null"].iloc[0]),
                "window": t["window"].iloc[0],
                "seed": int(t["seed"].iloc[0]),
                "probes": sorted(self.units.loc[t.index, "probe"].unique()),
            }
        names = self._level_names(q["split"], self.task.comparisons[q["split"]].levels)
        higher_b = t["selective"] & (t["auroc"] > 0.5)
        return self._status(t) | {
            "condition": self._condition_label(q["split"]),
            "a": names[0],
            "b": names[-1],
            "n_tests": int(t["n_tests"].iloc[0]),
            "n_selective": int(t["selective"].sum()),
            "n_higher_b": int(higher_b.sum()),
            "n_higher_a": int((t["selective"] & ~higher_b).sum()),
            "n_a": int(t["n_a"].iloc[0]),
            "n_b": int(t["n_b"].iloc[0]),
            "null": t["null"].iloc[0],
            "n_null": int(t["n_null"].iloc[0]),
            "window": t["window"].iloc[0],
            "seed": int(t["seed"].iloc[0]),
            "probes": sorted(self.units.loc[t.index, "probe"].unique()),
        }

    def probe_json(self, q: dict) -> dict:
        """One probe's units and region runs along the shank, for the probe strip."""
        probe = self.units.at[q["unit"], "probe"]
        on_probe = self.units.loc[self._select({**q, "node": ""})]
        on_probe = on_probe[on_probe["probe"] == probe]
        regions = self._regions(q)[on_probe.index]
        info = region_info(regions.dropna().unique()) if self.has_regions else {}
        runs = depth_runs(on_probe["depth_um"].to_numpy(), regions.to_numpy())
        for run in runs:
            run["colour"] = info[run["region"]]["colour"]
        units = on_probe[["depth_um", "lateral_um"]].assign(
            region=regions, colour=[info.get(r, {}).get("colour") for r in regions]
        )
        return {
            "probe": probe,
            "runs": runs,
            "units": _records(units.reset_index().rename(columns={"unit_id": "id"})),
        }

    def geometry_json(self, q: dict) -> dict:
        """CCF positions (ap, dv, ml µm) of units and probe tracks, and meshes to draw."""
        if not self.has_positions:
            return {"missing": self.session.available.missing["units.x"]}
        shown = self.units.loc[self._select({**q, "node": ""})]
        ccf = ccf_um(self.session.units.loc[shown.index, ["x", "y", "z"]].to_numpy())
        regions = self._regions(q)[shown.index]
        info = region_info(regions.dropna().unique())
        tracks = []
        for probe, rows in shown.groupby("probe"):
            if len(rows) >= 2:
                a, b = probe_track(ccf[shown.index.get_indexer(rows.index)])
                tracks.append({"probe": probe, "a": a.tolist(), "b": b.tolist()})
        meshes = [{"acronym": a, **info[a]} for a in sorted(info) if a != ROOT]
        return {
            "brain_id": BRAIN_ID,
            "units": [
                {"id": u, "ccf": p.tolist(), "colour": info.get(r, {}).get("colour")}
                for u, p, r in zip(shown.index, ccf, regions)
            ],
            "tracks": tracks,
            "meshes": meshes,
        }

    def unit_data(self, q: dict) -> dict:
        """Every number the selected unit's figure plots, and its caption."""
        window, bin_width, baseline, events = self._params(q)
        spikes = self.session.spikes[q["unit"]]
        region = self._regions(q)[q["unit"]]
        where = "" if pd.isna(region) else f" · {region}"
        trials, sel = self._trials(q)
        note = self._trial_note(sel)
        split = q.get("split", "")
        if split:
            cond = condition(trials, split, self.task)
            colours = self._colours(split, cond.levels, q.get("theme", "light"))
            groups = []
            raster_trials = []
            parts = split_event_times(trials, q["event"], cond, self.task)
            for part, colour in zip(parts, colours):
                if not np.isfinite(part.times).any():
                    continue  # e.g. error trials when aligned to reward feedback
                rows = trials.index[cond.values == part.level][np.isfinite(part.times)]
                raster_trials += self._trial_numbers(rows)
                trial, rel = raster(spikes, part.times, window)
                p = psth(spikes, part.times, window, bin_width, baseline)
                groups.append(TraceGroup(part.name, colour, p, trial, rel))
            n = sum(g.psth.n_trials for g in groups)
            excluded = f", {cond.n_excluded} {cond.excluded} excluded" if cond.n_excluded else ""
            caption = (
                f"{q['unit']}{where} · {self._event_label(q['event'])} · split by "
                f"{self._condition_label(split).lower()} · n = {n} trials{excluded}{note}"
            )
        else:
            p = psth(spikes, events, window, bin_width, baseline)
            trial, rel = raster(spikes, events, window)
            groups = [TraceGroup("all trials", None, p, trial, rel)]
            rows = trials.index[np.isfinite(trial_event_times(trials, q["event"], self.task))]
            raster_trials = self._trial_numbers(rows)
            caption = (
                f"{q['unit']}{where} · {self._event_label(q['event'])} · n = {p.n_trials} trials"
                + (f", {p.n_excluded} without this event excluded" if p.n_excluded else "")
                + note
            )
        assert len(raster_trials) == sum(g.psth.n_trials for g in groups)
        return {
            "groups": groups,
            "window": window,
            "baseline": baseline,
            "caption": caption,
            "raster_trials": raster_trials,
        }

    def _trial_numbers(self, rows: pd.Index) -> list[int]:
        """Trials-table row labels -> trial numbers (0-based rows of the session's table)."""
        return self.session.trials.index.get_indexer(rows).tolist()

    def tuning_data(self, q: dict) -> dict:
        """Mean ± SEM response rate per level of the split condition, with n per level."""
        split = q.get("split", "")
        if not split:
            raise ValueError("choose a condition to split by for a tuning curve")
        # The response window, or the seconds a split's own test declares (Allen's
        # gratings: the whole presentation), so the curve shows what the test tests.
        window = self.response_cfg.response_window
        comparison = self.task.comparisons.get(split)
        if comparison is not None and not isinstance(comparison.window, str):
            window = comparison.window
        trials, sel = self._trials(q)
        spikes = self.session.spikes[q["unit"]]
        curve = tuning_curve(spikes, trials, q["event"], split, window, self.task)
        cond = condition(trials, split, self.task)
        caption = (
            f"{q['unit']} · response rate {window[0] * 1000:g} to {window[1] * 1000:g} ms after "
            f"{self._event_label(q['event']).lower()}, by "
            f"{self._condition_label(split).lower()} · mean ± SEM"
            f"{self._trial_note(sel)}"
        )
        return {"curve": curve, "levels": cond.levels, "caption": caption}

    def tuning_png(self, q: dict) -> tuple[bytes, dict]:
        d = self.tuning_data(q)
        curve, split = d["curve"], q["split"]
        colours = self._colours(split, d["levels"], q.get("theme", "light"), for_axis=True)
        png = tuning_figure(
            list(curve.index),
            curve["mean_hz"].to_numpy(),
            curve["sem_hz"].to_numpy(),
            curve["n"].tolist(),
            colours,
            self.task.conditions[split].type == "ordinal",
            q.get("theme", "light"),
        )
        return png, {"X-Caption": quote(d["caption"])}

    def unit_png(self, q: dict) -> tuple[bytes, dict]:
        d = self.unit_data(q)
        theme = q.get("theme", "light")
        png, box = unit_figure(d["groups"], d["window"], d["baseline"] is not None, theme)
        return png, {
            "X-Caption": quote(d["caption"]),
            "X-Trials": ",".join(map(str, d["raster_trials"])),
            "X-Box": ",".join(f"{v:.4f}" for v in box),
        }

    def _trial_args(self, q: dict) -> dict:
        """The trial view's settings from a request; defaults from configs/trial_view.yaml."""
        cfg = self.trial_cfg
        if q.get("trial") in (None, ""):
            raise ValueError("choose a trial")
        try:
            trial, n_trials = int(q["trial"]), int(q.get("trial_n", 1))
        except ValueError:
            raise ValueError("the trial and the number of trials must be whole numbers") from None
        return {
            "trial": trial,
            "n_trials": n_trials,
            "align": q.get("trial_align", TRIAL_START),
            "pre": float(q.get("trial_pre_s", cfg.pre_pad_s)),
            "post": float(q.get("trial_post_s", cfg.post_pad_s)),
            "traces": tuple(t for t in q.get("trial_traces", "").split(",") if t),
        }

    def trial_data(self, q: dict) -> dict:
        """Every number the single-trial figure plots (analysis.trial_view), its caption,
        and each row's region at the current level."""
        ids = self._select(q)
        if not ids:
            raise ValueError("no units match this filter")
        keep = self._trials(q)[1].mask
        args = self._trial_args(q)
        view = trial_view(
            self.session, self.units.loc[ids], cfg=self.trial_cfg, keep=keep, task=self.task, **args
        )
        regions = [None if pd.isna(r) else r for r in self._regions(q)[view.rows]]
        info = region_info({r for r in regions if r}) if self.has_regions else {}
        included = sorted(set(view.probes))
        shown = view.window.trials
        which = "responsive " if q.get("responsive") == "1" else ""
        caption = (
            f"Single trial, descriptive (no test) · trial {view.window.trial} (0-based) · "
            f"{len(view.rows)} {which}units ({q.get('node') or 'all regions'}) · "
            f"{'probe' if len(included) == 1 else 'probes'} {', '.join(included)} · "
            f"zero at {view.align_label}"
            + (f" · trials {shown[0]} to {shown[-1]}" if len(shown) > 1 else "")
            + f" · pads {args['pre']:g} s before, {args['post']:g} s after"
        )
        return {
            "view": view,
            "regions": regions,  # (n_rows,) region at the current level, None if none
            "region_colours": {r: i["colour"] for r, i in info.items()},
            "caption": caption,
        }

    def trial_json(self, q: dict) -> dict:
        """The trial's header, what is not drawn and why, and where next/previous go."""
        d = self.trial_data(q)
        view, keep = d["view"], self._trials(q)[1].mask
        filtered = q.get("trial_all") != "1"
        k = view.window.trial
        place, n = position(keep, k, filtered)
        return {
            "header": view.header,
            "trials": list(view.window.trials),
            "boundaries": view.boundaries,
            "not_recorded": view.not_recorded,
            "absent_events": view.absent_events,
            "wheel_missing": view.wheel_missing,
            "traces_missing": view.traces_missing,
            "caption": d["caption"],
            "place": place,
            "n": n,
            "filtered": filtered,
            "previous": step(keep, k, -1, filtered),
            "next": step(keep, k, +1, filtered),
            "n_total": len(keep),
        }

    def trials_json(self, q: dict) -> dict:
        """Every trial for the Trials workspace's strip: kept by the page's trial filter
        or not (with the reasons), its reaction time, and its level of a condition
        (q["colour"]; default the split, then the task's outcome; "none" for none), with
        the page's colours for those levels. The page only draws it."""
        trials = self.session.trials
        sel = self._trials(q)[1]
        name = (
            q.get("colour")
            or q.get("split")
            or ("outcome" if "outcome" in self.task.conditions else "")
        )
        if name == "none":
            name = ""
        if name and name not in self.task.conditions:
            raise ValueError(
                f"unknown condition {name!r}; available: {sorted(self.task.conditions)}"
            )
        level, levels = np.full(len(trials), np.nan), []
        if name:
            c = condition(trials, name, self.task)
            level = c.values
            colours = self._colours(name, c.levels, q.get("theme", "light"), for_axis=True)
            levels = [
                {"value": float(v), "name": n, "colour": col}
                for v, n, col in zip(c.levels, c.names, colours)
            ]
        rt = np.full(len(trials), np.nan)
        m = self.task.movement
        if m is not None and self._has_movement_times():
            rt = reaction_times(trials, self.task)
        start = trials["intervals_0"].to_numpy(np.float64)
        rows = [
            {
                "i": i,
                "start_s": float(start[i]),
                "kept": bool(sel.mask[i]),
                "why": [r for r, failed in sel.failed.items() if failed[i]],
                "rt_s": None if np.isnan(rt[i]) else float(rt[i]),
                "level": None if np.isnan(level[i]) else float(level[i]),
            }
            for i in range(len(trials))
        ]
        kept = np.flatnonzero(sel.mask)
        return {
            "n": len(trials),
            "n_kept": sel.n_kept,
            "first_kept": int(kept[0]) if kept.size else None,
            "condition": (
                {"name": name, "label": self.task.conditions[name].label} if name else None
            ),
            "levels": levels,
            "rt_label": (
                None
                if m is None
                else f"{self.task.events[m.movement].label} − {self.task.events[m.stimulus].label}".lower()
            ),
            "trials": rows,
        }

    def _has_movement_times(self) -> bool:
        """Whether this session's trials hold the task's movement and stimulus times."""
        m = self.task.movement
        return m is not None and all(
            c in self.session.trials
            for e in (m.stimulus, m.movement)
            for c in self.task.events[e].columns
        )

    def trial_png(self, q: dict) -> tuple[bytes, dict]:
        d = self.trial_data(q)
        view = d["view"]
        png, box = trial_figure(
            view, d["regions"], d["region_colours"], q.get("unit"), q.get("theme", "light")
        )
        return png, {
            "X-Caption": quote(d["caption"]),
            "X-Rows": ",".join(view.rows),
            "X-Box": ",".join(f"{v:.4f}" for v in box),
        }

    def population_data(self, q: dict) -> dict:
        """Every number the population figure plots, rows in plotted order, and its caption."""
        window, bin_width, baseline, events = self._params(q)
        ids = self._select(q)
        if not ids:
            raise ValueError("no units match this filter")
        # Sort on odd trials and show even ones, so the order is not fitted to what it shows.
        sort_on, show = alternate_halves(events)
        if show.size == 0:
            raise ValueError("needs at least 2 trials with this event: one half sorts, one shows")
        spikes = self.session.spikes
        order = peak_order(population_psth(spikes, ids, sort_on, window, bin_width, baseline))
        pop = population_psth(spikes, ids, show, window, bin_width, baseline)[order]
        mean, sem = selection_average(pop)
        edges = bin_edges(window, bin_width)
        which = "responsive " if q.get("responsive") == "1" else ""
        rows = [ids[i] for i in order]
        probes = self.units.loc[rows, "probe"].tolist()
        included = sorted(set(probes))
        caption = (
            f"{len(ids)} {which}units ({q.get('node') or 'all regions'}) · "
            f"{'probe' if len(included) == 1 else 'probes'} {', '.join(included)} · "
            f"{self._event_label(q['event'])} · sorted by peak time on odd trials "
            f"(n = {sort_on.size}), "
            f"showing even trials (n = {show.size}){self._trial_note(self._trials(q)[1])}"
        )
        return {
            "units": rows,
            "probes": probes,
            "rates_hz": pop,
            "scaled": scale_rows_for_display(pop),
            "bin_centers": (edges[:-1] + edges[1:]) / 2,
            "mean": mean,
            "sem": sem,
            "window": window,
            "baseline": baseline,
            "n_sort_trials": int(sort_on.size),
            "n_show_trials": int(show.size),
            "caption": caption,
        }

    def population_png(self, q: dict) -> tuple[bytes, dict]:
        d = self.population_data(q)
        theme = q.get("theme", "light")
        png, box = population_figure(
            d["scaled"],
            d["bin_centers"],
            d["mean"],
            d["sem"],
            d["window"],
            theme,
            row_groups=d["probes"],
            group_colours=probe_colours(self.probes, theme),
        )
        headers = {
            "X-Caption": quote(d["caption"]),
            "X-Rows": ",".join(d["units"]),
            "X-Box": ",".join(f"{v:.4f}" for v in box),
        }
        return png, headers

    def _cluster_folder(self, probe: str) -> Path | str:
        """The probe's IBL spike-sorting folder, or why there is none."""
        if self.source is None or self.source.kind != "ibl":
            return "not an IBL session"
        if self.ibl_alf is None:
            return (
                "this session's spike sorting is not in the local ONE cache "
                "(IBL's waveforms and per-criterion metrics are downloaded with ONE)"
            )
        folder = ibl_sorting_folder(self.ibl_alf, probe)
        return folder or f"the ONE cache has no {probe} spike sorting for this session"

    def _criteria(self, unit: str) -> tuple[dict | None, str]:
        """IBL's label criteria for one unit, or why they are not shown."""
        probe = self.units.at[unit, "probe"]
        if probe not in self._ibl_criteria:
            folder = self._cluster_folder(probe)
            try:
                self._ibl_criteria[probe] = (
                    folder if isinstance(folder, str) else ibl_criteria(folder)
                )
            except ValueError as e:
                self._ibl_criteria[probe] = str(e)
        table = self._ibl_criteria[probe]
        if isinstance(table, str):
            return None, table
        cluster = int(self.session.units.at[unit, "cluster_id"])
        if cluster not in table.index:
            return None, f"cluster {cluster} is not in IBL's metrics table"
        row = table.loc[cluster]
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}, ""

    def _waveform(self, unit: str):
        """(Waveform, "") or (None, why there is none)."""
        if self.source is not None and self.source.kind == "nwb":
            return None, "waveforms aren't read from NWB files"
        cluster = int(self.session.units.at[unit, "cluster_id"])
        try:
            if self.source is not None and self.source.kind == "phy":
                return phy_waveform(Path(self.source.folder), cluster), ""
            folder = self._cluster_folder(self.units.at[unit, "probe"])
            if isinstance(folder, str):
                return None, folder
            return ibl_waveform(folder, cluster), ""
        except ValueError as e:
            return None, str(e)

    def quality_data(self, q: dict) -> dict:
        """The unit quality panel (analysis.unit_quality) and its caption."""
        unit = q["unit"]
        if unit not in self.units.index:
            raise ValueError(f"no unit {unit!r} in this session")
        phy = isinstance(self.qc, PhyUnitQC | SpikeQC)  # rules that run the refractory test
        contamination, alpha = (
            (self.qc.refractory_contamination, self.qc.refractory_alpha)
            if phy
            else (IBL_RP_CONTAMINATION, IBL_RP_ALPHA)
        )
        probe = self.units.at[unit, "probe"]
        waveform, waveform_missing = self._waveform(unit)
        if self.source is not None and self.source.kind == "nwb":
            criteria, ibl_missing = None, "only for IBL sessions (from IBL's own metrics)"
        elif phy:
            criteria, ibl_missing = (
                None,
                "not for Phy folders (IBL's amplitude criteria need volts)",
            )
        else:
            criteria, ibl_missing = self._criteria(unit)
        uq = unit_quality(
            self.session.spikes,
            unit,
            self.units.index[self.units["probe"] == probe],
            self.units.loc[unit],
            self.quality_cfg,
            contamination,
            alpha,
            waveform,
            waveform_missing,
            criteria,
            ibl_missing,
        )
        span = uq.span_s[1] - uq.span_s[0]
        caption = (
            f"{unit} · quality, descriptive (the QC verdicts already computed; no new test) · "
            f"{uq.n_spikes:,} spikes over {span / 60:.1f} min of the probe's recording"
        )
        return {"quality": uq, "caption": caption, "rp_used_by_qc": phy}

    def quality_json(self, q: dict) -> dict:
        d = self.quality_data(q)
        uq, rd = d["quality"], d["quality"].refractory
        span = uq.span_s[1] - uq.span_s[0]
        return {
            "unit": uq.unit,
            "caption": d["caption"],
            "n_spikes": uq.n_spikes,
            "rate_hz": uq.n_spikes / span if span > 0 else None,
            "presence_ratio": uq.presence[0],
            "presence_window_s": self.quality_cfg.presence_window_s,
            "qc": {
                "passed": uq.qc_passed,
                "reasons": uq.qc_reasons,
                "config": self._qc_info()["config"],
            },
            "spike_qc": {
                "passed": bool(self.units.at[uq.unit, "spike_qc_passed"]),
                "reasons": [
                    r for r in str(self.units.at[uq.unit, "spike_qc_reason"]).split("; ") if r
                ],
                "presence_ratio": _number(self.units.at[uq.unit, "presence_ratio"]),
                "is_the_qc": self._qc_info()["rule"] == "spike times",
                **{k: v for k, v in self._qc_info().items() if k.startswith("spike_")},
            },
            "refractory": {
                "passed": rd.passed,
                "first_pass_rp_ms": (
                    None if rd.first_pass_rp_s is None else rd.first_pass_rp_s * 1000
                ),
                "contamination": rd.contamination,
                "confidence": 1 - rd.alpha,
                "used_by_qc": d["rp_used_by_qc"],
                "why": rd.why,
            },
            "ibl_criteria": uq.ibl_criteria,
            "ibl_missing": uq.ibl_missing,
            "waveform": (
                None
                if uq.waveform is None
                else {"unit": uq.waveform.unit, "source": uq.waveform.source}
            ),
            "waveform_missing": uq.waveform_missing,
        }

    def quality_png(self, q: dict) -> tuple[bytes, dict]:
        d = self.quality_data(q)
        return quality_figure(d["quality"], q.get("theme", "light")), {
            "X-Caption": quote(d["caption"])
        }

    def trajectory_data(self, q: dict) -> dict:
        """Cross-validated population trajectories of the shown units (descriptive)."""
        ids = self._select(q)
        if len(ids) < 2:
            raise ValueError("trajectories need at least 2 shown units")
        trials, sel = self._trials(q)
        window, bin_width = (float(q["t0"]), float(q["t1"])), float(q["bin"])
        theme, split = q.get("theme", "light"), q.get("split", "")
        if split:
            cond = condition(trials, split, self.task)
            split_parts = split_event_times(trials, q["event"], cond, self.task)
            parts = [(p.name, p.times) for p in split_parts]
            colour_of = dict(zip(cond.names, self._colours(split, cond.levels, theme)))
            how = f"split by {self._condition_label(split).lower()}"
            dropped = f" · {cond.n_excluded} {cond.excluded} excluded" if cond.n_excluded else ""
        else:
            parts = [("all trials", trial_event_times(trials, q["event"], self.task))]
            colour_of = {"all trials": THEMES[theme]["series"]}
            how, dropped = "all trials", ""
        r = trajectories(self.session.spikes, ids, parts, window, bin_width, self.traj_cfg)
        shown = ", ".join(f"{a} {v:.0%}" for a, v in zip(r.axis_names, r.explained_held_out))
        not_shown = "".join(f" · {name} not shown: {why}" for name, why in r.excluded.items())
        caption = (
            f"{len(ids)} units ({q.get('node') or 'all regions'}) · "
            f"{self._event_label(q['event'])} · "
            f"{how} · principal components of condition-averaged rates, fit on "
            f"{sum(r.n_fit)} trials (1st, 3rd, ...) and shown on the other {sum(r.n_show)} · "
            "descriptive (no test) · "
            f"variance of the shown trials on {shown}{not_shown}{dropped}"
            f"{self._trial_note(sel)}"
        )
        return {"result": r, "colours": [colour_of[n] for n in r.names], "caption": caption}

    def trajectory_json(self, q: dict) -> dict:
        d = self.trajectory_data(q)
        r = d["result"]
        return {
            "caption": d["caption"],
            "names": r.names,
            "n_fit": r.n_fit,
            "n_show": r.n_show,
            "excluded": r.excluded,
            "axis_names": r.axis_names,
            "explained_fit": r.explained_fit.tolist(),
            "explained_held_out": r.explained_held_out.tolist(),
        }

    def trajectory_png(self, q: dict) -> tuple[bytes, dict]:
        d = self.trajectory_data(q)
        dims = 3 if q.get("traj_dims") == "3" else 2
        png = trajectory_figure(d["result"], d["colours"], dims, q.get("theme", "light"))
        return png, {"X-Caption": quote(d["caption"])}

    def mesh(self, structure_id: int) -> bytes:
        return mesh_path(structure_id, self.atlas_root).read_bytes()


def _static(path: str) -> Path | None:
    """A file under static/, or None; never a path outside it."""
    target = (STATIC / path.removeprefix("/static/")).resolve()
    return target if target.is_file() and STATIC.resolve() in target.parents else None


class App:
    """The running server: the homepage catalog, and at most one open session."""

    def __init__(
        self,
        data: DataConfig,
        studio: Studio | None = None,
        manifest: Manifest | None = None,
        loader=load_source,
        runs_dir: Path = RUNS,
    ):
        self.data = data
        self.studio = studio
        self.runs_dir = Path(runs_dir)  # region-summary runs are read from here (S5)
        self.catalog_cfg = load_catalog_config()
        self._manifest = manifest
        self._loader = loader

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:

            def build() -> Manifest:
                return build_manifest(self.data.bwm_ephys_root, self.data.bwm_behavior_root)

            self._manifest = load_catalog(self.data.derived_root, build)
        return self._manifest

    def require(self) -> Studio:
        if self.studio is None:
            raise ValueError("No session is open: choose one on the homepage")
        return self.studio

    def _cached(self, eid: str) -> bool:
        return SessionCache(self.data.cache_root).path(key_parts(eid, "bwm")).exists()

    def summaries(self, q: dict) -> list[dict]:
        """Finished region-summary runs (cli.summarise), newest first."""
        return list_summaries(self.runs_dir)

    def summary_json(self, q: dict) -> dict:
        """One summary run: its caption, regions with a claim, regions refused."""
        m, regions, _ = read_summary(self.runs_dir, q.get("run", ""))
        cfg = m["summary_config"]
        failed = m["sessions_failed"]
        refused = regions[regions["refused"] != ""]
        left_out = f" ({len(failed)} left out: {'; '.join(sorted(set(failed.values())))})"
        the_set = m["set"]
        if the_set.get("kind") == "nwb":  # files named on the command line (step 8c)
            of = (
                f"NWB files '{the_set['name']}' ({the_set['layout_label']}, task {the_set['task']})"
            )
        else:
            of = f"sessions of set '{the_set['name']}'"
        caption = (
            f"{m['label_text']} · {m['level']} regions · {len(m['sessions_used'])} of "
            f"{len(the_set['eids'])} {of}"
            f"{left_out if failed else ''} · {m['n_units']} units · per region: labelled "
            "units summed over sessions against the exact null with region labels permuted "
            "within each session, two-sided, Benjamini–Hochberg across "
            f"{m['n_tested']} region{'' if m['n_tested'] == 1 else 's'} with at least "
            f"{cfg['min_sessions']} sessions, "
            f"α = {cfg['alpha']} · {len(refused)} region{'' if len(refused) == 1 else 's'} "
            f"refused (too few sessions) · run {m['run_id']}"
        )
        status = m.get("plan_status", {"status": "exploratory", "why": "run before plans existed"})
        caption += (
            f" · confirmatory, the planned run of plan '{status['plan']}'"
            if status["status"] == "confirmatory"
            else f" · exploratory ({status['why']})"
        )

        def rows(table):
            out = table.reset_index()[
                ["region", "n_sessions", "n_units", "observed", "expected", "q", "direction"]
            ]
            return _records(out.assign(q=out["q"].astype(float)))

        claims = regions[regions["claim"]].sort_values("q")
        return {
            "caption": caption,
            "claims": rows(claims),
            "refused": rows(refused),
            "n_tested": m["n_tested"],
            "min_sessions": cfg["min_sessions"],
        }

    def summary_png(self, q: dict) -> tuple[bytes, dict]:
        """A summary run's flatmap or per-session spread, in the page's theme."""
        _, regions, sessions = read_summary(self.runs_dir, q.get("run", ""))
        theme = q.get("theme", "light")
        if q.get("kind") == "flatmap":
            return region_flatmap(regions, theme), {}
        if q.get("kind") == "spread":
            return region_spread(regions, sessions, theme), {}
        raise ValueError("choose flatmap or spread")

    def home_json(self, q: dict) -> dict:
        """Filter options, live counts, the region tree and the matching sessions."""
        raw = json.loads(q.get("f") or "{}")
        known = set(SessionFilter.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown session filters {unknown}")
        if raw.get("region") and "min_region_units" not in raw:
            raw["min_region_units"] = self.catalog_cfg.min_region_units
        f = SessionFilter(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
        m = self.manifest
        matching = filter_sessions(m, f)
        # The tree offers regions among sessions matching every filter but the region.
        others = filter_sessions(m, dataclasses.replace(f, region=None, min_region_units=0))
        column = q.get("sort") or "date"
        if column not in _SORTABLE:
            raise ValueError(f"cannot sort by {column!r}")
        matching = matching.sort_values(
            [column, "eid"], ascending=q.get("desc") != "1", kind="stable"
        )
        rows = matching.assign(
            n_regions=matching["regions"].map(len),
            cached=matching["eid"].map(self._cached),
        )
        keep = [c for c in (*_SORTABLE, "eid", "regions", "cached", "region_units") if c in rows]
        s = m.sessions
        return {
            "summary": summary(m, matching),
            "sessions": _records(rows[keep]),
            "tree": region_counts(m, others["eid"], q.get("level") or DEFAULT_LEVEL),
            "options": {
                "labs": sorted(s["lab"].unique()),
                "subjects": sorted(s["subject"].unique()),
                "dates": [s["date"].min(), s["date"].max()],
                "n_probes": sorted(int(n) for n in s["n_probes"].unique()),
                "modalities": sorted({x for mods in s["modalities"] for x in mods}),
                "min_region_units": self.catalog_cfg.min_region_units,
                "sync_tolerance_ms": load_sync_config().tolerance_ms,
                "trial_levels": _BWM_TRIAL_LEVELS,
                "tasks": list_tasks(),
                "layouts": list_layouts(),
                "default_task": DEFAULT_TASK,
                "default_trial_filter": TrialFilter.from_dict(
                    self.catalog_cfg.default_trial_filter
                ).to_dict(),
            },
            "notes": {
                "units": "Units are the release's good units, not Studio's QC count "
                "(d23a44ef: the release lists 398, Studio's QC passes 390).",
                "manifest": m.provenance,
            },
            "open": None if self.studio is None else self.studio.session.eid,
            "probes": self._probes(matching),
        }

    def _probes(self, matching: pd.DataFrame) -> dict:
        """The 3D overview: every matching probe as a CCF line, coloured by lab."""
        lines = probe_lines(self.manifest, matching["eid"])
        by_size = self.manifest.sessions["lab"].value_counts()
        labs = sorted(by_size.index, key=lambda lab: (-by_size[lab], lab))
        colours = {theme: lab_colours(labs, theme) for theme in THEMES}
        n_slots = len(THEMES["light"]["categorical"])
        return {
            "brain_id": BRAIN_ID,
            "lines": _records(lines),
            "labs": [
                {"lab": lab, "n_sessions": int(by_size[lab]), "other": i >= n_slots}
                for i, lab in enumerate(labs)
            ],
            "colours": colours,
        }

    @property
    def phy_root(self) -> Path:
        root = Path(self.catalog_cfg.phy_root).expanduser()
        return root if root.is_absolute() else self.data.data_root / root

    @property
    def nwb_root(self) -> Path:
        root = Path(self.catalog_cfg.nwb_root).expanduser()
        return root if root.is_absolute() else self.data.data_root / root

    def nwb_complete(self, q: dict) -> dict:
        return {
            "root": str(self.nwb_root),
            "choices": complete_nwb_path(self.nwb_root, q.get("prefix", "")),
        }

    def phy_complete(self, q: dict) -> dict:
        return {
            "root": str(self.phy_root),
            "choices": complete_phy_path(self.phy_root, q.get("prefix", "")),
        }

    def projects(self, q: dict) -> dict:
        return {"projects": recent_projects(self.data.data_root / "projects")}

    @property
    def sets_dir(self) -> Path:
        return self.data.data_root / "sets"

    def sets(self, q: dict) -> dict:
        return {"sets": list_sets(self.sets_dir), "folder": str(self.sets_dir)}

    def save_set(self, body: dict) -> dict:
        path = save_set(
            self.sets_dir,
            body.get("name", ""),
            body.get("eids", []),
            self.manifest.provenance["manifest_version"],
            body.get("session_filter", {}),
            body.get("trial_filter", {}),
        )
        return {"saved": path.name, "sets": list_sets(self.sets_dir)}

    def open_set(self, body: dict) -> dict:
        found, warnings = read_set(set_path(self.sets_dir, body.get("name", "")), MANIFEST_VERSION)
        known = set(self.manifest.sessions["eid"])
        missing = [e for e in found["eids"] if e not in known]
        if missing:
            warnings.append(f"{len(missing)} of its sessions are not in this manifest")
        return {"set": found, "warnings": warnings}

    def open(self, body: dict) -> dict:
        """Load a session and make it the open one. Everything cached is dropped."""
        kind = body.get("kind")
        if kind == "project":
            path = project_path(self.data.data_root / "projects", str(body.get("name", "")))
            project, session, qc, warnings = open_project(path)
            source = Source.from_record(project["source"])
            atlas = self.data.data_root / "atlas"
            self.studio = Studio(
                session,
                qc,
                atlas,
                source,
                path,
                project["view"],
                warnings,
                self.ibl_alf(source),
                analysis_log=project.get("analysis_log", []),
            )
            return {"eid": session.eid, "url": "/session", "warnings": warnings}
        recording = None
        if kind == "phy":  # a folder with a recording.yaml opens as that recording (step 14a)
            recording = resolve_recording(self.phy_root, str(body.get("path", "")))
        if recording is not None:
            task = str(body.get("task") or DEFAULT_TASK)
            source = Source(kind="recording", file=str(recording), task=task)
            name = recording.parent.name
        elif kind == "phy":
            folder, events = resolve_phy_folder(self.phy_root, str(body.get("path", "")))
            task = str(body.get("task") or DEFAULT_TASK)
            sync = {}
            if body.get("sync_probe") or body.get("sync_events"):  # step 13a: both, or neither
                if not (body.get("sync_probe") and body.get("sync_events")):
                    raise ValueError(
                        "give the sync pulses on both clocks: the probe's and the events'"
                    )
                sync = {
                    role: str(resolve_sync_file(self.phy_root, str(body[role])))
                    for role in ("sync_probe", "sync_events")
                }
            if body.get("locations"):  # step 13b
                sync["locations"] = str(
                    resolve_locations_file(self.phy_root, str(body["locations"]))
                )
            source = Source(kind="phy", folder=str(folder), events=str(events), task=task, **sync)
            name = folder.name
        elif kind == "nwb":
            file = resolve_nwb_file(self.nwb_root, str(body.get("path", "")))
            layout = body.get("layout") or None
            task = str(body.get("task") or load_layout(layout).task or DEFAULT_TASK)
            source = Source(kind="nwb", file=str(file), layout=layout, task=task)
            name = file.stem
        elif kind == "ibl" and body.get("eid"):
            source = Source(kind="ibl", eid=str(body["eid"]), backend="bwm")
            name = source.eid[:8]
        else:
            raise ValueError(
                "open needs {kind: 'ibl', eid}, {kind: 'phy', path}, {kind: 'nwb', path} "
                "or {kind: 'project', name}"
            )
        task = load_task(source.task)  # refuses an unknown or malformed definition
        session, qc = self._loader(source)
        task.require(session.trials)  # refuses a table without the required columns
        if "trials" in body:
            chosen = TrialFilter.from_dict(body["trials"], task)
            apply_trial_filter(session.trials, chosen)  # refuses a filter it can't apply
        else:
            chosen = default_trials(session.trials, self.catalog_cfg, task)
        view = {**DEFAULT_VIEW, "trials": chosen.to_dict()}
        path = _new_project_path(self.data.data_root / "projects", name)
        atlas = self.data.data_root / "atlas"
        self.studio = Studio(session, qc, atlas, source, path, view, [], self.ibl_alf(source), task)
        return {"eid": session.eid, "url": "/session"}

    def ibl_alf(self, source: Source) -> Path | None:
        """An IBL session's alf folder in the local ONE cache, found from the manifest."""
        if source.kind != "ibl":
            return None
        return ibl_session_folder(self.data.one_cache_root, self.manifest.sessions, source.eid)


def default_trials(trials: pd.DataFrame, cfg, task: TaskDefinition | None = None) -> TrialFilter:
    """configs/catalog.yaml's default trial filter (IBL's), minus what this session can't
    support. Another task starts with the filters its definition turns on (e.g. Allen's
    invalid data), where the table supports them; else with every trial."""
    task = task if task is not None else load_task()
    if task.name != DEFAULT_TASK:
        offered = available_trial_filters(trials, task)
        on = {
            name: True
            for name, f in task.trial_filters.items()
            if f.default and offered[name]["available"]
        }
        return TrialFilter(task, **on)
    offered = available_trial_filters(trials, task)
    default = cfg.default_trial_filter
    chosen = {k: v for k, v in default.items() if offered[k]["available"]}
    return TrialFilter.from_dict(chosen, task)


# Session-list columns the homepage can sort by.
_SORTABLE = (
    "lab",
    "subject",
    "date",
    "n_probes",
    "n_good_units",
    "n_trials",
    "n_included_trials",
    "n_regions",
)
# Every BWM session runs the same task, so the homepage offers these trial filters.
_BWM_TRIAL_LEVELS = {
    "contrasts": [0.0, 0.0625, 0.125, 0.25, 1.0],
    "blocks": [0.2, 0.5, 0.8],
    "outcomes": [-1.0, 1.0],
}
# path -> Studio method answering it; these need an open session.
_SESSION_ROUTES = {
    "/api/session": ("application/json", "session_json"),
    "/api/units": ("application/json", "units_json"),
    "/api/probe": ("application/json", "probe_json"),
    "/api/geometry": ("application/json", "geometry_json"),
    "/api/test": ("application/json", "test_json"),
    "/api/selectivity": ("application/json", "selectivity_json"),
    "/api/locking": ("application/json", "locking_json"),
    "/api/wheel.png": ("image/png", "wheel_png"),
    "/api/trial": ("application/json", "trial_json"),
    "/api/trials": ("application/json", "trials_json"),
    "/api/trial.png": ("image/png", "trial_png"),
    "/api/quality": ("application/json", "quality_json"),
    "/api/connections": ("application/json", "connections_json"),
    "/api/decode/status": ("application/json", "decode_status"),
    "/api/pair": ("application/json", "pair_json"),
    "/api/pair.png": ("image/png", "pair_png"),
    "/api/trajectories": ("application/json", "trajectory_json"),
    "/api/recipes": ("application/json", "recipes_json"),
    "/api/recipe": ("application/json", "recipe_run"),
    "/api/log": ("application/json", "log_json"),
    "/api/trajectories.png": ("image/png", "trajectory_png"),
    "/api/quality.png": ("image/png", "quality_png"),
    "/api/tuning.png": ("image/png", "tuning_png"),
    "/api/unit.png": ("image/png", "unit_png"),
    "/api/population.png": ("image/png", "population_png"),
}
# path -> App method answering it; these work with no session open.
_APP_ROUTES = {
    "/api/phy/complete": "phy_complete",
    "/api/nwb/complete": "nwb_complete",
    "/api/projects": "projects",
    "/api/sets": "sets",
    "/api/summaries": "summaries",
    "/api/summary": "summary_json",
}
# path -> App method returning (png, headers); no session needed.
_APP_IMAGES = {"/api/summary.png": "summary_png"}
_JSON_METHODS = {name for ctype, name in _SESSION_ROUTES.values() if ctype == "application/json"}


def make_handler(app: "App | Studio", freshness: Freshness | None = None):
    """The request handler. freshness: whether the code changed since the server
    started (studio.freshness); made now when not given."""
    if isinstance(app, Studio):  # a single session, as before the homepage
        app = App(load_data_config(), studio=app)
    fresh = freshness if freshness is not None else Freshness()

    def page(name: str) -> bytes:
        return (HERE / name).read_bytes()

    posts = {
        "/api/open": app.open,
        "/api/sets/save": app.save_set,
        "/api/sets/open": app.open_set,
        "/api/project": lambda view: app.require().save(view),
        "/api/export": lambda view: app.require().export(view),
        # The BWM manifest only for a BWM session: other sessions get a one-session catalogue.
        "/api/decode": lambda body: app.require().decode_start(
            body, app.manifest if app.require()._bwm() else None
        ),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/session") and fresh.stale():
                    return self._send(503, "text/html; charset=utf-8", fresh.restart_page())
                if url.path == "/":
                    return self._send(200, "text/html; charset=utf-8", page("home.html"))
                if url.path == "/session":
                    if app.studio is None:  # nothing open: back to the homepage
                        return self._redirect("/")
                    return self._send(200, "text/html; charset=utf-8", page("index.html"))
                if url.path == "/api/home":
                    body = json.dumps(app.home_json(q)).encode()
                    return self._send(200, "application/json", body)
                if url.path in _SESSION_ROUTES:
                    ctype, name = _SESSION_ROUTES[url.path]
                    result = getattr(app.require(), name)(q)
                    if name in _JSON_METHODS:
                        return self._send(200, ctype, json.dumps(result).encode())
                    body, headers = result
                    return self._send(200, ctype, body, headers)
                if match := _MESH.match(url.path):  # the homepage's 3D view needs no session
                    mesh = mesh_path(int(match.group(1)), app.data.data_root / "atlas")
                    return self._send(200, "text/plain", mesh.read_bytes())
                if url.path in _APP_ROUTES:
                    body = json.dumps(getattr(app, _APP_ROUTES[url.path])(q)).encode()
                    return self._send(200, "application/json", body)
                if url.path in _APP_IMAGES:
                    body, headers = getattr(app, _APP_IMAGES[url.path])(q)
                    return self._send(200, "image/png", body, headers)
                if url.path.startswith("/static/") and (path := _static(url.path)):
                    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    return self._send(200, ctype, path.read_bytes())
                self._send(404, "text/plain", b"not found")
            except (ValueError, KeyError, OSError) as e:
                # Plain-language refusal for the page to show, never a traceback.
                self._send(400, "text/plain; charset=utf-8", f"Cannot show this: {e}".encode())

        def do_POST(self):
            # Only this page may post: JSON bodies force a CORS preflight, which this
            # server never answers, and the Origin must be this server's own.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self._send(403, "text/plain", b"cross-origin request refused")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._send(415, "text/plain", b"expected application/json")
            length = int(self.headers.get("Content-Length", 0))
            if length > _MAX_BODY:
                return self._send(413, "text/plain", b"request too large")
            fn = posts.get(urlparse(self.path).path)
            if fn is None:
                return self._send(404, "text/plain", b"not found")
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, "application/json", json.dumps(fn(body)).encode())
            except (ValueError, KeyError, OSError) as e:
                self._send(400, "text/plain; charset=utf-8", f"Cannot do this: {e}".encode())

        def _redirect(self, where: str):
            self.send_response(302)
            self.send_header("Location", where)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send(self, code, ctype, body, headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return Handler


def _new_project_path(root: Path, name: str) -> Path:
    """root/<name>.unitwave.json, or <name>-2, -3, ... so no project is overwritten."""
    path, n = root / f"{name}{SUFFIX}", 2
    while path.exists():
        path, n = root / f"{name}-{n}{SUFFIX}", n + 1
    return path


def build_app(argv: list[str]) -> tuple[App, str]:
    """(App, the path to open first). No data flags: the homepage. Otherwise a session."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eid", help="an IBL session; skips the homepage")
    ap.add_argument("--backend", default="bwm")
    ap.add_argument("--phy", help="a Kilosort/Phy output folder (one probe); needs --events")
    ap.add_argument("--events", help="CSV of trial events, seconds on the probe's clock")
    ap.add_argument("--sync-probe", help="with --phy: sync pulses on the probe's clock")
    ap.add_argument("--sync-events", help="with --phy: the same pulses on the events' clock")
    ap.add_argument("--locations", help="with --phy: channel_locations.json or a .csv")
    ap.add_argument("--recording", help="a recording.yaml: several probes, events, behaviour")
    ap.add_argument("--nwb", help="an NWB file with sorted units and a trials table")
    ap.add_argument(
        "--layout",
        default=None,
        help="with --nwb: the NWB layout its specifics are declared in, a built-in name "
        "(configs/nwb/) or a YAML file; none reads it generically",
    )
    ap.add_argument(
        "--task",
        default=None,
        help="with --phy or --nwb: the task definition the trials are read with, a built-in "
        "name (configs/tasks/) or a YAML file (default: IBL's, or the NWB layout's)",
    )
    ap.add_argument(
        "--project",
        type=Path,
        help=f"a *{SUFFIX} file: alone, opens it; with a data source, saves a new one there",
    )
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if bool(args.phy) != bool(args.events):
        ap.error("--phy and --events go together")
    if sum(bool(x) for x in (args.phy, args.eid, args.nwb)) > 1:
        ap.error("choose one data source: --eid, --phy or --nwb")
    data = load_data_config()
    if not (args.project or args.phy or args.eid or args.nwb or args.recording):
        return App(data), "/"
    view, warnings = None, []
    if args.project and not (args.phy or args.eid or args.nwb or args.recording):
        if not args.project.exists():
            ap.error(f"{args.project} does not exist; give a data source to start a new project")
        project, session, qc, warnings = open_project(args.project)
        source = Source.from_record(project["source"])
        view, path = project["view"], args.project
        log = project.get("analysis_log", [])
    else:
        if args.recording:
            source = Source(
                kind="recording",
                file=str(Path(args.recording).resolve()),
                task=args.task or DEFAULT_TASK,
            )
        elif args.phy:
            sync = {
                role: str(Path(path).resolve())
                for role, path in (
                    ("sync_probe", args.sync_probe),
                    ("sync_events", args.sync_events),
                    ("locations", args.locations),
                )
                if path
            }
            source = Source(
                kind="phy",
                folder=str(Path(args.phy).resolve()),
                events=str(Path(args.events).resolve()),
                task=args.task or DEFAULT_TASK,
                **sync,
            )
        elif args.nwb:
            source = Source(
                kind="nwb",
                file=str(Path(args.nwb).resolve()),
                layout=args.layout,
                task=args.task or load_layout(args.layout).task or DEFAULT_TASK,
            )
        else:
            source = Source(kind="ibl", eid=args.eid, backend=args.backend)
        if args.project and args.project.exists():
            ap.error(f"{args.project} exists; open it with --project alone, or choose a new name")
        name = {
            "phy": lambda: Path(source.folder).name,
            "nwb": lambda: Path(source.file).stem,
            "recording": lambda: Path(source.file).parent.name,
        }
        name = name.get(source.kind, lambda: source.eid[:8])()
        path = args.project or _new_project_path(data.data_root / "projects", name)
        log = []
        session, qc = load_source(source)
        # The same default as opening from the homepage; a project keeps its own.
        task = load_task(source.task)
        task.require(session.trials)
        trials = default_trials(session.trials, load_catalog_config(), task)
        view = {**DEFAULT_VIEW, "trials": trials.to_dict()}
    app = App(data)
    atlas = data.data_root / "atlas"
    app.studio = Studio(
        session, qc, atlas, source, path, view, warnings, app.ibl_alf(source), analysis_log=log
    )
    return app, "/session"


def main() -> None:
    import sys

    app, first = build_app(sys.argv[1:])
    port = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--port=")), None)
    if port is None and "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    port = port or 8765
    opened = "" if app.studio is None else f"  ({app.studio.session.eid})"
    print(f"{APP_NAME}: http://127.0.0.1:{port}{first}{opened}", flush=True)
    if app.studio is not None:
        print(f"Project file: {app.studio.project_path}", flush=True)
        for warning in app.studio.warnings:
            print(f"Warning: {warning}", flush=True)
    # Threads, so a long responsiveness test doesn't hold up the plots.
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
