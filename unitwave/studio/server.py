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
import json
import mimetypes
import re
import sys
import threading
import time
import traceback
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
    CONDITIONS,
    TrialFilter,
    TrialSelection,
    apply_trial_filter,
    available_conditions,
    available_trial_filters,
    condition,
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
from unitwave.analysis.events import EVENTS, available_events, event_times, trial_event_times
from unitwave.analysis.movement import (
    load_movement_config,
    movement_free,
    movement_locking,
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
from unitwave.analysis.responsiveness import load_response_config, responsiveness
from unitwave.analysis.trajectories import load_trajectory_config, trajectories
from unitwave.analysis.trial_view import (
    EVENT_COLUMNS,
    NUMBERING,
    TRACES,
    TRIAL_START,
    alignment_label,
    load_trial_view_config,
    position,
    step,
    trial_view,
)
from unitwave.analysis.tuning import (
    COMPARISONS,
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
from unitwave.qc.phy import PhyUnitQC
from unitwave.studio import APP_NAME
from unitwave.studio.entry import (
    complete_phy_path,
    project_path,
    recent_projects,
    resolve_phy_folder,
)
from unitwave.studio.export import export_view
from unitwave.studio.project import (
    DEFAULT_VIEW,
    SUFFIX,
    Source,
    load_source,
    make_project,
    open_project,
    save_project,
    saving_path,
)
from unitwave.studio.sets import list_sets, read_set, save_set, set_path
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
    trajectory_figure,
    trial_figure,
    tuning_figure,
    unit_figure,
    wheel_figure,
)

HERE = Path(__file__).parent
STATIC = HERE / "static"
DEFAULT_LEVEL = "Beryl"
BRAIN_ID = 997  # Allen structure id of the whole brain ("root")
DEFAULT_EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
_MESH = re.compile(r"^/mesh/(\d+)\.obj$")
_MAX_BODY = 1 << 20
RUNS = Path(__file__).resolve().parents[2] / "runs"


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
    ):
        self.session = session
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
        self.units = unit_table(session, qc)
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

    def _test_key(self, q: dict) -> tuple:
        """A result belongs to one event, unit set, probe and trial filter."""
        return q.get("event", ""), q.get("all") == "1", q.get("probe", ""), self._filter(q).key()

    def _response_key(self, q: dict) -> tuple:
        """Responsiveness also depends on whether only movement-free trials were used."""
        return (*self._test_key(q), q.get("movement_free") == "1")

    def _filter(self, q: dict) -> TrialFilter:
        """The request's trial filter; no `tf` means every trial."""
        return TrialFilter.from_dict(json.loads(q["tf"])) if q.get("tf") else TrialFilter()

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
        events = event_times(self._trials(q)[0], q["event"])
        return window, float(q["bin"]), baseline, events

    def session_json(self, q: dict) -> dict:
        missing = self.session.available.missing
        return {
            "eid": self.session.eid,
            "n_trials": self.session.n_trials,
            "n_units_total": len(self.units),
            "n_units_passing": int(self.units["qc_passed"].sum()),
            "events": available_events(self.session.trials),
            "levels": list(LEVELS),
            "default_level": DEFAULT_LEVEL,
            "missing": {k: v for k, v in missing.items() if k.startswith("units.")},
            "response": dict(self.response_cfg.__dict__),
            "probes": self.probes,
            "probe_colours": {theme: probe_colours(self.probes, theme) for theme in THEMES},
            "conditions": available_conditions(self.session.trials),
            "comparisons": {
                name: [condition(self.session.trials, name).names[i] for i in (0, -1)]
                for name in available_conditions(self.session.trials)
                if name in COMPARISONS
            },
            "selectivity": dict(self.selectivity_cfg.__dict__),
            "trial_filters": available_trial_filters(self.session.trials),
            "trial_levels": self._trial_levels(),
            "movement": {
                "wheel": "wheel" in self.session.behaviour,
                "wheel_missing": self.session.available.missing.get("behaviour.wheel", ""),
                "first_movement": "firstMovement_times" in self.session.trials,
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
                        c: alignment_label(c).capitalize()
                        for c in EVENT_COLUMNS
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
                    for name in TRACES
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
        """The values the trial filters can choose from, in this session."""
        trials, out = self.session.trials, {}
        if "contrast" in available_conditions(trials):
            levels = np.abs(condition(trials, "contrast").levels)
            out["contrasts"] = sorted({float(v) for v in levels})
        if "probabilityLeft" in trials:
            out["blocks"] = list(condition(trials, "block").levels)
        if "feedbackType" in trials:
            out["outcomes"] = list(condition(trials, "outcome").levels)
        return out

    def save(self, view: dict) -> dict:
        """Write the project file: source, hashes, configs and this view, never results."""
        if self.source is None or self.project_path is None:
            raise ValueError("this Studio was started without a data source to save")
        save_project(make_project(self.source, self.session, self.qc, view), self.project_path)
        self.view = {**DEFAULT_VIEW, **view}
        return {"path": str(self.project_path)}

    def export(self, view: dict) -> dict:
        out = export_view(self, view, RUNS)
        return {"folder": str(out), "files": sorted(p.name for p in out.iterdir())}

    def test_json(self, q: dict) -> dict:
        """Run (or reuse) the responsiveness test on every shown unit for one event."""
        key = self._response_key(q)
        if key not in self._tests:
            ids = self._select({"all": q.get("all", "0"), "probe": q.get("probe", "")})
            if not ids:
                raise ValueError("no units to test")
            trials = self._trials(q)[0]
            if q.get("movement_free") == "1":
                if q["event"] != "stim_on":
                    raise ValueError("movement-free trials are defined for stimulus onset only")
                trials = trials[movement_free(trials, self.response_cfg.response_window[1])]
            events = event_times(trials, q["event"])
            self._tests[key] = responsiveness(self.session.spikes, ids, events, self.response_cfg)
        return self._summary(self._tests[key], key[-1])

    def _summary(self, t: pd.DataFrame, movement_free_only: bool = False) -> dict:
        up = t["responsive"] & (t["statistic_hz"] > 0)
        return {
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
            rows = rows.assign(sel_auroc=t["auroc"], sel_p=t["p"], sel_q=t["q"], sel=t["selective"])
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
        ibl = self.source is not None and self.source.kind == "ibl" and self.source.backend == "bwm"

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
                    "registry, built on the release's session manifest, which a Phy folder lacks."
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

    def decode_start(self, body: dict, manifest) -> dict:
        """Start decoding body["target"] from the units shown by body["query"], in the
        background (analysis.decoding). One run at a time."""
        if self.source is None or self.source.kind != "ibl" or self.source.backend != "bwm":
            raise ValueError(
                "decoding needs a Brain Wide Map session: splits come from the split "
                "registry, which is built on the release's session manifest, and a Phy "
                "folder has none yet"
            )
        target = body.get("target")
        if target not in self.decode_cfg.targets:
            raise ValueError(
                f"{target!r} is not a Studio decoding target: one of {list(self.decode_cfg.targets)}"
            )
        q = {k: str(v) for k, v in (body.get("query") or {}).items()}
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
            target=self._decode_job, args=(target, shown, manifest), daemon=True
        ).start()
        return self.decode_status({})

    def _decode_job(self, target: str, shown: list, manifest) -> None:
        def progress(stage: str, done: int, total: int) -> None:
            with self._decode_lock:
                self._decode.update(stage=stage, done=done, total=total)

        try:
            run = decode(
                self.session.eid,
                target,
                unit_ids=shown,
                cfg=self.decode_cfg,
                manifest=manifest,
                load=lambda eid: self.session,
                progress=progress,
            )
            update = {"state": "done", "summary": run.summary()}
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

    def _pairs_key(self, q: dict) -> tuple:
        """A connections result belongs to exactly the units shown when it ran."""
        return tuple(self._select(q))

    def connections_json(self, q: dict) -> dict:
        """Run (or reuse) the connection test on every pair of shown units."""
        key = self._pairs_key(q)
        if key not in self._connections:
            self._connections[key] = connections(
                self.session.spikes, list(key), self.units, self.ccg_cfg, self.response_cfg.alpha
            )
        return self._connections_summary(self._connections[key])

    def _connections_summary(self, t: pd.DataFrame) -> dict:
        hit = t[t["connected"]]
        units = sorted(set(t["pre"]))
        return {
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
            self._locking[key] = movement_locking(
                self.session.spikes, ids, trials, self.movement_cfg, self.response_cfg.alpha
            )
        return self._locking_summary(self._locking[key])

    def _locking_summary(self, t: pd.DataFrame) -> dict:
        return {
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
            f"Wheel speed · {EVENTS[q['event']][0]} · n = {p.n_trials} trials"
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
            self._selectivity[key] = selectivity(
                self.session.spikes,
                ids,
                self.session.trials,
                q["event"],
                q["split"],
                self.response_cfg,
                self.selectivity_cfg,
                trial_mask=self._trials(q)[1].mask,
            )
        return self._selectivity_summary(self._selectivity[key], q)

    def _selectivity_summary(self, t: pd.DataFrame, q: dict) -> dict:
        names = condition(self.session.trials, q["split"]).names
        higher_b = t["selective"] & (t["auroc"] > 0.5)
        return {
            "condition": CONDITIONS[q["split"]][0],
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
            cond = condition(trials, split)
            colours = condition_colours(split, cond.levels, q.get("theme", "light"))
            groups = []
            raster_trials = []
            for part, colour in zip(split_event_times(trials, q["event"], cond), colours):
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
                f"{q['unit']}{where} · {EVENTS[q['event']][0]} · split by "
                f"{CONDITIONS[split][0].lower()} · n = {n} trials{excluded}{note}"
            )
        else:
            p = psth(spikes, events, window, bin_width, baseline)
            trial, rel = raster(spikes, events, window)
            groups = [TraceGroup("all trials", None, p, trial, rel)]
            rows = trials.index[np.isfinite(trial_event_times(trials, q["event"]))]
            raster_trials = self._trial_numbers(rows)
            caption = (
                f"{q['unit']}{where} · {EVENTS[q['event']][0]} · n = {p.n_trials} trials"
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
        window = self.response_cfg.response_window
        trials, sel = self._trials(q)
        curve = tuning_curve(self.session.spikes[q["unit"]], trials, q["event"], split, window)
        cond = condition(trials, split)
        caption = (
            f"{q['unit']} · response rate {window[0] * 1000:g} to {window[1] * 1000:g} ms after "
            f"{EVENTS[q['event']][0].lower()}, by {CONDITIONS[split][0].lower()} · mean ± SEM"
            f"{self._trial_note(sel)}"
        )
        return {"curve": curve, "levels": cond.levels, "caption": caption}

    def tuning_png(self, q: dict) -> tuple[bytes, dict]:
        d = self.tuning_data(q)
        curve, split = d["curve"], q["split"]
        colours = condition_colours(split, d["levels"], q.get("theme", "light"))
        png = tuning_figure(
            list(curve.index),
            curve["mean_hz"].to_numpy(),
            curve["sem_hz"].to_numpy(),
            curve["n"].tolist(),
            colours,
            split in ("contrast", "block"),
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
        view = trial_view(self.session, self.units.loc[ids], cfg=self.trial_cfg, keep=keep, **args)
        regions = [None if pd.isna(r) else r for r in self._regions(q)[view.rows]]
        info = region_info({r for r in regions if r}) if self.has_regions else {}
        included = sorted(set(view.probes))
        shown = view.window.trials
        which = "responsive " if q.get("responsive") == "1" else ""
        caption = (
            f"Single trial, descriptive (no test) · trial {view.window.trial} (0-based) · "
            f"{len(view.rows)} {which}units ({q.get('node') or 'all regions'}) · "
            f"{'probe' if len(included) == 1 else 'probes'} {', '.join(included)} · "
            f"zero at {alignment_label(view.window.align)}"
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
            f"{EVENTS[q['event']][0]} · sorted by peak time on odd trials (n = {sort_on.size}), "
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
        phy = isinstance(self.qc, PhyUnitQC)
        contamination, alpha = (
            (self.qc.refractory_contamination, self.qc.refractory_alpha)
            if phy
            else (IBL_RP_CONTAMINATION, IBL_RP_ALPHA)
        )
        probe = self.units.at[unit, "probe"]
        waveform, waveform_missing = self._waveform(unit)
        criteria, ibl_missing = (
            self._criteria(unit)
            if not phy
            else (None, "not for Phy folders (IBL's amplitude criteria need volts)")
        )
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
                "config": "configs/qc_phy.yaml" if d["rp_used_by_qc"] else "configs/qc.yaml",
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
            cond = condition(trials, split)
            parts = [(p.name, p.times) for p in split_event_times(trials, q["event"], cond)]
            colour_of = dict(zip(cond.names, condition_colours(split, cond.levels, theme)))
            how = f"split by {CONDITIONS[split][0].lower()}"
            dropped = f" · {cond.n_excluded} {cond.excluded} excluded" if cond.n_excluded else ""
        else:
            parts = [("all trials", trial_event_times(trials, q["event"]))]
            colour_of = {"all trials": THEMES[theme]["series"]}
            how, dropped = "all trials", ""
        r = trajectories(self.session.spikes, ids, parts, window, bin_width, self.traj_cfg)
        shown = ", ".join(f"{a} {v:.0%}" for a, v in zip(r.axis_names, r.explained_held_out))
        not_shown = "".join(f" · {name} not shown: {why}" for name, why in r.excluded.items())
        caption = (
            f"{len(ids)} units ({q.get('node') or 'all regions'}) · {EVENTS[q['event']][0]} · "
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
    ):
        self.data = data
        self.studio = studio
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
                "trial_levels": _BWM_TRIAL_LEVELS,
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
        return {"sets": list_sets(self.sets_dir)}

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
            source = Source(**{k: v for k, v in project["source"].items() if k != "release"})
            atlas = self.data.data_root / "atlas"
            self.studio = Studio(
                session, qc, atlas, source, path, project["view"], warnings, self.ibl_alf(source)
            )
            return {"eid": session.eid, "url": "/session", "warnings": warnings}
        if kind == "phy":
            folder, events = resolve_phy_folder(self.phy_root, str(body.get("path", "")))
            source = Source(kind="phy", folder=str(folder), events=str(events))
            name = folder.name
        elif kind == "ibl" and body.get("eid"):
            source = Source(kind="ibl", eid=str(body["eid"]), backend="bwm")
            name = source.eid[:8]
        else:
            raise ValueError(
                "open needs {kind: 'ibl', eid}, {kind: 'phy', path} or {kind: 'project', name}"
            )
        session, qc = self._loader(source)
        if "trials" in body:
            chosen = TrialFilter.from_dict(body["trials"])
            apply_trial_filter(session.trials, chosen)  # refuses a filter it can't apply
        else:
            chosen = default_trials(session.trials, self.catalog_cfg)
        view = {**DEFAULT_VIEW, "trials": chosen.to_dict()}
        path = _new_project_path(self.data.data_root / "projects", name)
        atlas = self.data.data_root / "atlas"
        self.studio = Studio(session, qc, atlas, source, path, view, [], self.ibl_alf(source))
        return {"eid": session.eid, "url": "/session"}

    def ibl_alf(self, source: Source) -> Path | None:
        """An IBL session's alf folder in the local ONE cache, found from the manifest."""
        if source.kind != "ibl":
            return None
        return ibl_session_folder(self.data.one_cache_root, self.manifest.sessions, source.eid)


def default_trials(trials: pd.DataFrame, cfg) -> TrialFilter:
    """configs/catalog.yaml's default trial filter, minus what this session can't support."""
    offered = available_trial_filters(trials)
    default = cfg.default_trial_filter
    return TrialFilter.from_dict({k: v for k, v in default.items() if offered[k]["available"]})


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
    "/api/trial.png": ("image/png", "trial_png"),
    "/api/quality": ("application/json", "quality_json"),
    "/api/connections": ("application/json", "connections_json"),
    "/api/decode/status": ("application/json", "decode_status"),
    "/api/pair": ("application/json", "pair_json"),
    "/api/pair.png": ("image/png", "pair_png"),
    "/api/trajectories": ("application/json", "trajectory_json"),
    "/api/trajectories.png": ("image/png", "trajectory_png"),
    "/api/quality.png": ("image/png", "quality_png"),
    "/api/tuning.png": ("image/png", "tuning_png"),
    "/api/unit.png": ("image/png", "unit_png"),
    "/api/population.png": ("image/png", "population_png"),
}
# path -> App method answering it; these work with no session open.
_APP_ROUTES = {
    "/api/phy/complete": "phy_complete",
    "/api/projects": "projects",
    "/api/sets": "sets",
}
_JSON_METHODS = {name for ctype, name in _SESSION_ROUTES.values() if ctype == "application/json"}


def make_handler(app: "App | Studio"):
    if isinstance(app, Studio):  # a single session, as before the homepage
        app = App(load_data_config(), studio=app)

    def page(name: str) -> bytes:
        return (HERE / name).read_bytes()

    posts = {
        "/api/open": app.open,
        "/api/sets/save": app.save_set,
        "/api/sets/open": app.open_set,
        "/api/project": lambda view: app.require().save(view),
        "/api/export": lambda view: app.require().export(view),
        "/api/decode": lambda body: app.require().decode_start(body, app.manifest),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
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
    ap.add_argument(
        "--project",
        type=Path,
        help=f"a *{SUFFIX} file: alone, opens it; with a data source, saves a new one there",
    )
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if bool(args.phy) != bool(args.events):
        ap.error("--phy and --events go together")
    if args.phy and args.eid:
        ap.error("choose one data source: --eid or --phy")
    data = load_data_config()
    if not (args.project or args.phy or args.eid):
        return App(data), "/"
    view, warnings = None, []
    if args.project and not (args.phy or args.eid):
        if not args.project.exists():
            ap.error(f"{args.project} does not exist; give a data source to start a new project")
        project, session, qc, warnings = open_project(args.project)
        source = Source(**{k: v for k, v in project["source"].items() if k != "release"})
        view, path = project["view"], args.project
    else:
        if args.phy:
            source = Source(
                kind="phy",
                folder=str(Path(args.phy).resolve()),
                events=str(Path(args.events).resolve()),
            )
        else:
            source = Source(kind="ibl", eid=args.eid, backend=args.backend)
        if args.project and args.project.exists():
            ap.error(f"{args.project} exists; open it with --project alone, or choose a new name")
        name = Path(source.folder).name if source.kind == "phy" else source.eid[:8]
        path = args.project or _new_project_path(data.data_root / "projects", name)
        session, qc = load_source(source)
        # The same default as opening from the homepage; a project keeps its own.
        trials = default_trials(session.trials, load_catalog_config())
        view = {**DEFAULT_VIEW, "trials": trials.to_dict()}
    app = App(data)
    atlas = data.data_root / "atlas"
    app.studio = Studio(session, qc, atlas, source, path, view, warnings, app.ibl_alf(source))
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
