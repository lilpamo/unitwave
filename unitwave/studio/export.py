"""Figure export: the current view as SVG and PDF, with every plotted number beside it.

Each export is a new `runs/<run_id>/` folder (run_id = UTC time + "_studio"):
- `unit.svg`, `unit.pdf`, `unit.json`: the selected unit's raster and PSTH;
- `population.svg`, `population.pdf`, `population.json`: the heatmap and mean;
- `wheel.svg`, `wheel.pdf`, `wheel.json`: wheel speed on the same trials, when the
  session has a wheel;
- `trial.svg`, `trial.pdf`, `trial.json`: the single-trial view, when a trial is
  chosen, with every spike time and event time it plots;
- `quality.svg`, `quality.pdf`, `quality.json`: the selected unit's quality panel,
  with every count, test row and waveform sample it plots;
- `trajectories.svg`, `trajectories.pdf`, `trajectories.json`: the population
  trajectories, when the Population card shows them;
- `ccg.svg`, `ccg.pdf`, `ccg.json`: the cross-correlogram with the chosen partner;
  `connections.csv` when the connection test was run on the shown units;
- `responsiveness.csv`, when the test was run for this event (on all trials or on
  movement-free trials only, as the view says);
- `selectivity.csv` and `movement_locking.csv`, when those tests were run;
- `manifest.json`: git SHA and dirty flag, the project snapshot (source, file
  hashes, data fingerprint, config hashes), the view and the software versions.

Figures come from the same viz/ builders as the page, in the light theme at a fixed
size, titled with the page's caption, with text kept editable. The JSON sidecars hold
the arrays the figures were drawn from, so any number in a figure traces to a file.
Nothing is random, so there is no seed.
"""

import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from unitwave.studio import APP_NAME
from unitwave.studio.project import REPO, make_project, view_to_query
from unitwave.viz.studio_plots import (
    build_ccg_figure,
    build_population_figure,
    build_quality_figure,
    build_trajectory_figure,
    build_trial_figure,
    build_tuning_figure,
    build_unit_figure,
    build_wheel_figure,
    probe_colours,
    save_vector,
)

THEME = "light"


def _jsonable(value):
    """Arrays become lists; NaN and infinities become null; numpy scalars become numbers."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return None if not math.isfinite(value) else float(value)
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=1, allow_nan=False))


def _git() -> dict:
    def git(*args):
        run = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
        return run.stdout.strip()

    return {
        "sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def export_view(studio, view: dict, runs_dir: str | os.PathLike) -> Path:
    """Write the view's figures, sidecars and manifest to a new run folder; return it."""
    q = view_to_query(view)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_studio"
    out = Path(runs_dir) / run_id
    out.mkdir(parents=True)
    files = []

    if view.get("unit"):
        d = studio.unit_data(q)
        fig = build_unit_figure(
            d["groups"], d["window"], d["baseline"] is not None, THEME, d["caption"]
        )
        for suffix in ("svg", "pdf"):
            save_vector(fig, out / f"unit.{suffix}")

        def trace(g):
            p = g.psth
            return {
                "condition": g.name,
                "bin_centers": p.bin_centers,
                "mean": p.mean,
                "sem": p.sem,
                "n_trials": p.n_trials,
                "n_excluded": p.n_excluded,
                "raster": {"trial": g.trial, "time_s": g.rel},
            }

        traces = [trace(g) for g in d["groups"]]
        sidecar = {
            "caption": d["caption"],
            "trials": studio._trial_summary(q),
            "unit": view["unit"],
            "split": view.get("split") or None,
            "window_s": d["window"],
            "baseline_s": d["baseline"],
        }
        if view.get("split"):
            sidecar["groups"] = traces
        else:
            sidecar["psth"] = {
                k: v for k, v in traces[0].items() if k not in ("condition", "raster")
            }
            sidecar["raster"] = traces[0]["raster"]
        _write_json(out / "unit.json", sidecar)
        files += ["unit.svg", "unit.pdf", "unit.json"]

        if view.get("split"):
            t = studio.tuning_data(q)
            curve = t["curve"]
            fig = build_tuning_figure(
                list(curve.index),
                curve["mean_hz"].to_numpy(),
                curve["sem_hz"].to_numpy(),
                curve["n"].tolist(),
                studio._colours(view["split"], t["levels"], THEME, for_axis=True),
                studio.task.conditions[view["split"]].type == "ordinal",
                THEME,
                t["caption"],
            )
            for suffix in ("svg", "pdf"):
                save_vector(fig, out / f"tuning.{suffix}")
            _write_json(
                out / "tuning.json",
                {
                    "caption": t["caption"],
                    "trials": studio._trial_summary(q),
                    "levels": curve.reset_index().to_dict(orient="list"),
                },
            )
            files += ["tuning.svg", "tuning.pdf", "tuning.json"]

    d = studio.population_data(q)
    fig, _ = build_population_figure(
        d["scaled"],
        d["bin_centers"],
        d["mean"],
        d["sem"],
        d["window"],
        THEME,
        d["caption"],
        row_groups=d["probes"],
        group_colours=probe_colours(studio.probes, THEME),
    )
    for suffix in ("svg", "pdf"):
        save_vector(fig, out / f"population.{suffix}")
    _write_json(out / "population.json", {**d, "trials": studio._trial_summary(q)})
    files += ["population.svg", "population.pdf", "population.json"]

    if "wheel" in studio.session.behaviour:
        w = studio.wheel_data(q)
        p = w["psth"]
        fig = build_wheel_figure(p, w["window"], THEME, w["caption"])
        for suffix in ("svg", "pdf"):
            save_vector(fig, out / f"wheel.{suffix}")
        _write_json(
            out / "wheel.json",
            {
                "caption": w["caption"],
                "trials": studio._trial_summary(q),
                "unit": "rad/s",
                "bin_centers_s": p.bin_centers.tolist(),
                "mean": p.mean.tolist(),
                "sem": p.sem.tolist(),
                "n_trials": p.n_trials,
                "n_excluded": p.n_excluded,
            },
        )
        files += ["wheel.svg", "wheel.pdf", "wheel.json"]

    if view.get("unit"):
        d = studio.quality_data(q)
        uq = d["quality"]
        fig = build_quality_figure(uq, THEME, d["caption"])
        for suffix in ("svg", "pdf"):
            save_vector(fig, out / f"quality.{suffix}")
        rd = uq.refractory
        _write_json(
            out / "quality.json",
            {
                **studio.quality_json(q),
                "span_s": uq.span_s,
                "isi": {"edges_s": uq.isi[0], "counts": uq.isi[1], "beyond_max": uq.isi[2]},
                "autocorrelogram": {"lags_s": uq.acg[0], "counts": uq.acg[1]},
                "refractory_test": {
                    "rp_s": rd.rp_s,
                    "violations": rd.violations,
                    "max_acceptable": rd.max_acceptable,
                    "rate_hz": rd.rate_hz,
                    "duration_s": rd.duration_s,
                },
                "presence_counts": uq.presence[1],
                "rate": {"centres_s": uq.rate[0], "rate_hz": uq.rate[1]},
                "waveform_samples": None if uq.waveform is None else uq.waveform.samples_uv,
                "waveform_channels": None if uq.waveform is None else uq.waveform.channels,
            },
        )
        files += ["quality.svg", "quality.pdf", "quality.json"]

    if view.get("pop_view") == "trajectories":
        d = studio.trajectory_data(q)
        r = d["result"]
        fig = build_trajectory_figure(
            r, d["colours"], int(view.get("traj_dims", 2)), THEME, d["caption"]
        )
        for suffix in ("svg", "pdf"):
            save_vector(fig, out / f"trajectories.{suffix}")
        _write_json(
            out / "trajectories.json",
            {
                "caption": d["caption"],
                "trials": studio._trial_summary(q),
                "conditions": r.names,
                "n_fit": r.n_fit,
                "n_show": r.n_show,
                "excluded": r.excluded,
                "axis_names": r.axis_names,
                "bin_centers_s": r.bin_centers,
                "trajectories": r.trajectories,  # (n_conditions, n_bins, k), shown half
                "components": r.components,  # (n_units, k), rows in "units" order
                "units": studio._select(q),
                "explained_fit": r.explained_fit,
                "explained_held_out": r.explained_held_out,
                "config": asdict(studio.traj_cfg),
            },
        )
        files += ["trajectories.svg", "trajectories.pdf", "trajectories.json"]

    if view.get("unit") and view.get("partner"):
        d = studio.pair_data(q)
        fig = build_ccg_figure(
            d["lags"],
            d["observed"],
            d["expected"],
            studio.ccg_cfg.synaptic_window_s,
            THEME,
            d["caption"],
        )
        for suffix in ("svg", "pdf"):
            save_vector(fig, out / f"ccg.{suffix}")
        _write_json(
            out / "ccg.json",
            {
                **{k: d[k] for k in ("caption", "close", "close_why", "tests")},
                "unit": view["unit"],
                "partner": view["partner"],
                "lags_s": d["lags"],
                "observed": d["observed"],
                "expected_under_jitter": d["expected"],
                "config": asdict(studio.ccg_cfg),
            },
        )
        files += ["ccg.svg", "ccg.pdf", "ccg.json"]
    pairs = studio._connections.get(studio._pairs_key(q))
    if pairs is not None:
        pairs.to_csv(out / "connections.csv", index=False)
        files.append("connections.csv")

    if view.get("trial") is not None:
        t = studio.trial_data(q)
        v = t["view"]
        fig, _ = build_trial_figure(
            v, t["regions"], t["region_colours"], view.get("unit"), THEME, t["caption"]
        )
        for suffix in ("svg", "pdf"):
            save_vector(fig, out / f"trial.{suffix}")
        _write_json(
            out / "trial.json",
            {
                "caption": t["caption"],
                "trials": studio._trial_summary(q),
                "header": v.header,
                "window": {
                    "trials": list(v.window.trials),
                    "trial": v.window.trial,
                    "zero": v.window.align,
                    "zero_s": v.window.zero_s,
                    "start_rel_s": v.window.start_rel_s,
                    "stop_rel_s": v.window.stop_rel_s,
                },
                "times": "seconds relative to zero_s (session clock)",
                "rows": [
                    {"unit": u, "probe": p, "region": r, "spikes_s": x}
                    for u, p, r, x in zip(v.rows, v.probes, t["regions"], v.spikes)
                ],
                "events": v.events,
                "not_recorded": v.not_recorded,
                "absent_events": v.absent_events,
                "boundaries": v.boundaries,
                "wheel": v.wheel,
                "wheel_missing": v.wheel_missing,
                "traces": {k: {"t_s": x, "values": y} for k, (x, y) in v.traces.items()},
                "traces_missing": v.traces_missing,
            },
        )
        files += ["trial.svg", "trial.pdf", "trial.json"]

    tested = studio._tested(q)
    if tested is not None:
        tested.to_csv(out / "responsiveness.csv")
        files.append("responsiveness.csv")
    chosen = studio._selectivity.get(studio._selectivity_key(q))
    if chosen is not None:
        chosen.to_csv(out / "selectivity.csv")
        files.append("selectivity.csv")
    locking = studio._locking.get(studio._locking_key(q))
    if locking is not None:
        locking.to_csv(out / "movement_locking.csv")
        files.append("movement_locking.csv")

    manifest = {
        "run_id": run_id,
        "app": APP_NAME,
        "created": datetime.now(UTC).isoformat(),
        "command": "unitwave.studio export",
        "git": _git(),
        "seed": None,
        "project": make_project(studio.source, studio.session, studio.qc, view),
        "view": view,
        "responsiveness": None if tested is None else asdict(studio.response_cfg),
        "selectivity": None if chosen is None else asdict(studio.selectivity_cfg),
        "movement": None if locking is None else asdict(studio.movement_cfg),
        "correlograms": (
            None if pairs is None and not view.get("partner") else asdict(studio.ccg_cfg)
        ),
        "files": files,
        "versions": {
            "python": platform.python_version(),
            **{m: sys.modules[m].__version__ for m in ("numpy", "pandas", "scipy", "matplotlib")},
        },
    }
    _write_json(out / "manifest.json", manifest)
    return out
