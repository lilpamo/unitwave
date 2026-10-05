"""Studio project files (`*.unitwave.json`): what was loaded and how it was viewed.

Files saved before the rename end in `.ndstudio.json`. They still open, and saving
one writes the new ending beside it, never over it (docs/DECISIONS.md, "Rename:
UnitWave Studio").

A project holds no results. It records:
- the data source: an IBL session, or a Phy folder plus events CSV, and the task
  definition its trials are read with (analysis.tasks; files saved before step 3 have
  none and open with IBL's, which is what they were computed with);
- a sha256 of each source file (Phy) and a fingerprint of the loaded Session;
- the QC and analysis configs, by hash and content;
- the view: event, window, bin, baseline, region level and node, filters, unit,
  and the single-trial view's trial, alignment, pads and number of trials.

Opening a project reloads the data and recomputes everything, so it can never show
numbers that disagree with the data. Anything that changed since it was saved is
reported as a plain-language warning naming the file, not silently accepted.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from unitwave.analysis.correlograms import DEFAULT_CONFIG as CORRELOGRAM_CONFIG
from unitwave.analysis.movement import DEFAULT_CONFIG as MOVEMENT_CONFIG
from unitwave.analysis.responsiveness import DEFAULT_CONFIG as ANALYSIS_CONFIG
from unitwave.analysis.tasks import DEFAULT_TASK, load_task, task_path
from unitwave.analysis.trajectories import DEFAULT_CONFIG as TRAJECTORY_CONFIG
from unitwave.analysis.trial_view import DEFAULT_CONFIG as TRIAL_VIEW_CONFIG
from unitwave.analysis.trial_view import load_trial_view_config
from unitwave.analysis.tuning import DEFAULT_CONFIG as SELECTIVITY_CONFIG
from unitwave.data.backends.phy import load_session_phy
from unitwave.data.load import key_parts, load_session
from unitwave.data.session import Session
from unitwave.data.sync import fit_clock, load_sync_config, read_pulses
from unitwave.nwb.intake import REPORTS, layout_path, load_layout, read_nwb
from unitwave.qc.nwb import DEFAULT_CONFIG as NWB_QC_CONFIG
from unitwave.qc.nwb import NwbUnitQC, load_nwb_qc_config
from unitwave.qc.phy import DEFAULT_CONFIG as PHY_QC_CONFIG
from unitwave.qc.phy import PhyUnitQC, load_phy_qc_config
from unitwave.qc.spike_times import DEFAULT_CONFIG as SPIKE_QC_CONFIG
from unitwave.qc.spike_times import SpikeQC, load_spike_qc_config
from unitwave.qc.units import DEFAULT_CONFIG as QC_CONFIG
from unitwave.qc.units import load_qc_config

PROJECT_VERSION = 1
SUFFIX = ".unitwave.json"
# Endings read but never written: files saved before the rename.
OLD_SUFFIXES = (".ndstudio.json",)
SUFFIXES = (SUFFIX, *OLD_SUFFIXES)
REPO = Path(__file__).resolve().parents[2]
# Phy's label files: a folder with neither is judged on spike times (qc.spike_times).
PHY_LABEL_FILES = ("cluster_group.tsv", "cluster_KSLabel.tsv")
# Files a Phy folder may hold that change what Studio shows.
PHY_FILES = (
    "params.py",
    "spike_times.npy",
    "spike_clusters.npy",
    "cluster_group.tsv",
    "cluster_KSLabel.tsv",
    "templates.npy",
    "spike_templates.npy",
    "channel_positions.npy",
)
_TRIAL_CFG = load_trial_view_config()
DEFAULT_VIEW = {
    "event": "stim_on",
    "t0": -0.5,
    "t1": 1.0,
    "bin": 0.02,
    "baseline": False,
    "b0": -0.5,
    "b1": 0.0,
    "level": "Beryl",
    "node": "",
    "all": False,
    "responsive": False,
    "unit": None,
    "probe": "",  # "" is every probe; added after version 1 files, which open on all probes
    "split": "",  # "" is no condition split; added later, older files open unsplit
    # analysis.conditions.TrialFilter as a dict; {} keeps every trial. Added later: older
    # files, computed on all trials, open on all trials.
    "trials": {},
    # Responsiveness on movement-free trials only (analysis.movement); added later.
    "movement_free": False,
    # The single-trial view (analysis.trial_view); added later. trial None: none chosen.
    "trial": None,
    "trial_align": "trial_start",
    "trial_pre_s": _TRIAL_CFG.pre_pad_s,
    "trial_post_s": _TRIAL_CFG.post_pad_s,
    "trial_n": 1,
    "trial_all": False,  # step through every trial, not only those passing the filters
    "trial_traces": [],  # optional behaviour traces (analysis.trial_view.TRACES)
    # The selected unit's partner in the Pairs tab (analysis.correlograms); added later.
    "partner": None,
    # The Population card: "heatmap" or "trajectories" (analysis.trajectories), and
    # the trajectories' dimensions (2 or 3); added later.
    "pop_view": "heatmap",
    "traj_dims": 2,
}


@dataclass(frozen=True)
class Source:
    """kind "ibl" (eid, backend), "phy" (folder, events) or "nwb" (file, and the layout
    its specifics are declared in: a built-in name, a YAML path, or None for generic);
    task: the task definition, a built-in name or a YAML file's path. A Phy folder may
    also name its sync pulses on each clock (sync_probe, sync_events; data.sync, step
    13a); projects saved before have neither, and open as before."""

    kind: str
    eid: str | None = None
    backend: str | None = None
    folder: str | None = None
    events: str | None = None
    task: str = DEFAULT_TASK
    file: str | None = None
    layout: str | None = None
    sync_probe: str | None = None
    sync_events: str | None = None

    def __post_init__(self):
        needs = {"ibl": ("eid", "backend"), "phy": ("folder", "events"), "nwb": ("file",)}
        if self.kind not in needs:
            raise ValueError(f"unknown data source kind {self.kind!r}")
        if any(getattr(self, f) is None for f in needs[self.kind]):
            raise ValueError(f"a {self.kind} source needs {needs[self.kind]}")
        if (self.sync_probe is None) != (self.sync_events is None):
            raise ValueError("sync needs the pulses on both clocks: the probe's and the events'")
        if self.sync_probe is not None and self.kind != "phy":
            raise ValueError("sync pulses are read for Phy folders only")

    @classmethod
    def from_record(cls, record: dict) -> "Source":
        """A project's source record, without what was recorded beside it (the IBL
        release, the task definition's label and hash)."""
        return cls(**{k: v for k, v in record.items() if k not in _RECORDED_BESIDE})


# Recorded with a source in a project file, never read back as its fields.
_RECORDED_BESIDE = ("release", "task_label", "task_sha256", "layout_label", "layout_sha256")


def load_source(source: Source):
    """(Session, unit QC) for a source, with the QC that fits its data: IBL's label
    rule, the Phy group rule for a Phy folder with label files, and the spike-time rule
    for one without (qc.spike_times)."""
    if source.kind == "phy":
        task = load_task(source.task)
        clock = None
        if source.sync_probe is not None:
            clock = fit_clock(
                read_pulses(source.sync_probe), read_pulses(source.sync_events), load_sync_config()
            )
        session = load_session_phy(source.folder, source.events, task=task, clock=clock)
        REPORTS[session.eid] = {
            "kind": "phy",
            "clock": None if clock is None else clock.describe(),
        }
        labelled = any((Path(source.folder) / name).exists() for name in PHY_LABEL_FILES)
        return session, load_phy_qc_config() if labelled else load_spike_qc_config()
    if source.kind == "nwb":
        layout = load_layout(source.layout)
        intake = read_nwb(source.file, layout, task=source.task)
        REPORTS[intake.session.eid] = intake.report
        if layout.quality is None:
            return intake.session, load_spike_qc_config()
        return intake.session, load_nwb_qc_config(layout.quality)
    return load_session(source.eid, source.backend), load_qc_config()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(source: Source) -> dict[str, str]:
    """name -> sha256 of each Phy file present, of the events CSV and of any sync pulse
    files (keyed sync_probe/<name> and sync_events/<name>; IBL's with their channels and
    polarities), or of the NWB file. {} for IBL."""
    if source.kind == "nwb":
        return {Path(source.file).name: _sha256(Path(source.file))}
    if source.kind != "phy":
        return {}
    folder = Path(source.folder)
    hashes = {name: _sha256(folder / name) for name in PHY_FILES if (folder / name).exists()}
    hashes[Path(source.events).name] = _sha256(Path(source.events))
    for role in ("sync_probe", "sync_events"):
        path = getattr(source, role)
        if path is None:
            continue
        path = Path(path)
        files = [path]
        if path.name.startswith("_spikeglx_sync.times"):
            files += [
                path.with_name(path.name.replace(".times", f".{part}"))
                for part in ("channels", "polarities")
            ]
        for f in files:
            hashes[f"{role}/{f.name}"] = _sha256(f)
    return hashes


def session_fingerprint(session: Session) -> str:
    """sha256 over every spike train, the units table and the trials table."""
    digest = hashlib.sha256()
    for unit in sorted(session.spikes):
        digest.update(unit.encode())
        digest.update(session.spikes[unit].tobytes())
    for table in (session.units, session.trials):
        digest.update(json.dumps(list(map(str, table.columns))).encode())
        digest.update(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def qc_config_path(qc) -> Path:
    """The config file of a unit QC rule."""
    if isinstance(qc, SpikeQC):
        return Path(SPIKE_QC_CONFIG)
    if isinstance(qc, NwbUnitQC):
        return Path(NWB_QC_CONFIG)
    return Path(PHY_QC_CONFIG if isinstance(qc, PhyUnitQC) else QC_CONFIG)


def _configs(qc) -> dict:
    return {
        name: {"path": str(p.relative_to(REPO)), "sha256": _sha256(p)}
        for name, p in (
            ("qc", qc_config_path(qc)),
            ("spike_qc", Path(SPIKE_QC_CONFIG)),
            ("analysis", Path(ANALYSIS_CONFIG)),
            ("selectivity", Path(SELECTIVITY_CONFIG)),
            ("movement", Path(MOVEMENT_CONFIG)),
            ("trial_view", Path(TRIAL_VIEW_CONFIG)),
            ("correlograms", Path(CORRELOGRAM_CONFIG)),
            ("trajectories", Path(TRAJECTORY_CONFIG)),
        )
    }


def _git_sha() -> str:
    run = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    return run.stdout.strip()


def make_project(source: Source, session: Session, qc, view: dict) -> dict:
    unknown = sorted(set(view) - set(DEFAULT_VIEW))
    if unknown:
        raise ValueError(f"unknown view settings {unknown}; known: {sorted(DEFAULT_VIEW)}")
    project = {
        "version": PROJECT_VERSION,
        "source": asdict(source),
        "files": file_hashes(source),
        "fingerprint": session_fingerprint(session),
        "configs": _configs(qc),
        "view": {**DEFAULT_VIEW, **view},
        "saved_with": {"git_sha": _git_sha(), "at": datetime.now(UTC).isoformat()},
    }
    if source.kind == "ibl":
        project["source"]["release"] = key_parts(source.eid, source.backend)["source"]
    project["source"]["task_label"] = load_task(source.task).label
    project["source"]["task_sha256"] = _sha256(task_path(source.task))
    if source.kind == "nwb":
        project["source"]["layout_label"] = load_layout(source.layout).label
        if source.layout not in (None, "", "generic"):
            project["source"]["layout_sha256"] = _sha256(layout_path(source.layout))
    return project


def save_project(project: dict, path: str | os.PathLike) -> Path:
    """Write atomically: a crash mid-write never leaves half a project."""
    path = Path(path)
    if not path.name.endswith(SUFFIX):
        raise ValueError(f"project files are written as *{SUFFIX}, got {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(project, indent=1))
    tmp.replace(path)
    return path


def project_stem(path: str | os.PathLike) -> str | None:
    """A project file's name without its ending, for either ending; None otherwise."""
    name = Path(path).name
    for suffix in SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


def saving_path(opened: str | os.PathLike) -> Path:
    """Where a project opened from `opened` saves: the same file when it has the new
    ending; otherwise the new ending beside it, at the next free name, so neither the
    old file nor any other is overwritten."""
    opened = Path(opened)
    if opened.name.endswith(SUFFIX):
        return opened
    stem = project_stem(opened) or opened.name
    path, n = opened.with_name(f"{stem}{SUFFIX}"), 2
    while path.exists():
        path, n = opened.with_name(f"{stem}-{n}{SUFFIX}"), n + 1
    return path


def open_project(path: str | os.PathLike):
    """(project, Session, unit QC, warnings) with everything reloaded from the source."""
    project = json.loads(Path(path).read_text())
    if project.get("version") != PROJECT_VERSION:
        raise ValueError(
            f"{path} is project version {project.get('version')}; this Studio reads "
            f"version {PROJECT_VERSION}"
        )
    source = Source.from_record(project["source"])
    session, qc = load_source(source)
    warnings = []
    saved_task = project["source"].get("task_sha256")
    if saved_task is not None and saved_task != _sha256(task_path(source.task)):
        warnings.append(
            f"the task definition {task_path(source.task)} changed since the project was saved"
        )
    saved_layout = project["source"].get("layout_sha256")
    if saved_layout is not None and saved_layout != _sha256(layout_path(source.layout)):
        warnings.append(
            f"the NWB layout {layout_path(source.layout)} changed since the project was saved"
        )
    saved, now = project["files"], file_hashes(source)
    for name in sorted(set(saved) | set(now)):
        if name not in now:
            warnings.append(f"{name} is gone since the project was saved")
        elif name not in saved:
            warnings.append(f"{name} is new since the project was saved")
        elif saved[name] != now[name]:
            warnings.append(f"{name} changed since the project was saved")
    if session_fingerprint(session) != project["fingerprint"]:
        where = {"phy": source.folder, "nwb": source.file}.get(
            source.kind, f"IBL session {source.eid}"
        )
        warnings.append(f"the loaded data differ from when the project was saved ({where})")
    for name, config in _configs(qc).items():
        if name not in project["configs"]:  # files saved before this config was recorded
            warnings.append(f"{config['path']} was not recorded when the project was saved")
        elif config["sha256"] != project["configs"][name]["sha256"]:
            warnings.append(f"{config['path']} changed since the project was saved")
    return project, session, qc, warnings


def view_to_query(view: dict) -> dict[str, str]:
    """A view as the server's query strings: booleans "1"/"0", lists comma-joined, the
    trial filters as JSON under "tf" (as the page sends them), and no key for None."""
    unknown = sorted(set(view) - set(DEFAULT_VIEW))
    if unknown:
        raise ValueError(f"unknown view settings {unknown}")
    query = {}
    for key, value in {**DEFAULT_VIEW, **view}.items():
        if value is None:
            continue
        if key == "trials":
            query["tf"] = json.dumps(value)
        elif isinstance(value, bool):
            query[key] = "1" if value else "0"
        elif isinstance(value, list | tuple):
            query[key] = ",".join(map(str, value))
        else:
            query[key] = str(value)
    return query
