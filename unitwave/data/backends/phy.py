"""A Kilosort/Phy output folder plus a CSV of task events -> Session.

One folder is one probe. Read from the folder:
- `params.py` for `sample_rate`. It is parsed, never executed: only assignments of
  literals are read.
- `spike_times.npy` (integer samples) and `spike_clusters.npy`.
- Each cluster's group (good/mua/noise): from `cluster_group.tsv` if the cluster is
  there, else from `cluster_KSLabel.tsv`, else missing. `group_file` records which.
  Kilosort writes `cluster_group.tsv` as a copy of its own labels, so a group is not
  proof of manual curation.
- Depth, when `templates.npy`, `spike_templates.npy` and `channel_positions.npy` are
  all present: the y position of the peak channel (largest peak-to-peak) of the
  template most of the cluster's spikes use.

The events CSV has one row per trial, numeric columns, and the column names of the
task definition it is read with (analysis.tasks; IBL's canonical names,
session.TRIAL_FIELDS, by default). `intervals_0` and `intervals_1` (trial start and
end) are required; the task period and unit QC are defined by them. Times are in seconds
on the probe's clock, or on another clock with a clock fit from sync pulses
(data.sync, step 13a), which moves every time column, and only those, onto the probe's.
Without one, the loader refuses events outside the span of the recorded spikes, but
cannot detect a smaller offset between clocks.

A Phy folder has no IBL QC label or recording length, so those unit fields are declared
missing. Brain regions and 3-D positions are missing too, unless histology-aligned
channel locations are given (data.channel_locations, step 13b).
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from unitwave.analysis.tasks import TaskDefinition
from unitwave.data.channel_locations import assign_units, read_channel_locations
from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    TRIAL_TIME_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
)
from unitwave.data.sync import ClockFit

_GROUP_FILES = (("cluster_group.tsv", "group"), ("cluster_KSLabel.tsv", "KSLabel"))
_TEMPLATE_FILES = ("templates.npy", "spike_templates.npy", "channel_positions.npy")
_REQUIRED_EVENTS = ("intervals_0", "intervals_1")
_UNIT_MISSING = {
    "acronym": "Phy folders have no brain region",
    "x": "Phy folders have no 3-D position",
    "y": "Phy folders have no 3-D position",
    "z": "Phy folders have no 3-D position",
    "label": "IBL's numeric QC label; Phy gives a group instead, kept as 'phy_group'",
    "firing_rate": "Phy folders don't record the recording's length; QC uses the task-period rate",
}


def read_params(folder: str | Path) -> dict:
    """The literal assignments in params.py. Never executes the file."""
    path = Path(folder) / "params.py"
    if not path.exists():
        raise ValueError(f"{folder} has no params.py, so the sampling rate is unknown")
    params = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    params[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
    rate = params.get("sample_rate")
    if not isinstance(rate, int | float) or not rate > 0:
        raise ValueError(f"{path} has no positive sample_rate")
    return params


def _load(folder: Path, name: str) -> np.ndarray:
    path = folder / name
    if not path.exists():
        raise ValueError(f"{folder} has no {name}; is this a Kilosort/Phy output folder?")
    return np.load(path, allow_pickle=False)


def _groups(folder: Path, cluster_ids: np.ndarray) -> tuple[list, list]:
    """(group, group_file) per cluster id; None where neither file labels it."""
    group, source = [None] * cluster_ids.size, [None] * cluster_ids.size
    index = {int(c): i for i, c in enumerate(cluster_ids)}
    for filename, column in reversed(_GROUP_FILES):  # the first file wins
        path = folder / filename
        if not path.exists():
            continue
        table = pd.read_csv(path, sep="\t")
        if list(table.columns) != ["cluster_id", column]:
            raise ValueError(f"{path}: expected columns cluster_id, {column}")
        for cid, label in zip(table["cluster_id"], table[column]):
            if int(cid) in index and isinstance(label, str):
                group[index[int(cid)]], source[index[int(cid)]] = label, filename
    return group, source


def _peak_channels(folder: Path, order: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    """((n_clusters,) each cluster's peak channel, a row of channel_positions; (n_channels,
    2) channel_positions), or None when a template file is absent.

    The peak channel (largest peak-to-peak) of the template most of the cluster's spikes
    use. order sorts spikes by cluster; cluster i's spikes are order[starts[i]:ends[i]].
    """
    if not all((folder / f).exists() for f in _TEMPLATE_FILES):
        return None
    templates = _load(folder, "templates.npy")  # (n_templates, n_time, n_channels)
    spike_templates = _load(folder, "spike_templates.npy").ravel()  # (n_spikes,)
    positions = _load(folder, "channel_positions.npy")  # (n_channels, 2)
    assert templates.ndim == 3 and positions.shape == (templates.shape[2], 2)
    assert spike_templates.shape == order.shape
    peak_channel = np.ptp(templates, axis=1).argmax(axis=1)  # (n_templates,)
    by_cluster = spike_templates[order]
    dominant = [np.bincount(by_cluster[a:b]).argmax() for a, b in zip(starts, ends)]
    return peak_channel[dominant], positions


def _events(
    path: str | Path,
    span: tuple[float, float],
    task: TaskDefinition | None = None,
    clock: ClockFit | None = None,
) -> pd.DataFrame:
    trials = pd.read_csv(path)
    known = list(dict.fromkeys([*TRIAL_FIELDS, *(task.columns if task else ())]))
    unknown = sorted(set(trials) - set(known))
    if unknown:
        which = f"the task definition {task.label!r}" if task else "IBL's names"
        raise ValueError(
            f"{path}: unknown event columns {unknown}, not read by {which}. Rename them to "
            f"one of {known}, or read the file with another task definition"
        )
    missing = [c for c in _REQUIRED_EVENTS if c not in trials]
    if missing:
        raise ValueError(f"{path}: needs trial start and end columns {missing}")
    try:
        trials = trials.astype(np.float64)
    except ValueError as e:
        raise ValueError(f"{path}: every event column must be numeric ({e})") from None
    empty = [c for c in trials if trials[c].isna().all()]
    trials = trials.drop(columns=empty)
    if not np.isfinite(trials[list(_REQUIRED_EVENTS)].to_numpy()).all():
        raise ValueError(f"{path}: every trial needs a finite intervals_0 and intervals_1")
    times = [*TRIAL_TIME_FIELDS, *(task.time_columns if task else ())]
    for column in [c for c in dict.fromkeys(times) if c in trials]:
        if clock is not None:
            trials[column] = clock.to_probe(trials[column].to_numpy())
        t = trials[column].to_numpy()
        outside = np.isfinite(t) & ((t < span[0]) | (t > span[1]))
        if outside.any():
            raise ValueError(
                f"{path}: {column} has {outside.sum()} times outside the recorded spikes "
                f"({span[0]:.3f}–{span[1]:.3f} s)"
                + (
                    " after the clock fit."
                    if clock is not None
                    else ". Event times must be in seconds on the probe's clock, or come with "
                    "sync pulses."
                )
            )
    return trials


def load_session_phy(
    folder: str | Path,
    events_csv: str | Path,
    probe_name: str | None = None,
    task: TaskDefinition | None = None,
    clock: ClockFit | None = None,
    locations: str | Path | None = None,
) -> Session:
    """Session with one unit per cluster in spike_clusters.npy, ids '<probe_name>_<cluster_id>'.

    probe_name defaults to the folder's name. task: the definition the events CSV is read
    with; its columns are accepted besides IBL's (None: IBL's names only). clock: the
    events' clock mapped onto the probe's (data.sync.fit_clock); None: already the same.
    locations: histology-aligned channel locations (data.channel_locations, step 13b):
    each unit takes its peak channel's region and position.
    """
    folder = Path(folder)
    probe_name = probe_name or folder.name
    rate = float(read_params(folder)["sample_rate"])
    samples = _load(folder, "spike_times.npy").ravel()  # (n_spikes,)
    clusters = _load(folder, "spike_clusters.npy").ravel()  # (n_spikes,)
    if not np.issubdtype(samples.dtype, np.integer):
        raise ValueError(f"{folder}/spike_times.npy must hold integer samples, not {samples.dtype}")
    if samples.shape != clusters.shape or samples.size == 0:
        raise ValueError(f"{folder}: spike_times and spike_clusters differ in length or are empty")
    times = samples.astype(np.float64) / rate  # (n_spikes,) seconds

    order = np.lexsort((times, clusters))  # by cluster, then time
    cluster_ids, starts = np.unique(clusters[order], return_index=True)
    ends = np.append(starts[1:], order.size)
    ids = [f"{probe_name}_{int(c)}" for c in cluster_ids]
    spikes = {u: times[order[a:b]] for u, a, b in zip(ids, starts, ends)}

    group, group_file = _groups(folder, cluster_ids)
    units = pd.DataFrame(
        {
            "probe_name": probe_name,
            "cluster_id": cluster_ids.astype(np.int64),
            "phy_group": group,
            "group_file": group_file,
            "n_spikes": ends - starts,
        },
        index=pd.Index(ids, name="unit_id"),
    )
    missing = {f"units.{f}": why for f, why in _UNIT_MISSING.items()}
    peaks = _peak_channels(folder, order, starts, ends)
    if peaks is None:
        missing["units.depths"] = f"needs all of {list(_TEMPLATE_FILES)} in the folder"
    else:
        units["depths"] = peaks[1][peaks[0], 1]
    if locations is not None:
        if peaks is None:
            raise ValueError(
                f"{folder}: channel locations need templates.npy, spike_templates.npy and "
                "channel_positions.npy, to find each unit's peak channel"
            )
        channel_map = folder / "channel_map.npy"
        placed = assign_units(
            read_channel_locations(locations),
            peaks[1],
            peaks[0],
            np.load(channel_map).ravel() if channel_map.exists() else None,
            Path(locations).name,
        )
        for column in ("acronym", "x", "y", "z"):
            units[column] = placed[column].to_numpy()
            missing.pop(f"units.{column}")

    span = (float(times.min()), float(times.max()))
    trials = _events(events_csv, span, task, clock)
    missing |= {
        f"trials.{f}": "not in the events file, or empty there"
        for f in TRIAL_FIELDS
        if f not in trials
    }
    missing |= {
        f"behaviour.{f}": "Phy import reads spikes and events only" for f in BEHAVIOUR_FIELDS
    }
    present = {f"units.{f}" for f in UNIT_FIELDS if f in units}
    present |= {f"trials.{f}" for f in TRIAL_FIELDS if f in trials}
    return Session(
        eid=f"phy:{folder.resolve()}",
        time_bounds=(0.0, span[1]),
        spikes=spikes,
        units=units,
        trials=trials,
        behaviour={},
        available=Capabilities(present=frozenset(present), missing=missing),
    )
