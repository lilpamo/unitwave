"""Brain Wide Map session manifest, built from the compressed releases' metadata.

`sessions` has one row per session: eid, subject, lab, date, probe/unit/trial counts,
the Beryl regions of its good units, and the behavioural modalities the bwm_behavior
release holds for it (canonical names: wheel, pose_left, motion_energy_left,
pupil_left, ...). `insertions` has one row per probe, with unit counts and tip/top
coordinates. `region_units` counts each probe's good units per Allen acronym (the
release's finest region), for filtering by any region of the hierarchy.

Version 2 added `region_units` and the motion-energy and pupil modalities, which are
read from each session shard's meta.json by the same rule the BWM backend uses.

Counts are computed from the underlying tables and must agree with the release's own
precomputed per-session counts, or building raises.
"""

import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from unitwave.data.backends import bwm_compressed

MANIFEST_VERSION = 2
BEHAVIOUR_DATASET = "bwm_behavior"
BEHAVIOUR_VERSION = "2.0.0"
_CAMERAS = {"leftCamera": "left", "rightCamera": "right", "bodyCamera": "body"}
_MLAPDV = ["mlapdv_x", "mlapdv_y", "mlapdv_z"]
# Camera signals read from shard metadata, as data/backends/bwm_compressed.py loads them.
_CAMERA_SIGNALS = {
    "leftCamera": {"motion_energy": "whiskerMotionEnergy", "pupil": "pupilDiameter_raw"},
    "rightCamera": {"motion_energy": "whiskerMotionEnergy", "pupil": "pupilDiameter_raw"},
    "bodyCamera": {"motion_energy": "bodyMotionEnergy"},
}


@dataclass(frozen=True)
class Manifest:
    sessions: pd.DataFrame
    insertions: pd.DataFrame
    provenance: dict
    region_units: pd.DataFrame = field(default_factory=pd.DataFrame)


def manifest_versions() -> dict:
    """What a manifest built by this code depends on: its code version and release versions."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "sources": {
            bwm_compressed.DATASET_NAME: bwm_compressed.DATASET_VERSION,
            BEHAVIOUR_DATASET: BEHAVIOUR_VERSION,
        },
    }


def _check_release(root: Path, name: str, version: str) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    found = (manifest.get("dataset_name"), manifest.get("dataset_version"))
    if found != (name, version):
        raise ValueError(f"{root}: found {found}, expected {(name, version)}")


def _probe_ends(channels: pd.DataFrame) -> pd.DataFrame:
    """Per insertion, the mean position (m) of its channels nearest and farthest from the tip.

    Channel positions are IBL mlapdv in um relative to bregma; units' x/y/z use the same
    frame in meters, so the ends are converted to meters.
    """
    depth = channels["localCoordinates_y"]
    by_pid = depth.groupby(channels["pid"])
    ends = {}
    for end, extreme in (("tip", by_pid.transform("min")), ("top", by_pid.transform("max"))):
        at_end = channels[depth == extreme].groupby("pid")[_MLAPDV].mean() / 1e6
        ends[end] = at_end.set_axis([f"{end}_x", f"{end}_y", f"{end}_z"], axis=1)
    return pd.concat([ends["tip"], ends["top"]], axis=1)


def _modalities(behaviour_root: Path) -> pd.Series:
    meta = behaviour_root / "metadata"
    wheel = pd.read_parquet(meta / "wheel_availability.parquet")
    pose = pd.read_parquet(meta / "pose_availability.parquet")
    pose = pose[pose["pose_present"]]
    unknown = sorted(set(pose["camera"]) - set(_CAMERAS))
    if unknown:
        raise ValueError(f"unexpected pose cameras: {unknown}")
    rows = [(eid, "wheel") for eid in wheel.loc[wheel["wheel_present"], "eid"]]
    rows += [(eid, f"pose_{_CAMERAS[cam]}") for eid, cam in zip(pose["eid"], pose["camera"])]
    rows += _camera_signals(behaviour_root / "sessions")
    names = pd.DataFrame(rows, columns=["eid", "modality"])
    eids = pd.Index(wheel["eid"].unique(), name="eid")
    return names.groupby("eid")["modality"].apply(lambda s: sorted(set(s))).reindex(eids)


def _camera_signals(shards: Path) -> list[tuple[str, str]]:
    """(eid, modality) for motion energy and pupil, from each shard's meta.json only.

    Present when the camera has features, the column is listed, and the bwm_behavior
    build didn't skip it: the same rule as the backend's _behaviour_from_shard.
    """
    rows = []
    for shard in sorted(shards.glob("*.zip")) if shards.exists() else []:
        with zipfile.ZipFile(shard) as z:
            meta = json.loads(z.read(next(n for n in z.namelist() if n.endswith("meta.json"))))
        arrays = meta.get("arrays", {})
        for name, signals in _CAMERA_SIGNALS.items():
            info = meta.get("cameras", {}).get(name)
            if info is None or f"{name}.features" not in arrays:
                continue
            columns, skipped = set(info["columns"]), set(info.get("skipped_sources", []))
            for signal, column in signals.items():
                if column in columns and column not in skipped:
                    rows.append((shard.stem, f"{signal}_{_CAMERAS[name]}"))
    return rows


def _count(frame: pd.DataFrame, by: str, index: pd.Index) -> pd.Series:
    return frame.groupby(by).size().reindex(index, fill_value=0).astype(np.int64)


def _agree(ours: pd.Series, theirs: pd.Series, name: str) -> None:
    bad = ours.index[ours.to_numpy() != theirs.reindex(ours.index).to_numpy()]
    if len(bad):
        raise ValueError(
            f"{name} disagrees with the release for {len(bad)} sessions, e.g. {list(bad[:3])}"
        )


def build_manifest(ephys_root: str | os.PathLike, behaviour_root: str | os.PathLike) -> Manifest:
    """Build the manifest from extracted bwm_ephys 1.2.1 and bwm_behavior 2.0.0 releases."""
    ephys_root, behaviour_root = Path(ephys_root), Path(behaviour_root)
    _check_release(ephys_root, bwm_compressed.DATASET_NAME, bwm_compressed.DATASET_VERSION)
    _check_release(behaviour_root, BEHAVIOUR_DATASET, BEHAVIOUR_VERSION)

    meta = ephys_root / "metadata"
    release = pd.read_parquet(meta / "sessions.parquet").set_index("eid")
    insertions = pd.read_parquet(meta / "insertions.parquet", columns=["pid", "eid", "probe_name"])
    clusters = pd.read_parquet(
        ephys_root / "clusters.pqt", columns=["pid", "eid", "cluster_id", "label"]
    )
    units = pd.read_parquet(
        meta / "units.parquet", columns=["pid", "eid", "cluster_id", "acronym", "beryl_acronym"]
    )
    trials = pd.read_parquet(meta / "trials.parquet", columns=["eid", "bwm_include"])
    channels = pd.read_parquet(
        meta / "channels.parquet", columns=["pid", "localCoordinates_y", *_MLAPDV]
    )

    # The release's good units are a strict subset of label == 1 clusters: it also drops
    # units located in void/root and a few others (see docs/DECISIONS.md).
    label1 = clusters[clusters["label"] == 1.0]
    key = ["pid", "cluster_id"]
    not_label1 = units[key].merge(label1[key], on=key, how="left", indicator=True)
    if (not_label1["_merge"] == "left_only").any():
        raise ValueError("units table lists good units whose clusters.pqt label is not 1")

    pids = pd.Index(insertions["pid"], name="pid")
    probe_table = insertions.set_index("pid")
    probe_table["n_units"] = _count(clusters, "pid", pids)
    probe_table["n_label1_units"] = _count(label1, "pid", pids)
    probe_table["n_good_units"] = _count(units, "pid", pids)
    probe_table["n_channels"] = _count(channels, "pid", pids)
    probe_table = probe_table.join(_probe_ends(channels)).reset_index()

    eids = pd.Index(release.index, name="eid")
    sessions = release[["subject", "lab", "date", "session_number"]].copy()
    sessions["n_probes"] = _count(insertions, "eid", eids)
    sessions["n_units"] = _count(clusters, "eid", eids)
    sessions["n_label1_units"] = _count(label1, "eid", eids)
    sessions["n_good_units"] = _count(units, "eid", eids)
    sessions["n_trials"] = _count(trials, "eid", eids)
    sessions["n_included_trials"] = (
        trials.groupby("eid")["bwm_include"].sum().reindex(eids, fill_value=0).astype(np.int64)
    )
    for ours, theirs in [
        ("n_probes", "n_insertions"),
        ("n_good_units", "n_good_units"),
        ("n_trials", "n_trials"),
        ("n_included_trials", "n_included_trials"),
    ]:
        _agree(sessions[ours], release[theirs], ours)

    regions = units.dropna(subset=["beryl_acronym"]).groupby("eid")["beryl_acronym"]
    sessions["regions"] = regions.apply(lambda s: sorted(set(s))).reindex(eids)
    modalities = _modalities(behaviour_root)
    if set(modalities.index) != set(eids):
        raise ValueError("bwm_behavior and bwm_ephys list different sessions")
    sessions["modalities"] = modalities.reindex(eids)
    for column in ("regions", "modalities"):
        sessions[column] = sessions[column].map(lambda v: v if isinstance(v, list) else [])

    sessions = sessions.reset_index().sort_values(
        ["lab", "subject", "date", "session_number", "eid"]
    )
    probe_table = probe_table.sort_values(["eid", "probe_name", "pid"])
    region_units = (
        units.dropna(subset=["acronym"])
        .groupby(["eid", "pid", "acronym"])
        .size()
        .rename("n_good_units")
        .reset_index()
        .sort_values(["eid", "pid", "acronym"])
        .reset_index(drop=True)
    )
    if region_units["n_good_units"].sum() != len(units.dropna(subset=["acronym"])):
        raise ValueError("region_units lost units")
    provenance = {
        **manifest_versions(),
        "n_sessions": int(len(sessions)),
        "n_insertions": int(len(probe_table)),
    }
    return Manifest(
        sessions=sessions.reset_index(drop=True),
        insertions=probe_table.reset_index(drop=True),
        provenance=provenance,
        region_units=region_units,
    )


def write_manifest(manifest: Manifest, directory: str | os.PathLike) -> Path:
    """Write sessions.parquet, insertions.parquet and provenance.json, replacing atomically."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-manifest-", dir=directory.parent))
    try:
        manifest.sessions.to_parquet(tmp / "sessions.parquet")
        manifest.insertions.to_parquet(tmp / "insertions.parquet")
        manifest.region_units.to_parquet(tmp / "region_units.parquet")
        (tmp / "provenance.json").write_text(json.dumps(manifest.provenance, indent=2))
        if directory.exists():
            shutil.rmtree(directory)
        os.rename(tmp, directory)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return directory


def read_manifest(directory: str | os.PathLike) -> Manifest:
    directory = Path(directory)
    return Manifest(
        sessions=pd.read_parquet(directory / "sessions.parquet"),
        insertions=pd.read_parquet(directory / "insertions.parquet"),
        provenance=json.loads((directory / "provenance.json").read_text()),
        region_units=(
            pd.read_parquet(directory / "region_units.parquet")
            if (directory / "region_units.parquet").exists()
            else pd.DataFrame()
        ),
    )
