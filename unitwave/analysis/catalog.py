"""The homepage's session catalog: the BWM manifest, filtered, with live counts.

The manifest (data/manifest.py) is built once from the releases and written to
`derived/manifest-v<MANIFEST_VERSION>/`. Later starts read that copy; a new manifest
version is a new folder, so an old copy is never read by new code.

Unit counts here are the release's good units (its units table), not Studio's QC
count: d23a44ef has 398 release good units, of which Studio's QC passes 390.

Region filtering uses the Allen hierarchy (analysis/atlas.py): a node matches its own
units and all its descendants' (HPF includes CA1 and DG-mo).
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unitwave.analysis.atlas import ccf_um, region_at_level, region_tree, units_in_node
from unitwave.data.manifest import (
    MANIFEST_VERSION,
    Manifest,
    read_manifest,
    write_manifest,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "catalog.yaml"


@dataclass(frozen=True)
class CatalogConfig:
    min_region_units: int
    default_trial_filter: dict = field(default_factory=dict)
    phy_root: str = "phy"  # relative to data_root unless absolute
    nwb_root: str = "dandi"  # relative to data_root unless absolute


def load_catalog_config(path: str | os.PathLike = DEFAULT_CONFIG) -> CatalogConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    keys = {"min_region_units", "default_trial_filter", "phy_root", "nwb_root"}
    if set(raw) != keys:
        raise ValueError(f"{path}: keys {sorted(raw)}, expected {sorted(keys)}")
    return CatalogConfig(
        int(raw["min_region_units"]),
        dict(raw["default_trial_filter"]),
        str(raw["phy_root"]),
        str(raw["nwb_root"]),
    )


def load_catalog(derived_root: str | os.PathLike, build: Callable[[], Manifest]) -> Manifest:
    """The written manifest for this MANIFEST_VERSION, building and writing it if absent."""
    directory = Path(derived_root) / f"manifest-v{MANIFEST_VERSION}"
    if directory.is_dir():
        manifest = read_manifest(directory)
        if manifest.provenance.get("manifest_version") == MANIFEST_VERSION:
            return manifest
    manifest = build()
    write_manifest(manifest, directory)
    return manifest


@dataclass(frozen=True)
class SessionFilter:
    """Empty tuples, None and 0 match everything."""

    labs: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    date_from: str | None = None  # "YYYY-MM-DD", inclusive
    date_to: str | None = None
    region: str | None = None  # an Allen acronym; descendants included
    min_region_units: int = 0
    min_good_units: int = 0
    min_included_trials: int = 0
    n_probes: tuple[int, ...] = ()
    modalities: tuple[str, ...] = ()  # all required


def _region_units_per_session(manifest: Manifest, region: str) -> pd.Series:
    """eid -> the session's good units in `region` or its descendants."""
    ru = manifest.region_units
    acronyms = ru["acronym"].unique()
    inside = dict(zip(acronyms, units_in_node(acronyms, region)))
    hits = ru[ru["acronym"].map(inside)]
    return hits.groupby("eid")["n_good_units"].sum()


def filter_sessions(manifest: Manifest, f: SessionFilter) -> pd.DataFrame:
    """Matching sessions, in manifest order; with `region_units` when a region is set."""
    s = manifest.sessions
    keep = np.ones(len(s), bool)
    if f.labs:
        keep &= s["lab"].isin(f.labs).to_numpy()
    if f.subjects:
        keep &= s["subject"].isin(f.subjects).to_numpy()
    if f.date_from:
        keep &= (s["date"] >= f.date_from).to_numpy()
    if f.date_to:
        keep &= (s["date"] <= f.date_to).to_numpy()
    keep &= (s["n_good_units"] >= f.min_good_units).to_numpy()
    keep &= (s["n_included_trials"] >= f.min_included_trials).to_numpy()
    if f.n_probes:
        keep &= s["n_probes"].isin(f.n_probes).to_numpy()
    if f.modalities:
        need = set(f.modalities)
        keep &= s["modalities"].map(lambda m: need <= set(m)).to_numpy(bool)
    out = s[keep].copy()
    if f.region:
        counts = _region_units_per_session(manifest, f.region)
        out["region_units"] = out["eid"].map(counts).fillna(0).astype(int)
        out = out[out["region_units"] >= max(f.min_region_units, 1)]
    return out.reset_index(drop=True)


def summary(manifest: Manifest, matching: pd.DataFrame) -> dict:
    """{n_sessions, n_probes, n_units, n_region_units}: release good units, not Studio QC."""
    return {
        "n_sessions": len(matching),
        "n_probes": int(manifest.insertions["eid"].isin(matching["eid"]).sum()),
        "n_units": int(matching["n_good_units"].sum()),
        "n_region_units": (
            int(matching["region_units"].sum()) if "region_units" in matching else None
        ),
    }


def region_counts(manifest: Manifest, eids, level: str) -> list[dict]:
    """The region tree at `level` over the given sessions' good units, with counts."""
    ru = manifest.region_units[manifest.region_units["eid"].isin(list(eids))]
    if ru.empty:
        return []
    per_acronym = ru.groupby("acronym")["n_good_units"].sum()
    at_level = region_at_level(per_acronym.index.to_numpy(object), level)
    regions = np.repeat(at_level, per_acronym.to_numpy())
    return region_tree(regions)


def probe_lines(manifest: Manifest, eids) -> pd.DataFrame:
    """One row per probe of the given sessions: its tip and top in CCF µm (ap, dv, ml).

    The manifest stores IBL xyz in metres; they are converted with atlas.ccf_um, the same
    conversion as unit positions, so a probe line runs through its units.
    """
    ins = manifest.insertions[manifest.insertions["eid"].isin(list(eids))]
    sessions = manifest.sessions.set_index("eid")[["lab", "subject", "date"]]
    ins = ins.join(sessions, on="eid")
    out = ins[["pid", "eid", "probe_name", "lab", "subject", "date"]].copy()
    for end in ("tip", "top"):
        xyz = ins[[f"{end}_x", f"{end}_y", f"{end}_z"]].to_numpy(np.float64)
        out[end] = list(ccf_um(xyz))
    return out.reset_index(drop=True)
