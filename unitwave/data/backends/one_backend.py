"""IBL's ONE API (public Alyx) -> Session.

**Revisions are pinned.** By default ONE serves the newest revision of a dataset, so
the same eid can silently yield different data over time (Phase 0 found exactly that:
see docs/PRIOR_ART.md §C). This backend always asks for a fixed spike-sorting and
trials revision, and both go into the cache key.

Checked against the NWB backend on d23a44ef: identical units, identical spike times
(both come from the same sorting run), identical trials and wheel.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    TRIAL_TIME_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
    TimeSeries,
)

PUBLIC_ALYX = "https://openalyx.internationalbrainlab.org"
# The IBL spike sorting NWB and the compressed BWM both use, and the only trials
# revision on Alyx for the Brain Wide Map sessions as of 2026-09-28.
SORTER_REVISION = "2024-05-06"
TRIALS_REVISION = "2025-03-03"
SORTER_COLLECTION = "pykilosort"
# Bump when this module's mapping into a Session changes; it invalidates cached sessions.
LOADER_VERSION = 1

_PROBE_COLLECTION = re.compile(rf"^alf/(probe\d+[a-z]?)/{SORTER_COLLECTION}$")
# Trial fields ONE names differently, or that need reshaping.
_UNIT_EXTRAS = ["cluster_id", "cluster_uuid", "atlas_id", "peak_channel"]
_CLUSTER_ATTRIBUTES = ["channels", "depths", "metrics", "uuids"]
_CHANNEL_ATTRIBUTES = ["mlapdv", "brainLocationIds_ccf_2017"]
_ACRONYM_REASON = (
    "ONE gives Allen CCF ids (kept as 'atlas_id'); mapping ids to acronyms needs iblatlas"
)
_BEHAVIOUR_REASON = "not loaded by this backend yet"


def make_one(cache_dir: str | os.PathLike, base_url: str = PUBLIC_ALYX):
    """A ONE client on IBL's public database, caching downloads under `cache_dir`."""
    from one.api import ONE

    return ONE(base_url=base_url, cache_dir=str(Path(cache_dir).expanduser()), silent=True)


def _probe_collections(one, eid: str) -> list[tuple[str, str]]:
    """(probe_name, collection) for every spike-sorted probe in the session, sorted."""
    found = {}
    for collection in one.list_collections(eid, collection=f"alf/*/{SORTER_COLLECTION}"):
        match = _PROBE_COLLECTION.match(collection)
        if match:
            found[match.group(1)] = collection
    if not found:
        raise ValueError(f"{eid}: no spike-sorted probes under alf/*/{SORTER_COLLECTION}")
    return sorted(found.items())


def _units_and_spikes(
    one, eid: str, probes: list[tuple[str, str]], revision: str
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    tables, spikes = [], {}
    for probe_name, collection in probes:
        clusters = one.load_object(
            eid, "clusters", collection=collection, revision=revision, attribute=_CLUSTER_ATTRIBUTES
        )
        channels = one.load_object(
            eid, "channels", collection=collection, revision=revision, attribute=_CHANNEL_ATTRIBUTES
        )
        spike = one.load_object(
            eid, "spikes", collection=collection, revision=revision, attribute=["times", "clusters"]
        )

        metrics = clusters.metrics.reset_index(drop=True)
        n_units = len(metrics)
        peak_channel = np.asarray(clusters.channels, dtype=np.int64)
        mlapdv = np.asarray(channels.mlapdv, dtype=np.float64)[peak_channel] / 1e6
        atlas_id = np.asarray(channels.brainLocationIds_ccf_2017, dtype=np.int64)[peak_channel]
        cluster_id = np.asarray(metrics["cluster_id"], dtype=np.int64)

        table = pd.DataFrame(
            {
                "probe_name": probe_name,
                "cluster_id": cluster_id,
                "cluster_uuid": np.asarray(clusters.uuids, dtype=object).astype(str),
                "label": np.asarray(metrics["label"], dtype=np.float64),
                "firing_rate": np.asarray(metrics["firing_rate"], dtype=np.float64),
                "depths": np.asarray(clusters.depths, dtype=np.float64),
                "x": mlapdv[:, 0],
                "y": mlapdv[:, 1],
                "z": mlapdv[:, 2],
                "atlas_id": atlas_id,
                "peak_channel": peak_channel,
            },
            index=pd.Index([f"{probe_name}_{c}" for c in cluster_id], name="unit_id"),
        )
        tables.append(table)

        # spikes.clusters indexes the cluster tables, it is not the cluster_id.
        index = np.asarray(spike.clusters, dtype=np.int64)
        if index.size and (index.min() < 0 or index.max() >= n_units):
            raise ValueError(
                f"{eid} {probe_name}: spikes.clusters out of range for {n_units} clusters"
            )
        times = np.asarray(spike.times, dtype=np.float64)
        order = np.argsort(index, kind="stable")
        bounds = np.searchsorted(index[order], np.arange(n_units + 1))
        by_unit = times[order]
        for i, unit_id in enumerate(table.index):
            spikes[unit_id] = by_unit[bounds[i] : bounds[i + 1]]
    return pd.concat(tables), spikes


def _trials(one, eid: str, revision: str) -> pd.DataFrame:
    raw = one.load_object(eid, "trials", revision=revision)
    intervals = np.asarray(raw["intervals"], dtype=np.float64)
    out = pd.DataFrame({"intervals_0": intervals[:, 0], "intervals_1": intervals[:, 1]})
    for field in TRIAL_FIELDS:
        if field in out or field not in raw:
            continue
        out[field] = np.asarray(raw[field], dtype=np.float64)
    for extra in ("rewardVolume", "quiescencePeriod"):
        if extra in raw:
            out[extra] = np.asarray(raw[extra], dtype=np.float64)
    return out[[f for f in TRIAL_FIELDS if f in out] + [c for c in out if c not in TRIAL_FIELDS]]


def _behaviour(one, eid: str) -> tuple[dict[str, TimeSeries], dict[str, str]]:
    missing = {f: _BEHAVIOUR_REASON for f in BEHAVIOUR_FIELDS if f != "wheel"}
    try:
        wheel = one.load_object(eid, "wheel")
    except FileNotFoundError:
        return {}, {**missing, "wheel": f"{eid} has no wheel object in ONE"}
    series = TimeSeries(
        np.asarray(wheel.timestamps, dtype=np.float64), np.asarray(wheel.position, dtype=np.float64)
    )
    return {"wheel": series}, missing


def _time_bounds(spikes, trials: pd.DataFrame, behaviour: dict[str, TimeSeries]) -> tuple:
    firsts = [t[0] for t in spikes.values() if t.size]
    lasts = [t[-1] for t in spikes.values() if t.size]
    times = trials[[f for f in TRIAL_TIME_FIELDS if f in trials]].to_numpy(np.float64)
    times = times[np.isfinite(times)]
    if times.size:
        firsts.append(times.min())
        lasts.append(times.max())
    for series in behaviour.values():
        if series.timestamps.size:
            firsts.append(series.timestamps[0])
            lasts.append(series.timestamps[-1])
    return float(min(firsts)), float(max(lasts))


def _declare_missing(table: pd.DataFrame, group: str, fields, missing: dict) -> pd.DataFrame:
    for field in fields:
        if field not in table:
            missing.setdefault(f"{group}.{field}", "not provided by ONE for this session")
        elif table[field].isna().all():
            missing[f"{group}.{field}"] = "all values are NaN in ONE"
    return table.drop(columns=[f for f in fields if f"{group}.{f}" in missing and f in table])


def load_session_one(
    eid: str,
    one,
    *,
    sorter_revision: str = SORTER_REVISION,
    trials_revision: str = TRIALS_REVISION,
) -> Session:
    """Read one IBL session through ONE into a Session, at pinned revisions.

    All units are returned, not just IBL's good ones: filter on `units.label` (1.0 is
    IBL's "good" label). Times are seconds on the session clock, the wheel is the raw
    encoder position (radians), and unit x/y/z are metres relative to bregma.
    """
    probes = _probe_collections(one, eid)
    units, spikes = _units_and_spikes(one, eid, probes, sorter_revision)
    trials = _trials(one, eid, trials_revision)
    behaviour, missing_behaviour = _behaviour(one, eid)

    missing = {"units.acronym": _ACRONYM_REASON}
    missing.update({f"behaviour.{f}": why for f, why in missing_behaviour.items()})
    trials = _declare_missing(trials, "trials", TRIAL_FIELDS, missing)
    units = _declare_missing(units, "units", UNIT_FIELDS, missing)

    present = {f"trials.{f}" for f in TRIAL_FIELDS if f in trials}
    present |= {f"units.{f}" for f in UNIT_FIELDS if f in units}
    present |= {f"behaviour.{f}" for f in BEHAVIOUR_FIELDS if f in behaviour}

    return Session(
        eid=eid,
        time_bounds=_time_bounds(spikes, trials, behaviour),
        spikes=spikes,
        units=units[[*[f for f in UNIT_FIELDS if f in units], *_UNIT_EXTRAS]],
        trials=trials,
        behaviour=behaviour,
        available=Capabilities(present=frozenset(present), missing=missing),
    )
