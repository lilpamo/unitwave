"""Task-period firing rates of every unit in the BWM release, precomputed.  [R6]

Builders that work from release metadata (held_out_region, held_out_config) apply
unit QC to the release's units table without loading sessions, so they need every
unit's task-period rate. Computing it decodes every spike shard (about a minute on 6
processes), so `python -m unitwave.cli.build_task_rates` builds it once into

    <derived>/bwm_ephys-<release version>/task_rates-v<TASK_RATES_VERSION>.parquet

with a provenance.json beside it. `load_task_rates` refuses a table built from
another release or by other code. The rates use qc.units.task_period and in_task, the
definition `apply_unit_qc` applies to a loaded session, and the same shard decoder
as the BWM backend, so both paths give identical rates.
"""

import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from unitwave.data.backends import bwm_compressed
from unitwave.data.backends.bwm_compressed import _decode_spike_shard
from unitwave.qc.units import TASK_RATE, in_task, task_period

# Bump when this module's output changes for the same release.
TASK_RATES_VERSION = 1
_COLUMNS = ["pid", "eid", "cluster_id", "n_task_spikes", "task_start", "task_end", TASK_RATE]


def table_dir(derived_root: str | os.PathLike) -> Path:
    return Path(derived_root) / f"{bwm_compressed.DATASET_NAME}-{bwm_compressed.DATASET_VERSION}"


def _provenance(n_units: int) -> dict:
    return {
        "dataset": bwm_compressed.DATASET_NAME,
        "dataset_version": bwm_compressed.DATASET_VERSION,
        "task_rates_version": TASK_RATES_VERSION,
        "task_period": "first trial's intervals_0 to last trial's intervals_1, inclusive",
        "n_units": n_units,
    }


def probe_task_rates(ephys_root: str | os.PathLike, pid: str, eid: str, period) -> pd.DataFrame:
    """One insertion's units: spikes inside the task period and the rate."""
    t0, t1 = period
    times, cluster_ids, local = _decode_spike_shard(Path(ephys_root) / "spikes" / pid)
    counts = np.bincount(local[in_task(times, t0, t1)], minlength=len(cluster_ids))
    return pd.DataFrame(
        {
            "pid": pid,
            "eid": eid,
            "cluster_id": cluster_ids,
            "n_task_spikes": counts.astype(np.int64),
            "task_start": t0,
            "task_end": t1,
            TASK_RATE: counts / (t1 - t0),
        }
    )


def _probe_job(args) -> pd.DataFrame:
    return probe_task_rates(*args)


def build_task_rates(
    ephys_root: str | os.PathLike, derived_root: str | os.PathLike, workers: int = 6
) -> Path:
    """Decode every shard of the release and write the table; returns its path."""
    ephys_root = Path(ephys_root)
    bwm_compressed._check_version(ephys_root)
    trials = pd.read_parquet(
        ephys_root / "metadata/trials.parquet", columns=["eid", "intervals_0", "intervals_1"]
    )
    periods = {eid: task_period(t) for eid, t in trials.groupby("eid")}
    insertions = pd.read_parquet(ephys_root / "metadata/insertions.parquet", columns=["pid", "eid"])
    jobs = [(ephys_root, pid, eid, periods[eid]) for pid, eid in insertions.itertuples(index=False)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(_probe_job, jobs, chunksize=4))
    table = pd.concat(frames, ignore_index=True).sort_values(["pid", "cluster_id"])
    table = table[_COLUMNS].reset_index(drop=True)

    out = table_dir(derived_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-task-rates-", dir=out.parent))
    try:
        if out.exists():
            shutil.copytree(out, tmp, dirs_exist_ok=True)
        name = f"task_rates-v{TASK_RATES_VERSION}"
        table.to_parquet(tmp / f"{name}.parquet", index=False)
        (tmp / f"{name}.provenance.json").write_text(json.dumps(_provenance(len(table)), indent=2))
        if out.exists():
            shutil.rmtree(out)
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out / f"task_rates-v{TASK_RATES_VERSION}.parquet"


def load_task_rates(derived_root: str | os.PathLike) -> pd.DataFrame:
    """The table for the pinned release and this code, or an error saying how to build it."""
    path = table_dir(derived_root) / f"task_rates-v{TASK_RATES_VERSION}.parquet"
    provenance_path = path.with_suffix(".provenance.json")
    if not path.exists() or not provenance_path.exists():
        raise FileNotFoundError(
            f"{path} not found; build it with `python -m unitwave.cli.build_task_rates`"
        )
    table = pd.read_parquet(path)
    expected = _provenance(len(table))
    found = json.loads(provenance_path.read_text())
    if found != expected:
        raise ValueError(f"{path}: provenance {found} does not match {expected}; rebuild it")
    return table


def release_units(ephys_root: str | os.PathLike, derived_root: str | os.PathLike) -> pd.DataFrame:
    """The release's units table with each unit's task_firing_rate joined on."""
    ephys_root = Path(ephys_root)
    bwm_compressed._check_version(ephys_root)
    units = pd.read_parquet(ephys_root / "metadata/units.parquet")
    rates = load_task_rates(derived_root)[["pid", "cluster_id", TASK_RATE]]
    merged = units.merge(rates, on=["pid", "cluster_id"], how="left", validate="one_to_one")
    if merged[TASK_RATE].isna().any() or len(rates) != len(units):
        raise ValueError("task-rate table and units table list different units; rebuild it")
    return merged
