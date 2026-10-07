"""Write a folder in Kilosort/Phy's output format, for tests.

Mirrors what Kilosort writes: spike_times.npy and spike_templates.npy as (n_spikes, 1)
integers, tab-separated label files with a header row, and a params.py of literals.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def write_phy_folder(
    folder: Path,
    spike_samples: np.ndarray,
    spike_clusters: np.ndarray,
    sample_rate: float = 30000.0,
    *,
    group: dict | None = None,
    ks_label: dict | None = None,
    templates: np.ndarray | None = None,
    spike_templates: np.ndarray | None = None,
    channel_positions: np.ndarray | None = None,
    params_extra: str = "",
) -> Path:
    """group, ks_label: cluster_id -> label, written to cluster_group.tsv / cluster_KSLabel.tsv."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "spike_times.npy", np.asarray(spike_samples, np.uint64).reshape(-1, 1))
    np.save(folder / "spike_clusters.npy", np.asarray(spike_clusters, np.uint32))
    params = (
        "dat_path = r'continuous.dat'\nn_channels_dat = 385\ndtype = 'int16'\noffset = 0\n"
        f"sample_rate = {sample_rate!r}\nhp_filtered = False\n"
    )
    (folder / "params.py").write_text(params + params_extra)
    for name, column, labels in (
        ("cluster_group", "group", group),
        ("cluster_KSLabel", "KSLabel", ks_label),
    ):
        if labels is not None:
            table = pd.DataFrame({"cluster_id": list(labels), column: list(labels.values())})
            table.to_csv(folder / f"{name}.tsv", sep="\t", index=False)
    if templates is not None:
        np.save(folder / "templates.npy", np.asarray(templates, np.float32))
        np.save(
            folder / "spike_templates.npy", np.asarray(spike_templates, np.uint32).reshape(-1, 1)
        )
        np.save(folder / "channel_positions.npy", np.asarray(channel_positions, np.float64))
    return folder
