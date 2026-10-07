"""Per-cluster files beyond spike times: mean waveforms and IBL's per-criterion metrics.

Read from local files only, never downloaded:
- **IBL:** the ONE cache, at the spike-sorting revision the ONE backend pins
  (`alf/<probe>/pykilosort/#2024-05-06#`). `clusters.waveforms.npy` is each cluster's
  mean waveform in volts, (n_clusters, n_samples, n_channels), on the channels in
  `clusters.waveformsChannels.npy`. `clusters.metrics.pqt` holds the metrics behind
  IBL's `label`. Rows of `clusters.*` are cluster ids.
- **Phy:** the cluster's spike-weighted mean template, unwhitened with
  `whitening_mat_inv.npy`. Templates are in Kilosort's template units, not volts, so
  the unit says so.

A file that is not there is a plain refusal naming it, never a default.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from unitwave.data.backends.one_backend import SORTER_COLLECTION, SORTER_REVISION
from unitwave.data.backends.phy import read_params

# IBL's label is the mean of three criteria (brainbox.metrics.single_units.compute_labels,
# METRICS_PARAMS): fixed parts of IBL's definition, reproduced to explain a stored label.
IBL_MAX_CONFIDENCE = 90.0  # sliding RP test passed with at least this confidence (%)
IBL_NOISE_CUTOFF = 5.0  # noise_cutoff below this
IBL_AMP_MEDIAN_UV = 50.0  # median amplitude above this, µV
# IBL's sliding RP settings (METRICS_PARAMS RPslide_thresh, RPmax_confidence).
IBL_RP_CONTAMINATION = 0.1
IBL_RP_ALPHA = 1 - IBL_MAX_CONFIDENCE / 100
_IBL = "the ONE cache's spike sorting for this probe"
_PHY_WAVEFORM_FILES = (
    "templates.npy",
    "spike_templates.npy",
    "spike_clusters.npy",
    "whitening_mat_inv.npy",
)


@dataclass(frozen=True)
class Waveform:
    """samples_uv: (n_samples, n_channels) in `unit`; channels: (n_channels,) probe
    channel indices; source names the file it came from."""

    samples_uv: np.ndarray
    channels: np.ndarray
    sample_rate: float
    unit: str
    source: str

    @property
    def peak(self) -> int:
        """Column of the channel with the largest peak-to-peak amplitude."""
        return int(np.ptp(self.samples_uv, axis=0).argmax())


def ibl_session_folder(one_cache_root, sessions: pd.DataFrame, eid: str) -> Path | None:
    """The session's `alf` folder in the ONE cache, from the manifest's lab, subject,
    date and session number; None when the session is not in the manifest or not
    downloaded."""
    row = sessions[sessions["eid"] == eid]
    if len(row) != 1:
        return None
    r = row.iloc[0]
    folder = (
        Path(one_cache_root)
        / r["lab"]
        / "Subjects"
        / r["subject"]
        / str(r["date"])[:10]
        / f"{int(r['session_number']):03d}"
        / "alf"
    )
    return folder if folder.is_dir() else None


def ibl_sorting_folder(alf: Path, probe: str) -> Path | None:
    """The probe's spike-sorting folder at the pinned revision, if downloaded."""
    folder = Path(alf) / probe / SORTER_COLLECTION / f"#{SORTER_REVISION}#"
    return folder if folder.is_dir() else None


def _need(folder: Path, what: str, *names: str) -> None:
    missing = [n for n in names if not (Path(folder) / n).exists()]
    if missing:
        listed = ", ".join(missing[:-1]) + (" or " if len(missing) > 1 else "") + missing[-1]
        raise ValueError(f"{what} has no {listed} (in {folder})")


def ibl_waveform(folder: Path, cluster_id: int, sample_rate: float = 30000.0) -> Waveform:
    """IBL's mean waveform for one cluster, in µV. IBL's probes sample at 30 kHz."""
    _need(folder, _IBL, "clusters.waveforms.npy", "clusters.waveformsChannels.npy")
    waveforms = np.load(Path(folder) / "clusters.waveforms.npy", mmap_mode="r")
    channels = np.load(Path(folder) / "clusters.waveformsChannels.npy", mmap_mode="r")
    assert waveforms.ndim == 3 and channels.shape == (waveforms.shape[0], waveforms.shape[2])
    if not 0 <= cluster_id < waveforms.shape[0]:
        raise ValueError(f"no waveform for cluster {cluster_id} in {folder}")
    return Waveform(
        np.asarray(waveforms[cluster_id], np.float64) * 1e6,
        np.asarray(channels[cluster_id]),
        sample_rate,
        "µV",
        "clusters.waveforms.npy (IBL)",
    )


def ibl_criteria(folder: Path) -> pd.DataFrame:
    """(n_clusters, ...) indexed by cluster_id: each criterion of IBL's label, its value
    and whether it passes, and the stored label. A missing value fails, as in IBL."""
    _need(folder, _IBL, "clusters.metrics.pqt")
    m = pd.read_parquet(Path(folder) / "clusters.metrics.pqt").set_index("cluster_id")
    amp_uv = m["amp_median"] * 1e6
    return pd.DataFrame(
        {
            "max_confidence": m["max_confidence"],
            "refractory_pass": m["max_confidence"] >= IBL_MAX_CONFIDENCE,
            "noise_cutoff": m["noise_cutoff"],
            "noise_cutoff_pass": m["noise_cutoff"] < IBL_NOISE_CUTOFF,
            "amp_median_uv": amp_uv,
            "amplitude_pass": amp_uv > IBL_AMP_MEDIAN_UV,
            "label": m["label"],
        }
    )


def phy_waveform(folder: Path, cluster_id: int) -> Waveform:
    """The cluster's mean template over its spikes, unwhitened, in template units."""
    folder = Path(folder)
    _need(folder, "this Phy folder", *_PHY_WAVEFORM_FILES)
    templates = np.load(folder / "templates.npy", mmap_mode="r")  # (n_templates, n_t, n_ch)
    spike_templates = np.load(folder / "spike_templates.npy").ravel()  # (n_spikes,)
    spike_clusters = np.load(folder / "spike_clusters.npy").ravel()  # (n_spikes,)
    w_inv = np.load(folder / "whitening_mat_inv.npy")  # (n_ch, n_ch)
    assert templates.ndim == 3 and spike_templates.shape == spike_clusters.shape
    assert w_inv.shape == (templates.shape[2], templates.shape[2])
    used = spike_templates[spike_clusters == cluster_id]
    if used.size == 0:
        raise ValueError(f"cluster {cluster_id} has no spikes in {folder}")
    ids, counts = np.unique(used, return_counts=True)
    mean = np.tensordot(counts / counts.sum(), np.asarray(templates[ids], np.float64), axes=1)
    return Waveform(
        mean @ w_inv,
        np.arange(templates.shape[2]),
        float(read_params(folder)["sample_rate"]),
        "template units",
        "templates.npy (Kilosort)",
    )
