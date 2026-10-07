"""Sliding windows over normalised counts, kept inside the split.  [R1, R2]

`window_plan(split, context_bins=..., stride_bins=...)` runs `assert_split_valid` once,
so no window comes from a split whose gap is shorter than the context. For a session
and a partition, the plan allows a span of bins:
- within_session: the partition's block, so train and test windows share no bin and
  sit at least max(context, 2 s) apart;
- any other kind: the whole binned session, if the session is in that partition.

A window ending at bin e covers bins e - context_bins + 1 .. e; aligning a target to
it is the job of targets/. `ends` gives every stride_bins-th end from the first full
window in the span. `extract` also takes ends chosen elsewhere, and refuses any window
that leaves the span. Build plans with `window_plan`, not `WindowPlan(...)`, so the
guard runs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from unitwave.preprocess.binning import BinnedSpikes
from unitwave.splits.guards import assert_split_valid
from unitwave.splits.registry import Split, bin_range


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class WindowPlan:
    split: Split
    context_bins: int
    stride_bins: int

    def span(self, binned: BinnedSpikes, partition: str) -> tuple[int, int]:
        """First and last absolute bin (inclusive) a window for this session may use."""
        eid = binned.eid
        if eid not in self.split.partitions.get(partition, ()):
            raise ValueError(f"{eid} is not in {partition}")
        if binned.fingerprint != self.split.preproc["fingerprint"]:
            raise ValueError(f"{eid}: binned with another preprocessing fingerprint than the split")
        first, last = binned.first_bin, binned.first_bin + binned.n_bins - 1
        if self.split.kind != "within_session":
            return first, last
        block = self.split.sessions[eid]["blocks"][partition]
        b0, b1 = bin_range(block, self.split.preproc["bin_ms"])
        if b0 < first or b1 > last:
            raise ValueError(
                f"{eid}: {partition} block bins {b0}..{b1} are outside the binned data "
                f"{first}..{last}"
            )
        return b0, b1

    def ends(self, binned: BinnedSpikes, partition: str) -> np.ndarray:
        """(n_windows,) int64 absolute end bins, stride_bins apart."""
        first, last = self.span(binned, partition)
        return np.arange(first + self.context_bins - 1, last + 1, self.stride_bins, dtype=np.int64)

    def extract(
        self,
        z: np.ndarray,
        binned: BinnedSpikes,
        partition: str,
        ends: np.ndarray | None = None,
    ) -> np.ndarray:
        """(n_windows, n_units, context_bins) float32 windows of z.

        z: (n_units, n_bins), aligned with binned (e.g. Normalizer.transform(binned)).
        ends: absolute end bins; by default `self.ends(binned, partition)`.
        """
        if z.shape != binned.counts.shape:
            raise ValueError(f"z has shape {z.shape}, binned counts {binned.counts.shape}")
        first, last = self.span(binned, partition)
        ends = self.ends(binned, partition) if ends is None else np.asarray(ends, np.int64)
        context = self.context_bins
        outside = (ends - context + 1 < first) | (ends > last)
        if outside.any():
            raise ValueError(
                f"{binned.eid}: the window ending at bin {ends[outside][0]} reaches outside "
                f"the {partition} span {first}..{last}"
            )
        # (n_all_windows, n_units, context) view; indexing copies only the chosen ones.
        view = sliding_window_view(z, context, axis=1).transpose(1, 0, 2)
        x = view[ends - context + 1 - binned.first_bin].astype(np.float32, copy=False)
        assert x.shape == (len(ends), z.shape[0], context)
        return x


def window_plan(
    split: Split,
    *,
    context_bins: int,
    stride_bins: int,
    manifest_provenance: dict | None = None,
    preproc_fingerprint: str | None = None,
) -> WindowPlan:
    """Validate the split for this context, then plan windows over it.

    manifest_provenance / preproc_fingerprint are passed to `assert_split_valid`.
    """
    context_bins = _positive_int("context_bins", context_bins)
    stride_bins = _positive_int("stride_bins", stride_bins)
    assert_split_valid(
        split,
        context_bins=context_bins,
        manifest_provenance=manifest_provenance,
        preproc_fingerprint=preproc_fingerprint,
    )
    return WindowPlan(split, context_bins, stride_bins)


def unwindow(windows: np.ndarray, ends: np.ndarray, first_bin: int, n_bins: int) -> np.ndarray:
    """Put (n_windows, n_units, context) windows back on the session's bins.

    Returns (n_units, n_bins) float32, NaN where no window reaches. Where windows
    overlap they hold the same values, so the order they are written in does not matter.
    """
    n_windows, n_units, context = windows.shape
    if len(ends) != n_windows:
        raise ValueError(f"{len(ends)} ends for {n_windows} windows")
    out = np.full((n_units, n_bins), np.nan, dtype=np.float32)
    for window, end in zip(windows, ends):
        start = int(end) - context + 1 - first_bin
        if start < 0 or start + context > n_bins:
            raise ValueError(f"the window ending at bin {end} is outside the session's bins")
        out[:, start : start + context] = window
    return out
