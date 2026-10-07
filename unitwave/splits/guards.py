"""assert_split_valid: the first call of every training and evaluation entry point.  [R1, R2]

It raises and never warns. It checks (docs/SPLITS_AND_LEAKAGE.md, "The guard API"):
- the manifest and preprocessing the split was built against are the current ones;
- partitions are known and non-empty, and include train and test;
- across partitions (calibration included), animal-level splits share no subject,
  lab-level splits share no lab, and non-temporal splits share no session;
- within a session, the blocks share no bin and are at least max(context, 2 s) apart;
- every listed trial is in one partition only and lies wholly inside its block;
- for a held-out region, by the split's recorded unit counts, test sessions have at
  least 20% of their units in the region and train sessions none that is or could be;
- for a held-out configuration, the recorded cutoffs are each lab's percentiles of
  the recorded per-session unit counts, and each session is on the right side of its
  lab's cutoffs.
"""

import math
from itertools import combinations

import numpy as np

from unitwave.data.manifest import manifest_versions
from unitwave.preprocess.binning import load_preproc_config
from unitwave.splits.registry import (
    GROUP_KINDS,
    KINDS,
    MIN_GAP_S,
    MIN_REGION_FRACTION,
    config_cutoffs,
    PARTITIONS,
    Split,
    bin_range,
)

_GROUP_KEYS = {kind: key for key, kind in GROUP_KINDS.items()}


def assert_split_valid(
    split: Split,
    *,
    context_bins: int,
    manifest_provenance: dict | None = None,
    preproc_fingerprint: str | None = None,
) -> None:
    """Raise ValueError if the split could leak or does not match the current data.

    context_bins: the model's context length, in bins of the split's bin width.
    manifest_provenance / preproc_fingerprint: what to compare against; by default,
    this code's manifest versions and the fingerprint of configs/preprocess.yaml.
    """
    if isinstance(context_bins, bool) or not isinstance(context_bins, int) or context_bins < 1:
        raise ValueError(f"context_bins must be a positive integer, got {context_bins!r}")
    _check_versions(split, manifest_provenance, preproc_fingerprint)
    _check_partitions(split)
    if split.kind == "within_session":
        for eid in sorted(set().union(*split.partitions.values())):
            _check_session_blocks(split, eid, context_bins)
        return
    if split.kind == "leave_one_block_out":
        for eid in sorted(set().union(*split.partitions.values())):
            _check_session_folds(split, eid, context_bins)
        return
    pairs = list(combinations(split.partitions, 2))
    for a, b in pairs:
        shared = sorted(set(split.partitions[a]) & set(split.partitions[b]))
        if shared:
            raise ValueError(f"session {shared[0]} is in both {a} and {b}")
    if split.kind == "held_out_region":
        _check_region(split)
    if split.kind == "held_out_config":
        _check_config(split)
    key = _GROUP_KEYS.get(split.kind)
    if key is None:
        return
    for a, b in pairs:
        values = [{split.sessions[e][key] for e in split.partitions[p]} for p in (a, b)]
        shared = sorted(values[0] & values[1])
        if shared:
            raise ValueError(f"{key} {shared[0]} is in both {a} and {b}")


def _check_region(split: Split) -> None:
    """From the recorded unit counts: test sessions are mostly in the region, and train
    sessions have no unit that is or could be in it."""
    region = split.params["region"]
    for eid in split.partitions["test"]:
        s = split.sessions[eid]
        if not s["n_units"] or s["n_in_region"] / s["n_units"] < MIN_REGION_FRACTION:
            raise ValueError(
                f"test session {eid} has {s['n_in_region']} of {s['n_units']} units in "
                f"{region}, under {MIN_REGION_FRACTION:.0%}"
            )
    for eid in split.partitions["train"]:
        s = split.sessions[eid]
        if s["n_in_region"] or s["n_possibly_in_region"]:
            raise ValueError(
                f"train session {eid} has {s['n_in_region']} units in {region} and "
                f"{s['n_possibly_in_region']} possibly in it"
            )


def _check_config(split: Split) -> None:
    """Recompute each lab's cutoffs from every session's recorded unit count, then check
    each test session is below its lab's test cutoff and each train session at or above
    its lab's train cutoff.
    """
    recomputed = config_cutoffs(split.sessions)
    if recomputed != split.params["cutoffs"]:
        raise ValueError(
            "recorded cutoffs are not the within-lab percentiles of the recorded unit counts"
        )
    for eid in split.partitions["test"]:
        s = split.sessions[eid]
        cutoff = recomputed[s["lab"]]["test_below_units"]
        if not s["n_units"] < cutoff:
            raise ValueError(
                f"test session {eid} has {s['n_units']} units, not under {s['lab']}'s "
                f"cutoff {cutoff}"
            )
    for eid in split.partitions["train"]:
        s = split.sessions[eid]
        cutoff = recomputed[s["lab"]]["train_from_units"]
        if not s["n_units"] >= cutoff:
            raise ValueError(
                f"train session {eid} has {s['n_units']} units, under {s['lab']}'s cutoff {cutoff}"
            )


def _check_versions(split: Split, provenance: dict | None, fingerprint: str | None) -> None:
    expected = manifest_versions() if provenance is None else provenance
    for key in ("manifest_version", "sources"):
        if split.manifest.get(key) != expected.get(key):
            raise ValueError(
                f"split was built against manifest {key} {split.manifest.get(key)!r}, "
                f"current is {expected.get(key)!r}; rebuild the split"
            )
    if fingerprint is None:
        fingerprint = load_preproc_config().fingerprint()
    if split.preproc.get("fingerprint") != fingerprint:
        raise ValueError(
            f"split was built for preprocessing fingerprint {split.preproc.get('fingerprint')}, "
            f"current is {fingerprint}; rebuild the split"
        )


def _check_partitions(split: Split) -> None:
    if split.kind not in KINDS:
        raise ValueError(f"unknown split kind {split.kind!r}")
    unknown = sorted(set(split.partitions) - set(PARTITIONS))
    if unknown or not {"train", "test"} <= set(split.partitions):
        raise ValueError(
            f"partitions {sorted(split.partitions)}; need train and test, and only {PARTITIONS}"
        )
    for name, eids in split.partitions.items():
        if not eids:
            raise ValueError(f"partition {name} is empty")
        if len(set(eids)) != len(eids):
            raise ValueError(f"partition {name} lists a session more than once")
        missing = sorted(set(eids) - set(split.sessions))
        if missing:
            raise ValueError(f"partition {name} lists {missing[0]}, which has no session record")


def _check_session_folds(split: Split, eid: str, context_bins: int) -> None:
    """leave_one_block_out: per fold, the test span and each training interval share no
    bin and sit at least max(context, 2 s) apart; trials lie inside their partition's
    intervals and on one side only; the test and training trials each hold both block
    labels (probabilityLeft 0.2 and 0.8); across folds, a trial is tested at most once."""
    record = split.sessions[eid]
    bin_ms = split.preproc["bin_ms"]
    need = max(context_bins, math.ceil(MIN_GAP_S * 1000 / bin_ms))
    intervals = record["trial_intervals"]
    if "trial_prior" not in record:
        raise ValueError(
            f"{eid}: the split has no trial_prior; it predates the block-pair fix and "
            "cannot be checked for one-label test sets"
        )
    prior = np.asarray(record["trial_prior"], dtype=np.float64)
    tested: set[int] = set()
    for k, fold in enumerate(record.get("folds", [])):
        for partition, listed in fold["trials"].items():
            labels = set(prior[list(listed)].tolist()) & {0.2, 0.8}
            if labels != {0.2, 0.8}:
                raise ValueError(
                    f"{eid} fold {k}: the {partition} trials hold only block label(s) "
                    f"{sorted(labels)}; each fold needs both 0.2 and 0.8"
                )
        t0, t1 = bin_range(fold["blocks"]["test"], bin_ms)
        for block in fold["blocks"]["train"]:
            a0, a1 = bin_range(block, bin_ms)
            if a0 <= t1 and t0 <= a1:
                raise ValueError(
                    f"{eid} fold {k}: a training interval shares bins with the test block"
                )
            gap = max(t0 - a1, a0 - t1) - 1
            if gap < need:
                raise ValueError(
                    f"{eid} fold {k}: {gap} bins between a training interval and the test "
                    f"block; the gap must be at least max(context {context_bins}, "
                    f"{MIN_GAP_S} s) = {need} bins"
                )
        train, test = set(fold["trials"]["train"]), set(fold["trials"]["test"])
        if train & test:
            raise ValueError(f"{eid} fold {k}: trial {min(train & test)} is in both train and test")
        if tested & test:
            raise ValueError(f"{eid}: trial {min(tested & test)} is tested in more than one fold")
        tested |= test
        spans = {"train": fold["blocks"]["train"], "test": [fold["blocks"]["test"]]}
        for partition, listed in fold["trials"].items():
            for i in listed:
                s0, s1 = intervals[i]
                if not any(b0 <= s0 <= s1 <= b1 for b0, b1 in spans[partition]):
                    raise ValueError(
                        f"{eid} fold {k}: trial {i} is in {partition} but outside its intervals"
                    )


def _check_session_blocks(split: Split, eid: str, context_bins: int) -> None:
    record = split.sessions[eid]
    names = [p for p in split.partitions if eid in split.partitions[p]]
    blocks, trials = record.get("blocks", {}), record.get("trials", {})
    if set(blocks) != set(names) or set(trials) != set(names):
        raise ValueError(
            f"{eid}: blocks {sorted(blocks)} and trials {sorted(trials)} "
            f"must match its partitions {sorted(names)}"
        )

    bin_ms = split.preproc["bin_ms"]
    need = max(context_bins, math.ceil(MIN_GAP_S * 1000 / bin_ms))
    bins = {p: bin_range(blocks[p], bin_ms) for p in names}
    for p in names:
        if blocks[p][1] < blocks[p][0]:
            raise ValueError(f"{eid}: {p} block {blocks[p]} ends before it starts")
    for a, b in combinations(names, 2):
        (a0, a1), (b0, b1) = bins[a], bins[b]
        if a0 <= b1 and b0 <= a1:
            raise ValueError(f"{eid}: the {a} and {b} blocks share bins")
        gap = max(b0 - a1, a0 - b1) - 1
        if gap < need:
            raise ValueError(
                f"{eid}: {gap} bins between the {a} and {b} blocks; the gap must be at least "
                f"max(context {context_bins}, {MIN_GAP_S} s) = {need} bins"
            )

    intervals = record["trial_intervals"]
    seen: dict[int, str] = {}
    for p in names:
        start, stop = blocks[p]
        for i in trials[p]:
            if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(intervals):
                raise ValueError(f"{eid}: {p} lists trial {i!r}, not one of its trials")
            if i in seen:
                raise ValueError(f"{eid}: trial {i} is in both {seen[i]} and {p}")
            seen[i] = p
            t0, t1 = intervals[i]
            if not start <= t0 <= t1 <= stop:
                raise ValueError(
                    f"{eid}: trial {i} [{t0}, {t1}] is in {p} but not wholly inside "
                    f"its block [{start}, {stop}]"
                )
