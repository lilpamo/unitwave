"""Train / calibration / test partitions, saved as JSON with a hash.  [R1, R2, R7]

Every partition used for training or evaluation is built here and passed around as a
`Split`. Models never receive raw session lists. A split records the manifest
provenance and preprocessing fingerprint it was built against. Its hash covers all of
its content, so `load_split` refuses an edited or corrupted file, and
`guards.assert_split_valid` refuses a split built against other data or preprocessing.

Kinds (docs/SPLITS_AND_LEAKAGE.md):
- held_out_animal / held_out_lab: every session of an animal (lab) is in one partition.
  Calibration, when requested, is its own set of animals (labs).
- held_out_session: one session of each animal with >= 2 sessions is test; the rest train.
- within_session: per session, early trials train and late trials test, as two
  contiguous time blocks separated by a gap. Trials in the gap belong to neither.
- held_out_region: sessions with >= 20% of their QC-passing units in Beryl region R
  are test; sessions with no QC-passing unit that is or could be in R are train; the
  rest are dropped.
- held_out_config: sessions with fewer QC-passing units than their lab's 10th
  percentile are test; sessions at or above their lab's 25th percentile are train.
- leave_one_block_out (block only; adopted after the first table): within each
  session, the biased blocks are taken in adjacent pairs (one 0.2 and one 0.8 block;
  with an odd count the last block joins the final pair). Each pair is held out once
  as a fold's test set, and the fold trains on the trials before and after it,
  leaving out any trial within a gap of the pair. The first version held out single
  blocks, so each test set held one label; it inverted decoders with no signal (see
  docs/NEGATIVE_RESULTS.md).

Random choices use numpy's default_rng with the required seed. The saved file, not
the seed, is the record: a different numpy could draw differently from the same seed.
"""

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from unitwave.data.manifest import Manifest
from unitwave.preprocess.binning import PreprocConfig
from unitwave.qc.units import unit_qc

SPLIT_FORMAT_VERSION = 1
PARTITIONS = ("train", "calibration", "test")
GROUP_KINDS = {"subject": "held_out_animal", "lab": "held_out_lab"}
KINDS = (
    *GROUP_KINDS.values(),
    "held_out_session",
    "within_session",
    "held_out_region",
    "held_out_config",
    "leave_one_block_out",
)
# leave_one_block_out holds out adjacent pairs of biased blocks, so every test set holds
# both block labels, and needs this many biased blocks, so every fold's training data
# does too.
LOBO_BLOCKS_PER_FOLD = 2
MIN_LOBO_BLOCKS = 4
# Behavioural autocorrelation outlives the neural context, so temporal splits keep at
# least this much time between train and test whatever the context.
MIN_GAP_S = 2.0
# docs/SPLITS_AND_LEAKAGE.md: a held-out-region test session has at least this
# fraction of its QC-passing units in the region.
MIN_REGION_FRACTION = 0.2
# The user's definition of a non-standard configuration (2026-09-28): a session is
# held-out-config test when its QC-passing unit count is below the 10th percentile of
# its lab's sessions, and train when at or above its lab's 25th. Percentiles are
# within lab because across the release the bottom 10% is mostly single-probe labs,
# which would make the split measure lab shift. numpy's default (linear) percentiles.
CONFIG_TEST_PERCENTILE = 10
CONFIG_TRAIN_PERCENTILE = 25


def config_cutoffs(sessions: Mapping[str, dict]) -> dict:
    """lab -> its cutoffs, from eid -> {"lab", "n_units"}.

    Each lab gets test_below_units and train_from_units (its percentiles), the same
    cutoffs as whole unit counts (test_max_units, train_min_units), and n_sessions.
    """
    by_lab: dict[str, list[int]] = {}
    for record in sessions.values():
        by_lab.setdefault(record["lab"], []).append(record["n_units"])
    cutoffs = {}
    for lab, counts in sorted(by_lab.items()):
        test_below, train_from = np.percentile(
            np.asarray(counts, dtype=np.float64),
            [CONFIG_TEST_PERCENTILE, CONFIG_TRAIN_PERCENTILE],
        )
        cutoffs[lab] = {
            "n_sessions": len(counts),
            "test_below_units": float(test_below),
            "train_from_units": float(train_from),
            "test_max_units": math.ceil(test_below) - 1,
            "train_min_units": math.ceil(train_from),
        }
    return cutoffs


@dataclass(frozen=True)
class Split:
    """One partition of sessions (and, within a session, of time and trials).

    partitions: partition name -> sorted eids.
    sessions: eid -> {"subject", "lab"}; within_session adds "blocks" (partition ->
      [start_s, stop_s]), "trials" (partition -> row indices into Session.trials) and
      "trial_intervals" (one [intervals_0, intervals_1] per row of Session.trials);
      held_out_region adds QC-passing unit counts "n_units", "n_in_region" and
      "n_possibly_in_region", for every manifest session, including dropped ones.
    manifest: provenance of the manifest the split was built from.
    preproc: {"fingerprint", "bin_ms"} of the preprocessing it was built for.
    """

    kind: str
    params: dict
    partitions: dict
    sessions: dict
    manifest: dict
    preproc: dict

    def content(self) -> dict:
        return {"format_version": SPLIT_FORMAT_VERSION, **asdict(self)}

    @property
    def hash(self) -> str:
        canonical = json.dumps(
            self.content(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bin_range(block: list, bin_ms: int) -> tuple[int, int]:
    """First and last bin index (inclusive) touched by a [start_s, stop_s] block.

    Same grid as preprocess.binning: bin k covers [k * w, (k + 1) * w).
    """
    rate = 1000 // bin_ms
    return int(np.floor(block[0] * rate)), int(np.floor(block[1] * rate))


def _check_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    return int(seed)


def _session_meta(manifest: Manifest) -> dict:
    table = manifest.sessions[["eid", "subject", "lab"]]
    if table["eid"].duplicated().any():
        raise ValueError("manifest lists an eid more than once")
    if table.isna().any().any():
        raise ValueError("manifest has sessions without a subject or lab")
    return {
        str(eid): {"subject": str(subject), "lab": str(lab)}
        for eid, subject, lab in table.itertuples(index=False)
    }


def _split(kind: str, params: dict, partitions: dict, sessions: dict, manifest, preproc) -> Split:
    return Split(
        kind=kind,
        params=params,
        partitions={p: sorted(eids) for p, eids in partitions.items()},
        sessions=sessions,
        manifest=dict(manifest.provenance),
        preproc={"fingerprint": preproc.fingerprint(), "bin_ms": preproc.bin_ms},
    )


def held_out_groups(
    manifest: Manifest,
    preproc: PreprocConfig,
    *,
    by: str,
    n_test: int,
    n_calibration: int,
    seed: int,
) -> Split:
    """Hold out whole animals (by="subject") or labs (by="lab").

    The groups are shuffled with `seed`; the first n_test are test, the next
    n_calibration are calibration, and the rest are train. With n_calibration == 0
    there is no calibration partition.
    """
    if by not in GROUP_KINDS:
        raise ValueError(f"by must be one of {sorted(GROUP_KINDS)}, got {by!r}")
    seed = _check_seed(seed)
    meta = _session_meta(manifest)
    groups = sorted({m[by] for m in meta.values()})
    if n_test < 1 or n_calibration < 0 or n_test + n_calibration >= len(groups):
        raise ValueError(
            f"{n_test} test + {n_calibration} calibration groups leave no {by} for training "
            f"(there are {len(groups)} groups)"
        )
    shuffled = [groups[i] for i in np.random.default_rng(seed).permutation(len(groups))]
    assigned = {g: "test" for g in shuffled[:n_test]}
    assigned |= {g: "calibration" for g in shuffled[n_test : n_test + n_calibration]}

    partitions = {p: [] for p in PARTITIONS if p != "calibration" or n_calibration}
    for eid, m in meta.items():
        partitions[assigned.get(m[by], "train")].append(eid)
    params = {"n_test": int(n_test), "n_calibration": int(n_calibration), "seed": seed}
    return _split(GROUP_KINDS[by], params, partitions, meta, manifest, preproc)


def held_out_session(manifest: Manifest, preproc: PreprocConfig, *, seed: int) -> Split:
    """Test one session, drawn with `seed`, of every animal that has at least two."""
    seed = _check_seed(seed)
    meta = _session_meta(manifest)
    by_subject: dict[str, list[str]] = {}
    for eid in sorted(meta):
        by_subject.setdefault(meta[eid]["subject"], []).append(eid)

    rng = np.random.default_rng(seed)
    test = [
        eids[rng.integers(len(eids))] for _, eids in sorted(by_subject.items()) if len(eids) > 1
    ]
    if not test:
        raise ValueError("no animal has two sessions, so no session can be held out")
    partitions = {"train": set(meta) - set(test), "test": test}
    return _split("held_out_session", {"seed": seed}, partitions, meta, manifest, preproc)


def _check_intervals(eid: str, intervals: np.ndarray) -> None:
    if not np.all(np.isfinite(intervals)):
        raise ValueError(f"{eid}: trial intervals must be finite")
    if np.any(intervals[:, 1] < intervals[:, 0]):
        raise ValueError(f"{eid}: a trial ends before it starts")
    if np.any(intervals[1:, 0] < intervals[:-1, 1]):
        raise ValueError(f"{eid}: trials must be in time order and must not overlap")


def within_session(
    manifest: Manifest,
    trials: Mapping[str, pd.DataFrame],
    preproc: PreprocConfig,
    *,
    train_fraction: float,
    gap_s: float,
) -> Split:
    """Split each session's trials in time: the first train_fraction train, the rest test.

    trials: eid -> that session's Session.trials (row i is trial i). The train block runs
    from the first trial's start to the last train trial's end. Test trials are the
    later ones that start at least gap_s after that, measured in whole bins; any trial
    in between is in neither partition. gap_s must be >= 2 s, and also covers the
    model's context; `assert_split_valid` checks that.
    """
    if gap_s < MIN_GAP_S:
        raise ValueError(f"gap_s must be at least {MIN_GAP_S} s, got {gap_s}")
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    meta = _session_meta(manifest)
    unknown = sorted(set(trials) - set(meta))
    if unknown or not trials:
        raise ValueError(f"trials must be given for manifest sessions; unknown: {unknown}")

    rate = 1000 // preproc.bin_ms
    gap_bins = math.ceil(gap_s * rate)
    sessions = {}
    for eid in sorted(trials):
        intervals = trials[eid][["intervals_0", "intervals_1"]].to_numpy(np.float64)
        _check_intervals(eid, intervals)
        n_train = round(train_fraction * len(intervals))
        if not 1 <= n_train < len(intervals):
            raise ValueError(f"{eid}: {len(intervals)} trials cannot be split at {train_fraction}")
        train_stop = intervals[n_train - 1, 1]
        first_test_bin = int(np.floor(train_stop * rate)) + 1 + gap_bins
        start_bins = np.floor(intervals[:, 0] * rate).astype(np.int64)
        first_test = n_train + int(np.searchsorted(start_bins[n_train:], first_test_bin))
        if first_test == len(intervals):
            raise ValueError(f"{eid}: no trial starts after the {gap_s} s gap")
        sessions[eid] = {
            **meta[eid],
            "blocks": {
                "train": [float(intervals[0, 0]), float(train_stop)],
                "test": [float(intervals[first_test, 0]), float(intervals[-1, 1])],
            },
            "trials": {
                "train": list(range(n_train)),
                "test": list(range(first_test, len(intervals))),
            },
            "trial_intervals": intervals.tolist(),
        }
    partitions = {"train": list(sessions), "test": list(sessions)}
    params = {"train_fraction": float(train_fraction), "gap_s": float(gap_s)}
    return _split("within_session", params, partitions, sessions, manifest, preproc)


def _region_relatives(region: str) -> tuple[set[str], str]:
    """Allen acronyms that contain or lie inside Beryl `region`, and the iblatlas version.

    A unit labelled with one of these but with no Beryl region (e.g. STR for CP) could
    be in the region.
    """
    from importlib.metadata import version

    from iblatlas.regions import BrainRegions

    regions = BrainRegions()
    beryl = set(regions.acronym[np.unique(regions.mappings["Beryl"])]) - {"root", "void"}
    if region not in beryl:
        raise ValueError(f"{region!r} is not a Beryl region")
    ids = regions.acronym2id(region)
    relatives = set(regions.ancestors(ids)["acronym"]) | set(regions.descendants(ids)["acronym"])
    return relatives, version("iblatlas")


def held_out_region(
    manifest: Manifest, units: pd.DataFrame, preproc: PreprocConfig, *, region: str
) -> Split:
    """Hold out Beryl region `region`, by the stricter definition in the split doc.

    units: every unit of every manifest session, with eid, label, acronym,
    task_firing_rate and beryl_acronym (qc.task_rates.release_units). Only units passing
    preproc.qc count. A unit is in the region when its beryl_acronym is the region, and
    possibly in it when it has no Beryl region and its Allen acronym contains or lies
    inside the region. Test sessions have >= MIN_REGION_FRACTION of their units in the
    region; train sessions have none in it or possibly in it; the rest are dropped.
    """
    meta = _session_meta(manifest)
    covered = set(units["eid"])
    missing, unknown = sorted(set(meta) - covered), sorted(covered - set(meta))
    if missing or unknown:
        raise ValueError(f"units table lacks sessions {missing[:3]}, has unknown {unknown[:3]}")
    relatives, atlas_version = _region_relatives(region)

    passing = units[unit_qc(units, preproc.qc)["passed"].to_numpy()]
    beryl = passing["beryl_acronym"]
    counts = (
        pd.DataFrame(
            {
                "n_units": 1,
                "n_in_region": beryl == region,
                "n_possibly_in_region": beryl.isna() & passing["acronym"].isin(relatives),
            }
        )
        .groupby(passing["eid"].to_numpy())
        .sum()
    )
    counts = counts.reindex(sorted(meta), fill_value=0).astype(np.int64)

    sessions, partitions = {}, {"train": [], "test": []}
    for eid, n in counts.iterrows():
        sessions[eid] = {**meta[eid], **{k: int(v) for k, v in n.items()}}
        # k / n, not k >= 0.2 * n: 0.2 * 35 is 7.000000000000001, but 7 / 35 is 0.2.
        if n["n_units"] and n["n_in_region"] / n["n_units"] >= MIN_REGION_FRACTION:
            partitions["test"].append(eid)
        elif n["n_units"] and n["n_in_region"] == n["n_possibly_in_region"] == 0:
            partitions["train"].append(eid)
    if not partitions["test"]:
        raise ValueError(
            f"no session has {MIN_REGION_FRACTION:.0%} of its QC-passing units in {region}"
        )
    if not partitions["train"]:
        raise ValueError(f"no session is free of units that are or could be in {region}")
    params = {
        "region": region,
        "min_region_fraction": MIN_REGION_FRACTION,
        "atlas": f"iblatlas {atlas_version}",
    }
    return _split("held_out_region", params, partitions, sessions, manifest, preproc)


def held_out_config(manifest: Manifest, units: pd.DataFrame, preproc: PreprocConfig) -> Split:
    """Hold out sessions with unusually few QC-passing units for their lab.

    units: every unit of every manifest session, with eid, label, acronym and
    task_firing_rate (qc.task_rates.release_units). Test sessions have fewer
    QC-passing units than the CONFIG_TEST_PERCENTILE-th percentile of their lab's
    sessions; train sessions have at least their lab's CONFIG_TRAIN_PERCENTILE-th; the
    rest are dropped. Every session's count and every lab's cutoffs are recorded, so
    the guard can recompute them.
    """
    meta = _session_meta(manifest)
    covered = set(units["eid"])
    missing, unknown = sorted(set(meta) - covered), sorted(covered - set(meta))
    if missing or unknown:
        raise ValueError(f"units table lacks sessions {missing[:3]}, has unknown {unknown[:3]}")
    passing = units.loc[unit_qc(units, preproc.qc)["passed"].to_numpy(), "eid"]
    counts = passing.value_counts().reindex(sorted(meta), fill_value=0).astype(np.int64)

    sessions = {eid: {**meta[eid], "n_units": int(n)} for eid, n in counts.items()}
    cutoffs = config_cutoffs(sessions)
    partitions: dict[str, list[str]] = {"train": [], "test": []}
    for eid, record in sessions.items():
        lab = cutoffs[record["lab"]]
        if record["n_units"] < lab["test_below_units"]:
            partitions["test"].append(eid)
        elif record["n_units"] >= lab["train_from_units"]:
            partitions["train"].append(eid)
    if not partitions["test"] or not partitions["train"]:
        raise ValueError("the within-lab cutoffs leave an empty partition")
    params = {
        "test_percentile": CONFIG_TEST_PERCENTILE,
        "train_percentile": CONFIG_TRAIN_PERCENTILE,
        "percentile_method": "linear",
        "percentiles_within": "lab",
        "cutoffs": cutoffs,
    }
    return _split("held_out_config", params, partitions, sessions, manifest, preproc)


def _bin(t: float, rate: int) -> int:
    return int(np.floor(t * rate))


def leave_one_block_out(
    manifest: Manifest,
    trials: Mapping[str, pd.DataFrame],
    preproc: PreprocConfig,
    *,
    gap_s: float,
) -> Split:
    """One fold per adjacent pair of biased blocks (probabilityLeft 0.2 and 0.8) of each
    session; with an odd number of blocks, the last one joins the final pair.

    trials: eid -> that session's Session.trials (row i is trial i), with intervals and
    probabilityLeft. A fold's test span runs from its pair's first trial's start to
    its last trial's end. Its training trials are every earlier trial whose last bin
    is at least ceil(gap_s * rate) empty bins before the test span's first bin, and
    every later trial whose first bin is that far after its last; trials in between
    are in neither. The opening unbiased block is never tested. Sessions with fewer
    than MIN_LOBO_BLOCKS biased blocks are refused, not dropped. Each session record
    keeps every trial's probabilityLeft (trial_prior) so the guard can check labels.
    """
    if gap_s < MIN_GAP_S:
        raise ValueError(f"gap_s must be at least {MIN_GAP_S} s, got {gap_s}")
    meta = _session_meta(manifest)
    unknown = sorted(set(trials) - set(meta))
    if unknown or not trials:
        raise ValueError(f"trials must be given for manifest sessions; unknown: {unknown}")
    rate = 1000 // preproc.bin_ms
    gap_bins = math.ceil(gap_s * rate)
    sessions = {}
    for eid in sorted(trials):
        table = trials[eid]
        intervals = table[["intervals_0", "intervals_1"]].to_numpy(np.float64)
        _check_intervals(eid, intervals)
        prior = table["probabilityLeft"].to_numpy(np.float64)
        change = np.flatnonzero(np.diff(prior) != 0) + 1
        edges = np.concatenate([[0], change, [len(prior)]])
        blocks = [(a, b - 1) for a, b in zip(edges[:-1], edges[1:]) if prior[a] in (0.2, 0.8)]
        if len(blocks) < MIN_LOBO_BLOCKS:
            raise ValueError(
                f"{eid}: {len(blocks)} biased blocks; leave-one-block-out needs "
                f"{MIN_LOBO_BLOCKS} so every fold trains on both labels"
            )
        k = LOBO_BLOCKS_PER_FOLD
        groups = [(blocks[i][0], blocks[i + k - 1][1]) for i in range(0, len(blocks) - k + 1, k)]
        groups[-1] = (groups[-1][0], blocks[-1][1])  # a leftover block joins the final pair
        start_bins = np.floor(intervals[:, 0] * rate).astype(np.int64)
        end_bins = np.floor(intervals[:, 1] * rate).astype(np.int64)
        folds = []
        for first, last in groups:
            test_first, test_last = start_bins[first], end_bins[last]
            before = [i for i in range(first) if end_bins[i] < test_first - gap_bins]
            after = [
                i for i in range(last + 1, len(intervals)) if start_bins[i] > test_last + gap_bins
            ]
            train_blocks = []
            for part in (before, after):
                if part:
                    train_blocks.append(
                        [float(intervals[part[0], 0]), float(intervals[part[-1], 1])]
                    )
            folds.append(
                {
                    "blocks": {
                        "train": train_blocks,
                        "test": [float(intervals[first, 0]), float(intervals[last, 1])],
                    },
                    "trials": {"train": before + after, "test": list(range(first, last + 1))},
                }
            )
        sessions[eid] = {
            **meta[eid],
            "folds": folds,
            "trial_intervals": intervals.tolist(),
            "trial_prior": prior.tolist(),
        }
    partitions = {"train": list(sessions), "test": list(sessions)}
    return _split(
        "leave_one_block_out",
        {"gap_s": float(gap_s), "blocks_per_fold": LOBO_BLOCKS_PER_FOLD},
        partitions,
        sessions,
        manifest,
        preproc,
    )


def save_split(split: Split, path: str | os.PathLike) -> Path:
    """Write the split and its hash. An existing file is never replaced by a different split."""
    path = Path(path)
    content = {**split.content(), "hash": split.hash}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if path.exists():
        if path.read_text() == text:
            return path
        raise FileExistsError(f"{path} already holds a different split; splits are not replaced")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-split-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_split(path: str | os.PathLike) -> Split:
    """Read a split file, refusing it if its content no longer matches its hash."""
    raw = json.loads(Path(path).read_text())
    stored = raw.pop("hash", None)
    version = raw.pop("format_version", None)
    if version != SPLIT_FORMAT_VERSION:
        raise ValueError(f"{path}: split format {version}, this code reads {SPLIT_FORMAT_VERSION}")
    expected = {f.name for f in fields(Split)}
    if set(raw) != expected:
        raise ValueError(f"{path}: split fields {sorted(raw)}, expected {sorted(expected)}")
    split = Split(**raw)
    if split.hash != stored:
        raise ValueError(
            f"{path}: content does not match its hash; the file was edited or corrupted"
        )
    return split
