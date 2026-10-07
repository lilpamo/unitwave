"""Real sessions as contract inputs: the DataProvider for one split and one target.

SplitData prepares every session of the split once: BWM session -> unit QC and bins
(preprocess_session) -> the target (targets/) -> a Normalizer fit on the split's
training data. It keeps the binned counts and targets, not the spike trains, and
normalises on each request. Every session is prepared up front, which suits the
within-session splits of Phase 3; cross-session runs over hundreds of sessions will
need a lazier provider (Phase 4).

Samples for one session and partition (contract.SessionData):
- per-bin targets (wheel_velocity, movement_state): every window end in the
  partition's span (preprocess.windows) whose bin lies in the task period and whose
  target is defined; with train_stride k > 1, training keeps every k-th bin only.
- trial targets (choice, block): the target's trials (bwm_include, label defined)
  that the split lists in the partition, at the target's own window. A trial whose
  window reaches outside the partition's span is dropped and counted in `dropped`.
With a shift (null_shuffle), the target is rotated first (evaluation.nulls) and the
same ends are sampled; a bin whose rotated target is undefined is dropped.

leave_one_block_out splits (trial targets only; adopted after the first table): the
provider has one fold per held-out block index (`folds()`); a fold serves the
session's fold-k training and test trials, keeps a window only if it lies wholly in
one of that partition's intervals (others dropped and counted), and normalises with a
Normalizer fit on that fold's training intervals. For block, `pseudo=seed` replaces
the labels with a pseudo-session from IBL's generator (null_pseudosession).

Studio (S3): `unit_ids` restricts a session to the units shown on the page. Only
QC-passing units are binned, so a shown unit that fails QC is left out and listed in
`units_excluded`. `sample_trials` maps test samples to their trials for the
single-session bootstrap.
"""

import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from unitwave.data.load import load_data_config, load_session
from unitwave.data.session import Session
from unitwave.evaluation.contract import SessionData
from unitwave.evaluation.nulls import (
    NullConfig,
    bin_trialstruct_features,
    draw_shifts,
    generate_pseudo_blocks,
    load_null_config,
    shift_bin_target,
    shift_trial_target,
    task_bins,
    trial_trialstruct_features,
)
from unitwave.preprocess.binning import (
    BinnedSpikes,
    PreprocConfig,
    load_preproc_config,
    preprocess_session,
)
from unitwave.preprocess.normalize import fit_normalizer
from unitwave.preprocess.windows import window_plan
from unitwave.qc.rules import apply_qc
from unitwave.splits.guards import assert_split_valid
from unitwave.splits.registry import Split, bin_range
from unitwave.targets import bins as bin_targets
from unitwave.targets import task as task_targets
from unitwave.targets import trials as trial_targets
from unitwave.targets.config import TargetConfig, load_target_config

KINDS = {
    "wheel_velocity": "regression",
    "movement_state": "classification",
    "choice": "classification",
    "block": "classification",
    "stimulus_side": "classification",
}
PER_BIN = ("wheel_velocity", "movement_state")


@dataclass(frozen=True)
class _Trials:
    """A session's id and trials: all the null builders read (not its spike trains)."""

    eid: str
    trials: pd.DataFrame


@dataclass(frozen=True)
class _Prepared:
    trials: _Trials
    units: pd.DataFrame
    binned: BinnedSpikes
    target: object  # targets.bins.BinTarget or targets.trials.TrialTarget


class SplitData:
    """contract.DataProvider over real sessions, for one split and one target."""

    def __init__(
        self,
        split: Split,
        target: str,
        *,
        context_bins: int | None = None,
        train_stride: int = 1,
        load: Callable[[str], Session] | None = None,
        behaviour_root=None,
        preproc: PreprocConfig | None = None,
        targets: TargetConfig | None = None,
        nulls: NullConfig | None = None,
        unit_ids: Mapping[str, Collection[str]] | None = None,
        task=None,
    ):
        # A task's own trial target ("task:<name>", step 8a): its definition says what it is.
        self.task = task
        self.task_target = target.startswith(task_targets.PREFIX)
        if self.task_target:
            name = task_targets.target_name(target)
            if task is None or task.decoding is None or name not in task.decoding.targets:
                raise ValueError(f"{target}: the task definition declares no such target")
            kind = "classification"
        elif target not in KINDS:
            raise ValueError(f"unknown target {target!r}; one of {sorted(KINDS)}")
        else:
            kind = KINDS[target]
        if isinstance(train_stride, bool) or not isinstance(train_stride, int) or train_stride < 1:
            raise ValueError(f"train_stride must be a positive integer, got {train_stride!r}")
        self.split, self.target, self.kind = split, target, kind
        self.train_stride = train_stride
        self.preproc = preproc or load_preproc_config()
        self.targets = targets or load_target_config()
        self.nulls = nulls or load_null_config()
        self.context_bins = self._context(context_bins)
        self.lobo = split.kind == "leave_one_block_out"
        if self.lobo and target in PER_BIN:
            raise ValueError("leave_one_block_out splits are for trial targets only")
        # A task's target is checked against the preprocessing it actually uses (the
        # session's own QC) and, for a one-session catalogue (analysis.decoding), against
        # that catalogue: there is no release manifest for it to be out of date with.
        expected = {}
        if self.task_target:
            expected["preproc_fingerprint"] = self.preproc.fingerprint()
            if "catalog" in split.manifest:
                expected["manifest_provenance"] = split.manifest
        self.split_expectations = expected  # read by contract.evaluate too
        if self.lobo:
            assert_split_valid(split, context_bins=self.context_bins, **expected)
            self.plan = None
        else:
            self.plan = window_plan(
                split, context_bins=self.context_bins, stride_bins=1, **expected
            )
        self.pseudo_sessions = target == "block"
        if target == "movement_state" and behaviour_root is None:
            behaviour_root = load_data_config().bwm_behavior_root
        self._behaviour_root = behaviour_root
        load = load or (lambda eid: load_session(eid, "bwm"))

        self._prepared: dict[str, _Prepared] = {}
        self.units_excluded: dict[str, list[str]] = {}
        for eid in sorted(set().union(*split.partitions.values())):
            session = load(eid)
            binned = preprocess_session(session, self.preproc)
            units = apply_qc(session, self.preproc.qc).units.loc[list(binned.unit_ids)]
            if unit_ids is not None and eid in unit_ids:
                binned, units = self._subset(session, binned, units, list(unit_ids[eid]))
            self._prepared[eid] = _Prepared(
                _Trials(eid, session.trials), units, binned, self._build_target(session, binned)
            )
        binned = {e: p.binned for e, p in self._prepared.items()}
        if self.lobo:
            self.n_folds = max(len(split.sessions[e]["folds"]) for e in self._prepared)
            # One Normalizer per fold, each fit on that fold's training intervals only.
            self.normalizers = [
                fit_normalizer(split, binned, self.preproc, fold=k) for k in range(self.n_folds)
            ]
            self.normalizer = None
        else:
            self.n_folds = 1
            self.normalizer = fit_normalizer(split, binned, self.preproc)
            self.normalizers = [self.normalizer]
        self.dropped: dict[tuple[str, str], int] = {}

    def _subset(self, session, binned: BinnedSpikes, units: pd.DataFrame, chosen: list):
        """binned and units restricted to the chosen units that pass QC."""
        unknown = [u for u in chosen if u not in session.units.index]
        if unknown:
            raise ValueError(f"{session.eid}: units not in the session: {unknown}")
        wanted = set(chosen)
        keep = [u for u in binned.unit_ids if u in wanted]
        self.units_excluded[session.eid] = [u for u in chosen if u not in set(keep)]
        if not keep:
            raise ValueError(
                f"{session.eid}: no unit of the {len(chosen)} chosen passes unit QC, so "
                "there is nothing to decode from"
            )
        rows = [binned.unit_ids.index(u) for u in keep]
        assert len(rows) == len(keep)
        return replace(binned, unit_ids=tuple(keep), counts=binned.counts[rows]), units.loc[keep]

    def sample_trials(self, eid: str, ends) -> np.ndarray:
        """(n,) the trial (row of Session.trials) of each sample, by its window's end bin:
        from the target's own table for trial targets; for per-bin targets, the last
        trial starting at or before the bin."""
        p = self._prepared[eid]
        ends = np.asarray(ends, np.int64)
        if self.target in PER_BIN:
            rate = 1000 // p.binned.bin_ms
            starts = p.trials.trials["intervals_0"].to_numpy(np.float64)
            return np.searchsorted(np.floor(starts * rate).astype(np.int64), ends, "right") - 1
        lookup = p.target.table.set_index("end_bin")["trial"]
        if lookup.index.duplicated().any() or not np.isin(ends, lookup.index).all():
            raise ValueError(f"{eid}: samples don't map one-to-one onto the target's trials")
        return lookup.loc[ends].to_numpy(np.int64)

    def folds(self) -> list:
        """One provider per fold: [self] unless the split is leave_one_block_out."""
        return [_Fold(self, k) for k in range(self.n_folds)] if self.lobo else [self]

    def _context(self, context_bins: int | None) -> int:
        if self.target in PER_BIN:
            if context_bins is None:
                raise ValueError(f"{self.target} needs context_bins (the model's window)")
            return context_bins
        if self.task_target:
            name = task_targets.target_name(self.target)
            window = task_targets.window_bins(self.task, name, self.preproc.bin_ms)
        else:
            window = self.targets.windows[self.target].n_bins(self.preproc.bin_ms)
        if context_bins is not None and context_bins != window:
            raise ValueError(f"{self.target}'s window is {window} bins, not {context_bins}")
        return window

    def _build_target(self, session: Session, binned: BinnedSpikes):
        if self.task_target:
            name = task_targets.target_name(self.target)
            return task_targets.task_trial_target(session, binned, self.task, name)
        if self.target == "wheel_velocity":
            return bin_targets.wheel_velocity(session, binned, self.targets)
        if self.target == "movement_state":
            return bin_targets.movement_state(session, binned, self._behaviour_root, self.targets)
        return getattr(trial_targets, self.target)(session, binned, self.targets)

    def data(
        self,
        eid: str,
        partition: str,
        *,
        shift: int | None = None,
        pseudo: int | None = None,
        fold: int | None = None,
    ) -> SessionData:
        if eid not in self.split.partitions.get(partition, ()):
            raise ValueError(f"{eid} is not in {partition}")
        if self.lobo and fold is None:
            raise ValueError("a leave_one_block_out provider serves data through folds()")
        p = self._prepared[eid]
        if self.target in PER_BIN:
            if pseudo is not None:
                raise ValueError(f"{self.target} has no pseudo-sessions")
            ends, y, features = self._bin_samples(p, partition, shift)
        else:
            ends, y, features = self._trial_samples(p, partition, shift, pseudo, fold)
        normalizer = self.normalizers[fold] if self.lobo else self.normalizer
        return SessionData(
            eid=eid,
            partition=partition,
            z=normalizer.transform(p.binned),
            first_bin=p.binned.first_bin,
            bin_ms=p.binned.bin_ms,
            units=p.units,
            ends=ends,
            context_bins=self.context_bins,
            y=y,
            task_features=features,
        )

    def _bin_samples(self, p: _Prepared, partition: str, shift: int | None):
        b = p.binned
        ends = self.plan.ends(b, partition)
        rel = ends - b.first_bin
        task = task_bins(p.trials, b.first_bin, b.n_bins, b.bin_ms)
        keep = (rel >= task.start) & (rel < task.stop)
        if partition == "train" and self.train_stride > 1:
            keep &= ends % self.train_stride == 0
        target = p.target if shift is None else shift_bin_target(p.target, p.trials, shift)
        y = target.values[rel].astype(np.float64)
        keep &= np.isfinite(y)
        features, _ = bin_trialstruct_features(p.trials, b, self.nulls)
        return ends[keep], y[keep], features[rel[keep]]

    def _trial_target(self, p: _Prepared, shift: int | None, pseudo: int | None):
        target = p.target if shift is None else shift_trial_target(p.target, shift)
        if pseudo is None:
            return target
        if self.target != "block":
            raise ValueError(f"{self.target} has no pseudo-sessions")
        prior = generate_pseudo_blocks(len(p.trials.trials), seed=pseudo)
        table = target.table
        pseudo_prior = prior[table["trial"].to_numpy()]
        keep = pseudo_prior != 0.5  # the generator's unbiased opening has no block label
        table = table[keep].assign(label=(pseudo_prior[keep] == 0.8).astype(np.int64))
        return replace(target, table=table.reset_index(drop=True), name=f"{target.name}:pseudo")

    def _trial_samples(self, p, partition, shift, pseudo=None, fold=None):
        target = self._trial_target(p, shift, pseudo)
        table = target.table
        eid = p.trials.eid
        if self.lobo:
            record = self.split.sessions[eid]["folds"][fold]
            table = table[table["trial"].isin(record["trials"][partition])]
            spans = (
                record["blocks"]["train"] if partition == "train" else [record["blocks"]["test"]]
            )
            ranges = [bin_range(span, self.preproc.bin_ms) for span in spans]
            start = table["end_bin"] - self.context_bins + 1
            inside = np.zeros(len(table), dtype=bool)
            for first, last in ranges:
                inside |= ((start >= first) & (table["end_bin"] <= last)).to_numpy()
            key = f"{partition}/fold{fold}"
        else:
            if self.split.kind == "within_session":
                listed = self.split.sessions[eid]["trials"][partition]
                table = table[table["trial"].isin(listed)]
            first, last = self.plan.span(p.binned, partition)
            inside = (
                (table["end_bin"] - self.context_bins + 1 >= first) & (table["end_bin"] <= last)
            ).to_numpy()
            key = partition
        if pseudo is None and shift is None:
            self.dropped[(eid, key)] = int((~inside).sum())
        table = table[inside].reset_index(drop=True)
        if self.task_target:
            features, _ = task_targets.task_trialstruct_features(
                p.trials.trials,
                replace(target, table=table),
                self.task,
                history_trials=self.nulls.history_trials,
            )
        else:
            features, _ = trial_trialstruct_features(
                p.trials, replace(target, table=table), self.nulls
            )
        ends = table["end_bin"].to_numpy(np.int64)
        return ends, table["label"].to_numpy(np.float64), features

    def shifts(self, eid: str, n_shifts: int, *, seed: int) -> np.ndarray:
        p = self._prepared[eid]
        if self.target in PER_BIN:
            b = p.binned
            task = task_bins(p.trials, b.first_bin, b.n_bins, b.bin_ms)
            minimum = math.ceil(self.nulls.min_shift_s * 1000 / b.bin_ms)
            return draw_shifts(task.stop - task.start, minimum, n_shifts, seed=seed)
        # A task's own targets use the minimum shift its definition declares.
        minimum = (
            self.task.decoding.min_shift_trials if self.task_target else self.nulls.min_shift_trials
        )
        return draw_shifts(len(p.target.table), minimum, n_shifts, seed=seed)


class _Fold:
    """One fold of a leave_one_block_out SplitData: the sessions that have fold k."""

    def __init__(self, parent: SplitData, k: int):
        self.parent, self.k = parent, k
        eids = [e for e in parent._prepared if k < len(parent.split.sessions[e]["folds"])]
        self.split = _FoldSplit({"train": eids, "test": eids})

    def data(self, eid: str, partition: str, **labels) -> SessionData:
        return self.parent.data(eid, partition, fold=self.k, **labels)


@dataclass(frozen=True)
class _FoldSplit:
    partitions: dict
