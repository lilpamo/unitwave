"""The evaluation contract: every evaluation produces the full table.  [CLAUDE.md §5, R4]

`evaluate` is the only entry point that scores a model, and it always fits and scores
every row, in this order:

    null_shuffle        the model under test, refit on targets circularly shifted
                        within each session (per-session median over the shifts)
    null_pseudosession  the model under test, refit on pseudo-session labels from the
                        task's own generator (per-session median); only for targets
                        whose provider generates them (block)
    null_trialstruct    a decoder given task-variable features only (never spikes)
    baseline_ridge      ridge / logistic, same split
    baseline_rrr        multi-session reduced-rank regression, same split
    model               the model under test (spikes)
    model_with_task     the model under test given spikes and the task features
    ceiling_within      the model on a within-session split of the test sessions
                        (the model row itself when the split is within-session)

model_with_task and null_pseudosession were added after the first table (see
docs/DECISIONS.md, "Phase 3 gate: incremental row, leave-one-block-out, pseudo-sessions").

Every fit receives the train partition only; predictions are scored per test
session (evaluation.metrics). A provider may split each session into folds (e.g.
leave-one-block-out); each fold is fit and scored separately, and a session's
metrics are the mean over its folds whose primary metric is defined (a fold whose
test labels hold one class has no AUROC), with n_samples and n_folds counting those
folds. Folds are never pooled before scoring: each fold's model has its own offset,
set by its training data's class balance, and pooled predictions let those offsets
rank the samples. Pooling inverted decoders with no signal (docs/NEGATIVE_RESULTS.md).

Verdicts use a one-sided Wilcoxon signed-rank test over test sessions on the primary
metric (R² or AUROC), alpha 0.05, with the median difference and win count; fewer
than 5 sessions can never reach alpha. **The gate** is model_with_task vs
null_trialstruct: does neural activity add information beyond task variables? The
spikes-alone verdict (model vs null_trialstruct) is kept as the stricter, descriptive
result, and the shuffle, pseudo-session and ridge verdicts are printed too.

Data reaches decoders as SessionData from a DataProvider, loaded lazily, one session
at a time; decoders build their own inputs from it (e.g. windows via
preprocess.windows).

For single-session use (Studio, S3): the result keeps every real-label row's test
predictions, which evaluation.single_session resamples; `progress` is told after each
fit; and baseline_rrr, multi-session by design, may be left out with a stated reason.
"""

import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from unitwave.evaluation.metrics import (
    CLASSIFICATION,
    REGRESSION,
    EvalConfig,
    load_eval_config,
    per_session,
    summarise,
)
from unitwave.evaluation.nulls import NullConfig, load_null_config
from unitwave.splits.guards import assert_split_valid
from unitwave.splits.registry import Split

ROWS = (
    "null_shuffle",
    "null_pseudosession",
    "null_trialstruct",
    "baseline_ridge",
    "baseline_rrr",
    "model",
    "model_with_task",
    "ceiling_within",
)
PRIMARY = {"regression": "r2", "classification": "auroc"}
# (subject row, compared row); the second is the gate.
COMPARISONS = (
    ("model", "null_trialstruct"),
    ("model_with_task", "null_trialstruct"),
    ("model", "null_shuffle"),
    ("model", "null_pseudosession"),
    ("model", "baseline_ridge"),
)
GATE = ("model_with_task", "null_trialstruct")
MIN_SHIFTS = 5
MIN_PSEUDO = 20
# Split kinds whose test sessions are also training sessions: the ceiling is the model.
WITHIN_KINDS = ("within_session", "leave_one_block_out")
ALPHA = 0.05
# A one-sided exact Wilcoxon test over n sessions has p >= 2**-n: n < 5 can't reach 0.05.
MIN_SESSIONS = 5
_FAILURE = {
    ("model", "null_trialstruct"): "spikes alone don't outpredict the task variables",
    ("model_with_task", "null_trialstruct"): (
        "neural activity adds nothing beyond the task variables: this is not decoding"
    ),
    ("model", "null_shuffle"): "no signal beyond the target's own autocorrelation",
    ("model", "null_pseudosession"): "no signal beyond the task's own block structure",
    ("model", "baseline_ridge"): "a more complex model is not justified",
}


@dataclass(frozen=True)
class SessionData:
    """One session's samples in one partition.

    z: (n_units, n_bins) float32 normalised counts (Normalizer.transform), bins
      first_bin + j; None for the trial-structure row, which never sees spikes.
    units: (n_units, ...) unit metadata in z's row order; None with z.
    ends: (n,) int64 absolute end bin of each sample's context_bins-long window.
    y: (n,) targets: values, or 0/1 labels for classification.
    task_features: (n, n_features) null_trialstruct features (evaluation.nulls).
    """

    eid: str
    partition: str
    z: np.ndarray | None
    first_bin: int
    bin_ms: int
    units: pd.DataFrame | None
    ends: np.ndarray
    context_bins: int
    y: np.ndarray
    task_features: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.ends)
        if self.y.shape != (n,) or self.task_features.ndim != 2 or len(self.task_features) != n:
            raise ValueError(f"{self.eid}: ends, y and task_features disagree on n = {n}")
        if self.z is not None and (self.units is None or len(self.units) != self.z.shape[0]):
            raise ValueError(f"{self.eid}: units must list z's rows")

    def without_spikes(self) -> "SessionData":
        return replace(self, z=None, units=None)


class Decoder(Protocol):
    kind: str  # "regression" or "classification"

    def fit(self, train: Sequence[SessionData], *, seed: int) -> None: ...

    def predict(self, data: SessionData) -> np.ndarray:
        """(n,) predictions: values, or P(class 1) for classification."""


class DataProvider(Protocol):
    split: Split
    kind: str
    context_bins: int

    def data(
        self, eid: str, partition: str, *, shift: int | None = None, pseudo: int | None = None
    ) -> SessionData:
        """shift: rotate the session's target by this many samples (null_shuffle).
        pseudo: replace its labels with the pseudo-session drawn with this seed."""

    def shifts(self, eid: str, n_shifts: int, *, seed: int) -> np.ndarray:
        """Valid shifts for the session (evaluation.nulls.draw_shifts); may be empty."""

    # Optional: folds() -> providers, one per fold of the split (default: [self]);
    # pseudo_sessions: True when data(..., pseudo=...) is supported.


class _Sessions(Sequence):
    """The train sessions, loaded one at a time when a decoder reads them."""

    def __init__(self, provider, eids, label_of, strip):
        self._provider, self._label_of, self._strip = provider, label_of, strip
        self._eids = [e for e in eids if label_of(e) is not _SKIP]

    def __len__(self) -> int:
        return len(self._eids)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        eid = self._eids[i]
        data = self._provider.data(eid, "train", **self._label_of(eid))
        return data.without_spikes() if self._strip else data

    def __iter__(self) -> Iterator[SessionData]:
        return (self[i] for i in range(len(self)))


_SKIP = object()  # a session with no shift (or pseudo-session) at this draw sits it out


@dataclass(frozen=True)
class Verdict:
    row: str
    beats: bool
    p_value: float
    median_difference: float
    wins: int
    n_sessions: int
    subject: str = "model"

    @property
    def is_gate(self) -> bool:
        return (self.subject, self.row) == GATE

    def line(self, metric: str) -> str:
        who = f"{self.subject} vs {self.row}"
        if self.n_sessions == 0:
            return f"{who}: undefined, no test session has a defined {self.row}."
        detail = (
            f"median Δ{metric} {self.median_difference:+.3f}, wins {self.wins}/{self.n_sessions} "
            f"sessions, Wilcoxon p = {self.p_value:.3g}"
        )
        if self.n_sessions < MIN_SESSIONS:
            detail = f"too few sessions ({self.n_sessions}) for the test; {detail}"
        if self.beats:
            return f"{self.subject} beats {self.row} ({detail})."
        return (
            f"{self.subject} does NOT beat {self.row} ({detail}): "
            f"{_FAILURE[(self.subject, self.row)]}."
        )


@dataclass(frozen=True)
class ContractResult:
    kind: str
    primary: str
    per_session: dict  # row -> (n_test_sessions, metrics) DataFrame, index eid
    shuffle: pd.DataFrame  # (n_test_sessions, n_shifts) primary metric per shift draw
    verdicts: tuple
    seed: int
    n_shifts: int
    split_hash: str
    ceiling_split_hash: str | None
    ceiling_is_model: bool
    pseudo: pd.DataFrame | None = None  # (n_test_sessions, n_pseudo) per pseudo-session
    n_pseudo: int = 0
    n_folds: int = 1
    # row -> (n_test_samples, 5) eid, fold, end, y, prediction, for the real-label rows.
    predictions: dict = field(default_factory=dict)
    not_run: dict = field(default_factory=dict)  # row -> why it was not run

    @property
    def gate(self) -> Verdict:
        return next(v for v in self.verdicts if v.is_gate)

    def summary(self) -> pd.DataFrame:
        """Rows in contract order: the primary metric's distribution over test sessions,
        then the median of every other metric."""
        out = {}
        for row in (r for r in ROWS if r in self.per_session):
            s = summarise(self.per_session[row])
            first = s.loc[self.primary, ["median", "q25", "q75", "n_sessions"]]
            others = s.loc[[m for m in s.index if m != self.primary], "median"]
            out[row] = {**first.to_dict(), **{f"median_{m}": v for m, v in others.items()}}
        table = pd.DataFrame.from_dict(out, orient="index")
        return table.astype({"n_sessions": int})

    def report(self) -> str:
        table = self.summary()
        extra = f", {self.n_pseudo} pseudo-sessions" if self.n_pseudo else ""
        folds = (
            f", {self.n_folds} folds per session at most, scored per fold"
            if self.n_folds > 1
            else ""
        )
        lines = [
            (
                f"Evaluation contract: {self.kind}, primary metric {self.primary} over test "
                f"sessions (split {self.split_hash[:12]}, seed {self.seed}, "
                f"{self.n_shifts} shifts{extra}{folds})"
            ),
            table.to_string(float_format=lambda v: f"{v:.3f}"),
        ]
        if self.ceiling_is_model:
            lines.append("ceiling_within = model: the split is itself within-session.")
        lines += [f"{row}: not run ({why})." for row, why in self.not_run.items()]
        lines += [v.line(self.primary) for v in self.verdicts]
        gate = self.gate
        lines.append(
            f"GATE ({gate.subject} vs {gate.row}): {'PASSED' if gate.beats else 'NOT PASSED'}."
        )
        return "\n".join(lines)

    __str__ = report


def _session_seed(seed: int, eid: str) -> int:
    return int(hashlib.sha256(f"{seed}:{eid}".encode()).hexdigest()[:8], 16)


def _check_predictions(data: SessionData, prediction, kind: str) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(data.ends),) or not np.all(np.isfinite(prediction)):
        raise ValueError(
            f"{data.eid}: predictions must be ({len(data.ends)},) finite, got {prediction.shape}"
        )
    if kind == "classification" and np.any((prediction < 0) | (prediction > 1)):
        raise ValueError(f"{data.eid}: classification predictions must be probabilities")
    return prediction


def _undefined(kind: str, eids) -> pd.DataFrame:
    metrics = CLASSIFICATION if kind == "classification" else REGRESSION
    table = pd.DataFrame(np.nan, index=pd.Index(list(eids), name="eid"), columns=list(metrics))
    return table.assign(n_samples=0)


def _folds(provider) -> list:
    return provider.folds() if hasattr(provider, "folds") else [provider]


def _mean_over_folds(tables: list[pd.DataFrame], kind: str) -> pd.DataFrame:
    """Per session: each metric's mean over the folds where the primary metric is
    defined, the samples in those folds (n_samples) and their number (n_folds)."""
    stacked = pd.concat(tables, keys=range(len(tables)), names=["fold", "eid"])
    scored = stacked[np.isfinite(stacked[PRIMARY[kind]].to_numpy(np.float64))]
    metrics = [c for c in stacked.columns if not c.startswith("n_")]
    by_session = scored.groupby(level="eid")
    table = by_session[metrics].mean()
    table["n_samples"] = by_session["n_samples"].sum()
    table["n_folds"] = by_session.size()
    return table


def _run(
    factory, provider, kind, eval_config, seed, label_of=None, strip=False, keep=None
) -> pd.DataFrame:
    """Fit on each fold's train partition, predict its test partition, and score each
    session: on its held-out predictions with one fold, or as the mean of its per-fold
    scores with several (never pooled; see the module docstring).

    label_of(eid) -> the data() keywords for the session's labels ({} for the real
    ones, {"shift": k}, {"pseudo": seed}), or _SKIP to leave the session out.
    keep: a list to which each fold's test predictions are appended as a DataFrame
    (eid, fold, end, y, prediction).
    """
    label_of = label_of or (lambda eid: {})
    test_eids = list(provider.split.partitions["test"])
    folds = _folds(provider)
    tables = []
    for k, fold in enumerate(folds):
        ys, predictions, eids, ends = [], [], [], []
        decoder = factory()
        if decoder.kind != kind:
            raise ValueError(f"a {decoder.kind} decoder cannot fit a {kind} target")
        train = _Sessions(fold, fold.split.partitions["train"], label_of, strip)
        tested = [e for e in fold.split.partitions["test"] if label_of(e) is not _SKIP]
        if not len(train) or not tested:
            continue  # no session in this fold has these labels (e.g. too short to shift)
        decoder.fit(train, seed=seed)
        for eid in tested:
            data = fold.data(eid, "test", **label_of(eid))
            if strip:
                data = data.without_spikes()
            predictions.append(_check_predictions(data, decoder.predict(data), kind))
            ys.append(np.asarray(data.y, dtype=np.float64))
            eids.append(np.full(len(data.ends), eid, dtype=object))
            ends.append(np.asarray(data.ends, dtype=np.int64))
        if eids:
            y, prediction, sessions = map(np.concatenate, (ys, predictions, eids))
            tables.append(per_session(kind, y, prediction, sessions, eval_config))
            if keep is not None:
                keep.append(
                    pd.DataFrame(
                        {
                            "eid": sessions,
                            "fold": k,
                            "end": np.concatenate(ends),
                            "y": y,
                            "prediction": prediction,
                        }
                    )
                )
    if not tables:
        return _undefined(kind, test_eids)
    if len(folds) == 1:
        return tables[0].reindex(test_eids)
    table = _mean_over_folds(tables, kind).reindex(test_eids)
    counts = ["n_samples", "n_folds"]
    table[counts] = table[counts].fillna(0).astype(np.int64)
    return table


def _verdict(subject: pd.Series, other: pd.Series, row: str, name: str = "model") -> Verdict:
    diff = (subject - other.reindex(subject.index)).dropna()
    n, wins = len(diff), int((diff > 0).sum())
    median = float(diff.median()) if n else np.nan
    p = np.nan
    if n and np.any(diff != 0):
        p = float(wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)
    beats = n >= MIN_SESSIONS and np.isfinite(p) and p < ALPHA and median > 0
    return Verdict(row, bool(beats), p, median, wins, n, subject=name)


def _null_draws(factory, task, kind, eval_config, seed, draws: dict, keyword: str, step):
    """Per draw k: every session with a k-th draw gets those labels; per-draw tables.
    step() is called before each draw."""
    tables = []
    for k in range(max((len(d) for d in draws.values()), default=0)):
        step()

        def label_of(eid, k=k):
            return {keyword: int(draws[eid][k])} if k < len(draws[eid]) else _SKIP

        tables.append(_run(factory, task, kind, eval_config, seed, label_of=label_of))
    return tables


def evaluate(
    task: DataProvider,
    *,
    model: Callable[[], Decoder],
    model_with_task: Callable[[], Decoder],
    baseline_ridge: Callable[[], Decoder],
    baseline_rrr: Callable[[], Decoder] | None,
    trialstruct: Callable[[], Decoder],
    ceiling: DataProvider | None,
    seed: int,
    n_shifts: int | None = None,
    n_pseudo: int | None = None,
    eval_config: EvalConfig | None = None,
    null_config: NullConfig | None = None,
    not_run: dict | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ContractResult:
    """Fit and score every row; decoders are factories, called once per fit and fold.

    ceiling: a within-session provider over the task's test sessions, or None when the
    task's split is itself within-session. n_shifts: shift draws for null_shuffle
    (default configs/nulls.yaml), at least MIN_SHIFTS. n_pseudo: pseudo-sessions for
    null_pseudosession, required (>= MIN_PSEUDO) when the provider generates them.
    baseline_rrr may be None only with a reason in not_run["baseline_rrr"].
    progress(done, total) is called before the first fit and after each row or draw.
    """
    eval_config = eval_config or load_eval_config()
    null_config = null_config or load_null_config()
    n_shifts = null_config.n_shifts if n_shifts is None else n_shifts
    if n_shifts < MIN_SHIFTS:
        raise ValueError(f"null_shuffle needs at least {MIN_SHIFTS} shifts, got {n_shifts}")
    kind = task.kind
    if kind not in PRIMARY:
        raise ValueError(f"unknown target kind {kind!r}")

    not_run = dict(not_run or {})
    if baseline_rrr is None and not not_run.get("baseline_rrr"):
        raise ValueError("baseline_rrr can be left out only with a reason in not_run")
    if set(not_run) - {"baseline_rrr"} or (baseline_rrr is not None and not_run):
        raise ValueError(f"not_run names rows that are run or can't be skipped: {sorted(not_run)}")
    pseudo = bool(getattr(task, "pseudo_sessions", False))
    if pseudo and (n_pseudo is None or n_pseudo < MIN_PSEUDO):
        raise ValueError(f"null_pseudosession needs at least {MIN_PSEUDO} pseudo-sessions")
    if not pseudo and n_pseudo:
        raise ValueError("this provider doesn't generate pseudo-sessions")

    assert_split_valid(task.split, context_bins=task.context_bins)
    ceiling_is_model = task.split.kind in WITHIN_KINDS
    if ceiling_is_model:
        ceiling = None
    elif ceiling is None:
        raise ValueError("a cross-session split needs a within-session ceiling provider")
    else:
        if ceiling.split.kind != "within_session":
            raise ValueError("the ceiling provider's split must be within_session")
        if set(ceiling.split.partitions["test"]) != set(task.split.partitions["test"]):
            raise ValueError("the ceiling must cover exactly the task's test sessions")
        assert_split_valid(ceiling.split, context_bins=ceiling.context_bins)

    total = 4 + (baseline_rrr is not None) + n_shifts + (n_pseudo or 0) * pseudo
    total += not ceiling_is_model
    done = 0

    def step():
        nonlocal done
        if progress is not None:
            progress(done, total)
        done += 1

    per: dict[str, pd.DataFrame] = {}
    kept: dict[str, list] = {}
    real_rows = {
        "model": model,
        "model_with_task": model_with_task,
        "baseline_ridge": baseline_ridge,
        "baseline_rrr": baseline_rrr,
        "null_trialstruct": trialstruct,
    }
    for row, factory in real_rows.items():
        if factory is None:
            continue
        step()
        kept[row] = []
        strip = row == "null_trialstruct"
        per[row] = _run(factory, task, kind, eval_config, seed, strip=strip, keep=kept[row])

    split = task.split
    test_eids = list(split.partitions["test"])
    eids = sorted(set(split.partitions["train"]) | set(test_eids))

    def median_over(tables):
        stacked = pd.concat(tables, keys=range(len(tables)), names=["draw", "eid"])
        return stacked.groupby(level="eid").median().reindex(test_eids)

    shifts = {e: np.asarray(task.shifts(e, n_shifts, seed=_session_seed(seed, e))) for e in eids}
    for e in eids:
        shifts[e] = shifts[e][:n_shifts]
    tables = _null_draws(model, task, kind, eval_config, seed, shifts, "shift", step)
    for _ in range(n_shifts - len(tables)):  # draws no session has still count
        step()
        tables.append(_undefined(kind, test_eids))
    per["null_shuffle"] = median_over(tables)
    shuffle = pd.concat([t[PRIMARY[kind]] for t in tables], axis=1, keys=range(n_shifts))

    pseudo_table = None
    if pseudo:
        seeds = {
            e: np.array([_session_seed(seed, f"{e}:pseudo:{k}") for k in range(n_pseudo)])
            for e in eids
        }
        tables = _null_draws(model, task, kind, eval_config, seed, seeds, "pseudo", step)
        per["null_pseudosession"] = median_over(tables)
        pseudo_table = pd.concat([t[PRIMARY[kind]] for t in tables], axis=1, keys=range(n_pseudo))

    if ceiling_is_model:
        per["ceiling_within"] = per["model"]
    else:
        step()
        per["ceiling_within"] = _run(model, ceiling, kind, eval_config, seed)
    step()  # done == total
    primary = PRIMARY[kind]
    verdicts = tuple(
        _verdict(per[subject][primary], per[row][primary], row, subject)
        for subject, row in COMPARISONS
        if row in per
    )
    return ContractResult(
        kind=kind,
        primary=primary,
        per_session={row: per[row] for row in ROWS if row in per},
        shuffle=shuffle,
        verdicts=verdicts,
        seed=seed,
        n_shifts=n_shifts,
        split_hash=split.hash,
        ceiling_split_hash=None if ceiling is None else ceiling.split.hash,
        ceiling_is_model=ceiling_is_model,
        pseudo=pseudo_table,
        n_pseudo=n_pseudo or 0,
        n_folds=max(len(_folds(task)), 1),
        predictions={
            row: pd.concat(frames, ignore_index=True) for row, frames in kept.items() if frames
        },
        not_run=not_run,
    )
