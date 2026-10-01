"""Verdicts for one session (Studio, S3; signed off by the user 2026-10-01).  [R4, R7]

The contract's verdicts are a Wilcoxon test over test sessions and need at least 5, so
one session never reaches one. For one session, each comparison of the contract gets
a within-session test on its primary metric (AUROC):
- **vs null_shuffle and null_pseudosession:** the model's score ranked among the null
  draws, each the model refit on that draw's labels:
  p = (1 + #draws scoring >= the model) / (1 + #draws).
- **vs null_trialstruct and baseline_ridge** (scored on the same test samples): a
  paired bootstrap over whole test trials. Each resample draws each fold's test trials
  with replacement (a trial's bins move together), scores both rows per fold, and
  averages over the folds where both classes are present, as the contract does;
  p = (1 + #resamples where the difference <= 0) / (1 + #resamples with a score).
- **Correction:** Benjamini-Hochberg across the session's defined p-values. A
  comparison beats its row when q < alpha and the observed difference is positive.

Classification only: every Studio target is a class. Resamples are integer weights
on the samples, and AUROC is computed on the weighted samples (equal to sklearn's on
the repeated samples), in chunks of resamples. Seeds are derived from the run's seed
and the comparison's name.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import false_discovery_control

from unitwave.evaluation.contract import COMPARISONS, GATE, ContractResult, _session_seed

_RANKED = {
    "null_shuffle": ("shuffle", "shifted-target"),
    "null_pseudosession": ("pseudo", "pseudo-session"),
}
_CHUNK = 200  # resamples scored at once


@dataclass(frozen=True)
class SessionTest:
    """subject vs row. difference: subject's primary metric minus row's (observed).
    n: null draws or bootstrap resamples used. note: why it is undefined or vacuous."""

    subject: str
    row: str
    method: str
    difference: float
    p: float
    q: float
    beats: bool
    n: int
    note: str = ""

    @property
    def is_gate(self) -> bool:
        return (self.subject, self.row) == GATE


def weighted_auroc(y, scores, weights) -> np.ndarray:
    """(n_resamples,) AUROC of scores for 0/1 y, each resample weighting the samples by
    a row of weights (n_resamples, n_samples); NaN where a resample lacks a class.
    Ties count one half, as in sklearn."""
    y, scores = np.asarray(y), np.asarray(scores, np.float64)
    weights = np.atleast_2d(np.asarray(weights, np.float64))
    assert y.shape == scores.shape and weights.shape[1] == y.size
    order = np.argsort(scores, kind="stable")
    ranked = scores[order]
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])  # each tied value
    w = weights[:, order]
    positive = (y[order] == 1)[None, :]
    neg = np.add.reduceat(w * ~positive, starts, axis=1)  # (n_resamples, n_values)
    pos = np.add.reduceat(w * positive, starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    numerator = (pos * (below + 0.5 * neg)).sum(axis=1)
    denominator = pos.sum(axis=1) * neg.sum(axis=1)
    out = np.full(len(w), np.nan)
    ok = denominator > 0
    out[ok] = numerator[ok] / denominator[ok]
    return out


def rank_test(score: float, draws) -> tuple[float, int]:
    """p of score among the finite null draws, and how many there were."""
    draws = np.asarray(draws, np.float64)
    draws = draws[np.isfinite(draws)]
    if not draws.size or not np.isfinite(score):
        return np.nan, int(draws.size)
    return float((1 + (draws >= score).sum()) / (1 + draws.size)), int(draws.size)


def paired_bootstrap(
    y, subject, other, groups, folds, *, n_bootstrap: int, seed: int
) -> tuple[float, int]:
    """One-sided p that subject's AUROC exceeds other's, resampling whole groups
    (trials) within each fold; and the number of resamples with a score. All (n,)."""
    y, subject, other = (np.asarray(a) for a in (y, subject, other))
    groups, folds = np.asarray(groups), np.asarray(folds)
    assert y.shape == subject.shape == other.shape == groups.shape == folds.shape
    rng = np.random.default_rng(seed)
    parts = []
    for f in np.unique(folds):
        idx = np.flatnonzero(folds == f)
        _, member = np.unique(groups[idx], return_inverse=True)
        parts.append((idx, member, member.max() + 1))
    deltas = []
    for start in range(0, n_bootstrap, _CHUNK):
        b = min(_CHUNK, n_bootstrap - start)
        total, scored = np.zeros(b), np.zeros(b)
        for idx, member, n_groups in parts:
            counts = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=b)
            w = counts[:, member]  # (b, n_fold_samples)
            a = weighted_auroc(y[idx], subject[idx], w)
            o = weighted_auroc(y[idx], other[idx], w)
            ok = np.isfinite(a) & np.isfinite(o)
            total += np.where(ok, a - o, 0.0)
            scored += ok
        deltas.append(total[scored > 0] / scored[scored > 0])
    deltas = np.concatenate(deltas)
    if not deltas.size:
        return np.nan, 0
    return float((1 + (deltas <= 0).sum()) / (1 + deltas.size)), int(deltas.size)


def session_tests(
    result: ContractResult, groups_of, *, n_bootstrap: int, alpha: float, seed: int
) -> tuple[SessionTest, ...]:
    """Every contract comparison for the result's one test session, with BH.

    groups_of(eid, ends) -> (n,) the trial each test sample belongs to.
    """
    if result.kind != "classification":
        raise ValueError("single-session verdicts are defined for classification targets")
    eids = list(result.per_session["model"].index)
    if len(eids) != 1:
        raise ValueError(f"single-session verdicts need one test session, got {len(eids)}")
    eid, primary = eids[0], result.primary
    score = {row: float(t.loc[eid, primary]) for row, t in result.per_session.items()}
    raw = []
    for subject, row in COMPARISONS:
        if row not in result.per_session:
            continue
        name = f"{subject}:{row}"
        difference = score[subject] - score[row]
        if row in _RANKED:
            attribute, label = _RANKED[row]
            p, n = rank_test(score[subject], getattr(result, attribute).loc[eid])
            method = f"rank among {n} {label} refits of the model"
            note = "" if n else f"no valid {label} draw for this session"
            if row == "null_shuffle" and not n:
                note = "no valid shift: the session is too short for the minimum shift"
        else:
            a, o = result.predictions[subject], result.predictions[row]
            a, o = a[a["eid"] == eid], o[o["eid"] == eid]
            merged = a.merge(o, on=["fold", "end"], suffixes=("", "_other"), validate="1:1")
            if len(merged) != len(a):
                raise ValueError(f"{subject} and {row} were scored on different samples")
            groups = np.asarray(groups_of(eid, merged["end"].to_numpy()))
            p, n = paired_bootstrap(
                merged["y"].to_numpy(),
                merged["prediction"].to_numpy(),
                merged["prediction_other"].to_numpy(),
                groups,
                merged["fold"].to_numpy(),
                n_bootstrap=n_bootstrap,
                seed=_session_seed(seed, name),
            )
            trials = len({(f, g) for f, g in zip(merged["fold"], groups)})
            method = f"paired bootstrap over {trials} test trials, {n} resamples"
            same = np.array_equal(merged["prediction"], merged["prediction_other"])
            note = (
                f"{subject} and {row} made the same predictions (the same decoder)" if same else ""
            )
        raw.append((subject, row, method, difference, p, n, note))
    ps = np.array([r[4] for r in raw], np.float64)
    q = np.full(ps.size, np.nan)
    defined = np.isfinite(ps)
    if defined.any():
        q[defined] = false_discovery_control(ps[defined], method="bh")
    return tuple(
        SessionTest(
            subject=s,
            row=r,
            method=m,
            difference=d,
            p=p,
            q=float(qq),
            beats=bool(np.isfinite(qq) and qq < alpha and d > 0),
            n=n,
            note=note,
        )
        for (s, r, m, d, p, n, note), qq in zip(raw, q)
    )
