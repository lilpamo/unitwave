import dataclasses
import hashlib

import numpy as np
import pandas as pd
import pytest

from unitwave.data.manifest import Manifest, manifest_versions
from unitwave.evaluation.contract import ROWS, SessionData, _mean_over_folds, evaluate
from unitwave.evaluation.nulls import draw_shifts
from unitwave.preprocess.binning import load_preproc_config
from unitwave.splits.registry import bin_range, held_out_groups, within_session

PREPROC = load_preproc_config()
N_BINS, N_UNITS, CONTEXT = 600, 3, 5


def _manifest() -> Manifest:
    rows = [
        {"eid": f"e{s}{k}", "subject": f"s{s}", "lab": "lab"} for s in range(10) for k in range(2)
    ]
    return Manifest(
        sessions=pd.DataFrame(rows), insertions=pd.DataFrame(), provenance=manifest_versions()
    )


def _cross_split():
    # 4 test animals x 2 sessions = 8 test sessions; 12 train sessions.
    return held_out_groups(_manifest(), PREPROC, by="subject", n_test=4, n_calibration=0, seed=0)


def _within_split(eids):
    starts = 0.2 + 0.55 * np.arange(20)
    trials = {e: pd.DataFrame({"intervals_0": starts, "intervals_1": starts + 0.5}) for e in eids}
    return within_session(_manifest(), trials, PREPROC, train_fraction=0.7, gap_s=2.0)


def _rng(eid: str, salt: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"{eid}:{salt}".encode()).hexdigest()[:8], 16))


class Provider:
    """Synthetic sessions; the target follows unit 0 ("spikes") or task feature 0 ("task")."""

    def __init__(self, split, signal="spikes", kind="regression", log=None):
        self.split, self.signal, self.kind, self.context_bins = split, signal, kind, CONTEXT
        self.log = log if log is not None else []

    def _session(self, eid):
        z = _rng(eid, "z").normal(size=(N_UNITS, N_BINS)).astype(np.float32)
        task = _rng(eid, "task").normal(size=(N_BINS, 2)).astype(np.float32)
        source = {"spikes": z[0], "task": task[:, 0], "both": z[0] + task[:, 0]}[self.signal]
        y = source + 0.1 * _rng(eid, "noise").normal(size=N_BINS)
        if self.kind == "classification":
            y = (y > 0).astype(np.float64)
        return z, task, y

    def data(self, eid, partition, *, shift=None, pseudo=None):
        self.log.append((eid, partition, shift if pseudo is None else ("pseudo", pseudo)))
        z, task, y = self._session(eid)
        if shift is not None:
            y = np.roll(y, -shift)
        if pseudo is not None:
            y = np.random.default_rng(pseudo).permutation(y)
        if self.split.kind == "within_session":
            first, last = bin_range(self.split.sessions[eid]["blocks"][partition], 20)
        else:
            first, last = 0, N_BINS - 1
        ends = np.arange(first + CONTEXT - 1, last + 1)
        units = pd.DataFrame(index=[f"u{i}" for i in range(N_UNITS)])
        return SessionData(eid, partition, z, 0, 20, units, ends, CONTEXT, y[ends], task[ends])

    def shifts(self, eid, n_shifts, *, seed):
        return draw_shifts(N_BINS, 50, n_shifts, seed=seed)


class LeastSquares:
    """Linear fit on unit activity at each window's last bin."""

    kind = "regression"

    def __init__(self, log=None):
        self.log = log

    def _x(self, d):
        return np.column_stack([d.z[:, d.ends - d.first_bin].T, np.ones(len(d.ends))])

    def fit(self, train, *, seed):
        train = list(train)
        if self.log is not None:
            self.log.extend((d.eid, d.partition) for d in train)
        x = np.vstack([self._x(d) for d in train])
        y = np.concatenate([d.y for d in train])
        self.w = np.linalg.lstsq(x, y, rcond=None)[0]

    def predict(self, d):
        return self._x(d) @ self.w


class TaskLeastSquares(LeastSquares):
    def _x(self, d):
        assert d.z is None and d.units is None, "the trial-structure null saw spikes"
        return np.column_stack([d.task_features, np.ones(len(d.ends))])


class SpikesTaskLeastSquares(LeastSquares):
    def _x(self, d):
        return np.column_stack(
            [d.z[:, d.ends - d.first_bin].T, d.task_features, np.ones(len(d.ends))]
        )


class Logistic(LeastSquares):
    kind = "classification"

    def predict(self, d):
        return 1.0 / (1.0 + np.exp(-4.0 * (self._x(d) @ self.w - 0.5)))


def _run(provider=None, ceiling="auto", **kwargs):
    provider = provider or Provider(_cross_split())
    if ceiling == "auto":
        ceiling = Provider(_within_split(provider.split.partitions["test"]), provider.signal)
    rows = {
        "model": LeastSquares,
        "model_with_task": SpikesTaskLeastSquares,
        "baseline_ridge": LeastSquares,
        "baseline_rrr": LeastSquares,
        "trialstruct": TaskLeastSquares,
    }
    rows.update(kwargs.pop("rows", {}))
    return evaluate(provider, ceiling=ceiling, seed=0, **rows, **kwargs)


def test_every_row_runs_in_order_and_prints():
    result = _run()
    rows = tuple(r for r in ROWS if r != "null_pseudosession")  # no pseudo-sessions here
    assert tuple(result.summary().index) == rows
    text = str(result)
    for row in rows:
        assert row in text
    assert "GATE (model_with_task vs null_trialstruct)" in text
    assert result.primary == "r2"
    assert all(len(table) == 8 for table in result.per_session.values())


def test_decodable_target_beats_both_nulls():
    result = _run()
    verdicts = {(v.subject, v.row): v for v in result.verdicts}
    assert (
        verdicts[("model", "null_trialstruct")].beats and verdicts[("model", "null_shuffle")].beats
    )
    assert result.summary().loc["model", "median"] > 0.9
    assert abs(result.summary().loc["null_shuffle", "median"]) < 0.2


def test_task_driven_target_is_reported_as_not_decoding():
    result = _run(Provider(_cross_split(), signal="task"))
    verdicts = {(v.subject, v.row): v for v in result.verdicts}
    assert not verdicts[("model", "null_trialstruct")].beats
    # Adding spikes to the task features adds nothing: the gate fails.
    assert not result.gate.beats
    assert "model_with_task does NOT beat null_trialstruct" in str(result)
    assert "not decoding" in str(
        result
    ) and "GATE (model_with_task vs null_trialstruct): NOT PASSED" in str(result)


def test_gate_passes_when_spikes_add_to_the_task_variables():
    # The target needs both: task features alone and spikes alone each explain half.
    result = _run(Provider(_cross_split(), signal="both"))
    verdicts = {(v.subject, v.row): v for v in result.verdicts}
    assert result.gate.beats and "GATE (model_with_task vs null_trialstruct): PASSED" in str(result)
    # Spikes alone don't beat the task variables here; the gate still passes.
    assert not verdicts[("model", "null_trialstruct")].beats
    assert result.summary().loc["model_with_task", "median"] > 0.9


class FoldProvider(Provider):
    """Whole sessions, split into two folds: each half of the bins is tested once."""

    HALVES = ((0, 280), (320, N_BINS - 1))

    def data(self, eid, partition, *, shift=None, pseudo=None):
        z, task, y = self._session(eid)
        if shift is not None:
            y = np.roll(y, -shift)
        ends = np.arange(CONTEXT - 1, N_BINS)
        units = pd.DataFrame(index=[f"u{i}" for i in range(N_UNITS)])
        return SessionData(eid, partition, z, 0, 20, units, ends, CONTEXT, y[ends], task[ends])

    def folds(self):
        return [_FoldView(self, k) for k in range(2)]


class _FoldView:
    def __init__(self, parent, k):
        eids = list(parent.split.partitions["test"])
        self.parent, self.k = parent, k
        self.split = type("FoldSplit", (), {"partitions": {"train": eids, "test": eids}})()

    def data(self, eid, partition, **labels):
        d = self.parent.data(eid, partition, **labels)
        half = FoldProvider.HALVES[self.k if partition == "test" else 1 - self.k]
        keep = (d.ends - CONTEXT + 1 >= half[0]) & (d.ends <= half[1])
        return dataclasses.replace(
            d, ends=d.ends[keep], y=d.y[keep], task_features=d.task_features[keep]
        )


def test_folds_are_fit_and_scored_separately_per_session():
    provider = FoldProvider(_within_split(["e00", "e01", "e10", "e11", "e20", "e21"]))
    result = _run(provider, ceiling=None, n_shifts=5)
    assert result.n_folds == 2 and "scored per fold" in result.report()
    # Each half is held out once: every sample whose window fits inside one half.
    per_half = [(b - a + 1) - CONTEXT + 1 for a, b in FoldProvider.HALVES]
    model = result.per_session["model"]
    assert (model["n_samples"] == sum(per_half)).all() and (model["n_folds"] == 2).all()
    assert result.summary().loc["model", "median"] > 0.9


def test_fold_scores_are_averaged_never_pooled():
    # Fold 0 ranks its samples perfectly around a low offset, fold 1 around a high one.
    # Pooled, the offsets would rank fold 1's zeros above fold 0's ones; per fold they
    # can't. Fold 2's test labels hold one class: no AUROC, so it is left out entirely.
    def fold(auroc, log_loss, n):
        return pd.DataFrame(
            {"auroc": [auroc], "log_loss": [log_loss], "n_samples": [n]},
            index=pd.Index(["e0"], name="eid"),
        )

    table = _mean_over_folds(
        [fold(1.0, 0.2, 10), fold(0.8, 0.4, 30), fold(np.nan, 9.0, 5)], "classification"
    )
    assert table.loc["e0", "auroc"] == pytest.approx(0.9)
    assert table.loc["e0", "log_loss"] == pytest.approx(0.3)  # fold 2 excluded here too
    assert table.loc["e0", "n_samples"] == 40 and table.loc["e0", "n_folds"] == 2


class PseudoProvider(Provider):
    pseudo_sessions = True


def test_pseudo_session_row_uses_the_model_under_test():
    log = []
    provider = PseudoProvider(_cross_split(), log=log)
    result = _run(provider, n_shifts=5, n_pseudo=20)
    assert "null_pseudosession" in result.per_session and result.pseudo.shape == (8, 20)
    assert ("model", "null_pseudosession") in {(v.subject, v.row) for v in result.verdicts}
    assert result.summary().index.tolist().index("null_pseudosession") == 1
    assert {s for _, _, s in log if isinstance(s, tuple)}  # pseudo labels were requested
    assert abs(result.summary().loc["null_pseudosession", "median"]) < 0.2


def test_pseudo_session_counts_are_checked():
    with pytest.raises(ValueError, match="at least 20"):
        _run(PseudoProvider(_cross_split()), n_shifts=5, n_pseudo=10)
    with pytest.raises(ValueError, match="doesn't generate"):
        _run(Provider(_cross_split()), n_shifts=5, n_pseudo=20)


def test_fit_never_receives_test_data():
    model_log, ridge_log = [], []
    result = _run(
        rows={
            "model": lambda: LeastSquares(model_log),
            "baseline_ridge": lambda: LeastSquares(ridge_log),
        }
    )
    # Every fit, including the shuffle fits and the within-session ceiling, sees train only.
    assert model_log and all(partition == "train" for _, partition in model_log)
    # A cross-session row never trains on a test session at all.
    test = set(result.per_session["model"].index)
    assert ridge_log and not {eid for eid, _ in ridge_log} & test


def test_trialstruct_row_never_sees_spikes():
    # TaskLeastSquares asserts z and units are None on every call.
    _run()


def test_shuffle_retrains_the_model_with_one_shift_per_session_and_draw():
    log = []
    provider = Provider(_cross_split(), log=log)
    result = _run(provider, n_shifts=5)
    shifted = [(e, p, s) for e, p, s in log if s is not None]
    assert {s for _, _, s in shifted} and result.n_shifts == 5
    by_session: dict[str, set] = {}
    for eid, _, shift in shifted:
        by_session.setdefault(eid, set()).add(shift)
    assert all(len(shifts) == 5 for shifts in by_session.values())
    assert result.shuffle.shape == (8, 5)


def test_within_session_shift_is_shared_by_train_and_test():
    log = []
    within = _within_split(["e00", "e01", "e10", "e11", "e20", "e21"])
    result = _run(Provider(within, log=log), ceiling=None, n_shifts=5)
    assert result.ceiling_is_model
    seen: dict[tuple, set] = {}
    for eid, partition, shift in log:
        if shift is not None:
            seen.setdefault((eid, shift), set()).add(partition)
    assert all(parts == {"train", "test"} for parts in seen.values())
    assert "ceiling_within = model" in str(result)


def test_every_row_is_required():
    with pytest.raises(TypeError):
        evaluate(Provider(_cross_split()), model=LeastSquares, seed=0)


def test_cross_session_split_needs_a_within_session_ceiling():
    with pytest.raises(ValueError, match="ceiling"):
        _run(ceiling=None)
    wrong = Provider(_within_split(["e00", "e01"]))
    with pytest.raises(ValueError, match="test sessions"):
        _run(ceiling=wrong)


def test_too_few_shifts_are_refused():
    with pytest.raises(ValueError, match="at least 5"):
        _run(n_shifts=4)


def test_sessions_without_valid_shifts_have_no_shuffle_null():
    split = _cross_split()
    blocked = split.partitions["test"][:2]

    class NoShifts(Provider):
        def shifts(self, eid, n_shifts, *, seed):
            if eid in blocked:
                return np.array([], dtype=np.int64)
            return super().shifts(eid, n_shifts, seed=seed)

    result = _run(NoShifts(split))
    null = result.per_session["null_shuffle"]["r2"]
    assert all(np.isnan(null[e]) for e in blocked) and null.notna().sum() == 6
    assert {(v.subject, v.row): v for v in result.verdicts}[
        ("model", "null_shuffle")
    ].n_sessions == 6


def test_bad_predictions_are_refused():
    class Broken(LeastSquares):
        def predict(self, d):
            return np.full(len(d.ends) + 1, 0.0)

    with pytest.raises(ValueError, match="predictions"):
        _run(rows={"model": Broken})


def test_guard_runs_on_the_split():
    stale = dataclasses.replace(_cross_split(), preproc={"fingerprint": "0" * 64, "bin_ms": 20})
    with pytest.raises(ValueError, match="preprocessing"):
        _run(Provider(stale))


def test_classification_uses_auroc():
    rows = {k: Logistic for k in ("model", "baseline_ridge", "baseline_rrr")}

    class SpikesTaskLogistic(Logistic, SpikesTaskLeastSquares):
        pass

    rows["model_with_task"] = SpikesTaskLogistic

    class TaskLogistic(Logistic, TaskLeastSquares):
        pass

    provider = Provider(_cross_split(), kind="classification")
    ceiling = Provider(_within_split(provider.split.partitions["test"]), kind="classification")
    result = evaluate(provider, ceiling=ceiling, seed=0, trialstruct=TaskLogistic, **rows)
    assert result.primary == "auroc"
    assert {(v.subject, v.row): v for v in result.verdicts}[("model", "null_trialstruct")].beats


def test_few_sessions_never_count_as_beating():
    split = held_out_groups(_manifest(), PREPROC, by="subject", n_test=2, n_calibration=0, seed=0)
    provider = Provider(split)
    ceiling = Provider(_within_split(split.partitions["test"]))
    result = _run(provider, ceiling=ceiling)
    verdict = {(v.subject, v.row): v for v in result.verdicts}[("model", "null_trialstruct")]
    assert verdict.n_sessions == 4 and not verdict.beats
    assert "too few sessions" in str(result)


def test_a_null_with_no_valid_session_is_undefined_not_failed():
    class NoShifts(Provider):
        def shifts(self, eid, n_shifts, *, seed):
            return np.array([], dtype=np.int64)

    result = _run(NoShifts(_cross_split()))
    assert result.per_session["null_shuffle"]["r2"].isna().all()
    assert "model vs null_shuffle: undefined" in str(result)
    assert not {(v.subject, v.row): v for v in result.verdicts}[("model", "null_shuffle")].beats


def test_test_predictions_are_kept_for_every_real_label_row():
    # S3: single-session verdicts resample the test predictions, so the contract keeps
    # them: one row per test sample, per fold, for each row fit on the real labels.
    result = _run(n_shifts=5)
    assert set(result.predictions) == {
        "model",
        "model_with_task",
        "baseline_ridge",
        "baseline_rrr",
        "null_trialstruct",
    }
    for row, table in result.predictions.items():
        assert list(table.columns) == ["eid", "fold", "end", "y", "prediction"], row
        counts = table.groupby("eid").size()
        expected = result.per_session[row]["n_samples"].reindex(counts.index)
        assert (counts == expected).all(), row
    # The kept predictions reproduce the row's score.
    model = result.predictions["model"]
    e0 = model[model["eid"] == model["eid"].iloc[0]]
    r2 = 1 - ((e0["y"] - e0["prediction"]) ** 2).sum() / ((e0["y"] - e0["y"].mean()) ** 2).sum()
    assert r2 == pytest.approx(result.per_session["model"].loc[e0["eid"].iloc[0], "r2"])


def test_progress_counts_every_fit_up_to_the_total():
    seen = []
    _run(
        PseudoProvider(_cross_split()),
        n_shifts=5,
        n_pseudo=20,
        progress=lambda d, t: seen.append((d, t)),
    )
    done, totals = zip(*seen)
    # model, model_with_task, ridge, rrr, trialstruct, 5 shifts, 20 pseudo, ceiling.
    assert set(totals) == {5 + 5 + 20 + 1}
    assert list(done) == list(range(32))  # 0 before the first fit, 31 when done


def test_baseline_rrr_can_be_left_out_only_with_a_reason():
    provider = Provider(_within_split(["e00", "e01"]))
    with pytest.raises(ValueError, match="reason"):
        _run(provider, ceiling=None, n_shifts=5, rows={"baseline_rrr": None})
    result = _run(
        provider,
        ceiling=None,
        n_shifts=5,
        rows={"baseline_rrr": None},
        not_run={"baseline_rrr": "multi-session by design"},
    )
    assert "baseline_rrr" not in result.per_session and "baseline_rrr" not in result.predictions
    assert result.not_run == {"baseline_rrr": "multi-session by design"}
    assert "baseline_rrr: not run (multi-session by design)" in result.report()
