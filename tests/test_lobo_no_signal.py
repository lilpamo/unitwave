"""No-signal check for block under leave_one_block_out (added after the confirmation set).

Spikes are Poisson noise, independent of everything; block labels come from IBL's
generator. Through the real split, provider, decoders and contract, every row that
sees only noise must score AUROC ~0.5. The first version of the split (one block per
fold) scored ~0 here: each fold's training class balance moved against its test label
(docs/NEGATIVE_RESULTS.md). Pseudo-session and shifted labels can make a fold's test
set mostly one class, so the nulls are checked, not just the model.
"""

import numpy as np
import pandas as pd
import pytest

from unitwave.data.manifest import Manifest, manifest_versions
from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
    TimeSeries,
)
from unitwave.evaluation.contract import evaluate
from unitwave.evaluation.data import SplitData
from unitwave.evaluation.nulls import generate_pseudo_blocks
from unitwave.models.baselines.features import load_baseline_config
from unitwave.models.baselines.linear import (
    LogisticDecoder,
    SpikesAndTaskLogistic,
    TrialStructureLogistic,
)
from unitwave.models.baselines.rrr import RRRClassification
from unitwave.preprocess.binning import load_preproc_config
from unitwave.splits.registry import leave_one_block_out

PREPROC = load_preproc_config()
N_TRIALS, TRIAL_S, N_UNITS, RATE_HZ = 450, 4.5, 8, 5.0
EIDS = tuple(f"noise{i}" for i in range(6))
NOISE_ROWS = ("model", "model_with_task", "null_shuffle", "null_pseudosession")
TOLERANCE = 0.07  # on the median over sessions; the old split scored ~0, pooled pairs ~0.37


def _trials(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = 10.0 + TRIAL_S * np.arange(N_TRIALS)
    t = pd.DataFrame(index=range(N_TRIALS))
    t["intervals_0"], t["intervals_1"] = start, start + TRIAL_S - 0.5
    t["stimOn_times"] = t["goCue_times"] = start + 0.5
    t["firstMovement_times"] = t["stimOn_times"] + 0.3
    t["response_times"] = t["firstMovement_times"] + 0.2
    t["feedback_times"] = t["response_times"] + 0.01
    t["stimOff_times"] = t["feedback_times"] + 1.0
    t["choice"] = rng.choice([-1.0, 1.0], N_TRIALS)
    t["feedbackType"] = rng.choice([-1.0, 1.0], N_TRIALS)
    right = rng.random(N_TRIALS) < 0.5
    t["contrastLeft"] = np.where(right, np.nan, 0.5)
    t["contrastRight"] = np.where(right, 0.5, np.nan)
    # Real IBL block structure: 90 unbiased trials, then exponential-length blocks.
    t["probabilityLeft"] = generate_pseudo_blocks(N_TRIALS, seed=10_000 + seed)
    t = t[list(TRIAL_FIELDS)]
    t["bwm_include"] = True
    return t


def _session(eid: str) -> Session:
    seed = EIDS.index(eid)
    rng = np.random.default_rng(seed)
    duration = 20.0 + TRIAL_S * N_TRIALS
    ids = [f"p_{i}" for i in range(N_UNITS)]
    units = pd.DataFrame(
        {f: [1.0] * N_UNITS for f in UNIT_FIELDS}, index=pd.Index(ids, name="unit_id")
    )
    units["acronym"] = "CA1"
    n_spikes = int(RATE_HZ * duration)
    spikes = {u: np.sort(rng.uniform(0.0, duration, n_spikes)) for u in ids}
    t = 0.0031 + np.arange(int((duration - 0.01) * 100)) / 100
    present = (
        {f"trials.{f}" for f in TRIAL_FIELDS}
        | {f"units.{f}" for f in UNIT_FIELDS}
        | {"behaviour.wheel"}
    )
    return Session(
        eid=eid,
        time_bounds=(0.0, duration),
        spikes=spikes,
        units=units,
        trials=_trials(seed),
        behaviour={"wheel": TimeSeries(t, np.zeros_like(t))},
        available=Capabilities(
            present=frozenset(present),
            missing={f"behaviour.{f}": "fixture" for f in BEHAVIOUR_FIELDS if f != "wheel"},
        ),
    )


@pytest.fixture(scope="module")
def noise_result():
    sessions = {e: _session(e) for e in EIDS}
    rows = [{"eid": e, "subject": f"s{i}", "lab": "lab"} for i, e in enumerate(EIDS)]
    manifest = Manifest(
        sessions=pd.DataFrame(rows), insertions=pd.DataFrame(), provenance=manifest_versions()
    )
    split = leave_one_block_out(
        manifest, {e: s.trials for e, s in sessions.items()}, PREPROC, gap_s=2.0
    )
    provider = SplitData(split, "block", load=sessions.__getitem__)
    assert provider.pseudo_sessions and provider.n_folds >= 2
    cfg = load_baseline_config()
    decoders = dict(
        model=lambda: LogisticDecoder(cfg.trial_chunks, cfg.cv),
        model_with_task=lambda: SpikesAndTaskLogistic(
            cfg.trial_chunks, cfg.cv, cfg.task_penalty_ratios
        ),
        baseline_ridge=lambda: LogisticDecoder(cfg.trial_chunks, cfg.cv),
        baseline_rrr=lambda: RRRClassification(provider.context_bins, cfg.cv),
        trialstruct=lambda: TrialStructureLogistic(cfg.cv),
    )
    return evaluate(provider, ceiling=None, seed=0, n_shifts=5, n_pseudo=20, **decoders)


@pytest.mark.parametrize("row", NOISE_ROWS)
def test_noise_scores_chance_on_every_row(noise_result, row):
    auroc = noise_result.per_session[row]["auroc"]
    assert auroc.notna().sum() == len(EIDS)
    median = float(auroc.median())
    assert abs(median - 0.5) < TOLERANCE, (
        f"{row}: median AUROC {median:.3f} on pure noise; per session "
        f"{auroc.round(3).to_dict()}"
    )
