from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unitwave.data.manifest import Manifest
from unitwave.env import env
from unitwave.preprocess.binning import BinnedSpikes, PreprocConfig
from unitwave.preprocess.normalize import fit_normalizer
from unitwave.preprocess.windows import unwindow, window_plan
from unitwave.qc.units import UnitQC
from unitwave.splits.registry import held_out_session, within_session

PREPROC = PreprocConfig(bin_ms=20, qc=UnitQC(1.0, ("void", "root"), 0.1))
FP = PREPROC.fingerprint()
PROVENANCE = {"manifest_version": 1, "sources": {"bwm_ephys": "1.2.1", "bwm_behavior": "2.0.0"}}
EXPECT = {"manifest_provenance": PROVENANCE, "preproc_fingerprint": FP}
N_BINS = 5000  # 100 s from t = 0


def _manifest() -> Manifest:
    rows = [
        {"eid": f"e{s}{k}", "subject": f"s{s}", "lab": "lab"} for s in range(3) for k in range(2)
    ]
    return Manifest(
        sessions=pd.DataFrame(rows), insertions=pd.DataFrame(), provenance=dict(PROVENANCE)
    )


def _binned(eid="e00", seed=0, n_units=3, first_bin=0) -> BinnedSpikes:
    counts = np.random.default_rng(seed).poisson(0.5, (n_units, N_BINS)).astype(np.uint16)
    return BinnedSpikes(eid, tuple(f"u{i}" for i in range(n_units)), counts, 20, first_bin, FP)


def _within():
    # Trials every 4 s from 10 s: train block [10, 73] s = bins 500..3650,
    # test block [78, 89] s = bins 3900..4450.
    starts = 10.0 + 4.0 * np.arange(20)
    trials = {"e00": pd.DataFrame({"intervals_0": starts, "intervals_1": starts + 3.0})}
    return within_session(_manifest(), trials, PREPROC, train_fraction=0.8, gap_s=2.0)


def _bins(ends, context):
    return {int(b) for e in ends for b in range(e - context + 1, e + 1)}


def test_no_temporal_leakage():
    plan = window_plan(_within(), context_bins=50, stride_bins=1, **EXPECT)
    b = _binned()
    train, test = (_bins(plan.ends(b, p), 50) for p in ("train", "test"))
    assert train and test
    assert min(train) == 500 and max(train) == 3650
    assert min(test) == 3900 and max(test) == 4450
    assert not train & test
    assert min(test) - max(train) - 1 >= 50


def test_plan_refuses_a_context_longer_than_the_gap():
    # The gap is 249 bins; a 300-bin context could see across it.
    with pytest.raises(ValueError, match="gap"):
        window_plan(_within(), context_bins=300, stride_bins=1, **EXPECT)


def test_windows_are_the_context_before_each_end():
    plan = window_plan(_within(), context_bins=50, stride_bins=25, **EXPECT)
    b = _binned()
    z = b.counts.astype(np.float32)
    ends = plan.ends(b, "train")
    assert ends[0] == 549 and np.all(np.diff(ends) == 25) and ends[-1] <= 3650
    x = plan.extract(z, b, "train")
    assert x.shape == (len(ends), 3, 50) and x.dtype == np.float32
    for i in (0, len(ends) // 2, len(ends) - 1):
        np.testing.assert_array_equal(x[i], z[:, ends[i] - 49 : ends[i] + 1])


def test_ends_respect_the_binned_first_bin():
    plan = window_plan(
        held_out_session(_manifest(), PREPROC, seed=0), context_bins=50, stride_bins=50, **EXPECT
    )
    eid = plan.split.partitions["test"][0]
    b = _binned(eid, first_bin=1000)
    ends = plan.ends(b, "test")
    # Cross-session splits allow the whole binned session: bins 1000..5999.
    assert ends[0] == 1049 and ends[-1] == 5999
    x = plan.extract(b.counts.astype(np.float32), b, "test", ends[:1])
    np.testing.assert_array_equal(x[0], b.counts[:, :50])


def test_a_session_only_windows_in_its_own_partition():
    plan = window_plan(
        held_out_session(_manifest(), PREPROC, seed=0), context_bins=50, stride_bins=50, **EXPECT
    )
    eid = plan.split.partitions["test"][0]
    with pytest.raises(ValueError, match="not in train"):
        plan.ends(_binned(eid), "train")


def test_extract_refuses_windows_that_leave_the_block():
    plan = window_plan(_within(), context_bins=50, stride_bins=1, **EXPECT)
    b = _binned()
    z = b.counts.astype(np.float32)
    for bad_end in (548, 3651, 3700):  # starts before the block, ends after it, in the gap
        with pytest.raises(ValueError, match="outside"):
            plan.extract(z, b, "train", np.array([bad_end]))


def test_extract_checks_the_array_matches_the_session():
    plan = window_plan(_within(), context_bins=50, stride_bins=1, **EXPECT)
    b = _binned()
    with pytest.raises(ValueError, match="shape"):
        plan.extract(np.zeros((2, N_BINS), dtype=np.float32), b, "train")


def test_block_outside_the_binned_data_raises():
    plan = window_plan(_within(), context_bins=50, stride_bins=1, **EXPECT)
    short = BinnedSpikes("e00", ("u0",), np.zeros((1, 3000), np.uint16), 20, 0, FP)
    with pytest.raises(ValueError, match="binned"):
        plan.ends(short, "train")


def test_context_and_stride_must_be_positive_integers():
    with pytest.raises(ValueError, match="stride"):
        window_plan(_within(), context_bins=50, stride_bins=0, **EXPECT)


def test_unwindow_inverts_extract_with_overlap():
    plan = window_plan(_within(), context_bins=50, stride_bins=7, **EXPECT)
    b = _binned()
    z = b.counts.astype(np.float32)
    ends = plan.ends(b, "train")
    back = unwindow(plan.extract(z, b, "train"), ends, b.first_bin, b.n_bins)
    covered = ~np.isnan(back[0])
    assert covered.sum() == ends[-1] - 500 + 1
    np.testing.assert_array_equal(back[:, covered], z[:, covered])


def test_synthetic_counts_round_trip_through_normalised_windows():
    split, b = _within(), _binned()
    norm = fit_normalizer(split, {"e00": b}, PREPROC)
    plan = window_plan(split, context_bins=50, stride_bins=50, **EXPECT)
    z = norm.transform(b)
    for partition in ("train", "test"):
        ends = plan.ends(b, partition)
        back = unwindow(plan.extract(z, b, partition), ends, b.first_bin, b.n_bins)
        covered = ~np.isnan(back[0])
        counts = norm.inverse_transform(np.nan_to_num(back), "e00", b.unit_ids)
        np.testing.assert_allclose(counts[:, covered], b.counts[:, covered], atol=1e-4)


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EPHYS = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
BEHAVIOUR = DATA_ROOT / "bwm_compressed/bwm_behavior/2.0.0"
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"


@pytest.mark.skipif(
    not (EPHYS.exists() and BEHAVIOUR.exists()), reason="BWM releases not available"
)
def test_real_session_round_trips_to_windows_and_back():
    """Phase 2 success criterion: a session to tensors and back to spike counts."""
    from unitwave.data.backends.bwm_compressed import load_session_bwm
    from unitwave.data.manifest import build_manifest
    from unitwave.preprocess.binning import preprocess_session

    manifest = build_manifest(EPHYS, BEHAVIOUR)
    session = load_session_bwm(EID, EPHYS)
    split = within_session(manifest, {EID: session.trials}, PREPROC, train_fraction=0.8, gap_s=2.0)
    binned = preprocess_session(session, PREPROC)
    norm = fit_normalizer(split, {EID: binned}, PREPROC)
    plan = window_plan(
        split, context_bins=50, stride_bins=50, manifest_provenance=manifest.provenance
    )
    z = norm.transform(binned)
    for partition in ("train", "test"):
        ends = plan.ends(binned, partition)
        windows = plan.extract(z, binned, partition)
        assert windows.shape == (len(ends), len(binned.unit_ids), 50)
        back = unwindow(windows, ends, binned.first_bin, binned.n_bins)
        covered = ~np.isnan(back[0])
        counts = norm.inverse_transform(np.nan_to_num(back), EID, binned.unit_ids)
        assert np.array_equal(np.rint(counts[:, covered]), binned.counts[:, covered])
        np.testing.assert_allclose(counts[:, covered], binned.counts[:, covered], atol=1e-3)
