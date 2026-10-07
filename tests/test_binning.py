from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unitwave.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
)
from unitwave.env import env
from unitwave.preprocess import binning
from unitwave.preprocess.binning import (
    PreprocConfig,
    bin_spikes,
    load_preproc_config,
    preprocess_session,
)
from unitwave.qc.units import UnitQC

QC = UnitQC(min_label=1.0, exclude_regions=("void", "root"), min_firing_rate_hz=0.1)


def _session(spikes: dict, time_bounds=(0.0, 1.0)) -> Session:
    ids = list(spikes)
    units = pd.DataFrame(
        {f: [1.0] * len(ids) for f in UNIT_FIELDS}, index=pd.Index(ids, name="unit_id")
    )
    units["acronym"] = "CA1"
    units["firing_rate"] = 5.0
    present = {f"trials.{f}" for f in TRIAL_FIELDS} | {f"units.{f}" for f in UNIT_FIELDS}
    trials = pd.DataFrame({f: [0.5] for f in TRIAL_FIELDS})
    trials["intervals_0"], trials["intervals_1"] = 0.45, 0.6  # the task period for QC
    return Session(
        eid="e",
        time_bounds=time_bounds,
        spikes={u: np.asarray(t, dtype=float) for u, t in spikes.items()},
        units=units,
        trials=trials,
        behaviour={},
        available=Capabilities(
            present=frozenset(present),
            missing={f"behaviour.{f}": "fixture" for f in BEHAVIOUR_FIELDS},
        ),
    )


def test_default_config_is_20_ms_with_the_qc_config():
    cfg = load_preproc_config()
    assert cfg.bin_ms == 20
    assert cfg.qc == QC


def test_bin_width_must_divide_one_second():
    with pytest.raises(ValueError, match="divide"):
        PreprocConfig(bin_ms=30, qc=QC)
    with pytest.raises(ValueError, match="divide"):
        PreprocConfig(bin_ms=0, qc=QC)


def test_counts_and_half_open_bins():
    # 20 ms bins: [0, .02), [.02, .04), ... A spike exactly on an edge goes up.
    s = _session({"a": [0.001, 0.019, 0.02, 0.06], "b": [0.5]})
    b = bin_spikes(s, bin_ms=20)
    assert b.counts.shape == (2, 51)
    assert b.counts.dtype == np.uint16
    assert b.first_bin == 0
    assert b.counts[0, :4].tolist() == [2, 1, 0, 1]
    assert b.counts[1, 25] == 1


def test_bins_are_anchored_at_zero_not_at_the_session_start():
    early = bin_spikes(_session({"a": [0.51]}, time_bounds=(0.0, 1.0)), bin_ms=20)
    late = bin_spikes(_session({"a": [0.51]}, time_bounds=(0.45, 1.0)), bin_ms=20)
    assert late.first_bin == 22
    assert early.counts[0, 25] == late.counts[0, 25 - late.first_bin] == 1


def test_every_spike_lands_in_the_bin_that_contains_it():
    rng = np.random.default_rng(1)
    times = np.sort(rng.uniform(0.0, 1.0, 1000))
    b = bin_spikes(_session({"a": times}), bin_ms=10)
    assert int(b.counts.sum()) == 1000
    starts = b.bin_start_times()
    idx = np.searchsorted(starts, times, side="right") - 1
    np.testing.assert_array_equal(np.bincount(idx, minlength=b.n_bins), b.counts[0])


def test_unit_with_no_spikes_is_a_zero_row():
    b = bin_spikes(_session({"a": [0.1], "b": []}), bin_ms=20)
    assert b.unit_ids == ("a", "b")
    assert b.counts[1].sum() == 0


def test_overflow_raises_instead_of_wrapping():
    with pytest.raises(ValueError, match="uint16"):
        bin_spikes(_session({"a": np.full(70_000, 0.5)}), bin_ms=20)


def test_binning_is_byte_identical_for_the_same_input():
    rng = np.random.default_rng(2)
    spikes = {f"u{i}": np.sort(rng.uniform(0, 1, 200)) for i in range(5)}
    cfg = PreprocConfig(bin_ms=20, qc=QC)
    first, second = preprocess_session(_session(spikes), cfg), preprocess_session(
        _session(spikes), cfg
    )
    assert first.counts.tobytes() == second.counts.tobytes()
    assert first.fingerprint == second.fingerprint == cfg.fingerprint()


def test_fingerprint_changes_with_bin_width_qc_or_code_version(monkeypatch):
    base = PreprocConfig(bin_ms=20, qc=QC).fingerprint()
    assert PreprocConfig(bin_ms=10, qc=QC).fingerprint() != base
    assert PreprocConfig(bin_ms=20, qc=UnitQC(1.0, ("void", "root"), 1.0)).fingerprint() != base
    monkeypatch.setattr(binning, "PREPROC_VERSION", binning.PREPROC_VERSION + 1)
    assert PreprocConfig(bin_ms=20, qc=QC).fingerprint() != base


def test_preprocess_applies_qc_before_binning():
    # b's only spike is outside the task period [0.45, 0.6] s, so it fails the rate floor.
    s = _session({"a": [0.5], "b": [0.2]})
    b = preprocess_session(s, PreprocConfig(bin_ms=20, qc=QC))
    assert b.unit_ids == ("a",)


DATA_ROOT = Path(env("DATA_ROOT", "~/data/neurodecoder")).expanduser()
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
BWM = DATA_ROOT / "bwm_compressed/bwm_ephys/1.2.1"
NWB = (
    DATA_ROOT
    / "dandi/000409/sub-DY-016"
    / f"sub-DY-016_ses-{EID}_desc-processed_behavior+ecephys.nwb"
)
ONE_CACHE = DATA_ROOT / "one"
CFG = PreprocConfig(bin_ms=20, qc=QC)


@pytest.fixture(scope="module")
def nwb_binned():
    from unitwave.data.backends.dandi_nwb import load_session_nwb

    return preprocess_session(load_session_nwb(NWB), CFG)


@pytest.mark.skipif(not NWB.exists(), reason="NWB file not available")
def test_real_round_trip_counts_equal_spike_counts(nwb_binned):
    from unitwave.data.backends.dandi_nwb import load_session_nwb
    from unitwave.qc.units import apply_unit_qc

    session = apply_unit_qc(load_session_nwb(NWB), QC)
    assert nwb_binned.counts.shape[0] == 390  # 397 before the task-period rate rule
    per_unit = nwb_binned.counts.sum(axis=1, dtype=np.int64)
    expected = np.array([len(session.spikes[u]) for u in nwb_binned.unit_ids])
    np.testing.assert_array_equal(per_unit, expected)


@pytest.mark.skipif(
    not (NWB.exists() and (ONE_CACHE / "danlab").exists()), reason="NWB and ONE data needed"
)
def test_real_one_and_nwb_bin_identically(nwb_binned):
    from unitwave.data.backends.one_backend import load_session_one, make_one

    one = preprocess_session(load_session_one(EID, make_one(ONE_CACHE)), CFG)
    assert one.unit_ids == nwb_binned.unit_ids
    assert one.first_bin == nwb_binned.first_bin
    assert one.counts.tobytes() == nwb_binned.counts.tobytes()


@pytest.mark.skipif(not (NWB.exists() and BWM.exists()), reason="NWB and BWM data needed")
def test_real_bwm_differs_from_nwb_only_by_edge_rounding(nwb_binned):
    from unitwave.data.backends.bwm_compressed import load_session_bwm

    bwm = preprocess_session(load_session_bwm(EID, BWM), CFG)
    assert bwm.unit_ids == nwb_binned.unit_ids
    offset = bwm.first_bin - nwb_binned.first_bin
    n = min(bwm.n_bins, nwb_binned.n_bins - offset) if offset >= 0 else None
    assert offset == 0 and n is not None
    a, b = bwm.counts[:, :n].astype(np.int64), nwb_binned.counts[:, :n].astype(np.int64)
    np.testing.assert_array_equal(a.sum(axis=1), b.sum(axis=1))
    # BWM rounds spike times to 100 us ticks, so only spikes within 50 us of an edge
    # can change bin: at most 100 us / 20 ms = 0.5% of spikes.
    moved = np.abs(a - b).sum() / 2
    assert moved / b.sum() <= 0.005
