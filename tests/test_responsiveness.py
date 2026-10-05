import numpy as np
import pytest
from scipy.stats import kstest

from unitwave.analysis.psth import alternate_halves
from unitwave.analysis.responsiveness import (
    ResponseConfig,
    benjamini_hochberg,
    circular_statistics,
    load_response_config,
    responsiveness,
)

CFG = ResponseConfig(
    baseline_window=(-0.2, 0.0),
    response_window=(0.0, 0.3),
    grid_s=0.001,
    min_shift_s=10.0,
    alpha=0.05,
    max_self_overlap=0.5,
    max_realigned_fraction=0.25,
    max_close_fraction=0.5,
)
# Irregular event times, so no shift other than 0 lines all events up again.
EVENTS = np.array([1.0, 3.7, 7.1, 9.4, 14.2, 17.9, 21.3, 26.0, 29.8, 33.1])


def test_default_config():
    # The tests use a 1 ms grid; the app's default is 5 ms (configs/analysis.yaml).
    assert load_response_config() == ResponseConfig(**{**CFG.__dict__, "grid_s": 0.005})


def test_one_spike_after_every_event_has_the_smallest_possible_p():
    # One spike 50 ms after each event: response rate 1 / 0.3 s, baseline 0.
    result = responsiveness({"u": EVENTS + 0.05}, ["u"], EVENTS, CFG)
    row = result.loc["u"]
    assert row["statistic_hz"] == pytest.approx(1 / 0.3)
    assert row["p"] == 1 / (1 + row["n_shifts"])
    assert row["n_trials"] == 10 and row["n_excluded"] == 0


def test_the_fft_null_equals_counting_spikes_at_each_shift():
    rng = np.random.default_rng(0)
    spikes = np.sort(rng.uniform(0.5, 34, 400))
    stats, allowed, (x, bins, _c0) = circular_statistics({"u": spikes}, ["u"], EVENTS, CFG)
    n_bins = x.shape[1]
    b0, b1 = (round(t / CFG.grid_s) for t in CFG.baseline_window)
    r0, r1 = (round(t / CFG.grid_s) for t in CFG.response_window)
    for d in [0, 1, 12345, n_bins - 7]:
        idx = (bins[:, None] + d) % n_bins
        resp = [x[0, (i + np.arange(r0, r1)) % n_bins].sum() for i in idx[:, 0]]
        base = [x[0, (i + np.arange(b0, b1)) % n_bins].sum() for i in idx[:, 0]]
        direct = np.mean(np.array(resp) / 0.3 - np.array(base) / 0.2)
        assert stats[0, d] == pytest.approx(direct, abs=1e-9)
    # Shifts closer than min_shift_s to zero, either way round the circle, are excluded.
    assert not allowed[0] and not allowed[9999] and allowed[10000]
    assert not allowed[n_bins - 9999] and allowed[n_bins - 10000]


def test_null_units_give_uniform_p_values():
    rng = np.random.default_rng(1)
    events = np.sort(rng.uniform(5, 115, 40))
    units = {f"u{i}": np.sort(rng.uniform(0, 120, rng.poisson(600))) for i in range(200)}
    result = responsiveness(units, list(units), events, CFG)
    assert kstest(result["p"], "uniform").pvalue > 0.01


def test_benjamini_hochberg_controls_false_discoveries_on_null_data():
    # Under the complete null, any rejection is a false discovery: it should happen in
    # about alpha of the datasets. 40 datasets of 50 null units; expect ~2.
    rng = np.random.default_rng(2)
    with_any = 0
    for _ in range(40):
        events = np.sort(rng.uniform(5, 115, 40))
        units = {f"u{i}": np.sort(rng.uniform(0, 120, rng.poisson(600))) for i in range(50)}
        with_any += bool(responsiveness(units, list(units), events, CFG)["responsive"].any())
    assert with_any <= 5


def test_benjamini_hochberg_by_hand():
    # Sorted p 0.01, 0.03, 0.04, 0.2 with m = 4: raw 0.04, 0.06, 0.0533, 0.2, then the
    # running minimum from the largest down.
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.2]))
    np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def test_is_deterministic_and_refuses_bad_windows():
    rng = np.random.default_rng(3)
    units = {"u": np.sort(rng.uniform(0, 35, 300))}
    first = responsiveness(units, ["u"], EVENTS, CFG)
    assert first.equals(responsiveness(units, ["u"], EVENTS, CFG))
    with pytest.raises(ValueError, match="overlap"):
        responsiveness(
            units, ["u"], EVENTS, CFG.__class__(**{**CFG.__dict__, "response_window": (-0.1, 0.3)})
        )
    with pytest.raises(ValueError, match="grid"):
        responsiveness(
            units,
            ["u"],
            EVENTS,
            CFG.__class__(**{**CFG.__dict__, "response_window": (0.0, 0.3005)}),
        )


def test_missing_events_are_excluded_and_counted():
    events = np.append(EVENTS, np.nan)
    row = responsiveness({"u": EVENTS + 0.05}, ["u"], events, CFG).loc["u"]
    assert row["n_trials"] == 10 and row["n_excluded"] == 1


def test_alternate_halves_split_valid_trials_in_order():
    events = np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0])
    sort_on, show = alternate_halves(events)
    np.testing.assert_array_equal(sort_on, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(show, [2.0, 4.0])
