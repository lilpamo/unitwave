"""Responsiveness refuses presentations its shift null can't test (the user's decision,
2026-10-05; docs/NEGATIVE_RESULTS.md, 2026-10-02 and 2026-10-05).

With strictly periodic presentations, shifts a whole number of periods apart give
almost the same statistic: kept, they tie with the true value and the test has no power;
dropped, too few distinct draws remain and the test invents responses (4-12 of 200 null
units here). So periodic presentations (more than `max_realigned_fraction` of shifts
lining them up again) and back-to-back ones (closer than the windows) are refused,
saying why. The null itself is unchanged. Spike trains here are test inputs, except the
last tests, which read the cached IBL session and the downloaded Allen file."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from unitwave.analysis.responsiveness import (
    circular_statistics,
    load_response_config,
    responsiveness,
)

CFG = replace(load_response_config(), grid_s=0.005)
RNG = np.random.default_rng(7)
DESIGNS = {
    "near-regular trials (MC_Maze-like, CV 0.11)": 5.0 + np.cumsum(RNG.normal(3.0, 0.33, 200)),
    "periodic blocks with gaps (drifting gratings)": np.concatenate(
        [b + 3.003 * np.arange(75) for b in (5.0, 400.0, 800.0)]
    ),
    "irregular trials (IBL-like)": 5.0 + np.cumsum(RNG.uniform(1.5, 8.0, 300)),
}


def _null_units(rng, t1, n=200):
    return {
        f"u{i}": np.sort(rng.uniform(0, t1, rng.poisson(rng.uniform(2, 20) * t1))) for i in range(n)
    }


def test_periodic_presentations_are_refused_saying_why():
    flashes = 5.0 + 2.002 * np.arange(300)
    with pytest.raises(ValueError, match=r"periodic \(about 2 s apart\): 50% of the null's shifts"):
        responsiveness({"u": np.array([1.0])}, ["u"], flashes, CFG)


def test_back_to_back_presentations_are_refused_saying_why():
    static = 5.0 + 0.25 * np.arange(2400)
    with pytest.raises(ValueError, match="0.25 s apart, closer than the test's windows"):
        responsiveness({"u": np.array([1.0])}, ["u"], static, CFG)


@pytest.mark.parametrize("name", list(DESIGNS))
def test_other_designs_keep_the_test_and_its_calibration(name):
    events = DESIGNS[name]
    _, allowed, _ = circular_statistics({"u": np.array([1.0])}, ["u"], events, CFG)
    n, shift = allowed.size, np.arange(allowed.size)
    # The null is today's: every shift past the minimum.
    assert np.array_equal(
        allowed, np.minimum(shift, n - shift) >= round(CFG.min_shift_s / CFG.grid_s)
    )
    units = _null_units(np.random.default_rng(1), events.max() + 10)
    result = responsiveness(units, list(units), events, CFG)
    assert result["responsive"].sum() <= 1  # BH at 0.05 over 200 units ignoring the events
    assert (result["p"] < 0.05).mean() < 0.1


def test_ibl_is_never_refused_and_its_null_unchanged():
    from unitwave.analysis.events import event_times
    from unitwave.data.load import load_session

    try:
        session = load_session("d23a44ef-1402-4ed7-97f5-47e9a7a504d9", "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")
    cfg = load_response_config()
    for event in ("stim_on", "first_movement", "feedback_reward", "feedback_error"):
        events = event_times(session.trials, event)
        _, allowed, _ = circular_statistics({"u": np.array([1.0])}, ["u"], events, cfg)
        n, shift = allowed.size, np.arange(allowed.size)
        minimum = np.minimum(shift, n - shift) >= round(cfg.min_shift_s / cfg.grid_s)
        assert np.array_equal(allowed, minimum), event


ALLEN = Path.home() / "data/neurodecoder/dandi/000021/sub-707296975/sub-707296975_ses-721123822.nwb"


@pytest.mark.skipif(not ALLEN.exists(), reason="the Allen session isn't downloaded")
def test_allen_flashes_and_static_gratings_are_refused_drifting_gratings_tested():
    from unitwave.analysis.events import event_times
    from unitwave.analysis.tasks import load_task
    from unitwave.studio.project import Source, load_source

    verdicts = {}
    for task in ("allen_flashes", "allen_static_gratings", "allen_drifting_gratings"):
        session, _ = load_source(
            Source(kind="nwb", file=str(ALLEN), layout="allen_visual_coding", task=task)
        )
        events = event_times(session.trials, "stim_on", load_task(task))
        try:
            circular_statistics({"u": np.array([1.0])}, ["u"], events, load_response_config())
            verdicts[task] = "tested"
        except ValueError as e:
            verdicts[task] = "periodic" if "periodic" in str(e) else "back to back"
    assert verdicts == {
        "allen_flashes": "periodic",
        "allen_static_gratings": "back to back",
        "allen_drifting_gratings": "tested",
    }
