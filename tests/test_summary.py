"""Across-session region summaries (S5). Unit tables here are test inputs, never data."""

from math import comb

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from unitwave.analysis.summary import SummaryConfig, load_summary_config, region_summary

CFG = SummaryConfig(level="Beryl", min_sessions=2, alpha=0.05)


def _units(rows) -> pd.DataFrame:
    """rows: (eid, region, labelled) per unit."""
    t = pd.DataFrame(rows, columns=["eid", "region", "labelled"])
    t.insert(1, "unit_id", [f"u{i}" for i in range(len(t))])
    return t


def test_default_config():
    assert load_summary_config() == SummaryConfig(level="Beryl", min_sessions=5, alpha=0.05)


def test_aggregation_by_hand_on_two_sessions():
    # Session a: 10 units, 4 labelled; region R has 3 of them, 2 labelled.
    # Session b: 6 units, 3 labelled; region R has 2 of them, none labelled.
    a = [("a", "R", True)] * 2 + [("a", "R", False)] + [("a", "S", True)] * 2
    a += [("a", "S", False)] * 5
    b = [("b", "R", False)] * 2 + [("b", "S", True)] * 3 + [("b", "S", False)]
    regions, sessions = region_summary(_units(a + b), CFG)
    r = regions.loc["R"]
    assert (r["n_sessions"], r["n_units"], r["observed"]) == (2, 5, 2)
    assert r["expected"] == pytest.approx(3 * 4 / 10 + 2 * 3 / 6)
    # The null: in each session the region's labelled count is hypergeometric; the
    # sum's distribution is their convolution.
    pa = {k: comb(4, k) * comb(6, 3 - k) / comb(10, 3) for k in range(4)}
    pb = {k: comb(3, k) * comb(3, 2 - k) / comb(6, 2) for k in range(3)}
    total = {}
    for i, x in pa.items():
        for j, y in pb.items():
            total[i + j] = total.get(i + j, 0) + x * y
    p_high = sum(v for t, v in total.items() if t >= 2)
    p_low = sum(v for t, v in total.items() if t <= 2)
    assert r["p_high"] == pytest.approx(p_high) and r["p_low"] == pytest.approx(p_low)
    assert r["p"] == pytest.approx(min(1.0, 2 * min(p_high, p_low)))
    # Every region gets its per-session rows, never only the pooled numbers (§5).
    s = sessions.set_index(["region", "eid"])
    assert s.loc[("R", "a"), ["n_units", "n_labelled"]].tolist() == [3, 2]
    assert s.loc[("R", "a"), "fraction"] == pytest.approx(2 / 3)
    assert s.loc[("R", "a"), "fraction_rest"] == pytest.approx(2 / 7)
    assert s.loc[("R", "b"), "fraction"] == 0 and s.loc[("R", "b"), "fraction_rest"] == 0.75
    assert set(sessions["region"]) == {"R", "S"}


def test_a_region_below_the_session_minimum_is_refused_with_the_reason():
    rows = [("a", "R", True), ("a", "S", False), ("b", "S", True), ("b", "S", False)]
    regions, sessions = region_summary(_units(rows), CFG)
    r = regions.loc["R"]
    assert r["n_sessions"] == 1 and np.isnan(r["p"]) and np.isnan(r["q"]) and not r["claim"]
    assert r["refused"] == "in 1 session; a verdict needs at least 2 (configs/summary.yaml)"
    assert len(sessions[sessions["region"] == "R"]) == 1  # its session rows are still shown


def test_units_without_a_region_count_in_their_session_but_are_never_a_region():
    rows = [("a", "R", True), ("a", None, False), ("a", "root", False)]
    rows += [("b", "R", False), ("b", None, True)]
    regions, sessions = region_summary(_units(rows), CFG)
    assert list(regions.index) == ["R"]
    s = sessions.set_index("eid")
    assert s.loc["a", "fraction_rest"] == 0 and s.loc["b", "fraction_rest"] == 1


def _uneven(rng, n_sessions=6):
    """No region effect inside any session, but sessions with many labelled units
    contribute many units of region R, and the others few (Simpson's setting)."""
    rows = []
    for s in range(n_sessions):
        rate = 0.8 if s % 2 else 0.1
        n_r = 30 if s % 2 else 4
        n_other = 6 if s % 2 else 40
        for region, n in (("R", n_r), ("S", n_other)):
            labels = rng.random(n) < rate
            rows += [(f"e{s}", region, bool(x)) for x in labels]
    return _units(rows)


def test_the_region_null_stays_calibrated_when_units_are_pooled_unevenly():
    rng = np.random.default_rng(0)
    ours, naive = [], []
    for _ in range(400):
        units = _uneven(rng)
        regions, _ = region_summary(units, CFG)
        ours.append(regions.loc["R", "p"])
        r = units["region"] == "R"
        table = [
            [(r & units["labelled"]).sum(), (r & ~units["labelled"]).sum()],
            [(~r & units["labelled"]).sum(), (~r & ~units["labelled"]).sum()],
        ]
        naive.append(fisher_exact(table)[1])
    ours, naive = np.array(ours), np.array(naive)
    se = np.sqrt(0.05 * 0.95 / ours.size)
    assert (ours < 0.05).mean() <= 0.05 + 2 * se
    # The setting is a real test: pooling the units across sessions calls R enriched.
    assert (naive < 0.05).mean() > 0.5


def test_a_real_enrichment_is_found_and_regions_are_corrected_together():
    rng = np.random.default_rng(1)
    rows = []
    for s in range(6):
        rows += [(f"e{s}", "R", bool(x)) for x in rng.random(20) < 0.7]
        rows += [(f"e{s}", "S", bool(x)) for x in rng.random(20) < 0.2]
        rows += [(f"e{s}", "T", bool(x)) for x in rng.random(20) < 0.2]
    regions, _ = region_summary(_units(rows), CFG)
    assert regions.loc["R", "claim"] and regions.loc["R", "direction"] == "more"
    assert regions.loc["R", "q"] >= regions.loc["R", "p"]
    assert regions["n_tests"].eq(3).all()


def test_re_running_gives_identical_results():
    units = _uneven(np.random.default_rng(2))
    first, second = region_summary(units, CFG), region_summary(
        units.sample(frac=1, random_state=3), CFG
    )
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


# Real data: the summary's per-unit labels are Studio's, on the same session and filter.
EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
TRIAL_FILTER = {"bwm_include": True, "exclude_nogo": True}


@pytest.fixture(scope="module")
def real():
    from unitwave.data.load import load_session

    try:
        return load_session(EID, "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")


@pytest.mark.parametrize(
    "spec, column, call",
    [
        (("responsive", "stim_on", ""), "responsive", "test_json"),
        (("selective", "stim_on", "choice"), "selective", "selectivity_json"),
        (("locked", "", ""), "locked", "locking_json"),
    ],
)
def test_session_labels_are_studios_on_the_same_session_and_filter(real, spec, column, call):
    import json

    from unitwave.analysis.summary import LabelSpec, session_labels
    from unitwave.data.load import load_data_config
    from unitwave.qc.units import load_qc_config
    from unitwave.studio.project import Source
    from unitwave.studio.server import Studio

    label = LabelSpec(*spec)
    ours = session_labels(real, label, TRIAL_FILTER, level="Beryl")
    assert list(ours.columns) == ["eid", "unit_id", "region", "labelled"]
    studio = Studio(
        real,
        load_qc_config(),
        load_data_config().data_root / "atlas",
        Source(kind="ibl", eid=EID, backend="bwm"),
    )
    q = {"event": spec[1] or "stim_on", "split": spec[2], "tf": json.dumps(TRIAL_FILTER)}
    q.update(all="0", level="Beryl")
    getattr(studio, call)(q)
    store = {"test_json": studio._tests, "selectivity_json": studio._selectivity}
    table = next(iter(store.get(call, studio._locking).values()))
    assert ours["unit_id"].tolist() == list(table.index)
    assert ours["labelled"].tolist() == table[column].astype(bool).tolist()
    assert ours["region"].tolist() == studio._regions(q).loc[table.index].tolist()
    assert (ours["eid"] == EID).all()


def test_label_specs_are_checked():
    from unitwave.analysis.summary import LabelSpec

    with pytest.raises(ValueError, match="event"):
        LabelSpec("responsive", "", "")
    with pytest.raises(ValueError, match="condition"):
        LabelSpec("selective", "stim_on", "")
    with pytest.raises(ValueError, match="kind"):
        LabelSpec("tuned", "stim_on", "")
