"""Step 4: unit QC from spike times only, for sources without their own labels.
Spike trains here are hand-built test inputs, never shown as data."""

import numpy as np
import pandas as pd
import pytest
from phy_folder import write_phy_folder

from unitwave.analysis.units import unit_table
from unitwave.data.session import Session
from unitwave.qc.phy import PhyUnitQC
from unitwave.qc.spike_times import (
    UNAVAILABLE,
    SpikeQC,
    agreement,
    load_spike_qc_config,
    presence_ratio,
    spike_time_metrics,
    spike_unit_qc,
)
from unitwave.qc.units import UnitQC
from unitwave.studio.project import Source, load_source

CFG = SpikeQC(
    min_firing_rate_hz=0.1,
    refractory_contamination=0.1,
    refractory_alpha=0.1,
    presence_window_s=10.0,
    min_presence_ratio=0.9,
)


def test_the_signed_off_defaults():
    assert load_spike_qc_config() == CFG
    assert set(UNAVAILABLE) >= {"amplitude_cutoff", "amplitude_median"}
    assert all("amplitude" in why for why in UNAVAILABLE.values())


def test_presence_ratio_by_hand():
    # Task period 0-35 s: three whole 10 s bins, [0, 10), [10, 20), [20, 30); the last
    # 5 s are less than a bin and are not binned.
    spikes = np.array([1.0, 2.0, 20.0, 29.9, 33.0])
    assert presence_ratio(spikes, 0.0, 35.0, 10.0) == pytest.approx(2 / 3)
    assert presence_ratio(np.array([10.0]), 0.0, 35.0, 10.0) == pytest.approx(1 / 3)  # [10, 20)
    assert presence_ratio(np.array([-1.0, 40.0]), 0.0, 35.0, 10.0) == 0.0  # outside the task
    assert np.isnan(presence_ratio(spikes, 0.0, 9.0, 10.0))  # not one whole bin


def _session(spikes: dict, t1: float, units: pd.DataFrame | None = None) -> Session:
    from unitwave.data.session import (
        BEHAVIOUR_FIELDS,
        TRIAL_FIELDS,
        UNIT_FIELDS,
        Capabilities,
    )

    ids = list(spikes)
    units = units if units is not None else pd.DataFrame({"probe_name": "p0"}, index=ids)
    units.index.name = "unit_id"
    trials = pd.DataFrame({"intervals_0": [0.0, t1 / 2], "intervals_1": [t1 / 2, t1]})
    present = {f"units.{f}" for f in UNIT_FIELDS if f in units}
    present |= {f"trials.{f}" for f in TRIAL_FIELDS if f in trials}
    fields = [f"units.{f}" for f in UNIT_FIELDS] + [f"trials.{f}" for f in TRIAL_FIELDS]
    fields += [f"behaviour.{f}" for f in BEHAVIOUR_FIELDS]
    missing = {f: "not in this test session" for f in fields if f not in present}
    return Session(
        eid="hand-built",
        time_bounds=(0.0, t1 + 1.0),
        spikes={u: np.asarray(t, float) for u, t in spikes.items()},
        units=units,
        trials=trials,
        behaviour={},
        available=Capabilities(present=present, missing=missing),
    )


def _trains():
    """Over a 100 s task: a clean 10 Hz train, the same with every spike doubled 0.5 ms
    later (refractory violations), a train silent for the last 30 s, and a sparse one."""
    clean = np.arange(0.05, 100.0, 0.1)
    return {
        "clean": clean,
        "doubled": np.sort(np.concatenate([clean, clean + 0.0005])),
        "gap": clean[clean < 70.0],
        "sparse": np.array([5.0, 55.0, 95.0]),
    }


def test_each_metric_and_reason_by_hand():
    s = _session(_trains(), 100.0)
    m = spike_time_metrics(s, CFG)
    assert m.loc["clean", "task_firing_rate"] == pytest.approx(1000 / 100.0)
    assert m.loc["sparse", "task_firing_rate"] == pytest.approx(3 / 100.0)
    assert m.loc["clean", "presence_ratio"] == 1.0
    assert m.loc["gap", "presence_ratio"] == pytest.approx(0.7)
    assert bool(m.loc["clean", "sliding_rp_pass"]) and not bool(m.loc["doubled", "sliding_rp_pass"])
    v = spike_unit_qc(m, CFG)
    assert v.loc["clean", "passed"] and v.loc["clean", "reason"] == ""
    assert "refractory violations" in v.loc["doubled", "reason"]
    assert v.loc["gap", "reason"] == "presence ratio 0.7 < 0.9 (10 s bins over the task)"
    sparse = v.loc["sparse", "reason"]
    assert "task firing rate 0.03 Hz < 0.1 Hz" in sparse and "presence ratio 0.3" in sparse


def test_a_source_with_no_labels_is_judged_on_spike_times():
    s = _session(_trains(), 100.0)
    table = unit_table(s, CFG)
    assert table["label"].isna().all()
    assert table["qc_passed"].tolist() == [True, False, False, False]
    assert table["qc_passed"].equals(table["spike_qc_passed"])
    assert table.loc["gap", "presence_ratio"] == pytest.approx(0.7)


def test_labelled_sources_keep_their_rule_and_show_both_verdicts():
    labels = pd.DataFrame(
        {"probe_name": "p0", "label": [1.0, 1.0, 0.33, 1.0], "acronym": ["CA1"] * 4},
        index=["clean", "doubled", "gap", "sparse"],
    )
    s = _session(_trains(), 100.0, labels)
    table = unit_table(s, UnitQC(1.0, ("void", "root"), 0.1))
    # The IBL rule: label 1, located, rate >= 0.1 Hz. "doubled" passes it.
    assert table["qc_passed"].tolist() == [True, True, False, False]
    assert table["spike_qc_passed"].tolist() == [True, False, False, False]


def _phy(tmp_path, **labels):
    rate = 30000.0
    trains = _trains()
    times = np.concatenate([trains["clean"], trains["gap"]])
    clusters = np.concatenate([np.full(trains["clean"].size, 1), np.full(trains["gap"].size, 2)])
    order = np.argsort(times, kind="stable")
    folder = write_phy_folder(
        tmp_path / "imec0",
        np.round(times[order] * rate).astype(np.int64),
        clusters[order],
        rate,
        **labels,
    )
    # Within the recorded spikes (the loader refuses events outside them): 9 whole bins.
    pd.DataFrame({"intervals_0": [0.1, 50.0], "intervals_1": [50.0, 99.9]}).to_csv(
        tmp_path / "events.csv", index=False
    )
    return Source(kind="phy", folder=str(folder), events=str(tmp_path / "events.csv"))


def test_a_phy_folder_without_label_files_uses_the_spike_time_rule(tmp_path):
    session, qc = load_source(_phy(tmp_path))
    assert isinstance(qc, SpikeQC)
    table = unit_table(session, qc)
    assert table["qc_passed"].tolist() == [True, False]  # the gap unit is absent 30% of the task
    assert "presence ratio" in table.loc["imec0_2", "qc_reason"]


def test_a_phy_folder_with_label_files_keeps_the_phy_rule(tmp_path):
    session, qc = load_source(_phy(tmp_path, ks_label={1: "good", 2: "good"}))
    assert isinstance(qc, PhyUnitQC)
    table = unit_table(session, qc)
    assert table["qc_passed"].tolist() == [True, True]  # presence is not part of the Phy rule
    assert table["spike_qc_passed"].tolist() == [True, False]


def test_agreement_counts_and_says_why_they_differ():
    table = pd.DataFrame(
        {
            "qc_passed": [True, True, False, False, True],
            "qc_reason": ["", "", "label 0.667 < 1", "located in root", ""],
            "spike_qc_passed": [True, False, True, False, False],
            "spike_qc_reason": [
                "",
                "presence ratio 0.5 < 0.9 (10 s bins over the task)",
                "",
                "task firing rate 0.05 Hz < 0.1 Hz",
                (
                    "refractory violations: contamination below 0.1 not shown at 0.9 "
                    "confidence; presence ratio 0.8 < 0.9 (10 s bins over the task)"
                ),
            ],
        }
    )
    a = agreement(table)
    assert (a["n_units"], a["both"], a["source_only"], a["spikes_only"], a["neither"]) == (
        5,
        1,
        2,
        1,
        1,
    )
    assert a["agree"] == pytest.approx(2 / 5)
    # Why units passing the source's rule fail on spike times, criterion by criterion.
    assert a["source_only_fail"] == {"presence ratio": 2, "refractory violations": 1}
    assert a["spikes_only_fail"] == {"label": 1}


def test_the_agreement_run_writes_per_session_counts_and_a_manifest(tmp_path):
    import json

    from unitwave.cli.qc_agreement import run

    labels = pd.DataFrame(
        {"probe_name": "p0", "label": [1.0, 1.0, 0.33, 1.0], "acronym": ["CA1"] * 4},
        index=["clean", "doubled", "gap", "sparse"],
    )
    sessions = {"a" * 8: _session(_trains(), 100.0, labels), "b" * 8: None}

    def load(eid):
        if sessions[eid] is None:
            raise ValueError("not in the local cache")
        return sessions[eid]

    out = run(list(sessions), runs_dir=tmp_path, load=load, progress=lambda _: None)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["sessions_used"] == ["a" * 8]
    assert manifest["sessions_failed"] == {"b" * 8: "not in the local cache"}
    assert set(manifest["configs"]) == {"qc", "spike_qc"}
    per_session = json.loads((out / "agreement.json").read_text())
    row = per_session["sessions"]["a" * 8]
    assert (row["both"], row["source_only"], row["spikes_only"], row["neither"]) == (1, 1, 0, 2)
    assert row["source_only_fail"] == {"refractory violations": 1}
    assert per_session["total"]["n_units"] == 4
    units = pd.read_csv(out / "units.csv")
    assert list(units.columns[:3]) == ["eid", "unit_id", "qc_passed"]


def test_studio_names_the_rule_and_shows_both_verdicts(tmp_path):
    from unitwave.studio.server import Studio

    for labels, rule in (({}, "spike times"), ({"ks_label": {1: "good", 2: "good"}}, "Phy group")):
        root = tmp_path / rule.replace(" ", "_")
        session, qc = load_source(_phy(root, **labels))
        studio = Studio(session, qc, root / "atlas")
        s = studio.session_json({})
        assert s["qc"]["rule"] == rule
        assert s["qc"]["spike_config"] == "configs/qc_spikes.yaml"
        assert s["n_units_passing_spikes"] == 1
        row = {u["id"]: u for u in studio.units_json({"all": "1"})["units"]}["imec0_2"]
        assert row["spike_qc_passed"] is False and row["presence_ratio"] == pytest.approx(7 / 9)
        q = studio.quality_json({"unit": "imec0_2"})
        assert q["spike_qc"]["is_the_qc"] is (rule == "spike times")
        assert any("presence ratio" in r for r in q["spike_qc"]["reasons"])
        assert "amplitude_cutoff" in q["spike_qc"]["spike_unavailable"]
