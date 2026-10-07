"""Step 3: a task that is not IBL's, defined in its own YAML file, works end to end:
Phy folder + events CSV -> Studio's numbers, captions, trial view, export and project
file, all in the task's own names. Hand-built test data, never shown as real."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from phy_folder import write_phy_folder

from unitwave.analysis.movement import MovementConfig
from unitwave.qc.phy import PhyUnitQC
from unitwave.studio.export import export_view
from unitwave.studio.project import (
    DEFAULT_VIEW,
    Source,
    load_source,
    make_project,
    open_project,
    save_project,
)
from unitwave.studio.server import Studio

RATE = 30000.0
# A tone/lick task: a low or high tone at one of two loudnesses, a lick, water on some
# trials. Nothing in it is IBL's.
TASK = {
    "name": "tone_lick",
    "label": "Tone–lick task",
    "required_columns": ["intervals_0", "intervals_1", "tone_times"],
    "events": {
        "tone": {"label": "Tone onset", "column": "tone_times"},
        "lick": {"label": "First lick", "column": "lick_times"},
        "water": {
            "label": "Water",
            "column": "outcome_times",
            "when": {"column": "rewarded", "equals": 1},
        },
    },
    "trial_view_events": {
        "tone_times": "tone",
        "lick_times": "first lick",
        "outcome_times": "outcome",
    },
    "conditions": {
        "pitch": {
            "label": "Pitch",
            "type": "categorical",
            "column": "pitch",
            "level_names": {1: "low", 2: "high"},
            "excluded": "trials without a tone",
        },
        "loudness": {
            "label": "Loudness",
            "type": "ordinal",
            "column": "loudness_db",
            "level_format": "{v:g} dB",
            "excluded": "trials without a loudness",
        },
        "rewarded": {
            "label": "Reward",
            "type": "categorical",
            "column": "rewarded",
            "level_names": {0: "none", 1: "water"},
            "excluded": "trials without an outcome",
        },
        "latency": {
            "label": "Lick latency",
            "type": "continuous",
            "derive": {"name": "median_split_difference", "columns": ["lick_times", "tone_times"]},
            "excluded": "trials without a lick",
        },
    },
    "strata": {"loudness": {"condition": "loudness"}},
    "comparisons": {
        "pitch": {"levels": [1, 2], "null_model": {"permute_within": "loudness"}},
        "rewarded": {"levels": [0, 1], "null_model": {"permute": "all"}},
    },
    "movement": {"stimulus": "tone", "movement": "lick", "strata": "loudness"},
    "behaviour": {"traces": []},
    "trialstruct": ["pitch", "loudness_db"],
    "trial_filters": {
        "engaged": {
            "label": "Engaged trials only",
            "kind": "flag",
            "column": "engaged",
            "reason": "not engaged",
        },
        "loud": {
            "label": "Loudness",
            "kind": "select",
            "column": "loudness_db",
            "level_format": "{v:g} dB",
            "reason": "loudness not selected",
        },
    },
}
N_TRIALS = 80


def _trials(rng) -> pd.DataFrame:
    start = 1.0 + 2.0 * np.arange(N_TRIALS)
    tone = start + 0.2
    lick = tone + rng.uniform(0.15, 0.45, N_TRIALS)
    lick[[5, 17]] = np.nan  # no lick: latency excluded, not filled
    rewarded = rng.integers(0, 2, N_TRIALS).astype(float)
    return pd.DataFrame(
        {
            "intervals_0": start,
            "intervals_1": start + 1.5,
            "tone_times": tone,
            "lick_times": lick,
            "outcome_times": np.where(np.isfinite(lick), lick + 0.05, tone + 1.0),
            "pitch": rng.integers(1, 3, N_TRIALS).astype(float),
            "loudness_db": rng.choice([50.0, 70.0], N_TRIALS),
            "rewarded": rewarded,
            "engaged": np.where(np.arange(N_TRIALS) % 10 == 9, 0.0, 1.0),
        }
    )


def _spikes(rng, trials) -> dict[int, np.ndarray]:
    """Cluster 1 answers high tones, cluster 2 fires at the lick, cluster 3 at neither."""
    end = trials["intervals_1"].iloc[-1] + 1.0

    def background():
        return rng.uniform(0.0, end, rng.poisson(5.0 * end))

    high = trials["tone_times"][trials["pitch"] == 2].to_numpy()
    licks = trials["lick_times"].dropna().to_numpy()
    return {
        1: np.concatenate([background(), *(t + rng.uniform(0.0, 0.2, 8) for t in high)]),
        2: np.concatenate([background(), *(t + rng.uniform(0.0, 0.1, 6) for t in licks)]),
        3: background(),
    }


@pytest.fixture
def custom(tmp_path):
    rng = np.random.default_rng(3)
    trials = _trials(rng)
    spikes = _spikes(rng, trials)
    times = np.concatenate(list(spikes.values()))
    clusters = np.concatenate([np.full(t.size, c) for c, t in spikes.items()])
    order = np.argsort(times)
    folder = write_phy_folder(
        tmp_path / "imec0",
        np.round(times[order] * RATE).astype(np.int64),
        clusters[order],
        RATE,
        ks_label={1: "good", 2: "good", 3: "good"},
    )
    trials.to_csv(tmp_path / "events.csv", index=False)
    task = tmp_path / "tone_lick.yaml"
    task.write_text(yaml.safe_dump(TASK, sort_keys=False, allow_unicode=True))
    events = str(tmp_path / "events.csv")
    return Source(kind="phy", folder=str(folder), events=events, task=str(task)), tmp_path


def _studio(source: Source, tmp_path: Path) -> Studio:
    session, _ = load_source(source)
    qc = PhyUnitQC(("good",), 0.1, refractory_contamination=0.5, refractory_alpha=0.1)
    studio = Studio(session, qc, tmp_path / "atlas", source)
    studio.movement_cfg = MovementConfig((-0.1, 0.0), (0.0, 0.1), 200, 0)
    return studio


Q = {"event": "tone", "t0": "-0.2", "t1": "0.5", "bin": "0.05", "all": "1", "theme": "light"}


def test_the_session_speaks_the_tasks_own_names(custom):
    studio = _studio(*custom)
    s = studio.session_json({})
    assert s["task"]["name"] == "tone_lick" and s["task"]["label"] == "Tone–lick task"
    assert s["events"] == {"tone": "Tone onset", "lick": "First lick", "water": "Water"}
    assert s["conditions"] == {
        "pitch": "Pitch",
        "loudness": "Loudness",
        "rewarded": "Reward",
        "latency": "Lick latency",
    }
    assert s["comparisons"] == {"pitch": ["low", "high"], "rewarded": ["none", "water"]}
    assert s["trial_levels"] == {"loud": [50.0, 70.0]}
    loud = s["task"]["trial_filters"]["loud"]["levels"]
    assert [lv["name"] for lv in loud] == ["50 dB", "70 dB"]
    m = s["movement"]
    assert m["first_movement"] is True and m["stimulus"] == "tone"
    assert m["null"] == "reaction times permuted within loudness"
    assert set(s["trial_view"]["alignments"]) == {
        "trial_start",
        "tone_times",
        "lick_times",
        "outcome_times",
    }


def test_split_psths_tuning_and_captions_use_its_conditions(custom):
    studio = _studio(*custom)
    d = studio.unit_data({**Q, "unit": "imec0_1", "split": "pitch"})
    assert [g.name for g in d["groups"]] == ["low", "high"]
    assert " · Tone onset · split by pitch · n = 80 trials" in d["caption"]
    latency = studio.unit_data({**Q, "unit": "imec0_1", "split": "latency"})
    assert "2 trials without a lick excluded" in latency["caption"]
    t = studio.tuning_data({**Q, "unit": "imec0_1", "split": "loudness"})
    assert list(t["curve"].index) == ["50 dB", "70 dB"]
    assert "after tone onset, by loudness" in t["caption"]


def test_selectivity_uses_the_declared_nulls(custom):
    studio = _studio(*custom)
    pitch = studio.selectivity_json({**Q, "split": "pitch"})
    assert pitch["condition"] == "Pitch" and (pitch["a"], pitch["b"]) == ("low", "high")
    assert pitch["null"] == "pitch permuted within loudness"
    table = studio._selectivity[studio._selectivity_key({**Q, "split": "pitch"})]
    assert table.loc["imec0_1", "selective"] and not table.loc["imec0_3", "selective"]
    assert table.loc["imec0_1", "auroc"] > 0.5  # higher for the high tone
    plain = studio.selectivity_json({**Q, "split": "rewarded"})
    assert plain["null"] == "rewarded permuted across trials (no strata declared)"
    with pytest.raises(ValueError, match="loudness has many levels"):
        studio.selectivity_json({**Q, "split": "loudness"})


def test_movement_controls_use_its_stimulus_and_movement(custom):
    studio = _studio(*custom)
    locking = studio.locking_json({"all": "1"})
    assert locking["null"] == "reaction times permuted within loudness"
    assert locking["n_trials"] == N_TRIALS - 2  # two trials without a lick
    table = studio._locking[studio._locking_key({"all": "1"})]
    assert table.loc["imec0_2", "locked"] and not table.loc["imec0_3", "locked"]
    with pytest.raises(ValueError, match="tone onset only"):
        studio.test_json({**Q, "event": "lick", "movement_free": "1"})


def test_its_trial_filters_exclude_and_count(custom):
    studio = _studio(*custom)
    tf = json.dumps({"engaged": True, "loud": [70.0]})
    summary = studio._trial_summary({**Q, "tf": tf})
    trials = studio.session.trials
    not_engaged = int((trials["engaged"] == 0).sum())
    quiet = int((trials["loudness_db"] == 50).sum())
    assert summary["excluded"] == {"not engaged": not_engaged, "loudness not selected": quiet}
    with pytest.raises(ValueError, match="unknown trial filters"):
        studio._filter({"tf": json.dumps({"bwm_include": True})})


def test_the_single_trial_view_names_its_events_and_conditions(custom):
    studio = _studio(*custom)
    trials = studio.session.trials
    k = int(np.flatnonzero((trials["rewarded"] == 1) & trials["lick_times"].notna())[0])
    q = {**Q, "unit": "imec0_1", "trial": str(k), "trial_align": "tone_times"}
    header = studio.trial_json(q)["header"]
    assert header["task"] == "Tone–lick task"
    levels = {c["name"]: c["level"] for c in header["conditions"]}
    assert levels["rewarded"] == "water" and levels["pitch"] in ("low", "high")
    assert header["flags"] == {"engaged": bool(trials["engaged"].iloc[k])}
    assert "side" not in header  # IBL's own fields belong to IBL's definition
    view = studio.trial_data(q)["view"]
    assert {e["label"] for e in view.events} == {"tone", "first lick", "outcome: water"}
    assert view.align_label == "tone"
    assert "zero at tone" in studio.trial_data(q)["caption"]


def test_export_and_project_file_record_the_task(custom):
    source, tmp_path = custom
    studio = _studio(source, tmp_path)
    view = {**DEFAULT_VIEW, "event": "tone", "split": "pitch", "unit": "imec0_1", "all": True}
    out = export_view(studio, view, tmp_path / "runs")
    unit = json.loads((out / "unit.json").read_text())
    assert "split by pitch" in unit["caption"]
    assert [g["condition"] for g in unit["groups"]] == ["low", "high"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["project"]["source"]["task"] == source.task
    assert manifest["project"]["source"]["task_label"] == "Tone–lick task"

    session, qc = load_source(source)
    path = tmp_path / "tone.unitwave.json"
    save_project(make_project(source, session, qc, view), path)
    project, _, _, warnings = open_project(path)
    assert warnings == [] and project["source"]["task"] == source.task
    changed = {**TASK, "label": "Tone–lick task, edited"}
    Path(source.task).write_text(yaml.safe_dump(changed, sort_keys=False, allow_unicode=True))
    _, _, _, warnings = open_project(path)
    assert any("task definition" in w and "changed" in w for w in warnings)


def test_a_table_without_the_required_columns_is_refused_naming_them(custom):
    source, tmp_path = custom
    events = pd.read_csv(source.events).drop(columns="tone_times")
    events.to_csv(source.events, index=False)
    session, qc = load_source(source)
    with pytest.raises(ValueError, match="no tone_times column.*'Tone–lick task'"):
        Studio(session, qc, tmp_path / "atlas", source)


def test_columns_no_definition_reads_are_refused(custom):
    source, _ = custom
    ibl = Source(kind="phy", folder=source.folder, events=source.events)  # IBL's names
    with pytest.raises(ValueError, match=r"unknown event columns \['engaged'"):
        load_source(ibl)
