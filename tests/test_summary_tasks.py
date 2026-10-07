"""Step 8c: region summaries for any task with regions. A label is checked against its
task (events, comparisons, movement), each session is labelled with its own unit QC
rule and the same test Studio runs, angles take the circular test, and a session or a
layout without brain regions is refused with the reason. Spike trains here are test
inputs, except the tests that read downloaded files."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from test_circular_selectivity import TASK as GRATINGS
from test_qc_spike_times import _session as hand_built

from unitwave.analysis.events import EVENTS, event_times
from unitwave.analysis.responsiveness import load_response_config, responsiveness
from unitwave.analysis.summary import LabelSpec, session_labels
from unitwave.analysis.tuning import circular_selectivity, load_selectivity_config
from unitwave.qc.spike_times import load_spike_qc_config


def _task_file(tmp_path) -> str:
    path = tmp_path / "gratings_test.yaml"
    path.write_text(yaml.safe_dump(GRATINGS))
    return str(path)


def test_ibl_labels_read_as_before():
    label = LabelSpec("responsive", "stim_on")
    assert label.task == "ibl"
    assert label.describe() == f"responsive to {EVENTS['stim_on'][0].lower()}"
    assert LabelSpec("locked").describe() == "movement-locked"


def test_a_label_is_checked_against_its_task(tmp_path):
    steinmetz = LabelSpec("selective", "stim_on", "choice", task="steinmetz")
    assert steinmetz.describe() == "selective for choice at stimulus onset"
    with pytest.raises(ValueError, match="comparison.*'choice', 'outcome'"):
        LabelSpec("selective", "stim_on", "contrasts", task="steinmetz")
    with pytest.raises(ValueError, match="declares no movement events"):
        LabelSpec("locked", task="steinmetz")
    with pytest.raises(ValueError, match="needs an event, one of .*'stim_on'"):
        LabelSpec("responsive", "go_cue", task="allen_flashes")
    angles = LabelSpec("selective", "stim_on", "direction", task=_task_file(tmp_path))
    assert angles.describe() == "selective for direction at stimulus onset"
    with pytest.raises(ValueError, match="no task definition"):
        LabelSpec("responsive", "stim_on", task="no_such_task")


def _gratings_session():
    """The circular test's 96 presentations (2 blanks), with a tuned unit in VISp, a
    flat one in CA1, and a sparse one in VISp that the spike-time rule fails. Spikes are
    evenly spaced in each presentation's first second, so no train has refractory
    violations."""
    from test_circular_selectivity import _session

    trials, _ = _session(np.random.default_rng(1))
    # Jittered starts, as real trials are: strictly periodic presentations are refused
    # by the responsiveness test (tests/test_responsiveness_periodic.py).
    jitter = np.random.default_rng(3).uniform(0.0, 1.0, len(trials))
    trials = trials.assign(intervals_0=trials["intervals_0"] + np.cumsum(jitter))
    trials = trials.assign(intervals_1=trials["intervals_0"] + 2.0)
    directions = trials["orientation"].fillna(0.0).to_numpy()
    starts = trials["intervals_0"].to_numpy()

    def unit(count_of):
        out = []
        for t, d in zip(starts, directions):
            n = int(count_of(d))
            out.append(t + (np.arange(n) + 0.5) / n)
        return np.concatenate(out)

    spikes = {
        "tuned": unit(lambda d: round(2 + 18 * np.exp(3 * np.cos(np.radians(d - 90)) - 3))),
        "flat": unit(lambda d: 8),
        "sparse": np.array([5.0, 120.0, 230.0]),
    }
    units = pd.DataFrame(
        {"probe_name": "p0", "acronym": ["VISp", "CA1", "VISp"]}, index=list(spikes)
    )
    session = hand_built(spikes, float(trials["intervals_1"].max()) + 1.0, units)
    return session.__class__(**{**session.__dict__, "trials": trials})


def test_session_labels_use_the_sessions_qc_and_the_circular_test(tmp_path):
    session = _gratings_session()
    task = _task_file(tmp_path)
    spike_qc = load_spike_qc_config()
    label = LabelSpec("selective", "stim_on", "direction", task=task)
    ours = session_labels(session, label, {}, level="Beryl", qc=spike_qc)
    assert ours["unit_id"].tolist() == ["tuned", "flat"]  # sparse fails the spike-time rule
    assert ours["region"].tolist() == ["VISp", "CA1"]
    from unitwave.analysis.tasks import load_task

    direct = circular_selectivity(
        session.spikes,
        ["tuned", "flat"],
        session.trials,
        "stim_on",
        "direction",
        load_response_config(),
        load_selectivity_config(),
        trial_mask=np.ones(len(session.trials), bool),
        task=load_task(task),
    )
    assert ours["labelled"].tolist() == direct["selective"].tolist() == [True, False]

    responsive = session_labels(
        session, LabelSpec("responsive", "stim_on", task=task), {}, level="Beryl", qc=spike_qc
    )
    events = event_times(session.trials, "stim_on", load_task(task))
    expected = responsiveness(session.spikes, ["tuned", "flat"], events, load_response_config())
    assert responsive["labelled"].tolist() == expected["responsive"].tolist()


def test_a_session_without_regions_is_refused_with_the_reason(tmp_path):
    session = _gratings_session()
    missing = {**session.available.missing, "units.acronym": "no locations in this file"}
    present = session.available.present - {"units.acronym"}
    no_regions = session.__class__(
        **{
            **session.__dict__,
            "units": session.units.drop(columns="acronym"),
            "available": session.available.__class__(present=present, missing=missing),
        }
    )
    label = LabelSpec("responsive", "stim_on", task=_task_file(tmp_path))
    with pytest.raises(ValueError, match="no brain regions: no locations in this file"):
        session_labels(no_regions, label, {}, level="Beryl", qc=load_spike_qc_config())


# Downloaded files: Steinmetz (regions) and MC_Maze (none).
RICHARDS = (
    Path.home() / "data/neurodecoder/dandi/000017/sub-Richards/sub-Richards_ses-20171031T120000.nwb"
)
MC_MAZE = (
    Path.home()
    / "data/neurodecoder/dandi/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
)
ENGAGED = {"included": True}


@pytest.fixture(scope="module")
def richards():
    if not RICHARDS.exists():
        pytest.skip("the Steinmetz Richards file isn't downloaded")
    from unitwave.studio.project import Source, load_source

    source = Source(kind="nwb", file=str(RICHARDS), layout="steinmetz_2019", task="steinmetz")
    return source, *load_source(source)


@pytest.mark.parametrize(
    "spec, column, call",
    [
        (("responsive", "stim_on", ""), "responsive", "test_json"),
        (("selective", "stim_on", "choice"), "selective", "selectivity_json"),
    ],
)
def test_steinmetz_labels_are_studios_on_the_same_session_and_filter(richards, spec, column, call):
    from unitwave.data.load import load_data_config
    from unitwave.studio.server import Studio

    source, session, qc = richards
    ours = session_labels(
        session, LabelSpec(*spec, task="steinmetz"), ENGAGED, level="Beryl", qc=qc
    )
    studio = Studio(session, qc, load_data_config().data_root / "atlas", source)
    q = {"event": spec[1], "split": spec[2], "tf": json.dumps(ENGAGED), "all": "0"}
    getattr(studio, call)({**q, "level": "Beryl"})
    store = {"test_json": studio._tests, "selectivity_json": studio._selectivity}[call]
    table = next(iter(store.values()))
    assert ours["unit_id"].tolist() == list(table.index)
    assert ours["labelled"].tolist() == table[column].astype(bool).tolist()
    assert ours["region"].tolist() == studio._regions({"level": "Beryl"}).loc[table.index].tolist()


def test_a_dataset_without_regions_is_refused_before_anything_is_read(tmp_path):
    from unitwave.cli.summarise import run_nwb

    label = LabelSpec("responsive", "go_cue", task="nlb_mc_maze")
    never_read = tmp_path / "absent.nwb"
    with pytest.raises(ValueError, match="no brain regions.*per-array"):
        run_nwb([never_read], label, layout="nlb_mc_maze", name="maze", runs_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []  # no run folder was started


def test_an_nwb_run_records_its_files_layout_task_and_qc(richards, tmp_path):
    import dataclasses

    from unitwave.analysis.summary import load_summary_config
    from unitwave.cli.summarise import run_nwb
    from unitwave.studio.project import file_hashes
    from unitwave.studio.summaries import list_summaries

    source, _, _ = richards
    label = LabelSpec("responsive", "stim_on", task="steinmetz")
    cfg = dataclasses.replace(load_summary_config(), min_sessions=1)
    with pytest.raises(ValueError, match="named twice"):
        run_nwb([RICHARDS, RICHARDS], label, layout="steinmetz_2019", name="x", runs_dir=tmp_path)
    out = run_nwb(
        [RICHARDS],
        label,
        layout="steinmetz_2019",
        name="Richards one",
        trial_filter=ENGAGED,
        cfg=cfg,
        runs_dir=tmp_path,
        progress=lambda _: None,
    )
    m = json.loads((out / "manifest.json").read_text())
    s = m["set"]
    assert s["kind"] == "nwb" and (s["layout"], s["task"]) == ("steinmetz_2019", "steinmetz")
    assert s["layout_label"] == "Steinmetz et al. 2019 (DANDI 000017)"
    assert s["files"] == [{"path": str(RICHARDS), "sha256": file_hashes(source)[RICHARDS.name]}]
    assert s["trial_filter"] == {"included": True, "exclude_nogo": False, "outcomes": []}
    assert m["sessions_used"] == [f"nwb:{RICHARDS}"] == s["eids"] and m["task"] == "steinmetz"
    assert m["configs"]["qc"]["path"].endswith("configs/qc_nwb.yaml")
    assert m["configs"]["layout"]["path"].endswith("configs/nwb/steinmetz_2019.yaml")
    assert m["configs"]["task"]["path"].endswith("configs/tasks/steinmetz.yaml")
    assert m["label_text"] == "responsive to stimulus onset"
    units = pd.read_parquet(out / "units.parquet")
    assert set(units["eid"]) == {f"nwb:{RICHARDS}"}
    assert [r["set"] for r in list_summaries(tmp_path)] == ["Richards one"]


SIX = [
    Path.home() / "data/neurodecoder/dandi/000017" / f
    for f in (
        "sub-Cori/sub-Cori_ses-20161214T120000.nwb",
        "sub-Forssmann/sub-Forssmann_ses-20171101T120000.nwb",
        "sub-Lederberg/sub-Lederberg_ses-20171205T120000.nwb",
        "sub-Lederberg/sub-Lederberg_ses-20171207T120000.nwb",
        "sub-Richards/sub-Richards_ses-20171029T120000.nwb",
        "sub-Theiler/sub-Theiler_ses-20171011T120000.nwb",
    )
]


@pytest.mark.skipif(not all(f.exists() for f in SIX), reason="the six Steinmetz files aren't here")
def test_six_steinmetz_sessions_give_regions_a_verdict(tmp_path):
    """Chosen by streaming every file's regions (docs/DECISIONS.md, "Step 8c"): CA1, DG,
    MOs, SUB and VISp hold good units in at least 5 of the 6 sessions."""
    from unitwave.cli.summarise import run_nwb

    label = LabelSpec("responsive", "stim_on", task="steinmetz")
    out = run_nwb(
        SIX,
        label,
        layout="steinmetz_2019",
        name="Steinmetz six",
        trial_filter=ENGAGED,
        runs_dir=tmp_path,
        progress=lambda _: None,
    )
    m = json.loads((out / "manifest.json").read_text())
    assert m["sessions_failed"] == {} and len(m["sessions_used"]) == 6
    regions = pd.read_parquet(out / "regions.parquet")
    tested = set(regions.index[regions["refused"] == ""])
    assert {"CA1", "DG", "MOs", "SUB", "VISp"} <= tested
    assert (regions.loc[list(tested), "n_sessions"] >= 5).all()
    assert (regions["n_tests"] == len(tested)).all()
    sessions = pd.read_parquet(out / "sessions.parquet")
    assert set(sessions["region"]) == set(regions.index)  # per-session rows for every region
