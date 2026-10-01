"""The homepage reads region-summary runs (S5); it computes nothing."""

import json

import pytest
from test_summary_plots import _summary

from unitwave.cli.summarise import write_results
from unitwave.studio.summaries import list_summaries, read_summary

RUN = "20261001T120000123456Z_summary_responsive_ca1-examples"


def _write_run(runs, name=RUN, status="complete"):
    out = runs / name
    out.mkdir(parents=True)
    regions, sessions = _summary()
    units = sessions[["eid"]].assign(unit_id="u", region="VISp", labelled=True)
    write_results(out, units, regions, sessions)
    manifest = {
        "run_id": name,
        "status": status,
        "created": "2026-10-01T12:00:00+00:00",
        "label_text": "responsive to stimulus onset",
        "level": "Beryl",
        "summary_config": {"level": "Beryl", "min_sessions": 3, "alpha": 0.05},
        "set": {"name": "CA1 examples", "eids": [f"e{i}" for i in range(6)]},
        "sessions_used": [f"e{i}" for i in range(5)],
        "sessions_failed": {"e5": "no wheel in this session"},
        "n_units": 400,
        "n_regions": 4,
        "n_tested": 3,
        "n_claims": 3,
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def test_finished_summary_runs_are_listed_newest_first_and_nothing_else(tmp_path):
    _write_run(tmp_path)
    _write_run(tmp_path, "20261002T090000000000Z_summary_locked_ca1-examples")
    _write_run(tmp_path, "20261003T090000000000Z_summary_locked_x", status="running")
    (tmp_path / "20261001T090000Z_studio").mkdir()  # an export, not a summary
    rows = list_summaries(tmp_path)
    assert [r["run"] for r in rows] == [
        "20261002T090000000000Z_summary_locked_ca1-examples",
        RUN,
    ]
    assert rows[1] == {
        "run": RUN,
        "label": "responsive to stimulus onset",
        "set": "CA1 examples",
        "level": "Beryl",
        "created": "2026-10-01T12:00:00+00:00",
        "n_sessions": 5,
        "n_failed": 1,
        "n_tested": 3,
        "n_claims": 3,
    }
    assert list_summaries(tmp_path / "missing") == []


def test_a_run_is_read_by_name_only(tmp_path):
    _write_run(tmp_path)
    manifest, regions, sessions = read_summary(tmp_path, RUN)
    assert manifest["n_claims"] == 3 and set(regions.index) == {"VISp", "CA1", "LP", "MOs"}
    assert set(sessions["region"]) == set(regions.index)
    for bad in ("../" + RUN, RUN + "/..", "20261001T120000Z_studio", "nope"):
        with pytest.raises(ValueError, match="no region summary"):
            read_summary(tmp_path, bad)


def test_the_app_serves_the_list_the_table_and_both_figures(tmp_path):
    from unitwave.data.load import load_data_config
    from unitwave.studio.server import App

    _write_run(tmp_path)
    app = App(load_data_config(), runs_dir=tmp_path)
    assert [r["run"] for r in app.summaries({})] == [RUN]
    d = app.summary_json({"run": RUN})
    claims = {r["region"]: r["direction"] for r in d["claims"]}
    assert claims == {"CA1": "fewer", "LP": "fewer", "VISp": "more"}
    assert [r["region"] for r in d["refused"]] == ["MOs"]
    assert (d["n_tested"], d["min_sessions"]) == (3, 3)
    caption = d["caption"]
    for words in (
        "responsive to stimulus onset",
        "5 of 6 sessions",
        "1 left out",
        "permuted within each session",
        "Benjamini–Hochberg across 3 regions",
        "1 region refused",
    ):
        assert words in caption, words
    for kind in ("flatmap", "spread"):
        png, _ = app.summary_png({"run": RUN, "kind": kind, "theme": "dark"})
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="flatmap or spread"):
        app.summary_png({"run": RUN, "kind": "other"})
