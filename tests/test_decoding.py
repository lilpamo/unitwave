"""Decoding in Studio (S3): one session, the shown units, the CLI's table."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from unitwave.analysis.decoding import DecodingConfig, decode, load_decoding_config
from unitwave.cli.evaluate import RunConfig, run
from unitwave.data.load import load_data_config, load_session
from unitwave.qc.units import apply_unit_qc, load_qc_config

EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
DATA = load_data_config()
needs_bwm = pytest.mark.skipif(
    not (DATA.bwm_ephys_root.exists() and DATA.bwm_behavior_root.exists()),
    reason="BWM releases not available",
)


def test_default_config():
    cfg = load_decoding_config()
    assert cfg == DecodingConfig(
        seed=0,
        train_fraction=0.8,
        gap_s=2.0,
        leave_one_block_out=("block",),
        train_stride={"movement_state": 5},
        targets=("choice", "stimulus_side", "block", "movement_state"),
        model="ridge",
        n_shifts=100,
        n_bootstrap=2000,
        alpha=0.05,
    )


def test_unknown_targets_are_refused():
    with pytest.raises(ValueError, match="wheel_velocity"):
        decode(EID, "wheel_velocity", cfg=load_decoding_config())


@pytest.fixture(scope="module")
def quick():
    """The default config with fewer null draws, for speed; both paths use it."""
    return dataclasses.replace(load_decoding_config(), n_shifts=5, n_bootstrap=200)


@needs_bwm
def test_studio_table_equals_the_cli_table_for_the_same_session_units_split_and_seed(
    tmp_path, quick
):
    run_cfg = RunConfig(
        name="studio_check",
        seed=quick.seed,
        fixed_sessions=(EID,),
        random_sessions=0,
        train_fraction=quick.train_fraction,
        gap_s=quick.gap_s,
        targets=("choice",),
        train_stride=dict(quick.train_stride),
        model=quick.model,
        n_shifts=quick.n_shifts,
        leave_one_block_out=quick.leave_one_block_out,
    )
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"name": run_cfg.name}))
    cli = run(run_cfg, config_path=path, runs_dir=tmp_path / "cli")
    cli_table = pd.read_parquet(cli / "choice" / "per_session.parquet")

    studio = decode(EID, "choice", cfg=quick, runs_dir=tmp_path / "studio")
    studio_table = pd.read_parquet(studio.folder / "choice" / "per_session.parquet")
    rows = studio_table.index.get_level_values("row").unique()
    assert "baseline_rrr" not in rows and "baseline_rrr" in cli_table.index.get_level_values("row")
    pd.testing.assert_frame_equal(studio_table, cli_table.loc[list(rows)])

    # The run folder carries the section 7 manifest and the single-session verdicts.
    manifest = json.loads((studio.folder / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["sessions"] == [EID]
    assert {"studio_decoding", "qc", "preprocess", "targets"} <= set(manifest["configs"])
    assert manifest["split_hash"] == studio.result.split_hash
    verdicts = json.loads((studio.folder / "choice" / "single_session.json").read_text())
    assert {(t["subject"], t["row"]) for t in verdicts["tests"]} == {
        ("model", "null_trialstruct"),
        ("model_with_task", "null_trialstruct"),
        ("model", "null_shuffle"),
        ("model", "baseline_ridge"),
    }
    report = (studio.folder / "choice" / "report.txt").read_text()
    assert "baseline_rrr: not run (multi-session by design" in report


@needs_bwm
def test_shown_units_are_decoded_and_qc_failures_counted(tmp_path, quick):
    session = load_session(EID, "bwm")
    passing = list(apply_unit_qc(session, load_qc_config()).units.index)
    failing = next(u for u in session.units.index if u not in set(passing))
    chosen = passing[:8] + [failing]
    out = decode(EID, "stimulus_side", unit_ids=chosen, cfg=quick, runs_dir=tmp_path)
    assert out.units_used == passing[:8] and out.units_excluded == [failing]
    summary = out.summary()
    assert summary["units"] == {"used": 8, "excluded_failing_qc": 1}
    assert summary["trials"]["excluded"] == {
        "0% contrast": int(
            (
                session.trials["bwm_include"]
                & (session.trials["contrastLeft"].fillna(session.trials["contrastRight"]) == 0)
            ).sum()
        )
    }
    rows = [r["row"] for r in summary["rows"]]
    assert rows[:2] == ["null_shuffle", "null_trialstruct"] and "baseline_rrr" not in rows
    assert summary["not_applicable"]["baseline_rrr"].startswith("multi-session")
    assert all(np.isfinite(r["auroc"]) for r in summary["rows"] if r["row"] != "null_shuffle")
    assert summary["null"]["n_shifts_used"] <= summary["null"]["n_shifts"] == quick.n_shifts
    assert json.dumps(summary, allow_nan=False)  # strict JSON for the page


def test_a_random_trial_split_is_refused_on_studios_path():
    # R2: a random split interleaves training and test trials in time. Built by hand
    # here (the registry can't make one), it must be refused before anything is fit.
    from test_eval_data import SESSIONS, SPLIT

    from unitwave.cli.evaluate import evaluate_target
    from unitwave.models.baselines.features import load_baseline_config

    record = SPLIT.sessions["e0"]
    n = len(record["trial_intervals"])
    order = np.random.default_rng(0).permutation(n)
    span = [record["trial_intervals"][0][0], record["trial_intervals"][-1][1]]
    shuffled = dataclasses.replace(
        SPLIT,
        partitions={"train": ["e0"], "test": ["e0"]},
        sessions={
            "e0": {
                **record,
                "blocks": {"train": span, "test": span},
                "trials": {
                    "train": sorted(int(i) for i in order[: int(0.8 * n)]),
                    "test": sorted(int(i) for i in order[int(0.8 * n) :]),
                },
            }
        },
    )
    with pytest.raises(ValueError, match="share bins"):
        evaluate_target(
            "choice",
            shuffled,
            None,
            leave_one_block_out=(),
            seed=0,
            n_shifts=5,
            train_stride={},
            baselines=load_baseline_config(),
            load=SESSIONS.__getitem__,
            rrr=False,
        )
