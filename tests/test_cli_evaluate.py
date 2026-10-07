import dataclasses
import json

import pandas as pd
import pytest
from test_eval_data import EIDS, SESSIONS, _manifest

from unitwave.cli.evaluate import REPO, load_run_config, run, select_sessions
from unitwave.preprocess.normalize import Normalizer
from unitwave.splits.registry import load_split

FIRST_TABLE = REPO / "configs/runs/phase3_first_table.yaml"
CONFIRMATION = REPO / "configs/runs/phase3_confirmation.yaml"


def test_fixed_sessions_then_seeded_extras():
    chosen = select_sessions(_manifest(), ["e3", "e0"], 2, seed=0)
    assert chosen[:2] == ["e3", "e0"] and len(set(chosen)) == 4
    assert chosen == select_sessions(_manifest(), ["e3", "e0"], 2, seed=0)
    with pytest.raises(ValueError, match="not in the manifest"):
        select_sessions(_manifest(), ["nope"], 1, seed=0)


def test_the_first_table_config():
    cfg = load_run_config(FIRST_TABLE)
    assert len(cfg.fixed_sessions) == 8 and cfg.random_sessions == 2
    assert cfg.targets == ("choice", "block", "wheel_velocity", "movement_state")
    assert cfg.train_stride == {"wheel_velocity": 1, "movement_state": 5}
    assert (cfg.train_fraction, cfg.gap_s, cfg.model, cfg.n_shifts) == (0.8, 2.0, "ridge", 20)
    assert cfg.leave_one_block_out == ("block",)


def test_the_confirmation_config_differs_only_in_its_sessions():
    first, confirmation = load_run_config(FIRST_TABLE), load_run_config(CONFIRMATION)
    assert len(confirmation.fixed_sessions) == 10 and confirmation.random_sessions == 0
    assert not set(confirmation.fixed_sessions) & set(first.fixed_sessions)
    same = ("targets", "train_stride", "train_fraction", "gap_s", "model", "n_shifts", "seed")
    for field in same + ("leave_one_block_out",):
        assert getattr(confirmation, field) == getattr(first, field), field


def test_bad_configs_are_refused(tmp_path):
    text = FIRST_TABLE.read_text().replace("[choice, block", "[choice, pupil")
    (tmp_path / "bad.yaml").write_text(text)
    with pytest.raises(ValueError, match="pupil"):
        load_run_config(tmp_path / "bad.yaml")
    text = FIRST_TABLE.read_text().replace(
        "leave_one_block_out: [block]", "leave_one_block_out: [wheel_velocity]"
    )
    (tmp_path / "bad.yaml").write_text(text)
    with pytest.raises(ValueError, match="trial targets"):
        load_run_config(tmp_path / "bad.yaml")


def test_a_run_logs_everything(tmp_path):
    cfg = dataclasses.replace(
        load_run_config(FIRST_TABLE),
        name="synthetic",
        fixed_sessions=("e0", "e1", "e2"),
        random_sessions=2,
        targets=("choice", "wheel_velocity"),
        n_shifts=5,
        # The fixtures' 40 trials are too few for pseudo-sessions (90 unbiased trials
        # first), so the leave-one-block-out path runs on choice here.
        leave_one_block_out=("choice",),
    )
    out = run(
        cfg,
        config_path=FIRST_TABLE,
        runs_dir=tmp_path,
        manifest=_manifest(),
        load=SESSIONS.__getitem__,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["git"]["sha"]) == 40
    assert set(manifest["configs"]) == {
        "run",
        "qc",
        "preprocess",
        "targets",
        "nulls",
        "evaluation",
        "baselines",
    }
    assert all(len(c["sha256"]) == 64 for c in manifest["configs"].values())
    assert sorted(manifest["sessions"]) == sorted(EIDS)
    assert load_split(out / "split.json").hash == manifest["split_hash"]
    lobo_hash = load_split(out / "split_leave_one_block_out.json").hash
    assert lobo_hash == manifest["split_leave_one_block_out_hash"] != manifest["split_hash"]
    for target, split_hash in (("choice", lobo_hash), ("wheel_velocity", manifest["split_hash"])):
        folder = out / target
        report = (folder / "report.txt").read_text()
        assert "vacuous" in report and "ceiling_within = model" in report
        metrics = json.loads((folder / "metrics.json").read_text())  # strict JSON: no NaN tokens
        assert set(metrics["rows"]) == set(metrics["summary"])
        assert "model_with_task" in metrics["rows"] and "null_pseudosession" not in metrics["rows"]
        assert set(metrics["rows"]["model"]) == set(EIDS)
        assert metrics["gate"]["subject"] == "model_with_task"
        assert metrics["gate"]["row"] == "null_trialstruct"
        assert "GATE (model_with_task vs null_trialstruct)" in report
        assert metrics["split_hash"] == split_hash == manifest["targets"][target]["split_hash"]
        table = pd.read_parquet(folder / "per_session.parquet")
        assert table.index.names == ["row", "eid"]
        assert pd.read_parquet(folder / "shuffle.parquet").shape[1] == 5
        assert manifest["targets"][target]["seconds"] >= 0
        # R3: the training-only normalisation statistics are stored with the run, one per
        # fold for the leave-one-block-out split.
        if target == "choice":
            raw = json.loads((folder / "normalizers.json").read_text())
            assert metrics["n_folds"] == len(raw) > 1 and not (folder / "normalizer.json").exists()
            normalizers = [Normalizer.from_dict(n) for n in raw]
            assert [n.split_hash for n in normalizers] == [
                f"{split_hash}#fold{k}" for k in range(len(raw))
            ]
        else:
            normalizers = [
                Normalizer.from_dict(json.loads((folder / "normalizer.json").read_text()))
            ]
            assert metrics["n_folds"] == 1 and normalizers[0].split_hash == split_hash
        hashes = [n.hash for n in normalizers]
        assert (
            hashes
            == metrics["normalizer_hashes"]
            == manifest["targets"][target]["normalizer_hashes"]
        )
