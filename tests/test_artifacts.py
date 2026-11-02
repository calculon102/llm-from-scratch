"""Tests for the per-run artifact manager (T-004)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from core.artifacts import ArtifactError, create_run
from core.config import load_config, validate_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_config(tmp_path: Path, **overrides) -> dict:
    return validate_config(
        {"run_name": "demo", "output_dir": str(tmp_path / "artifacts"), **overrides}
    )


def test_run_layout_created(tmp_path):
    run = create_run(make_config(tmp_path))
    assert run.run_dir.is_dir()
    assert run.run_dir.parent == tmp_path / "artifacts"
    assert run.config_path.is_file()
    assert run.log_path.is_file()
    assert run.meta_path.is_file()
    assert run.checkpoints_dir.is_dir()
    assert run.plots_dir.is_dir()
    assert run.index_path.is_file()


def test_run_dir_name_has_timestamp_and_run_name(tmp_path):
    run = create_run(make_config(tmp_path, run_name="xor_tiny"))
    assert run.run_dir.name.endswith("_xor_tiny")
    prefix = run.run_dir.name[: -len("_xor_tiny")]
    # prefix is YYYYMMDD-HHMMSS
    assert len(prefix) == 15 and prefix[8] == "-"


def test_two_runs_get_distinct_dirs(tmp_path):
    a = create_run(make_config(tmp_path))
    b = create_run(make_config(tmp_path))
    assert a.run_dir != b.run_dir
    assert a.run_dir.is_dir() and b.run_dir.is_dir()


def test_create_run_requires_mapping(tmp_path):
    with pytest.raises(ArtifactError, match="mapping"):
        create_run(["not", "a", "dict"])


def test_output_dir_and_run_name_overrides(tmp_path):
    cfg = make_config(tmp_path)
    other = tmp_path / "elsewhere"
    run = create_run(cfg, output_dir=other, run_name="custom")
    assert run.run_dir.parent == other
    assert run.run_dir.name.endswith("_custom")


def test_config_snapshot_round_trips(tmp_path):
    cfg = make_config(tmp_path)
    run = create_run(cfg)
    reloaded = load_config(run.config_path)
    assert reloaded == cfg


def test_meta_lifecycle(tmp_path):
    cfg = make_config(tmp_path, profile="small", phase="xor", seed=7)
    run = create_run(cfg)
    meta = json.loads(run.meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "running"
    assert meta["run_name"] == "demo"
    assert meta["profile"] == "small"
    assert meta["phase"] == "xor"
    assert meta["seed"] == 7
    assert meta["end_time"] is None

    run.mark_finished()
    meta = json.loads(run.meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "finished"
    assert meta["end_time"] is not None


def test_index_accumulates_across_runs(tmp_path):
    a = create_run(make_config(tmp_path))
    b = create_run(make_config(tmp_path))
    assert a.index_path == b.index_path
    lines = [
        json.loads(line)
        for line in a.index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    assert [line["run_dir"] for line in lines] == [str(a.run_dir), str(b.run_dir)]


def test_logger_writes_to_file(tmp_path):
    run = create_run(make_config(tmp_path))
    run.log("hello artifacts")
    run.logger.warning("careful")
    text = run.log_path.read_text(encoding="utf-8")
    assert "hello artifacts" in text
    assert "WARNING" in text
    assert "careful" in text


def test_checkpoint_roundtrip(tmp_path):
    import torch

    run = create_run(make_config(tmp_path))
    state = {"w": torch.tensor([1.0, -2.0]), "step": 5}
    path = run.save_checkpoint(state, step=5)
    assert path.is_file()
    assert path.name == "step_000005.pt"
    assert run.latest_checkpoint() == path

    run.save_checkpoint(state, step=10)
    assert run.latest_checkpoint().name == "step_000010.pt"

    loaded = run.load_checkpoint()
    assert torch.equal(loaded["w"], state["w"])
    assert loaded["step"] == 5


def test_checkpoint_without_step_is_final(tmp_path):
    import torch

    run = create_run(make_config(tmp_path))
    path = run.save_checkpoint({"w": torch.zeros(1)})
    assert path.name == "final.pt"
    assert run.latest_checkpoint() == path


def test_load_checkpoint_missing_raises(tmp_path):
    run = create_run(make_config(tmp_path))
    with pytest.raises(ArtifactError, match="no checkpoints"):
        run.load_checkpoint()
    with pytest.raises(ArtifactError, match="not found"):
        run.load_checkpoint(tmp_path / "nope.pt")


def test_save_plot(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = create_run(make_config(tmp_path))
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [3, 2, 1])
    path = run.save_plot(fig, "loss")
    assert path.is_file()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0
    plt.close(fig)


def test_save_plot_requires_viz_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    run = create_run(make_config(tmp_path))
    with pytest.raises(ArtifactError, match="viz extra"):
        run.save_plot(object(), "loss")


def test_sample_profile_creates_run_in_tmp(tmp_path):
    cfg = load_config(REPO_ROOT / "configs" / "tiny_debug.yaml")
    run = create_run(cfg, output_dir=tmp_path)
    assert run.run_dir.is_dir()
    assert run.config_path.is_file()
