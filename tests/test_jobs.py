"""Tests for the UI job runner (T-008).

A toy XOR phase (the shape of M1, T-101) is registered so the job runner
can be driven end-to-end: start, status polling, log streaming, stop,
error handling.  The runner is exercised directly (not through the UI);
the ``/jobs`` page wiring is covered in ``tests/test_ui.py``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apps.ui.jobs import JobError, JobRunner
from core.config import ConfigError, validate_config
from core.device import DeviceError
from core.trainer import PhaseBundle, TrainerError, register_phase


# ---------------------------------------------------------------------------
# Toy XOR phase (same shape as the M1 phase)
# ---------------------------------------------------------------------------

class XORDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [(a, b, a ^ b) for a in (0, 1) for b in (0, 1)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        a, b, y = self.samples[i]
        return {
            "x": torch.tensor([a, b], dtype=torch.float32),
            "y": torch.tensor([y], dtype=torch.float32),
        }


def xor_factory(cfg: dict) -> PhaseBundle:
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))

    # Identical to the toy XOR in tests/test_trainer.py: the phase registry
    # is global and last-write-wins, so both "xor" registrations must agree
    # (same model, dataset and metrics) or import order decides which one
    # wins and breaks the other suite.
    def loss_fn(model: nn.Module, batch: dict):
        logits = model(batch["x"])  # (B, 1)
        loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
        with torch.no_grad():
            acc = (
                ((torch.sigmoid(logits) >= 0.5).float() == batch["y"]).float().mean().item()
            )
        return loss, {"acc": acc}

    return PhaseBundle(
        model=model,
        train_dataset=XORDataset(),
        val_dataset=XORDataset(),
        loss_fn=loss_fn,
    )


register_phase("xor", xor_factory)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(tmp_path: Path, **overrides) -> dict:
    return validate_config(
        {
            "run_name": "job_test",
            "output_dir": str(tmp_path / "artifacts"),
            "device": "cpu",
            "seed": 42,
            "profile": "tiny_debug",
            "batch_size": 4,
            "learning_rate": 0.05,
            "epochs": 10,
            "log_interval": 5,
            "save_interval": 5,
            "val_split": 0.25,
            "phase": "xor",
            **overrides,
        }
    )


def write_cfg(tmp_path: Path, name: str, **overrides) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(make_cfg(tmp_path, **overrides)), encoding="utf-8")
    return path


def wait_for_state(runner: JobRunner, states: set[str], timeout: float = 60.0) -> dict:
    """Poll ``runner.status()`` until the state is in ``states``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = runner.status()
        if status["state"] in states:
            return status
        time.sleep(0.05)
    raise AssertionError(f"job did not reach {states}; state={runner.status()}")


# ---------------------------------------------------------------------------
# Start / finish
# ---------------------------------------------------------------------------

def test_start_and_finish(tmp_path):
    runner = JobRunner()
    runner.start(write_cfg(tmp_path, "xor"))
    status = wait_for_state(runner, {"finished"})

    assert status["state"] == "finished"
    assert status["error"] is None
    assert status["phase"] == "xor"
    assert status["step"] == 10  # 4 samples / batch 4 = 1 step per epoch
    assert status["epoch"] == 10
    assert status["elapsed_s"] is not None and status["elapsed_s"] >= 0

    run_dir = Path(status["run_dir"])
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "run.log").is_file()
    assert (run_dir / "checkpoints" / "final.pt").is_file()
    assert (run_dir / "checkpoints" / "step_000005.pt").is_file()
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished"

    logs = "\n".join(runner.drain_logs())
    assert "job started" in logs
    assert "startup:" in logs
    assert "training started" in logs
    assert "training finished" in logs
    assert "checkpoint saved" in logs
    assert runner.drain_logs() == []  # queue fully drained


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

def test_stop_stops_job_early(tmp_path):
    runner = JobRunner()
    runner.start(write_cfg(tmp_path, "xor", epochs=1000))
    wait_for_state(runner, {"running"})
    runner.stop()
    status = wait_for_state(runner, {"stopped"})

    assert status["state"] == "stopped"
    assert status["step"] is not None and status["step"] < 1000

    run_dir = Path(status["run_dir"])
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "interrupted"
    # The Trainer still saves the final checkpoint on a requested stop.
    assert (run_dir / "checkpoints" / "final.pt").is_file()


def test_stop_without_job_raises(tmp_path):
    runner = JobRunner()
    with pytest.raises(JobError, match="no active job"):
        runner.stop()


def test_reset_while_active_raises(tmp_path):
    runner = JobRunner()
    runner.start(write_cfg(tmp_path, "xor", epochs=1000))
    wait_for_state(runner, {"running"})
    with pytest.raises(JobError, match="cannot reset"):
        runner.reset()
    runner.stop()
    wait_for_state(runner, {"stopped"})


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_start_rejects_second_job(tmp_path):
    runner = JobRunner()
    runner.start(write_cfg(tmp_path, "xor", epochs=1000))
    wait_for_state(runner, {"running"})
    with pytest.raises(JobError, match="already active"):
        runner.start(write_cfg(tmp_path, "xor2"))
    runner.stop()
    wait_for_state(runner, {"stopped"})


def test_start_rejects_missing_config(tmp_path):
    runner = JobRunner()
    with pytest.raises(ConfigError, match="not found"):
        runner.start(tmp_path / "missing.yaml")
    assert runner.status()["state"] == "idle"


def test_start_rejects_unregistered_phase(tmp_path):
    runner = JobRunner()
    cfg_path = write_cfg(tmp_path, "tok", phase="tokenizer")
    with pytest.raises(TrainerError, match="no registered factory"):
        runner.start(cfg_path)
    assert runner.status()["state"] == "idle"


def test_failed_job_marks_run_failed(tmp_path, monkeypatch):
    from apps.ui import jobs as jobs_module

    def boom(cfg):
        raise DeviceError("simulated device failure")

    monkeypatch.setattr(jobs_module, "get_device_info", boom)
    runner = JobRunner()
    runner.start(write_cfg(tmp_path, "xor"))
    status = wait_for_state(runner, {"failed"})

    assert status["state"] == "failed"
    assert "simulated device failure" in (status["error"] or "")

    run_dir = Path(status["run_dir"])
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_state(tmp_path):
    runner = JobRunner()
    runner.start(write_cfg(tmp_path, "xor"))
    wait_for_state(runner, {"finished"})
    assert runner.status()["state"] == "finished"

    runner.reset()
    status = runner.status()
    assert status["state"] == "idle"
    assert status["run_dir"] is None
    assert status["step"] is None
    assert status["error"] is None
    assert status["elapsed_s"] is None
