"""Tests for the device manager (T-006): ROCm/CPU detection + AMP toggle.

The GPU paths (ROCm/NVIDIA) are exercised when a GPU is present and
skipped otherwise; the CPU fallback path is always tested.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.artifacts import create_run
from core.config import validate_config
from core.device import (
    DeviceError,
    detect_backend,
    describe_device,
    get_device_info,
    log_device_startup,
    resolve_device,
    resolve_precision,
)
from core.trainer import PhaseBundle, Trainer, register_phase
from scripts.train import main as train_main

HAS_GPU = torch.cuda.is_available()


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

    def loss_fn(model: nn.Module, batch: dict):
        logits = model(batch["x"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
        return loss, {}

    return PhaseBundle(
        model=model,
        train_dataset=XORDataset(),
        val_dataset=XORDataset(),
        loss_fn=loss_fn,
    )


register_phase("xor", xor_factory)


def make_cfg(tmp_path: Path, **overrides) -> dict:
    return validate_config(
        {
            "run_name": "device_test",
            "output_dir": str(tmp_path / "artifacts"),
            "device": "cpu",
            "seed": 42,
            "profile": "tiny_debug",
            "batch_size": 4,
            "learning_rate": 0.05,
            "epochs": 10,
            "log_interval": 5,
            "save_interval": 50,
            "val_split": 0.25,
            **overrides,
        }
    )


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def test_detect_backend_is_known_and_consistent():
    backend = detect_backend()
    assert backend in ("rocm", "cuda", "cpu")
    if HAS_GPU:
        assert backend in ("rocm", "cuda")
        if getattr(torch.version, "hip", None):
            assert backend == "rocm"
    else:
        assert backend == "cpu"


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def test_resolve_device_auto():
    device = resolve_device({"device": "auto"})
    if HAS_GPU:
        assert device.type == "cuda"
    else:
        assert device == torch.device("cpu")


def test_resolve_device_cpu():
    assert resolve_device({"device": "cpu"}) == torch.device("cpu")


@pytest.mark.skipif(HAS_GPU, reason="only testable without a GPU")
def test_resolve_device_cuda_unavailable_raises():
    with pytest.raises(DeviceError, match="no CUDA/ROCm device"):
        resolve_device({"device": "cuda"})


@pytest.mark.skipif(not HAS_GPU, reason="requires a GPU")
def test_get_device_info_gpu_reports_vram():
    info = get_device_info({"device": "cuda", "amp": False})
    assert info.device.type == "cuda"
    assert info.backend in ("rocm", "cuda")
    assert info.name
    assert info.vram_total_mb is not None and info.vram_total_mb > 0
    assert info.vram_free_mb is not None and info.vram_free_mb > 0


# ---------------------------------------------------------------------------
# Precision (AMP toggle)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("backend", "amp", "label", "dtype"),
    [
        ("cpu", False, "fp32", None),
        ("cpu", True, "bf16 (AMP)", torch.bfloat16),
        ("rocm", False, "fp32", None),
        ("rocm", True, "bf16 (AMP)", torch.bfloat16),
        ("cuda", False, "fp32", None),
        ("cuda", True, "fp16 (AMP)", torch.float16),
    ],
)
def test_resolve_precision_matrix(backend, amp, label, dtype):
    assert resolve_precision(backend, amp) == (label, dtype)


def test_get_device_info_cpu_precision():
    info = get_device_info({"device": "cpu", "amp": False})
    assert info.device == torch.device("cpu")
    assert info.backend == "cpu"
    assert info.name == "CPU"
    assert info.vram_total_mb is None and info.vram_free_mb is None
    assert info.precision == "fp32"
    assert info.amp_dtype is None

    info_amp = get_device_info({"device": "cpu", "amp": True})
    assert info_amp.precision == "bf16 (AMP)"
    assert info_amp.amp_dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Startup logging
# ---------------------------------------------------------------------------

def test_describe_device_contains_device_and_precision(tmp_path):
    info = get_device_info(make_cfg(tmp_path))
    line = describe_device(info)
    assert f"device={info.device}" in line
    assert f"precision={info.precision}" in line
    assert f"backend={info.backend}" in line


def test_log_device_startup_logs_clear_line():
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("device_test_capture")
    logger.setLevel(logging.INFO)
    logger.addHandler(Capture())
    try:
        info = log_device_startup({"device": "cpu", "amp": True}, logger=logger)
    finally:
        logger.handlers.clear()
    assert len(records) == 1
    message = records[0].getMessage()
    assert message.startswith("startup: ")
    assert f"device={info.device}" in message
    assert f"precision={info.precision}" in message


def test_trainer_startup_log_reports_device_and_precision(tmp_path):
    cfg = make_cfg(tmp_path, phase="xor", epochs=2)
    run = create_run(cfg)
    from core.trainer import build_phase

    trainer = Trainer.from_bundle(build_phase(cfg), config=cfg, run=run)
    trainer.train()
    run.mark_finished()
    log_text = run.log_path.read_text(encoding="utf-8")
    started = next(line for line in log_text.splitlines() if "training started" in line)
    assert f"device={trainer.device}" in started
    assert f"precision={trainer.precision}" in started
    assert f"amp={trainer.amp}" in started


def test_train_script_startup_log_reports_device_and_precision(tmp_path):
    cfg = make_cfg(tmp_path, phase="xor", epochs=2, amp=True)
    cfg_path = tmp_path / "device_e2e.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert train_main(["--config", str(cfg_path)]) == 0

    run_dir = next(p for p in (tmp_path / "artifacts").iterdir() if p.is_dir())
    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    startup = next(line for line in log_text.splitlines() if "startup: " in line)
    assert "device=" in startup
    assert "precision=" in startup
    assert "backend=" in startup


def test_train_script_rejects_unavailable_cuda(tmp_path):
    if HAS_GPU:
        pytest.skip("only testable without a GPU")
    cfg = make_cfg(tmp_path, phase="xor", device="cuda")
    cfg_path = tmp_path / "cuda.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert train_main(["--config", str(cfg_path)]) == 1


# ---------------------------------------------------------------------------
# AMP actually runs
# ---------------------------------------------------------------------------

def test_amp_training_runs_on_cpu(tmp_path):
    cfg = make_cfg(tmp_path, phase="xor", epochs=30, amp=True)
    from core.trainer import build_phase

    trainer = Trainer.from_bundle(build_phase(cfg), config=cfg)
    assert trainer.precision == "bf16 (AMP)"
    summary = trainer.train()
    assert summary["steps"] == 30
    val = trainer.evaluate()
    assert val["val_loss"] < 0.2  # bf16 on CPU is slightly noisier than fp32
