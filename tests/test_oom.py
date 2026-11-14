"""Tests for OOM fallback hints and auto-save on interrupt (T-503).

Acceptance: "failures provide actionable fallback suggestions".  The tests
drive a real Trainer onto the failure paths (simulated OOM, Ctrl+C) and
assert both the checkpoint and the human-readable guidance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pytest

from core.artifacts import create_run
from core.config import validate_config
from core.oom import (
    TrainingOOMError,
    fallback_suggestions,
    format_oom_help,
    is_oom_error,
    next_smaller_profile,
)
from core.trainer import PhaseBundle, Trainer
from scripts.train import _sigterm_to_interrupt


class OneDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict:
        return {"x": torch.tensor([1.0])}


def make_trainer(tmp_path, **overrides) -> tuple[Trainer, object]:
    cfg = validate_config(
        {
            "run_name": "oom_test",
            "output_dir": str(tmp_path / "artifacts"),
            "device": "cpu",
            "epochs": 10,
            "batch_size": 2,
            "log_interval": 1,
            "save_interval": 100,
            "profile": "small",
            **overrides,
        }
    )
    model = nn.Linear(1, 1)
    bundle = PhaseBundle(
        model=model,
        train_dataset=OneDataset(),
        loss_fn=lambda model, batch: (model(batch["x"]).sum() * 0, {}),
    )
    run = create_run(cfg)
    return Trainer.from_bundle(bundle, config=cfg, run=run), run


def test_is_oom_error_recognises_cuda_and_rocm_messages() -> None:
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert is_oom_error(RuntimeError("HIP out of memory"))
    assert is_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert is_oom_error(torch.cuda.OutOfMemoryError("boom"))
    assert not is_oom_error(RuntimeError("shape mismatch"))
    assert not is_oom_error(ValueError("bad config"))


def test_next_smaller_profile_walks_the_ladder() -> None:
    assert next_smaller_profile("max_7900xtx") == "medium"
    assert next_smaller_profile("medium") == "small"
    assert next_smaller_profile("small") == "tiny_train"
    assert next_smaller_profile("tiny_train") == "tiny_debug"
    assert next_smaller_profile("tiny_debug") is None
    assert next_smaller_profile("unknown") is None


def test_fallback_suggestions_are_actionable() -> None:
    cfg = validate_config(
        {
            "profile": "small",
            "profile_defaults": True,
            "phase": "gpt",
            "batch_size": 16,
        }
    )
    hints = fallback_suggestions(cfg, checkpoint="artifacts/run/checkpoints/final.pt")
    text = "\n".join(hints)
    assert "batch_size: 16 -> 8" in text
    assert "context_length: 256 -> 128" in text
    assert "AMP is already on" in text
    assert "small -> tiny_train" in text
    assert "--resume artifacts/run/checkpoints/final.pt" in text

    tiny_train = fallback_suggestions(
        validate_config(
            {"profile": "tiny_train", "profile_defaults": True, "phase": "gpt"}
        )
    )
    assert any("Enable AMP: set amp: true" in hint for hint in tiny_train)

    smallest = fallback_suggestions(
        validate_config({"profile": "tiny_debug", "profile_defaults": True, "phase": "gpt"})
    )
    assert any("smallest profile" in hint for hint in smallest)


def test_format_oom_help_includes_resume_command() -> None:
    cfg = validate_config({"profile": "medium", "profile_defaults": True, "phase": "gpt"})
    message = format_oom_help(
        RuntimeError("CUDA out of memory"),
        cfg,
        checkpoint="artifacts/x/checkpoints/step_000010.pt",
        config_path="configs/gpt_medium.yaml",
    )
    assert "out of memory while training (profile=medium)" in message
    assert "Fallback suggestions:" in message
    assert "1. " in message
    assert (
        "python -m scripts.train --config configs/gpt_medium.yaml "
        "--resume artifacts/x/checkpoints/step_000010.pt"
    ) in message


def test_oom_saves_checkpoint_and_raises_with_hints(tmp_path, monkeypatch) -> None:
    trainer, run = make_trainer(tmp_path)

    def raise_oom(batch):
        raise RuntimeError("CUDA out of memory. Tried to allocate 3.00 GiB")

    monkeypatch.setattr(trainer, "_train_step", raise_oom)
    with pytest.raises(TrainingOOMError) as excinfo:
        trainer.train()

    error = excinfo.value
    assert error.hints
    assert error.checkpoint is not None
    assert error.checkpoint.is_file()
    assert "batch_size" in str(error)
    assert "Fallback suggestions:" in str(error)

    log_text = run.log_path.read_text(encoding="utf-8")
    assert "out of memory while training" in log_text
    assert "Fallback suggestions:" in log_text
    assert "checkpoint saved" in log_text


def test_keyboard_interrupt_saves_last_step_checkpoint(tmp_path, monkeypatch) -> None:
    trainer, run = make_trainer(tmp_path)
    calls = {"n": 0}

    def interrupt_on_second_step(batch):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return torch.tensor(0.0, requires_grad=True), {}

    monkeypatch.setattr(trainer, "_train_step", interrupt_on_second_step)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()

    checkpoint = run.checkpoints_dir / "step_000001.pt"
    assert checkpoint.is_file()
    assert run.latest_checkpoint() == checkpoint
    log_text = run.log_path.read_text(encoding="utf-8")
    assert "interrupted at step 1" in log_text
    assert "checkpoint saved to" in log_text


def test_sigterm_handler_raises_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt):
        _sigterm_to_interrupt(15, None)
