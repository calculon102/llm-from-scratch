"""Tests for the finite-difference gradient check (T-103).

Covers the standalone functions in :mod:`core.gradcheck`, the
``Trainer.check_gradients`` method (report file + run log), the
``gradcheck`` config key, and the full CLI path
(``scripts.train`` with ``gradcheck: true``).

A "broken" model is built with a custom ``torch.autograd.Function`` whose
``backward`` deliberately returns *half* the gradient for one weight —
the classic silent-gradient bug — and the check must flag exactly that
parameter.  Models are built inline (like ``test_trainer.py``) instead of
importing ``core.xor`` so this file does not touch the phase registry.
"""

from __future__ import annotations

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
from core.config import ConfigError, validate_config
from core.gradcheck import (
    DEFAULT_EPS,
    DEFAULT_RTOL,
    compare_gradients,
    format_report,
    gradient_check,
)
from core.trainer import Trainer
from scripts.train import main as train_main


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_model(hidden_size: int = 8) -> nn.Module:
    """A correct XOR MLP (same shape as the M1 phase)."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
    )


def xor_loss(model: nn.Module, batch: dict):
    logits = model(batch["x"])
    loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
    return loss, {"acc": 0.0}


def make_batch(repeats: int = 25) -> dict:
    xs, ys = [], []
    for a in (0, 1):
        for b in (0, 1):
            for _ in range(repeats):
                xs.append([float(a), float(b)])
                ys.append([float(a ^ b)])
    return {
        "x": torch.tensor(xs, dtype=torch.float32),
        "y": torch.tensor(ys, dtype=torch.float32),
    }


class XorDataset(Dataset):
    """Dict-of-tensors dataset in the shared Trainer's batch convention."""

    def __init__(self, repeats: int = 25) -> None:
        self._batch = make_batch(repeats)

    def __len__(self) -> int:
        return self._batch["x"].shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"x": self._batch["x"][i], "y": self._batch["y"][i]}


class HalfGrad(torch.autograd.Function):
    """Identity matmul whose ``backward`` is *wrong* on purpose: it sends
    back only half the gradient into the weight.  A perfectly typical bug
    in hand-written backward code (a missing factor, a copied formula)."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        # BUG: the true gradient is grad_out.t() @ x — half of it here.
        return grad_out.t() @ x * 0.5, None


class BrokenModel(nn.Module):
    """MLP whose first layer goes through the buggy ``HalfGrad`` function,
    so the analytic gradient of ``buggy.weight`` is exactly half the true
    value while every other parameter is correct."""

    def __init__(self) -> None:
        super().__init__()
        self.buggy = nn.Linear(2, 4)
        self.out = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = HalfGrad.apply(self.buggy.weight, x)
        return self.out(torch.tanh(h))


def make_broken_model() -> nn.Module:
    torch.manual_seed(42)
    return BrokenModel()


def make_trainer(tmp_path: Path, model: nn.Module | None = None, **overrides):
    cfg = validate_config(
        {
            "run_name": "gradcheck_test",
            "output_dir": str(tmp_path / "artifacts"),
            "device": "cpu",
            "seed": 42,
            "profile": "tiny_debug",
            "batch_size": 32,
            "learning_rate": 0.05,
            "epochs": 1,
            "log_interval": 1,
            "val_split": 0.25,
            **overrides,
        }
    )
    model = model or make_model()
    run = create_run(cfg)
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"]),
        XorDataset(),
        config=cfg,
        run=run,
        loss_fn=xor_loss,
    )
    return trainer, cfg, run


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def test_correct_model_passes():
    report = gradient_check(make_model(), xor_loss, make_batch())
    assert report.passed
    assert report.n_failed_params == 0
    assert report.max_rel_err <= DEFAULT_RTOL


def test_all_parameters_are_in_the_report():
    report = gradient_check(make_model(), xor_loss, make_batch())
    assert [row.name for row in report.rows] == [
        "0.weight", "0.bias", "2.weight", "2.bias"
    ]
    assert report.num_params == 4
    assert report.total_elements == 16 + 8 + 8 + 1  # 2x8 + 8 + 8x1 + 1
    for row in report.rows:
        assert row.num_elements == int(torch.tensor(row.shape).prod())


def test_broken_backward_is_reported():
    report = gradient_check(make_broken_model(), xor_loss, make_batch())
    assert not report.passed
    # The bug must be localized to the parameter with the wrong backward.
    assert report.worst_param == "buggy.weight"
    buggy = next(r for r in report.rows if r.name == "buggy.weight")
    assert not buggy.ok
    assert buggy.max_rel_err > DEFAULT_RTOL
    # ...while the untouched parameters stay within tolerance.
    out = next(r for r in report.rows if r.name == "out.weight")
    assert out.ok


def test_check_does_not_modify_the_model():
    model = make_model()
    before = [p.detach().clone() for p in model.parameters()]
    gradient_check(model, xor_loss, make_batch())
    for param, original in zip(model.parameters(), before):
        assert torch.equal(param.detach(), original)


def test_gradient_check_requires_batch():
    with pytest.raises(ValueError, match="needs a batch"):
        gradient_check(make_model(), xor_loss, None)


def test_compare_gradients_normalization():
    # Element 1: |1.0 - 1.0001| = 1e-4, scale max(1, |a|, |n|) = 1.0001
    # -> rel_err = 1e-4 / 1.0001  (~9.999e-5, below rtol)
    # Element 2: |0.01 - 0.01002| = 2e-5, scale = 1 (|a|, |n| < 1)
    # -> rel_err = 2e-5 (absolute regime for small gradients)
    report = compare_gradients(
        {"w": torch.tensor([1.0, 0.01])},
        {"w": torch.tensor([1.0001, 0.01002])},
        eps=DEFAULT_EPS,
        rtol=DEFAULT_RTOL,
    )
    row = report.rows[0]
    assert row.max_rel_err == pytest.approx(1e-4 / 1.0001, rel=1e-3)
    assert row.ok and report.passed


def test_compare_gradients_flags_real_mismatch():
    # First element off by 0.5: rel_err = 0.5 / max(1, 1, 0.5) = 0.5 >> rtol.
    report = compare_gradients(
        {"w": torch.tensor([1.0, 0.01])},
        {"w": torch.tensor([0.5, 0.01])},
        eps=DEFAULT_EPS,
        rtol=DEFAULT_RTOL,
    )
    assert not report.rows[0].ok
    assert not report.passed
    assert report.worst_param == "w"


def test_compare_gradients_empty_model():
    report = compare_gradients({}, {}, eps=DEFAULT_EPS, rtol=DEFAULT_RTOL)
    assert report.passed
    assert report.num_params == 0
    assert report.worst_param is None


def test_format_report_contains_verdict_and_parameters():
    report = gradient_check(make_model(), xor_loss, make_batch())
    text = format_report(report)
    assert "PASS" in text
    assert "gradient check" in text.lower()
    for name in ("0.weight", "0.bias", "2.weight", "2.bias"):
        assert name in text


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------

def test_trainer_check_gradients_writes_report(tmp_path):
    trainer, cfg, run = make_trainer(tmp_path)
    report = trainer.check_gradients()
    assert report.passed
    report_path = run.run_dir / "gradcheck_report.txt"
    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    assert "PASS" in text
    assert "0.weight" in text
    assert "gradient check" in (run.run_dir / "run.log").read_text(encoding="utf-8")


def test_trainer_check_gradients_reports_fail(tmp_path):
    trainer, cfg, run = make_trainer(tmp_path, model=make_broken_model())
    report = trainer.check_gradients()
    assert not report.passed
    text = (run.run_dir / "gradcheck_report.txt").read_text(encoding="utf-8")
    assert "FAIL" in text
    assert "buggy.weight" in text  # the report points at the culprit


def test_trainer_check_gradients_without_run_does_not_write(tmp_path):
    cfg = validate_config({"device": "cpu", "profile": "tiny_debug"})
    model = make_model()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"]),
        XorDataset(),
        config=cfg,
        loss_fn=xor_loss,
    )
    assert trainer.run is None
    report = trainer.check_gradients()  # must not raise
    assert report.passed


# ---------------------------------------------------------------------------
# Config key
# ---------------------------------------------------------------------------

def test_gradcheck_config_default_is_false():
    cfg = validate_config({})
    assert cfg["gradcheck"] is False


def test_gradcheck_config_rejects_non_bool():
    with pytest.raises(ConfigError):
        validate_config({"gradcheck": "yes"})


# ---------------------------------------------------------------------------
# End-to-end: CLI
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, *, gradcheck: bool) -> Path:
    config = {
        "run_name": "e2e_gradcheck",
        "phase": "xor",
        "seed": 42,
        "device": "cpu",
        "amp": False,
        "profile": "tiny_debug",
        "output_dir": str(tmp_path / "artifacts"),
        "log_interval": 10,
        "batch_size": 32,
        "learning_rate": 0.05,
        "epochs": 2,
        "val_split": 0.25,
        "gradcheck": gradcheck,
    }
    path = tmp_path / f"gradcheck_{gradcheck}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_train_cli_with_gradcheck_writes_report(tmp_path):
    config_path = _write_config(tmp_path, gradcheck=True)
    rc = train_main(["--config", str(config_path)])
    assert rc == 0
    run_dirs = [p for p in (tmp_path / "artifacts").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    report = run_dir / "gradcheck_report.txt"
    assert report.is_file()
    assert "PASS" in report.read_text(encoding="utf-8")
    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "gradient check" in log_text


def test_train_cli_without_gradcheck_writes_no_report(tmp_path):
    config_path = _write_config(tmp_path, gradcheck=False)
    rc = train_main(["--config", str(config_path)])
    assert rc == 0
    run_dirs = [p for p in (tmp_path / "artifacts").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "gradcheck_report.txt").exists()
    assert "gradient check" not in (run_dirs[0] / "run.log").read_text(encoding="utf-8")
