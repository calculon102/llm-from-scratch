"""Tests for the XOR page (T-104).

Covers the page's acceptance criterion — *retraining from the UI updates
the outputs* — plus the runner features the page relies on: UI-typed
config overrides, the recorded per-step/per-epoch training history, and
the loss-curve figure.

Headless UI tests use the NiceGUI ``user`` fixture (see
``tests/test_ui.py``): it re-executes ``apps/ui/main.py`` per test, so
all routes (including ``/xor`` from ``apps/ui/xor_page.py``) are
registered.  The XOR page owns a *module-level* job runner
(``xor_job_runner``) that persists across those per-test app
re-runs — it lives in ``apps.ui.jobs``, whose module is not purged by
NiceGUI's test teardown — so every UI test resets it first.

Like ``tests/test_jobs.py`` and ``tests/test_ui.py``, a toy XOR phase
(4 corner samples, 2→8→1 MLP, BCE + accuracy) is registered with the
global phase registry.  The registry is global and last-write-wins, and
this file imports last (alphabetically), so its registration governs the
whole session — it must stay behaviorally identical to the other two,
otherwise the ``step == epochs`` arithmetic breaks (4 samples, 25%
validation split, batch 4 ⇒ exactly one step per epoch).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from nicegui import ui
from nicegui.testing import User
from torch.utils.data import Dataset

from apps.ui.jobs import JobRunner, xor_job_runner
from core.config import ConfigError, validate_config
from core.trainer import PhaseBundle, register_phase
from core.xor import plots

# ---------------------------------------------------------------------------
# Toy XOR phase (identical to tests/test_jobs.py and tests/test_ui.py)
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

    # Identical to the toy XOR in tests/test_trainer.py, tests/test_jobs.py
    # and tests/test_ui.py: the phase registry is global and
    # last-write-wins, so all "xor" registrations must agree (same model,
    # dataset and metrics) or import order decides which one wins and
    # breaks the other suite.
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


def write_xor_cfg(tmp_path: Path, **overrides) -> Path:
    """Write a validated, fast XOR config (1 step per epoch) into ``tmp_path``."""
    cfg = validate_config(
        {
            "run_name": "xor_ui",
            "output_dir": str(tmp_path / "artifacts"),
            "device": "cpu",
            "seed": 42,
            "profile": "tiny_debug",
            "batch_size": 4,
            "learning_rate": 0.05,
            "epochs": 10,
            # log_interval=1: every step is a logged step, so the runner's
            # history records one row per step (10 rows for 10 epochs).
            "log_interval": 1,
            "save_interval": 5,
            "val_split": 0.25,
            "phase": "xor",
            **overrides,
        }
    )
    path = tmp_path / "xor_ui.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def wait_for_state_sync(runner: JobRunner, states: set[str], timeout: float = 60.0) -> dict:
    """Poll ``runner.status()`` (sync variant for non-UI tests)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = runner.status()
        if status["state"] in states:
            return status
        time.sleep(0.05)
    raise AssertionError(f"job did not reach {states}; state={runner.status()}")


async def wait_for_state(runner: JobRunner, states: set[str], timeout: float = 60.0) -> dict:
    """Poll ``runner.status()`` until its state is in ``states``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = runner.status()
        if status["state"] in states:
            return status
        await asyncio.sleep(0.05)
    raise AssertionError(f"job did not reach {states}; state={runner.status()}")


async def reset_xor_runner() -> None:
    """Reset the module-level XOR runner before each UI test.

    xor_job_runner lives in apps.ui.jobs (not in xor_page) so it survives
    NiceGUI's per-test module purge and is the same object in test code and
    page code.  A previous test's job thread may still be finishing when
    the next test starts; reset() raises while the state is active, so wait.
    """
    deadline = time.time() + 30
    while time.time() < deadline:
        if xor_job_runner.status()["state"] not in ("starting", "running", "stopping"):
            break
        await asyncio.sleep(0.05)
    xor_job_runner.reset()


def base_config_select(user: User) -> ui.select:
    """The page's "Config file" select (the other select is the activation)."""
    for select in user.find(kind=ui.select).elements:
        if select.label == "Config file":
            return select
    raise AssertionError("no ui.select with label 'Config file' on the page")


def number_by_label(user: User, label: str) -> ui.number:
    for element in user.find(kind=ui.number).elements:
        if element.label == label:
            return element
    raise AssertionError(f"no ui.number with label {label!r} on the page")


def select_base_config(user: User, config_path: Path) -> None:
    """Point the page's config select at ``config_path``.

    The select only offers configs from ``configs/``; the test's temp
    config lives elsewhere, so register it as an option first (a
    ``ui.select`` resets a value that is not one of its options to
    ``None``).  Setting the value fires the change handler — and
    therefore the controls' prefill — *synchronously*, so the control
    values are already updated when this returns.
    """
    select = base_config_select(user)
    if str(config_path) not in select.options:
        select.options = list(select.options) + [str(config_path)]
    select.value = str(config_path)


# ---------------------------------------------------------------------------
# Page wiring
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_xor_route_reachable(user: User, anyio_backend: str) -> None:
    await reset_xor_runner()
    await user.open("/xor")
    await user.should_see("XOR Learning")
    for marker in (
        "xor-train",
        "xor-stop",
        "xor-status",
        "xor-run-dir",
        "xor-log",
        "xor-loss-plot",
        "xor-boundary-plot",
        "xor-activations-plot",
    ):
        await user.should_see(marker=marker)


@pytest.mark.anyio
async def test_base_config_change_prefills_controls(
    user: User, anyio_backend: str, tmp_path: Path
) -> None:
    await reset_xor_runner()
    cfg_path = write_xor_cfg(tmp_path, learning_rate=0.123, epochs=7, seed=1234)

    await user.open("/xor")
    # The page opens on configs/xor_tiny_debug.yaml and prefills its
    # values (epochs 50, hidden 8, activation tanh, ...).
    select_base_config(user, cfg_path)

    # The temp config's values replace the file's values ...
    assert number_by_label(user, "learning rate").value == 0.123
    assert number_by_label(user, "epochs").value == 7
    assert number_by_label(user, "seed").value == 1234
    # ... while settings the temp config does not mention (it has no
    # phase_config section) keep the values prefilled from the default
    # profile: a changed base config only updates what it itself defines.
    assert number_by_label(user, "hidden units").value == 8


# ---------------------------------------------------------------------------
# Acceptance: retrain from the UI updates the outputs
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_retrain_from_ui_updates_outputs(
    user: User, anyio_backend: str, tmp_path: Path
) -> None:
    await reset_xor_runner()
    cfg_path = write_xor_cfg(tmp_path, epochs=20)

    await user.open("/xor")
    select_base_config(user, cfg_path)
    # UI-typed override: 6 hidden units instead of the config's 8.
    number_by_label(user, "hidden units").value = 6
    user.find(marker="xor-train").click()

    # The job runs in a background thread; wait for it to finish.
    status = await wait_for_state(xor_job_runner, {"finished", "failed"})
    assert status["state"] == "finished", status["error"]
    # 4 samples, batch 4, 25% validation split → 1 step per epoch.
    assert status["step"] == 20

    # The on_end hook wrote all three final plots into the run folder.
    run_dir = Path(status["run_dir"])
    for name in ("loss_curve.png", "decision_boundary.png", "activations.png"):
        assert (run_dir / "plots" / name).is_file(), f"missing {name}"

    # The config snapshot is the *override-applied* config: the UI values
    # (hidden 6, epochs 20) are what actually ran.
    snapshot = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert snapshot["epochs"] == 20
    assert snapshot["phase_config"]["hidden_size"] == 6

    # The runner recorded the training history the loss curve is built from:
    # per-step rows with the training loss, per-epoch rows with val_loss
    # (the temp config has a 25% validation split).
    history = xor_job_runner.history()
    assert any("loss" in r for r in history)
    assert any("val_loss" in r for r in history)

    # The page timer picked the finished state up into the status label
    # and the run-dir line.
    await user.should_see(marker="xor-status", content="finished", retries=100)
    await user.should_see(marker="xor-run-dir", content="run:", retries=100)


@pytest.mark.anyio
async def test_stop_from_ui(user: User, anyio_backend: str, tmp_path: Path) -> None:
    await reset_xor_runner()
    cfg_path = write_xor_cfg(tmp_path, epochs=10000)

    await user.open("/xor")
    select_base_config(user, cfg_path)
    user.find(marker="xor-train").click()

    # Wait until the job is running and the timer has enabled Stop.
    stop_btn = next(iter(user.find(marker="xor-stop").elements))
    deadline = time.time() + 30
    while time.time() < deadline:
        if xor_job_runner.status()["state"] == "running" and stop_btn.enabled:
            break
        await asyncio.sleep(0.05)
    assert xor_job_runner.status()["state"] == "running" and stop_btn.enabled

    user.find(marker="xor-stop").click()
    status = await wait_for_state(xor_job_runner, {"stopped", "failed"})
    assert status["state"] == "stopped", status["error"]
    assert status["step"] is not None and status["step"] < 10000

    await user.should_see(marker="xor-status", content="stopped", retries=100)


# ---------------------------------------------------------------------------
# JobRunner features the page relies on (no UI involved)
# ---------------------------------------------------------------------------


def test_runner_records_history(tmp_path: Path) -> None:
    runner = JobRunner()
    runner.start(write_xor_cfg(tmp_path, epochs=3, log_interval=1))
    status = wait_for_state_sync(runner, {"finished"})
    assert status["state"] == "finished", status["error"]

    history = runner.history()
    # One per-step row (with the training loss and the extra "acc"
    # metric) and one per-epoch row (with the evaluation's val_loss).
    step_rows = [r for r in history if "loss" in r]
    epoch_rows = [r for r in history if "val_loss" in r]
    assert len(step_rows) == 3
    assert len(epoch_rows) == 3
    assert all({"step", "epoch", "loss", "acc"} <= set(r) for r in step_rows)
    assert all("epoch" in r and "step" in r for r in epoch_rows)


def test_start_applies_overrides(tmp_path: Path) -> None:
    runner = JobRunner()
    cfg_path = write_xor_cfg(tmp_path)
    runner.start(
        cfg_path,
        overrides={"epochs": 3, "phase_config": {"hidden_size": 4}},
    )
    status = wait_for_state_sync(runner, {"finished", "failed"})
    assert status["state"] == "finished", status["error"]
    assert status["step"] == 3  # overrides took effect: 3 epochs, 1 step each

    snapshot = yaml.safe_load(
        (Path(status["run_dir"]) / "config.yaml").read_text(encoding="utf-8")
    )
    assert snapshot["epochs"] == 3
    assert snapshot["phase_config"]["hidden_size"] == 4


def test_invalid_overrides_raise_config_error(tmp_path: Path) -> None:
    runner = JobRunner()
    cfg_path = write_xor_cfg(tmp_path)

    # learning_rate < 0 fails the base schema ...
    with pytest.raises(ConfigError):
        runner.start(cfg_path, overrides={"learning_rate": -1.0})
    assert runner.status()["state"] == "idle"  # nothing was started

    # ... and "sigmoid" is not an XOR activation (choices: tanh/relu).
    with pytest.raises(ConfigError):
        runner.start(cfg_path, overrides={"phase_config": {"activation": "sigmoid"}})
    assert runner.status()["state"] == "idle"


# ---------------------------------------------------------------------------
# Loss curve figure
# ---------------------------------------------------------------------------


def test_loss_curve_fig_lines() -> None:
    import matplotlib.pyplot as plt

    # Mixed per-step and per-epoch records → train + val line.
    history = [
        {"step": 1, "epoch": 1, "loss": 0.7, "acc": 0.5},
        {"step": 1, "epoch": 1, "val_loss": 0.75, "val_acc": 0.5},
        {"step": 2, "epoch": 2, "loss": 0.5, "acc": 1.0},
        {"step": 2, "epoch": 2, "val_loss": 0.55, "val_acc": 1.0},
    ]
    fig = plots.loss_curve_fig(history)
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)

    # Empty history → no lines (save_loss_curve skips it entirely).
    fig = plots.loss_curve_fig([])
    assert len(fig.axes[0].lines) == 0
    plt.close(fig)
