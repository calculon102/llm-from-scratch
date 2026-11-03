"""App shell tests (T-007) + jobs page tests (T-008).

Uses the NiceGUI headless user simulation (``user`` fixture from
``nicegui.testing.user_plugin``, loaded in ``conftest.py``).  The ``user``
fixture executes ``apps/ui/main.py`` (``main_file`` ini option) so all
routes are registered, then each test opens a route and checks the page
content.

The ``user`` fixture and the ``User`` methods are async; they are run by
the anyio pytest plugin (a hard dependency of NiceGUI).  Requesting
``anyio_backend`` is what makes the plugin wrap the async fixture.

The jobs tests register a toy XOR phase so a real (tiny) training job can
be started/stopped through the page; the ``job_runner`` singleton is
reset at the start of each test because it persists across the
per-test app re-runs.
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

from apps.ui.jobs import job_runner
from apps.ui.main import PHASE_PAGES
from core.config import validate_config
from core.trainer import PhaseBundle, register_phase

# (route, text that must be visible on the page)
ROUTES: list[tuple[str, str]] = [("/", "llm-from-scratch")] + [
    (phase.route, phase.title) for phase in PHASE_PAGES
]


@pytest.mark.anyio
@pytest.mark.parametrize("route,expected", ROUTES)
async def test_route_reachable(
    user: User, anyio_backend: str, route: str, expected: str
) -> None:
    await user.open(route)
    await user.should_see(expected)


@pytest.mark.anyio
@pytest.mark.parametrize("phase", PHASE_PAGES, ids=[p.route for p in PHASE_PAGES])
async def test_phase_page_lists_tickets(user: User, anyio_backend: str, phase) -> None:
    await user.open(phase.route)
    # Quasar renders table rows client-side, so check the row data on the
    # Python-side table element instead of looking for row text.
    table = next(iter(user.find(kind=ui.table).elements))
    assert [row["ticket"] for row in table.rows] == [
        ticket_id for ticket_id, _title, _priority in phase.tickets
    ]


@pytest.mark.anyio
async def test_home_links_to_all_phases(user: User, anyio_backend: str) -> None:
    await user.open("/")
    for phase in PHASE_PAGES:
        await user.should_see(phase.title)


@pytest.mark.anyio
async def test_shell_has_dark_mode_toggle(user: User, anyio_backend: str) -> None:
    await user.open("/")
    await user.should_see(kind=ui.switch)


@pytest.mark.anyio
async def test_header_has_menu_button(user: User, anyio_backend: str) -> None:
    # The menu button is the drawer's only opener on mobile (off-canvas).
    await user.open("/")
    await user.should_see(marker="app-menu-button")


# ---------------------------------------------------------------------------
# Jobs page (T-008)
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


def write_job_cfg(tmp_path: Path, **overrides) -> Path:
    cfg = validate_config(
        {
            "run_name": "ui_job",
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
    path = tmp_path / "ui_job.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


async def wait_for_state(state: set[str], timeout: float = 60.0) -> dict:
    """Poll the app-wide job runner until it reaches one of ``state``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = job_runner.status()
        if status["state"] in state:
            return status
        await asyncio.sleep(0.05)
    raise AssertionError(f"job did not reach {state}; state={job_runner.status()}")


async def select_config(user: User, config_path: Path) -> None:
    """Point the jobs page config select at ``config_path``.

    The select only offers configs from ``configs/``; the test's temp
    config lives elsewhere, so register it as an option first -- a
    ``ui.select`` resets a value that is not one of its options to
    ``None``.
    """
    select = next(iter(user.find(kind=ui.select).elements))
    if str(config_path) not in select.options:
        select.options = list(select.options) + [str(config_path)]
    select.value = str(config_path)


@pytest.mark.anyio
async def test_jobs_route_reachable(user: User, anyio_backend: str) -> None:
    job_runner.reset()
    await user.open("/jobs")
    await user.should_see("Jobs")
    await user.should_see(marker="job-start")
    await user.should_see(marker="job-stop")
    await user.should_see(marker="job-status")
    await user.should_see(marker="job-log")
    await user.should_see(marker="job-run-dir")


@pytest.mark.anyio
async def test_start_job_from_ui(user: User, anyio_backend: str, tmp_path: Path) -> None:
    job_runner.reset()
    cfg_path = write_job_cfg(tmp_path)

    await user.open("/jobs")
    await user.should_see(marker="job-status", content="idle")
    await select_config(user, cfg_path)
    user.find(marker="job-start").click()

    # The job runs in a background thread; wait for it to finish.
    status = await wait_for_state({"finished", "failed"})
    assert status["state"] == "finished", status["error"]
    assert status["step"] == 10

    # The page timer updates the status label and streams the run log.
    await user.should_see(marker="job-status", content="finished", retries=100)
    await user.should_see("training finished", retries=100)
    await user.should_see(marker="job-run-dir", content="run:", retries=100)


@pytest.mark.anyio
async def test_stop_job_from_ui(user: User, anyio_backend: str, tmp_path: Path) -> None:
    job_runner.reset()
    cfg_path = write_job_cfg(tmp_path, epochs=10000)

    await user.open("/jobs")
    await select_config(user, cfg_path)
    user.find(marker="job-start").click()

    # Wait until the job is running and the timer has enabled Stop.
    stop_btn = next(iter(user.find(marker="job-stop").elements))
    deadline = time.time() + 30
    while time.time() < deadline:
        if job_runner.status()["state"] == "running" and stop_btn.enabled:
            break
        await asyncio.sleep(0.05)
    assert job_runner.status()["state"] == "running" and stop_btn.enabled

    user.find(marker="job-stop").click()
    status = await wait_for_state({"stopped", "failed"})
    assert status["state"] == "stopped", status["error"]
    assert status["step"] is not None and status["step"] < 10000

    await user.should_see(marker="job-status", content="stopped", retries=100)
