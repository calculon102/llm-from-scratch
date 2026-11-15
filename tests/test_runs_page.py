"""Tests for the runs/artifacts browser page (T-604).

Acceptance criterion: "recent runs and artifacts browsable in UI".  The
tests point ``LLM_ARTIFACTS_ROOT`` at a temporary output folder, create
real runs through the artifact manager, open ``/runs`` headlessly and
check the run table, run detail, pipeline summary, artifact list and an
in-place text preview.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nicegui import ui
from nicegui.testing import User

from core.artifacts import artifact_entries, create_run
from core.config import validate_config


def _element(user: User, marker: str) -> ui.element:
    return next(iter(user.find(marker=marker).elements))


def _make_run(tmp_path: Path, name: str = "demo_run") -> Path:
    cfg = validate_config(
        {
            "run_name": name,
            "output_dir": str(tmp_path / "artifacts"),
            "phase": "gpt",
            "profile": "tiny_debug",
            "seed": 42,
        }
    )
    run = create_run(cfg)
    run.log(f"{name} started")
    (run.run_dir / "notes.txt").write_text(
        "hello from the artifact browser\n", encoding="utf-8"
    )
    gpt_dir = run.run_dir / "gpt"
    gpt_dir.mkdir()
    (gpt_dir / "meta.json").write_text(
        json.dumps({"tokens": 123, "parameters": 456}), encoding="utf-8"
    )
    (run.run_dir / "pipeline.json").write_text(
        json.dumps(
            {
                "format": "llm-from-scratch-pipeline",
                "status": "finished",
                "options": {"preprocess_profile": "moderate_clean"},
                "stages": {
                    "train": {"steps": 40},
                    "eval": {"val_loss": 6.9, "val_perplexity": 995.98},
                },
            }
        ),
        encoding="utf-8",
    )
    run.mark_finished()
    return run.run_dir


@pytest.fixture
def artifacts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setenv("LLM_ARTIFACTS_ROOT", str(root))
    return root


@pytest.mark.anyio
async def test_runs_page_lists_and_previews_artifacts(
    user: User,
    anyio_backend: str,
    artifacts_root: Path,
) -> None:
    run_dir = _make_run(tmp_path=artifacts_root.parent)

    await user.open("/runs")
    await user.should_see("Runs & artifacts")
    await user.should_see(marker="runs-table")
    await user.should_see(marker="run-detail", content="demo_run")
    await user.should_see(marker="run-pipeline", content="995.98")
    await user.should_see(marker="run-pipeline", content="40 steps")

    table = next(iter(user.find(marker="runs-table").elements))
    assert [row["run"] for row in table.rows] == ["demo_run"]
    assert table.rows[0]["status"] == "finished"

    entries = artifact_entries(run_dir)
    notes_index = next(
        index
        for index, entry in enumerate(entries)
        if entry.relative_path == "notes.txt"
    )

    user.find(marker=f"artifact-{notes_index}").click()
    await user.should_see(marker="artifact-path", content="notes.txt")
    preview = _element(user, "artifact-content")
    assert "hello from the artifact browser" in preview.content


@pytest.mark.anyio
async def test_runs_page_empty_state_points_at_pipeline(
    user: User,
    anyio_backend: str,
    artifacts_root: Path,
) -> None:
    await user.open("/runs")
    await user.should_see("No runs found")
    await user.should_see("scripts.pipeline")


@pytest.mark.anyio
async def test_runs_page_handles_multiple_runs(
    user: User,
    anyio_backend: str,
    artifacts_root: Path,
) -> None:
    first = _make_run(tmp_path=artifacts_root.parent, name="first_run")
    second = _make_run(tmp_path=artifacts_root.parent, name="second_run")

    await user.open("/runs")
    table = next(iter(user.find(marker="runs-table").elements))
    assert {row["run"] for row in table.rows} == {"first_run", "second_run"}
    assert first.name != second.name
    # The newest indexed run is selected and rendered by default.
    await user.should_see(marker="run-detail", content="second_run")
