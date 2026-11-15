"""Runs and artifacts browser page (T-604).

``/runs`` answers the everyday question after a training run: *what did I
produce, and what is inside it?*

* recent runs are read from the artifact manager's ``runs.jsonl`` index
  (T-004) and from any run folder that has a ``run_meta.json``;
* selecting a run shows its status, phase/profile, timings and pipeline
  metrics when the run came from ``scripts.pipeline``;
* every file in the run folder is listed with its size.  Text, JSON,
  logs and small PNG/SVG plots can be previewed in place, while large
  checkpoints are labelled as binary rather than loaded.

Everything is local; the page never calls a network service.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from functools import partial
from pathlib import Path

from nicegui import ui

from core.artifacts import (
    DEFAULT_ARTIFACTS_ROOT,
    ArtifactEntry,
    ArtifactError,
    RunSummary,
    artifact_entries,
    artifact_preview,
    list_runs,
)

__all__ = ["register_runs_page", "runs_page", "artifacts_root"]

#: Overridable in tests (and useful for demos with a temporary output dir).
_ARTIFACTS_ROOT: Path | None = None

_MAX_IMAGE_BYTES = 2 * 1024 * 1024


def artifacts_root() -> Path:
    """Return the artifact root shown by the page.

    ``LLM_ARTIFACTS_ROOT`` lets a demo point the UI at a non-default
    output directory without editing code.
    """
    if _ARTIFACTS_ROOT is not None:
        return Path(_ARTIFACTS_ROOT)
    return Path(os.environ.get("LLM_ARTIFACTS_ROOT", DEFAULT_ARTIFACTS_ROOT))


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _started(summary: RunSummary) -> str:
    if not summary.start_time:
        return "—"
    return summary.start_time.replace("T", " ")


def _duration(summary: RunSummary) -> str:
    duration = summary.duration_s
    return "—" if duration is None else f"{duration:.1f}s"


def _status_color(status: str) -> str:
    return {
        "finished": "positive",
        "failed": "negative",
        "interrupted": "warning",
        "running": "info",
    }.get(status, "grey")


def _run_rows(runs: list[RunSummary]) -> list[dict]:
    return [
        {
            "id": str(run.run_dir),
            "run": run.run_name,
            "phase": run.phase or "—",
            "profile": run.profile or "—",
            "status": run.status,
            "started": _started(run),
            "duration": _duration(run),
        }
        for run in runs
    ]


def _pipeline_summary(run: RunSummary) -> dict | None:
    """Read ``pipeline.json`` (T-603) when present and valid."""
    path = run.run_dir / "pipeline.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _image_data_url(entry: ArtifactEntry) -> str | None:
    if entry.size > _MAX_IMAGE_BYTES:
        return None
    try:
        raw = entry.path.read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(entry.path.name)[0] or "image/png"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _run_detail(run: RunSummary) -> None:
    """Render the selected run's metadata, pipeline summary and files."""
    ui.label(run.run_name).classes("text-h6").mark("run-detail")
    with ui.row().classes("items-center gap-2"):
        ui.badge(run.status, color=_status_color(run.status))
        if run.phase:
            ui.badge(f"phase: {run.phase}", outline=True)
        if run.profile:
            ui.badge(f"profile: {run.profile}", outline=True)
        if run.amp is not None:
            ui.badge("AMP" if run.amp else "fp32", outline=True)

    with ui.grid(columns="repeat(auto-fit, minmax(220px, 1fr))").classes(
        "w-full gap-2 q-my-sm"
    ):
        ui.label(f"Started: {_started(run)}")
        ui.label(f"Duration: {_duration(run)}")
        ui.label(f"Device: {run.device or '—'}")
        ui.label(f"Folder: {run.run_dir}").classes("break-all")

    pipeline = _pipeline_summary(run)
    if pipeline:
        stages = pipeline.get("stages") or {}
        options = pipeline.get("options") or {}
        eval_stage = stages.get("eval") or {}
        parts = [
            f"pipeline: {options.get('preprocess_profile', '?')} → "
            f"tokenizer → GPT → eval"
        ]
        if eval_stage.get("val_loss") is not None:
            parts.append(f"val_loss {float(eval_stage['val_loss']):.4f}")
        if eval_stage.get("val_perplexity") is not None:
            parts.append(f"perplexity {float(eval_stage['val_perplexity']):.2f}")
        if stages.get("train", {}).get("steps") is not None:
            parts.append(f"{stages['train']['steps']} steps")
        ui.label(" · ".join(parts)).classes("text-body2").mark("run-pipeline")

    ui.separator()
    try:
        entries = artifact_entries(run.run_dir)
    except ArtifactError as exc:
        ui.label(f"cannot list artifacts: {exc}").classes("text-negative")
        return

    ui.label(f"Artifacts ({len(entries)})").classes("text-subtitle1")

    def show_preview(entry: ArtifactEntry) -> None:
        preview_area.clear()
        with preview_area:
            ui.label(entry.relative_path).classes("text-subtitle2 font-mono").mark(
                "artifact-path"
            )
            if entry.kind == "image":
                data_url = _image_data_url(entry)
                if data_url is None:
                    ui.label(
                        f"image too large to preview ({_human_size(entry.size)})"
                    ).classes("text-caption").mark("artifact-content")
                else:
                    ui.image(data_url).classes("max-w-full").mark("artifact-content")
                return
            preview = artifact_preview(entry)
            if preview.error:
                ui.label(f"cannot read: {preview.error}").classes(
                    "text-negative"
                ).mark("artifact-content")
                return
            if preview.text is None:
                ui.label(
                    f"binary file ({_human_size(entry.size)}); not previewed"
                ).classes("text-caption").mark("artifact-content")
                return
            language = "json" if entry.kind == "json" else "text"
            ui.code(preview.text, language=language).classes(
                "w-full max-h-96 overflow-auto"
            ).mark("artifact-content")
            if preview.truncated:
                ui.label("preview truncated").classes("text-caption")

    with ui.row().classes("w-full items-start gap-4"):
        with ui.card().classes("w-80 shrink-0"):
            ui.label("Files").classes("text-subtitle2")
            with ui.scroll_area().classes("h-96 w-full"):
                with ui.column().classes("w-full gap-0"):
                    for index, entry in enumerate(entries):
                        ui.button(
                            f"{entry.relative_path}  ({_human_size(entry.size)})",
                            on_click=partial(show_preview, entry),
                        ).props("flat dense align=left no-caps").classes(
                            "w-full justify-start font-mono text-xs"
                        ).mark(f"artifact-{index}")
        with ui.card().classes("grow min-w-0"):
            ui.label("Preview").classes("text-subtitle2")
            preview_area = ui.column().classes("w-full")
        if entries:
            show_preview(entries[0])
        else:
            with preview_area:
                ui.label("this run has no files").classes("text-caption")


def runs_page() -> None:
    """Build the ``/runs`` artifact browser."""
    # Imported lazily to avoid a circular import with apps/ui/main.py.
    from apps.ui.main import _shell

    root = artifacts_root()
    with _shell("/runs"):
        ui.label("Runs & artifacts").classes("text-h4")
        ui.label(
            "Browse recent training runs and inspect the files they produced: "
            "configs, logs, metrics, checkpoints, plots and pipeline reports."
        ).classes("text-subtitle1")
        ui.label(f"Artifact root: {root}").classes("text-caption break-all")
        ui.separator()

        @ui.refreshable
        def body() -> None:
            runs = list_runs(root)
            if not runs:
                ui.label("No runs found").classes("text-h6")
                ui.label(
                    "Start a run from the Jobs page, or run the end-to-end "
                    "pipeline; it appears here automatically."
                ).classes("q-my-sm")
                ui.code(
                    "python -m scripts.pipeline --config configs/gpt_tiny_debug.yaml"
                ).classes("w-full")
                return

            rows = _run_rows(runs)
            detail = ui.column().classes("w-full")

            def render(run_id: str) -> None:
                detail.clear()
                selected = next((run for run in runs if str(run.run_dir) == run_id), None)
                if selected is not None:
                    with detail:
                        _run_detail(selected)

            def selected(event) -> None:
                selection = list(getattr(event, "selection", None) or [])
                if selection:
                    render(str(selection[0].get("id")))

            table = (
                ui.table(
                    rows=rows,
                    columns=[
                        {"name": "run", "label": "Run", "field": "run", "align": "left"},
                        {"name": "phase", "label": "Phase", "field": "phase"},
                        {"name": "profile", "label": "Profile", "field": "profile"},
                        {"name": "status", "label": "Status", "field": "status"},
                        {"name": "started", "label": "Started", "field": "started"},
                        {"name": "duration", "label": "Duration", "field": "duration"},
                    ],
                    row_key="id",
                    selection="single",
                    on_select=selected,
                )
                .classes("w-full")
                .mark("runs-table")
            )
            table.selected = [rows[0]]
            render(rows[0]["id"])

        with ui.row().classes("items-center"):
            ui.button("Refresh", icon="refresh", on_click=body.refresh).mark(
                "runs-refresh"
            )
        body()


def register_runs_page() -> None:
    """Register ``/runs`` (called from ``main.register_pages``)."""

    @ui.page("/runs", title="Runs & artifacts – llm-from-scratch")
    def runs_page_route() -> None:
        runs_page()
