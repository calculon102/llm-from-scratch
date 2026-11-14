"""Embeddings explorer page (T-305): query tokens, neighbours and a 2-D map.

The page is the visual front-end for the M3 stack:

* a run selector discovers every ``<run>/embeddings`` artifact written by
  T-302 (:func:`core.embeddings.pipeline.find_embedding_dirs`);
* a searchable token selector and top-k control call
  :class:`core.embeddings.neighbors.EmbeddingIndex` — the same cosine
  search the CLI uses (T-303), so the page and the command line agree;
* an optional **Project** button runs t-SNE, UMAP or PCA through
  :func:`core.embeddings.projection.project_embeddings` (T-304), saves the
  PNG into the run's ``plots/`` folder and shows it inline.

Everything is local: no API calls, no network, and no model download.  If
no embedding run exists yet, the page explains how to create one instead of
showing an empty explorer.

The projection is computed synchronously on the UI callback.  That is fine
for the small vocabularies in this repository (hundreds of tokens, well
under a second for t-SNE); a larger corpus should move it to the job
runner, which is deliberately out of scope for this page.
"""

from __future__ import annotations

import base64
from pathlib import Path

from nicegui import ui

from core.embeddings.neighbors import EmbeddingIndex
from core.embeddings.pipeline import EmbeddingsError, find_embedding_dirs
from core.embeddings.projection import project_embeddings, save_projection_plot

__all__ = ["register_embeddings_page", "embeddings_page"]

#: How many neighbours the table shows at once.
MAX_TOP_K = 50

#: Loaded indexes are cached by path + mtime.  A newly trained run has a new
#: directory, and re-training into the same folder invalidates the mtime.
_INDEX_CACHE: dict[tuple[str, int], EmbeddingIndex] = {}


def _source_options() -> dict[str, str]:
    """Return ``{embeddings_dir: label}`` for all discovered runs."""
    options: dict[str, str] = {}
    for embeddings_dir in find_embedding_dirs():
        run_name = embeddings_dir.parent.name
        label = f"{run_name}  ({embeddings_dir})"
        if label in options.values():
            label = f"{label} [{len(options)}]"
        options[str(embeddings_dir)] = label
    return options


def _cached_index(source: str) -> EmbeddingIndex:
    path = Path(source)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    key = (str(path), mtime)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = EmbeddingIndex.load(path)
    return _INDEX_CACHE[key]


def _default_token(tokens: list[str]) -> str | None:
    """Prefer a word-like token over a single character for the first query."""
    for token in tokens:
        if len(token.strip()) >= 3:
            return token
    return tokens[0] if tokens else None


def _neighbor_rows(index: EmbeddingIndex, token: str, k: int) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "token": neighbor.token,
            "score": round(neighbor.score, 4),
            "id": neighbor.index,
        }
        for rank, neighbor in enumerate(index.neighbors(token, k=k), start=1)
    ]


def _plot_data_url(index: EmbeddingIndex, source: str, method: str) -> tuple[str, Path]:
    """Project, save the PNG into the run's plots folder, return a data URL."""
    embeddings_dir = Path(source)
    run_dir = embeddings_dir.parent if embeddings_dir.name == "embeddings" else embeddings_dir
    projection = project_embeddings(index, method=method)
    plot_path = save_projection_plot(
        projection,
        run_dir / "plots" / f"embeddings_{projection.method}_ui.png",
        title=f"Embedding projection — {run_dir.name}",
    )
    encoded = base64.b64encode(plot_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}", plot_path


def _no_artifacts_message() -> None:
    """Explain the missing prerequisite instead of rendering an empty page."""
    ui.label("No embedding runs found yet").classes("text-h6 q-mt-md")
    ui.label(
        "Train a skip-gram run first (the Jobs page can start any config), "
        "then come back; the run appears automatically."
    ).classes("q-my-sm")
    ui.code(
        "python -m scripts.train --config configs/embeddings_tiny_debug.yaml"
    ).classes("w-full")


def embeddings_page() -> None:
    """Build the ``/embeddings`` explorer page."""
    # Imported lazily: apps/ui/main.py imports this module inside
    # register_pages(), so a module-level import would be circular.
    from apps.ui.main import _shell

    options = _source_options()
    with _shell("/embeddings"):
        ui.label("Embeddings explorer").classes("text-h4")
        ui.label(
            "Skip-gram vectors learned from the German sample corpus. "
            "Pick a run, choose a token, and inspect its cosine neighbours; "
            "optionally project the whole vocabulary to 2-D."
        ).classes("text-subtitle1")
        ui.separator()

        if not options:
            _no_artifacts_message()
            return

        source_values = list(options)
        with ui.row().classes("w-full items-end gap-4 flex-wrap"):
            source_el = (
                ui.select(
                    options=options,
                    value=source_values[0],
                    label="Embedding run",
                )
                .classes("w-[34rem] max-w-full")
                .mark("emb-source")
            )
            query_el = (
                ui.select(
                    options=[],
                    value=None,
                    with_input=True,
                    label="Query token",
                )
                .classes("w-64")
                .mark("emb-query")
            )
            top_k_el = (
                ui.number("top-k", value=10, min=1, max=MAX_TOP_K, step=1)
                .classes("w-28")
                .mark("emb-top-k")
            )
            query_btn = (
                ui.button("Neighbours", icon="travel_explore")
                .props("color=primary")
                .mark("emb-query-btn")
            )

        meta_label = ui.label("").classes("text-caption").mark("emb-meta")
        status_label = ui.label("").classes("text-body2").mark("emb-status")

        neighbor_table = ui.table(
            rows=[],
            columns=[
                {"name": "rank", "label": "#", "field": "rank", "align": "right"},
                {"name": "token", "label": "token", "field": "token", "align": "left"},
                {"name": "score", "label": "cosine", "field": "score", "align": "right"},
                {"name": "id", "label": "id", "field": "id", "align": "right"},
            ],
            row_key="rank",
        ).classes("w-full max-w-3xl").mark("emb-neighbors-table")

        ui.separator()
        ui.label("2-D map").classes("text-h6")
        ui.label(
            "t-SNE and UMAP reveal local clusters; PCA is the linear "
            "baseline. The plot is saved into the run's plots/ folder."
        ).classes("text-caption")
        with ui.row().classes("items-end gap-4 flex-wrap q-my-sm"):
            method_el = (
                ui.select(
                    options={
                        "tsne": "t-SNE (local clusters)",
                        "umap": "UMAP (local + global)",
                        "pca": "PCA (linear baseline)",
                    },
                    value="tsne",
                    label="Projection",
                )
                .classes("w-64")
                .mark("emb-method")
            )
            project_btn = ui.button("Project", icon="scatter_plot").mark("emb-project")

        plot_box = ui.column().classes("w-full max-w-4xl")
        with plot_box:
            ui.label("Press Project to generate a scatter plot.").classes(
                "text-caption"
            ).mark("emb-plot-placeholder")

        project_btn.set_enabled(False)

        def load_source(value: str | None) -> None:
            source = value or source_values[0]
            try:
                index = _cached_index(source)
            except (EmbeddingsError, OSError, ValueError) as exc:
                meta_label.text = f"error: {exc}"
                ui.notify(str(exc), type="negative")
                project_btn.set_enabled(False)
                return

            tokens = index.tokens
            query_el.options = tokens
            query_el.value = _default_token(tokens)
            query_el.update()
            meta_label.text = (
                f"{index.vocab_size} tokens · {index.embedding_dim} dimensions · "
                f"run: {Path(source).parent.name}"
            )
            neighbor_table.rows = []
            neighbor_table.update()
            status_label.text = ""
            plot_box.clear()
            with plot_box:
                ui.label(
                    "Press Project to generate a scatter plot."
                ).classes("text-caption").mark("emb-plot-placeholder")
            project_btn.set_enabled(True)

        def find_neighbors() -> None:
            token = query_el.value
            if not token:
                status_label.text = "choose a query token first"
                return
            source = source_el.value or source_values[0]
            try:
                index = _cached_index(source)
                k = int(top_k_el.value or 10)
                rows = _neighbor_rows(index, token, k)
            except (KeyError, EmbeddingsError, OSError, ValueError) as exc:
                status_label.text = f"error: {exc}"
                ui.notify(str(exc), type="negative")
                return
            if not rows:
                status_label.text = "no neighbours (vocabulary is too small)"
                return
            neighbor_table.rows = rows
            neighbor_table.update()
            status_label.text = (
                f"top {len(rows)} cosine neighbours of {token!r} "
                f"({index.vocab_size} tokens, {index.embedding_dim}-d)"
            )

        def project() -> None:
            source = source_el.value or source_values[0]
            method = method_el.value or "tsne"
            project_btn.set_enabled(False)
            status_label.text = f"projecting with {method}…"
            try:
                index = _cached_index(source)
                data_url, plot_path = _plot_data_url(index, source, method)
            except (EmbeddingsError, OSError, ValueError) as exc:
                status_label.text = f"projection error: {exc}"
                ui.notify(str(exc), type="negative")
                project_btn.set_enabled(True)
                return
            plot_box.clear()
            with plot_box:
                ui.image(data_url).classes("w-full").mark("emb-plot")
            status_label.text = f"{method} projection saved to {plot_path}"
            project_btn.set_enabled(True)

        query_btn.on_click(find_neighbors)
        project_btn.on_click(project)
        source_el.on_value_change(lambda event: load_source(event.value))
        load_source(source_el.value)
        # Show a first result immediately; the default token comes from the
        # selected run, so the page is useful before the user clicks anything.
        find_neighbors()


def register_embeddings_page() -> None:
    """Register the ``/embeddings`` route (called from ``main.register_pages``).

    Registration happens from ``register_pages`` rather than at module
    import because the headless UI tests reset the app and re-execute
    ``main.py`` per test while this module stays cached in ``sys.modules``.
    """

    @ui.page("/embeddings", title="Embeddings – llm-from-scratch")
    def embeddings_page_route() -> None:
        embeddings_page()
