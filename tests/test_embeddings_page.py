"""Tests for the embeddings explorer page (T-305).

Acceptance: a query token produces ranked neighbours, and the optional
scatter projection can be generated from the same page.  Artifacts are
written to a temp folder and discovered through a monkeypatched
``find_embedding_dirs`` so the tests do not depend on the repository's
``artifacts/`` state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from nicegui import ui
from nicegui.testing import User
import pytest

from apps.ui import embeddings_page as page_module
from core.embeddings.pipeline import write_embedding_artifacts

TOKENS = ["könig", "königin", "mann", "frau", "hund", "katze"]
VECTORS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.0, 0.0, 1.0],
        [0.1, 0.1, 0.9],
    ],
    dtype=np.float32,
)


def _element(user: User, marker: str) -> ui.element:
    return next(iter(user.find(marker=marker).elements))


def _make_run(tmp_path: Path, name: str = "run1") -> Path:
    return write_embedding_artifacts(
        tmp_path / "artifacts" / name / "embeddings",
        VECTORS,
        TOKENS,
        {"run_name": name, "embedding_dim": 3, "vocab_size": len(TOKENS)},
    )


def _patch_discovery(
    monkeypatch: pytest.MonkeyPatch, directories: list[Path]
) -> None:
    """Patch the page module the running route actually holds.

    The ``user`` fixture executes ``apps/ui/main.py`` with ``runpy``, which
    imports a fresh ``apps.ui.embeddings_page`` module.  The object imported
    at test-collection time can therefore be stale; patch both the imported
    reference and the current ``sys.modules`` entry.
    """
    modules = {page_module, sys.modules.get("apps.ui.embeddings_page")}
    for module in modules:
        if module is not None:
            monkeypatch.setattr(module, "find_embedding_dirs", lambda: directories)


@pytest.fixture
def one_run(tmp_path: Path) -> Path:
    return _make_run(tmp_path)


@pytest.mark.anyio
async def test_route_shows_query_and_neighbours(
    user: User,
    anyio_backend: str,
    one_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_discovery(monkeypatch, [one_run])
    await user.open("/embeddings")
    await user.should_see("Embeddings explorer")
    for marker in (
        "emb-source",
        "emb-query",
        "emb-top-k",
        "emb-query-btn",
        "emb-meta",
        "emb-neighbors-table",
        "emb-method",
        "emb-project",
    ):
        await user.should_see(marker=marker)

    # The page populates itself on first load.
    table = _element(user, "emb-neighbors-table")
    assert table.rows
    assert len(table.rows) == len(TOKENS) - 1
    assert [row["rank"] for row in table.rows] == list(range(1, len(TOKENS)))
    scores = [row["score"] for row in table.rows]
    assert scores == sorted(scores, reverse=True)
    await user.should_see(marker="emb-meta", content="6 tokens")
    await user.should_see(marker="emb-meta", content="3 dimensions")


@pytest.mark.anyio
async def test_clicking_query_updates_neighbours(
    user: User,
    anyio_backend: str,
    one_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_discovery(monkeypatch, [one_run])
    await user.open("/embeddings")
    query_el = _element(user, "emb-query")
    k_el = _element(user, "emb-top-k")
    query_el.value = "hund"
    k_el.value = 2
    user.find(marker="emb-query-btn").click()

    table = _element(user, "emb-neighbors-table")
    assert len(table.rows) == 2
    assert table.rows[0]["token"] == "katze"
    assert "hund" not in [row["token"] for row in table.rows]
    await user.should_see(marker="emb-status", content="neighbours of 'hund'")


@pytest.mark.anyio
async def test_project_button_generates_scatter_plot(
    user: User,
    anyio_backend: str,
    one_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_discovery(monkeypatch, [one_run])
    await user.open("/embeddings")
    user.find(marker="emb-project").click()

    await user.should_see(marker="emb-plot")
    image = _element(user, "emb-plot")
    assert image.source.startswith("data:image/png;base64,")
    plot = one_run.parent / "plots" / "embeddings_tsne_ui.png"
    assert plot.exists()
    assert plot.read_bytes().startswith(b"\x89PNG")
    await user.should_see(marker="emb-status", content="tsne projection saved")


@pytest.mark.anyio
async def test_no_artifacts_explains_prerequisite(
    user: User, anyio_backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_discovery(monkeypatch, [])
    await user.open("/embeddings")
    await user.should_see("No embedding runs found yet")
    await user.should_see("embeddings_tiny_debug.yaml")


def test_source_options_and_index_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _make_run(tmp_path, "run_a")
    second = _make_run(tmp_path, "run_b")
    monkeypatch.setattr(page_module, "find_embedding_dirs", lambda: [first, second])

    options = page_module._source_options()
    assert list(options) == [str(first), str(second)]
    assert "run_a" in options[str(first)]
    assert "run_b" in options[str(second)]

    index = page_module._cached_index(str(first))
    assert index.vocab_size == len(TOKENS)
    # Cache hit returns the same object; no re-read.
    assert page_module._cached_index(str(first)) is index


def test_default_token_prefers_a_word() -> None:
    assert page_module._default_token(["a", "der", "die"]) == "der"
    assert page_module._default_token([]) is None
    assert page_module._default_token(["a", "b"]) == "a"
