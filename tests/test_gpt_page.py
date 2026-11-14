"""Tests for the Mini GPT generation page (T-406).

Acceptance: interactive generation works from a trained artifact.  The
small model and tokenizer are written to a temp folder and discovered
through a monkeypatched ``find_gpt_dirs`` so the tests do not depend on the
repository's ``artifacts/`` state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from nicegui import ui
from nicegui.testing import User

from apps.ui import gpt_page as page_module
from core.gpt.model import MiniGPT
from core.gpt.pipeline import write_gpt_artifacts
from core.tokenizer import save_bundle, train_bundle

TRAIN_TEXT = "der die das ein eine und oder mit von zu den dem des " * 30


def _element(user: User, marker: str) -> ui.element:
    return next(iter(user.find(marker=marker).elements))


def _make_run(tmp_path: Path, name: str = "gpt_run") -> Path:
    bundle = train_bundle(
        TRAIN_TEXT, vocab_size=60, min_frequency=1, max_token_length=4
    )
    model = MiniGPT(
        vocab_size=bundle.bpe.vocab_size,
        context_length=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        tie_weights=False,
    )
    directory = write_gpt_artifacts(
        tmp_path / "artifacts" / name / "gpt",
        model,
        {
            "run_name": name,
            "best_val": 1.2345,
            "val_perplexity": 3.42,
        },
    )
    save_bundle(bundle, directory.parent / "tokenizers")
    return directory


def _patch_discovery(
    monkeypatch: pytest.MonkeyPatch, directories: list[Path]
) -> None:
    """Patch the page module the running route actually holds.

    The ``user`` fixture executes ``apps/ui/main.py`` with ``runpy``, which
    imports a fresh ``apps.ui.gpt_page`` module.  The object imported at
    test-collection time can therefore be stale; patch both the imported
    reference and the current ``sys.modules`` entry.
    """
    modules = {page_module, sys.modules.get("apps.ui.gpt_page")}
    for module in modules:
        if module is not None:
            monkeypatch.setattr(module, "find_gpt_dirs", lambda: directories)


@pytest.fixture
def one_run(tmp_path: Path) -> Path:
    return _make_run(tmp_path)


@pytest.mark.anyio
async def test_route_shows_generation_controls(
    user: User,
    anyio_backend: str,
    one_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_discovery(monkeypatch, [one_run])
    await user.open("/gpt")
    await user.should_see("Mini GPT generation")
    for marker in (
        "gpt-source",
        "gpt-meta",
        "gpt-prompt",
        "gpt-max-new-tokens",
        "gpt-temperature",
        "gpt-top-k",
        "gpt-top-p",
        "gpt-seed",
        "gpt-greedy",
        "gpt-generate",
        "gpt-status",
        "gpt-output",
    ):
        await user.should_see(marker=marker)
    await user.should_see(marker="gpt-meta", content="context 8")
    await user.should_see(marker="gpt-meta", content="perplexity 3.42")


@pytest.mark.anyio
async def test_greedy_generation_is_interactive(
    user: User,
    anyio_backend: str,
    one_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_discovery(monkeypatch, [one_run])
    await user.open("/gpt")

    prompt_el = _element(user, "gpt-prompt")
    max_tokens_el = _element(user, "gpt-max-new-tokens")
    greedy_el = _element(user, "gpt-greedy")
    prompt_el.value = "der"
    max_tokens_el.value = 5
    greedy_el.value = True
    user.find(marker="gpt-generate").click()

    output = _element(user, "gpt-output")
    assert output.text.startswith("der")
    await user.should_see(marker="gpt-status", content="generated")
    await user.should_see(marker="gpt-status", content="(greedy)")


@pytest.mark.anyio
async def test_sampling_controls_are_used(
    user: User,
    anyio_backend: str,
    one_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_discovery(monkeypatch, [one_run])
    await user.open("/gpt")

    _element(user, "gpt-prompt").value = "die"
    _element(user, "gpt-max-new-tokens").value = 4
    _element(user, "gpt-temperature").value = 0.7
    _element(user, "gpt-top-k").value = 3
    _element(user, "gpt-top-p").value = 0.9
    _element(user, "gpt-seed").value = 7
    user.find(marker="gpt-generate").click()

    output = _element(user, "gpt-output")
    assert output.text.startswith("die")
    await user.should_see(marker="gpt-status", content="(sampled)")


@pytest.mark.anyio
async def test_no_artifacts_explains_prerequisite(
    user: User, anyio_backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_discovery(monkeypatch, [])
    await user.open("/gpt")
    await user.should_see("No GPT runs found yet")
    await user.should_see("gpt_tiny_debug.yaml")


def test_source_options_and_artifact_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _make_run(tmp_path, "run_a")
    second = _make_run(tmp_path, "run_b")
    monkeypatch.setattr(page_module, "find_gpt_dirs", lambda: [first, second])

    options = page_module._source_options()
    assert list(options) == [str(first), str(second)]
    assert "run_a" in options[str(first)]
    assert "run_b" in options[str(second)]

    artifacts = page_module._cached_artifacts(str(first))
    assert artifacts.vocab_size > 0
    # Cache hit returns the same object; no re-read.
    assert page_module._cached_artifacts(str(first)) is artifacts
    assert artifacts.tokenizer is not None
