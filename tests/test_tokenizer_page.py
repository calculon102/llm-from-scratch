"""Tests for the tokenizer compare page (T-205).

The acceptance criterion is "side-by-side tokenizer view works": both
tokenizers must be visible at once, show their own token tables, and
re-tokenize when the text changes.  The page builds and trains the sample
bundle on first load, so these tests also exercise the offline path a
fresh clone takes (no artifact folder required).
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui
from nicegui.testing import User
import pytest

from apps.ui import tokenizer_page as page_module
from core.tokenizer import save_bundle, train_bundle


def _element(user: User, marker: str) -> ui.element:
    return next(iter(user.find(marker=marker).elements))


@pytest.mark.anyio
async def test_tokenizer_route_reachable_and_split_view(
    user: User, anyio_backend: str
) -> None:
    await user.open("/tokenizer")
    await user.should_see("Tokenizer compare")
    for marker in (
        "tok-source",
        "tok-vocab-size",
        "tok-text",
        "tok-compare",
        "tok-status",
        "tok-char-metrics",
        "tok-bpe-metrics",
        "tok-char-roundtrip",
        "tok-bpe-roundtrip",
        "tok-char-table",
        "tok-bpe-table",
        "tok-char-count",
        "tok-bpe-count",
    ):
        await user.should_see(marker=marker)

    char_table = _element(user, "tok-char-table")
    bpe_table = _element(user, "tok-bpe-table")
    # The page populates itself on first build; both sides must have real
    # token rows, and BPE must be the shorter sequence on the German sample
    # text (the whole point of the comparison).
    assert char_table.rows
    assert bpe_table.rows
    assert len(bpe_table.rows) < len(char_table.rows)

    await user.should_see(marker="tok-char-roundtrip", content="round-trip ok")
    await user.should_see(marker="tok-bpe-roundtrip", content="round-trip ok")


@pytest.mark.anyio
async def test_tokenize_click_updates_both_tables(
    user: User, anyio_backend: str
) -> None:
    await user.open("/tokenizer")
    text_el = _element(user, "tok-text")
    text_el.value = "Hallo Welt"
    user.find(marker="tok-compare").click()

    char_table = _element(user, "tok-char-table")
    bpe_table = _element(user, "tok-bpe-table")
    # 10 characters: each character is one token on the char side.
    assert len(char_table.rows) == 10
    assert len(bpe_table.rows) <= len(char_table.rows)
    await user.should_see(marker="tok-status", content="10 characters")
    await user.should_see(marker="tok-char-count", content="showing 10 of 10 tokens")


@pytest.mark.anyio
async def test_whitespace_is_visible_in_token_tables(
    user: User, anyio_backend: str
) -> None:
    await user.open("/tokenizer")
    text_el = _element(user, "tok-text")
    text_el.value = "a b\nc"
    user.find(marker="tok-compare").click()

    char_table = _element(user, "tok-char-table")
    tokens = [row["token"] for row in char_table.rows]
    # A space and a newline are separate character tokens; the table must
    # escape them instead of rendering invisible cells.
    assert "␣" in tokens
    assert "\\n" in tokens


def test_cached_sample_bundle_respects_vocab_size() -> None:
    small = page_module._cached_bundle(page_module.SAMPLE_SOURCE, 120)
    large = page_module._cached_bundle(page_module.SAMPLE_SOURCE, 300)
    # The character vocabulary is fixed by the corpus alphabet; only BPE
    # grows with the requested vocab size.
    assert small.char.vocab_size == large.char.vocab_size
    assert large.bpe.vocab_size > small.bpe.vocab_size
    # Cache hit returns the exact same object, not a retrain.
    assert page_module._cached_bundle(page_module.SAMPLE_SOURCE, 300) is large


def test_source_options_include_and_load_artifact_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = train_bundle(
        "Hallo Welt. Grüße aus Köln. " * 5, vocab_size=60, source="unit-test"
    )
    bundle_dir = save_bundle(bundle, tmp_path / "artifacts" / "run1" / "tokenizers")
    monkeypatch.setattr(page_module, "find_bundle_dirs", lambda: [bundle_dir])

    options = page_module._source_options()
    assert options[page_module.SAMPLE_SOURCE].startswith("Bundled sample")
    assert options[str(bundle_dir)] == "artifact: run1"

    loaded = page_module._cached_bundle(str(bundle_dir), 9999)
    assert loaded.bpe.vocab == bundle.bpe.vocab
    assert loaded.bpe.merges == bundle.bpe.merges
    # Artifact vocab size is fixed on disk; the UI control is ignored.
    assert loaded.bpe.vocab_size == bundle.bpe.vocab_size
