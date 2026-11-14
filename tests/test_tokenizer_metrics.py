"""Tests for tokenizer quality metrics (T-204).

Acceptance criterion: compression / token-length / OOV metrics are logged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.tokenizer.bpe import BPETokenizer
from core.tokenizer.char import CharTokenizer
from core.tokenizer.common import TokenizerError
from core.tokenizer.metrics import (
    compare_tokenizers,
    format_metrics,
    log_metrics,
    tokenizer_metrics,
    write_metrics,
)

TEXT = "Die Straße war frei. " * 4


class FakeRun:
    """Minimal RunArtifacts stand-in for log_metrics."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def trained_pair():
    char = CharTokenizer.from_text(TEXT)
    bpe = BPETokenizer.train(TEXT, vocab_size=100, min_frequency=1)
    return char, bpe


# ---------------------------------------------------------------------------
# Single-tokenizer metrics
# ---------------------------------------------------------------------------


def test_character_metrics_are_one_token_per_character():
    tok = CharTokenizer.from_text("Hallo Welt")
    metrics = tokenizer_metrics(tok, "Hallo Welt")
    assert metrics["tokens"] == 10
    assert metrics["chars"] == 10
    assert metrics["words"] == 2
    assert metrics["compression_ratio"] == 1.0
    assert metrics["tokens_per_word"] == 5.0
    assert metrics["avg_token_chars"] == 1.0
    assert metrics["min_token_chars"] == 1
    assert metrics["max_token_chars"] == 1
    assert metrics["unk_tokens"] == 0
    assert metrics["unk_rate"] == 0.0
    assert metrics["oov_chars_total"] == 0
    assert metrics["roundtrip_ok"] is True


def test_bpe_metrics_compress():
    char, bpe = trained_pair()
    char_metrics = tokenizer_metrics(char, TEXT)
    bpe_metrics = tokenizer_metrics(bpe, TEXT)
    assert bpe_metrics["tokens"] < char_metrics["tokens"]
    assert bpe_metrics["compression_ratio"] > 1.5
    assert bpe_metrics["avg_token_chars"] > 1.0
    assert bpe_metrics["max_token_chars"] >= bpe_metrics["min_token_chars"]
    assert bpe_metrics["roundtrip_ok"] is True


def test_oov_metrics():
    tok = CharTokenizer.from_text("abc")
    metrics = tokenizer_metrics(tok, "ab😀")
    assert metrics["tokens"] == 3
    assert metrics["unk_tokens"] == 1
    assert metrics["unk_rate"] == pytest.approx(1 / 3)
    assert metrics["oov_chars_total"] == 1
    assert metrics["oov_char_types"] == 1
    assert metrics["oov_chars"] == ["😀"]
    assert metrics["roundtrip_ok"] is False


def test_empty_text_metrics():
    tok = CharTokenizer.from_text("abc")
    metrics = tokenizer_metrics(tok, "")
    assert metrics["tokens"] == 0
    assert metrics["compression_ratio"] == 0.0
    assert metrics["tokens_per_word"] == 0.0
    assert metrics["unique_token_ratio"] == 0.0
    assert metrics["roundtrip_ok"] is True


def test_metrics_require_text():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(TokenizerError, match="must be a str"):
        tokenizer_metrics(tok, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_comparison_reports_gain():
    char, bpe = trained_pair()
    comparison = compare_tokenizers(char, bpe, TEXT)
    assert comparison["text"]["chars"] == len(TEXT)
    assert comparison["char"]["tokens"] == len(TEXT)
    assert comparison["bpe"]["tokens"] < len(TEXT)
    assert comparison["compression_gain"] > 1.5
    assert comparison["token_reduction"] > 0.0
    assert comparison["both_roundtrip_ok"] is True


def test_comparison_with_empty_text_does_not_divide_by_zero():
    char, bpe = trained_pair()
    comparison = compare_tokenizers(char, bpe, "")
    assert comparison["compression_gain"] == 0.0
    assert comparison["token_reduction"] == 0.0


def test_format_metrics_is_human_readable():
    _, bpe = trained_pair()
    line = format_metrics("bpe", tokenizer_metrics(bpe, TEXT))
    assert line.startswith("bpe:")
    assert "chars/token" in line
    assert "tokens/word" in line
    assert "OOV" in line
    assert "round-trip=ok" in line


# ---------------------------------------------------------------------------
# Acceptance: metrics are written to the run
# ---------------------------------------------------------------------------


def test_log_metrics_writes_json_and_log(tmp_path: Path):
    char, bpe = trained_pair()
    run = FakeRun(tmp_path / "run")
    comparison = compare_tokenizers(char, bpe, TEXT)

    path = log_metrics(run, comparison)
    assert path == tmp_path / "run" / "metrics.json"
    assert path.is_file()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["char"]["tokens"] == len(TEXT)
    assert data["bpe"]["roundtrip_ok"] is True

    assert any(message.startswith("char:") for message in run.messages)
    assert any(message.startswith("bpe:") for message in run.messages)
    assert any("BPE compression:" in message for message in run.messages)
    assert any("metrics written:" in message for message in run.messages)


def test_log_metrics_warns_on_failed_roundtrip(tmp_path: Path):
    char = CharTokenizer.from_text("abc")
    bpe = BPETokenizer.train("abcabc", vocab_size=20, min_frequency=1)
    run = FakeRun(tmp_path / "run")
    log_metrics(run, compare_tokenizers(char, bpe, "ab😀"))
    assert any("WARNING" in message for message in run.messages)


def test_write_metrics_preserves_unicode(tmp_path: Path):
    path = write_metrics(tmp_path / "m.json", {"text": "Grüße"})
    assert "Grüße" in path.read_text(encoding="utf-8")
