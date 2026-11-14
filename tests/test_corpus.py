"""Tests for noisy corpus ingestion (T-201)."""

from __future__ import annotations

import json

import pytest

from core.tokenizer.corpus import (
    CorpusOptions,
    clean_text,
    decode_bytes,
    ingest_corpus,
    iter_corpus_files,
    looks_binary,
)
from core.tokenizer.common import TokenizerError
from core.tokenizer.pipeline import DEFAULT_SAMPLE_CORPUS


# ---------------------------------------------------------------------------
# Decoding and binary sniffing
# ---------------------------------------------------------------------------


def test_decode_prefers_utf8_and_strips_bom():
    result = decode_bytes(b"\xef\xbb\xbfGr\xc3\xbc\xc3\x9fe")
    assert result.text == "Grüße"
    assert result.encoding == "utf-8-sig"
    assert result.replaced_chars == 0
    assert result.fell_back is False


def test_decode_falls_back_to_cp1252():
    data = "Grüße aus Köln".encode("cp1252")
    result = decode_bytes(data)
    assert result.text == "Grüße aus Köln"
    assert result.encoding == "cp1252"


def test_decode_final_fallback_replaces_bad_bytes():
    result = decode_bytes(b"abc\xff", encodings=("utf-8",))
    assert result.encoding == "utf-8+replace"
    assert result.replaced_chars == 1
    assert result.fell_back is True
    assert "\ufffd" in result.text


def test_decode_rejects_empty_encoding_list():
    with pytest.raises(TokenizerError, match="encodings"):
        decode_bytes(b"abc", encodings=())


def test_looks_binary():
    assert looks_binary(b"normaler Text\n") is False
    assert looks_binary(b"abc\x00def") is True
    assert looks_binary(b"\x01\x02\x03\x04") is True
    assert looks_binary(b"") is False


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def test_clean_normalizes_nfc_crlf_controls_and_spaces():
    text = "  A\x00B\u200bC\u00a0D  \r\nCafe\u0301\r\n"
    cleaned, stats = clean_text(text)
    assert cleaned == "A B C D\nCafé\n"
    assert stats.controls_removed >= 3
    assert stats.lines_out == 2


def test_clean_drops_blank_short_and_duplicate_lines():
    cleaned, stats = clean_text("Hallo\r\nHallo\r\nA\r\n\r\nWelt\r\n")
    assert cleaned == "Hallo\nWelt\n"
    assert stats.duplicate_dropped == 1
    assert stats.short_dropped == 1
    assert stats.blank_dropped >= 1


def test_clean_truncates_long_lines():
    cleaned, stats = clean_text(
        "abcdefgh\n", CorpusOptions(min_line_chars=1, max_line_chars=5)
    )
    assert cleaned == "abcde\n"
    assert stats.truncated_lines == 1


def test_clean_zero_max_line_chars_means_no_limit():
    cleaned, stats = clean_text(
        "abcdefgh\n", CorpusOptions(min_line_chars=1, max_line_chars=0)
    )
    assert cleaned == "abcdefgh\n"
    assert stats.truncated_lines == 0


def test_clean_drops_replacement_char_junk():
    cleaned, stats = clean_text("ok\n" + "\ufffd" * 10 + "\n")
    assert cleaned == "ok\n"
    assert stats.junk_dropped == 1


def test_clean_alpha_ratio_filter():
    opts = CorpusOptions(min_alpha_ratio=0.5)
    cleaned, stats = clean_text("123456\nabc123\n", opts)
    assert cleaned == "abc123\n"
    assert stats.low_alpha_dropped == 1


def test_clean_lowercase_option_is_lossy_by_design():
    cleaned, _ = clean_text("ÄBC\n", CorpusOptions(lowercase=True))
    assert cleaned == "äbc\n"


def test_clean_empty_text():
    cleaned, stats = clean_text("")
    assert cleaned == ""
    assert stats.lines_out == 0


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def test_iter_corpus_files_is_recursive_and_deterministic(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("m", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("h", encoding="utf-8")
    (tmp_path / "sub" / "c.txt").write_text("c", encoding="utf-8")
    (tmp_path / "sub" / ".hidden2.txt").write_text("h", encoding="utf-8")

    found = [path.name for path in iter_corpus_files(tmp_path)]
    assert found == ["a.txt", "b.txt", "c.txt"]


def test_iter_corpus_files_accepts_explicit_file_of_any_suffix(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("notiz", encoding="utf-8")
    assert list(iter_corpus_files(path)) == [path]


def test_iter_corpus_files_missing_path_raises(tmp_path):
    with pytest.raises(TokenizerError, match="not found"):
        list(iter_corpus_files(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_ingest_sample_corpus_reports_noise():
    corpus = ingest_corpus(DEFAULT_SAMPLE_CORPUS)
    assert corpus.stats["files_seen"] == 4
    assert corpus.stats["files_ok"] == 4
    assert corpus.stats["encodings"] == {"cp1252": 1, "utf-8-sig": 3}
    assert corpus.stats["duplicate_dropped"] == 1
    assert "\r" not in corpus.text
    assert "Grüße" in corpus.text
    assert "Straße" in corpus.text
    # UTF-8 BOM was stripped; the text starts with real content.
    assert corpus.text.startswith("Grüße aus dem Archiv")
    assert "Ein Tokenizer ist kein Wörterbuch!" in corpus.text
    assert corpus.stats["chars_out"] == len(corpus.text)


def test_ingest_reports_each_file():
    corpus = ingest_corpus(DEFAULT_SAMPLE_CORPUS)
    by_name = {report.path.split("/")[-1]: report for report in corpus.files}
    assert by_name["archiv_cp1252.txt"].encoding == "cp1252"
    assert by_name["rechtschreibung_bom.txt"].encoding == "utf-8-sig"
    assert all(report.status == "ok" for report in corpus.files)
    assert all(report.chars_out > 0 for report in corpus.files)


def test_ingest_dedupes_across_files(tmp_path):
    (tmp_path / "a.txt").write_text("Hallo Welt\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Hallo Welt\n", encoding="utf-8")
    corpus = ingest_corpus(tmp_path)
    assert corpus.text == "Hallo Welt\n"
    assert corpus.stats["duplicate_dropped"] == 1


def test_ingest_skips_binary_and_empty_files(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"\x00\x01\x02")
    (tmp_path / "empty.txt").write_bytes(b"")
    (tmp_path / "good.txt").write_text("Guter Text\n", encoding="utf-8")

    corpus = ingest_corpus(tmp_path)
    statuses = {report.path.split("/")[-1]: report for report in corpus.files}
    assert statuses["bad.txt"].status == "skipped"
    assert statuses["bad.txt"].reason == "binary content"
    assert statuses["empty.txt"].reason == "empty file"
    assert statuses["good.txt"].status == "ok"
    assert corpus.stats["files_ok"] == 1
    assert corpus.stats["files_skipped"] == 2


def test_ingest_raises_when_nothing_usable(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"\x00\x01\x02")
    with pytest.raises(TokenizerError, match="no usable corpus text"):
        ingest_corpus(tmp_path)


def test_corpus_report_is_json_serializable(tmp_path):
    corpus = ingest_corpus(DEFAULT_SAMPLE_CORPUS)
    path = corpus.write_report(tmp_path / "report.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"stats", "files"}
    assert data["stats"]["files_ok"] == 4
    assert len(data["files"]) == 4
    assert data["files"][0]["status"] == "ok"


def test_corpus_options_validation():
    with pytest.raises(TokenizerError, match="min_line_chars"):
        CorpusOptions(min_line_chars=-1)
    with pytest.raises(TokenizerError, match="min_alpha_ratio"):
        CorpusOptions(min_alpha_ratio=1.5)
    with pytest.raises(TokenizerError, match="encodings"):
        CorpusOptions(encodings=())
