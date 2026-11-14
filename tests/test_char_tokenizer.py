"""Tests for the character tokenizer (T-202).

Acceptance criterion: deterministic encode/decode round-trip passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.tokenizer.char import CharTokenizer
from core.tokenizer.common import SPECIAL_TOKENS, TokenizerError

GERMAN_TEXT = "Grüße aus Köln – die Straße ist frei!\nÄrger? Öl, Übung, ß."


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_vocab_is_deterministic_and_sorted_by_code_point():
    first = CharTokenizer.from_text("cbaßüöä")
    second = CharTokenizer.from_text("äöüßabc")
    assert first.vocab == second.vocab
    assert list(first.vocab) == list(SPECIAL_TOKENS) + sorted("cbaßüöä")


def test_special_ids_are_fixed():
    tok = CharTokenizer.from_text("abc")
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.bos_id == 2
    assert tok.eos_id == 3
    assert tok.special_ids == frozenset({0, 1, 2, 3})
    assert tok.token_to_id("<pad>") == 0


def test_from_text_min_frequency_and_extra_chars():
    tok = CharTokenizer.from_text("aaab", min_frequency=2, extra_chars="z")
    assert "a" in tok
    assert "b" not in tok
    assert "z" in tok
    assert tok.encode("b") == [tok.unk_id]


def test_from_text_empty_raises():
    with pytest.raises(TokenizerError, match="no characters"):
        CharTokenizer.from_text("")


def test_from_corpus_accepts_corpus_object():
    class FakeCorpus:
        text = "abc"

    tok = CharTokenizer.from_corpus(FakeCorpus())
    assert tok.vocab_size == len(SPECIAL_TOKENS) + 3


def test_invalid_vocab_is_rejected():
    with pytest.raises(TokenizerError, match="contiguous"):
        CharTokenizer({"<pad>": 0, "<unk>": 0, "<bos>": 2, "<eos>": 3, "a": 4})
    # Contiguous ids, but <bos> and <eos> are swapped.
    with pytest.raises(TokenizerError, match="special token"):
        CharTokenizer({"<pad>": 0, "<unk>": 1, "<bos>": 3, "<eos>": 2, "a": 4})


# ---------------------------------------------------------------------------
# Acceptance: deterministic round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_is_exact_for_in_vocabulary_text():
    tok = CharTokenizer.from_text(GERMAN_TEXT)
    ids = tok.encode(GERMAN_TEXT)
    assert ids == tok.encode(GERMAN_TEXT)  # deterministic
    assert tok.decode(ids) == GERMAN_TEXT
    assert tok.roundtrip_ok(GERMAN_TEXT)


def test_add_special_tokens():
    tok = CharTokenizer.from_text("ab")
    ids = tok.encode("ab", add_special_tokens=True)
    assert ids == [tok.bos_id, tok.vocab["a"], tok.vocab["b"], tok.eos_id]
    assert tok.decode(ids) == "ab"
    assert tok.decode(ids, skip_special_tokens=False) == "<bos>ab<eos>"


def test_unknown_character_maps_to_unk():
    tok = CharTokenizer.from_text("abc")
    ids = tok.encode("ab😀")
    assert ids[-1] == tok.unk_id
    assert tok.decode(ids) == "ab\ufffd"
    assert not tok.roundtrip_ok("ab😀")
    assert tok.oov_chars("ab😀") == {"😀": 1}
    assert tok.oov_count("ab😀") == 1


def test_strict_encoding_names_the_character():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(TokenizerError, match="U\\+1F600"):
        tok.encode("ab😀", strict=True)


def test_decode_error_modes():
    tok = CharTokenizer.from_text("ab")
    ids = [tok.vocab["a"], tok.unk_id, tok.vocab["b"]]
    assert tok.decode(ids, errors="replace") == "a\ufffdb"
    assert tok.decode(ids, errors="ignore") == "ab"
    with pytest.raises(TokenizerError, match="strict"):
        tok.decode(ids, errors="strict")
    with pytest.raises(TokenizerError, match="errors must be"):
        tok.decode(ids, errors="explode")


def test_invalid_id_raises():
    tok = CharTokenizer.from_text("ab")
    with pytest.raises(TokenizerError, match="out of range"):
        tok.decode([999])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path):
    tok = CharTokenizer.from_text(GERMAN_TEXT)
    directory = tok.save(tmp_path / "char")
    assert (directory / "vocab.json").is_file()
    assert (directory / "meta.json").is_file()

    loaded = CharTokenizer.load(directory)
    assert loaded.vocab == tok.vocab
    assert loaded.encode(GERMAN_TEXT) == tok.encode(GERMAN_TEXT)
    assert loaded.decode(loaded.encode(GERMAN_TEXT)) == GERMAN_TEXT


def test_load_rejects_wrong_directory(tmp_path: Path):
    with pytest.raises(TokenizerError, match="not a character tokenizer"):
        CharTokenizer.load(tmp_path / "missing")


def test_load_rejects_wrong_type(tmp_path: Path):
    from core.tokenizer.bpe import BPETokenizer

    BPETokenizer.train("abcabc", vocab_size=20, min_frequency=1).save(tmp_path / "bpe")
    with pytest.raises(TokenizerError, match="not 'char'"):
        CharTokenizer.load(tmp_path / "bpe")
