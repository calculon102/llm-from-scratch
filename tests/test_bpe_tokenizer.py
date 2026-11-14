"""Tests for the from-scratch BPE trainer (T-203).

Acceptance criterion: vocabulary and merges persist and reload exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.tokenizer.bpe import BPETokenizer, split_chunks
from core.tokenizer.char import CharTokenizer
from core.tokenizer.common import SPECIAL_TOKENS, TokenizerError

TEXT = (
    "Die Straße war frei. Grüße aus Köln! "
    "Die Straße war frei. Grüße aus Köln! "
    "Die Straße war frei. Grüße aus Köln! "
)


# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------


def test_split_chunks_reconstructs_input():
    for text in (
        "Grüße, 2025!\nTschüß…",
        "a  b\tc\r\nd",
        "keine-leerzeichen",
        "",
    ):
        assert "".join(split_chunks(text)) == text


def test_split_chunks_separates_words_whitespace_punctuation():
    assert split_chunks("Hallo, Welt!") == ["Hallo", ",", " ", "Welt", "!"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_training_is_deterministic():
    first = BPETokenizer.train(TEXT, vocab_size=120, min_frequency=1)
    second = BPETokenizer.train(TEXT, vocab_size=120, min_frequency=1)
    assert first.vocab == second.vocab
    assert first.merges == second.merges
    assert first.encode(TEXT) == second.encode(TEXT)


def test_training_stats_are_consistent():
    tok = BPETokenizer.train(TEXT, vocab_size=120, min_frequency=1)
    stats = tok.training_stats
    assert stats["target_vocab_size"] == 120
    assert stats["base_vocab_size"] == len(SPECIAL_TOKENS) + stats["alphabet_size"]
    assert stats["vocab_size"] == tok.vocab_size
    assert stats["merges"] == len(tok.merges)
    assert tok.vocab_size == stats["base_vocab_size"] + len(tok.merges)
    assert stats["stop_reason"] in {
        "target_vocab_size",
        "min_frequency",
        "max_token_length",
    }


def test_roundtrip_is_exact_and_compresses():
    tok = BPETokenizer.train(TEXT, vocab_size=120, min_frequency=1)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    assert tok.roundtrip_ok(TEXT)
    # Far fewer tokens than characters: BPE actually compresses.
    assert len(ids) < len(TEXT) / 2


def test_whitespace_and_newlines_roundtrip():
    text = "Zeile eins\n\tZeile zwei   mit  Abstand\n\nEnde ohne Punkt"
    tok = BPETokenizer.train(text * 3, vocab_size=120, min_frequency=1)
    assert tok.decode(tok.encode(text)) == text


def test_merges_never_cross_chunk_boundaries():
    tok = BPETokenizer.train("ab cd ef " * 5, vocab_size=60, min_frequency=1)
    tokens = tok.tokenize("ab cd ef")
    assert "".join(tokens) == "ab cd ef"
    assert not any(
        any(char.isalnum() for char in token)
        and any(char.isspace() for char in token)
        for token in tokens
    )


def test_max_token_length_is_respected():
    text = "Donaudampfschifffahrtsgesellschaft " * 5
    tok = BPETokenizer.train(
        text, vocab_size=200, min_frequency=1, max_token_length=8
    )
    assert all(len(left + right) <= 8 for left, right in tok.merges)
    assert all(len(token) <= 8 for token in tok.vocab if token not in SPECIAL_TOKENS)


def test_min_frequency_stops_training():
    tok = BPETokenizer.train("abcdefghij", vocab_size=100, min_frequency=2)
    assert tok.merges == []
    assert tok.training_stats["stop_reason"] == "min_frequency"


def test_target_vocab_size_stops_training():
    tok = BPETokenizer.train("abababab" * 20, vocab_size=10, min_frequency=1)
    assert tok.vocab_size == 10
    assert tok.training_stats["stop_reason"] == "target_vocab_size"


def test_vocab_size_below_alphabet_raises():
    with pytest.raises(TokenizerError, match="too small"):
        BPETokenizer.train("abcdefghij", vocab_size=8)


def test_invalid_training_parameters_raise():
    with pytest.raises(TokenizerError, match="min_frequency"):
        BPETokenizer.train(TEXT, min_frequency=0)
    with pytest.raises(TokenizerError, match="max_token_length"):
        BPETokenizer.train(TEXT, max_token_length=1)
    with pytest.raises(TokenizerError, match="empty text"):
        BPETokenizer.train("")


def test_from_corpus_wrapper():
    class FakeCorpus:
        text = "abcabcabc"

    tok = BPETokenizer.from_corpus(FakeCorpus(), vocab_size=20, min_frequency=1)
    assert tok.roundtrip_ok("abcabc")


# ---------------------------------------------------------------------------
# Encoding / OOV
# ---------------------------------------------------------------------------


def test_unknown_character_becomes_unk():
    tok = BPETokenizer.train("abcabc", vocab_size=20, min_frequency=1)
    ids = tok.encode("ab😀")
    assert ids[-1] == tok.unk_id
    assert tok.decode(ids) == "ab\ufffd"
    assert tok.oov_chars("ab😀") == {"😀": 1}
    with pytest.raises(TokenizerError, match="U\\+1F600"):
        tok.encode("ab😀", strict=True)


def test_special_tokens_and_token_strings():
    tok = BPETokenizer.train("abcabc", vocab_size=20, min_frequency=1)
    ids = tok.encode("abc", add_special_tokens=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    strings = tok.token_strings(ids)
    assert strings[0] == "<bos>"
    assert tok.decode(ids) == "abc"


def test_manual_merges_apply_in_rank_order():
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token in ("a", "b", "c", "ab", "bc", "abc"):
        vocab[token] = len(vocab)
    tok = BPETokenizer(vocab, [("a", "b"), ("b", "c"), ("ab", "c")])
    assert tok.tokenize("abc") == ["abc"]
    assert tok.encode("abc") == [vocab["abc"]]


def test_invalid_merges_are_rejected():
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    vocab.update({"a": 4, "b": 5})
    with pytest.raises(TokenizerError, match="no merged token"):
        BPETokenizer(vocab, [("a", "b")])
    with pytest.raises(TokenizerError, match="duplicate merge"):
        BPETokenizer({**vocab, "ab": 6}, [("a", "b"), ("a", "b")])


# ---------------------------------------------------------------------------
# Acceptance: persistence
# ---------------------------------------------------------------------------


def test_save_and_reload_is_exact(tmp_path: Path):
    tok = BPETokenizer.train(TEXT, vocab_size=120, min_frequency=1)
    directory = tok.save(tmp_path / "bpe")
    assert (directory / "vocab.json").is_file()
    assert (directory / "merges.json").is_file()
    assert (directory / "meta.json").is_file()

    loaded = BPETokenizer.load(directory)
    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges
    assert loaded.training_stats == tok.training_stats
    assert loaded.encode(TEXT) == tok.encode(TEXT)
    assert loaded.decode(loaded.encode(TEXT)) == TEXT

    raw_merges = json.loads((directory / "merges.json").read_text(encoding="utf-8"))
    assert [tuple(pair) for pair in raw_merges] == tok.merges


def test_reload_preserves_merge_priority(tmp_path: Path):
    tok = BPETokenizer.train("abababab" * 20, vocab_size=30, min_frequency=1)
    loaded = BPETokenizer.load(tok.save(tmp_path / "bpe"))
    for text in ("abab", "ba", "abababab"):
        assert loaded.tokenize(text) == tok.tokenize(text)


def test_load_rejects_missing_or_wrong_directory(tmp_path: Path):
    with pytest.raises(TokenizerError, match="not a BPE tokenizer"):
        BPETokenizer.load(tmp_path / "missing")

    CharTokenizer.from_text("abc").save(tmp_path / "char")
    with pytest.raises(TokenizerError, match="missing merges.json"):
        BPETokenizer.load(tmp_path / "char")


def test_load_rejects_wrong_version(tmp_path: Path):
    directory = BPETokenizer.train("abcabc", vocab_size=20, min_frequency=1).save(
        tmp_path / "bpe"
    )
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    meta["version"] = 999
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(TokenizerError, match="unsupported"):
        BPETokenizer.load(directory)
