"""Tests for the tokenizer pipeline glue and config schema (T-201…T-205)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import core.tokenizer  # noqa: F401  (registers the tokenizer phase schema)
from core.config import ConfigError, load_config
from core.tokenizer import tokenizer_defaults
from core.tokenizer.common import TokenizerError
from core.tokenizer.pipeline import (
    DEFAULT_SAMPLE_CORPUS,
    DEFAULT_SAMPLE_TEXT,
    REPO_ROOT,
    find_bundle_dirs,
    load_bundle,
    metrics_text,
    resolve_corpus_path,
    save_bundle,
    save_corpus_artifacts,
    train_bundle,
    train_bundle_from_config,
)

CONFIG = REPO_ROOT / "configs" / "tokenizer_tiny_debug.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_sample_tokenizer_config_loads():
    cfg = load_config(CONFIG)
    assert cfg["phase"] == "tokenizer"
    assert cfg["phase_config"]["corpus_path"] == "data/sample_de"
    assert cfg["phase_config"]["vocab_size"] == 300
    assert cfg["phase_config"]["encodings"][0] == "utf-8-sig"


def test_tokenizer_config_defaults_are_complete():
    defaults = tokenizer_defaults()
    assert defaults["vocab_size"] == 500
    assert defaults["min_frequency"] == 2
    assert defaults["max_token_length"] == 32
    assert defaults["drop_duplicates"] is True
    assert defaults["lowercase"] is False
    assert defaults["metrics_sample_chars"] == 4000
    assert defaults["corpus_path"] == "data/sample_de"


def _write_config(tmp_path: Path, phase_config: dict, **base) -> Path:
    # Write the raw mapping: load_config is what must reject invalid
    # phase configs, so the test helper must not pre-validate it.
    cfg = {
        "run_name": "tokenizer_test",
        "phase": "tokenizer",
        "phase_config": phase_config,
        **base,
    }
    path = tmp_path / "tokenizer.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_unknown_phase_config_key_rejected(tmp_path: Path):
    path = _write_config(tmp_path, {"vocab_size": 100, "vocab_sise": 100})
    with pytest.raises(ConfigError, match="unknown key 'vocab_sise'"):
        load_config(path)


@pytest.mark.parametrize(
    "phase_config, message",
    [
        ({"vocab_size": 4}, "below minimum"),
        ({"vocab_size": True}, "expected int, got bool"),
        ({"min_frequency": 0}, "below minimum"),
        ({"max_token_length": 1}, "below minimum"),
        ({"min_alpha_ratio": 2.0}, "above maximum"),
        ({"drop_duplicates": "yes"}, "expected bool"),
        ({"encodings": []}, "must not be empty"),
        ({"encodings": ["utf-8", 3]}, "list of str"),
    ],
)
def test_invalid_phase_config_values_rejected(tmp_path: Path, phase_config, message):
    path = _write_config(tmp_path, phase_config)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_empty_phase_config_uses_runtime_defaults(tmp_path: Path):
    path = _write_config(tmp_path, {})
    cfg = load_config(path)
    assert cfg["phase_config"] == {}
    bundle, corpus = train_bundle_from_config(cfg)
    assert corpus.stats["files_ok"] == 4
    assert bundle.bpe.vocab_size <= 500


# ---------------------------------------------------------------------------
# Paths, text and artifact helpers
# ---------------------------------------------------------------------------


def test_resolve_corpus_path():
    assert resolve_corpus_path("data/sample_de") == DEFAULT_SAMPLE_CORPUS
    absolute = Path("/tmp/corpus")
    assert resolve_corpus_path(absolute) == absolute


def test_metrics_text_slices_and_zero_means_all():
    assert metrics_text("abcdef", sample_chars=3) == "abc"
    assert metrics_text("abcdef", sample_chars=0) == "abcdef"
    assert metrics_text("abcdef", sample_chars=999) == "abcdef"


def test_default_sample_text_has_german_signal():
    assert "Straße" in DEFAULT_SAMPLE_TEXT
    assert "Grüße" in DEFAULT_SAMPLE_TEXT
    assert "Fußgängerüberwege" in DEFAULT_SAMPLE_TEXT


# ---------------------------------------------------------------------------
# Training and bundle persistence
# ---------------------------------------------------------------------------


def test_train_bundle_from_config():
    cfg = load_config(CONFIG)
    bundle, corpus = train_bundle_from_config(cfg)
    assert bundle.char.vocab_size > 4
    assert bundle.bpe.vocab_size > bundle.char.vocab_size
    assert bundle.source == "data/sample_de"
    assert bundle.char.roundtrip_ok(corpus.text)
    assert bundle.bpe.roundtrip_ok(corpus.text)


def test_train_bundle_rejects_empty_text():
    with pytest.raises(TokenizerError, match="empty text"):
        train_bundle("   \n")


def test_bundle_save_and_load_is_exact(tmp_path: Path):
    bundle = train_bundle(
        "Die Straße war frei. " * 5, vocab_size=80, min_frequency=1, source="unit-test"
    )
    directory = save_bundle(bundle, tmp_path / "tokenizers")
    assert (directory / "bundle.json").is_file()
    assert (directory / "char" / "vocab.json").is_file()
    assert (directory / "bpe" / "merges.json").is_file()

    loaded = load_bundle(directory)
    assert loaded.source == "unit-test"
    assert loaded.char.vocab == bundle.char.vocab
    assert loaded.bpe.merges == bundle.bpe.merges
    for text in ("Die Straße", "Grüße aus Köln!"):
        assert loaded.bpe.encode(text) == bundle.bpe.encode(text)
        assert loaded.char.encode(text) == bundle.char.encode(text)


def test_load_bundle_rejects_missing_bundle(tmp_path: Path):
    with pytest.raises(TokenizerError, match="not a tokenizer bundle"):
        load_bundle(tmp_path / "missing")


def test_find_bundle_dirs(tmp_path: Path):
    bundle = train_bundle("abcabcabc", vocab_size=20, min_frequency=1)
    save_bundle(bundle, tmp_path / "run_a" / "tokenizers")
    (tmp_path / "run_b" / "tokenizers").mkdir(parents=True)  # incomplete
    found = find_bundle_dirs(tmp_path)
    assert found == [tmp_path / "run_a" / "tokenizers"]


def test_save_corpus_artifacts(tmp_path: Path):
    from core.tokenizer import ingest_corpus

    corpus = ingest_corpus(DEFAULT_SAMPLE_CORPUS)
    paths = save_corpus_artifacts(corpus, tmp_path / "corpus")
    assert paths["corpus"].read_text(encoding="utf-8") == corpus.text
    assert "files" in paths["report"].read_text(encoding="utf-8")
