"""M2 – Tokenizer module (T-201…T-206).

Everything needed to turn a noisy German-first text collection into token
ids, implemented in pure Python:

* :mod:`core.tokenizer.corpus` — robust ingestion (encoding fallback,
  Unicode/whitespace cleanup, line filters, dedupe, JSON report) — T-201
* :mod:`core.tokenizer.char` — deterministic character tokenizer with
  exact round-trip — T-202
* :mod:`core.tokenizer.bpe` — byte-pair encoding trained from scratch,
  persisted as ``vocab.json`` + ``merges.json`` — T-203
* :mod:`core.tokenizer.metrics` — compression / token-length / OOV
  metrics — T-204
* :mod:`core.tokenizer.pipeline` — config → corpus → bundle glue used by
  the CLI and the UI compare page — T-205/T-206

This module also registers the ``"tokenizer"`` phase schema with the
config loader (T-003)::

    from core.tokenizer import validate_tokenizer_phase_config
    # registered at import time

Phase config (all optional; defaults shown)::

    phase_config:
      corpus_path: data/sample_de   # file or directory (UTF-8/CP1252/...)
      vocab_size: 500               # specials + alphabet + merges
      min_frequency: 2              # stop when the best pair is rarer
      max_token_length: 32          # longest learned subword
      min_line_chars: 2             # drop navigation crumbs
      max_line_chars: 2000          # truncate pathological lines
      min_alpha_ratio: 0.0          # 0 keeps punctuation/tables
      drop_duplicates: true         # drop repeated boilerplate lines
      lowercase: false              # keep German casing (recommended)
      encodings: [utf-8-sig, utf-8, cp1252, latin-1]
      extensions: [.txt]            # scanned inside directories
      metrics_sample_chars: 4000    # 0 = measure the whole corpus
"""

from __future__ import annotations

from typing import Any

from core.config import register_phase_schema
from core.tokenizer.bpe import BPETokenizer, split_chunks
from core.tokenizer.char import CharTokenizer
from core.tokenizer.common import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    TokenizerError,
)
from core.tokenizer.corpus import (
    DEFAULT_ENCODINGS,
    DEFAULT_EXTENSIONS,
    Corpus,
    CorpusOptions,
    CleaningStats,
    ingest_corpus,
    clean_text,
)
from core.tokenizer.metrics import (
    compare_tokenizers,
    format_metrics,
    log_metrics,
    tokenizer_metrics,
)
from core.tokenizer.pipeline import (
    DEFAULT_SAMPLE_CORPUS,
    DEFAULT_SAMPLE_TEXT,
    REPO_ROOT,
    TokenizerBundle,
    find_bundle_dirs,
    load_bundle,
    metrics_text,
    resolve_corpus_path,
    save_bundle,
    save_corpus_artifacts,
    train_bundle,
    train_bundle_from_config,
)

__all__ = [
    # common
    "SPECIAL_TOKENS",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "TokenizerError",
    # corpus (T-201)
    "Corpus",
    "CorpusOptions",
    "CleaningStats",
    "ingest_corpus",
    "clean_text",
    "DEFAULT_ENCODINGS",
    "DEFAULT_EXTENSIONS",
    # tokenizers (T-202/T-203)
    "CharTokenizer",
    "BPETokenizer",
    "split_chunks",
    # metrics (T-204)
    "tokenizer_metrics",
    "compare_tokenizers",
    "format_metrics",
    "log_metrics",
    # pipeline / UI (T-205/T-206)
    "TokenizerBundle",
    "train_bundle",
    "train_bundle_from_config",
    "save_bundle",
    "load_bundle",
    "find_bundle_dirs",
    "save_corpus_artifacts",
    "resolve_corpus_path",
    "metrics_text",
    "DEFAULT_SAMPLE_CORPUS",
    "DEFAULT_SAMPLE_TEXT",
    "REPO_ROOT",
    # config
    "TOKENIZER_PHASE_SCHEMA",
    "validate_tokenizer_phase_config",
    "tokenizer_defaults",
]


# ---------------------------------------------------------------------------
# Phase config schema
# ---------------------------------------------------------------------------

TOKENIZER_PHASE_SCHEMA: dict[str, dict[str, Any]] = {
    "corpus_path": {"type": str, "default": "data/sample_de"},
    "vocab_size": {"type": int, "default": 500, "min": 8},
    "min_frequency": {"type": int, "default": 2, "min": 1},
    "max_token_length": {"type": int, "default": 32, "min": 2},
    "min_line_chars": {"type": int, "default": 2, "min": 0},
    "max_line_chars": {"type": int, "default": 2000, "min": 0},
    "min_alpha_ratio": {"type": float, "default": 0.0, "min": 0.0, "max": 1.0},
    "drop_duplicates": {"type": bool, "default": True},
    "lowercase": {"type": bool, "default": False},
    "encodings": {
        "type": list,
        "default": list(DEFAULT_ENCODINGS),
        "item_type": str,
    },
    "extensions": {
        "type": list,
        "default": list(DEFAULT_EXTENSIONS),
        "item_type": str,
    },
    "metrics_sample_chars": {"type": int, "default": 4000, "min": 0},
}


def tokenizer_defaults() -> dict[str, Any]:
    """Return a fresh copy of the phase-config defaults."""
    defaults: dict[str, Any] = {}
    for key, spec in TOKENIZER_PHASE_SCHEMA.items():
        value = spec["default"]
        defaults[key] = list(value) if isinstance(value, list) else value
    return defaults


def _check_type(value: Any, spec: dict[str, Any]) -> bool:
    expected = spec["type"]
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is bool:
        return isinstance(value, bool)
    if expected is list:
        item_type = spec.get("item_type", str)
        return isinstance(value, list) and all(
            isinstance(item, item_type) for item in value
        )
    return isinstance(value, expected)


def validate_tokenizer_phase_config(phase_config: dict, source: str) -> list[str]:
    """Validate the ``phase_config`` section of a tokenizer config.

    Unknown keys are rejected so typos surface at config-load time rather
    than after a corpus has been read and a BPE training run wasted.
    """
    errors: list[str] = []
    for key, value in phase_config.items():
        if key not in TOKENIZER_PHASE_SCHEMA:
            errors.append(
                f"phase_config: unknown key '{key}' "
                f"(allowed: {', '.join(TOKENIZER_PHASE_SCHEMA)})"
            )
            continue
        spec = TOKENIZER_PHASE_SCHEMA[key]
        if not _check_type(value, spec):
            expected_name = spec["type"].__name__
            if spec["type"] is list:
                expected_name = f"list of {spec.get('item_type', str).__name__}"
            errors.append(
                f"phase_config.{key}: expected {expected_name}, "
                f"got {type(value).__name__} ({value!r})"
            )
            continue
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and "min" in spec
            and value < spec["min"]
        ):
            errors.append(
                f"phase_config.{key}: {value} is below minimum {spec['min']}"
            )
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and "max" in spec
            and value > spec["max"]
        ):
            errors.append(
                f"phase_config.{key}: {value} is above maximum {spec['max']}"
            )
        if key in ("encodings", "extensions") and isinstance(value, list):
            if not value:
                errors.append(f"phase_config.{key}: must not be empty")
    return errors


# Registration happens at import time, before any config is loaded by the
# tokenizer CLI or the UI page.
register_phase_schema("tokenizer", validate_tokenizer_phase_config)
