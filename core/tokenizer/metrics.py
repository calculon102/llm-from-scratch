"""Tokenizer quality metrics (T-204).

A tokenizer is judged on how much text it packs into a token, how many
tokens a word costs, and how often it falls back to ``<unk>``.  This
module computes those numbers for one tokenizer and compares the
character and BPE tokenizers on the same text::

    from core.tokenizer.metrics import compare_tokenizers, log_metrics

    comparison = compare_tokenizers(char_tok, bpe_tok, sample_text)
    log_metrics(run, comparison)        # run.log lines + metrics.json

Definitions
-----------
* **compression ratio** — characters per token, ``len(text) / n_tokens``.
  Higher is better: BPE at 3.5 means one token carries ~3.5 characters.
* **tokens per word** — ``n_tokens / n_words`` (a.k.a. fertility).  A
  German word costs one token only if it was frequent enough to be
  learned whole; otherwise it is split into pieces.
* **OOV rate** — share of tokens that are ``<unk>``.  The character
  tokenizer can only fail on characters it never saw; BPE additionally
  needs its *alphabet* to cover the character.
* **round-trip** — ``decode(encode(text)) == text``.  Any OOV character
  makes this false, which is exactly the signal we want.

Caveat: BPE merges are learned on the training corpus, so measuring
compression on that same text is mildly optimistic.  ``scripts/tokenizer.py``
measures on the cleaned corpus (and notes the caveat in the run log); the
UI compares on arbitrary user-provided text, where the numbers are honest
held-out measurements.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.tokenizer.common import TokenizerError

__all__ = [
    "compare_tokenizers",
    "format_metrics",
    "log_metrics",
    "tokenizer_metrics",
    "write_metrics",
]

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenizer_metrics(tokenizer: Any, text: str) -> dict[str, Any]:
    """Compute quality metrics for one tokenizer on ``text``.

    Works with both :class:`~core.tokenizer.char.CharTokenizer` and
    :class:`~core.tokenizer.bpe.BPETokenizer`: they share the
    ``encode/decode/token_strings/oov_chars/unk_id`` interface.
    """
    if not isinstance(text, str):
        raise TokenizerError("text must be a str")

    ids = tokenizer.encode(text)
    token_strings = tokenizer.token_strings(ids)
    special_ids = tokenizer.special_ids
    content_lengths = [
        len(token)
        for token_id, token in zip(ids, token_strings)
        if token_id not in special_ids
    ]
    oov = tokenizer.oov_chars(text)
    unk_id = tokenizer.unk_id
    unk_tokens = sum(1 for token_id in ids if token_id == unk_id)
    words = _WORD_RE.findall(text)
    char_count = len(text)
    token_count = len(ids)

    return {
        "tokenizer": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "chars": char_count,
        "utf8_bytes": len(text.encode("utf-8")),
        "words": len(words),
        "tokens": token_count,
        "unique_tokens": len(set(ids)),
        "unique_token_ratio": len(set(ids)) / token_count if token_count else 0.0,
        "compression_ratio": char_count / token_count if token_count else 0.0,
        "tokens_per_word": token_count / len(words) if words else 0.0,
        "avg_token_chars": (
            sum(content_lengths) / len(content_lengths) if content_lengths else 0.0
        ),
        "min_token_chars": min(content_lengths) if content_lengths else 0,
        "max_token_chars": max(content_lengths) if content_lengths else 0,
        "unk_tokens": unk_tokens,
        "unk_rate": unk_tokens / token_count if token_count else 0.0,
        "oov_chars_total": sum(oov.values()),
        "oov_char_types": len(oov),
        "oov_chars": [char for char, _count in oov.most_common(20)],
        "roundtrip_ok": tokenizer.decode(ids) == text,
    }


def compare_tokenizers(char_tokenizer: Any, bpe_tokenizer: Any, text: str) -> dict:
    """Compute metrics for both tokenizers and their relative gain.

    ``compression_gain`` is how many character-tokenizer tokens one BPE
    token replaces (e.g. 3.4 means a 3.4× shorter sequence);
    ``token_reduction`` is the same number as a fraction.
    """
    char_metrics = tokenizer_metrics(char_tokenizer, text)
    bpe_metrics = tokenizer_metrics(bpe_tokenizer, text)
    char_tokens = char_metrics["tokens"]
    bpe_tokens = bpe_metrics["tokens"]
    return {
        "text": {
            "chars": len(text),
            "words": char_metrics["words"],
            "utf8_bytes": char_metrics["utf8_bytes"],
        },
        "char": char_metrics,
        "bpe": bpe_metrics,
        "compression_gain": char_tokens / bpe_tokens if bpe_tokens else 0.0,
        "token_reduction": 1.0 - bpe_tokens / char_tokens if char_tokens else 0.0,
        "both_roundtrip_ok": bool(
            char_metrics["roundtrip_ok"] and bpe_metrics["roundtrip_ok"]
        ),
    }


def format_metrics(name: str, metrics: dict[str, Any]) -> str:
    """One human-readable line for a run log."""
    return (
        f"{name}: {metrics['tokens']} tokens / {metrics['chars']} chars "
        f"= {metrics['compression_ratio']:.2f} chars/token, "
        f"{metrics['tokens_per_word']:.2f} tokens/word, "
        f"OOV {metrics['oov_chars_total']} chars "
        f"({metrics['unk_rate']:.2%} of tokens), "
        f"round-trip={'ok' if metrics['roundtrip_ok'] else 'FAILED'}"
    )


def write_metrics(path: str | Path, metrics: dict) -> Path:
    """Write ``metrics`` as pretty JSON (Unicode preserved)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def log_metrics(run: Any, comparison: dict, *, filename: str = "metrics.json") -> Path:
    """Log a comparison to a run (``run.log``) and save ``metrics.json``.

    ``run`` is a :class:`core.artifacts.RunArtifacts` (anything with
    ``log`` and ``run_dir`` works).  Returns the written metrics path.
    """
    text = comparison["text"]
    run.log(
        f"tokenizer metrics: {text['chars']} chars, {text['words']} words, "
        f"{text['utf8_bytes']} UTF-8 bytes"
    )
    run.log(format_metrics("char", comparison["char"]))
    run.log(format_metrics("bpe", comparison["bpe"]))
    run.log(
        f"BPE compression: {comparison['compression_gain']:.2f}× shorter than "
        f"char ({comparison['token_reduction']:.1%} fewer tokens)"
    )
    if not comparison["both_roundtrip_ok"]:
        run.log("WARNING: decode(encode(text)) round-trip failed (OOV characters)")
    path = write_metrics(Path(run.run_dir) / filename, comparison)
    run.log(f"metrics written: {path}")
    return path
