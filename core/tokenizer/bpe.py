"""Byte-pair-encoding trainer and tokenizer, from scratch (T-203).

BPE is a greedy compression algorithm that learns a vocabulary of
frequent *subword* pieces.  It starts from the character alphabet and
repeatedly merges the most frequent adjacent pair of symbols::

    corpus words:  n i e d e r   n i e d e r l a n d
    step 1:        ni e d e r   ni e d e r l a n d
    step 2:        nie d e r   nie d e r l a n d
    ...
    learned merges: (n,i)->ni, (ni,e)->nie, (d,e)->de, (de,r)->der, ...

The result is a vocabulary between characters and whole words: common
words become one token, rare words stay as a few meaningful pieces, and
the sequence length for German text drops by roughly 3–4× versus a
character tokenizer (T-204 measures it).

What is implemented here
------------------------
* **Pre-tokenization.** Text is split into chunks (``\\s+``, ``\\w+``,
  punctuation runs).  Merges never cross a chunk boundary, so a token
  cannot silently swallow a space plus half of the next word.  Because
  every chunk is kept verbatim and decoding just concatenates token
  strings, ``decode(encode(text)) == text`` for any text whose characters
  are in the vocabulary.
* **Training.** Pair counts are weighted by chunk frequency, ties are
  broken lexicographically (deterministic!), and the trainer stops at the
  target vocabulary size, when no pair is frequent enough, or when every
  remaining merge would exceed ``max_token_length``.
* **Persistence.** ``vocab.json`` + ``merges.json`` + ``meta.json`` are
  written to a directory and can be reloaded into an identical encoder.
  The vocab is a plain token→id mapping; the merges file preserves the
  *order*, which is the priority order used at encode time.
* **OOV.** Characters never seen during training become ``<unk>``; the
  character tokenizer's ids 0–3 are reused so downstream phases see one
  id convention.

Usage::

    from core.tokenizer.bpe import BPETokenizer

    tok = BPETokenizer.train(corpus.text, vocab_size=500)
    ids = tok.encode("Die Straße ist frei.")
    assert tok.decode(ids) == "Die Straße ist frei."
    tok.save("artifacts/.../tokenizers/bpe")
    same = BPETokenizer.load("artifacts/.../tokenizers/bpe")
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from core.tokenizer.common import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    TokenizerError,
)

__all__ = ["BPETokenizer", "split_chunks", "CHUNK_RE"]

FORMAT_NAME = "llm-from-scratch-tokenizer"
FORMAT_VERSION = 1
VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.json"
META_FILE = "meta.json"

#: Pre-tokenization: whitespace runs, word runs, punctuation runs.  The
#: three alternatives together cover every character; concatenating the
#: chunks reproduces the input exactly.  ``\w`` is Unicode-aware, so
#: German umlauts and ``ß`` belong to words.
CHUNK_RE = re.compile(r"\s+|\w+|[^\w\s]+", re.UNICODE)

_DECODE_ERRORS = ("replace", "strict", "ignore")


def split_chunks(text: str) -> list[str]:
    """Split ``text`` into BPE chunks (whitespace / word / punctuation)."""
    return CHUNK_RE.findall(text)


def _count_pairs(word_symbols: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
    """Count adjacent symbol pairs, weighted by chunk frequency."""
    counts: Counter[tuple[str, str]] = Counter()
    for symbols, frequency in word_symbols.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += frequency
    return counts


def _merge_symbols(
    symbols: tuple[str, ...], pair: tuple[str, str]
) -> tuple[str, ...]:
    """Replace every non-overlapping occurrence of ``pair`` in ``symbols``."""
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if (
            index + 1 < len(symbols)
            and symbols[index] == pair[0]
            and symbols[index + 1] == pair[1]
        ):
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return tuple(merged)


class BPETokenizer:
    """A trained BPE vocabulary plus its ordered merge rules.

    Parameters
    ----------
    vocab:
        ``token -> id``; ids contiguous from 0 with specials first.
    merges:
        Ordered ``(left, right)`` pairs.  Each merge's concatenation must
        be a token in ``vocab``; the order is the encode-time priority.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]] | tuple[tuple[str, str], ...],
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
        *,
        max_token_length: int = 32,
        training_stats: dict | None = None,
    ) -> None:
        if not vocab:
            raise TokenizerError("vocabulary must not be empty")
        ids = sorted(vocab.values())
        if ids != list(range(len(vocab))):
            raise TokenizerError(
                "token ids must be contiguous starting at 0 "
                f"(got {len(vocab)} entries with ids {ids[:5]}…)"
            )
        self._special_tokens = tuple(special_tokens)
        for index, token in enumerate(self._special_tokens):
            if vocab.get(token) != index:
                raise TokenizerError(f"special token {token!r} must have id {index}")

        pairs = [(str(left), str(right)) for left, right in merges]
        if len(set(pairs)) != len(pairs):
            raise TokenizerError("duplicate merge rules")
        for left, right in pairs:
            merged = left + right
            if left not in vocab or right not in vocab:
                raise TokenizerError(
                    f"merge ({left!r}, {right!r}) references a token outside the vocabulary"
                )
            if merged not in vocab:
                raise TokenizerError(
                    f"merge ({left!r}, {right!r}) has no merged token {merged!r} in the vocabulary"
                )
        if max_token_length < 2:
            raise TokenizerError("max_token_length must be >= 2")

        self._vocab = dict(vocab)
        self._itos = {index: token for token, index in self._vocab.items()}
        self._special_ids = frozenset(range(len(self._special_tokens)))
        self._merges = pairs
        self._ranks = {pair: rank for rank, pair in enumerate(pairs)}
        self._max_token_length = max_token_length
        self._training_stats = dict(training_stats or {})
        #: Characters that are directly representable (single-symbol tokens).
        self._base_chars = frozenset(
            token
            for token in self._vocab
            if len(token) == 1 and token not in self._special_tokens
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        text: str,
        *,
        vocab_size: int = 500,
        min_frequency: int = 2,
        max_token_length: int = 32,
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
    ) -> "BPETokenizer":
        """Train BPE on ``text`` and return the tokenizer.

        ``vocab_size`` includes the special tokens and the character
        alphabet.  If it is smaller than that base, training raises rather
        than silently dropping characters.  The returned tokenizer exposes
        :attr:`training_stats` with the stop reason and counts.
        """
        if vocab_size < len(special_tokens) + 1:
            raise TokenizerError(
                f"vocab_size must be at least {len(special_tokens) + 1}, got {vocab_size}"
            )
        if min_frequency < 1:
            raise TokenizerError(f"min_frequency must be >= 1, got {min_frequency}")
        if max_token_length < 2:
            raise TokenizerError("max_token_length must be >= 2")

        chunks = Counter(split_chunks(text))
        if not chunks:
            raise TokenizerError("cannot train BPE on empty text")
        alphabet = sorted({char for chunk in chunks for char in chunk})
        base_size = len(special_tokens) + len(alphabet)
        if vocab_size < base_size:
            raise TokenizerError(
                f"vocab_size={vocab_size} is too small for this corpus: "
                f"{len(special_tokens)} special tokens + {len(alphabet)} characters "
                f"= {base_size}"
            )

        vocab = {token: index for index, token in enumerate(special_tokens)}
        for char in alphabet:
            vocab[char] = len(vocab)

        word_symbols: Counter[tuple[str, ...]] = Counter(
            {tuple(chunk): frequency for chunk, frequency in chunks.items()}
        )
        merges: list[tuple[str, str]] = []
        stop_reason = "target_vocab_size"

        while len(vocab) < vocab_size:
            pair_counts = _count_pairs(word_symbols)
            frequent = [
                (pair, count)
                for pair, count in pair_counts.items()
                if count >= min_frequency
            ]
            if not frequent:
                stop_reason = "min_frequency"
                break
            candidates = [
                (pair, count)
                for pair, count in frequent
                if len(pair[0]) + len(pair[1]) <= max_token_length
                and (pair[0] + pair[1]) not in vocab
            ]
            if not candidates:
                stop_reason = "max_token_length"
                break
            # Highest count wins; lexicographic tie-break makes training
            # reproducible across runs and Python hash seeds.
            pair, count = min(candidates, key=lambda item: (-item[1], item[0]))
            merged = pair[0] + pair[1]
            vocab[merged] = len(vocab)
            merges.append(pair)
            word_symbols = Counter(
                {
                    _merge_symbols(symbols, pair): frequency
                    for symbols, frequency in word_symbols.items()
                }
            )

        stats = {
            "text_chars": len(text),
            "chunks": len(chunks),
            "alphabet_size": len(alphabet),
            "base_vocab_size": base_size,
            "target_vocab_size": vocab_size,
            "vocab_size": len(vocab),
            "merges": len(merges),
            "min_frequency": min_frequency,
            "max_token_length": max_token_length,
            "stop_reason": stop_reason,
        }
        return cls(
            vocab,
            merges,
            special_tokens=special_tokens,
            max_token_length=max_token_length,
            training_stats=stats,
        )

    @classmethod
    def from_corpus(cls, corpus, **kwargs) -> "BPETokenizer":
        """Train on ``corpus.text`` (or a plain string)."""
        text = getattr(corpus, "text", corpus)
        return cls.train(text, **kwargs)

    # ------------------------------------------------------------------
    # Vocabulary access
    # ------------------------------------------------------------------

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def merges(self) -> list[tuple[str, str]]:
        return list(self._merges)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def max_token_length(self) -> int:
        return self._max_token_length

    @property
    def training_stats(self) -> dict:
        return dict(self._training_stats)

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return self._special_tokens

    @property
    def special_ids(self) -> frozenset[int]:
        return self._special_ids

    @property
    def pad_id(self) -> int:
        return self._vocab[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._vocab[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._vocab[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._vocab[EOS_TOKEN]

    def token_to_id(self, token: str) -> int | None:
        return self._vocab.get(token)

    def id_to_token(self, token_id: int) -> str:
        try:
            return self._itos[token_id]
        except KeyError as exc:
            raise TokenizerError(f"token id out of range: {token_id}") from exc

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, token: object) -> bool:
        return token in self._vocab

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"merges={len(self._merges)})"
        )

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        """Apply the learned merges greedily by rank (lowest rank first)."""
        while len(symbols) > 1:
            best_rank: int | None = None
            best_index = -1
            for index in range(len(symbols) - 1):
                rank = self._ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_rank is None:
                break
            symbols[best_index : best_index + 2] = [
                symbols[best_index] + symbols[best_index + 1]
            ]
        return symbols

    def tokenize(self, text: str) -> list[str]:
        """Return the BPE token strings for ``text``."""
        tokens: list[str] = []
        for chunk in split_chunks(text):
            tokens.extend(self._apply_merges(list(chunk)))
        return tokens

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        strict: bool = False,
    ) -> list[int]:
        """Encode ``text``; unknown characters map to ``<unk>``."""
        ids: list[int] = [self.bos_id] if add_special_tokens else []
        unk_id = self.unk_id
        for token in self.tokenize(text):
            token_id = self._vocab.get(token)
            if token_id is None:
                if strict:
                    raise TokenizerError(
                        f"character {token!r} (U+{ord(token):04X}) "
                        "is not in the vocabulary"
                    )
                token_id = unk_id
            ids.append(token_id)
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def token_strings(self, ids: list[int]) -> list[str]:
        return [self.id_to_token(token_id) for token_id in ids]

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "replace",
    ) -> str:
        """Decode ids back into text (see :meth:`CharTokenizer.decode`)."""
        if errors not in _DECODE_ERRORS:
            raise TokenizerError(
                f"errors must be one of {_DECODE_ERRORS}, got {errors!r}"
            )
        parts: list[str] = []
        for token_id in ids:
            token = self.id_to_token(token_id)
            if token_id in self._special_ids:
                if token == UNK_TOKEN:
                    if errors == "strict":
                        raise TokenizerError("cannot decode <unk> in strict mode")
                    if errors == "replace":
                        parts.append("\ufffd")
                    continue
                if skip_special_tokens:
                    continue
                parts.append(token)
                continue
            parts.append(token)
        return "".join(parts)

    def oov_chars(self, text: str) -> Counter[str]:
        """Characters of ``text`` that are not representable by the vocab."""
        return Counter(
            char for char in text if char not in self._base_chars
        )

    def oov_count(self, text: str) -> int:
        return sum(self.oov_chars(text).values())

    def roundtrip_ok(self, text: str) -> bool:
        return self.decode(self.encode(text)) == text

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_meta(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "type": "bpe",
            "special_tokens": list(self._special_tokens),
            "vocab_size": self.vocab_size,
            "merges": len(self._merges),
            "max_token_length": self._max_token_length,
            "training_stats": self._training_stats,
        }

    def save(self, directory: str | Path) -> Path:
        """Write ``vocab.json``, ``merges.json`` and ``meta.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / VOCAB_FILE).write_text(
            json.dumps(self._vocab, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (directory / MERGES_FILE).write_text(
            json.dumps(
                [[left, right] for left, right in self._merges],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (directory / META_FILE).write_text(
            json.dumps(self.to_meta(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        """Reload a tokenizer written by :meth:`save`.

        The reload is exact: same ids, same merge order, same encoding.
        """
        directory = Path(directory)
        paths = {
            name: directory / name for name in (VOCAB_FILE, MERGES_FILE, META_FILE)
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise TokenizerError(
                f"not a BPE tokenizer directory: {directory} "
                f"(missing {', '.join(missing)})"
            )
        try:
            meta = json.loads(paths[META_FILE].read_text(encoding="utf-8"))
            raw_vocab = json.loads(paths[VOCAB_FILE].read_text(encoding="utf-8"))
            raw_merges = json.loads(paths[MERGES_FILE].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"cannot read tokenizer {directory}: {exc}") from exc
        if meta.get("type") != "bpe":
            raise TokenizerError(
                f"{directory} contains a {meta.get('type')!r} tokenizer, not 'bpe'"
            )
        if meta.get("version") != FORMAT_VERSION:
            raise TokenizerError(
                f"unsupported tokenizer format version {meta.get('version')!r}"
            )
        try:
            vocab = {str(token): int(index) for token, index in raw_vocab.items()}
            merges = [(str(left), str(right)) for left, right in raw_merges]
        except (AttributeError, TypeError, ValueError) as exc:
            raise TokenizerError(f"invalid tokenizer data in {directory}: {exc}") from exc
        return cls(
            vocab,
            merges,
            special_tokens=tuple(meta.get("special_tokens", SPECIAL_TOKENS)),
            max_token_length=int(meta.get("max_token_length", 32)),
            training_stats=meta.get("training_stats") or {},
        )
