"""Character tokenizer (T-202).

A character tokenizer is the simplest possible tokenizer: every Unicode
code point (after NFC normalization) is one token, plus four reserved
special tokens.  It is the baseline every more clever tokenizer must beat,
and it is also the *alphabet* on which the BPE trainer (T-203) starts.

Why start here?
--------------
* **No vocabulary explosion.**  A German corpus needs roughly 60–90
  characters (letters, umlauts, punctuation, digits, whitespace); the
  resulting vocabulary is tiny and every word is representable.
* **No OOV for normal text.**  Any text drawn from the training corpus
  round-trips exactly.  Only genuinely new symbols (emoji, rare scripts)
  become ``<unk>``.
* **A clean lower bound for compression.**  One token per character means
  compression ratio = 1 char/token.  BPE's ratio is measured against this
  number (T-204).

The trade-off is sequence length: German averages ~5 characters per word,
so a 1,000-character passage becomes ~1,000 tokens.  Transformer attention
cost grows quadratically with that length; BPE exists to shorten it.

Determinism
-----------
The vocabulary is built by sorting code points, not by corpus frequency:
the same text always yields exactly the same ids, on every machine and
Python version, and the special ids ``0..3`` never move.

Usage::

    from core.tokenizer.char import CharTokenizer

    tok = CharTokenizer.from_text("Grüße aus Köln! 👋")
    ids = tok.encode("Grüße")
    assert tok.decode(ids) == "Grüße"
    tok.save("artifacts/.../tokenizers/char")
    same = CharTokenizer.load("artifacts/.../tokenizers/char")
"""

from __future__ import annotations

import json
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

__all__ = ["CharTokenizer"]

FORMAT_NAME = "llm-from-scratch-tokenizer"
FORMAT_VERSION = 1
VOCAB_FILE = "vocab.json"
META_FILE = "meta.json"

#: Decoding modes for unknown symbols / malformed ids.
_DECODE_ERRORS = ("replace", "strict", "ignore")


class CharTokenizer:
    """One token per character, with fixed ``<pad>/<unk>/<bos>/<eos>`` ids.

    Parameters
    ----------
    vocab:
        Mapping ``token -> id``.  Ids must be contiguous from 0 and the
        four special tokens must occupy ids 0–3.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
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
                raise TokenizerError(
                    f"special token {token!r} must have id {index}"
                )
        self._vocab = dict(vocab)
        self._itos = {index: token for token, index in self._vocab.items()}
        self._special_ids = frozenset(range(len(self._special_tokens)))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        min_frequency: int = 1,
        extra_chars: str = "",
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
    ) -> "CharTokenizer":
        """Build a character vocabulary from ``text``.

        Characters are ordered by code point, so the result is
        deterministic.  ``min_frequency`` drops rare characters (they
        become ``<unk>``); ``extra_chars`` force-includes characters that
        do not occur in the training text.
        """
        if min_frequency < 1:
            raise TokenizerError(f"min_frequency must be >= 1, got {min_frequency}")
        counts = Counter(text)
        chars = {char for char, count in counts.items() if count >= min_frequency}
        chars.update(extra_chars)
        chars.difference_update(special_tokens)
        if not chars:
            raise TokenizerError("no characters found to build a vocabulary")
        vocab = {token: index for index, token in enumerate(special_tokens)}
        for char in sorted(chars):
            vocab[char] = len(vocab)
        return cls(vocab, special_tokens=special_tokens)

    @classmethod
    def from_corpus(
        cls,
        corpus,
        *,
        min_frequency: int = 1,
        extra_chars: str = "",
    ) -> "CharTokenizer":
        """Convenience wrapper around :meth:`from_text` for a Corpus."""
        text = getattr(corpus, "text", corpus)
        return cls.from_text(
            text, min_frequency=min_frequency, extra_chars=extra_chars
        )

    # ------------------------------------------------------------------
    # Vocabulary access
    # ------------------------------------------------------------------

    @property
    def vocab(self) -> dict[str, int]:
        """Copy of the ``token -> id`` mapping."""
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

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
        return f"CharTokenizer(vocab_size={self.vocab_size})"

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def token_strings(self, ids: list[int]) -> list[str]:
        """Map ids to token strings (special tokens included)."""
        return [self.id_to_token(token_id) for token_id in ids]

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        strict: bool = False,
    ) -> list[int]:
        """Encode ``text`` into token ids.

        Unknown characters become ``<unk>`` unless ``strict`` is true, in
        which case :class:`TokenizerError` names the first unknown
        character.  ``add_special_tokens`` wraps the result in
        ``<bos> … <eos>``.
        """
        ids: list[int] = [self.bos_id] if add_special_tokens else []
        unk_id = self.unk_id
        for char in text:
            token_id = self._vocab.get(char)
            if token_id is None:
                if strict:
                    raise TokenizerError(
                        f"character {char!r} (U+{ord(char):04X}) "
                        "is not in the vocabulary"
                    )
                token_id = unk_id
            ids.append(token_id)
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "replace",
    ) -> str:
        """Decode ids back into text.

        ``<unk>`` becomes U+FFFD (``errors="replace"``), raises
        (``"strict"``), or disappears (``"ignore"``).  With
        ``skip_special_tokens`` the remaining specials are omitted; with
        ``False`` they are written literally, which is useful for
        debugging and token displays.
        """
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
        """Count characters of ``text`` that are not in the vocabulary."""
        return Counter(char for char in text if char not in self._vocab)

    def oov_count(self, text: str) -> int:
        """Total number of characters that would encode to ``<unk>``."""
        return sum(self.oov_chars(text).values())

    def roundtrip_ok(self, text: str) -> bool:
        """True when ``decode(encode(text)) == text``."""
        return self.decode(self.encode(text)) == text

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_meta(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "type": "char",
            "special_tokens": list(self._special_tokens),
            "vocab_size": self.vocab_size,
        }

    def save(self, directory: str | Path) -> Path:
        """Write ``vocab.json`` + ``meta.json`` into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / VOCAB_FILE).write_text(
            json.dumps(self._vocab, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (directory / META_FILE).write_text(
            json.dumps(self.to_meta(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "CharTokenizer":
        """Load a tokenizer previously written by :meth:`save`."""
        directory = Path(directory)
        meta_path = directory / META_FILE
        vocab_path = directory / VOCAB_FILE
        if not meta_path.is_file() or not vocab_path.is_file():
            raise TokenizerError(
                f"not a character tokenizer directory: {directory} "
                f"(expected {META_FILE} and {VOCAB_FILE})"
            )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"cannot read tokenizer {directory}: {exc}") from exc
        if meta.get("type") != "char":
            raise TokenizerError(
                f"{directory} contains a {meta.get('type')!r} tokenizer, not 'char'"
            )
        if meta.get("version") != FORMAT_VERSION:
            raise TokenizerError(
                f"unsupported tokenizer format version {meta.get('version')!r}"
            )
        try:
            vocab = {str(token): int(index) for token, index in raw_vocab.items()}
        except (AttributeError, TypeError, ValueError) as exc:
            raise TokenizerError(f"invalid vocabulary in {vocab_path}: {exc}") from exc
        special_tokens = tuple(meta.get("special_tokens", SPECIAL_TOKENS))
        return cls(vocab, special_tokens=special_tokens)
