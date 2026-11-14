"""Shared constants and errors for the M2 tokenizer module (T-201…T-206).

The tokenizer is deliberately independent of torch: it is pure Python
plus the standard library, so it can be trained and inspected in
milliseconds on any machine.
"""

from __future__ import annotations

__all__ = [
    "SPECIAL_TOKENS",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "TokenizerError",
]

#: Reserved tokens.  They occupy ids ``0..3`` in every tokenizer built by
#: this module, so the ids are stable across character and BPE models:
#:
#: * ``<pad>`` -- batch padding (ignored by the loss),
#: * ``<unk>`` -- a symbol that was not in the vocabulary (OOV),
#: * ``<bos>`` / ``<eos>`` -- optional sequence delimiters.
#:
#: Keeping them first and fixed means downstream phases (embeddings, GPT)
#: can reserve the same rows without knowing which tokenizer produced the
#: ids.
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")

PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = SPECIAL_TOKENS


class TokenizerError(Exception):
    """Raised for invalid tokenizer input, training, or persistence."""
