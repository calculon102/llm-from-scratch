"""Causal self-attention and the MiniGPT language model (T-401/T-402).

Attention is the mechanism that lets a language model look at earlier
tokens while predicting the next one.  For a sequence of token embeddings
``x`` this module computes queries, keys and values::

    Q = x W_q,   K = x W_k,   V = x W_v

    scores = Q K^T / sqrt(head_dim)        # (B, heads, T, T)
    scores = scores masked to the past      # no peeking at the future
    weights = softmax(scores, dim=-1)
    out = weights V

The causal mask is the only line that makes this a *language model*
rather than a bidirectional encoder: position ``t`` may attend to
``0..t``, never to ``t+1..T-1``.  ``CausalSelfAttention`` stores that mask
as a buffer so tests can inspect it directly, and
:meth:`CausalSelfAttention.attention_weights` exposes the normalized
weights for diagnostics.

``MiniGPT`` stacks pre-LayerNorm transformer blocks, adds learned token
and position embeddings, and projects the final hidden state to vocabulary
logits.  With ``tie_weights=True`` (the default) the output projection
shares its matrix with the token embedding, the standard parameter-saving
trick from GPT-2.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CausalSelfAttention",
    "causal_mask",
]

#: Default architecture for the tiny profiles.  Real runs override these
#: through ``phase_config`` (see :mod:`core.gpt`).
DEFAULT_CONTEXT_LENGTH = 64
DEFAULT_N_LAYER = 2
DEFAULT_N_HEAD = 4
DEFAULT_N_EMBD = 64
DEFAULT_DROPOUT = 0.0


def causal_mask(context_length: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a lower-triangular boolean ``(T, T)`` mask.

    ``mask[i, j]`` is true when position ``i`` is allowed to attend to
    position ``j``.  A causal language model allows ``j <= i`` only, so
    the result is the lower triangle including the diagonal.
    """
    if context_length < 1:
        raise ValueError("context_length must be >= 1")
    return torch.tril(
        torch.ones(context_length, context_length, dtype=torch.bool, device=device)
    )


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with a causal mask.

    Parameters
    ----------
    n_embd:
        Width of the residual stream.  Must be divisible by ``n_head``.
    n_head:
        Number of attention heads.  Every head sees ``n_embd / n_head``
        channels and the heads are concatenated afterwards.
    context_length:
        Maximum sequence length the layer can process.  The causal mask is
        this size, so a longer input raises a clear error instead of
        silently attending beyond the trained context.
    dropout:
        Dropout on the attention weights and on the output projection.
    bias:
        Whether the QKV and output projections use a bias term.  GPT-2
        uses bias; many modern models do not.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        *,
        dropout: float = DEFAULT_DROPOUT,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if n_embd < 1:
            raise ValueError("n_embd must be >= 1")
        if n_head < 1:
            raise ValueError("n_head must be >= 1")
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd={n_embd} must be divisible by n_head={n_head}"
            )
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.n_embd = int(n_embd)
        self.n_head = int(n_head)
        self.context_length = int(context_length)
        self.head_dim = self.n_embd // self.n_head

        # One fused projection is slightly faster than three separate ones
        # and keeps the three matrices next to each other for inspection.
        self.qkv_proj = nn.Linear(self.n_embd, 3 * self.n_embd, bias=bias)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        # persistent=False: the mask is a constant, not a learned parameter,
        # and deriving it again after loading is cheaper than shipping it in
        # every checkpoint.  It still appears in state_dict-free inspection.
        self.register_buffer(
            "mask",
            causal_mask(self.context_length),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """``(B, T, C) -> (B, n_head, T, head_dim)``."""
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return the causal attention probabilities, shape ``(B, H, T, T)``.

        This is the diagnostic view used by tests and by anyone who wants
        to see what the layer is looking at.  It performs the same
        computation as :meth:`forward` up to (and including) the softmax,
        without dropout.
        """
        if x.dim() != 3 or x.shape[-1] != self.n_embd:
            raise ValueError(
                f"expected (B, T, {self.n_embd}) input, got {tuple(x.shape)}"
            )
        length = x.shape[1]
        if length > self.context_length:
            raise ValueError(
                f"sequence length {length} exceeds context_length "
                f"{self.context_length}"
            )
        q, k, _ = self.qkv_proj(x).split(self.n_embd, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.mask[:length, :length]
        scores = scores.masked_fill(~mask, float("-inf"))
        return F.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head attention; shape is preserved."""
        if x.dim() != 3 or x.shape[-1] != self.n_embd:
            raise ValueError(
                f"expected (B, T, {self.n_embd}) input, got {tuple(x.shape)}"
            )
        batch, length, _ = x.shape
        if length > self.context_length:
            raise ValueError(
                f"sequence length {length} exceeds context_length "
                f"{self.context_length}"
            )

        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.mask[:length, :length]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        out = weights @ v                              # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, length, self.n_embd)
        return self.residual_dropout(self.out_proj(out))
