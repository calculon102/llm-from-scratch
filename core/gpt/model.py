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
    "GPTConfig",
    "MLP",
    "MiniGPT",
    "TransformerBlock",
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


@dataclass(frozen=True)
class GPTConfig:
    """Architecture hyper-parameters for :class:`MiniGPT`.

    The field names mirror the ``phase_config`` keys (see
    :mod:`core.gpt`), so a validated YAML config can be passed almost
    verbatim.  The dataclass validates the shape constraints early: a
    wrong ``n_embd``/``n_head`` combination should fail while building the
    model, not halfway through the first forward pass.
    """

    vocab_size: int
    context_length: int = DEFAULT_CONTEXT_LENGTH
    n_layer: int = DEFAULT_N_LAYER
    n_head: int = DEFAULT_N_HEAD
    n_embd: int = DEFAULT_N_EMBD
    dropout: float = DEFAULT_DROPOUT
    bias: bool = True
    tie_weights: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.context_length < 1:
            raise ValueError("context_length must be >= 1")
        if self.n_layer < 1:
            raise ValueError("n_layer must be >= 1")
        if self.n_head < 1:
            raise ValueError("n_head must be >= 1")
        if self.n_embd < 1:
            raise ValueError("n_embd must be >= 1")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.init_std <= 0.0:
            raise ValueError("init_std must be > 0")

    @classmethod
    def from_dict(cls, data: dict) -> "GPTConfig":
        """Build a config from a plain mapping (e.g. a saved artifact)."""
        known = {field for field in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown GPT config keys: {sorted(unknown)} (allowed: {sorted(known)})"
            )
        return cls(**data)

    def to_dict(self) -> dict:
        """Plain serialisable mapping for ``model.pt``/``meta.json``."""
        return asdict(self)


class MLP(nn.Module):
    """The feed-forward block inside a transformer layer.

    It expands the residual width by 4x, applies GELU, and projects back.
    Attention mixes information *across* positions; this MLP transforms
    each position independently, which is where most of the model's
    parameters and nonlinear capacity live.
    """

    def __init__(self, n_embd: int, *, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.fc_in = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.fc_out = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block: attention then MLP, both residual.

    The order is::

        x = x + attention(ln_1(x))
        x = x + mlp(ln_2(x))

    Normalizing *before* each sublayer (pre-LN) keeps the residual path a
    clean identity, which is why it trains stably without learning-rate
    warmup even for small models.  The original GPT used post-LN
    (``ln(x + attention(x))``); pre-LN is the modern default.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        context_length: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attention = CausalSelfAttention(
            n_embd,
            n_head,
            context_length,
            dropout=dropout,
            bias=bias,
        )
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout=dropout, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-only transformer language model.

    Input is a ``(batch, time)`` tensor of token ids; output is a
    ``(batch, time, vocab_size)`` tensor of next-token logits.  The model
    itself does not compute a loss — that keeps it usable for sampling and
    inspection — but :func:`core.gpt.language_model_loss` turns logits and
    shifted targets into cross-entropy for the shared Trainer.

    Parameters
    ----------
    config:
        A :class:`GPTConfig`.  For convenience, keyword arguments are
        accepted alternatively (``MiniGPT(vocab_size=300, n_layer=2)``).
    """

    def __init__(self, config: GPTConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = GPTConfig(**kwargs)
        elif kwargs:
            raise ValueError("pass either a GPTConfig or keyword arguments, not both")
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.n_embd,
                    config.n_head,
                    config.context_length,
                    dropout=config.dropout,
                    bias=config.bias,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2 scales the residual output projections by 1/sqrt(2*n_layer)
        # so that the variance of the residual stream does not grow with
        # depth.  Without it, deeper tiny models start with very large
        # logits and unstable gradients.
        residual_std = config.init_std / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def context_length(self) -> int:
        return self.config.context_length

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def num_parameters(self, *, non_embedding: bool = False) -> int:
        """Number of trainable parameters, optionally minus the embeddings."""
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total -= self.token_embedding.weight.numel()
            total -= self.position_embedding.weight.numel()
            if self.config.tie_weights:
                # lm_head.weight is the same tensor as token_embedding.weight;
                # parameters() yields it once because it is one Parameter.
                pass
        return int(total)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Map token ids ``(B, T)`` to next-token logits ``(B, T, V)``."""
        if idx.dim() != 2:
            raise ValueError(f"expected token ids of shape (B, T), got {tuple(idx.shape)}")
        batch, length = idx.shape
        if length < 1:
            raise ValueError("sequence length must be >= 1")
        if length > self.context_length:
            raise ValueError(
                f"sequence length {length} exceeds context_length {self.context_length}"
            )
        if idx.numel():
            if int(idx.min()) < 0 or int(idx.max()) >= self.vocab_size:
                raise ValueError(
                    f"token ids must be in [0, {self.vocab_size}); "
                    f"got [{int(idx.min())}, {int(idx.max())}]"
                )

        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
