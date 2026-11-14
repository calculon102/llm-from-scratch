"""Tests for the causal self-attention block (T-401).

Acceptance criterion: "mask correctness + shape tests pass".  The tests
below check the triangular mask itself, the forward-pass shapes, that the
softmax rows sum to one, that future positions get exactly zero attention,
and -- the property that matters most -- that changing a future token
cannot change an earlier token's representation.
"""

from __future__ import annotations

import pytest
import torch

from core.gpt.model import CausalSelfAttention, causal_mask


def test_causal_mask_is_lower_triangular() -> None:
    mask = causal_mask(4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)
    # The diagonal is always visible: a token attends to itself.
    assert torch.diagonal(mask).all()


def test_mask_is_a_registered_buffer() -> None:
    attention = CausalSelfAttention(
        n_embd=16, n_head=2, context_length=6, dropout=0.0
    )
    assert "mask" in dict(attention.named_buffers())
    assert attention.mask.shape == (6, 6)
    assert torch.equal(attention.mask, causal_mask(6))


def test_forward_preserves_shape_and_backpropagates() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(
        n_embd=32, n_head=4, context_length=8, dropout=0.0
    )
    x = torch.randn(2, 5, 32, requires_grad=True)
    out = attention(x)

    assert out.shape == (2, 5, 32)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert attention.qkv_proj.weight.grad is not None
    assert attention.out_proj.weight.grad is not None


def test_attention_weights_are_causal_and_normalized() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(
        n_embd=24, n_head=3, context_length=6, dropout=0.0
    )
    attention.eval()
    x = torch.randn(2, 5, 24)
    weights = attention.attention_weights(x)

    assert weights.shape == (2, 3, 5, 5)
    assert torch.isfinite(weights).all()
    # Every row is a probability distribution over the past.
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 5))
    # Strictly upper-triangular entries are exactly zero: no future access.
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(weights[..., future]) == 0


def test_changing_a_future_token_does_not_change_the_past() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(
        n_embd=16, n_head=2, context_length=8, dropout=0.0
    )
    attention.eval()

    x = torch.randn(1, 6, 16)
    changed = x.clone()
    changed[:, -1, :] = torch.randn(16)

    first = attention(x)
    second = attention(changed)

    # Positions before the changed token must be identical.
    assert torch.allclose(first[:, :-1, :], second[:, :-1, :], atol=1e-6)
    # The changed position itself is allowed to differ.
    assert not torch.allclose(first[:, -1, :], second[:, -1, :])


def test_overlong_sequence_is_rejected() -> None:
    attention = CausalSelfAttention(
        n_embd=8, n_head=2, context_length=4, dropout=0.0
    )
    with pytest.raises(ValueError, match="exceeds context_length"):
        attention(torch.randn(1, 5, 8))


def test_invalid_head_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(n_embd=10, n_head=4, context_length=4)
    with pytest.raises(ValueError, match="n_head"):
        CausalSelfAttention(n_embd=8, n_head=0, context_length=4)
