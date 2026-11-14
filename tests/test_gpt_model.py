"""Tests for the decoder transformer stack and LM head (T-402).

Acceptance criterion: "stable tiny forward pass".  The tests check the
logit shapes, that a tiny stack produces finite values and finite
gradients, that the causal property survives the full stack, and that the
weight-tied language-model head behaves as expected.
"""

from __future__ import annotations

import pytest
import torch

from core.gpt.model import GPTConfig, MLP, MiniGPT, TransformerBlock


def tiny_config(**overrides) -> GPTConfig:
    values = dict(
        vocab_size=37,
        context_length=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    values.update(overrides)
    return GPTConfig(**values)


def test_tiny_forward_pass_is_stable() -> None:
    torch.manual_seed(0)
    model = MiniGPT(tiny_config())
    model.eval()
    idx = torch.randint(0, 37, (3, 8))

    logits = model(idx)
    assert logits.shape == (3, 8, 37)
    assert torch.isfinite(logits).all()
    # Initial logits should be small: that is what makes the first
    # cross-entropy close to ln(vocab_size) instead of saturated.
    assert float(logits.detach().std()) < 2.0

    with torch.no_grad():
        again = model(idx)
    assert torch.allclose(logits, again)


def test_tiny_forward_pass_backpropagates_finite_gradients() -> None:
    torch.manual_seed(0)
    model = MiniGPT(tiny_config())
    idx = torch.randint(0, 37, (2, 8))
    targets = torch.randint(0, 37, (2, 8))

    logits = model(idx)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 37), targets.reshape(-1)
    )
    loss.backward()

    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert float(model.token_embedding.weight.grad.abs().sum()) > 0


def test_full_stack_is_causal() -> None:
    torch.manual_seed(0)
    model = MiniGPT(tiny_config(n_layer=3))
    model.eval()
    idx = torch.randint(0, 37, (1, 8))
    changed = idx.clone()
    changed[:, -1] = (int(changed[:, -1]) + 1) % 37

    first = model(idx)
    second = model(changed)
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6)
    assert not torch.allclose(first[:, -1], second[:, -1])


def test_weight_tying_and_parameter_count() -> None:
    tied = MiniGPT(tiny_config(tie_weights=True))
    untied = MiniGPT(tiny_config(tie_weights=False))

    assert tied.lm_head.weight is tied.token_embedding.weight
    assert untied.lm_head.weight is not untied.token_embedding.weight
    # Tying removes exactly one vocabulary-sized matrix.
    assert tied.num_parameters() + 37 * 16 == untied.num_parameters()
    assert tied.num_parameters(non_embedding=True) < tied.num_parameters()


def test_mlp_and_block_preserve_shape() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    assert MLP(16)(x).shape == (2, 5, 16)
    assert TransformerBlock(16, 2, 8)(x).shape == (2, 5, 16)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(vocab_size=10, n_embd=10, n_head=4)
    with pytest.raises(ValueError, match="vocab_size"):
        GPTConfig(vocab_size=0)
    with pytest.raises(ValueError, match="context_length"):
        GPTConfig(vocab_size=10, context_length=0)
    with pytest.raises(ValueError, match="unknown GPT config"):
        GPTConfig.from_dict({"vocab_size": 10, "width": 32})


def test_input_validation() -> None:
    model = MiniGPT(tiny_config())
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(8, dtype=torch.long))
    with pytest.raises(ValueError, match="exceeds context_length"):
        model(torch.zeros(1, 9, dtype=torch.long))
    with pytest.raises(ValueError, match="token ids"):
        model(torch.tensor([[0, 37]], dtype=torch.long))


def test_state_dict_roundtrip() -> None:
    torch.manual_seed(0)
    model = MiniGPT(tiny_config())
    clone = MiniGPT(tiny_config())
    clone.load_state_dict(model.state_dict())

    idx = torch.randint(0, 37, (2, 8))
    model.eval()
    clone.eval()
    with torch.no_grad():
        assert torch.allclose(model(idx), clone(idx))
