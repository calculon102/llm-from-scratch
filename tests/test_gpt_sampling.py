"""Tests for T-405: greedy / top-k / top-p / temperature sampling."""

from __future__ import annotations

import math

import pytest
import torch

from core.gpt import MiniGPT
from core.gpt.sampling import (
    generate,
    generate_text,
    sample_next_token,
    top_k_filter,
    top_p_filter,
)
from core.tokenizer import train_bundle


class FixedLM(torch.nn.Module):
    """A tiny deterministic model that returns the same logits everywhere."""

    def __init__(self, logits: list[float], context_length: int = 8) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))
        self.context_length = context_length
        self.seen_lengths: list[int] = []

    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, V)
        self.seen_lengths.append(int(idx.shape[1]))
        return self.logits.view(1, 1, -1).expand(idx.shape[0], idx.shape[1], -1)


def test_top_k_filter_keeps_only_largest() -> None:
    logits = torch.tensor([0.1, 3.0, -1.0, 2.0])
    filtered = top_k_filter(logits, 2)
    finite = torch.isfinite(filtered)
    assert finite.sum().item() == 2
    assert set(filtered[finite].tolist()) == {2.0, 3.0}
    assert torch.equal(top_k_filter(logits, 0), logits)
    assert torch.equal(top_k_filter(logits, 10), logits)


def test_top_p_filter_keeps_smallest_nucleus() -> None:
    # Probabilities are exactly 0.5 / 0.3 / 0.2.
    logits = torch.log(torch.tensor([0.5, 0.3, 0.2]))

    # 0.7 needs the first two tokens (0.5 then 0.8 cumulative).
    filtered = top_p_filter(logits, 0.7)
    assert filtered[:2].isfinite().all()
    assert not torch.isfinite(filtered[2])

    # 0.4 is reached by the top token alone; the highest token always stays.
    filtered = top_p_filter(logits, 0.4)
    assert torch.isfinite(filtered[0])
    assert not filtered[1:].isfinite().any()

    # p = 1 keeps the entire distribution, including the last token.
    assert torch.isfinite(top_p_filter(logits, 1.0)).all()


def test_top_p_filter_always_keeps_a_token() -> None:
    logits = torch.tensor([5.0, 4.0, 3.0])
    filtered = top_p_filter(logits, 1e-9)
    assert torch.isfinite(filtered).sum().item() == 1
    assert int(filtered.argmax()) == 0


def test_greedy_and_zero_temperature_match_argmax() -> None:
    logits = torch.tensor([0.4, 2.0, -0.5, 1.0])
    generator = torch.Generator().manual_seed(0)
    greedy = sample_next_token(logits, greedy=True, generator=generator)
    cold = sample_next_token(logits, temperature=0.0, generator=generator)
    assert greedy.tolist() == [1]
    assert cold.tolist() == [1]


def test_temperature_changes_sampling_sharpness() -> None:
    logits = torch.tensor([2.0, 1.0, 0.0])
    draws = 400

    def top_token_frequency(temperature: float) -> float:
        generator = torch.Generator().manual_seed(7)
        samples = [
            int(
                sample_next_token(
                    logits,
                    temperature=temperature,
                    generator=generator,
                )
            )
            for _ in range(draws)
        ]
        return samples.count(0) / draws

    sharp = top_token_frequency(0.25)
    flat = top_token_frequency(5.0)
    assert sharp > 0.95
    assert flat < 0.7
    assert sharp > flat


def test_sampling_with_a_generator_is_reproducible() -> None:
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    first = sample_next_token(logits, generator=torch.Generator().manual_seed(11))
    second = sample_next_token(logits, generator=torch.Generator().manual_seed(11))
    assert first.item() == second.item()


def test_sampling_handles_batches() -> None:
    logits = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
    token = sample_next_token(logits, greedy=True)
    assert token.shape == (2, 1)
    assert token.flatten().tolist() == [1, 0]


def test_top_k_one_equals_greedy() -> None:
    logits = torch.tensor([0.2, 3.0, 1.0])
    generator = torch.Generator().manual_seed(3)
    token = sample_next_token(logits, top_k=1, generator=generator)
    assert token.item() == 1


def test_invalid_sampling_parameters_raise() -> None:
    logits = torch.zeros(4)
    with pytest.raises(ValueError):
        sample_next_token(logits, temperature=-1.0)
    with pytest.raises(ValueError):
        sample_next_token(logits, top_k=-1)
    with pytest.raises(ValueError):
        sample_next_token(logits, top_p=0.0)
    with pytest.raises(ValueError):
        top_k_filter(logits, -2)
    with pytest.raises(ValueError):
        top_p_filter(logits, 1.5)


def test_generate_greedy_appends_argmax_tokens() -> None:
    model = FixedLM([0.0, 1.0, 3.0])
    result = generate(model, [5], max_new_tokens=3, greedy=True)
    assert result == [5, 2, 2, 2]


def test_generate_stops_at_eos() -> None:
    model = FixedLM([0.0, 1.0, 3.0])
    result = generate(model, [4], max_new_tokens=10, greedy=True, eos_id=2)
    assert result == [4, 2]


def test_generate_crops_to_context_length() -> None:
    model = FixedLM([0.0, 1.0, 3.0], context_length=4)
    result = generate(model, [1, 2, 3, 4, 5, 6], max_new_tokens=2, greedy=True)
    assert len(result) == 8
    assert model.seen_lengths == [4, 4]


def test_generate_seed_is_reproducible() -> None:
    model = FixedLM([0.1, 0.1, 0.1])
    first = generate(model, [0], max_new_tokens=8, temperature=1.0, seed=123)
    second = generate(model, [0], max_new_tokens=8, temperature=1.0, seed=123)
    assert first == second


def test_generate_validation() -> None:
    model = FixedLM([0.0, 1.0])
    with pytest.raises(ValueError):
        generate(model, [], max_new_tokens=1)
    with pytest.raises(ValueError):
        generate(model, [0], max_new_tokens=-1)
    with pytest.raises(ValueError):
        generate(model, [0], seed=1, generator=torch.Generator())
    with pytest.raises(ValueError):
        generate(model, [0], max_new_tokens=1, top_p=0.0)


def test_generate_text_round_trips_through_tokenizer() -> None:
    text = "der die das ein eine und oder mit von zu den dem des " * 30
    bundle = train_bundle(text, vocab_size=60, min_frequency=1, max_token_length=4)
    model = MiniGPT(
        vocab_size=bundle.bpe.vocab_size,
        context_length=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        tie_weights=False,
    )
    out = generate_text(model, bundle, "der", max_new_tokens=5, greedy=True)
    assert isinstance(out, str)
    assert out.startswith("der")

    # An empty prompt starts from <bos> rather than raising.
    empty = generate_text(model, bundle, "", max_new_tokens=3, greedy=True)
    assert isinstance(empty, str)


def test_sampling_probabilities_are_normalised_after_filtering() -> None:
    # Not exported, but exercising the private helper guards the maths.
    from core.gpt.sampling import _filtered_logits

    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    filtered = _filtered_logits(
        logits, temperature=1.0, top_k=2, top_p=0.9, greedy=False
    )
    probs = torch.softmax(filtered, dim=-1)
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-6)
    assert torch.isfinite(filtered).sum().item() == 2
