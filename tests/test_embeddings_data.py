"""Tests for the skip-gram pair generator and negative sampler (T-301).

The acceptance criterion is a "stable configurable batch pipeline".  These
tests pin down all three words: the positive pairs are exactly the ones a
hand-computed window produces, the same seed reproduces the same batches,
and the configurable knobs (window, dynamic window, negatives, max_pairs)
change the output in the expected way.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.embeddings.data import (
    NegativeSampler,
    PairOptions,
    SkipGramDataset,
    build_skipgram_dataset,
    iter_skipgram_pairs,
    pairs_to_dataset,
)


# ---------------------------------------------------------------------------
# Positive pairs
# ---------------------------------------------------------------------------

def test_fixed_window_pairs_are_hand_computable() -> None:
    pairs = list(iter_skipgram_pairs([0, 1, 2, 3, 4], window_size=1))
    assert pairs == [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
    ]


def test_window_is_clipped_at_sequence_boundaries() -> None:
    pairs = list(iter_skipgram_pairs([7, 8, 9], window_size=10))
    assert pairs == [
        (7, 8),
        (7, 9),
        (8, 7),
        (8, 9),
        (9, 7),
        (9, 8),
    ]
    # No self-pairs, no duplicates.
    assert all(center != context for center, context in pairs)
    assert len(pairs) == len(set(pairs))


def test_dynamic_window_is_reproducible_and_bounded() -> None:
    tokens = list(range(20))
    first = list(iter_skipgram_pairs(tokens, window_size=4, dynamic_window=True, seed=7))
    second = list(iter_skipgram_pairs(tokens, window_size=4, dynamic_window=True, seed=7))
    other = list(iter_skipgram_pairs(tokens, window_size=4, dynamic_window=True, seed=8))

    assert first == second
    assert first != other, "different seeds should produce different windows"
    # Every generated pair respects the maximum window.
    positions: dict[int, list[int]] = {}
    for index, token in enumerate(tokens):
        positions.setdefault(token, []).append(index)
    for center, context in first:
        assert abs(positions[center][0] - positions[context][0]) <= 4


def test_short_or_empty_sequences_produce_no_pairs() -> None:
    assert list(iter_skipgram_pairs([], window_size=3)) == []
    assert list(iter_skipgram_pairs([5], window_size=3)) == []


def test_pair_options_validate() -> None:
    assert PairOptions().window_size == 5
    with pytest.raises(ValueError):
        PairOptions(window_size=0)
    with pytest.raises(ValueError):
        list(iter_skipgram_pairs([1, 2], window_size=0))


# ---------------------------------------------------------------------------
# Negative sampler
# ---------------------------------------------------------------------------

def test_negative_sampler_follows_the_three_quarter_power() -> None:
    # Token 0 is 100x more frequent than tokens 1 and 2.  With the 0.75
    # exponent its probability is 100**0.75 / (100**0.75 + 1 + 1) ≈ 0.9405,
    # far below the raw frequency of 0.98 but still dominant.
    sampler = NegativeSampler([100, 1, 1], power=0.75, seed=0)
    assert sampler.probabilities == pytest.approx(
        [0.9405165, 0.0297417, 0.0297417], rel=1e-4
    )

    samples = sampler.sample(20_000, generator=torch.Generator().manual_seed(0))
    frequencies = np.bincount(samples.numpy(), minlength=3) / len(samples)
    assert frequencies[0] > 0.93
    assert frequencies[1] > 0.005 and frequencies[2] > 0.005


def test_negative_sampler_never_draws_zero_count_tokens() -> None:
    sampler = NegativeSampler([5, 0, 5], power=0.75, seed=0)
    samples = sampler.sample(2000, generator=torch.Generator().manual_seed(1))
    assert set(samples.tolist()) == {0, 2}


def test_negative_sampler_excludes_the_positive_context() -> None:
    sampler = NegativeSampler([1, 1, 1], seed=0)
    samples = sampler.sample(
        1000, generator=torch.Generator().manual_seed(3), exclude=1
    )
    assert 1 not in set(samples.tolist())


def test_negative_sampler_from_token_ids_uses_vocab_size() -> None:
    sampler = NegativeSampler.from_token_ids([0, 0, 2, 2, 2], vocab_size=5)
    assert sampler.vocab_size == 5
    # Ids 1, 3 and 4 never occurred, so they are never sampled.
    assert set(sampler.sample(500, generator=torch.Generator().manual_seed(0)).tolist()) <= {0, 2}


def test_negative_sampler_validates_arguments() -> None:
    with pytest.raises(ValueError):
        NegativeSampler([])
    with pytest.raises(ValueError):
        NegativeSampler([0, 0])
    with pytest.raises(ValueError):
        NegativeSampler([1, 2], power=1.5)
    with pytest.raises(ValueError):
        NegativeSampler([1, 2]).sample(10, exclude={0, 1})


# ---------------------------------------------------------------------------
# Dataset / batch pipeline
# ---------------------------------------------------------------------------

def test_dataset_batch_contract_and_shapes() -> None:
    dataset = build_skipgram_dataset(
        [0, 1, 2, 3, 0, 1, 2, 3],
        vocab_size=4,
        window_size=2,
        negatives=5,
        seed=42,
    )
    assert len(dataset) > 0
    assert dataset.negatives == 5

    loader = torch.utils.data.DataLoader(dataset, batch_size=6)
    batch = next(iter(loader))
    assert set(batch) == {"center", "context", "negatives"}
    assert batch["center"].shape == (6,)
    assert batch["context"].shape == (6,)
    assert batch["negatives"].shape == (6, 5)
    assert batch["negatives"].dtype == torch.long
    assert int(batch["negatives"].min()) >= 0
    assert int(batch["negatives"].max()) < 4


def test_dataset_items_are_stable_across_visit_order() -> None:
    dataset = build_skipgram_dataset(
        [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        vocab_size=6,
        window_size=2,
        negatives=4,
        seed=123,
    )
    forwards = [dataset[i] for i in range(len(dataset))]
    backwards = [dataset[i] for i in reversed(range(len(dataset)))][::-1]
    for left, right in zip(forwards, backwards):
        assert torch.equal(left["negatives"], right["negatives"])
        assert left["center"].item() == right["center"].item()
        assert left["context"].item() == right["context"].item()


def test_dataset_does_not_use_the_true_context_as_a_negative() -> None:
    dataset = build_skipgram_dataset(
        list(range(5)) * 4,
        vocab_size=5,
        window_size=2,
        negatives=4,
        seed=0,
    )
    for index in range(len(dataset)):
        item = dataset[index]
        assert item["context"].item() not in set(item["negatives"].tolist())
        assert item["center"].item() not in set(item["negatives"].tolist())


def test_max_pairs_caps_the_dataset_deterministically() -> None:
    tokens = list(range(10)) * 3
    full = build_skipgram_dataset(tokens, vocab_size=10, window_size=3, seed=1)
    capped = build_skipgram_dataset(
        tokens, vocab_size=10, window_size=3, seed=1, max_pairs=7
    )
    capped_again = build_skipgram_dataset(
        tokens, vocab_size=10, window_size=3, seed=1, max_pairs=7
    )
    assert len(full) > 7
    assert len(capped) == 7
    assert capped.pairs == capped_again.pairs


def test_build_validates_ids_and_vocab_size() -> None:
    with pytest.raises(ValueError):
        build_skipgram_dataset([], vocab_size=4)
    with pytest.raises(ValueError):
        build_skipgram_dataset([0, 4], vocab_size=4)
    with pytest.raises(ValueError):
        build_skipgram_dataset([0, 1], vocab_size=4, negatives=0)


def test_pairs_to_dataset_wraps_existing_pairs() -> None:
    dataset = pairs_to_dataset(
        [(0, 1), (1, 0)],
        vocab_size=3,
        token_counts=[10, 5, 1],
        negatives=2,
        seed=0,
    )
    assert isinstance(dataset, SkipGramDataset)
    assert dataset.pairs == [(0, 1), (1, 0)]
    item = dataset[0]
    assert item["negatives"].shape == (2,)
