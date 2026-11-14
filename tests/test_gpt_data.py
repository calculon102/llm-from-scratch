"""Tests for LM chunking and the train/validation split (T-403).

Acceptance criterion: "configurable context batches produced".  These tests
verify the window arithmetic, the input/target shift, the collated batch
shapes, and that the validation suffix shares no tokens with training.
"""

from __future__ import annotations

import pytest
import torch

from core.gpt.data import (
    LMChunkDataset,
    build_lm_datasets,
    chunk_starts,
    make_lm_dataloader,
    split_token_ids,
)


def test_chunk_starts_respects_stride() -> None:
    # 10 tokens, context 4: full windows start at 0,4 (stride 4) and
    # 0,2,4 (stride 2); a stride larger than the context skips tokens.
    assert chunk_starts(10, 4, stride=4) == [0, 4]
    assert chunk_starts(10, 4, stride=2) == [0, 2, 4]
    assert chunk_starts(10, 4, stride=6) == [0]
    assert chunk_starts(4, 4) == []
    assert chunk_starts(5, 4) == [0]


def test_dataset_returns_shifted_context_windows() -> None:
    ids = list(range(20))
    dataset = LMChunkDataset(ids, context_length=4, stride=4)

    assert len(dataset) == 4
    first = dataset[0]
    assert torch.equal(first["input_ids"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(first["targets"], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(dataset[1]["input_ids"], torch.tensor([4, 5, 6, 7]))

    for index in range(len(dataset)):
        item = dataset[index]
        assert torch.equal(item["input_ids"][1:], item["targets"][:-1])


def test_overlapping_windows_shift_by_stride() -> None:
    dataset = LMChunkDataset(list(range(12)), context_length=4, stride=2)
    assert len(dataset) == 4
    assert torch.equal(dataset[0]["input_ids"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(dataset[1]["input_ids"], torch.tensor([2, 3, 4, 5]))


def test_split_is_a_contiguous_disjoint_suffix() -> None:
    train_ids, val_ids, split = split_token_ids(list(range(100)), 0.2)
    assert split == 80
    assert torch.equal(train_ids, torch.arange(80))
    assert torch.equal(val_ids, torch.arange(80, 100))
    assert set(train_ids.tolist()).isdisjoint(val_ids.tolist())


def test_build_lm_datasets_splits_before_chunking() -> None:
    data = build_lm_datasets(
        list(range(100)), context_length=8, stride=4, val_split=0.2
    )
    assert data.split_index == 80
    assert data.train_windows == 18  # starts 0,4,...,68
    assert data.val_windows == 3  # starts 0,4,8 within the 20-token suffix
    train_tokens = set(data.train_dataset.token_ids.tolist())
    val_tokens = set(data.val_dataset.token_ids.tolist())
    assert train_tokens.isdisjoint(val_tokens)


def test_no_validation_split_when_val_split_is_zero() -> None:
    data = build_lm_datasets(list(range(40)), context_length=8, val_split=0.0)
    assert data.val_dataset is None
    assert data.val_windows == 0
    assert data.split_index == 40


def test_dataloader_produces_configurable_context_batches() -> None:
    data = build_lm_datasets(list(range(200)), context_length=16, val_split=0.1)
    loader = make_lm_dataloader(data.train_dataset, batch_size=8)

    batch = next(iter(loader))
    assert batch["input_ids"].shape == (8, 16)
    assert batch["targets"].shape == (8, 16)
    assert batch["input_ids"].dtype == torch.long
    assert torch.equal(batch["input_ids"][:, 1:], batch["targets"][:, :-1])

    val_batch = next(iter(make_lm_dataloader(data.val_dataset, batch_size=4)))
    assert val_batch["input_ids"].shape[0] <= 4
    assert val_batch["input_ids"].shape[1] == 16


def test_validation_split_must_leave_a_full_window() -> None:
    with pytest.raises(ValueError, match="validation split"):
        build_lm_datasets(list(range(20)), context_length=10, val_split=0.1)


def test_training_split_must_leave_a_full_window() -> None:
    with pytest.raises(ValueError, match="training split"):
        build_lm_datasets(list(range(20)), context_length=10, val_split=0.5)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="context_length"):
        LMChunkDataset([1, 2, 3], context_length=0)
    with pytest.raises(ValueError, match="stride"):
        LMChunkDataset([1, 2, 3], context_length=2, stride=0)
    with pytest.raises(ValueError, match="1-D"):
        LMChunkDataset([[1, 2], [3, 4]], context_length=1)
    with pytest.raises(ValueError, match="empty"):
        LMChunkDataset([], context_length=1)
    with pytest.raises(ValueError, match="non-negative"):
        LMChunkDataset([0, -1, 2], context_length=2)
    with pytest.raises(ValueError, match="val_split"):
        split_token_ids([1, 2, 3], 1.0)
    with pytest.raises(ValueError, match="batch_size"):
        make_lm_dataloader(LMChunkDataset(list(range(10)), context_length=2), batch_size=0)
