"""Language-model data: turn a token stream into next-token batches (T-403).

A causal language model is trained on one task: **given tokens 0..t,
predict token t+1**.  The corpus is therefore not split into labelled
examples by hand; it is cut into overlapping *windows*::

    token ids:  10 11 12 13 14 15 16 17 ...
    window 0:   [10 11 12 13]  ->  [11 12 13 14]
    window 1:   [14 15 16 17]  ->  [15 16 17 18]     (stride = 4)

Every position in the window is a training example: the model's output at
position ``t`` is compared with the token at ``t+1``.  With
``context_length`` tokens the model sees ``context_length`` prediction
problems at once, which is why batches are shaped ``(B, T)`` in and
``(B, T, vocab_size)`` out.

Two important details
---------------------
* **The split happens before chunking.**  If we chunked first and split
  the chunks randomly, neighbouring chunks would share almost all of
  their tokens and the validation loss would be optimistic.  We hold out
  a contiguous *suffix* of the token stream, then chunk train and
  validation independently.
* **The stride controls overlap.**  ``stride == context_length`` gives
  non-overlapping windows (the nanoGPT default and the cheapest choice).
  Smaller strides produce more windows and more overlap; a value larger
  than ``context_length`` skips tokens.  The last partial window is
  dropped rather than padded, so no loss is computed on padding.

Example::

    ids = list(range(1000))
    data = build_lm_datasets(ids, context_length=64, val_split=0.1)
    batch = next(iter(make_lm_dataloader(data.train_dataset, batch_size=8)))
    assert batch["input_ids"].shape == (8, 64)
    assert batch["targets"].shape == (8, 64)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_STRIDE",
    "LMChunkDataset",
    "LMDatasets",
    "build_lm_datasets",
    "chunk_starts",
    "make_lm_dataloader",
    "split_token_ids",
]

#: Default number of tokens the model conditions on per prediction.
DEFAULT_CONTEXT_LENGTH = 64

#: ``None`` means "use ``context_length``", i.e. non-overlapping windows.
DEFAULT_STRIDE: int | None = None


def _as_token_tensor(token_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Return a validated 1-D ``torch.long`` tensor of token ids."""
    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be a 1-D sequence")
        ids = token_ids.detach().to("cpu", torch.long)
    else:
        ids = torch.as_tensor(list(token_ids), dtype=torch.long)
        if ids.dim() != 1:
            raise ValueError("token_ids must be a 1-D sequence")
    if ids.numel() == 0:
        raise ValueError("token_ids must not be empty")
    if int(ids.min()) < 0:
        raise ValueError("token_ids must be non-negative")
    return ids


def _resolve_stride(context_length: int, stride: int | None) -> int:
    if context_length < 1:
        raise ValueError("context_length must be >= 1")
    resolved = context_length if stride is None else int(stride)
    if resolved < 1:
        raise ValueError("stride must be >= 1")
    return resolved


def chunk_starts(
    num_tokens: int, context_length: int, stride: int | None = None
) -> list[int]:
    """Return the start indices of all full windows in ``num_tokens`` ids.

    A window needs ``context_length + 1`` tokens (context plus the target
    after it), so the last possible start is ``num_tokens - context_length
    - 1``.  Partial windows are dropped.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0")
    stride = _resolve_stride(context_length, stride)
    if num_tokens <= context_length:
        return []
    return list(range(0, num_tokens - context_length, stride))


class LMChunkDataset(Dataset):
    """A view over a token stream yielding next-token training windows.

    Each item is a dict of two ``torch.long`` tensors of shape
    ``(context_length,)``::

        {"input_ids": ids[start : start + T],
         "targets":   ids[start + 1 : start + T + 1]}

    The dataset keeps only one copy of the token stream and computes the
    windows lazily, so a long corpus does not explode into Python lists of
    tuples.
    """

    def __init__(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        stride: int | None = DEFAULT_STRIDE,
    ) -> None:
        self._ids = _as_token_tensor(token_ids)
        self._context_length = int(context_length)
        self._stride = _resolve_stride(self._context_length, stride)
        self._starts = chunk_starts(
            int(self._ids.numel()), self._context_length, self._stride
        )

    @property
    def token_ids(self) -> torch.Tensor:
        """The underlying (not copied) token stream."""
        return self._ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def stride(self) -> int:
        return self._stride

    def __len__(self) -> int:
        return len(self._starts)

    def __repr__(self) -> str:
        return (
            f"LMChunkDataset(tokens={self._ids.numel()}, "
            f"context_length={self._context_length}, stride={self._stride}, "
            f"windows={len(self)})"
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self._starts[index]
        window = self._ids[start : start + self._context_length + 1]
        return {
            "input_ids": window[:-1].clone(),
            "targets": window[1:].clone(),
        }


@dataclass(frozen=True)
class LMDatasets:
    """Train/validation chunk datasets plus the split metadata.

    ``split_index`` is the token offset where the validation suffix begins.
    ``train_dataset`` and ``val_dataset`` never share a window, and their
    token streams are disjoint, so a validation loss measures generalisation
    rather than memorisation of a neighbouring window.
    """

    train_dataset: LMChunkDataset
    val_dataset: LMChunkDataset | None
    context_length: int
    stride: int
    split_index: int
    total_tokens: int

    @property
    def train_windows(self) -> int:
        return len(self.train_dataset)

    @property
    def val_windows(self) -> int:
        return len(self.val_dataset) if self.val_dataset is not None else 0


def split_token_ids(
    token_ids: Sequence[int] | torch.Tensor, val_split: float
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Split a token stream into a train prefix and validation suffix.

    Returns ``(train_ids, val_ids_or_none, split_index)``.  The split is
    contiguous and deterministic: no shuffling means the validation text is
    genuinely held out, and rerunning with the same data gives the same
    split.  ``val_split`` is a fraction of *tokens*; ``0`` disables
    validation.
    """
    ids = _as_token_tensor(token_ids)
    if not isinstance(val_split, (int, float)) or isinstance(val_split, bool):
        raise ValueError("val_split must be a number")
    if not 0.0 <= float(val_split) < 1.0:
        raise ValueError("val_split must be in [0, 1)")
    if val_split == 0.0:
        return ids, None, int(ids.numel())

    num_val = max(1, int(round(ids.numel() * float(val_split))))
    split_index = int(ids.numel()) - num_val
    if split_index < 1:
        raise ValueError(
            f"val_split={val_split} leaves no training tokens "
            f"(n={ids.numel()})"
        )
    return ids[:split_index].clone(), ids[split_index:].clone(), split_index


def build_lm_datasets(
    token_ids: Sequence[int] | torch.Tensor,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    stride: int | None = DEFAULT_STRIDE,
    val_split: float = 0.1,
    seed: int = 42,
) -> LMDatasets:
    """Split a token stream and build train/validation chunk datasets.

    Parameters
    ----------
    token_ids:
        Flat corpus ids, typically produced by the M2 BPE tokenizer.
    context_length:
        Number of conditioning tokens per window (the model's ``T``).
    stride:
        Step between window starts; ``None`` means ``context_length``.
    val_split:
        Fraction of tokens held out as a contiguous validation suffix.
    seed:
        Accepted for API symmetry with the other phases.  The split itself
        is deterministic and does not consume randomness.

    Raises :class:`ValueError` when either split is too short to yield at
    least one full window.  A language-model split needs at least
    ``2 * (context_length + 1)`` tokens in total.
    """
    _ = seed  # the contiguous split needs no randomness
    stride = _resolve_stride(context_length, stride)
    train_ids, val_ids, split_index = split_token_ids(token_ids, val_split)

    train_dataset = LMChunkDataset(
        train_ids, context_length=context_length, stride=stride
    )
    if len(train_dataset) == 0:
        raise ValueError(
            f"training split has {train_ids.numel()} tokens, but a context "
            f"window needs {context_length + 1}"
        )

    val_dataset: LMChunkDataset | None = None
    if val_ids is not None:
        val_dataset = LMChunkDataset(
            val_ids, context_length=context_length, stride=stride
        )
        if len(val_dataset) == 0:
            raise ValueError(
                f"validation split has {val_ids.numel()} tokens, but a context "
                f"window needs {context_length + 1}; lower context_length or "
                "increase val_split"
            )

    return LMDatasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        context_length=int(context_length),
        stride=stride,
        split_index=split_index,
        total_tokens=train_dataset.token_ids.numel()
        + (val_dataset.token_ids.numel() if val_dataset is not None else 0),
    )


def make_lm_dataloader(
    dataset: LMChunkDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap an :class:`LMChunkDataset` in a ``DataLoader`` of dict batches.

    The default collate stacks the per-window dicts into
    ``{"input_ids": (B, T), "targets": (B, T)}``.  ``shuffle=False`` is the
    right choice for validation; the Trainer builds its own shuffled
    training loader.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
