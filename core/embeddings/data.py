"""Skip-gram pair generation and negative sampling (T-301).

The skip-gram objective does not train on whole documents.  It trains on
**(center, context) pairs**: every token in the corpus is a *center*, and
every token inside a small window around it is a *context*::

    text:    Die Straße war frei
    tokens:  [Die] [Straße] [war] [frei]
    center "Straße", window 2:
        (Straße, Die)   (Straße, war)   (Straße, frei)

A model that sees the center must score the observed context tokens high.
Doing that alone has a trivial solution: give every pair a high score.  So
each positive pair is contrasted with a handful of random **negative**
tokens, drawn from the unigram distribution raised to the 3/4 power.  The
loss then pushes the true context up and the random distractors down.

This module builds that data pipeline:

* :func:`iter_skipgram_pairs` yields positive pairs from an id sequence.
  ``dynamic_window=True`` picks a random window in ``1..window_size`` per
  center (the word2vec trick: nearby pairs are sampled more often);
  ``False`` always uses the full window.  Both modes are deterministic for
  a given ``seed``.
* :class:`NegativeSampler` samples ids from ``counts ** 0.75``.  The
  exponent flattens the distribution, so frequent tokens still dominate
  but rare tokens are not starved of training signal.
* :class:`SkipGramDataset` turns pairs into the batch dicts the shared
  Trainer consumes: ``{"center": (B,), "context": (B,),
  "negatives": (B, K)}``.  Negatives are derived from the pair index and
  the dataset seed, not from a global RNG, so an item's batch is identical
  no matter in which order the DataLoader visits items.
* :func:`build_skipgram_dataset` is the one-call convenience wrapper used
  by the phase factory and the explorer.

False negatives (drawing the true context as a "negative") are excluded:
on a small German vocabulary they are common enough to visibly hurt the
embeddings.  The sampler rejects the excluded ids and redraws.

Example::

    dataset = build_skipgram_dataset(
        token_ids, vocab_size=300, window_size=3, negatives=5, seed=42
    )
    batch = next(iter(DataLoader(dataset, batch_size=32)))
    assert batch["negatives"].shape == (32, 5)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "PairOptions",
    "SkipGramDataset",
    "NegativeSampler",
    "build_skipgram_dataset",
    "iter_skipgram_pairs",
    "pairs_to_dataset",
]


@dataclass(frozen=True)
class PairOptions:
    """Options for the positive-pair generator.

    Attributes
    ----------
    window_size:
        Maximum distance (in tokens) between center and context.
    dynamic_window:
        When true, each center draws a window in ``1..window_size``.  This
        weights nearby pairs more heavily and is the standard word2vec
        behaviour.  When false, every center sees the full window.
    seed:
        Seed for the dynamic-window draws.  Ignored when ``dynamic_window``
        is false; the pair order itself is always deterministic.
    """

    window_size: int = 5
    dynamic_window: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool):
            raise ValueError("window_size must be an int")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an int")


def iter_skipgram_pairs(
    token_ids: Sequence[int],
    *,
    window_size: int = 5,
    dynamic_window: bool = False,
    seed: int = 42,
) -> Iterator[tuple[int, int]]:
    """Yield ``(center, context)`` pairs from a sequence of token ids.

    Pairs never cross the sequence boundaries: the first token has only
    right-hand context, the last only left-hand context.  The same id may
    appear on both sides of a pair; a token is never paired with itself.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    n = len(token_ids)
    if n < 2:
        return
    rng = random.Random(seed)
    for center_index, center in enumerate(token_ids):
        width = rng.randint(1, window_size) if dynamic_window else window_size
        start = max(0, center_index - width)
        end = min(n, center_index + width + 1)
        for context_index in range(start, end):
            if context_index != center_index:
                yield int(center), int(token_ids[context_index])


class NegativeSampler:
    """Draw negative token ids from ``counts ** power``.

    Parameters
    ----------
    counts:
        Per-token occurrence counts; index == token id, ``len(counts)`` ==
        vocabulary size.  Zero-count tokens are never sampled.
    power:
        Distribution exponent, normally 0.75 (word2vec's choice).  ``1.0``
        samples proportional to frequency, ``0.0`` samples uniformly from
        the seen tokens.
    seed:
        Seed for the sampler's own generator.  Callers that need a stable
        per-example result should pass their own ``generator`` to
        :meth:`sample`.
    """

    def __init__(
        self,
        counts: Sequence[int] | np.ndarray,
        *,
        power: float = 0.75,
        seed: int = 0,
    ) -> None:
        counts_array = np.asarray(counts, dtype=np.float64)
        if counts_array.ndim != 1 or counts_array.size == 0:
            raise ValueError("counts must be a non-empty 1-D sequence")
        if np.any(counts_array < 0):
            raise ValueError("counts must be non-negative")
        if not 0.0 <= power <= 1.0:
            raise ValueError("power must be between 0 and 1")
        weights = np.power(counts_array, power)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("counts must contain at least one positive value")
        probabilities = weights / total
        self._probabilities = torch.from_numpy(probabilities.astype(np.float32))
        self._counts = counts_array
        self._power = float(power)
        self._generator = torch.Generator().manual_seed(int(seed))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def probabilities(self) -> np.ndarray:
        """The normalized sampling distribution (a copy)."""
        return self._probabilities.numpy().astype(np.float64, copy=True)

    @property
    def power(self) -> float:
        return self._power

    @property
    def vocab_size(self) -> int:
        return int(self._counts.size)

    def __repr__(self) -> str:
        return (
            f"NegativeSampler(vocab_size={self.vocab_size}, "
            f"power={self._power:g})"
        )

    @classmethod
    def from_token_ids(
        cls,
        token_ids: Sequence[int] | Iterable[int],
        *,
        vocab_size: int | None = None,
        power: float = 0.75,
        seed: int = 0,
    ) -> "NegativeSampler":
        """Build a sampler from a token stream.

        ``vocab_size`` defaults to ``max(token_ids) + 1``; pass it explicitly
        when the tokenizer's vocabulary is larger than the ids that happen
        to occur in this corpus.
        """
        ids = np.asarray(list(token_ids), dtype=np.int64)
        if ids.size and ids.min() < 0:
            raise ValueError("token ids must be non-negative")
        if vocab_size is None:
            if ids.size == 0:
                raise ValueError("vocab_size is required when token_ids is empty")
            size = int(ids.max()) + 1
        else:
            size = int(vocab_size)
        if size < 1:
            raise ValueError("vocab_size must be >= 1")
        if ids.size and int(ids.max()) >= size:
            raise ValueError(
                f"token id {int(ids.max())} is outside vocab_size={size}"
            )
        counts = np.bincount(ids, minlength=size) if ids.size else np.zeros(size)
        return cls(counts, power=power, seed=seed)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
        exclude: Iterable[int] | int | None = None,
    ) -> torch.Tensor:
        """Draw ``num_samples`` negative ids (with replacement).

        ``exclude`` ids (e.g. the true context of the pair) are replaced by
        fresh draws, so a positive pair is not also treated as a negative.
        Raises :class:`ValueError` when every non-zero token is excluded.
        """
        if num_samples < 0:
            raise ValueError("num_samples must be >= 0")
        if num_samples == 0:
            return torch.empty(0, dtype=torch.long)

        excluded: set[int] = set()
        if exclude is not None:
            if isinstance(exclude, int):
                excluded.add(exclude)
            else:
                excluded.update(int(token) for token in exclude)
        if excluded:
            allowed = self._probabilities.clone()
            for token in excluded:
                if 0 <= token < allowed.numel():
                    allowed[token] = 0.0
            if float(allowed.sum()) <= 0.0:
                raise ValueError(
                    "cannot sample negatives: every seen token is excluded"
                )
            probabilities = allowed / allowed.sum()
        else:
            probabilities = self._probabilities

        source = generator if generator is not None else self._generator
        samples = torch.multinomial(
            probabilities, int(num_samples), replacement=True, generator=source
        )
        return samples.to(torch.long)


class SkipGramDataset(Dataset):
    """A fixed list of positive pairs with on-demand negative sampling.

    Each item is a batch-ready dict of scalar tensors::

        {"center": tensor(...), "context": tensor(...),
         "negatives": tensor([...])}   # shape (negatives,)

    The DataLoader stacks these into ``(B,)``, ``(B,)`` and ``(B, K)``.
    Negatives for pair ``i`` use ``seed * 1_000_003 + i``, so the result is
    stable across DataLoader order, epochs, and process restarts.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[int, int]] | np.ndarray,
        *,
        vocab_size: int,
        sampler: NegativeSampler,
        negatives: int = 5,
        seed: int = 42,
    ) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if negatives < 1:
            raise ValueError("negatives must be >= 1")
        if sampler.vocab_size != vocab_size:
            raise ValueError(
                f"sampler.vocab_size={sampler.vocab_size} does not match "
                f"vocab_size={vocab_size}"
            )
        if isinstance(pairs, np.ndarray):
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("pairs array must have shape (N, 2)")
            pair_list = [(int(c), int(x)) for c, x in pairs]
        else:
            pair_list = [(int(c), int(x)) for c, x in pairs]
        for center, context in pair_list:
            if not (0 <= center < vocab_size and 0 <= context < vocab_size):
                raise ValueError(
                    f"pair {(center, context)} is outside vocab_size={vocab_size}"
                )
        self._pairs = pair_list
        self._vocab_size = int(vocab_size)
        self._sampler = sampler
        self._negatives = int(negatives)
        self._seed = int(seed)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return list(self._pairs)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def negatives(self) -> int:
        return self._negatives

    @property
    def seed(self) -> int:
        return self._seed

    def __len__(self) -> int:
        return len(self._pairs)

    def __repr__(self) -> str:
        return (
            f"SkipGramDataset(pairs={len(self._pairs)}, "
            f"vocab_size={self._vocab_size}, negatives={self._negatives})"
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        center, context = self._pairs[index]
        generator = torch.Generator().manual_seed(
            (self._seed * 1_000_003 + int(index)) % (2**63 - 1)
        )
        negatives = self._sampler.sample(
            self._negatives,
            generator=generator,
            exclude=(center, context),
        )
        return {
            "center": torch.tensor(center, dtype=torch.long),
            "context": torch.tensor(context, dtype=torch.long),
            "negatives": negatives,
        }


def pairs_to_dataset(
    pairs: Sequence[tuple[int, int]],
    *,
    vocab_size: int,
    token_counts: Sequence[int] | np.ndarray,
    negatives: int = 5,
    seed: int = 42,
    power: float = 0.75,
) -> SkipGramDataset:
    """Wrap already-generated pairs into a :class:`SkipGramDataset`."""
    sampler = NegativeSampler(token_counts, power=power, seed=seed)
    return SkipGramDataset(
        pairs,
        vocab_size=vocab_size,
        sampler=sampler,
        negatives=negatives,
        seed=seed,
    )


def build_skipgram_dataset(
    token_ids: Sequence[int],
    *,
    vocab_size: int,
    window_size: int = 5,
    dynamic_window: bool = True,
    negatives: int = 5,
    seed: int = 42,
    power: float = 0.75,
    max_pairs: int = 0,
) -> SkipGramDataset:
    """Generate positive pairs and bind them to a negative sampler.

    Parameters
    ----------
    token_ids:
        Corpus encoded as a flat id sequence (special tokens should already
        be removed by the caller).
    vocab_size:
        Number of ids the model's embedding tables cover.
    window_size / dynamic_window:
        See :class:`PairOptions`.
    negatives:
        Distractors per positive pair (word2vec's usual 5–20).
    seed:
        Seed for dynamic windows, pair subsampling, and negative draws.
    power:
        Negative-sampling exponent (0.75 is the word2vec default).
    max_pairs:
        If positive, uniformly subsample at most this many pairs.  ``0``
        keeps all pairs.  Useful for keeping the tiny profiles fast.

    Raises :class:`ValueError` for ids outside ``[0, vocab_size)`` or an
    empty token stream.
    """
    ids = np.asarray(list(token_ids), dtype=np.int64)
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")
    if ids.size == 0:
        raise ValueError("token_ids must not be empty")
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError(
            f"token ids must be in [0, {vocab_size}); "
            f"got [{int(ids.min())}, {int(ids.max())}]"
        )

    pairs = list(
        iter_skipgram_pairs(
            ids,
            window_size=window_size,
            dynamic_window=dynamic_window,
            seed=seed,
        )
    )
    if max_pairs and len(pairs) > max_pairs:
        chosen = sorted(random.Random(seed).sample(range(len(pairs)), max_pairs))
        pairs = [pairs[index] for index in chosen]
    if not pairs:
        raise ValueError(
            "no skip-gram pairs produced; the corpus needs at least two tokens"
        )

    counts = np.bincount(ids, minlength=vocab_size)
    return pairs_to_dataset(
        pairs,
        vocab_size=vocab_size,
        token_counts=counts,
        negatives=negatives,
        seed=seed,
        power=power,
    )
