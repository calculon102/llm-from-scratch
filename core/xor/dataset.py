"""XOR dataset for the M1 learning module (T-101).

The XOR task is the classic "not linearly separable" toy problem:

    x1  x2  |  y
    0   0   |  0
    0   1   |  1
    1   0   |  1
    1   1   |  0

No single straight line (hyperplane) can separate the two classes, so a
single linear layer can never solve it; adding one hidden layer makes it
learnable.

:class:`XORDataset` repeats the four canonical points ``repeats`` times so
that the Trainer's train/val split (T-005) has enough samples, and
optionally adds Gaussian input noise so the model has to learn a margin
rather than memorize the four exact points.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

__all__ = ["XOR_POINTS", "XORDataset"]

#: The four canonical XOR input/label pairs, (x1, x2, y).
XOR_POINTS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
)


class XORDataset(Dataset):
    """XOR training data: the four canonical points, repeated ``repeats`` times.

    Each item is a dict of tensors ``{"x": (2,), "y": (1,)}`` — the same
    dict-of-tensors batch convention the shared Trainer (T-005) uses for
    every phase.

    Parameters
    ----------
    repeats:
        How many times each of the four points is repeated (default 100
        → 400 samples).
    noise_std:
        Standard deviation of Gaussian noise added to the inputs (default
        0.0 → the exact canonical points).
    seed:
        Seed for the noise RNG; the dataset is deterministic for a seed.
    """

    def __init__(self, repeats: int = 100, noise_std: float = 0.0, seed: int = 0) -> None:
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        # Interleave the four points so the dataset cycles through them
        # (rather than in blocks): row i uses point i % 4.
        points = [XOR_POINTS[i % len(XOR_POINTS)] for i in range(4 * repeats)]
        self._x = torch.tensor([p[:2] for p in points], dtype=torch.float32)
        self._y = torch.tensor([[p[2]] for p in points], dtype=torch.float32)
        if noise_std > 0:
            gen = torch.Generator().manual_seed(seed)
            self._x = self._x + noise_std * torch.randn(self._x.shape, generator=gen)

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self._x[idx], "y": self._y[idx]}
