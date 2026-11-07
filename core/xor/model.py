"""2-layer MLP for the XOR task (T-101).

Architecture::

    x (2,) ──> Linear(2, hidden) ──> activation ──> Linear(hidden, 1) ──> logits (1,)

One hidden layer is the minimal fix for non-linear separability: each
hidden unit computes a half-plane (a linear threshold of the input), and
the output layer draws a straight line *in the hidden feature space*.
XOR is representable with as few as two hidden units, e.g.::

    y = ReLU(x1 - x2) + ReLU(x2 - x1)      # = |x1 - x2| = XOR on {0,1}**2

The optimizer finds *some* separating solution (which one depends on the
initialization); ``hidden_size`` above the minimum of 2 makes training
robust across seeds.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ACTIVATIONS", "XorMLP"]

#: Supported hidden-layer activations.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


class XorMLP(nn.Module):
    """2-layer MLP: input -> hidden (activation) -> output (logits).

    The output is a *logit* (pre-sigmoid), intended to be paired with
    ``nn.BCEWithLogitsLoss`` (numerically stable: it fuses the sigmoid
    and the binary cross-entropy into one op).

    Parameters
    ----------
    in_dim:
        Input dimension (2 for XOR).
    hidden_size:
        Number of hidden units (default 8; the minimum workable is 2).
    activation:
        Hidden-layer activation: ``"tanh"`` (default) or ``"relu"``.
    """

    def __init__(
        self, in_dim: int = 2, hidden_size: int = 8, activation: str = "tanh"
    ) -> None:
        super().__init__()  # nn.Module registry must exist before self.net
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; "
                f"expected one of {list(ACTIVATIONS)}"
            )
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            ACTIVATIONS[activation](),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of inputs ``(..., in_dim)`` to logits ``(..., 1)``."""
        return self.net(x)
