"""2-layer MLP for the XOR task (T-101).

What a neural network *is* (the whole of the architecture)
-----------------------------------------------------------

Every neuron in a layer does the same tiny thing: it takes numbers in,
computes a **weighted sum** plus a bias, and hands the result to the next
layer::

    neuron_i = w_1 * x_1 + w_2 * x_2 + b_i

Nothing more.  A "layer" is just a matrix of those weights, so a Linear
layer is literally a matrix multiplication — that is why neural nets are
so fast to run on GPUs.  What makes the network non-linear (and therefore
able to solve XOR at all) is the **activation function** squashed in
between the two linear layers: tanh (or ReLU) bends the straight-line
mapping, so the composition of two bent half-planes can draw the
checkered XOR pattern.

Why XOR needs a hidden layer, in geometric terms:

* A single linear layer can only draw **one straight line** through the
  input space (the set of points where the output is zero).  XOR wants a
  checkerboard — two diagonal corners class 0, the other two class 1.
  No single line (or half-plane) can carve that out; provably impossible.
* Each *hidden unit* computes one half-plane (a linear threshold of the
  input), and the output layer then draws a straight line **in the hidden
  feature space** — a space where the four XOR points *are* linearly
  separable.  Two hidden units already suffice, e.g.::

      y = ReLU(x1 - x2) + ReLU(x2 - x1)     # = |x1 - x2| = XOR on {0,1}**2

  one unit detects "x2 > x1", the other "x1 > x2"; their sum is XOR.

How *training* finds such a wiring: the weights start as random numbers
so the network's first answers are garbage.  The loss (see
:mod:`core.xor`) measures how wrong the answers are; the gradient tells
us, for every single weight, which way to nudge it to make the loss
smaller.  The optimizer repeats ``W -= lr * grad`` thousands of times —
a walk downhill through the loss landscape — until the four points fall
on the right sides of the learned boundaries.  Which exact solution the
optimizer settles on depends on the random initialization; choosing
``hidden_size`` above the minimum of 2 leaves slack that makes training
robust across seeds.

Architecture::

    x (2,) ──> Linear(2, hidden) ──> activation ──> Linear(hidden, 1) ──> logits (1,)
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
        # nn.Linear initializes its weights randomly (Kaiming-uniform) and
        # the bias to zero.  The random start is essential, not an accident:
        # (a) it breaks symmetry — if all hidden units started with the
        #     same weights they would all compute the identical thing and
        #     never specialize (gradient descent can only *differentiate*
        #     units that already differ); (b) it gives the optimizer a
        #     concrete starting point on the loss landscape to walk downhill
        #     from.  That random start is exactly what the finite-difference
        #     gradient check (T-103) verifies step by step.
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),   # W1 (hidden x in_dim) + b1: one half-plane per hidden unit
            ACTIVATIONS[activation](),        # the bend that makes the composition non-linear
            nn.Linear(hidden_size, 1),        # W2 (1 x hidden) + b2: a straight line in hidden space
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of inputs ``(..., in_dim)`` to logits ``(..., 1)``.

        This is the *forward pass*: for a batch of shape ``(B, in_dim)``
        the computation is two matrix multiplies and one activation,
        producing ``B`` logits.  The logit is the pre-sigmoid score — a
        positive logit means "network leans toward 1", negative toward 0;
        the pair :func:`xor_loss`/``BCEWithLogitsLoss`` turns it into a
        probability and a loss.
        """
        return self.net(x)
