"""XOR visualization: decision boundary, activations, loss curves (T-102/T-104).

Three figures, each answering a question the loss number alone cannot:

- :func:`decision_boundary_fig` — the model's ``P(y=1 | x)`` over the whole
  input plane, with the ``P = 0.5`` decision contour.  Shows *where* the
  model draws the line between the two classes.
- :func:`activations_fig` — each hidden unit's activation over the input
  plane.  Shows *how* the network decomposed XOR into half-plane features
  (after a few epochs, individual units light up on different half-planes;
  the output layer then combines them into the two XOR "quadrants").
- :func:`loss_curve_fig` (T-104) — training loss over steps, plus the
  per-epoch validation loss.  Shows *that the learning actually happened*:
  the curve starts at ``ln(2) ≈ 0.693`` (the network is maximally "surprised"
  by every label at random initialization) and should fall toward 0.  A flat
  curve is the classic symptom of a learning rate that is too small.

:func:`save_xor_plots` builds the model figures and
:func:`save_loss_curve` the training-history figure; all land in a run's
``plots/`` folder via the artifact manager (T-004).  Matplotlib is an
optional dependency (the ``viz`` extra); when it is missing, a warning is
logged and no plots are written, so training never fails because of
plotting.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn

from core.artifacts import RunArtifacts
from core.xor.dataset import XOR_POINTS

__all__ = [
    "decision_boundary_fig",
    "activations_fig",
    "save_xor_plots",
    "loss_curve_fig",
    "save_loss_curve",
]


def _probability_grid(
    model: nn.Module, grid_size: int, x_range: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate ``sigmoid(model(x))`` on a square grid of the input plane.

    Returns ``(g1, g2, probs)`` where ``g1``/``g2`` are the coordinate
    grids and ``probs`` has shape ``(grid_size, grid_size)``.
    """
    model.eval()
    with torch.no_grad():
        x1 = torch.linspace(x_range[0], x_range[1], grid_size)
        x2 = torch.linspace(x_range[0], x_range[1], grid_size)
        g1, g2 = torch.meshgrid(x1, x2, indexing="ij")
        points = torch.stack([g1.flatten(), g2.flatten()], dim=1)
        probs = torch.sigmoid(model(points)).reshape(grid_size, grid_size)
    return g1, g2, probs


def decision_boundary_fig(
    model: nn.Module,
    grid_size: int = 200,
    x_range: tuple[float, float] = (-0.5, 1.5),
) -> "object":
    """Figure: P(y=1) heatmap + the P=0.5 decision contour + the 4 points."""
    import matplotlib.pyplot as plt

    g1, g2, probs = _probability_grid(model, grid_size, x_range)
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(
        g1.numpy(), g2.numpy(), probs.numpy(), levels=24, cmap="RdYlBu_r", alpha=0.85
    )
    ax.contour(
        g1.numpy(), g2.numpy(), probs.numpy(),
        levels=[0.5], colors="black", linewidths=1.5,
    )
    xs = [p[0] for p in XOR_POINTS]
    ys = [p[1] for p in XOR_POINTS]
    labels = [p[2] for p in XOR_POINTS]
    ax.scatter(
        xs, ys, c=labels, cmap="RdYlBu_r", s=90, edgecolors="black", zorder=5
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("XOR decision boundary (color = P(y=1), black = P=0.5)")
    fig.colorbar(cf, ax=ax, label="P(y=1)")
    return fig


def activations_fig(
    model: nn.Module,
    grid_size: int = 100,
    x_range: tuple[float, float] = (-0.5, 1.5),
) -> "object":
    """Figure: one panel per hidden unit, its activation over the input plane."""
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        x1 = torch.linspace(x_range[0], x_range[1], grid_size)
        x2 = torch.linspace(x_range[0], x_range[1], grid_size)
        g1, g2 = torch.meshgrid(x1, x2, indexing="ij")
        points = torch.stack([g1.flatten(), g2.flatten()], dim=1)
        # The network's layers: for a XorMLP the ``.net`` Sequential holds
        # [Linear, activation, Linear]; a bare nn.Sequential (as used by
        # tests and future phases) *is* that stack, so fall back to the
        # module itself.  units 0 and 1 are input-Linear and activation,
        # so composing them gives every hidden unit's response to the
        # input plane — before the final linear layer has collapsed it.
        net = getattr(model, "net", model)
        acts = net[1](net[0](points)).reshape(grid_size, grid_size, -1)
    n_units = acts.shape[2]
    ncols = min(4, n_units)
    nrows = math.ceil(n_units / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False
    )
    extent = [x_range[0], x_range[1], x_range[0], x_range[1]]
    for i, ax in enumerate(axes.flat):
        if i >= n_units:
            ax.axis("off")
            continue
        im = ax.imshow(
            acts[:, :, i].numpy(), origin="lower", extent=extent, cmap="viridis"
        )
        ax.set_title(f"unit {i}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        if i >= ncols * (nrows - 1):
            ax.set_xlabel("x1", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("x2", fontsize=9)
    fig.suptitle("Hidden-layer activations (one panel per unit)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save_xor_plots(run: RunArtifacts, model: nn.Module) -> list[Path]:
    """Build both XOR figures and save them to ``run.plots_dir``.

    Returns the list of written PNG paths.  When matplotlib (the optional
    ``viz`` extra) is not installed, logs a warning and returns ``[]`` —
    plotting must never break training.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        run.log(
            "plots skipped: matplotlib not installed "
            '(install the viz extra: pip install -e ".[viz]")',
            level=logging.WARNING,
        )
        return []

    paths: list[Path] = []
    fig = decision_boundary_fig(model)
    paths.append(run.save_plot(fig, "decision_boundary"))
    plt.close(fig)
    fig = activations_fig(model)
    paths.append(run.save_plot(fig, "activations"))
    plt.close(fig)
    return paths


def loss_curve_fig(history: list[dict]) -> "object":
    """Figure: training loss over steps (solid) + per-epoch validation loss (dashed).

    ``history`` is a :meth:`JobRunner.history` snapshot (T-104): a list of
    dicts, some per-step (``{"step", "epoch", "loss", ...}``) and some
    per-epoch (``{"epoch", "step", "val_loss"|"train_loss", ...}``).

    How to read the result:

    * The **training** line is the loss the optimizer actually descends —
      one point per logged step.  It should start near ``ln(2) ≈ 0.693``
      (maximum "surprise", i.e. the random network answers everything
      about 50/50) and fall smoothly toward 0 as the four points fall on
      the right side of the learned boundaries.
    * The **validation** line (dashed, per epoch, not per step) is the
      loss on data the optimizer never used for updates.  Tracking the
      training line here is expected (the val split is only 25% of a 100-
      point dataset, so it barely generalizes); a val line that *rises*
      while train falls would be the classic overfitting signature in
      larger models.

    The two lines are drawn on different x-scales by design: training
    loss is measured every step, validation only at epoch ends.
    """
    import matplotlib.pyplot as plt

    # Training curve: one point per logged step, keyed by the global
    # step counter (so the x-axis is comparable across runs with
    # different epochs/batch sizes).
    train_pts = [(r["step"], r["loss"]) for r in history if "loss" in r and r["step"] is not None]
    # Validation curve: per-epoch evaluation.  With a val split the key is
    # "val_loss"; without one the Trainer reports "train_loss" instead.
    val_pts = []
    for r in history:
        vloss = r.get("val_loss", r.get("train_loss"))
        if vloss is not None and r.get("step") is not None:
            val_pts.append((r["step"], vloss))

    fig, ax = plt.subplots(figsize=(6, 4))
    if train_pts:
        xs = [p[0] for p in train_pts]
        ys = [p[1] for p in train_pts]
        ax.plot(xs, ys, color="tab:blue", linewidth=1.2, label="train loss (per step)")
    if val_pts:
        xs = [p[0] for p in val_pts]
        ys = [p[1] for p in val_pts]
        ax.plot(xs, ys, color="tab:orange", linewidth=1.6, linestyle="--",
                marker="o", markersize=4, label="val loss (per epoch)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (binary cross-entropy)")
    ax.set_title("XOR training loss")
    ax.grid(True, alpha=0.3)
    if train_pts or val_pts:
        ax.legend()
    fig.tight_layout()
    return fig


def save_loss_curve(run: RunArtifacts, history: list[dict]) -> Path | None:
    """Save :func:`loss_curve_fig` to ``run.plots_dir`` as ``loss_curve.png``.

    Returns the written path, or ``None`` when there is no history or
    matplotlib (the optional ``viz`` extra) is missing — with a warning
    log in the latter case, mirroring :func:`save_xor_plots`: plotting
    must never break training.
    """
    if not history:
        run.log("loss curve skipped: no training history recorded", level=logging.WARNING)
        return None
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        run.log(
            "loss curve skipped: matplotlib not installed "
            '(install the viz extra: pip install -e ".[viz]")',
            level=logging.WARNING,
        )
        return None

    fig = loss_curve_fig(history)
    path = run.save_plot(fig, "loss_curve")
    plt.close(fig)
    return path
