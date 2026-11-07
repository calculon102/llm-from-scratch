"""XOR visualization: decision boundary + hidden activations (T-102).

Two figures, both answering questions the loss number cannot:

- :func:`decision_boundary_fig` — the model's ``P(y=1 | x)`` over the whole
  input plane, with the ``P = 0.5`` decision contour.  Shows *where* the
  model draws the line between the two classes.
- :func:`activations_fig` — each hidden unit's activation over the input
  plane.  Shows *how* the network decomposed XOR into half-plane features.

:func:`save_xor_plots` builds both and writes them to a run's ``plots/``
folder via the artifact manager (T-004).  Matplotlib is an optional
dependency (the ``viz`` extra); when it is missing, a warning is logged
and no plots are written, so training never fails because of plotting.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn

from core.artifacts import RunArtifacts
from core.xor.dataset import XOR_POINTS

__all__ = ["decision_boundary_fig", "activations_fig", "save_xor_plots"]


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
        acts = model.net[1](model.net[0](points)).reshape(grid_size, grid_size, -1)
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
