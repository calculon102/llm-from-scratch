r"""2-D projections of embedding spaces with UMAP/t-SNE/PCA (T-304).

Embeddings live in 32–400 dimensions, so the neighbour list from T-303 is
the honest way to inspect them; a 2-D scatter is the *visual* companion.
Projection methods preserve different things:

* **t-SNE** (van der Maaten & Hinton, 2008) converts pairwise distances
  into neighbour probabilities and minimises the KL divergence between
  high- and low-dimensional distributions.  It is excellent at revealing
  local clusters, but the distances *between* clusters are not meaningful
  and the result depends on the random seed and perplexity.
* **UMAP** (McInnes et al., 2018) builds a fuzzy nearest-neighbour graph
  and optimises a low-dimensional layout of it.  It is usually faster and
  preserves more global structure than t-SNE, at the cost of extra
  dependencies (``umap-learn``/numba).
* **PCA** is the linear baseline: project onto the two directions of
  largest variance.  It is deterministic, dependency-free and honest
  about global variance, but it cannot unfold a curved manifold.

Because skip-gram similarity is a **cosine** quantity, the vectors are L2
normalized before projection by default (for UMAP we additionally pass
``metric="cosine"``).  Projections are for exploration, not for measuring
similarity: use :mod:`core.embeddings.neighbors` for that.

Matplotlib is optional (the ``viz`` extra).  :func:`save_projection_plot`
pins the non-interactive Agg backend, as the XOR plots do, so it is safe to
call from the UI's background thread.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.embeddings.neighbors import EmbeddingIndex
from core.embeddings.pipeline import (
    EmbeddingArtifacts,
    EmbeddingsError,
    load_embedding_artifacts,
)

__all__ = [
    "Projection",
    "project_embeddings",
    "save_projection_plot",
]

_PROJECTION_METHODS = ("tsne", "umap", "pca")


@dataclass
class Projection:
    """A 2-D layout of an embedding vocabulary.

    ``coordinates[i]`` is the point for ``tokens[i]``.  ``parameters``
    records the effective settings (including a clamped perplexity), and
    ``explained_variance_ratio`` is filled only for PCA.
    """

    coordinates: np.ndarray
    tokens: list[str]
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)
    norms: np.ndarray | None = None
    explained_variance_ratio: list[float] | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise EmbeddingsError(
                f"projection coordinates must be (n, 2), got {coordinates.shape}"
            )
        if coordinates.shape[0] != len(self.tokens):
            raise EmbeddingsError(
                f"projection has {coordinates.shape[0]} points but "
                f"{len(self.tokens)} tokens"
            )
        self.coordinates = coordinates

    @property
    def n_points(self) -> int:
        return len(self.tokens)

    def points(self) -> list[tuple[str, float, float]]:
        return [
            (token, float(x), float(y))
            for token, (x, y) in zip(self.tokens, self.coordinates)
        ]


def _as_index(
    source: str | Path | EmbeddingIndex | EmbeddingArtifacts,
) -> EmbeddingIndex:
    if isinstance(source, EmbeddingIndex):
        return source
    if isinstance(source, EmbeddingArtifacts):
        return EmbeddingIndex.from_artifacts(source)
    return EmbeddingIndex.load(source)


def _canonical_method(method: str) -> str:
    normalized = method.lower().replace("-", "")
    if normalized not in _PROJECTION_METHODS:
        raise ValueError(
            f"unknown projection method {method!r}; "
            f"choose from {', '.join(_PROJECTION_METHODS)}"
        )
    return normalized


def _pca(
    vectors: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Two-component PCA via SVD; returns coordinates and explained variance."""
    centered = vectors.astype(np.float64, copy=False)
    centered = centered - centered.mean(axis=0, keepdims=True)
    # full_matrices=False keeps memory at O(n*d); we only need two axes.
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    total = float(np.sum(singular**2))
    explained = (
        [float(singular[i] ** 2 / total) for i in range(min(2, len(singular)))]
        if total > 0
        else [0.0, 0.0]
    )
    if coords.shape[1] < 2:  # degenerate 1-D input
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return coords.astype(np.float32), explained


def _tsne(
    vectors: np.ndarray,
    *,
    seed: int,
    perplexity: float,
    n_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EmbeddingsError(
            "t-SNE needs scikit-learn; install the viz extra: "
            'pip install -e ".[viz]"'
        ) from exc

    n_samples = vectors.shape[0]
    # sklearn requires 0 < perplexity < n_samples.  Clamp tiny vocabularies
    # rather than failing the whole UI page; the effective value is recorded.
    effective = float(min(perplexity, max(1.0, n_samples - 1)))
    kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": effective,
        "init": "pca" if n_samples >= 10 else "random",
        "random_state": seed,
        "learning_rate": "auto",
    }
    # sklearn renamed n_iter -> max_iter in 1.5; support both installed versions.
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = n_iter
    else:  # pragma: no cover - older sklearn
        kwargs["n_iter"] = n_iter
    coords = TSNE(**kwargs).fit_transform(vectors)
    return np.asarray(coords, dtype=np.float32), {
        "perplexity": effective,
        "n_iter": n_iter,
        "init": kwargs["init"],
    }


def _umap(
    vectors: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EmbeddingsError(
            "UMAP needs umap-learn; install the viz extra: "
            'pip install -e ".[viz]"'
        ) from exc

    effective_neighbors = int(min(n_neighbors, max(2, vectors.shape[0] - 1)))
    kwargs: dict[str, Any] = {
        "n_components": 2,
        "n_neighbors": effective_neighbors,
        "min_dist": min_dist,
        "metric": "cosine",
        "random_state": seed,
    }
    # umap-learn 0.6 announces a pipeline change through this flag; pin the
    # legacy layout so projections stay reproducible across versions.
    if "compatibility_layout" in inspect.signature(umap.UMAP).parameters:
        kwargs["compatibility_layout"] = True
    coords = umap.UMAP(**kwargs).fit_transform(vectors)
    return np.asarray(coords, dtype=np.float32), {
        "n_neighbors": effective_neighbors,
        "min_dist": min_dist,
        "metric": "cosine",
    }


def project_embeddings(
    source: str | Path | EmbeddingIndex | EmbeddingArtifacts,
    *,
    method: str = "tsne",
    seed: int = 42,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    normalize: bool = True,
) -> Projection:
    """Project an embedding matrix to 2-D with ``tsne``, ``umap`` or ``pca``.

    ``source`` may be a run directory, an ``embeddings/`` directory, a
    loaded :class:`~core.embeddings.pipeline.EmbeddingArtifacts`, or an
    :class:`~core.embeddings.neighbors.EmbeddingIndex`.

    The projection is seeded where the backend supports it, but t-SNE/UMAP
    are stochastic: the same seed and input reproduce a layout, different
    seeds legitimately look different.
    """
    index = _as_index(source)
    method = _canonical_method(method)
    if index.vocab_size < 3:
        raise EmbeddingsError("need at least 3 tokens for a 2-D projection")
    if n_iter < 250 and method == "tsne":
        raise ValueError("t-SNE needs n_iter >= 250")
    if perplexity <= 0:
        raise ValueError("perplexity must be > 0")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be >= 2")
    if normalize:
        vectors = index.normalized_vectors
    else:
        vectors = index.vectors
    vectors = np.asarray(vectors, dtype=np.float32)

    parameters: dict[str, Any] = {"normalize": normalize, "seed": seed}
    explained: list[float] | None = None
    if method == "pca":
        coords, explained = _pca(vectors)
        parameters["explained_variance_ratio"] = explained
    elif method == "tsne":
        coords, extra = _tsne(
            vectors, seed=seed, perplexity=perplexity, n_iter=n_iter
        )
        parameters.update(extra)
    else:
        coords, extra = _umap(
            vectors, seed=seed, n_neighbors=n_neighbors, min_dist=min_dist
        )
        parameters.update(extra)

    return Projection(
        coordinates=coords,
        tokens=list(index.tokens),
        method=method,
        parameters=parameters,
        norms=index.norms,
        explained_variance_ratio=explained,
    )


def _use_headless_backend() -> None:
    """Pin matplotlib to Agg before building a figure (thread-safe saves)."""
    import matplotlib
    import matplotlib.pyplot as plt

    backend = plt.get_backend().lower()
    if backend != "agg" and not backend.startswith("module://"):
        matplotlib.use("Agg")


def save_projection_plot(
    projection: Projection,
    path: str | Path,
    *,
    title: str | None = None,
    annotate: bool = True,
    max_labels: int = 120,
    figsize: tuple[float, float] = (10.0, 8.0),
    dpi: int = 150,
) -> Path:
    """Save a :class:`Projection` as a PNG scatter plot.

    Points are coloured by embedding norm (a rough frequency/salience
    signal) and the highest-norm tokens are labelled first, which keeps a
    255-token plot readable without hiding the rest of the vocabulary.
    Matplotlib is required; when it is missing an :class:`EmbeddingsError`
    with the install hint is raised.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EmbeddingsError(
            "plotting needs matplotlib; install the viz extra: "
            'pip install -e ".[viz]"'
        ) from exc

    _use_headless_backend()
    output = Path(path)
    if output.suffix == "":
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    coords = projection.coordinates
    fig, ax = plt.subplots(figsize=figsize)
    color = projection.norms
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color,
        cmap="viridis" if color is not None else None,
        s=20,
        alpha=0.8,
        edgecolors="none",
    )
    if color is not None:
        fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.03, label="embedding norm")

    if annotate and max_labels > 0:
        if projection.norms is not None:
            order = np.argsort(projection.norms)[::-1]
        else:
            order = np.arange(projection.n_points)
        for index in order[: min(max_labels, projection.n_points)]:
            token = projection.tokens[int(index)]
            x, y = coords[int(index)]
            ax.annotate(
                token,
                (x, y),
                fontsize=6,
                alpha=0.85,
                xytext=(3, 2),
                textcoords="offset points",
            )

    label = projection.method.upper().replace("TSNE", "t-SNE")
    ax.set_title(title or f"Embedding projection ({label})")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.2)
    note = "cosine geometry" if projection.parameters.get("normalize") else "raw vectors"
    ax.set_xlabel(f"{projection.n_points} tokens · {note}", fontsize=8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output
