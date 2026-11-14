r"""Cosine nearest-neighbor search over learned embeddings (T-303).

Word embeddings are useful because **direction** carries meaning: tokens
that occur in similar contexts end up with vectors pointing in similar
directions.  The standard way to read that out is cosine similarity,

$$
\cos(v_a, v_b) = \\frac{v_a \\cdot v_b}{\\lVert v_a\\rVert\\,\\lVert v_b\\rVert},
$$

which ignores vector length (frequency effects) and depends only on the
angle.  For a query token we rank every other token by this score and
return the top $k$.

This module keeps the operation small and inspectable:

* :class:`EmbeddingIndex` loads ``vectors.npy`` + ``vocab.json`` (via
  :func:`core.embeddings.pipeline.load_embedding_artifacts`), L2-normalizes
  the rows once, and answers queries with a single matrix-vector product.
* :class:`Neighbor` is a ``(token, score, index)`` record; ``neighbors()``
  never returns the query token itself.
* :func:`nearest_neighbors` is the one-call convenience used by the UI and
  the CLI.

Example::

    index = EmbeddingIndex.load("artifacts/<run>/embeddings")
    for hit in index.neighbors("Straße", k=5):
        print(hit.token, hit.score)

Normalization is cached, so repeated UI queries cost one dot product over
a ``(vocab_size, embedding_dim)`` matrix — microseconds for the small
models in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from core.embeddings.pipeline import (
    EmbeddingArtifacts,
    EmbeddingsError,
    load_embedding_artifacts,
)

__all__ = [
    "EmbeddingIndex",
    "Neighbor",
    "format_neighbors",
    "nearest_neighbors",
]


@dataclass(frozen=True)
class Neighbor:
    """One ranked neighbor: token string, cosine score, vocabulary index."""

    token: str
    score: float
    index: int

    def __str__(self) -> str:
        return f"{self.token!r} {self.score:+.4f}"


class EmbeddingIndex:
    """An in-memory, L2-normalized embedding matrix with token lookup."""

    def __init__(
        self,
        vectors: np.ndarray | Sequence[Sequence[float]],
        tokens: Sequence[str],
        *,
        metadata: dict[str, Any] | None = None,
        directory: str | Path | None = None,
    ) -> None:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise EmbeddingsError(
                f"vectors must be 2-D (vocab_size, embedding_dim), got {array.shape}"
            )
        if array.shape[0] != len(tokens):
            raise EmbeddingsError(
                f"vectors has {array.shape[0]} rows but {len(tokens)} tokens were given"
            )
        if not tokens:
            raise EmbeddingsError("tokens must not be empty")
        if not all(isinstance(token, str) for token in tokens):
            raise EmbeddingsError("tokens must all be strings")

        self.tokens = [str(token) for token in tokens]
        self.vectors = array
        self.metadata = dict(metadata or {})
        self.directory = Path(directory) if directory is not None else None
        self._token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self._normalized = _l2_normalize(array)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(cls, artifacts: EmbeddingArtifacts) -> "EmbeddingIndex":
        return cls(
            artifacts.vectors,
            artifacts.tokens,
            metadata=artifacts.metadata,
            directory=artifacts.directory,
        )

    @classmethod
    def load(cls, directory: str | Path) -> "EmbeddingIndex":
        """Load a run directory or an ``embeddings/`` directory."""
        return cls.from_artifacts(load_embedding_artifacts(directory))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def embedding_dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def normalized_vectors(self) -> np.ndarray:
        """Cached L2-normalized copy of the embedding matrix."""
        return self._normalized

    @property
    def norms(self) -> np.ndarray:
        """Row-wise vector lengths (a rough frequency/salience signal)."""
        return np.linalg.norm(self.vectors, axis=1)

    @property
    def token_to_id(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def __contains__(self, token: object) -> bool:
        return token in self._token_to_id

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return (
            f"EmbeddingIndex(vocab_size={self.vocab_size}, "
            f"embedding_dim={self.embedding_dim})"
        )

    def token_id(self, token: str) -> int:
        """Return the vocabulary id for ``token`` or raise a clear error."""
        try:
            return self._token_to_id[token]
        except KeyError as exc:
            raise KeyError(
                f"token {token!r} is not in the vocabulary "
                f"(vocab_size={self.vocab_size})"
            ) from exc

    def vector(self, token: str) -> np.ndarray:
        return self.vectors[self.token_id(token)].copy()

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def similarity(self, token_a: str, token_b: str) -> float:
        """Cosine similarity between two vocabulary tokens."""
        a = self._normalized[self.token_id(token_a)]
        b = self._normalized[self.token_id(token_b)]
        return float(np.dot(a, b))

    def query_vector(
        self,
        vector: np.ndarray | Sequence[float],
        *,
        k: int = 10,
        exclude: Iterable[int] | int | None = None,
    ) -> list[Neighbor]:
        """Rank vocabulary tokens by cosine similarity to ``vector``.

        ``exclude`` removes ids from the result (the query token itself,
        usually).  ``k`` larger than the number of candidates returns all
        candidates rather than raising.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        query = np.asarray(vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.embedding_dim:
            raise EmbeddingsError(
                f"query vector must have shape ({self.embedding_dim},), "
                f"got {query.shape}"
            )
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            scores = np.zeros(self.vocab_size, dtype=np.float32)
        else:
            scores = self._normalized @ (query / norm)

        excluded: set[int] = set()
        if exclude is not None:
            if isinstance(exclude, int):
                excluded.add(exclude)
            else:
                excluded.update(int(item) for item in exclude)
        candidates = [index for index in range(self.vocab_size) if index not in excluded]
        if not candidates:
            return []
        count = min(int(k), len(candidates))
        # Stable ordering on ties: score desc, then vocabulary id asc.
        order = sorted(candidates, key=lambda index: (-float(scores[index]), index))[:count]
        return [
            Neighbor(token=self.tokens[index], score=float(scores[index]), index=index)
            for index in order
        ]

    def neighbors(self, token: str, k: int = 10) -> list[Neighbor]:
        """Return the ``k`` tokens closest to ``token`` (excluding itself)."""
        token_id = self.token_id(token)
        return self.query_vector(self.vectors[token_id], k=k, exclude=token_id)

    def most_similar(
        self,
        positive: Iterable[str] = (),
        negative: Iterable[str] = (),
        *,
        k: int = 10,
    ) -> list[Neighbor]:
        """Word2vec-style analogy query: mean(positive) - mean(negative).

        The query terms are excluded from the result.  This is a small
        convenience for the explorer and for demonstrating that the space
        captures more than pairwise resemblance.
        """
        positive_tokens = list(positive)
        negative_tokens = list(negative)
        if not positive_tokens and not negative_tokens:
            raise ValueError("give at least one positive or negative token")

        used: list[int] = []

        def average(tokens: list[str], sign: float) -> np.ndarray:
            total = np.zeros(self.embedding_dim, dtype=np.float32)
            for token in tokens:
                index = self.token_id(token)
                used.append(index)
                total += sign * self._normalized[index]
            return total

        positive_mean = average(positive_tokens, 1.0)
        negative_mean = average(negative_tokens, -1.0)
        query = positive_mean + negative_mean
        return self.query_vector(query, k=k, exclude=used)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; all-zero rows stay zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (vectors / norms).astype(np.float32, copy=False)


def nearest_neighbors(
    source: str | Path | EmbeddingIndex | EmbeddingArtifacts,
    token: str,
    *,
    k: int = 10,
) -> list[Neighbor]:
    """Load ``source`` if needed and return ``token``'s top-k neighbors."""
    index = (
        source
        if isinstance(source, EmbeddingIndex)
        else EmbeddingIndex.from_artifacts(source)
        if isinstance(source, EmbeddingArtifacts)
        else EmbeddingIndex.load(source)
    )
    return index.neighbors(token, k=k)


def format_neighbors(
    token: str,
    neighbors: Sequence[Neighbor],
    *,
    limit: int | None = None,
) -> str:
    """Human-readable multi-line neighbor list for logs and the CLI."""
    shown = list(neighbors[:limit] if limit is not None else neighbors)
    lines = [f"neighbors of {token!r} ({len(shown)}):"]
    for rank, neighbor in enumerate(shown, start=1):
        lines.append(f"{rank:>3}. {neighbor.token:<24} {neighbor.score:+.4f}")
    return "\n".join(lines)
