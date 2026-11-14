"""Tests for cosine nearest-neighbor search (T-303)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.embeddings.neighbors import (
    EmbeddingIndex,
    Neighbor,
    format_neighbors,
    nearest_neighbors,
)
from core.embeddings.pipeline import (
    EmbeddingsError,
    load_embedding_artifacts,
    write_embedding_artifacts,
)
from scripts.embeddings import main


def _toy_index() -> EmbeddingIndex:
    # Two rough semantic clusters: people vs. animals.  Cosine similarity
    # should rank the other people above the animal for a person query.
    return EmbeddingIndex(
        np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.9, 0.1],
                [0.7, 0.3],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        ["könig", "königin", "mann", "frau", "hund"],
    )


def test_similarity_matches_manual_cosine():
    index = _toy_index()
    a = np.array([1.0, 0.0])
    b = np.array([0.8, 0.2])
    expected = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert index.similarity("könig", "königin") == pytest.approx(expected)
    assert index.similarity("könig", "könig") == pytest.approx(1.0)


def test_neighbors_are_ranked_and_exclude_query():
    index = _toy_index()
    hits = index.neighbors("könig", k=10)

    assert [hit.token for hit in hits] == ["mann", "königin", "frau", "hund"]
    assert "könig" not in [hit.token for hit in hits]
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(hit, Neighbor) for hit in hits)
    assert hits[0].index == index.token_id("mann")


def test_top_k_is_clamped_to_vocabulary_size():
    index = _toy_index()
    assert len(index.neighbors("hund", k=2)) == 2
    assert len(index.neighbors("hund", k=999)) == len(index) - 1


def test_k_must_be_positive():
    index = _toy_index()
    with pytest.raises(ValueError, match="k must be"):
        index.neighbors("könig", k=0)


def test_cosine_is_scale_invariant():
    index = _toy_index()
    scaled = EmbeddingIndex(index.vectors * 5.0, index.tokens)
    assert scaled.similarity("könig", "königin") == pytest.approx(
        index.similarity("könig", "königin"), abs=1e-6
    )


def test_zero_vectors_score_zero_instead_of_nan():
    index = EmbeddingIndex([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], ["null", "x", "y"])
    hits = index.neighbors("null", k=2)
    assert [hit.score for hit in hits] == [0.0, 0.0]


def test_unknown_token_has_clear_error():
    index = _toy_index()
    with pytest.raises(KeyError, match="not in the vocabulary"):
        index.neighbors("unbekannt")


def test_query_vector_accepts_raw_vector_and_exclusions():
    index = _toy_index()
    hits = index.query_vector([1.0, 0.0], k=2, exclude=[0, 2])
    assert [hit.token for hit in hits] == ["königin", "frau"]


def test_most_similar_arithmetic_excludes_query_terms():
    index = _toy_index()
    hits = index.most_similar(positive=["könig", "frau"], negative=["mann"], k=2)
    assert "könig" not in [hit.token for hit in hits]
    assert "frau" not in [hit.token for hit in hits]
    assert "mann" not in [hit.token for hit in hits]
    assert hits


def test_load_from_saved_artifacts(tmp_path: Path):
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    path = write_embedding_artifacts(
        tmp_path, vectors, ["könig", "königin", "hund"], {"run_name": "toy"}
    )

    index = EmbeddingIndex.load(path)
    assert index.vocab_size == 3
    assert index.embedding_dim == 2
    assert index.metadata["run_name"] == "toy"
    assert index.neighbors("könig", k=1)[0].token == "königin"

    # The one-call helper accepts both a loaded artifact object and a path.
    artifacts = load_embedding_artifacts(path)
    assert nearest_neighbors(artifacts, "könig", k=1)[0].token == "königin"
    assert nearest_neighbors(path, "hund", k=1)[0].token == "königin"


def test_nearest_neighbors_accepts_index_directly():
    index = _toy_index()
    assert nearest_neighbors(index, "hund", k=1)[0].token == "frau"


def test_format_neighbors_is_readable():
    text = format_neighbors("könig", _toy_index().neighbors("könig", k=2))
    assert "neighbors of 'könig'" in text
    assert "mann" in text
    assert "+0.99" in text


def test_cli_neighbors_prints_top_k(tmp_path: Path, capsys: pytest.CaptureFixture):
    path = write_embedding_artifacts(
        tmp_path / "embeddings",
        np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        ["könig", "königin", "hund"],
        {"run_name": "toy"},
    )

    code = main(
        ["neighbors", "--run", str(path), "--token", "könig", "--top-k", "1"]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "neighbors of 'könig'" in captured.out
    assert "königin" in captured.out
    assert "hund" not in captured.out


def test_cli_rejects_unknown_token(tmp_path: Path, capsys: pytest.CaptureFixture):
    path = write_embedding_artifacts(
        tmp_path, np.array([[1.0, 0.0]], dtype=np.float32), ["könig"], {}
    )
    code = main(["neighbors", "--run", str(path), "--token", "xyz"])
    captured = capsys.readouterr()
    assert code == 2
    assert "not in the vocabulary" in captured.err


def test_invalid_index_shapes_are_rejected():
    with pytest.raises(EmbeddingsError, match="2-D"):
        EmbeddingIndex([1.0, 0.0], ["a"])
    with pytest.raises(EmbeddingsError, match="rows"):
        EmbeddingIndex([[1.0, 0.0]], ["a", "b"])
    with pytest.raises(EmbeddingsError, match="empty"):
        EmbeddingIndex(np.zeros((0, 2), dtype=np.float32), [])
