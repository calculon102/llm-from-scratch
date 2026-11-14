"""Tests for 2-D embedding projections (T-304)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.embeddings.neighbors import EmbeddingIndex
from core.embeddings.pipeline import (
    EmbeddingsError,
    load_embedding_artifacts,
    write_embedding_artifacts,
)
from core.embeddings.projection import (
    Projection,
    project_embeddings,
    save_projection_plot,
)
from scripts.embeddings import main

_TOKENS = [
    "könig",
    "königin",
    "mann",
    "frau",
    "hund",
    "katze",
    "pferd",
    "kuh",
    "apfel",
    "brot",
    "milch",
    "wein",
]


def _toy_index(seed: int = 3) -> EmbeddingIndex:
    """Three well-separated clusters in 6-D with a little noise."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    vectors = np.vstack(
        [centers[i // 4] + 0.03 * rng.standard_normal(6) for i in range(len(_TOKENS))]
    ).astype(np.float32)
    return EmbeddingIndex(vectors, _TOKENS)


def test_pca_projection_shape_and_variance():
    projection = project_embeddings(_toy_index(), method="pca")
    assert isinstance(projection, Projection)
    assert projection.method == "pca"
    assert projection.coordinates.shape == (len(_TOKENS), 2)
    assert np.isfinite(projection.coordinates).all()
    assert projection.n_points == len(_TOKENS)
    assert projection.parameters["normalize"] is True
    assert projection.explained_variance_ratio is not None
    assert len(projection.explained_variance_ratio) == 2
    assert 0.0 <= sum(projection.explained_variance_ratio) <= 1.000001
    assert len(projection.points()) == len(_TOKENS)


def test_pca_is_deterministic():
    a = project_embeddings(_toy_index(), method="pca")
    b = project_embeddings(_toy_index(), method="pca")
    assert np.allclose(a.coordinates, b.coordinates)


def test_method_name_accepts_t_sne_alias():
    pytest.importorskip("sklearn")
    projection = project_embeddings(
        _toy_index(), method="t-SNE", n_iter=250, perplexity=5
    )
    assert projection.method == "tsne"
    assert projection.coordinates.shape == (len(_TOKENS), 2)


def test_tsne_projection_is_seeded_and_finite():
    pytest.importorskip("sklearn")
    a = project_embeddings(_toy_index(), method="tsne", seed=7, n_iter=250, perplexity=5)
    b = project_embeddings(_toy_index(), method="tsne", seed=7, n_iter=250, perplexity=5)
    assert np.isfinite(a.coordinates).all()
    assert np.allclose(a.coordinates, b.coordinates)


def test_tsne_clamps_perplexity_for_small_vocabularies():
    pytest.importorskip("sklearn")
    index = EmbeddingIndex(np.eye(3, dtype=np.float32), ["a", "b", "c"])
    projection = project_embeddings(index, method="tsne", perplexity=30, n_iter=250)
    assert projection.parameters["perplexity"] == pytest.approx(2.0)


def test_umap_projection_runs_when_installed():
    pytest.importorskip("umap")
    projection = project_embeddings(
        _toy_index(), method="umap", n_neighbors=3, min_dist=0.05, seed=11
    )
    assert projection.method == "umap"
    assert projection.coordinates.shape == (len(_TOKENS), 2)
    assert np.isfinite(projection.coordinates).all()
    assert projection.parameters["n_neighbors"] == 3


def test_input_can_be_artifacts_or_path(tmp_path: Path):
    path = write_embedding_artifacts(
        tmp_path, _toy_index().vectors, _TOKENS, {"run_name": "toy"}
    )
    artifacts = load_embedding_artifacts(path)
    from_artifacts = project_embeddings(artifacts, method="pca")
    from_path = project_embeddings(path, method="pca")
    assert np.allclose(from_artifacts.coordinates, from_path.coordinates)


def test_projection_with_too_few_tokens_is_rejected():
    index = EmbeddingIndex(np.eye(2, dtype=np.float32), ["a", "b"])
    with pytest.raises(EmbeddingsError, match="at least 3"):
        project_embeddings(index, method="pca")


def test_invalid_projection_settings_are_rejected():
    index = _toy_index()
    with pytest.raises(ValueError, match="unknown projection method"):
        project_embeddings(index, method="magic")
    with pytest.raises(ValueError, match="n_iter"):
        project_embeddings(index, method="tsne", n_iter=10)
    with pytest.raises(ValueError, match="perplexity"):
        project_embeddings(index, method="tsne", perplexity=0)
    with pytest.raises(ValueError, match="n_neighbors"):
        project_embeddings(index, method="umap", n_neighbors=1)


def test_save_projection_plot_writes_png(tmp_path: Path):
    projection = project_embeddings(_toy_index(), method="pca")
    path = save_projection_plot(projection, tmp_path / "scatter", title="Toy")
    assert path == tmp_path / "scatter.png"
    assert path.exists()
    assert path.stat().st_size > 1000
    assert path.read_bytes().startswith(b"\x89PNG")


def test_save_projection_plot_can_skip_labels(tmp_path: Path):
    projection = project_embeddings(_toy_index(), method="pca")
    path = save_projection_plot(
        projection, tmp_path / "plain.png", annotate=False, max_labels=0
    )
    assert path.read_bytes().startswith(b"\x89PNG")


def test_cli_project_writes_to_run_plots(tmp_path: Path, capsys: pytest.CaptureFixture):
    run_dir = tmp_path / "run"
    write_embedding_artifacts(
        run_dir / "embeddings", _toy_index().vectors, _TOKENS, {"run_name": "toy"}
    )

    code = main(["project", "--run", str(run_dir), "--method", "pca", "--seed", "5"])
    captured = capsys.readouterr()
    assert code == 0
    assert "projection: pca" in captured.out
    plot = run_dir / "plots" / "embeddings_pca.png"
    assert plot.exists()
    assert plot.read_bytes().startswith(b"\x89PNG")


def test_cli_project_honours_explicit_output(tmp_path: Path):
    path = write_embedding_artifacts(
        tmp_path / "embeddings", _toy_index().vectors, _TOKENS, {}
    )
    output = tmp_path / "custom.png"
    code = main(
        ["project", "--run", str(path), "--method", "pca", "--output", str(output)]
    )
    assert code == 0
    assert output.exists()
