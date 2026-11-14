"""CLI: inspect trained embeddings (T-303/T-304).

Usage::

    python -m scripts.embeddings neighbors --run artifacts/<run> --token Straße
    python -m scripts.embeddings neighbors --token Straße --top-k 5  # latest run
    python -m scripts.embeddings project --run artifacts/<run> --method tsne
    python -m scripts.embeddings project --method umap --output /tmp/umap.png

The neighbor command loads a run's ``embeddings/`` directory and prints the
cosine top-k.  The project command writes a 2-D scatter plot into the run's
``plots/`` folder (or an explicit ``--output`` path).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.embeddings.neighbors import EmbeddingIndex, format_neighbors
from core.embeddings.pipeline import EmbeddingsError, find_embedding_dirs


def _default_run() -> Path | None:
    candidates = find_embedding_dirs()
    return candidates[0] if candidates else None


def _resolve_run(args: argparse.Namespace) -> Path:
    if args.run:
        return Path(args.run)
    latest = _default_run()
    if latest is None:
        raise EmbeddingsError(
            "no embeddings artifact found under artifacts/; train one first "
            "with: python -m scripts.train --config configs/embeddings_tiny_debug.yaml"
        )
    return latest


def _neighbors(args: argparse.Namespace) -> int:
    run = _resolve_run(args)
    index = EmbeddingIndex.load(run)
    print(f"embeddings: {index.directory} (vocab={index.vocab_size}, dim={index.embedding_dim})")
    if args.token not in index:
        print(f"error: token {args.token!r} is not in the vocabulary", file=sys.stderr)
        return 2
    hits = index.neighbors(args.token, k=args.top_k)
    print(format_neighbors(args.token, hits))
    return 0


def _project(args: argparse.Namespace) -> int:
    # Imported lazily so the neighbor command works without the optional
    # viz dependencies (and without importing sklearn/umap for every query).
    from core.embeddings.projection import project_embeddings, save_projection_plot

    run = _resolve_run(args)
    index = EmbeddingIndex.load(run)
    projection = project_embeddings(
        index,
        method=args.method,
        seed=args.seed,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        normalize=args.normalize,
    )
    if args.output:
        output = Path(args.output)
    else:
        base = Path(run)
        run_dir = base.parent if base.name == "embeddings" else base
        output = run_dir / "plots" / f"embeddings_{projection.method}.png"
    path = save_projection_plot(
        projection,
        output,
        title=args.title,
        annotate=not args.no_annotate,
        max_labels=args.max_labels,
    )
    print(
        f"projection: {projection.method} on {projection.n_points} tokens "
        f"→ {path}"
    )
    if projection.explained_variance_ratio is not None:
        print(f"explained variance: {projection.explained_variance_ratio}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect trained embeddings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    neighbor_parser = subparsers.add_parser(
        "neighbors", help="print cosine nearest neighbors for a token"
    )
    neighbor_parser.add_argument("--run", help="run directory or embeddings/ directory")
    neighbor_parser.add_argument("--token", required=True, help="token string to query")
    neighbor_parser.add_argument("--top-k", type=int, default=10, help="number of neighbors")
    neighbor_parser.set_defaults(func=_neighbors)

    project_parser = subparsers.add_parser(
        "project", help="write a 2-D t-SNE/UMAP/PCA scatter plot"
    )
    project_parser.add_argument("--run", help="run directory or embeddings/ directory")
    project_parser.add_argument(
        "--method", choices=("tsne", "umap", "pca"), default="tsne"
    )
    project_parser.add_argument("--output", help="output PNG path")
    project_parser.add_argument("--title", default=None)
    project_parser.add_argument("--seed", type=int, default=42)
    project_parser.add_argument("--perplexity", type=float, default=30.0)
    project_parser.add_argument("--n-iter", type=int, default=1000)
    project_parser.add_argument("--n-neighbors", type=int, default=15)
    project_parser.add_argument("--min-dist", type=float, default=0.1)
    project_parser.add_argument("--max-labels", type=int, default=120)
    project_parser.add_argument(
        "--no-annotate", action="store_true", help="do not label points"
    )
    project_parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="project raw vectors instead of L2-normalized ones",
    )
    project_parser.set_defaults(func=_project, normalize=True)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (EmbeddingsError, KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
