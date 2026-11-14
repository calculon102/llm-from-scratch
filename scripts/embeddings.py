"""CLI: inspect trained embeddings (T-303).

Usage::

    python -m scripts.embeddings neighbors --run artifacts/<run> --token Straße
    python -m scripts.embeddings neighbors --token Straße --top-k 5  # latest run

The neighbor command loads a run's ``embeddings/`` directory and prints the
cosine top-k.  A projection command is added in T-304.
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

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (EmbeddingsError, KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
