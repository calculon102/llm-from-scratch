"""Embeddings pipeline: config → tokenizer → pairs → artifacts (T-302).

This module is the glue shared by the phase factory, the nearest-neighbor
tools, the projection utility, the UI explorer and the tests.  Keeping
path resolution and artifact persistence in one place means the CLI and
the UI cannot disagree about where vectors live or what a vocabulary
entry means.

Artifact layout (inside one run folder created by T-004)::

    <run>/
      tokenizers/              # copy of the bundle used to encode the corpus
        bundle.json
        char/
        bpe/
      embeddings/
        vectors.npy            # float32 (vocab_size, embedding_dim)
        vocab.json             # list of token strings, id == list index
        meta.json              # dims, objective settings, source, summary
        corpus.txt             # encoded corpus (when trained from text)

``vectors.npy`` stores the **centre** embedding table, because that is the
table word2vec exports.  ``vocab.json`` is a JSON list rather than a dict:
the id is the index, which makes the alignment with ``vectors.npy``
impossible to get wrong.  :func:`load_embedding_artifacts` returns a small
object with a ``token_to_id`` map for reverse lookups.

Path resolution follows the M2 convention: config paths are relative to
the repository root, independent of the current working directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from core.tokenizer.corpus import Corpus, CorpusOptions, ingest_corpus
from core.tokenizer.pipeline import (
    DEFAULT_SAMPLE_CORPUS,
    REPO_ROOT,
    TokenizerBundle,
    load_bundle,
    resolve_corpus_path,
    save_bundle,
    train_bundle,
)

__all__ = [
    "DEFAULT_EMBEDDINGS_DIR",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_NEGATIVE_SAMPLES",
    "DEFAULT_WINDOW_SIZE",
    "EMBEDDINGS_FORMAT",
    "EMBEDDINGS_VERSION",
    "EmbeddingArtifacts",
    "EmbeddingData",
    "EmbeddingsError",
    "effective_phase_config",
    "encode_token_ids",
    "find_embedding_dirs",
    "load_embedding_artifacts",
    "prepare_embedding_data",
    "save_embedding_artifacts",
    "write_embedding_artifacts",
]

#: Name of the per-run embeddings subdirectory.
DEFAULT_EMBEDDINGS_DIR = "embeddings"

#: Marker/version written into ``meta.json``.
EMBEDDINGS_FORMAT = "llm-from-scratch-embeddings"
EMBEDDINGS_VERSION = 1

#: Defaults used when a phase config omits a setting.  They intentionally
#: mirror the schema in :mod:`core.embeddings` so direct API calls and
#: validated YAML files behave the same.
DEFAULT_EMBEDDING_DIM = 32
DEFAULT_WINDOW_SIZE = 3
DEFAULT_NEGATIVE_SAMPLES = 5
DEFAULT_MAX_PAIRS = 20_000


class EmbeddingsError(Exception):
    """Raised for invalid corpus/tokenizer/artifact setups."""


def effective_phase_config(cfg: dict) -> dict[str, Any]:
    """Return ``phase_config`` with the embeddings defaults filled in.

    The config schema validates user values but does not inject phase
    defaults (that is the phase's job), so every consumer calls this one
    helper instead of repeating ``pc.get(key, default)`` everywhere.
    """
    pc = dict(cfg.get("phase_config", {}) or {})
    defaults = {
        "tokenizer_bundle": "",
        "corpus_path": str(DEFAULT_SAMPLE_CORPUS),
        "vocab_size": 300,
        "min_frequency": 2,
        "max_token_length": 16,
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "window_size": DEFAULT_WINDOW_SIZE,
        "dynamic_window": True,
        "negative_samples": DEFAULT_NEGATIVE_SAMPLES,
        "negative_power": 0.75,
        "max_pairs": DEFAULT_MAX_PAIRS,
        "min_line_chars": 2,
        "max_line_chars": 2000,
        "min_alpha_ratio": 0.0,
        "drop_duplicates": True,
        "lowercase": False,
        "encodings": ["utf-8-sig", "utf-8", "cp1252", "latin-1"],
        "extensions": [".txt"],
    }
    defaults.update(pc)
    return defaults


@dataclass
class EmbeddingData:
    """Everything the phase factory needs to construct a training bundle."""

    tokenizer: TokenizerBundle
    corpus: Corpus | None
    token_ids: list[int]
    tokens: list[str]
    dataset: Any  # SkipGramDataset, imported lazily to avoid a cycle
    config: dict[str, Any] = field(default_factory=dict)
    content_ids: list[int] = field(default_factory=list)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def corpus_text(self) -> str | None:
        return self.corpus.text if self.corpus is not None else None


def encode_token_ids(tokenizer: TokenizerBundle, text: str) -> list[int]:
    """Encode ``text`` with the bundle's BPE tokenizer, dropping specials.

    Special tokens (``<pad>``/``<unk>``/``<bos>``/``<eos>``) are structural
    markers, not words.  Leaving them in would teach the model a vector for
    ``<pad>`` that is dominated by padding positions; the skip-gram phase
    therefore trains on content ids only.
    """
    special_ids = set(tokenizer.bpe.special_ids)
    return [int(token) for token in tokenizer.bpe.encode(text) if int(token) not in special_ids]


def prepare_embedding_data(
    cfg: dict,
    *,
    base_dir: str | Path = REPO_ROOT,
) -> EmbeddingData:
    """Load or train a tokenizer, encode the corpus and build the pairs.

    If ``phase_config.tokenizer_bundle`` points at a saved bundle (a run's
    ``tokenizers/`` directory), that bundle is reused to encode the
    configured corpus.  Otherwise the corpus is ingested and a fresh
    character/BPE bundle is trained on it, exactly like the M2 CLI.  The
    tokenizer is copied into the embeddings run so the vectors remain
    reproducible.
    """
    from core.embeddings.data import build_skipgram_dataset

    pc = effective_phase_config(cfg)
    bundle_path = str(pc.get("tokenizer_bundle", "") or "").strip()
    corpus_path = resolve_corpus_path(pc["corpus_path"], base_dir)
    corpus = ingest_corpus(corpus_path, CorpusOptions.from_phase_config(pc))
    if not corpus.text.strip():
        raise EmbeddingsError(f"corpus is empty after cleaning: {corpus_path}")

    if bundle_path:
        resolved = Path(bundle_path)
        if not resolved.is_absolute():
            resolved = Path(base_dir) / resolved
        tokenizer = load_bundle(resolved)
        source = str(resolved)
    else:
        tokenizer = train_bundle(
            corpus.text,
            vocab_size=int(pc["vocab_size"]),
            min_frequency=int(pc["min_frequency"]),
            max_token_length=int(pc["max_token_length"]),
            source=str(pc["corpus_path"]),
        )
        source = str(corpus_path)

    special_ids = set(tokenizer.bpe.special_ids)
    content_ids = [
        token_id
        for token_id in range(tokenizer.bpe.vocab_size)
        if token_id not in special_ids
    ]
    tokens = [tokenizer.bpe.id_to_token(token_id) for token_id in content_ids]
    token_ids = encode_token_ids(tokenizer, corpus.text)
    if len(token_ids) < 2:
        raise EmbeddingsError(
            f"corpus {source!r} produced only {len(token_ids)} content tokens; "
            "at least two are needed for skip-gram pairs"
        )

    dataset = build_skipgram_dataset(
        token_ids,
        vocab_size=tokenizer.bpe.vocab_size,
        window_size=int(pc["window_size"]),
        dynamic_window=bool(pc["dynamic_window"]),
        negatives=int(pc["negative_samples"]),
        seed=int(cfg.get("seed", 42)),
        power=float(pc["negative_power"]),
        max_pairs=int(pc["max_pairs"]),
    )
    return EmbeddingData(
        tokenizer=tokenizer,
        corpus=corpus,
        token_ids=token_ids,
        tokens=tokens,
        dataset=dataset,
        config=pc,
        content_ids=content_ids,
    )


# ---------------------------------------------------------------------------
# Artifact persistence and discovery
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingArtifacts:
    """Loaded embedding vectors + vocabulary (see :func:`load_embedding_artifacts`)."""

    directory: Path
    vectors: np.ndarray
    tokens: list[str]
    metadata: dict[str, Any]

    @property
    def vocab_size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def __repr__(self) -> str:
        return (
            f"EmbeddingArtifacts(vocab_size={self.vocab_size}, "
            f"embedding_dim={self.embedding_dim}, dir={self.directory})"
        )


def write_embedding_artifacts(
    directory: str | Path,
    vectors: np.ndarray | torch.Tensor,
    tokens: Sequence[str],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write ``vectors.npy``, ``vocab.json`` and ``meta.json``.

    ``vectors`` must be ``(vocab_size, embedding_dim)`` and ``tokens`` must
    have exactly one entry per row.  The vectors are stored as float32
    (the format word2vec tooling expects) regardless of the training dtype.
    """
    directory = Path(directory)
    array = vectors.detach().cpu().numpy() if isinstance(vectors, torch.Tensor) else np.asarray(vectors)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise EmbeddingsError(
            f"vectors must be 2-D (vocab_size, embedding_dim), got shape {array.shape}"
        )
    if array.shape[0] != len(tokens):
        raise EmbeddingsError(
            f"vectors has {array.shape[0]} rows but {len(tokens)} tokens were given"
        )
    if not tokens:
        raise EmbeddingsError("tokens must not be empty")

    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "vectors.npy", array)
    (directory / "vocab.json").write_text(
        json.dumps(list(tokens), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    meta = {
        "format": EMBEDDINGS_FORMAT,
        "version": EMBEDDINGS_VERSION,
        "vocab_size": int(array.shape[0]),
        "embedding_dim": int(array.shape[1]),
        **(metadata or {}),
    }
    (directory / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return directory


def save_embedding_artifacts(
    run: Any,
    model: Any,
    tokens: Sequence[str],
    *,
    metadata: dict[str, Any] | None = None,
    tokenizer: TokenizerBundle | None = None,
    corpus_text: str | None = None,
    vector_ids: Sequence[int] | None = None,
) -> Path:
    """Save a trained model's centre vectors plus its reproducibility data.

    ``run`` is a :class:`core.artifacts.RunArtifacts` (anything with a
    ``run_dir`` and ``log`` works).  ``vector_ids`` selects which model rows
    are exported; the phase passes its non-special token ids so the saved
    vocabulary does not contain untrained ``<pad>``/``<bos>`` rows.  The
    tokenizer bundle is copied under ``<run>/tokenizers/`` and the corpus
    text under ``<run>/embeddings/corpus.txt``.
    """
    run_dir = Path(run.run_dir)
    vectors = model.vectors.detach().cpu().numpy()
    if vector_ids is not None:
        vectors = vectors[np.asarray(list(vector_ids), dtype=np.int64)]
    directory = write_embedding_artifacts(
        run_dir / DEFAULT_EMBEDDINGS_DIR,
        vectors,
        tokens,
        metadata,
    )
    if tokenizer is not None:
        save_bundle(tokenizer, run_dir / "tokenizers")
    if corpus_text is not None:
        (directory / "corpus.txt").write_text(corpus_text, encoding="utf-8")
    if hasattr(run, "log"):
        run.log(
            f"embeddings written: {directory} "
            f"({len(tokens)} tokens × {vectors.shape[1]} dims)"
        )
    return directory


def _resolve_embedding_dir(directory: str | Path) -> Path:
    path = Path(directory)
    if (path / "vectors.npy").is_file():
        return path
    nested = path / DEFAULT_EMBEDDINGS_DIR
    if (nested / "vectors.npy").is_file():
        return nested
    raise EmbeddingsError(
        f"not an embeddings artifact: {path} (expected vectors.npy or "
        f"{DEFAULT_EMBEDDINGS_DIR}/vectors.npy)"
    )


def load_embedding_artifacts(directory: str | Path) -> EmbeddingArtifacts:
    """Load a run directory or an ``embeddings/`` directory."""
    path = _resolve_embedding_dir(directory)
    try:
        vectors = np.load(path / "vectors.npy")
        tokens = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise EmbeddingsError(f"cannot read embeddings artifact {path}: {exc}") from exc
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise EmbeddingsError(f"vectors.npy must be 2-D, got {vectors.shape}")
    if not isinstance(tokens, list) or len(tokens) != vectors.shape[0]:
        raise EmbeddingsError(
            f"vocab.json must list {vectors.shape[0]} tokens, got "
            f"{len(tokens) if isinstance(tokens, list) else type(tokens).__name__}"
        )
    if not all(isinstance(token, str) for token in tokens):
        raise EmbeddingsError("vocab.json entries must be strings")
    return EmbeddingArtifacts(
        directory=path,
        vectors=vectors,
        tokens=[str(token) for token in tokens],
        metadata=dict(metadata),
    )


def find_embedding_dirs(root: str | Path | None = None) -> list[Path]:
    """Find ``*/embeddings`` artifact directories under ``root``, newest first."""
    root = Path(root) if root is not None else REPO_ROOT / "artifacts"
    if not root.is_dir():
        return []
    found = [
        path
        for path in root.glob("*/" + DEFAULT_EMBEDDINGS_DIR)
        if (path / "vectors.npy").is_file()
    ]
    found.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return found
