"""M3 – Embeddings module (T-301…T-306).

A from-scratch skip-gram word2vec trainer that reuses the shared Trainer
loop (T-005), plus cosine nearest-neighbor search, optional UMAP/t-SNE
projection and a NiceGUI explorer.

Package layout:

* :mod:`core.embeddings.data` — positive pairs + negative sampling (T-301)
* :mod:`core.embeddings.model` — skip-gram model + negative-sampling loss
  (T-302)
* :mod:`core.embeddings.pipeline` — config → tokenizer → pairs → artifacts
  (T-302)
* :mod:`core.embeddings.neighbors` — cosine top-k search (T-303)
* :mod:`core.embeddings.projection` — UMAP/t-SNE/PCA 2-D views (T-304)

This module registers the ``"embeddings"`` phase schema and factory with
the config loader and the Trainer registry::

    import core.embeddings  # registers the phase
    python -m scripts.train --config configs/embeddings_tiny_debug.yaml

Phase config (all optional; defaults shown)::

    phase_config:
      # If set, reuse this tokenizer bundle instead of training a new one.
      tokenizer_bundle: ""
      corpus_path: data/sample_de
      vocab_size: 300               # only used when training a tokenizer
      min_frequency: 2
      max_token_length: 16
      embedding_dim: 32
      window_size: 3
      dynamic_window: true
      negative_samples: 5
      negative_power: 0.75
      max_pairs: 20000              # 0 = keep every pair
      # Corpus cleaning (T-201) options, used when training the tokenizer:
      min_line_chars: 2
      max_line_chars: 2000
      min_alpha_ratio: 0.0
      drop_duplicates: true
      lowercase: false
      encodings: [utf-8-sig, utf-8, cp1252, latin-1]
      extensions: [.txt]
"""

from __future__ import annotations

from typing import Any

from core.config import register_phase_schema
from core.embeddings.data import (
    NegativeSampler,
    PairOptions,
    SkipGramDataset,
    build_skipgram_dataset,
    iter_skipgram_pairs,
    pairs_to_dataset,
)
from core.embeddings.model import SkipGramModel, skipgram_loss
from core.embeddings.pipeline import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_MAX_PAIRS,
    DEFAULT_NEGATIVE_SAMPLES,
    DEFAULT_WINDOW_SIZE,
    EMBEDDINGS_FORMAT,
    EMBEDDINGS_VERSION,
    EmbeddingArtifacts,
    EmbeddingData,
    EmbeddingsError,
    effective_phase_config,
    encode_token_ids,
    find_embedding_dirs,
    load_embedding_artifacts,
    prepare_embedding_data,
    save_embedding_artifacts,
    write_embedding_artifacts,
)
from core.trainer import PhaseBundle, register_phase, split_dataset

__all__ = [
    # data (T-301)
    "NegativeSampler",
    "PairOptions",
    "SkipGramDataset",
    "build_skipgram_dataset",
    "iter_skipgram_pairs",
    "pairs_to_dataset",
    # model + pipeline (T-302)
    "SkipGramModel",
    "skipgram_loss",
    "EmbeddingArtifacts",
    "EmbeddingData",
    "EmbeddingsError",
    "prepare_embedding_data",
    "encode_token_ids",
    "load_embedding_artifacts",
    "write_embedding_artifacts",
    "save_embedding_artifacts",
    "find_embedding_dirs",
    "effective_phase_config",
    "DEFAULT_EMBEDDINGS_DIR",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_NEGATIVE_SAMPLES",
    "DEFAULT_MAX_PAIRS",
    "EMBEDDINGS_FORMAT",
    "EMBEDDINGS_VERSION",
    # config / phase registration
    "EMBEDDINGS_PHASE_SCHEMA",
    "validate_embeddings_phase_config",
    "embeddings_defaults",
    "build_embeddings_phase",
]


# ---------------------------------------------------------------------------
# Phase config schema
# ---------------------------------------------------------------------------

EMBEDDINGS_PHASE_SCHEMA: dict[str, dict[str, Any]] = {
    "tokenizer_bundle": {"type": str, "default": ""},
    "corpus_path": {"type": str, "default": "data/sample_de"},
    "vocab_size": {"type": int, "default": 300, "min": 8},
    "min_frequency": {"type": int, "default": 2, "min": 1},
    "max_token_length": {"type": int, "default": 16, "min": 2},
    "embedding_dim": {"type": int, "default": DEFAULT_EMBEDDING_DIM, "min": 2},
    "window_size": {"type": int, "default": DEFAULT_WINDOW_SIZE, "min": 1},
    "dynamic_window": {"type": bool, "default": True},
    "negative_samples": {"type": int, "default": DEFAULT_NEGATIVE_SAMPLES, "min": 1},
    "negative_power": {"type": float, "default": 0.75, "min": 0.0, "max": 1.0},
    "max_pairs": {"type": int, "default": DEFAULT_MAX_PAIRS, "min": 0},
    "min_line_chars": {"type": int, "default": 2, "min": 0},
    "max_line_chars": {"type": int, "default": 2000, "min": 0},
    "min_alpha_ratio": {"type": float, "default": 0.0, "min": 0.0, "max": 1.0},
    "drop_duplicates": {"type": bool, "default": True},
    "lowercase": {"type": bool, "default": False},
    "encodings": {"type": list, "default": ["utf-8-sig", "utf-8", "cp1252", "latin-1"], "item_type": str},
    "extensions": {"type": list, "default": [".txt"], "item_type": str},
}


def embeddings_defaults() -> dict[str, Any]:
    """Return a fresh copy of the phase-config defaults."""
    defaults: dict[str, Any] = {}
    for key, spec in EMBEDDINGS_PHASE_SCHEMA.items():
        value = spec["default"]
        defaults[key] = list(value) if isinstance(value, list) else value
    return defaults


def _check_type(value: Any, spec: dict[str, Any]) -> bool:
    expected = spec["type"]
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is bool:
        return isinstance(value, bool)
    if expected is list:
        item_type = spec.get("item_type", str)
        return isinstance(value, list) and all(
            isinstance(item, item_type) for item in value
        )
    return isinstance(value, expected)


def validate_embeddings_phase_config(phase_config: dict, source: str) -> list[str]:
    """Validate the ``phase_config`` section of an embeddings config."""
    errors: list[str] = []
    for key, value in phase_config.items():
        if key not in EMBEDDINGS_PHASE_SCHEMA:
            errors.append(
                f"phase_config: unknown key '{key}' "
                f"(allowed: {', '.join(EMBEDDINGS_PHASE_SCHEMA)})"
            )
            continue
        spec = EMBEDDINGS_PHASE_SCHEMA[key]
        if not _check_type(value, spec):
            expected_name = spec["type"].__name__
            if spec["type"] is list:
                expected_name = f"list of {spec.get('item_type', str).__name__}"
            errors.append(
                f"phase_config.{key}: expected {expected_name}, "
                f"got {type(value).__name__} ({value!r})"
            )
            continue
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and "min" in spec
            and value < spec["min"]
        ):
            errors.append(
                f"phase_config.{key}: {value} is below minimum {spec['min']}"
            )
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and "max" in spec
            and value > spec["max"]
        ):
            errors.append(
                f"phase_config.{key}: {value} is above maximum {spec['max']}"
            )
        if key in ("encodings", "extensions") and isinstance(value, list):
            if not value:
                errors.append(f"phase_config.{key}: must not be empty")
    return errors


# ---------------------------------------------------------------------------
# Phase factory
# ---------------------------------------------------------------------------

def _make_save_callback(data: EmbeddingData, cfg: dict):
    """Build the Trainer ``on_end`` hook that persists the run artifacts."""

    pc = data.config

    def _on_end(trainer: Any, **kwargs: Any) -> None:
        if trainer.run is None:
            return
        if not isinstance(trainer.model, SkipGramModel):
            return
        summary = kwargs.get("summary") or {}
        metadata = {
            "run_name": cfg.get("run_name"),
            "profile": cfg.get("profile"),
            "seed": cfg.get("seed"),
            "tokenizer_source": data.tokenizer.source,
            "corpus_path": pc.get("corpus_path"),
            "window_size": pc.get("window_size"),
            "dynamic_window": pc.get("dynamic_window"),
            "negative_samples": pc.get("negative_samples"),
            "negative_power": pc.get("negative_power"),
            "pairs": len(data.dataset),
            "content_tokens": len(data.token_ids),
            "best_val": summary.get("best_val"),
            "steps": summary.get("steps"),
            "epochs": summary.get("epochs"),
        }
        save_embedding_artifacts(
            trainer.run,
            trainer.model,
            data.tokens,
            metadata=metadata,
            tokenizer=data.tokenizer,
            corpus_text=data.corpus_text,
            vector_ids=data.content_ids,
        )

    return _on_end


def build_embeddings_phase(cfg: dict) -> PhaseBundle:
    """Build the embeddings :class:`PhaseBundle` for the shared Trainer.

    Reads the corpus (or reuses a saved tokenizer bundle), generates
    skip-gram pairs, splits them into train/validation pairs, constructs
    the model and attaches the ``on_end`` artifact-saving callback.
    """
    pc = effective_phase_config(cfg)
    data = prepare_embedding_data(cfg)
    model = SkipGramModel(
        vocab_size=data.dataset.vocab_size,
        embedding_dim=int(pc["embedding_dim"]),
    )
    train_dataset, val_dataset = split_dataset(
        data.dataset,
        float(cfg.get("val_split", 0.1)),
        seed=int(cfg.get("seed", 42)),
    )
    return PhaseBundle(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=skipgram_loss,
        callbacks={"on_end": _make_save_callback(data, cfg)},
    )


# Registration happens at import time, before any config is loaded by the
# embeddings CLI, the jobs runner or the UI page.
register_phase_schema("embeddings", validate_embeddings_phase_config)
register_phase("embeddings", build_embeddings_phase)
