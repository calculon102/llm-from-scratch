"""M4 – Mini GPT module (T-401…T-407).

A decoder-only transformer language model built from scratch and trained
with the shared Trainer loop (T-005).  The package is split the same way
as the earlier phases:

* :mod:`core.gpt.model` — causal self-attention, transformer blocks and the
  MiniGPT language model (T-401, T-402)
* :mod:`core.gpt.data` — LM chunking and train/validation split (T-403)
* :mod:`core.gpt.pipeline` — config → tokenizer → chunks → artifacts
  (T-404)
* :mod:`core.gpt.sampling` — greedy / top-k / top-p / temperature sampling
  and text generation (T-405)

This module registers the ``"gpt"`` phase schema and factory with the
config loader and the Trainer registry::

    import core.gpt
    python -m scripts.train --config configs/gpt_tiny_debug.yaml

Phase config (all optional; defaults shown)::

    phase_config:
      # If set, reuse this tokenizer bundle instead of training a new one.
      tokenizer_bundle: ""
      corpus_path: data/sample_de
      vocab_size: 300               # only used when training a tokenizer
      min_frequency: 2
      max_token_length: 16
      context_length: 64            # tokens the model conditions on
      stride: 0                     # 0 = context_length (non-overlapping)
      n_layer: 2
      n_head: 4
      n_embd: 64
      dropout: 0.0
      bias: true
      tie_weights: true             # share token embedding and LM head
      init_std: 0.02
      add_special_tokens: true      # put <bos>/<eos> around the corpus
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

import math
from typing import Any

from core.config import register_phase_schema
from core.gpt.data import (
    DEFAULT_CONTEXT_LENGTH,
    LMChunkDataset,
    LMDatasets,
    build_lm_datasets,
    chunk_starts,
    make_lm_dataloader,
    split_token_ids,
)
from core.gpt.model import (
    CausalSelfAttention,
    GPTConfig,
    MLP,
    MiniGPT,
    TransformerBlock,
    causal_mask,
    language_model_loss,
)
from core.gpt.pipeline import (
    DEFAULT_GPT_DIR,
    GPT_FORMAT,
    GPT_VERSION,
    GPTArtifacts,
    GPTData,
    GPTPipelineError,
    build_gpt_model,
    effective_phase_config,
    evaluate_perplexity,
    find_gpt_dirs,
    load_gpt_artifacts,
    prepare_gpt_data,
    save_gpt_artifacts,
    write_gpt_artifacts,
)
from core.trainer import PhaseBundle, register_phase

__all__ = [
    # model (T-401, T-402)
    "CausalSelfAttention",
    "GPTConfig",
    "MLP",
    "MiniGPT",
    "TransformerBlock",
    "causal_mask",
    "language_model_loss",
    # data (T-403)
    "DEFAULT_CONTEXT_LENGTH",
    "LMChunkDataset",
    "LMDatasets",
    "build_lm_datasets",
    "chunk_starts",
    "make_lm_dataloader",
    "split_token_ids",
    # pipeline / training (T-404)
    "DEFAULT_GPT_DIR",
    "GPT_FORMAT",
    "GPT_VERSION",
    "GPTArtifacts",
    "GPTData",
    "GPTPipelineError",
    "build_gpt_model",
    "effective_phase_config",
    "evaluate_perplexity",
    "find_gpt_dirs",
    "load_gpt_artifacts",
    "prepare_gpt_data",
    "save_gpt_artifacts",
    "write_gpt_artifacts",
    # config / phase registration
    "GPT_PHASE_SCHEMA",
    "validate_gpt_phase_config",
    "gpt_defaults",
    "build_gpt_phase",
]


# ---------------------------------------------------------------------------
# Phase config schema
# ---------------------------------------------------------------------------

GPT_PHASE_SCHEMA: dict[str, dict[str, Any]] = {
    "tokenizer_bundle": {"type": str, "default": ""},
    "corpus_path": {"type": str, "default": "data/sample_de"},
    "vocab_size": {"type": int, "default": 300, "min": 8},
    "min_frequency": {"type": int, "default": 2, "min": 1},
    "max_token_length": {"type": int, "default": 16, "min": 2},
    "context_length": {"type": int, "default": DEFAULT_CONTEXT_LENGTH, "min": 1},
    "stride": {"type": int, "default": 0, "min": 0},
    "n_layer": {"type": int, "default": 2, "min": 1},
    "n_head": {"type": int, "default": 4, "min": 1},
    "n_embd": {"type": int, "default": 64, "min": 1},
    "dropout": {"type": float, "default": 0.0, "min": 0.0, "max": 0.95},
    "bias": {"type": bool, "default": True},
    "tie_weights": {"type": bool, "default": True},
    "init_std": {"type": float, "default": 0.02, "min": 1e-6},
    "add_special_tokens": {"type": bool, "default": True},
    "min_line_chars": {"type": int, "default": 2, "min": 0},
    "max_line_chars": {"type": int, "default": 2000, "min": 0},
    "min_alpha_ratio": {"type": float, "default": 0.0, "min": 0.0, "max": 1.0},
    "drop_duplicates": {"type": bool, "default": True},
    "lowercase": {"type": bool, "default": False},
    "encodings": {"type": list, "default": ["utf-8-sig", "utf-8", "cp1252", "latin-1"], "item_type": str},
    "extensions": {"type": list, "default": [".txt"], "item_type": str},
}


def gpt_defaults() -> dict[str, Any]:
    """Return a fresh copy of the phase-config defaults."""
    defaults: dict[str, Any] = {}
    for key, spec in GPT_PHASE_SCHEMA.items():
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


def validate_gpt_phase_config(phase_config: dict, source: str) -> list[str]:
    """Validate the ``phase_config`` section of a GPT config."""
    errors: list[str] = []
    for key, value in phase_config.items():
        if key not in GPT_PHASE_SCHEMA:
            errors.append(
                f"phase_config: unknown key '{key}' "
                f"(allowed: {', '.join(GPT_PHASE_SCHEMA)})"
            )
            continue
        spec = GPT_PHASE_SCHEMA[key]
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
    # Cross-field constraints: n_embd must divide into attention heads.
    if (
        isinstance(phase_config.get("n_embd"), int)
        and isinstance(phase_config.get("n_head"), int)
        and phase_config["n_head"] > 0
        and phase_config["n_embd"] % phase_config["n_head"] != 0
    ):
        errors.append(
            f"phase_config.n_embd ({phase_config['n_embd']}) must be divisible "
            f"by phase_config.n_head ({phase_config['n_head']})"
        )
    return errors


# ---------------------------------------------------------------------------
# Phase factory
# ---------------------------------------------------------------------------

def _make_save_callback(data: GPTData, cfg: dict):
    """Build the Trainer ``on_end`` hook that persists model + tokenizer."""

    def _on_end(trainer: Any, **kwargs: Any) -> None:
        if trainer.run is None:
            return
        if not isinstance(trainer.model, MiniGPT):
            return
        summary = kwargs.get("summary") or {}
        best_val = trainer.best_val
        metadata = {
            "run_name": cfg.get("run_name"),
            "profile": cfg.get("profile"),
            "seed": cfg.get("seed"),
            "tokenizer_source": data.tokenizer.source,
            "corpus_path": data.config.get("corpus_path"),
            "context_length": data.datasets.context_length,
            "stride": data.datasets.stride,
            "tokens": len(data.token_ids),
            "train_windows": data.datasets.train_windows,
            "val_windows": data.datasets.val_windows,
            "best_val": best_val,
            "val_perplexity": (
                math.exp(min(best_val, 50.0)) if best_val is not None else None
            ),
            "steps": summary.get("steps"),
            "epochs": summary.get("epochs"),
        }
        save_gpt_artifacts(trainer.run, trainer.model, data, metadata=metadata)

    return _on_end


def build_gpt_phase(cfg: dict) -> PhaseBundle:
    """Build the GPT :class:`PhaseBundle` for the shared Trainer.

    Ingests the corpus, trains/reuses the tokenizer, chunks the token
    stream, constructs :class:`MiniGPT`, and wires cross-entropy plus
    token-weighted perplexity evaluation.  The ``on_end`` callback writes
    the model and tokenizer into the run folder.
    """
    data = prepare_gpt_data(cfg)
    model = build_gpt_model(data)
    val_dataset = data.datasets.val_dataset
    return PhaseBundle(
        model=model,
        train_dataset=data.datasets.train_dataset,
        val_dataset=val_dataset,
        loss_fn=language_model_loss,
        metric_fn=evaluate_perplexity if val_dataset is not None else None,
        callbacks={"on_end": _make_save_callback(data, cfg)},
    )


# Registration happens at import time, before any config is loaded by the
# GPT CLI, the jobs runner or the UI page.
register_phase_schema("gpt", validate_gpt_phase_config)
register_phase("gpt", build_gpt_phase)
