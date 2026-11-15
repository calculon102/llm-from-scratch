"""GPT data/model pipeline and artifact persistence (T-404).

This module connects the pieces built in T-401…T-403: it ingests the
German corpus, loads or trains a tokenizer bundle, encodes the text into
one token stream, chunks it into train/validation windows, constructs a
:class:`~core.gpt.model.MiniGPT`, and attaches the shared Trainer's
cross-entropy objective and perplexity evaluation.

Why *perplexity*?
-----------------
Cross-entropy is measured in nats per token.  Exponentiating it puts the
number back on the vocabulary scale::

    perplexity = exp(loss)

A perplexity of 1 means the model is certain and always right; a
perplexity equal to the vocabulary size means it is guessing uniformly at
random.  It is the standard language-model report because it is
comparable across tokenizers and corpus sizes (unlike raw loss, which
depends on the tokenizer's vocabulary).

Artifact layout (inside one run folder)::

    <run>/
      tokenizers/             # exact tokenizer used for the token ids
        bundle.json
        char/
        bpe/
      gpt/
        model.pt              # MiniGPT state_dict
        model_config.json     # GPTConfig fields needed to rebuild it
        meta.json             # corpus, windows, training summary
        corpus.txt            # cleaned text (for inspection)

:func:`find_gpt_dirs` and :func:`load_gpt_artifacts` are also used by the
generation UI (T-406).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from core.gpt.data import LMDatasets, build_lm_datasets
from core.gpt.model import GPTConfig, MiniGPT
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
]

#: Name of the per-run GPT subdirectory.
DEFAULT_GPT_DIR = "gpt"

#: Marker/version written into ``meta.json``.
GPT_FORMAT = "llm-from-scratch-gpt"
GPT_VERSION = 1


class GPTPipelineError(Exception):
    """Raised for invalid corpus/tokenizer/artifact setups."""


def effective_phase_config(cfg: dict) -> dict[str, Any]:
    """Return ``phase_config`` with all GPT defaults filled in.

    The config schema validates user values but deliberately does not
    inject defaults; every consumer (phase factory, CLI, UI) calls this
    helper so they cannot disagree about what a missing key means.
    """
    pc = dict(cfg.get("phase_config", {}) or {})
    defaults: dict[str, Any] = {
        "tokenizer_bundle": "",
        "corpus_path": str(DEFAULT_SAMPLE_CORPUS),
        "vocab_size": 300,
        "min_frequency": 2,
        "max_token_length": 16,
        "context_length": 64,
        "stride": 0,  # 0 means "same as context_length" (no overlap)
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 64,
        "dropout": 0.0,
        "bias": True,
        "tie_weights": True,
        "init_std": 0.02,
        "add_special_tokens": True,
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
class GPTData:
    """Everything the phase factory needs to build a training bundle."""

    tokenizer: TokenizerBundle
    corpus: Corpus
    token_ids: list[int]
    datasets: LMDatasets
    config: dict[str, Any]

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.bpe.vocab_size)

    @property
    def corpus_text(self) -> str:
        return self.corpus.text


def prepare_gpt_data(
    cfg: dict,
    *,
    base_dir: str | Path = REPO_ROOT,
    corpus: Corpus | None = None,
    tokenizer: TokenizerBundle | None = None,
) -> GPTData:
    """Load/train a tokenizer, encode the corpus and chunk it for LM training.

    If ``phase_config.tokenizer_bundle`` points at a saved bundle (e.g. an
    earlier tokenizer or embeddings run), it is reused.  Otherwise a fresh
    bundle is trained on the cleaned corpus.  The tokenizer is copied into
    the GPT run by :func:`save_gpt_artifacts`, so generation can reproduce
    the exact id↔text mapping.

    The pipeline command (T-603) may pass an already-cleaned ``corpus``
    and an already-trained ``tokenizer`` so that preprocessing and
    tokenizer training happen exactly once and the GPT phase encodes that
    same stream.
    """
    pc = effective_phase_config(cfg)
    bundle_path = str(pc.get("tokenizer_bundle", "") or "").strip()
    corpus_path = resolve_corpus_path(pc["corpus_path"], base_dir)
    if corpus is None:
        corpus = ingest_corpus(corpus_path, CorpusOptions.from_phase_config(pc))
    if not corpus.text.strip():
        raise GPTPipelineError(f"corpus is empty after cleaning: {corpus_path}")

    if tokenizer is None:
        if bundle_path:
            resolved = Path(bundle_path)
            if not resolved.is_absolute():
                resolved = Path(base_dir) / resolved
            tokenizer = load_bundle(resolved)
        else:
            tokenizer = train_bundle(
                corpus.text,
                vocab_size=int(pc["vocab_size"]),
                min_frequency=int(pc["min_frequency"]),
                max_token_length=int(pc["max_token_length"]),
                source=str(pc["corpus_path"]),
            )

    token_ids = tokenizer.bpe.encode(
        corpus.text, add_special_tokens=bool(pc["add_special_tokens"])
    )
    if len(token_ids) < 2:
        raise GPTPipelineError(
            f"corpus {corpus_path!r} produced only {len(token_ids)} token(s)"
        )

    stride = int(pc["stride"]) or None
    try:
        datasets = build_lm_datasets(
            token_ids,
            context_length=int(pc["context_length"]),
            stride=stride,
            val_split=float(cfg.get("val_split", 0.1)),
            seed=int(cfg.get("seed", 42)),
        )
    except ValueError as exc:
        raise GPTPipelineError(
            f"{exc} (corpus {corpus_path} encoded to {len(token_ids)} tokens; "
            "point phase_config.corpus_path at a larger corpus or lower "
            "phase_config.context_length)"
        ) from exc

    return GPTData(
        tokenizer=tokenizer,
        corpus=corpus,
        token_ids=token_ids,
        datasets=datasets,
        config=pc,
    )


def build_gpt_model(data: GPTData) -> MiniGPT:
    """Construct the model whose vocabulary matches the tokenizer exactly."""
    pc = data.config
    try:
        config = GPTConfig(
            vocab_size=data.vocab_size,
            context_length=int(pc["context_length"]),
            n_layer=int(pc["n_layer"]),
            n_head=int(pc["n_head"]),
            n_embd=int(pc["n_embd"]),
            dropout=float(pc["dropout"]),
            bias=bool(pc["bias"]),
            tie_weights=bool(pc["tie_weights"]),
            init_std=float(pc["init_std"]),
        )
    except ValueError as exc:
        raise GPTPipelineError(f"invalid GPT architecture: {exc}") from exc
    return MiniGPT(config)


@torch.no_grad()
def evaluate_perplexity(
    model: MiniGPT, loader: DataLoader
) -> dict[str, float]:
    """Compute token-weighted validation loss and perplexity.

    The shared Trainer's default evaluation averages *batch losses*, which
    is only exact when every batch has the same number of tokens.  This
    metric sums the negative log-likelihood over every predicted token and
    divides by the token count, so the final window in a split cannot skew
    the result.  It returns the ``val_loss`` key the Trainer uses for
    best-model tracking plus the human-readable ``val_perplexity``.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    try:
        for batch in loader:
            input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
            targets = batch["targets"].to(device=device, dtype=torch.long)
            logits = model(input_ids)
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll)
            total_tokens += int(targets.numel())
    finally:
        if was_training:
            model.train()
    mean_loss = total_nll / max(total_tokens, 1)
    return {
        "val_loss": mean_loss,
        "val_perplexity": math.exp(min(mean_loss, 50.0)),
    }


# ---------------------------------------------------------------------------
# Artifact persistence and discovery
# ---------------------------------------------------------------------------

@dataclass
class GPTArtifacts:
    """A loaded GPT checkpoint plus the tokenizer needed to generate text."""

    directory: Path
    model: MiniGPT
    config: GPTConfig
    metadata: dict[str, Any]
    tokenizer: TokenizerBundle | None = None

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    @property
    def context_length(self) -> int:
        return self.model.context_length

    def __repr__(self) -> str:
        return (
            f"GPTArtifacts(vocab_size={self.vocab_size}, "
            f"context_length={self.context_length}, dir={self.directory})"
        )


def write_gpt_artifacts(
    directory: str | Path,
    model: MiniGPT,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write ``model.pt``, ``model_config.json`` and ``meta.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt")
    (directory / "model_config.json").write_text(
        json.dumps(model.config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    meta = {
        "format": GPT_FORMAT,
        "version": GPT_VERSION,
        "vocab_size": model.vocab_size,
        "context_length": model.context_length,
        "parameters": model.num_parameters(),
        "parameters_non_embedding": model.num_parameters(non_embedding=True),
        **(metadata or {}),
    }
    (directory / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return directory


def save_gpt_artifacts(
    run: Any,
    model: MiniGPT,
    data: GPTData,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a trained model and its reproducibility data in the run folder."""
    run_dir = Path(run.run_dir)
    directory = write_gpt_artifacts(run_dir / DEFAULT_GPT_DIR, model, metadata)
    save_bundle(data.tokenizer, run_dir / "tokenizers")
    (directory / "corpus.txt").write_text(data.corpus_text, encoding="utf-8")
    if hasattr(run, "log"):
        run.log(
            f"GPT written: {directory} "
            f"({model.num_parameters()} params, "
            f"{data.datasets.train_windows} train / "
            f"{data.datasets.val_windows} val windows)"
        )
    return directory


def _resolve_gpt_dir(directory: str | Path) -> Path:
    path = Path(directory)
    if (path / "model.pt").is_file():
        return path
    nested = path / DEFAULT_GPT_DIR
    if (nested / "model.pt").is_file():
        return nested
    raise GPTPipelineError(
        f"not a GPT artifact: {path} (expected model.pt or "
        f"{DEFAULT_GPT_DIR}/model.pt)"
    )


def _find_tokenizer(directory: Path) -> TokenizerBundle | None:
    candidates = [
        directory / "tokenizers",
        directory.parent / "tokenizers",
        directory.parent.parent / "tokenizers",
    ]
    for candidate in candidates:
        if (candidate / "bundle.json").is_file():
            return load_bundle(candidate)
    return None


def load_gpt_artifacts(
    directory: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> GPTArtifacts:
    """Load a run directory, a ``gpt/`` directory, or just a model file."""
    path = _resolve_gpt_dir(directory)
    try:
        config_data = json.loads(
            (path / "model_config.json").read_text(encoding="utf-8")
        )
        config = GPTConfig.from_dict(config_data)
        model = MiniGPT(config)
        state = torch.load(
            path / "model.pt", map_location=map_location, weights_only=True
        )
        model.load_state_dict(state)
        model.eval()
        metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
        raise GPTPipelineError(f"cannot read GPT artifact {path}: {exc}") from exc
    return GPTArtifacts(
        directory=path,
        model=model,
        config=config,
        metadata=dict(metadata),
        tokenizer=_find_tokenizer(path),
    )


def find_gpt_dirs(root: str | Path | None = None) -> list[Path]:
    """Find ``*/gpt`` artifact directories under ``root``, newest first."""
    root = Path(root) if root is not None else REPO_ROOT / "artifacts"
    if not root.is_dir():
        return []
    found = [
        path
        for path in root.glob("*/" + DEFAULT_GPT_DIR)
        if (path / "model.pt").is_file()
    ]
    found.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return found
