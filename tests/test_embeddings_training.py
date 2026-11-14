"""Tests for the skip-gram model and Trainer integration (T-302).

The ticket's acceptance criterion is "loss decreases on tiny/small
profiles".  The end-to-end test below builds the real phase from a
validated config, runs the shared Trainer, and checks that the training
loss after the run is lower than before it.  It also verifies that the
``on_end`` callback persisted a self-contained, loadable artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import core.embeddings  # noqa: F401  # registers schema + phase
from core.config import ConfigError, validate_config
from core.embeddings import (
    SkipGramModel,
    build_embeddings_phase,
    load_embedding_artifacts,
    prepare_embedding_data,
)
from core.embeddings.pipeline import save_embedding_artifacts
from core.trainer import Trainer, build_phase
from core.artifacts import create_run


GERMAN_TEXT = "\n".join(
    f"Die Straße war frei, und die Fußgänger gingen am Tag {index} "
    f"über den Platz. Grüße aus Köln an alle Besucherinnen und Besucher."
    for index in range(40)
)


def write_corpus(tmp_path: Path, text: str = GERMAN_TEXT) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return path


def embeddings_config(corpus: Path, output_dir: Path, **overrides) -> dict:
    data = {
        "run_name": "embeddings_test",
        "seed": 7,
        "device": "cpu",
        "amp": False,
        "gradcheck": False,
        "profile": "tiny_debug",
        "output_dir": str(output_dir),
        "log_interval": 1000,
        "batch_size": 8,
        "learning_rate": 0.05,
        "epochs": 8,
        "max_steps": 200,
        "val_split": 0.0,
        "save_interval": 1000,
        "phase": "embeddings",
        "phase_config": {
            "corpus_path": str(corpus),
            "vocab_size": 120,
            "min_frequency": 1,
            "max_token_length": 10,
            "embedding_dim": 16,
            "window_size": 3,
            "dynamic_window": True,
            "negative_samples": 5,
            "negative_power": 0.75,
            "max_pairs": 0,
        },
    }
    data.update(overrides)
    return validate_config(data)


# ---------------------------------------------------------------------------
# Model objective
# ---------------------------------------------------------------------------

def test_model_shapes_loss_and_gradients() -> None:
    torch.manual_seed(0)
    model = SkipGramModel(vocab_size=17, embedding_dim=8)
    center = torch.tensor([0, 1, 2, 3])
    context = torch.tensor([4, 5, 6, 7])
    negatives = torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]])

    loss, metrics = model(center, context, negatives)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert set(metrics) == {"acc", "pos_acc", "neg_acc"}
    assert 0.0 <= metrics["acc"] <= 1.0

    loss.backward()
    assert model.center_embeddings.weight.grad is not None
    assert model.context_embeddings.weight.grad is not None
    assert torch.isfinite(model.center_embeddings.weight.grad).all()
    assert float(model.center_embeddings.weight.grad.abs().sum()) > 0


def test_initial_logits_are_not_saturated() -> None:
    # With sensible initialisation the initial loss is close to 2*ln(2):
    # logsigmoid(0) = -ln(2), and the positive + negative terms each give
    # one such term.  This catches the common nn.Embedding N(0,1) mistake.
    torch.manual_seed(0)
    model = SkipGramModel(vocab_size=200, embedding_dim=32)
    center = torch.arange(64) % 200
    context = (torch.arange(64) + 17) % 200
    negatives = (torch.arange(64 * 5).reshape(64, 5) + 31) % 200
    loss, _ = model(center, context, negatives)
    assert float(loss.detach()) == pytest.approx(1.386, abs=0.08)


def test_model_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        SkipGramModel(vocab_size=0, embedding_dim=8)
    with pytest.raises(ValueError):
        SkipGramModel(vocab_size=10, embedding_dim=0)


# ---------------------------------------------------------------------------
# Config / pipeline
# ---------------------------------------------------------------------------

def test_embeddings_phase_schema_is_registered() -> None:
    cfg = validate_config(
        {
            "phase": "embeddings",
            "phase_config": {"embedding_dim": 24, "window_size": 2},
        }
    )
    assert cfg["phase_config"]["embedding_dim"] == 24
    with pytest.raises(ConfigError, match="unknown key"):
        validate_config({"phase": "embeddings", "phase_config": {"dimm": 4}})
    with pytest.raises(ConfigError, match="below minimum"):
        validate_config({"phase": "embeddings", "phase_config": {"window_size": 0}})


def test_prepare_embedding_data_builds_aligned_vocab_and_pairs(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path)
    cfg = embeddings_config(corpus, tmp_path / "artifacts")
    data = prepare_embedding_data(cfg)

    assert data.vocab_size == len(data.content_ids) == len(data.tokens)
    assert data.vocab_size < data.tokenizer.bpe.vocab_size
    assert all(token not in {"<pad>", "<unk>", "<bos>", "<eos>"} for token in data.tokens)
    assert len(data.dataset) > 100
    assert data.dataset.vocab_size == data.tokenizer.bpe.vocab_size
    assert data.corpus is not None and "Straße" in data.corpus.text
    assert 2 <= len(data.token_ids)


def test_prepare_embedding_data_can_reuse_a_saved_bundle(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path)
    cfg = embeddings_config(corpus, tmp_path / "artifacts")
    first = prepare_embedding_data(cfg)

    from core.tokenizer import save_bundle

    bundle_dir = tmp_path / "bundle"
    save_bundle(first.tokenizer, bundle_dir)
    cfg["phase_config"]["tokenizer_bundle"] = str(bundle_dir)
    second = prepare_embedding_data(cfg)

    assert second.tokens == first.tokens
    assert second.tokenizer.path == bundle_dir
    assert second.tokenizer.bpe.vocab_size == first.tokenizer.bpe.vocab_size
    assert [pair for pair in second.dataset.pairs] == [pair for pair in first.dataset.pairs]


def test_phase_builder_splits_pairs_and_attaches_callback(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path)
    cfg = embeddings_config(
        corpus, tmp_path / "artifacts", val_split=0.2, phase_config={
            **embeddings_config(corpus, tmp_path / "artifacts")["phase_config"],
            "embedding_dim": 8,
        }
    )
    bundle = build_phase(cfg)
    assert isinstance(bundle.model, SkipGramModel)
    assert bundle.model.embedding_dim == 8
    assert bundle.val_dataset is not None
    total = len(bundle.train_dataset) + len(bundle.val_dataset)
    assert total > 0
    assert "on_end" in (bundle.callbacks or {})


# ---------------------------------------------------------------------------
# End-to-end training + artifacts
# ---------------------------------------------------------------------------

def test_training_decreases_loss_and_writes_artifacts(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path)
    cfg = embeddings_config(corpus, tmp_path / "artifacts")
    run = create_run(cfg)
    bundle = build_embeddings_phase(cfg)
    trainer = Trainer.from_bundle(bundle, config=cfg, run=run, device=torch.device("cpu"))

    before = trainer.evaluate()["train_loss"]
    summary = trainer.train()
    after = trainer.evaluate()["train_loss"]

    assert summary["steps"] > 0
    assert after < before, f"loss did not decrease: before={before:.4f} after={after:.4f}"

    embeddings_dir = Path(run.run_dir) / "embeddings"
    assert (embeddings_dir / "vectors.npy").is_file()
    assert (embeddings_dir / "vocab.json").is_file()
    assert (embeddings_dir / "meta.json").is_file()
    assert (embeddings_dir / "corpus.txt").is_file()
    assert (Path(run.run_dir) / "tokenizers" / "bundle.json").is_file()

    artifacts = load_embedding_artifacts(Path(run.run_dir))
    assert artifacts.vectors.dtype == np.float32
    assert artifacts.vectors.shape == (artifacts.vocab_size, 16)
    assert artifacts.vocab_size == len(artifacts.tokens)
    assert artifacts.metadata["format"] == "llm-from-scratch-embeddings"
    assert artifacts.metadata["embedding_dim"] == 16
    assert artifacts.metadata["pairs"] > 0
    assert artifacts.metadata["steps"] == summary["steps"]


def test_save_filters_special_rows_and_aligns_tokens(tmp_path: Path) -> None:
    class FakeRun:
        run_dir = tmp_path / "run"

        def __init__(self) -> None:
            self.messages: list[str] = []

        def log(self, message: str, level: int = 0) -> None:
            self.messages.append(message)

    model = SkipGramModel(vocab_size=5, embedding_dim=3)
    with torch.no_grad():
        model.vectors.copy_(torch.arange(15).reshape(5, 3).float())
    run = FakeRun()

    directory = save_embedding_artifacts(
        run,
        model,
        ["a", "b", "c"],
        vector_ids=[1, 3, 4],
        tokenizer=None,
        corpus_text=None,
    )
    artifacts = load_embedding_artifacts(directory)
    assert artifacts.tokens == ["a", "b", "c"]
    assert artifacts.vectors[:, 0].tolist() == [3.0, 9.0, 12.0]
