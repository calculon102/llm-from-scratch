"""Tests for GPT training, perplexity evaluation and artifacts (T-404).

Acceptance criterion: "tiny overfit sanity + val perplexity logs".  The
training test deliberately runs a tiny model on the sample corpus long
enough to memorise the training windows, checks that the training loss
falls sharply, and checks that validation perplexity was logged and the
model/tokenizer artifacts can be loaded again.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import core.gpt
from core.config import ConfigError, load_config, validate_config
from core.gpt import (
    GPTConfig,
    LMChunkDataset,
    MiniGPT,
    build_gpt_phase,
    evaluate_perplexity,
    find_gpt_dirs,
    language_model_loss,
    load_gpt_artifacts,
    make_lm_dataloader,
    prepare_gpt_data,
    save_gpt_artifacts,
)
from core.trainer import Trainer, seed_everything


def tiny_cfg(tmp_path=None):
    cfg = load_config("configs/gpt_tiny_debug.yaml")
    cfg["phase_config"].update(
        {
            "vocab_size": 80,
            "context_length": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
        }
    )
    cfg["batch_size"] = 4
    cfg["val_split"] = 0.2
    if tmp_path is not None:
        cfg["output_dir"] = str(tmp_path)
        cfg["run_name"] = "gpt_test"
    return cfg


def mean_train_loss(model, loader):
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss = language_model_loss(model, batch)[0]
            total += float(loss) * int(batch["targets"].numel())
            tokens += int(batch["targets"].numel())
    model.train()
    return total / max(tokens, 1)


def test_effective_phase_config_and_validation() -> None:
    cfg = tiny_cfg()
    data = prepare_gpt_data(cfg)
    assert data.vocab_size <= 80
    assert len(data.token_ids) > 20
    assert data.datasets.train_windows > 0
    assert data.datasets.val_windows > 0

    with pytest.raises(ConfigError, match="divisible"):
        validate_config(
            {"phase": "gpt", "phase_config": {"n_embd": 10, "n_head": 4}}
        )
    with pytest.raises(ConfigError, match="unknown key"):
        validate_config({"phase": "gpt", "phase_config": {"width": 32}})


def test_phase_builds_model_with_tokenizer_vocabulary() -> None:
    cfg = tiny_cfg()
    bundle = build_gpt_phase(cfg)
    assert isinstance(bundle.model, MiniGPT)
    assert bundle.model.vocab_size == len(bundle.model.token_embedding.weight)
    assert bundle.model.context_length == 8
    assert bundle.metric_fn is evaluate_perplexity
    assert len(bundle.train_dataset) > 0


def test_language_model_loss_is_finite_and_backpropagates() -> None:
    model = MiniGPT(
        GPTConfig(vocab_size=11, context_length=5, n_layer=1, n_head=1, n_embd=8)
    )
    batch = {
        "input_ids": torch.randint(0, 11, (3, 5)),
        "targets": torch.randint(0, 11, (3, 5)),
    }
    loss, metrics = language_model_loss(model, batch)
    assert torch.isfinite(loss)
    assert metrics == {}
    loss.backward()
    assert any(
        parameter.grad is not None for parameter in model.parameters()
    )


def test_perplexity_is_token_weighted() -> None:
    # Zeroing every parameter makes the logits uniform, so the exact
    # perplexity must equal the vocabulary size.
    model = MiniGPT(
        GPTConfig(vocab_size=7, context_length=4, n_layer=1, n_head=1, n_embd=8)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    dataset = LMChunkDataset([i % 7 for i in range(20)], context_length=4)
    loader = make_lm_dataloader(dataset, batch_size=3)

    result = evaluate_perplexity(model, loader)
    assert result["val_loss"] == pytest.approx(math.log(7), rel=1e-6)
    assert result["val_perplexity"] == pytest.approx(7.0, rel=1e-6)


def test_artifacts_roundtrip(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    data = prepare_gpt_data(cfg)
    model = core.gpt.build_gpt_model(data)
    run = SimpleNamespace(run_dir=tmp_path, log=lambda message: None)

    directory = save_gpt_artifacts(
        run, model, data, metadata={"note": "roundtrip"}
    )
    assert (directory / "model.pt").is_file()
    assert (directory / "model_config.json").is_file()
    assert (directory / "meta.json").is_file()
    assert (directory / "corpus.txt").is_file()
    assert (tmp_path / "tokenizers" / "bundle.json").is_file()

    loaded = load_gpt_artifacts(tmp_path)
    assert loaded.tokenizer is not None
    assert loaded.metadata["note"] == "roundtrip"
    assert loaded.config == model.config
    idx = torch.randint(0, model.vocab_size, (2, 8))
    with torch.no_grad():
        assert torch.allclose(loaded.model(idx), model(idx), atol=1e-6)
    assert directory in find_gpt_dirs(tmp_path.parent)


def test_tiny_overfit_and_val_perplexity_logs(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    cfg["phase_config"].update(
        {"context_length": 16, "n_layer": 2, "n_head": 2, "n_embd": 32}
    )
    cfg.update(
        {
            "max_steps": 250,
            "epochs": 200,
            "batch_size": 8,
            "log_interval": 10,
            "save_interval": 0,
        }
    )
    seed_everything(cfg["seed"])
    bundle = build_gpt_phase(cfg)
    initial = mean_train_loss(bundle.model, make_lm_dataloader(bundle.train_dataset))
    assert initial > 3.0

    from core.artifacts import create_run

    run = create_run(cfg)
    trainer = Trainer.from_bundle(
        bundle, config=cfg, run=run, device=torch.device("cpu")
    )
    summary = trainer.train()
    run.mark_finished("finished")

    final = mean_train_loss(trainer.model, trainer.train_loader)
    assert summary["steps"] == 250
    assert final < initial - 2.0
    assert final < 2.0  # tiny model has memorised the training windows

    log_text = run.log_path.read_text(encoding="utf-8")
    assert "val_perplexity" in log_text
    assert "best_val" in log_text

    loaded = load_gpt_artifacts(run.run_dir)
    assert loaded.tokenizer is not None
    assert loaded.metadata["train_windows"] == len(bundle.train_dataset)
    assert loaded.metadata["best_val"] is not None
