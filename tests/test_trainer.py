"""Tests for the shared Trainer loop (T-005).

Two toy phases are registered to prove the Trainer is phase-agnostic:
an XOR MLP (the shape of the M1 phase) and a tiny sequence LM (the shape
of the M4 phase).  Both go through the exact same loop, config, and
checkpoint format.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.artifacts import create_run
from core.config import validate_config
from core.trainer import (
    PhaseBundle,
    Trainer,
    TrainerError,
    build_phase,
    register_phase,
    seed_everything,
    split_dataset,
)
from scripts.train import main as train_main


# ---------------------------------------------------------------------------
# Toy phase 1: XOR MLP (M1 shape)
# ---------------------------------------------------------------------------

class XORDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [(a, b, a ^ b) for a in (0, 1) for b in (0, 1)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        a, b, y = self.samples[i]
        return {
            "x": torch.tensor([a, b], dtype=torch.float32),
            "y": torch.tensor([y], dtype=torch.float32),
        }


def xor_factory(cfg: dict) -> PhaseBundle:
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))

    def loss_fn(model: nn.Module, batch: dict):
        logits = model(batch["x"])  # (B, 1)
        loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
        with torch.no_grad():
            acc = (
                ((torch.sigmoid(logits) >= 0.5).float() == batch["y"]).float().mean().item()
            )
        return loss, {"acc": acc}

    return PhaseBundle(
        model=model,
        train_dataset=XORDataset(),
        val_dataset=XORDataset(),
        loss_fn=loss_fn,
    )


# ---------------------------------------------------------------------------
# Toy phase 2: tiny sequence LM over a small DE corpus (M4 shape)
# ---------------------------------------------------------------------------

CORPUS = ["hallo welt", "gute bahn", "klein haus", "blau auto"]
VOCAB = {t: i for i, t in enumerate(sorted({t for s in CORPUS for t in s.split()}))}


class TokenLM(Dataset):
    def __init__(self, texts: list[str], vocab: dict[str, int]) -> None:
        self.items = [[vocab[t] for t in text.split()] for text in texts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        ids = self.items[i]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "target_ids": torch.tensor(ids[1:], dtype=torch.long),
        }


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 8, hidden: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self.emb(input_ids))
        return self.head(out)


def tiny_lm_factory(cfg: dict) -> PhaseBundle:
    vocab_size = len(VOCAB)
    model = TinyLM(vocab_size)

    def loss_fn(model: nn.Module, batch: dict):
        logits = model(batch["input_ids"])  # (B, T, V)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            batch["target_ids"].reshape(-1),
        )
        return loss, {}

    return PhaseBundle(
        model=model,
        train_dataset=TokenLM(CORPUS, VOCAB),
        val_dataset=TokenLM(CORPUS, VOCAB),
        loss_fn=loss_fn,
    )


# Both phases plug into the same registry the real phases will use.
register_phase("xor", xor_factory)
register_phase("gpt", tiny_lm_factory)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(tmp_path: Path, **overrides) -> dict:
    return validate_config(
        {
            "run_name": "trainer_test",
            "output_dir": str(tmp_path / "artifacts"),
            "device": "cpu",
            "seed": 42,
            "profile": "tiny_debug",
            "batch_size": 4,
            "learning_rate": 0.05,
            "epochs": 40,
            "log_interval": 5,
            "save_interval": 50,
            "val_split": 0.25,
            **overrides,
        }
    )


def make_trainer(tmp_path: Path, phase: str, run=None, trainer_kwargs=None, **overrides):
    cfg = make_cfg(tmp_path, phase=phase, **overrides)
    bundle = build_phase(cfg)
    return Trainer.from_bundle(
        bundle, config=cfg, run=run, **(trainer_kwargs or {})
    ), cfg


def list_run_dirs(root: Path) -> list[Path]:
    return [p for p in root.glob("*") if p.is_dir()]


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def test_xor_learns(tmp_path):
    trainer, cfg = make_trainer(tmp_path, "xor", epochs=60)
    summary = trainer.train()
    assert summary["steps"] == 60  # 4 samples / batch 4 = 1 step per epoch
    assert summary["epochs"] == 60
    val = trainer.evaluate()
    assert val["acc"] == 1.0
    assert val["val_loss"] < 0.05
    assert summary["best_val"] is not None and summary["best_val"] < 0.05


def test_tiny_lm_learns(tmp_path):
    trainer, _ = make_trainer(tmp_path, "gpt", epochs=150, learning_rate=0.02)
    before = trainer.evaluate()["val_loss"]
    trainer.train()
    after = trainer.evaluate()["val_loss"]
    assert after < before


def test_evaluate_falls_back_to_train_set(tmp_path):
    cfg = make_cfg(tmp_path, phase="xor", epochs=5)
    bundle = build_phase(cfg)
    trainer = Trainer(
        bundle.model,
        torch.optim.AdamW(bundle.model.parameters(), lr=cfg["learning_rate"]),
        bundle.train_dataset,
        None,  # no val set
        config=cfg,
        loss_fn=bundle.loss_fn,
    )
    metrics = trainer.evaluate()
    assert "train_loss" in metrics
    assert "val_loss" not in metrics


def test_evaluate_without_training(tmp_path):
    trainer, _ = make_trainer(tmp_path, "xor")
    metrics = trainer.evaluate()
    assert set(metrics) == {"val_loss", "acc"}


# ---------------------------------------------------------------------------
# Loop control
# ---------------------------------------------------------------------------

def test_max_steps_stops_early(tmp_path):
    trainer, _ = make_trainer(tmp_path, "xor", epochs=100, max_steps=7)
    summary = trainer.train()
    assert summary["steps"] == 7
    assert summary["epochs"] <= 7


def test_request_stop_interrupts(tmp_path):
    trainer, _ = make_trainer(
        tmp_path,
        "xor",
        epochs=100,
        log_interval=1,  # on_step fires every step
        trainer_kwargs={
            "callbacks": {
                "on_step": lambda t, step, **kw: t.request_stop() if step == 3 else None
            }
        },
    )
    summary = trainer.train()
    assert summary["steps"] == 3


def test_callbacks_fire(tmp_path):
    events: list = []
    trainer, _ = make_trainer(
        tmp_path,
        "xor",
        epochs=10,
        trainer_kwargs={
            "callbacks": {
                "on_start": lambda t: events.append("start"),
                "on_step": lambda t, step, **kw: events.append(step),
                "on_epoch_end": lambda t, epoch, **kw: events.append(epoch),
                "on_end": lambda t, summary, **kw: events.append("end"),
            }
        },
    )
    trainer.train()
    assert events[0] == "start"
    assert events[-1] == "end"
    assert 5 in events and 10 in events  # log_interval = 5
    assert 1 in events and 10 in events  # epoch ends


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def test_checkpoint_and_resume(tmp_path):
    cfg = make_cfg(tmp_path, phase="xor", epochs=10, save_interval=5)
    run = create_run(cfg)
    bundle = build_phase(cfg)
    trainer = Trainer.from_bundle(bundle, config=cfg, run=run)
    trainer.train()
    run.mark_finished()

    assert (run.checkpoints_dir / "step_000005.pt").is_file()
    assert (run.checkpoints_dir / "step_000010.pt").is_file()
    assert (run.checkpoints_dir / "final.pt").is_file()

    # Resume into a fresh trainer from the run directory.
    cfg2 = make_cfg(tmp_path, phase="xor", epochs=15, save_interval=5)
    run2 = create_run(cfg2)
    bundle2 = build_phase(cfg2)
    trainer2 = Trainer.from_bundle(bundle2, config=cfg2, run=run2)
    trainer2.resume(run.run_dir)
    assert trainer2.start_step == 10
    for p1, p2 in zip(bundle.model.parameters(), bundle2.model.parameters()):
        assert torch.equal(p1.detach().cpu(), p2.detach().cpu())

    summary = trainer2.train()
    # The step_000010 checkpoint was saved mid-loop, with epoch 9 still in
    # progress; resume restarts that epoch, so its one batch is re-processed
    # (16 total steps), and training stops after 15 completed epochs.
    assert summary["steps"] == 16
    assert summary["epochs"] == 15
    run2.mark_finished()
    final = torch.load(run2.checkpoints_dir / "final.pt", weights_only=True)
    assert final["step"] == 16
    assert final["epoch"] == 15


def test_resume_from_explicit_file(tmp_path):
    cfg = make_cfg(tmp_path, phase="xor", epochs=10, save_interval=5)
    run = create_run(cfg)
    bundle = build_phase(cfg)
    trainer = Trainer.from_bundle(bundle, config=cfg, run=run)
    trainer.train()
    run.mark_finished()

    bundle2 = build_phase(cfg)
    trainer2 = Trainer.from_bundle(bundle2, config=cfg)
    trainer2.resume(run.checkpoints_dir / "step_000005.pt")
    assert trainer2.start_step == 5


def test_resume_without_run_or_path_raises(tmp_path):
    trainer, _ = make_trainer(tmp_path, "xor")
    with pytest.raises(TrainerError, match="no run and no checkpoint path"):
        trainer.resume()


def test_resume_missing_checkpoint_raises(tmp_path):
    trainer, _ = make_trainer(tmp_path, "xor")
    with pytest.raises(TrainerError, match="checkpoint not found"):
        trainer.resume(tmp_path / "nope.pt")


def test_save_checkpoint_without_run_returns_none(tmp_path):
    trainer, _ = make_trainer(tmp_path, "xor")
    assert trainer.save_checkpoint(step=1) is None


# ---------------------------------------------------------------------------
# Phase registry
# ---------------------------------------------------------------------------

def test_build_phase_requires_phase(tmp_path):
    with pytest.raises(TrainerError, match="no 'phase'"):
        build_phase(make_cfg(tmp_path))


def test_build_phase_unregistered(tmp_path):
    with pytest.raises(TrainerError, match="no registered factory"):
        build_phase(make_cfg(tmp_path, phase="tokenizer"))


def test_register_phase_rejects_unknown_name():
    with pytest.raises(TrainerError, match="unknown phase"):
        register_phase("bogus", lambda cfg: None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_split_dataset_sizes_and_determinism(tmp_path):
    class Ranged(Dataset):
        def __len__(self) -> int:
            return 10

        def __getitem__(self, i: int) -> dict:
            return {"i": torch.tensor([i])}

    train_a, val_a = split_dataset(Ranged(), 0.25, seed=1)
    train_b, val_b = split_dataset(Ranged(), 0.25, seed=1)
    assert len(train_a) == 8 and len(val_a) == 2
    assert [train_a[i]["i"].item() for i in range(8)] == [
        train_b[i]["i"].item() for i in range(8)
    ]
    assert set(val_a[i]["i"].item() for i in range(2)) == set(
        val_b[i]["i"].item() for i in range(2)
    )


def test_split_dataset_no_val_when_too_small(tmp_path):
    class One(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, i: int) -> dict:
            return {"i": torch.tensor([i])}

    ds, val = split_dataset(One(), 0.25)
    assert ds is not None and val is None


def test_split_dataset_rejects_full_val(tmp_path):
    class Ranged(Dataset):
        def __len__(self) -> int:
            return 10

        def __getitem__(self, i: int) -> dict:
            return {"i": torch.tensor([i])}

    with pytest.raises(TrainerError, match="no training samples"):
        split_dataset(Ranged(), 1.0)


def test_seed_everything_is_deterministic():
    seed_everything(123)
    a = torch.randn(3, 3)
    seed_everything(123)
    b = torch.randn(3, 3)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# End-to-end via scripts.train
# ---------------------------------------------------------------------------

def write_cfg(tmp_path: Path, name: str, **overrides) -> Path:
    cfg = make_cfg(tmp_path, **overrides)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_train_script_end_to_end(tmp_path):
    cfg_path = write_cfg(tmp_path, "xor", phase="xor", epochs=10, save_interval=5)
    rc = train_main(["--config", str(cfg_path)])
    assert rc == 0

    run_dirs = list_run_dirs(tmp_path / "artifacts")
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "run.log").is_file()
    assert (run_dir / "checkpoints" / "final.pt").is_file()
    assert (run_dir / "checkpoints" / "step_000005.pt").is_file()
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished"
    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "training started" in log_text
    assert "training finished" in log_text
    final = torch.load(run_dir / "checkpoints" / "final.pt", weights_only=True)
    assert final["step"] == 10


def test_train_script_resume(tmp_path):
    cfg1 = write_cfg(tmp_path, "xor_a", phase="xor", epochs=5)
    assert train_main(["--config", str(cfg1)]) == 0
    run1 = list_run_dirs(tmp_path / "artifacts")[0]

    cfg2 = write_cfg(tmp_path, "xor_b", phase="xor", epochs=8)
    assert train_main(["--config", str(cfg2), "--resume", str(run1)]) == 0
    run2 = max(list_run_dirs(tmp_path / "artifacts"), key=lambda p: p.stat().st_mtime)
    assert run2 != run1
    log_text = (run2 / "run.log").read_text(encoding="utf-8")
    assert "resumed from" in log_text
    final = torch.load(run2 / "checkpoints" / "final.pt", weights_only=True)
    assert final["step"] == 8  # resumed at 5, trained 3 more steps


def test_train_script_rejects_unregistered_phase(tmp_path):
    cfg_path = write_cfg(tmp_path, "tok", phase="tokenizer")
    assert train_main(["--config", str(cfg_path)]) == 1
