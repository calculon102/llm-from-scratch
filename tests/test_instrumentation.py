"""Tests for throughput and memory instrumentation (T-502).

Acceptance: "tokens/sec and memory indicators logged".  The end-to-end test
runs the shared Trainer on a toy LM with a real run folder and asserts that
the console/run log carries ``tok/s`` + ``mem`` and that ``metrics.jsonl``
contains the machine-readable records.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from core.artifacts import create_run
from core.config import validate_config
from core.instrumentation import (
    MemoryStats,
    ThroughputMeter,
    format_memory,
    memory_stats,
)
from core.trainer import PhaseBundle, Trainer


class TinyLMDataset(Dataset):
    """16 windows of ``context_length=3`` over a tiny vocabulary."""

    def __init__(self, vocab_size: int = 8, length: int = 3, count: int = 16) -> None:
        self.windows = [
            torch.randint(0, vocab_size, (length + 1,)) for _ in range(count)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        return {"input_ids": window[:-1], "targets": window[1:]}


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 8, n_embd: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(input_ids))


def lm_loss(model: nn.Module, batch: dict) -> tuple[torch.Tensor, dict]:
    logits = model(batch["input_ids"])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch["targets"].reshape(-1))
    return loss, {}


def make_trainer(tmp_path) -> tuple[Trainer, object]:
    cfg = validate_config(
        {
            "run_name": "instrumentation",
            "output_dir": str(tmp_path / "artifacts"),
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "max_steps": 4,
            "log_interval": 2,
            "save_interval": 100,
            "profile": "tiny_debug",
        }
    )
    bundle = PhaseBundle(
        model=TinyLM(),
        train_dataset=TinyLMDataset(),
        loss_fn=lm_loss,
        token_count_fn=lambda batch: int(batch["targets"].numel()),
    )
    run = create_run(cfg)
    return Trainer.from_bundle(bundle, config=cfg, run=run), run


def test_memory_stats_and_format_on_cpu() -> None:
    stats = memory_stats(torch.device("cpu"))
    assert stats.cpu_peak_mb is not None and stats.cpu_peak_mb > 0
    if not torch.cuda.is_available():
        assert stats.gpu_peak_mb is None
    text = format_memory(stats)
    assert text.startswith("cpu=")
    assert "MiB" in text or "GiB" in text


def test_format_memory_with_gpu_values() -> None:
    stats = MemoryStats(
        cpu_peak_mb=512,
        gpu_allocated_mb=1024,
        gpu_reserved_mb=1536,
        gpu_peak_mb=2048,
        gpu_total_mb=24564,
    )
    text = format_memory(stats)
    assert "gpu=1.0GiB/1.5GiB" in text
    assert "peak=2.0GiB/24.0GiB" in text


def test_throughput_meter_counts_and_pauses() -> None:
    meter = ThroughputMeter()
    meter.update(100)
    meter.update(50)
    assert meter.tokens == 150
    assert meter.steps == 2
    assert meter.tokens_per_sec > 0

    meter.pause()
    elapsed = meter.elapsed_s
    meter.update(10)
    assert meter.elapsed_s == pytest.approx(elapsed, abs=1e-3)
    meter.resume()
    assert meter.tokens == 160


def test_trainer_logs_tokens_per_sec_and_memory(tmp_path) -> None:
    trainer, run = make_trainer(tmp_path)
    summary = trainer.train()
    run.mark_finished("finished")

    # 4 steps * batch 2 * context 3 = 24 predicted tokens.
    assert summary["steps"] == 4
    assert summary["tokens"] == 24
    assert summary["tokens_per_sec"] > 0
    assert summary["memory"]["cpu_peak_mb"] > 0

    log_text = run.log_path.read_text(encoding="utf-8")
    assert "memory at start:" in log_text
    assert "tok/s" in log_text
    assert "mem cpu=" in log_text
    assert "training finished:" in log_text

    metrics = [
        json.loads(line)
        for line in run.metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    step_records = [record for record in metrics if record["kind"] == "step"]
    summaries = [record for record in metrics if record["kind"] == "summary"]
    assert len(step_records) == 2
    assert len(summaries) == 1
    assert all(record["tokens"] > 0 for record in step_records)
    assert all(record["tokens_per_sec"] > 0 for record in step_records)
    assert all("cpu_peak_mb" in record for record in step_records)
    assert summaries[0]["tokens"] == 24
    assert summaries[0]["memory"]["cpu_peak_mb"] > 0


def test_non_lm_phase_reports_steps_only(tmp_path) -> None:
    """A phase without a token counter still runs; no bogus tok/s is logged."""

    class ScalarDataset(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict:
            return {"x": torch.tensor([1.0])}

    model = nn.Linear(1, 1)
    bundle = PhaseBundle(
        model=model,
        train_dataset=ScalarDataset(),
        loss_fn=lambda model, batch: (model(batch["x"]).sum() * 0, {}),
    )
    cfg = validate_config(
        {
            "run_name": "no_tokens",
            "output_dir": str(tmp_path / "artifacts"),
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "max_steps": 2,
            "log_interval": 1,
            "save_interval": 100,
        }
    )
    run = create_run(cfg)
    trainer = Trainer.from_bundle(bundle, config=cfg, run=run)
    summary = trainer.train()
    assert summary["tokens"] == 0
    log_text = run.log_path.read_text(encoding="utf-8")
    assert "tok/s" not in log_text
    assert "mem cpu=" in log_text
