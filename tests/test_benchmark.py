"""Tests for the M5 profile benchmark report (T-504).

Acceptance: "speed/quality comparison table is documented".  The tests run
a one-step slice of the smallest profile through the real GPT pipeline and
check the generated Markdown/JSON, plus the graceful skip for a profile
whose context does not fit the bundled sample corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from core.tokenizer import save_bundle, train_bundle
from scripts.benchmark import (
    BenchmarkResult,
    _parse_profiles,
    benchmark_profile,
    format_report,
    main,
)


def test_benchmark_tiny_debug_measures_throughput(tmp_path) -> None:
    result = benchmark_profile(
        "tiny_debug",
        steps=1,
        output_root=tmp_path,
        device=torch.device("cpu"),
        amp=False,
    )
    assert result.status == "ok", result.note
    assert result.steps == 1
    assert result.parameters > 0
    assert result.vocab_size > 0
    assert result.context_length == 16
    assert result.tokens and result.tokens > 0
    assert result.tokens_per_sec and result.tokens_per_sec > 0
    assert result.val_loss is not None
    assert result.cpu_peak_mb and result.cpu_peak_mb > 0
    assert Path(result.run_dir).is_dir()
    assert (Path(result.run_dir) / "metrics.jsonl").is_file()


def test_benchmark_reuses_tokenizer_bundle(tmp_path) -> None:
    """A supplied bundle must bypass BPE training (needed on big corpora)."""
    bundle = train_bundle(
        "Hallo Welt. Das ist ein Test. " * 80,
        vocab_size=60,
        source="benchmark-fixture",
    )
    save_bundle(bundle, tmp_path / "tokenizer")

    result = benchmark_profile(
        "tiny_debug",
        steps=1,
        output_root=tmp_path,
        device=torch.device("cpu"),
        tokenizer_bundle=str(tmp_path / "tokenizer"),
        amp=False,
    )
    assert result.status == "ok", result.note
    assert result.vocab_size == bundle.bpe.vocab_size
    meta = json.loads(
        (Path(result.run_dir) / "gpt" / "meta.json").read_text(encoding="utf-8")
    )
    assert meta["tokenizer_source"] == "benchmark-fixture"


def test_benchmark_skips_profiles_that_do_not_fit_the_corpus(tmp_path) -> None:
    result = benchmark_profile(
        "max_7900xtx",
        steps=1,
        output_root=tmp_path,
        device=torch.device("cpu"),
    )
    assert result.status == "skipped"
    assert "context window" in result.note


def test_format_report_has_comparison_table() -> None:
    results = [
        BenchmarkResult(
            profile="tiny_debug",
            status="ok",
            parameters=35776,
            context_length=16,
            batch_size=8,
            vocab_size=300,
            tokens_per_sec=12345.0,
            steps_per_sec=98.7,
            val_loss=4.5,
            val_perplexity=90.0,
            cpu_peak_mb=900.0,
        ),
        BenchmarkResult(
            profile="small",
            status="skipped",
            note="corpus too small",
        ),
    ]
    report = format_report(
        results,
        {
            "generated_at": "2026-11-15T00:00:00",
            "device": "device=cpu",
            "corpus": "data/sample_de",
            "steps": 10,
            "torch": "2.x",
        },
    )
    assert "| Profile | Params | Context | Batch | Vocab | Tok/s | Steps/s | Val loss | Perplexity | Peak mem | Status |" in report
    assert "| tiny_debug | 35,776 | 16 | 8 | 300 | 12345 | 98.7 | 4.5000 | 90.00 | 900 MiB CPU | ok |" in report
    assert "skipped: corpus too small" in report


def test_benchmark_cli_writes_report_and_json(tmp_path) -> None:
    rc = main(
        [
            "--profiles",
            "tiny_debug",
            "--steps",
            "1",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--amp",
        ]
    )
    assert rc == 0
    report_files = list(tmp_path.glob("*/report.md"))
    data_files = list(tmp_path.glob("*/benchmark.json"))
    assert len(report_files) == 1 and len(data_files) == 1

    report = report_files[0].read_text(encoding="utf-8")
    assert "| Profile |" in report
    assert "tiny_debug" in report
    assert "Tok/s" in report

    data = json.loads(data_files[0].read_text(encoding="utf-8"))
    assert data["steps"] == 1
    assert data["results"][0]["status"] == "ok"
    assert len(data["model_scale"]) == 5
    assert data["model_scale"][-1]["profile"] == "max_7900xtx"


def test_parse_profiles_supports_all_and_rejects_unknown() -> None:
    assert _parse_profiles(["all"]) == [
        "tiny_debug",
        "tiny_train",
        "small",
        "medium",
        "max_7900xtx",
    ]
    assert _parse_profiles(["tiny_train", "tiny_train"]) == ["tiny_train"]
    try:
        _parse_profiles(["huge"])
    except SystemExit as exc:
        assert "unknown profile" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown profile was accepted")
