"""Tests for the end-to-end pipeline command (T-603).

Acceptance criterion: "single command runs full educational pipeline".
The tests exercise the real CLI entrypoint with a tiny GPT config: corpus
cleaning, tokenizer training, shared-Trainer GPT training, validation
perplexity, sample generation and the combined ``pipeline.json`` report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import core.gpt  # noqa: F401  # registers the gpt phase
from core.config import load_config
from core.gpt import load_gpt_artifacts, prepare_gpt_data
from core.tokenizer.corpus import CorpusOptions, ingest_corpus
from core.tokenizer.pipeline import load_bundle, train_bundle
from scripts.pipeline import main as pipeline_main


def _write_config(tmp_path: Path, **overrides) -> Path:
    cfg = load_config("configs/gpt_tiny_debug.yaml")
    cfg["output_dir"] = str(tmp_path / "runs")
    cfg["run_name"] = "pipeline_test"
    cfg["max_steps"] = 2
    cfg["epochs"] = 1
    cfg["batch_size"] = 2
    cfg["log_interval"] = 1
    cfg["save_interval"] = 100
    cfg["phase_config"].update(
        {
            "vocab_size": 48,
            "min_frequency": 1,
            "max_token_length": 8,
            "context_length": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
        }
    )
    cfg.update(overrides)
    path = tmp_path / "pipeline_test.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _write_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lines = []
    for index in range(80):
        lines.append(
            f"Die deutsche Sprache ist ein wichtiges Beispiel Nummer {index} "
            "für sauberen Text mit genügend Buchstaben."
        )
    lines.append("kurz")  # dropped by moderate_clean, kept by minimal_clean
    (corpus / "de.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return corpus


def _run(tmp_path: Path, corpus: Path, *extra: str) -> tuple[int, Path]:
    config = _write_config(tmp_path)
    output = tmp_path / "runs"
    code = pipeline_main(
        [
            "--config",
            str(config),
            "--input",
            str(corpus),
            "--output-dir",
            str(output),
            "--max-new-tokens",
            "8",
            "--top-k",
            "0",
            *extra,
        ]
    )
    run_dir = next(path for path in output.iterdir() if path.is_dir())
    return code, run_dir


def test_pipeline_writes_every_stage(tmp_path: Path, capsys) -> None:
    corpus = _write_corpus(tmp_path)
    code, run_dir = _run(tmp_path, corpus)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "pipeline:" in captured.out

    # Stage 1: reproducible preprocess artifacts.
    cleaned = (run_dir / "corpus" / "cleaned.txt").read_text(encoding="utf-8")
    assert "Die deutsche Sprache" in cleaned
    assert "kurz\n" not in cleaned
    assert (run_dir / "corpus" / "report.json").is_file()
    assert (run_dir / "corpus" / "manifest.json").is_file()

    # Stage 2: tokenizer bundle usable on its own.
    bundle = load_bundle(run_dir / "tokenizers")
    assert bundle.bpe.vocab_size <= 48
    assert bundle.char.vocab_size > 0

    # Stage 3: GPT artifacts load back.
    artifacts = load_gpt_artifacts(run_dir / "gpt")
    assert artifacts.model.num_parameters() > 0
    assert artifacts.metadata["tokens"] > 0
    assert artifacts.tokenizer is not None

    # Stage 4: evaluation + deterministic sample report.
    eval_report = json.loads(
        (run_dir / "eval" / "report.json").read_text(encoding="utf-8")
    )
    assert eval_report["metrics"]["val_loss"] > 0
    assert eval_report["metrics"]["val_perplexity"] > 1
    assert eval_report["generation"]["prompt"] == "Die Sprache ist"
    assert (run_dir / "eval" / "sample.txt").read_text(encoding="utf-8")

    # Combined provenance report.
    report = json.loads((run_dir / "pipeline.json").read_text(encoding="utf-8"))
    assert report["format"] == "llm-from-scratch-pipeline"
    assert report["status"] == "finished"
    assert set(report["stages"]) == {
        "preprocess",
        "tokenizer",
        "gpt_setup",
        "train",
        "eval",
    }
    assert report["stages"]["preprocess"]["profile"] == "moderate_clean"
    assert report["stages"]["train"]["steps"] == 2
    assert report["stages"]["gpt_setup"]["tokens"] > 0
    assert report["artifacts"]["model"] == "gpt/model.pt"
    manifest = json.loads(
        (run_dir / "corpus" / "manifest.json").read_text(encoding="utf-8")
    )
    assert report["stages"]["preprocess"]["sha256"] == manifest["output"]["sha256"]

    # The run is registered in the artifact index for the UI (T-604).
    assert (run_dir.parent / "runs.jsonl").is_file()


def test_pipeline_minimal_profile_keeps_short_lines(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path)
    code, run_dir = _run(tmp_path, corpus, "--profile", "minimal_clean")
    assert code == 0
    cleaned = (run_dir / "corpus" / "cleaned.txt").read_text(encoding="utf-8")
    assert "kurz" in cleaned.splitlines()


def test_pipeline_rejects_non_gpt_config(tmp_path: Path, capsys) -> None:
    code = pipeline_main(
        [
            "--config",
            "configs/xor_tiny_debug.yaml",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 1
    assert "expected 'gpt'" in capsys.readouterr().err


def test_prepare_gpt_data_accepts_prepared_corpus_and_tokenizer(tmp_path: Path) -> None:
    corpus_path = _write_corpus(tmp_path)
    corpus = ingest_corpus(
        corpus_path,
        CorpusOptions(
            min_line_chars=1,
            max_line_chars=0,
            drop_duplicates=False,
        ),
    )
    bundle = train_bundle(
        corpus.text, vocab_size=60, min_frequency=1, max_token_length=8
    )
    cfg = load_config("configs/gpt_tiny_debug.yaml")
    cfg["phase_config"].update(
        {
            "corpus_path": str(tmp_path / "does-not-exist"),
            "context_length": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
        }
    )
    data = prepare_gpt_data(cfg, corpus=corpus, tokenizer=bundle)
    expected = bundle.bpe.encode(corpus.text, add_special_tokens=True)
    assert data.token_ids == expected
    assert data.tokenizer is bundle
    assert data.datasets.train_windows > 0


def test_pipeline_options_reject_bad_generation_settings() -> None:
    from core.pipeline import PipelineError, PipelineOptions

    with pytest.raises(PipelineError, match="top_p"):
        PipelineOptions(top_p=0.0)
    with pytest.raises(PipelineError, match="max_new_tokens"):
        PipelineOptions(max_new_tokens=-1)
