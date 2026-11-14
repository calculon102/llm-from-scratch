"""Tests for the tokenizer CLI (T-201…T-204).

The CLI is the reproducible entrypoint behind the UI: one run folder with
the cleaned corpus, the persisted char/BPE bundles and the metrics file.
These tests use a tiny noisy corpus in ``tmp_path`` so they exercise the
same path without depending on the bundled sample data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from core.tokenizer import load_bundle
from scripts.tokenizer import main


def _write_config(tmp_path: Path, corpus: Path, **phase_overrides: object) -> Path:
    phase = {
        "corpus_path": str(corpus),
        "vocab_size": 80,
        "min_frequency": 1,
        "max_token_length": 16,
        "metrics_sample_chars": 4000,
        **phase_overrides,
    }
    cfg = {
        "run_name": "tokenizer_cli_test",
        "seed": 42,
        "device": "cpu",
        "profile": "tiny_debug",
        "output_dir": str(tmp_path / "artifacts"),
        "phase": "tokenizer",
        "phase_config": phase,
    }
    path = tmp_path / "tokenizer.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_cli_writes_corpus_bundle_and_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    # A little noise: blank line, a one-character line, a duplicate line,
    # and a CP1252 file.  The CLI must still produce usable artifacts.
    (corpus_dir / "a.txt").write_text(
        "Hallo Welt.\nHallo Welt.\n\nx\nGrüße aus Köln!\n", encoding="utf-8"
    )
    (corpus_dir / "b.txt").write_bytes("Straße und Fußweg.\n".encode("cp1252"))

    cfg_path = _write_config(tmp_path, corpus_dir)
    assert main(["--config", str(cfg_path)]) == 0

    run_dir = Path(capsys.readouterr().out.strip().split("run directory: ")[-1])
    assert run_dir.is_dir()

    cleaned = run_dir / "corpus" / "cleaned.txt"
    report = run_dir / "corpus" / "report.json"
    assert cleaned.is_file() and report.is_file()
    text = cleaned.read_text(encoding="utf-8")
    assert "Hallo Welt." in text
    # Duplicate line dropped, junk line dropped.
    assert text.count("Hallo Welt.") == 1
    assert "\nx\n" not in f"\n{text}\n"
    report_data = json.loads(report.read_text(encoding="utf-8"))
    assert report_data["stats"]["files_ok"] == 2
    assert report_data["stats"]["duplicate_dropped"] == 1

    bundle = load_bundle(run_dir / "tokenizers")
    assert bundle.char.vocab_size > 4
    assert bundle.bpe.vocab_size > bundle.char.vocab_size
    assert bundle.bpe.merges
    assert bundle.bpe.decode(bundle.bpe.encode(text)) == text

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["char"]["roundtrip_ok"] is True
    assert metrics["bpe"]["roundtrip_ok"] is True
    assert metrics["bpe"]["compression_ratio"] > metrics["char"]["compression_ratio"]
    assert metrics["bpe"]["oov_chars_total"] == 0


def test_cli_input_and_vocab_size_overrides(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    # Many distinct words sharing prefixes/digit suffixes give BPE enough
    # pair candidates to reach the requested 60 tokens.
    (corpus_dir / "a.txt").write_text(
        " ".join(f"wort{i:03d}" for i in range(100)) + "\n", encoding="utf-8"
    )
    cfg_path = _write_config(tmp_path, corpus_dir, vocab_size=40)

    assert (
        main(
            [
                "--config",
                str(cfg_path),
                "--input",
                str(corpus_dir),
                "--vocab-size",
                "60",
            ]
        )
        == 0
    )
    run_dir = max((tmp_path / "artifacts").iterdir(), key=lambda p: p.stat().st_mtime)
    bundle = load_bundle(run_dir / "tokenizers")
    assert bundle.bpe.vocab_size == 60


def test_cli_rejects_non_tokenizer_phase(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("Hallo Welt.\n", encoding="utf-8")
    cfg_path = _write_config(tmp_path, corpus_dir)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["phase"] = "xor"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert main(["--config", str(cfg_path)]) == 1
