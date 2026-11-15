"""Tests for reproducible preprocessing profiles (T-602)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.corpus import (
    PREPROCESS_PROFILES,
    PreprocessError,
    get_profile,
    preprocess_corpus,
    record_preprocess,
)
from scripts.corpus import main


def _corpus_dir(tmp_path: Path) -> Path:
    corpus = tmp_path / "raw"
    corpus.mkdir()
    (corpus / "a.txt").write_text(
        "Hallo Welt und Grüße.\n"
        "Hallo Welt und Grüße.\n"
        "x\n"
        "1234 5678\n"
        "Ein kurzer Satz.\n"
        "Noch ein ordentlicher deutscher Satz mit Umlauten: Äpfel, Öl, Übermut.\n"
        + ("L" * 2100)
        + "\n",
        encoding="utf-8",
    )
    (corpus / "b.txt").write_bytes("Straße und Fußweg.\n".encode("cp1252"))
    return corpus


def test_profiles_are_registered():
    assert set(PREPROCESS_PROFILES) == {"minimal_clean", "moderate_clean"}
    assert get_profile("minimal_clean").options.drop_duplicates is False
    assert get_profile("moderate_clean").options.drop_duplicates is True
    with pytest.raises(PreprocessError, match="unknown preprocess profile"):
        get_profile("aggressive")


def test_minimal_keeps_short_and_duplicate_lines(tmp_path: Path) -> None:
    corpus = _corpus_dir(tmp_path)
    result = preprocess_corpus(corpus, "minimal_clean", tmp_path / "out_min")
    text = result.cleaned_path.read_text(encoding="utf-8")
    assert text.count("Hallo Welt und Grüße.") == 2
    assert "\nx\n" in f"\n{text}"
    assert "1234 5678" in text
    # No truncation in the minimal profile.
    assert "L" * 2100 in text


def test_moderate_drops_noise_duplicates_and_truncates(tmp_path: Path) -> None:
    corpus = _corpus_dir(tmp_path)
    result = preprocess_corpus(corpus, "moderate_clean", tmp_path / "out_mod")
    text = result.cleaned_path.read_text(encoding="utf-8")
    assert text.count("Hallo Welt und Grüße.") == 1
    assert "\nx\n" not in f"\n{text}"
    assert "1234 5678" not in text
    assert "L" * 2100 not in text
    assert "Straße und Fußweg." in text
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["profile"]["name"] == "moderate_clean"
    assert payload["output"]["sha256"] == result.output_sha256
    assert payload["stats"]["duplicate_dropped"] >= 1


def test_preprocessing_is_byte_for_byte_reproducible(tmp_path: Path) -> None:
    corpus = _corpus_dir(tmp_path)
    first = preprocess_corpus(corpus, "moderate_clean", tmp_path / "one")
    second = preprocess_corpus(corpus, "moderate_clean", tmp_path / "two")
    for name in ("cleaned.txt", "report.json", "manifest.json"):
        assert (first.output_dir / name).read_bytes() == (
            second.output_dir / name
        ).read_bytes(), name
    assert first.output_sha256 == second.output_sha256
    assert first.input_digest == second.input_digest


def test_register_preprocess_preserves_existing_manifest(tmp_path: Path) -> None:
    corpus = _corpus_dir(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"generated_at": "old", "sources": [{"name": "x"}]}),
        encoding="utf-8",
    )
    result = preprocess_corpus(corpus, "minimal_clean", tmp_path / "out")
    record = record_preprocess(result, manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["generated_at"] == "old"
    assert data["sources"] == [{"name": "x"}]
    assert data["preprocessed"]["minimal_clean"]["output_sha256"] == (
        result.output_sha256
    )
    assert record["inputs"]


def test_cli_preprocess_writes_and_registers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus_dir(tmp_path)
    out = tmp_path / "processed"
    manifest = tmp_path / "manifest.json"
    argv = [
        "preprocess",
        "--profile",
        "moderate_clean",
        "--input",
        str(corpus),
        "--output",
        str(out),
        "--manifest",
        str(manifest),
    ]
    assert main(argv) == 0
    assert (out / "cleaned.txt").is_file()
    assert (out / "report.json").is_file()
    assert (out / "manifest.json").is_file()
    first_sha = json.loads((out / "manifest.json").read_text())["output"]["sha256"]

    # A second run is reproducible and refreshes the registration.
    assert main(argv) == 0
    second_sha = json.loads((out / "manifest.json").read_text())["output"]["sha256"]
    assert first_sha == second_sha

    assert main(["status", "--manifest", str(manifest)]) == 0
    status = capsys.readouterr().out
    assert "preprocessed corpora" in status
    assert "moderate_clean" in status


def test_cli_preprocess_uses_fetched_sources_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    manifest = tmp_path / "manifest.json"
    assert (
        main(
            [
                "fetch",
                "--source",
                "sample_de",
                "--cache-dir",
                str(cache),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "preprocess",
                "--profile",
                "moderate_clean",
                "--output",
                str(tmp_path / "processed"),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    text = (tmp_path / "processed" / "cleaned.txt").read_text(encoding="utf-8")
    assert "Sprache" in text or "Hallo" in text
