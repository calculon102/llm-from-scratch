"""Tests for the corpus downloader/cache and manifest (T-601)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.corpus import (
    CorpusCacheError,
    CorpusSource,
    fetch_source,
    load_manifest,
    record_source,
    registered_sources,
)
from scripts.corpus import main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fetch_file_verifies_and_registers_without_clobbering(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("Grüße aus Köln.\n", encoding="utf-8")
    source = CorpusSource(
        name="tiny_de",
        url=str(raw),
        language="de",
        license="CC0-1.0",
        sha256=_sha256(raw),
        size_bytes=raw.stat().st_size,
    )
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "generated_at": "yesterday",
                "sources": [{"name": "existing"}],
                "corpus": {"path": "data/corpus_de"},
            }
        ),
        encoding="utf-8",
    )

    result = fetch_source(source, tmp_path / "cache")
    assert result.status == "downloaded"
    assert result.path.read_text(encoding="utf-8") == raw.read_text(encoding="utf-8")
    record = record_source(result, manifest)

    data = load_manifest(manifest)
    assert data["generated_at"] == "yesterday"
    assert data["sources"] == [{"name": "existing"}]
    assert data["corpus"] == {"path": "data/corpus_de"}
    assert data["cache"]["sources"]["tiny_de"]["sha256"] == source.sha256
    assert record["files"] == 1

    # A second fetch reuses the cache and refreshes the registration.
    again = fetch_source(source, tmp_path / "cache")
    assert again.status == "cached"
    record_source(again, manifest)
    assert registered_sources(manifest)["tiny_de"]["status"] == "cached"


def test_fetch_file_rejects_sha_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("text", encoding="utf-8")
    source = CorpusSource(name="bad", url=str(raw), sha256="0" * 64)

    with pytest.raises(CorpusCacheError, match="SHA-256 mismatch"):
        fetch_source(source, tmp_path / "cache")

    assert not (tmp_path / "cache" / "bad").exists()
    assert not list((tmp_path / "cache").glob("*.part"))


def test_offline_missing_source_has_actionable_error(tmp_path: Path) -> None:
    source = CorpusSource(name="missing", url="https://example.invalid/x.txt")
    with pytest.raises(CorpusCacheError, match="offline mode"):
        fetch_source(source, tmp_path / "cache", offline=True)


def test_fetch_directory_copies_tree_and_is_cached(tmp_path: Path) -> None:
    src = tmp_path / "sample"
    src.mkdir()
    (src / "a.txt").write_text("Hallo\n", encoding="utf-8")
    (src / "b.txt").write_text("Welt\n", encoding="utf-8")
    source = CorpusSource(name="sample", url=str(src), kind="directory")

    first = fetch_source(source, tmp_path / "cache")
    assert first.status == "copied"
    assert first.files == 2
    assert (first.path / "a.txt").is_file()

    second = fetch_source(source, tmp_path / "cache")
    assert second.status == "cached"
    assert second.sha256 == first.sha256


def test_cli_fetch_registers_source_and_status_reports_it(
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
    out = capsys.readouterr().out
    assert "registered in" in out
    assert registered_sources(manifest)["sample_de"]["files"] == 4

    assert main(["status", "--manifest", str(manifest)]) == 0
    status = capsys.readouterr().out
    assert "sample_de" in status
    assert "path:" in status
    assert "license: CC0-1.0" in status


def test_cli_offline_fails_when_cache_is_cold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "fetch",
            "--source",
            "sample_de",
            "--offline",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )
    assert code == 1
    assert "offline mode" in capsys.readouterr().err


def test_cli_rejects_unknown_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fetch", "--source", "nope"]) == 1
    assert "unknown corpus source" in capsys.readouterr().err


def test_cli_list_shows_default_marker(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "sample_de" in out
    assert "wikipedia_de_dump" in out
    assert "gutenberg_de_catalog" in out
