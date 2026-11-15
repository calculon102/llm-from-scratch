"""Reproducible corpus preprocessing profiles (T-602).

Two profiles sit on top of the T-201 ingestion pipeline:

``minimal_clean``
    Decode (UTF-8/CP1252 fallback), NFC-normalize, collapse whitespace and
    drop empty lines.  Nothing else: short lines, repeated boilerplate and
    casing all survive, so this is the profile to use when you want to
    inspect or experiment with raw-ish text.

``moderate_clean``
    Adds the usual corpus hygiene: very short lines (page numbers, menu
    crumbs) and low-alphabetic lines (tables of numbers, navigation) are
    dropped, repeated lines are deduplicated globally, and pathological
    long lines are truncated.  German casing, umlauts and ``ß`` are kept.

Every preprocessed corpus is a directory with byte-identical outputs for
byte-identical inputs and profile::

    <output>/
    ├── cleaned.txt    # the corpus stream used for tokenizer training
    ├── report.json    # per-file decoding/filter statistics
    └── manifest.json  # profile, input hashes, output hash/size

``manifest.json`` deliberately contains no timestamps: rerunning the same
command on the same inputs reproduces it exactly.  Timestamps and the
registration of the corpus in ``data/corpus_manifest.json`` live outside
the reproducible payload.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from core.corpus.manifest import ManifestError, load_manifest
from core.corpus.sources import REPO_ROOT
from core.tokenizer.common import TokenizerError
from core.tokenizer.corpus import (
    Corpus,
    CorpusOptions,
    ingest_corpus,
    iter_corpus_files,
)

__all__ = [
    "PREPROCESS_PROFILES",
    "PreprocessError",
    "PreprocessProfile",
    "PreprocessResult",
    "get_profile",
    "preprocess_corpus",
    "record_preprocess",
]

PROFILE_SCHEMA_VERSION = 1


class PreprocessError(Exception):
    """Raised when a profile is unknown or a processed corpus cannot be written."""


@dataclass(frozen=True)
class PreprocessProfile:
    """One named cleaning configuration."""

    name: str
    description: str
    options: CorpusOptions


PREPROCESS_PROFILES: dict[str, PreprocessProfile] = {
    "minimal_clean": PreprocessProfile(
        name="minimal_clean",
        description=(
            "Decode, NFC-normalize, collapse whitespace and drop empty lines; "
            "keep short lines, duplicates and original casing."
        ),
        options=CorpusOptions(
            min_line_chars=1,
            max_line_chars=0,
            min_alpha_ratio=0.0,
            drop_duplicates=False,
            lowercase=False,
        ),
    ),
    "moderate_clean": PreprocessProfile(
        name="moderate_clean",
        description=(
            "Drop short/low-alpha lines and duplicate boilerplate, truncate "
            "pathological lines; keep German casing and orthography."
        ),
        options=CorpusOptions(
            min_line_chars=12,
            max_line_chars=2000,
            min_alpha_ratio=0.3,
            drop_duplicates=True,
            lowercase=False,
        ),
    ),
}


def get_profile(name: str) -> PreprocessProfile:
    """Look up a preprocessing profile by name."""
    try:
        return PREPROCESS_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(PREPROCESS_PROFILES)
        raise PreprocessError(
            f"unknown preprocess profile {name!r}; choose one of: {known}"
        ) from exc


@dataclass(frozen=True)
class PreprocessResult:
    """A written processed corpus plus its reproducibility metadata."""

    profile: PreprocessProfile
    output_dir: Path
    cleaned_path: Path
    report_path: Path
    manifest_path: Path
    corpus: Corpus
    input_digest: str
    output_sha256: str
    output_bytes: int

    def manifest(self) -> dict:
        """The deterministic ``manifest.json`` payload."""
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile": {
                "name": self.profile.name,
                "description": self.profile.description,
                "options": _jsonable(asdict(self.profile.options)),
            },
            "inputs": [
                {
                    "path": _display_path(Path(report.path)),
                    "status": report.status,
                    "reason": report.reason,
                    "sha256": _file_digest(Path(report.path)),
                    "bytes": report.bytes,
                }
                for report in self.corpus.files
            ],
            "input_digest": self.input_digest,
            "output": {
                "cleaned": "cleaned.txt",
                "sha256": self.output_sha256,
                "bytes": self.output_bytes,
                "chars": len(self.corpus.text),
                "lines": self.corpus.stats.get("lines_out", 0),
                "files": self.corpus.stats.get("files_ok", 0),
            },
            "stats": _jsonable(self.corpus.stats),
        }


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _input_digest(inputs: Iterable[Path]) -> str:
    """Hash the ordered file list and every file's content."""
    digest = hashlib.sha256()
    for path in inputs:
        digest.update(_display_path(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise PreprocessError(f"cannot write {path}: {exc}") from exc


def preprocess_corpus(
    inputs: str | Path | Iterable[str | Path],
    profile: str | PreprocessProfile = "moderate_clean",
    output_dir: str | Path = "data/processed",
) -> PreprocessResult:
    """Clean ``inputs`` with ``profile`` and write a reproducible corpus.

    The same inputs and profile always produce the same ``cleaned.txt``
    and ``manifest.json`` bytes; the directory is overwritten in place.
    """
    prof = get_profile(profile) if isinstance(profile, str) else profile
    out = Path(output_dir).expanduser()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreprocessError(f"cannot create output directory {out}: {exc}") from exc

    # Resolve the exact file list up front so the input digest describes
    # what ingestion will actually read.
    try:
        files = list(iter_corpus_files(inputs, prof.options.extensions))
    except TokenizerError as exc:
        raise PreprocessError(str(exc)) from exc
    if not files:
        raise PreprocessError(
            f"no .txt files found under {inputs!r}; check the path or add "
            "sources with: python -m scripts.corpus fetch"
        )

    try:
        corpus = ingest_corpus(inputs, prof.options)
    except TokenizerError as exc:
        raise PreprocessError(
            f"no usable text after cleaning with {prof.name!r}: {exc}"
        ) from exc

    cleaned_path = out / "cleaned.txt"
    report_path = out / "report.json"
    manifest_path = out / "manifest.json"
    _write_atomic(cleaned_path, corpus.text)
    _write_atomic(
        report_path,
        json.dumps(corpus.report(), indent=2, ensure_ascii=False) + "\n",
    )

    output_bytes = cleaned_path.stat().st_size
    output_sha256 = _file_digest(cleaned_path)
    result = PreprocessResult(
        profile=prof,
        output_dir=out,
        cleaned_path=cleaned_path,
        report_path=report_path,
        manifest_path=manifest_path,
        corpus=corpus,
        input_digest=_input_digest(files),
        output_sha256=output_sha256,
        output_bytes=output_bytes,
    )
    _write_atomic(
        manifest_path,
        json.dumps(result.manifest(), indent=2, ensure_ascii=False) + "\n",
    )
    return result


def record_preprocess(
    result: PreprocessResult,
    manifest_path: str | Path = "data/corpus_manifest.json",
    *,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Register a processed corpus in ``data/corpus_manifest.json``.

    Additive like the T-601 cache records: existing keys and other
    profile entries are preserved.
    """
    resolved = Path(manifest_path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    data = load_manifest(resolved, repo_root=repo_root)
    preprocessed = data.get("preprocessed")
    if not isinstance(preprocessed, dict):
        preprocessed = {}
    preprocessed[result.profile.name] = {
        "path": _display_path(result.output_dir),
        "input_digest": result.input_digest,
        "output_sha256": result.output_sha256,
        "bytes": result.output_bytes,
        "chars": len(result.corpus.text),
        "lines": result.corpus.stats.get("lines_out", 0),
        "files": result.corpus.stats.get("files_ok", 0),
        "inputs": [report.path for report in result.corpus.files],
        "profile_options": _jsonable(asdict(result.profile.options)),
    }
    data["preprocessed"] = preprocessed

    tmp = resolved.with_name(resolved.name + ".tmp")
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, resolved)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ManifestError(f"cannot write corpus manifest {resolved}: {exc}") from exc
    return preprocessed[result.profile.name]
