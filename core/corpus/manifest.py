"""Non-destructive cache manifest updates (T-601).

``data/corpus_manifest.json`` predates this ticket: it documents the
staged German corpus (sizes, sources, licences).  The downloader must not
erase that information, so updates are additive: known top-level keys are
preserved, and a ``cache`` section records every fetched source with its
URL, local path, size, SHA-256 and timestamp.

Writes are atomic (temporary file + :func:`os.replace`) because the
manifest is the reproducibility record for processed corpora (T-602).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.corpus.cache import FetchResult
from core.corpus.sources import REPO_ROOT

__all__ = [
    "ManifestError",
    "load_manifest",
    "record_source",
    "registered_sources",
]

DEFAULT_MANIFEST = "data/corpus_manifest.json"


class ManifestError(Exception):
    """Raised when the corpus manifest cannot be read or written."""


def _display_path(path: Path, repo_root: Path) -> str:
    """Prefer a portable repository-relative path when possible."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Read a manifest, returning ``{}`` when it does not exist yet."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    if not resolved.exists():
        return {}
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read corpus manifest {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(
            f"corpus manifest {resolved} must contain a JSON object, "
            f"got {type(data).__name__}"
        )
    return data


def registered_sources(
    path: str | Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, dict[str, Any]]:
    """Return the ``cache.sources`` mapping from a manifest (may be empty)."""
    data = load_manifest(path, repo_root=repo_root)
    cache = data.get("cache")
    if not isinstance(cache, dict):
        return {}
    sources = cache.get("sources")
    return sources if isinstance(sources, dict) else {}


def record_source(
    result: FetchResult,
    path: str | Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Register one fetched source in the manifest and return the record.

    Existing top-level keys (``sources``, ``corpus``, ``quick``, ...) and
    unrelated cache entries are preserved.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    data = load_manifest(resolved, repo_root=repo_root)
    cache = data.get("cache")
    if not isinstance(cache, dict):
        cache = {}
    sources = cache.get("sources")
    if not isinstance(sources, dict):
        sources = {}

    source = result.source
    record = {
        "name": source.name,
        "url": source.url,
        "path": _display_path(result.path, repo_root),
        "kind": source.kind,
        "language": source.language,
        "license": source.license,
        "description": source.description,
        "bytes": result.bytes,
        "files": result.files,
        "sha256": result.sha256,
        "status": result.status,
        "fetched_at": result.fetched_at,
    }
    sources[source.name] = record
    cache.update(
        {
            "path": _display_path(
                result.path.parent, repo_root
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": sources,
        }
    )
    data["cache"] = cache

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
    return record
