"""Download corpora into a local, verifiable cache (T-601).

All fetching is plain :mod:`urllib` — no external Python packages, no
APIs.  The cache is a directory of files/trees keyed by source name.  A
download is written to a temporary ``.part`` entry and atomically renamed,
so an interrupted fetch never looks like a valid cached source.

``file://`` URLs and plain local paths are supported, which makes the
whole flow testable and usable on an offline machine:

    corpus fetch                 # bundled sample_de, no network
    corpus fetch --all           # also the large Wikipedia/Gutenberg sources
    corpus fetch --offline       # cache-only: fail if something is missing
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from core.corpus.sources import REPO_ROOT, CorpusSource, source_path

__all__ = [
    "CorpusCacheError",
    "FetchResult",
    "cache_path",
    "fetch_source",
    "sha256_file",
    "sha256_tree",
]

_CHUNK_SIZE = 1024 * 1024
_DEFAULT_TIMEOUT = 60


class CorpusCacheError(Exception):
    """Raised when a source cannot be fetched or a cached copy is invalid."""


@dataclass(frozen=True)
class FetchResult:
    """Outcome of fetching one source."""

    source: CorpusSource
    path: Path
    status: str  # "downloaded" | "cached" | "copied"
    bytes: int
    files: int
    sha256: str | None
    fetched_at: str

    @property
    def was_cached(self) -> bool:
        return self.status == "cached"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> tuple[str, int, int]:
    """Return ``(digest, bytes, files)`` for a directory tree.

    The digest covers relative paths as well as contents, so renames are
    visible.  Symlinks are followed for files but not descended into.
    """
    digest = hashlib.sha256()
    total_bytes = 0
    files = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
        digest.update(b"\0")
        files += 1
    return digest.hexdigest(), total_bytes, files


def cache_path(
    source: CorpusSource,
    cache_dir: str | Path = "data/raw/cache",
    *,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Destination path for ``source`` inside ``cache_dir``."""
    root = Path(cache_dir).expanduser()
    if not root.is_absolute():
        root = repo_root / root
    return root / (source.filename or source.name)


def _check_expected(source: CorpusSource, digest: str, size: int) -> None:
    if source.size_bytes is not None and size != source.size_bytes:
        raise CorpusCacheError(
            f"{source.name}: expected {source.size_bytes} bytes, got {size}; "
            "the download is truncated or the upstream file changed"
        )
    if source.sha256 and digest.lower() != source.sha256.lower():
        raise CorpusCacheError(
            f"{source.name}: SHA-256 mismatch (expected {source.sha256}, "
            f"got {digest}); refusing to register a corrupt download"
        )


def _local_file(source: CorpusSource, repo_root: Path) -> Path:
    """Resolve ``file://`` or a plain local path for ``source``."""
    if source.url.startswith("file://"):
        parsed = urllib.parse.urlparse(source.url)
        return Path(urllib.request.url2pathname(parsed.path))
    return source_path(source, repo_root)


def _copy_file(
    src: Path,
    dest: Path,
    source: CorpusSource,
) -> tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with src.open("rb") as infh, part.open("wb") as outfh:
            for chunk in iter(lambda: infh.read(_CHUNK_SIZE), b""):
                outfh.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size == 0:
            raise CorpusCacheError(f"{source.name}: source file {src} is empty")
        _check_expected(source, digest.hexdigest(), size)
        os.replace(part, dest)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise CorpusCacheError(f"{source.name}: cannot copy {src}: {exc}") from exc
    except CorpusCacheError:
        part.unlink(missing_ok=True)
        raise
    return size, digest.hexdigest()


def _download_file(
    source: CorpusSource,
    dest: Path,
    *,
    timeout: int,
) -> tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(
        source.url, headers={"User-Agent": "llm-from-scratch-corpus-fetch/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, part.open(
            "wb"
        ) as outfh:
            for chunk in iter(lambda: response.read(_CHUNK_SIZE), b""):
                outfh.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size == 0:
            raise CorpusCacheError(f"{source.name}: server returned an empty file")
        _check_expected(source, digest.hexdigest(), size)
        os.replace(part, dest)
    except CorpusCacheError:
        part.unlink(missing_ok=True)
        raise
    except Exception as exc:  # urllib raises several unrelated types
        part.unlink(missing_ok=True)
        raise CorpusCacheError(
            f"{source.name}: download failed from {source.url}: {exc}; "
            "check the URL/network or use --offline with a warmed cache"
        ) from exc
    return size, digest.hexdigest()


def _fetch_directory(
    source: CorpusSource,
    dest: Path,
    *,
    repo_root: Path,
    offline: bool,
) -> FetchResult:
    if dest.is_dir():
        digest, total_bytes, files = sha256_tree(dest)
        return FetchResult(
            source=source,
            path=dest,
            status="cached",
            bytes=total_bytes,
            files=files,
            sha256=digest,
            fetched_at=_utc_now(),
        )
    if offline:
        raise CorpusCacheError(
            f"{source.name}: not found in the cache at {dest} (offline mode)"
        )
    src = _local_file(source, repo_root)
    if not src.is_dir():
        raise CorpusCacheError(
            f"{source.name}: local source directory {src} does not exist"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    try:
        shutil.copytree(src, part)
        os.replace(part, dest)
    except OSError as exc:
        shutil.rmtree(part, ignore_errors=True)
        raise CorpusCacheError(
            f"{source.name}: cannot copy {src} into the cache: {exc}"
        ) from exc
    digest, total_bytes, files = sha256_tree(dest)
    return FetchResult(
        source=source,
        path=dest,
        status="copied",
        bytes=total_bytes,
        files=files,
        sha256=digest,
        fetched_at=_utc_now(),
    )


def fetch_source(
    source: CorpusSource,
    cache_dir: str | Path = "data/raw/cache",
    *,
    repo_root: Path = REPO_ROOT,
    offline: bool = False,
    timeout: int = _DEFAULT_TIMEOUT,
    log: Callable[[str], None] | None = None,
) -> FetchResult:
    """Fetch one source into the cache, reusing a valid cached copy.

    Parameters
    ----------
    source:
        Registry entry to fetch.
    cache_dir:
        Cache root (repository-relative by default).
    offline:
        Refuse all network/local-source access; only a warm cache works.
    log:
        Optional progress sink (the CLI passes a console logger).
    """
    dest = cache_path(source, cache_dir, repo_root=repo_root)
    say = log or (lambda _message: None)

    if source.kind == "directory":
        return _fetch_directory(source, dest, repo_root=repo_root, offline=offline)

    if dest.is_file():
        size = dest.stat().st_size
        digest = sha256_file(dest) if (source.sha256 or source.size_bytes) else None
        if source.size_bytes is not None or source.sha256:
            try:
                _check_expected(source, digest or "", size)
            except CorpusCacheError:
                if offline:
                    raise
                say(f"{source.name}: cached copy is invalid, re-downloading")
            else:
                return FetchResult(
                    source=source,
                    path=dest,
                    status="cached",
                    bytes=size,
                    files=1,
                    sha256=digest,
                    fetched_at=_utc_now(),
                )
        else:
            return FetchResult(
                source=source,
                path=dest,
                status="cached",
                bytes=size,
                files=1,
                sha256=None,
                fetched_at=_utc_now(),
            )

    if offline:
        raise CorpusCacheError(
            f"{source.name}: not found in the cache at {dest} (offline mode)"
        )

    say(f"{source.name}: fetching {source.url}")
    if source.url.startswith(("http://", "https://")):
        size, digest = _download_file(source, dest, timeout=timeout)
    else:
        src = _local_file(source, repo_root)
        if not src.is_file():
            raise CorpusCacheError(
                f"{source.name}: local source file {src} does not exist"
            )
        size, digest = _copy_file(src, dest, source)
    say(f"{source.name}: {size} bytes -> {dest}")
    return FetchResult(
        source=source,
        path=dest,
        status="downloaded",
        bytes=size,
        files=1,
        sha256=digest,
        fetched_at=_utc_now(),
    )
