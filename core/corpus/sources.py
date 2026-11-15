"""Registry of German-first corpus sources (T-601).

The registry is intentionally small and explicit: every source records
where it comes from, its licence and whether it is part of the default
(offline-friendly) fetch set.  Large dumps are opt-in so a fresh checkout
can bootstrap with one tiny command and no multi-gigabyte surprise.

Sources are addressed by an HTTP(S) URL, a ``file://`` URL, or a plain
path relative to the repository root.  The latter keeps the bundled
``data/sample_de`` corpus usable on a completely offline machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CorpusSource",
    "DEFAULT_SOURCES",
    "REPO_ROOT",
    "get_source",
    "select_sources",
    "source_path",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CorpusSource:
    """One downloadable or locally staged corpus source.

    Parameters
    ----------
    name:
        Stable identifier used by the CLI and in the cache manifest.
    url:
        HTTP(S)/``file://`` URL or repository-relative local path.
    kind:
        ``"file"`` for a single download, ``"directory"`` for a tree that
        is copied into the cache (used by the bundled sample corpus).
    filename:
        Cache entry name; defaults to the source name.
    language:
        ISO language code; the registry is German-first, but the field is
        explicit so future sources do not silently mix languages.
    license:
        Human-readable licence string shown by ``corpus list`` and stored
        in the manifest.
    description:
        Short explanation of what the source contains.
    sha256:
        Optional expected SHA-256 of a single-file source.  When set, the
        downloader verifies it and refuses corrupt/truncated downloads.
    size_bytes:
        Optional expected size of a single-file source.
    default:
        Include in a bare ``corpus fetch`` (offline-friendly set).
    optional:
        Large/network-heavy source; only fetched when requested or with
        ``--all``.  Documented so users know what they are opting into.
    """

    name: str
    url: str
    kind: str = "file"
    filename: str = ""
    language: str = "de"
    license: str = ""
    description: str = ""
    sha256: str | None = None
    size_bytes: int | None = None
    default: bool = False
    optional: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("file", "directory"):
            raise ValueError(f"unknown source kind {self.kind!r}")
        if not self.name:
            raise ValueError("source name must not be empty")
        if not self.url:
            raise ValueError(f"source {self.name!r}: url must not be empty")


#: The DE-first registry.  ``sample_de`` is the default because it is tiny
#: and ships with the repository; the real corpora are explicit opt-ins.
DEFAULT_SOURCES: tuple[CorpusSource, ...] = (
    CorpusSource(
        name="sample_de",
        url="data/sample_de",
        kind="directory",
        language="de",
        license="CC0-1.0",
        description="Bundled noisy German sample (4 files, UTF-8/CP1252/BOM/CRLF).",
        default=True,
    ),
    CorpusSource(
        name="gutenberg_de_catalog",
        url="https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",
        filename="pg_catalog.csv",
        language="de",
        license="Project Gutenberg terms; individual texts public domain in the US",
        description="Project Gutenberg catalogue used to select German-language books.",
        optional=True,
    ),
    CorpusSource(
        name="wikipedia_de_dump",
        url=(
            "https://dumps.wikimedia.org/dewiki/latest/"
            "dewiki-latest-pages-articles.xml.bz2"
        ),
        filename="dewiki-latest-pages-articles.xml.bz2",
        language="de",
        license="CC BY-SA 4.0",
        description="Full German Wikipedia article dump (~8 GiB, needs WikiExtractor).",
        optional=True,
    ),
)


def get_source(name: str) -> CorpusSource:
    """Look up a source by name; raise ``KeyError`` with the known names."""
    for source in DEFAULT_SOURCES:
        if source.name == name:
            return source
    known = ", ".join(source.name for source in DEFAULT_SOURCES)
    raise KeyError(f"unknown corpus source {name!r}; known sources: {known}")


def select_sources(
    names: list[str] | tuple[str, ...] | None = None,
    *,
    include_all: bool = False,
) -> list[CorpusSource]:
    """Resolve CLI selections to sources.

    ``names`` wins over ``include_all``; with neither, the default
    (offline, tiny) set is returned.  ``all`` is also accepted as a name.
    """
    if names:
        if "all" in names:
            return list(DEFAULT_SOURCES)
        return [get_source(name) for name in names]
    if include_all:
        return list(DEFAULT_SOURCES)
    return [source for source in DEFAULT_SOURCES if source.default]


def source_path(source: CorpusSource, repo_root: Path = REPO_ROOT) -> Path:
    """Local path for a source without a scheme (``file://`` handled by cache)."""
    if "://" in source.url:
        return Path(source.url)
    path = Path(source.url).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path
