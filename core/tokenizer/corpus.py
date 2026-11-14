"""Noisy, German-first corpus ingestion (T-201).

The corpus for this project is *not* a clean academic benchmark: it is
German-first text collected from mixed sources, and it arrives with the
usual damage -- UTF-8 files with a BOM, Windows-1252 files with umlauts,
stray ``\\r\\n`` line endings, zero-width characters, control bytes,
boilerplate lines repeated on every page, one-word navigation crumbs, and
the occasional binary file that slipped into the folder.

Ingestion turns all of that into one deterministic stream of clean text
plus a machine-readable report of what was kept, changed and dropped.
"Robust" here means that no single bad file can abort a run, and that
every decision is visible in the report instead of being silent.

Pipeline (per file)::

    bytes --(encoding fallback)--> str --(Unicode + whitespace cleanup)
          --(line filters)--> cleaned lines --(global dedupe)--> corpus text

Usage::

    from core.tokenizer.corpus import CorpusOptions, ingest_corpus

    corpus = ingest_corpus(["data/sample_de"], CorpusOptions())
    print(corpus.text[:200])
    print(corpus.report()["stats"])

Design notes
------------
* **Encoding fallback.**  Try each configured encoding in order with
  strict decoding.  Only when every codec fails do we fall back to UTF-8
  with replacement characters, and such files are counted as suspicious
  in the report.
* **NFC normalization.**  German umlauts can be represented as one
  precomposed code point (``ä``, U+00E4) or as ``a`` + combining diaeresis
  (U+0061 U+0308).  Both look identical but are different token sequences;
  NFC makes the corpus canonical.  We deliberately do *not* strip
  diacritics or map ``ß`` to ``ss``: German orthography is part of the
  signal the tokenizer should learn.
* **Line-level filtering.**  Very short lines (navigation crumbs, page
  numbers) and duplicate lines (boilerplate) add little language signal
  but cost vocabulary and training steps.  Both filters are configurable.
* **No external calls.**  Ingestion only reads local files; nothing is
  downloaded.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from core.tokenizer.common import TokenizerError

__all__ = [
    "DEFAULT_ENCODINGS",
    "DEFAULT_EXTENSIONS",
    "CleaningStats",
    "Corpus",
    "CorpusOptions",
    "DecodeResult",
    "FileReport",
    "clean_text",
    "decode_bytes",
    "ingest_corpus",
    "iter_corpus_files",
    "looks_binary",
]

#: Codecs tried in order.  ``utf-8-sig`` accepts both plain UTF-8 and a
#: UTF-8 BOM; ``cp1252`` is the common Windows German encoding; Latin-1
#: never fails, so it is the last resort before replacement decoding.
DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

#: File suffixes considered corpus text when a *directory* is ingested.
#: Explicitly named files are accepted regardless of suffix.
DEFAULT_EXTENSIONS: tuple[str, ...] = (".txt",)

#: Characters we normalize to a plain space before collapsing runs.
_SPACE_LIKE = {
    "\u00a0",  # no-break space
    "\u1680",  # ogham space mark
    "\u2000", "\u2001", "\u2002", "\u2003", "\u2004",
    "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200a",
    "\u202f",  # narrow no-break space
    "\u205f",  # medium mathematical space
    "\u3000",  # ideographic space
}

#: A run of horizontal whitespace (tabs, spaces) -> one space.
_HSPACE_RE = re.compile(r"[^\S\n]+")

#: Unicode categories that are *format* characters (zero-width spaces,
#: BOMs, bidi controls) and can be deleted outright.
_FORMAT_CATEGORIES = {"Cf"}

#: Line/paragraph separators that should behave like ``\n``.
_LINE_CATEGORIES = {"Zl", "Zp"}

#: Fraction of U+FFFD replacement characters above which a line is
#: treated as undecodable garbage and dropped.
_MAX_REPLACEMENT_RATIO = 0.05


@dataclass(frozen=True)
class CorpusOptions:
    """Knobs for :func:`clean_text` and :func:`ingest_corpus`.

    All values are intentionally conservative defaults for German text:
    keep almost everything, drop only obviously useless lines.
    """

    min_line_chars: int = 2
    max_line_chars: int = 2000
    min_alpha_ratio: float = 0.0
    drop_duplicates: bool = True
    lowercase: bool = False
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS

    def __post_init__(self) -> None:
        if self.min_line_chars < 0:
            raise TokenizerError("min_line_chars must be >= 0")
        if self.max_line_chars < 0:
            raise TokenizerError("max_line_chars must be >= 0")
        if not 0.0 <= self.min_alpha_ratio <= 1.0:
            raise TokenizerError("min_alpha_ratio must be in [0, 1]")
        if not self.encodings:
            raise TokenizerError("encodings must not be empty")

    @classmethod
    def from_phase_config(cls, phase_config: dict) -> "CorpusOptions":
        """Build options from a validated ``phase_config`` mapping."""
        pc = phase_config or {}
        defaults = cls()
        return cls(
            min_line_chars=pc.get("min_line_chars", defaults.min_line_chars),
            max_line_chars=pc.get("max_line_chars", defaults.max_line_chars),
            min_alpha_ratio=pc.get("min_alpha_ratio", defaults.min_alpha_ratio),
            drop_duplicates=pc.get("drop_duplicates", defaults.drop_duplicates),
            lowercase=pc.get("lowercase", defaults.lowercase),
            encodings=tuple(pc.get("encodings", defaults.encodings)),
            extensions=tuple(pc.get("extensions", defaults.extensions)),
        )


@dataclass
class CleaningStats:
    """What :func:`clean_text` did to one decoded file."""

    chars_in: int = 0
    chars_out: int = 0
    lines_in: int = 0
    lines_out: int = 0
    blank_dropped: int = 0
    short_dropped: int = 0
    duplicate_dropped: int = 0
    low_alpha_dropped: int = 0
    junk_dropped: int = 0
    truncated_lines: int = 0
    controls_removed: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DecodeResult:
    """Result of decoding one file's bytes."""

    text: str
    encoding: str
    replaced_chars: int = 0
    fell_back: bool = False


@dataclass(frozen=True)
class FileReport:
    """Per-file ingestion record (kept in the corpus report)."""

    path: str
    status: str  # "ok" or "skipped"
    reason: str | None = None
    encoding: str | None = None
    bytes: int = 0
    chars_in: int = 0
    chars_out: int = 0
    lines_in: int = 0
    lines_out: int = 0
    replaced_chars: int = 0
    cleaning: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Corpus:
    """One clean corpus stream plus the report of how it was produced."""

    text: str
    files: tuple[FileReport, ...]
    stats: dict[str, object]

    def report(self) -> dict[str, object]:
        """JSON-serializable ingestion report."""
        return {
            "stats": self.stats,
            "files": [report.to_dict() for report in self.files],
        }

    def write_report(self, path: str | Path) -> Path:
        """Write :meth:`report` as pretty JSON and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.report(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


# ---------------------------------------------------------------------------
# Decoding and binary sniffing
# ---------------------------------------------------------------------------

def decode_bytes(data: bytes, encodings: Sequence[str] = DEFAULT_ENCODINGS) -> DecodeResult:
    """Decode ``data`` using the first codec that accepts it.

    The final fallback is UTF-8 with ``errors="replace"`` so a single
    corrupt byte cannot abort ingestion.  ``replaced_chars`` counts U+FFFD
    characters in the result (including any that were already literal in
    the decoded text), which lets the caller flag doubtful files.
    """
    if not encodings:
        raise TokenizerError("encodings must not be empty")
    for encoding in encodings:
        try:
            text = data.decode(encoding, errors="strict")
        except (UnicodeDecodeError, LookupError):
            continue
        return DecodeResult(
            text=text,
            encoding=encoding,
            replaced_chars=text.count("\ufffd"),
            fell_back=encoding != encodings[0],
        )
    text = data.decode("utf-8", errors="replace")
    return DecodeResult(
        text=text,
        encoding="utf-8+replace",
        replaced_chars=text.count("\ufffd"),
        fell_back=True,
    )


def looks_binary(data: bytes) -> bool:
    """Heuristic: NUL bytes or many C0 control bytes mean "not text"."""
    sample = data[:8192]
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    control = sum(1 for byte in sample if byte < 9 or 13 < byte < 32)
    return control / len(sample) > 0.10


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

def _normalize_characters(text: str) -> tuple[str, int]:
    """Normalize Unicode form and remove/replace control characters.

    Returns the rewritten text and the number of characters removed or
    replaced.
    """
    removed = 0
    out: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if char == "\n":
            out.append(char)
        elif category in _FORMAT_CATEGORIES:
            # BOM and soft hyphen vanish; other format characters
            # (zero-width space, bidi controls) act as a word boundary
            # and become a space, so "A\u200bB" does not merge into "AB".
            if char in {"\ufeff", "\u00ad"}:
                removed += 1
            else:
                out.append(" ")
                removed += 1
        elif category in _LINE_CATEGORIES:
            out.append("\n")
            removed += 1
        elif category == "Cc":
            # NUL and other C0/C1 controls (except \n handled above and
            # \t handled by whitespace collapsing): become a space.
            out.append(" ")
            removed += 1
        elif char in _SPACE_LIKE:
            out.append(" ")
            removed += 1
        else:
            out.append(char)
    return "".join(out), removed


def clean_text(
    text: str, options: CorpusOptions | None = None
) -> tuple[str, CleaningStats]:
    """Clean one decoded file; return ``(clean_text, stats)``.

    The cleaned text ends with a newline when it is non-empty.  Line order
    is preserved, so document structure survives; empty lines are dropped
    (the tokenizer does not need blank lines, and paragraph breaks are
    still visible as line boundaries).
    """
    opts = options or CorpusOptions()
    stats = CleaningStats(chars_in=len(text))
    normalized = unicodedata.normalize("NFC", text)
    # Normalize line endings before splitting, then control/space cleanup.
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized, stats.controls_removed = _normalize_characters(normalized)
    normalized = _HSPACE_RE.sub(" ", normalized)

    kept: list[str] = []
    seen: set[str] = set()
    stats.lines_in = len(normalized.split("\n"))
    for raw in normalized.split("\n"):
        line = raw.strip()
        if not line:
            stats.blank_dropped += 1
            continue
        if len(line) < opts.min_line_chars:
            stats.short_dropped += 1
            continue
        if opts.max_line_chars and len(line) > opts.max_line_chars:
            line = line[: opts.max_line_chars].rstrip()
            stats.truncated_lines += 1
            if not line:
                stats.short_dropped += 1
                continue
        replacements = line.count("\ufffd")
        if replacements and replacements / len(line) > _MAX_REPLACEMENT_RATIO:
            stats.junk_dropped += 1
            continue
        if opts.min_alpha_ratio:
            alpha = sum(1 for char in line if char.isalpha()) / len(line)
            if alpha < opts.min_alpha_ratio:
                stats.low_alpha_dropped += 1
                continue
        if opts.drop_duplicates:
            key = line.casefold()
            if key in seen:
                stats.duplicate_dropped += 1
                continue
            seen.add(key)
        kept.append(line.lower() if opts.lowercase else line)

    cleaned = "\n".join(kept)
    if cleaned:
        cleaned += "\n"
    stats.lines_out = len(kept)
    stats.chars_out = len(cleaned)
    return cleaned, stats


# ---------------------------------------------------------------------------
# File discovery and ingestion
# ---------------------------------------------------------------------------

def _as_paths(inputs: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(inputs, (str, Path)):
        return [Path(inputs)]
    return [Path(item) for item in inputs]


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    return any(part.startswith(".") for part in relative.parts)


def iter_corpus_files(
    inputs: str | Path | Iterable[str | Path],
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> Iterator[Path]:
    """Yield corpus files for files/directories in deterministic order.

    Directories are searched recursively; hidden files/directories and
    files with other suffixes are skipped.  Explicit files are always
    yielded (a caller may deliberately ingest a file without ``.txt``).
    Missing paths produce :class:`TokenizerError`: a typo in a corpus path
    should be loud, not silently yield an empty corpus.
    """
    suffixes = {suffix.lower() for suffix in extensions}
    seen: set[Path] = set()
    for input_path in _as_paths(inputs):
        if not input_path.exists():
            raise TokenizerError(f"corpus path not found: {input_path}")
        if input_path.is_file():
            resolved = input_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield input_path
            continue
        if not input_path.is_dir():
            raise TokenizerError(f"corpus path is neither file nor directory: {input_path}")
        for path in sorted(input_path.rglob("*"), key=lambda p: p.as_posix()):
            if not path.is_file() or _is_hidden(path, input_path):
                continue
            if suffixes and path.suffix.lower() not in suffixes:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def _aggregate_stats(reports: Sequence[FileReport]) -> dict[str, object]:
    stats: dict[str, object] = {
        "files_seen": len(reports),
        "files_ok": 0,
        "files_skipped": 0,
        "bytes": 0,
        "chars_in": 0,
        "chars_out": 0,
        "lines_in": 0,
        "lines_out": 0,
        "controls_removed": 0,
        "blank_dropped": 0,
        "short_dropped": 0,
        "duplicate_dropped": 0,
        "low_alpha_dropped": 0,
        "junk_dropped": 0,
        "truncated_lines": 0,
        "replaced_chars": 0,
        "encodings": {},
    }
    # Only the *drop* counters are summed from ``report.cleaning``;
    # chars/lines are already on the report itself (summing them twice
    # would double-count).
    cleaning_counters = (
        "controls_removed",
        "blank_dropped",
        "short_dropped",
        "duplicate_dropped",
        "low_alpha_dropped",
        "junk_dropped",
        "truncated_lines",
    )
    encodings: Counter[str] = Counter()
    for report in reports:
        if report.status == "ok":
            stats["files_ok"] += 1
        else:
            stats["files_skipped"] += 1
        stats["bytes"] += report.bytes
        stats["chars_in"] += report.chars_in
        stats["chars_out"] += report.chars_out
        stats["lines_in"] += report.lines_in
        stats["lines_out"] += report.lines_out
        stats["replaced_chars"] += report.replaced_chars
        if report.encoding:
            encodings[report.encoding] += 1
        for key in cleaning_counters:
            stats[key] += int(report.cleaning.get(key, 0))
    stats["encodings"] = dict(sorted(encodings.items()))
    return stats


def ingest_corpus(
    inputs: str | Path | Iterable[str | Path],
    options: CorpusOptions | None = None,
) -> Corpus:
    """Read, decode, clean and concatenate corpus files.

    A file that is binary, unreadable, or cleans down to nothing is
    recorded as ``skipped`` with a reason; ingestion continues with the
    remaining files.  Raises :class:`TokenizerError` only when there is no
    usable text at all (an empty corpus cannot train a tokenizer).
    """
    opts = options or CorpusOptions()
    reports: list[FileReport] = []
    chunks: list[str] = []
    # Dedupe across files as well as within them, so the same boilerplate
    # copied into several files does not inflate the corpus.
    global_seen: set[str] = set() if opts.drop_duplicates else set()

    for path in iter_corpus_files(inputs, opts.extensions):
        try:
            data = path.read_bytes()
        except OSError as exc:
            reports.append(
                FileReport(path=str(path), status="skipped", reason=f"read error: {exc}")
            )
            continue
        if not data:
            reports.append(
                FileReport(path=str(path), status="skipped", reason="empty file", bytes=0)
            )
            continue
        if looks_binary(data):
            reports.append(
                FileReport(
                    path=str(path),
                    status="skipped",
                    reason="binary content",
                    bytes=len(data),
                )
            )
            continue

        decoded = decode_bytes(data, opts.encodings)
        cleaned, cleaning = clean_text(decoded.text, opts)
        if opts.drop_duplicates:
            kept: list[str] = []
            extra_duplicates = 0
            for line in cleaned.splitlines():
                key = line.casefold()
                if key in global_seen:
                    extra_duplicates += 1
                    continue
                global_seen.add(key)
                kept.append(line)
            if extra_duplicates:
                cleaning.duplicate_dropped += extra_duplicates
                cleaning.lines_out = len(kept)
                cleaned = "\n".join(kept)
                if cleaned:
                    cleaned += "\n"
                cleaning.chars_out = len(cleaned)

        if not cleaned.strip():
            reports.append(
                FileReport(
                    path=str(path),
                    status="skipped",
                    reason="no usable text after cleaning",
                    encoding=decoded.encoding,
                    bytes=len(data),
                    chars_in=len(decoded.text),
                    chars_out=0,
                    lines_in=cleaning.lines_in,
                    lines_out=0,
                    replaced_chars=decoded.replaced_chars,
                    cleaning=cleaning.to_dict(),
                )
            )
            continue

        chunks.append(cleaned)
        reports.append(
            FileReport(
                path=str(path),
                status="ok",
                encoding=decoded.encoding,
                bytes=len(data),
                chars_in=len(decoded.text),
                chars_out=cleaning.chars_out,
                lines_in=cleaning.lines_in,
                lines_out=cleaning.lines_out,
                replaced_chars=decoded.replaced_chars,
                cleaning=cleaning.to_dict(),
            )
        )

    if not chunks:
        details = "; ".join(
            f"{report.path}: {report.reason}"
            for report in reports
            if report.status == "skipped"
        )
        raise TokenizerError(
            "no usable corpus text found"
            + (f" ({details})" if details else "")
        )

    text = "\n".join(chunk.rstrip("\n") for chunk in chunks) + "\n"
    stats = _aggregate_stats(reports)
    stats["chars_out"] = len(text)
    stats["lines_out"] = sum(1 for line in text.splitlines() if line.strip())
    return Corpus(text=text, files=tuple(reports), stats=stats)
