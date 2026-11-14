"""Tokenizer pipeline: config → corpus → character + BPE → artifacts.

This is the glue shared by ``scripts/tokenizer.py`` and the ``/tokenizer``
UI page (T-205).  It keeps path resolution, config defaults, bundle
persistence and artifact discovery in one place so the CLI and the UI
cannot drift apart.

Artifact layout (inside one run folder created by T-004)::

    <run>/
      corpus/
        cleaned.txt          # the cleaned, deduplicated corpus
        report.json          # per-file ingestion report
      tokenizers/
        bundle.json          # source metadata
        char/                # vocab.json + meta.json
        bpe/                 # vocab.json + merges.json + meta.json
      metrics.json           # T-204 comparison
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.tokenizer.bpe import BPETokenizer
from core.tokenizer.char import CharTokenizer
from core.tokenizer.common import TokenizerError
from core.tokenizer.corpus import (
    Corpus,
    CorpusOptions,
    ingest_corpus,
)

__all__ = [
    "BUNDLE_FILE",
    "DEFAULT_SAMPLE_CORPUS",
    "DEFAULT_SAMPLE_TEXT",
    "REPO_ROOT",
    "TokenizerBundle",
    "find_bundle_dirs",
    "load_bundle",
    "metrics_text",
    "resolve_corpus_path",
    "save_bundle",
    "save_corpus_artifacts",
    "train_bundle",
    "train_bundle_from_config",
]

#: Repository root (``core/tokenizer/pipeline.py`` -> parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Bundled tiny German corpus used by the sample config and the UI when
#: no trained artifact exists.  It is small on purpose (fast on CPU) and
#: contains UTF-8, a BOM, CRLF and a Windows-1252 file to exercise the
#: ingestion fallbacks.
DEFAULT_SAMPLE_CORPUS = REPO_ROOT / "data" / "sample_de"

#: Default text in the UI textarea.  It contains the kind of material a
#: German corpus has: umlauts, ß, a compound noun, punctuation and a
#: number.  Keep it short enough to retokenize on every click.
DEFAULT_SAMPLE_TEXT = (
    "Die Straße war frei, und die Fußgängerüberwege glänzten im Regen. "
    "Grüße aus Köln! Am 3. Oktober 2025 zählte die Stadt über 1.000 "
    "Besucherinnen und Besucher beim Straßenfest."
)

BUNDLE_FILE = "bundle.json"
BUNDLE_VERSION = 1


@dataclass
class TokenizerBundle:
    """A character tokenizer and a BPE tokenizer trained together."""

    char: CharTokenizer
    bpe: BPETokenizer
    source: str = "in-memory"
    path: Path | None = None

    def save(self, directory: str | Path) -> Path:
        return save_bundle(self, directory)

    @classmethod
    def load(cls, directory: str | Path) -> "TokenizerBundle":
        return load_bundle(directory)


def train_bundle(
    text: str,
    *,
    vocab_size: int = 500,
    min_frequency: int = 2,
    max_token_length: int = 32,
    source: str = "in-memory",
) -> TokenizerBundle:
    """Train both tokenizers on the same text."""
    if not text.strip():
        raise TokenizerError("cannot train a tokenizer on empty text")
    char = CharTokenizer.from_text(text)
    bpe = BPETokenizer.train(
        text,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_token_length=max_token_length,
    )
    return TokenizerBundle(char=char, bpe=bpe, source=source)


def resolve_corpus_path(corpus_path: str | Path, base_dir: str | Path = REPO_ROOT) -> Path:
    """Resolve a config corpus path relative to ``base_dir``.

    Configs are written relative to the repository root, independent of
    the server/CLI working directory.
    """
    path = Path(corpus_path)
    return path if path.is_absolute() else Path(base_dir) / path


def train_bundle_from_config(
    cfg: dict,
    *,
    corpus: Corpus | None = None,
    base_dir: str | Path = REPO_ROOT,
) -> tuple[TokenizerBundle, Corpus]:
    """Ingest the configured corpus (unless given) and train the bundle."""
    pc = cfg.get("phase_config", {}) or {}
    if corpus is None:
        corpus_path = resolve_corpus_path(
            pc.get("corpus_path", str(DEFAULT_SAMPLE_CORPUS)), base_dir
        )
        corpus = ingest_corpus(corpus_path, CorpusOptions.from_phase_config(pc))
    bundle = train_bundle(
        corpus.text,
        vocab_size=pc.get("vocab_size", 500),
        min_frequency=pc.get("min_frequency", 2),
        max_token_length=pc.get("max_token_length", 32),
        source=str(pc.get("corpus_path", "in-memory")),
    )
    return bundle, corpus


def metrics_text(text: str, sample_chars: int = 4000) -> str:
    """Return the text slice used for metrics (``0`` means all)."""
    if sample_chars and sample_chars > 0 and len(text) > sample_chars:
        return text[:sample_chars]
    return text


# ---------------------------------------------------------------------------
# Bundle persistence and discovery
# ---------------------------------------------------------------------------

def save_bundle(bundle: TokenizerBundle, directory: str | Path) -> Path:
    """Save ``char/``, ``bpe/`` and ``bundle.json`` under ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundle.char.save(directory / "char")
    bundle.bpe.save(directory / "bpe")
    meta = {
        "format": "llm-from-scratch-tokenizer-bundle",
        "version": BUNDLE_VERSION,
        "source": bundle.source,
        "char_vocab_size": bundle.char.vocab_size,
        "bpe_vocab_size": bundle.bpe.vocab_size,
        "bpe_merges": len(bundle.bpe.merges),
        "bpe_training_stats": bundle.bpe.training_stats,
    }
    (directory / BUNDLE_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    bundle.path = directory
    return directory


def load_bundle(directory: str | Path) -> TokenizerBundle:
    """Load a bundle written by :func:`save_bundle`."""
    directory = Path(directory)
    meta_path = directory / BUNDLE_FILE
    if not meta_path.is_file():
        raise TokenizerError(
            f"not a tokenizer bundle: {directory} (missing {BUNDLE_FILE})"
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenizerError(f"cannot read bundle {meta_path}: {exc}") from exc
    if meta.get("version") != BUNDLE_VERSION:
        raise TokenizerError(
            f"unsupported bundle version {meta.get('version')!r} in {meta_path}"
        )
    return TokenizerBundle(
        char=CharTokenizer.load(directory / "char"),
        bpe=BPETokenizer.load(directory / "bpe"),
        source=str(meta.get("source", str(directory))),
        path=directory,
    )


def find_bundle_dirs(root: str | Path | None = None) -> list[Path]:
    """Find ``*/tokenizers`` bundle directories under ``root``, newest first.

    The default root is ``artifacts/``.  A directory qualifies when it has
    a ``bundle.json`` plus the ``char/`` and ``bpe/`` subdirectories.
    """
    root = Path(root) if root is not None else REPO_ROOT / "artifacts"
    if not root.is_dir():
        return []
    found: list[Path] = []
    for meta_path in root.glob("*/tokenizers/" + BUNDLE_FILE):
        directory = meta_path.parent
        if (directory / "char").is_dir() and (directory / "bpe").is_dir():
            found.append(directory)
    found.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return found


def save_corpus_artifacts(corpus: Corpus, directory: str | Path) -> dict[str, Path]:
    """Write ``cleaned.txt`` and ``report.json`` into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    text_path = directory / "cleaned.txt"
    report_path = directory / "report.json"
    text_path.write_text(corpus.text, encoding="utf-8")
    corpus.write_report(report_path)
    return {"corpus": text_path, "report": report_path}
