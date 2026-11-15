"""M6 – Corpus downloader and manifest (T-601).

* :mod:`core.corpus.sources` — German-first source registry
* :mod:`core.corpus.cache` — offline-friendly downloader/cache with
  SHA-256 and size verification
* :mod:`core.corpus.manifest` — additive updates to
  ``data/corpus_manifest.json``

The CLI is :mod:`scripts.corpus`::

    python -m scripts.corpus list
    python -m scripts.corpus fetch
    python -m scripts.corpus fetch --all
    python -m scripts.corpus status
"""

from __future__ import annotations

from core.corpus.cache import (
    CorpusCacheError,
    FetchResult,
    cache_path,
    fetch_source,
    sha256_file,
    sha256_tree,
)
from core.corpus.manifest import (
    DEFAULT_MANIFEST,
    ManifestError,
    load_manifest,
    record_source,
    registered_sources,
)
from core.corpus.preprocess import (
    PREPROCESS_PROFILES,
    PreprocessError,
    PreprocessProfile,
    PreprocessResult,
    get_profile,
    preprocess_corpus,
    record_preprocess,
)
from core.corpus.sources import (
    DEFAULT_SOURCES,
    REPO_ROOT,
    CorpusSource,
    get_source,
    select_sources,
    source_path,
)

__all__ = [
    "CorpusCacheError",
    "CorpusSource",
    "DEFAULT_MANIFEST",
    "DEFAULT_SOURCES",
    "FetchResult",
    "ManifestError",
    "PREPROCESS_PROFILES",
    "PreprocessError",
    "PreprocessProfile",
    "PreprocessResult",
    "REPO_ROOT",
    "cache_path",
    "fetch_source",
    "get_profile",
    "get_source",
    "load_manifest",
    "preprocess_corpus",
    "record_preprocess",
    "record_source",
    "registered_sources",
    "select_sources",
    "sha256_file",
    "sha256_tree",
    "source_path",
]
