"""CLI: fetch German-first corpora into a local cache and register them (T-601).

Examples::

    python -m scripts.corpus list
    python -m scripts.corpus fetch
    python -m scripts.corpus fetch --source sample_de
    python -m scripts.corpus fetch --source gutenberg_de_catalog
    python -m scripts.corpus fetch --all
    python -m scripts.corpus fetch --offline
    python -m scripts.corpus status

``fetch`` writes into ``data/raw/cache/`` (gitignored) and records every
source in ``data/corpus_manifest.json`` under a ``cache`` section.  The
default fetch uses the bundled ``sample_de`` corpus, so it works fully
offline; large Wikipedia/Gutenberg downloads are explicit opt-ins.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.corpus import (
    DEFAULT_MANIFEST,
    DEFAULT_SOURCES,
    PREPROCESS_PROFILES,
    CorpusCacheError,
    ManifestError,
    PreprocessError,
    fetch_source,
    get_profile,
    load_manifest,
    preprocess_corpus,
    record_preprocess,
    record_source,
    registered_sources,
    select_sources,
)
from core.corpus.sources import REPO_ROOT

DEFAULT_CACHE_DIR = "data/raw/cache"


def _print_sources() -> None:
    print(f"{'name':28} {'lang':5} {'kind':10} {'default':8} {'license'}")
    for source in DEFAULT_SOURCES:
        print(
            f"{source.name:28} {source.language:5} {source.kind:10} "
            f"{'yes' if source.default else 'no':8} {source.license}"
        )


def _print_status(manifest: str) -> int:
    sources = registered_sources(manifest)
    data = load_manifest(manifest)
    preprocessed = data.get("preprocessed")
    has_preprocessed = isinstance(preprocessed, dict) and bool(preprocessed)
    if not sources and not has_preprocessed:
        print(f"no cached sources registered in {manifest}")
        print("run: python -m scripts.corpus fetch")
        return 0
    if sources:
        print(f"registered cache sources in {manifest}:")
        for name in sorted(sources):
            record = sources[name]
            print(
                f"  {name}: {record.get('files', '?')} file(s), "
                f"{record.get('bytes', '?')} bytes, status={record.get('status')}, "
                f"sha256={str(record.get('sha256'))[:12]}…"
            )
            print(f"    path: {record.get('path')}")
            print(f"    url:  {record.get('url')}")
            print(f"    license: {record.get('license')}")

    if has_preprocessed:
        print("preprocessed corpora:")
        for name in sorted(preprocessed):
            record = preprocessed[name]
            print(
                f"  {name}: {record.get('files', '?')} file(s), "
                f"{record.get('chars', '?')} chars, "
                f"{record.get('lines', '?')} lines"
            )
            print(f"    path: {record.get('path')}")
            print(f"    sha256={str(record.get('output_sha256'))[:12]}…")
    return 0


def _default_inputs(manifest: str) -> list[str]:
    """Use cached directory/text sources when --input is omitted."""
    paths: list[str] = []
    for name, record in registered_sources(manifest).items():
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = REPO_ROOT / path
        if record.get("kind") == "directory" or path.suffix.lower() == ".txt":
            if path.exists():
                paths.append(str(path))
    return paths


def _preprocess(args: argparse.Namespace) -> int:
    try:
        profile = get_profile(args.profile)
    except PreprocessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    inputs = args.input or _default_inputs(args.manifest)
    if not inputs:
        print(
            "error: no inputs; pass --input DIR or run "
            "'python -m scripts.corpus fetch' first",
            file=sys.stderr,
        )
        return 1

    output = args.output or f"data/processed/{profile.name}"
    try:
        result = preprocess_corpus(inputs, profile, output)
    except PreprocessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    record = None
    if not args.no_register:
        try:
            record = record_preprocess(result, args.manifest)
        except ManifestError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    stats = result.corpus.stats
    print(
        f"{profile.name}: {stats.get('files_ok', 0)} file(s), "
        f"{stats.get('chars_out', 0)} chars, {stats.get('lines_out', 0)} lines -> "
        f"{result.cleaned_path}"
    )
    print(
        f"  sha256={result.output_sha256[:16]}… bytes={result.output_bytes} "
        f"input_digest={result.input_digest[:16]}…"
    )
    print(
        "  dropped: "
        f"blank={stats.get('blank_dropped', 0)} "
        f"short={stats.get('short_dropped', 0)} "
        f"duplicate={stats.get('duplicate_dropped', 0)} "
        f"low_alpha={stats.get('low_alpha_dropped', 0)}"
    )
    if record is not None:
        print(f"  registered as preprocessed.{profile.name} in {args.manifest}")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    try:
        sources = select_sources(args.source, include_all=args.all)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 1
    if not sources:
        print("error: no sources selected", file=sys.stderr)
        return 1

    failures = 0
    for source in sources:
        try:
            result = fetch_source(
                source,
                args.cache_dir,
                offline=args.offline,
                log=lambda message: print(message),
            )
        except CorpusCacheError as exc:
            failures += 1
            print(f"error: {exc}", file=sys.stderr)
            continue
        try:
            record = record_source(result, args.manifest)
        except ManifestError as exc:
            failures += 1
            print(f"error: {exc}", file=sys.stderr)
            continue
        print(
            f"{source.name}: {result.status} | {result.files} file(s), "
            f"{result.bytes} bytes | sha256={str(record.get('sha256'))[:12]}… | "
            f"registered in {args.manifest}"
        )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch German-first corpora and maintain the local cache manifest."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_parser = sub.add_parser("fetch", help="Download/copy sources and register them.")
    fetch_parser.add_argument(
        "--source",
        action="append",
        default=None,
        help=(
            "Source name to fetch (repeatable, or 'all'). Defaults to the "
            "bundled offline sample_de source."
        ),
    )
    fetch_parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch every registered source, including large optional dumps.",
    )
    fetch_parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only a warm cache; never touch the network or local raw data.",
    )
    fetch_parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    fetch_parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Manifest JSON path (default: {DEFAULT_MANIFEST}).",
    )

    sub.add_parser("list", help="List the registered source names and licences.")

    preprocess_parser = sub.add_parser(
        "preprocess",
        help="Create a reproducible cleaned corpus with a named profile.",
    )
    preprocess_parser.add_argument(
        "--profile",
        default="moderate_clean",
        choices=sorted(PREPROCESS_PROFILES),
        help="Cleaning profile (default: moderate_clean).",
    )
    preprocess_parser.add_argument(
        "--input",
        action="append",
        default=None,
        help=(
            "Input file/directory (repeatable). Defaults to directory/text "
            "sources already registered by 'corpus fetch'."
        ),
    )
    preprocess_parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: data/processed/<profile>).",
    )
    preprocess_parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Manifest JSON path (default: {DEFAULT_MANIFEST}).",
    )
    preprocess_parser.add_argument(
        "--no-register",
        action="store_true",
        help="Write the corpus without registering it in the manifest.",
    )

    status_parser = sub.add_parser("status", help="Show sources recorded in the manifest.")
    status_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)

    args = parser.parse_args(argv)
    if args.command == "list":
        _print_sources()
        return 0
    if args.command == "status":
        try:
            return _print_status(args.manifest)
        except ManifestError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.command == "preprocess":
        return _preprocess(args)
    return _fetch(args)


if __name__ == "__main__":
    raise SystemExit(main())
