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

from core.corpus import (
    DEFAULT_MANIFEST,
    DEFAULT_SOURCES,
    CorpusCacheError,
    ManifestError,
    fetch_source,
    record_source,
    registered_sources,
    select_sources,
)

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
    if not sources:
        print(f"no cached sources registered in {manifest}")
        print("run: python -m scripts.corpus fetch")
        return 0
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
    return _fetch(args)


if __name__ == "__main__":
    raise SystemExit(main())
