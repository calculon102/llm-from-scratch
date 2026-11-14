"""CLI: ingest a corpus, train char + BPE tokenizers, log metrics (T-201…T-204).

Usage::

    python -m scripts.tokenizer --config configs/tokenizer_tiny_debug.yaml
    python -m scripts.tokenizer --config ... --input data/sample_de
    python -m scripts.tokenizer --config ... --vocab-size 800

One run folder is created under the config's ``output_dir`` (T-004) with::

    corpus/cleaned.txt      cleaned, deduplicated text
    corpus/report.json      per-file encoding/filter report
    tokenizers/bundle.json  char + BPE bundle metadata
    tokenizers/char/        vocab.json + meta.json
    tokenizers/bpe/         vocab.json + merges.json + meta.json
    metrics.json            char-vs-BPE compression / OOV comparison

The command is offline-only and CPU-only: tokenizer training is pure
Python, so it runs in well under a second on the bundled sample corpus.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from pathlib import Path

import core.tokenizer  # noqa: F401  # registers the "tokenizer" phase schema
from core.artifacts import ArtifactError, create_run
from core.config import ConfigError, add_config_arg, load_config, validate_config
from core.tokenizer import (
    CorpusOptions,
    TokenizerError,
    compare_tokenizers,
    ingest_corpus,
    log_metrics,
    metrics_text,
    resolve_corpus_path,
    save_bundle,
    save_corpus_artifacts,
    train_bundle_from_config,
)


def _display_token(token: str) -> str:
    """Make whitespace visible in run logs."""
    return (
        token.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace(" ", "·")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train character and BPE tokenizers on a noisy German-first corpus."
    )
    add_config_arg(parser)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override phase_config.corpus_path (file or directory).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override phase_config.vocab_size (specials + alphabet + merges).",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if cfg["phase"] != "tokenizer":
        print(
            f"error: {args.config}: phase is {cfg['phase']!r}, expected 'tokenizer'",
            file=sys.stderr,
        )
        return 1

    # CLI overrides are re-validated exactly like file values.
    phase_config = dict(cfg["phase_config"])
    if args.input is not None:
        phase_config["corpus_path"] = str(Path(args.input).expanduser().resolve())
    if args.vocab_size is not None:
        phase_config["vocab_size"] = args.vocab_size
    cfg["phase_config"] = phase_config
    try:
        cfg = validate_config(cfg, source=f"{args.config} (with CLI overrides)")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        run = create_run(cfg)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    run.log(
        "startup: tokenizer "
        f"python={platform.python_version()} platform={platform.platform()}"
    )
    run.log(
        f"phase=tokenizer profile={cfg['profile']} "
        f"vocab_size={cfg['phase_config']['vocab_size']} seed={cfg['seed']}"
    )

    try:
        options = CorpusOptions.from_phase_config(cfg["phase_config"])
        corpus_path = resolve_corpus_path(cfg["phase_config"]["corpus_path"])
        run.log(f"corpus: ingesting {corpus_path}")
        corpus = ingest_corpus(corpus_path, options)
        run.log(
            f"corpus: {corpus.stats['files_ok']} file(s) ok, "
            f"{corpus.stats['files_skipped']} skipped, "
            f"{corpus.stats['chars_out']} chars, "
            f"{corpus.stats['lines_out']} lines, "
            f"encodings={corpus.stats['encodings']}"
        )
        run.log(
            "corpus filters: "
            f"blank={corpus.stats['blank_dropped']} "
            f"short={corpus.stats['short_dropped']} "
            f"duplicate={corpus.stats['duplicate_dropped']} "
            f"junk={corpus.stats['junk_dropped']} "
            f"low_alpha={corpus.stats['low_alpha_dropped']}"
        )
        for report in corpus.files:
            if report.status != "ok":
                run.log(
                    f"corpus skip: {report.path} ({report.reason})",
                    level=logging.WARNING,
                )
            elif report.encoding not in ("utf-8-sig", None):
                run.log(
                    f"corpus decode: {report.path} via {report.encoding} "
                    f"({report.replaced_chars} replacement chars)"
                )

        paths = save_corpus_artifacts(corpus, Path(run.run_dir) / "corpus")
        run.log(f"corpus written: {paths['corpus']}")
        run.log(f"corpus report: {paths['report']}")

        bundle, _ = train_bundle_from_config(cfg, corpus=corpus)
        bundle_dir = save_bundle(bundle, Path(run.run_dir) / "tokenizers")
        run.log(
            f"char tokenizer: vocab_size={bundle.char.vocab_size}"
        )
        run.log(
            f"bpe tokenizer: vocab_size={bundle.bpe.vocab_size} "
            f"merges={len(bundle.bpe.merges)} "
            f"stop={bundle.bpe.training_stats.get('stop_reason')}"
        )
        if bundle.bpe.merges:
            preview = ", ".join(
                f"{_display_token(left)}+{_display_token(right)}"
                f"→{_display_token(left + right)}"
                for left, right in bundle.bpe.merges[:10]
            )
            run.log(f"first merges: {preview}")
        run.log(f"tokenizers written: {bundle_dir}")

        sample = metrics_text(
            corpus.text, cfg["phase_config"]["metrics_sample_chars"]
        )
        if len(sample) < len(corpus.text):
            run.log(
                f"metrics: first {len(sample)} of {len(corpus.text)} chars "
                f"(metrics_sample_chars={cfg['phase_config']['metrics_sample_chars']})"
            )
        comparison = compare_tokenizers(bundle.char, bundle.bpe, sample)
        run.log(
            "metrics caveat: BPE merges were trained on this corpus, so "
            "compression here is optimistic (UI text is held out)"
        )
        log_metrics(run, comparison)
    except (TokenizerError, OSError) as exc:
        run.log(f"error: {exc}", level=logging.ERROR)
        run.mark_finished("failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1

    run.mark_finished("finished")
    print(f"run directory: {run.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
