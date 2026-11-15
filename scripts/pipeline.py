"""CLI: one-command preprocess → tokenizer → GPT → eval pipeline (T-603).

Usage::

    python -m scripts.pipeline --config configs/gpt_tiny_debug.yaml
    python -m scripts.pipeline --config configs/gpt_de_quick.yaml \
        --profile moderate_clean --max-new-tokens 120

The YAML config is a normal GPT config: ``phase: gpt`` plus its
``phase_config`` (corpus, tokenizer size, model architecture, training
schedule).  Pipeline-only choices are command-line flags; they are stored
in the run's ``pipeline.json`` so the run remains reproducible.

One run folder contains::

    corpus/cleaned.txt, corpus/report.json, corpus/manifest.json
    tokenizers/          # char + BPE bundle trained on that corpus
    gpt/                 # model.pt, model_config.json, meta.json, corpus.txt
    eval/report.json     # validation perplexity + generation settings
    eval/sample.txt      # deterministic generated sample
    pipeline.json        # stage timings, hashes, settings, summary
"""

from __future__ import annotations

import argparse
import signal
import sys

import core.gpt  # noqa: F401  # registers the "gpt" phase (T-404)
from core.artifacts import ArtifactError, create_run
from core.config import ConfigError, add_config_arg, load_config
from core.corpus.preprocess import PreprocessError
from core.oom import TrainingOOMError
from core.pipeline import PipelineError, PipelineOptions, run_pipeline
from core.trainer import TrainerError
from scripts.train import _sigterm_to_interrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full educational pipeline: preprocess corpus, train the "
            "tokenizer, train MiniGPT with the shared Trainer, then evaluate."
        )
    )
    add_config_arg(parser)
    parser.add_argument(
        "--profile",
        default="moderate_clean",
        help="Preprocess profile: minimal_clean or moderate_clean.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help=(
            "Corpus file/directory (repeatable). Defaults to "
            "phase_config.corpus_path from the config."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Die Sprache ist",
        help="Prompt for the evaluation sample (default: 'Die Sprache ist').",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for sample generation (default: config seed).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the config's output_dir for this run.",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if cfg["phase"] != "gpt":
        print(
            f"error: {args.config}: phase is {cfg['phase']!r}, "
            "expected 'gpt' for the end-to-end pipeline",
            file=sys.stderr,
        )
        return 1

    try:
        options = PipelineOptions(
            profile=args.profile,
            inputs=tuple(args.input or ()),
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            generation_seed=args.sample_seed,
        )
    except (PipelineError, PreprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        run = create_run(cfg, output_dir=args.output_dir)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # T-503: SIGTERM saves a checkpoint just like Ctrl+C.
    previous_sigterm = None
    try:
        previous_sigterm = signal.signal(signal.SIGTERM, _sigterm_to_interrupt)
    except ValueError:  # not the main thread (tests / embedding)
        previous_sigterm = None

    try:
        result = run_pipeline(cfg, options, run=run)
    except (PipelineError, TrainerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        run.mark_finished("failed")
        return 1
    except TrainingOOMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        run.mark_finished("failed")
        return 2
    except KeyboardInterrupt:
        run.mark_finished("interrupted")
        print(f"interrupted; checkpoint saved in {run.run_dir}", file=sys.stderr)
        return 130
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)

    run.mark_finished("finished")
    val_loss = result.eval_metrics.get("val_loss")
    val_ppl = result.eval_metrics.get("val_perplexity")
    print(f"run directory: {run.run_dir}")
    print(
        "pipeline: "
        f"{result.processed.profile.name} -> tokenizer "
        f"(vocab={result.tokenizer.bpe.vocab_size}) -> "
        f"GPT ({result.data.datasets.train_windows} train windows) -> "
        f"val_loss={val_loss:.4f} val_perplexity={val_ppl:.2f}"
    )
    print(f"sample: {result.sample[:200]!r}")
    print(f"pipeline report: {result.pipeline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
