"""Benchmark the M5 profile ladder (T-504).

Runs a short training slice for each requested profile through the real
GPT pipeline + shared Trainer, measures throughput and memory (T-502), and
writes a speed/quality comparison table::

    python -m scripts.benchmark --profiles tiny_debug tiny_train --steps 50
    python -m scripts.benchmark --profiles all --corpus data/full_de --steps 100

Each benchmark is a normal artifact run under
``artifacts/benchmarks/<timestamp>/runs/``, so its config snapshot, log,
``metrics.jsonl`` and final checkpoint stay inspectable.  The top-level
folder gets ``benchmark.json`` (machine-readable) and ``report.md`` (the
table to paste into docs/issues).

Important: a benchmark is a *short slice*, not a quality run.  Validation
loss/perplexity is included because it moves in the right direction, but
the run length (and the bundled sample corpus) is far too small for a
capability claim.  Use the profile's full ``max_steps`` for that.

Profiles whose context/validation split does not fit the supplied corpus
are reported as ``skipped`` instead of aborting the whole benchmark — the
bundled ``data/sample_de`` is only ~880 tokens, so ``small`` and above need
the larger M6 corpus (or ``--corpus``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

import core.gpt  # noqa: F401  # registers the phase schema + factory
from core.artifacts import ArtifactError, create_run
from core.config import ConfigError, validate_config
from core.device import DeviceError, describe_device, get_device_info
from core.gpt import GPTPipelineError, build_gpt_phase
from core.scaling import PROFILE_ORDER, profile_config, profile_table_rows
from core.trainer import Trainer, TrainerError, seed_everything

__all__ = ["BenchmarkResult", "benchmark_profile", "format_report", "main"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "benchmarks"


@dataclass
class BenchmarkResult:
    """One profile's measured slice (or the reason it was skipped)."""

    profile: str
    status: str
    parameters: int | None = None
    vocab_size: int | None = None
    context_length: int | None = None
    batch_size: int | None = None
    steps: int | None = None
    tokens: int | None = None
    tokens_per_sec: float | None = None
    steps_per_sec: float | None = None
    elapsed_s: float | None = None
    val_loss: float | None = None
    val_perplexity: float | None = None
    cpu_peak_mb: float | None = None
    gpu_peak_mb: float | None = None
    gpu_total_mb: float | None = None
    run_dir: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mib(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1024:
        return f"{value / 1024:.2f} GiB"
    return f"{value:.0f} MiB"


def _num(value: int | None) -> str:
    return f"{value:,}" if value is not None else "-"


def _float(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_report(results: Sequence[BenchmarkResult], metadata: dict[str, Any]) -> str:
    """Render the Markdown speed/quality comparison table."""
    lines = [
        "# Scaling benchmark",
        "",
        f"- Generated: {metadata.get('generated_at', '?')}",
        f"- Device: {metadata.get('device', '?')}",
        f"- Corpus: `{metadata.get('corpus', '?')}`",
        f"- Steps per profile: {metadata.get('steps', '?')}",
        f"- Torch: {metadata.get('torch', '?')}",
        "",
        "| Profile | Params | Context | Batch | Vocab | Tok/s | Steps/s | Val loss | Perplexity | Peak mem | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        memo = (
            f"{_mib(result.cpu_peak_mb)} CPU"
            if result.gpu_peak_mb is None
            else f"{_mib(result.gpu_peak_mb)} / {_mib(result.gpu_total_mb)} GPU"
        )
        note = result.status if not result.note else f"{result.status}: {result.note}"
        lines.append(
            "| {profile} | {params} | {ctx} | {batch} | {vocab} | {tps} | {sps} | "
            "{loss} | {ppl} | {mem} | {note} |".format(
                profile=result.profile,
                params=_num(result.parameters),
                ctx=_num(result.context_length),
                batch=_num(result.batch_size),
                vocab=_num(result.vocab_size),
                tps=_float(result.tokens_per_sec, 0),
                sps=_float(result.steps_per_sec, 1),
                loss=_float(result.val_loss, 4),
                ppl=_float(result.val_perplexity, 2),
                mem=memo,
                note=note,
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Throughput excludes evaluation/checkpoint time (T-502 clock is paused).",
            "- A short slice measures speed/memory, not capability; val loss is only a sanity signal.",
            "- Skip reasons usually mean the corpus is too small for that profile's context length.",
            "",
        ]
    )
    return "\n".join(lines)


def benchmark_profile(
    profile: str,
    *,
    steps: int,
    output_root: Path,
    device: torch.device,
    corpus: str | None = None,
    tokenizer_bundle: str | None = None,
    amp: bool | None = None,
    seed: int = 42,
) -> BenchmarkResult:
    """Run one profile slice and return its measurements.

    Pipeline/config errors for this profile are captured as a ``skipped``
    result so one oversized rung does not hide the other measurements.
    """
    cfg = profile_config(
        profile,
        base={
            "run_name": f"bench_{profile}",
            "output_dir": str(output_root / "runs"),
            "device": str(device),
            "seed": seed,
            "max_steps": int(steps),
            # Keep looping until max_steps even when the corpus is tiny;
            # a fully consumed data loader simply ends earlier.
            "epochs": 10_000,
            "save_interval": 1_000_000,
            "log_interval": max(1, min(int(steps), 10)),
        },
    )
    if corpus:
        cfg["phase_config"]["corpus_path"] = corpus
    if tokenizer_bundle:
        # Reuse a trained bundle instead of rerunning pure-Python BPE on the
        # benchmark corpus.  This is what makes full-corpus slices practical.
        cfg["phase_config"]["tokenizer_bundle"] = tokenizer_bundle
    if amp is not None:
        cfg["amp"] = amp

    try:
        cfg = validate_config(cfg)
    except ConfigError as exc:
        return BenchmarkResult(profile=profile, status="invalid", note=str(exc))

    try:
        bundle = build_gpt_phase(cfg)
    except (TrainerError, GPTPipelineError, ValueError, OSError, RuntimeError) as exc:
        return BenchmarkResult(
            profile=profile,
            status="skipped",
            note=f"{type(exc).__name__}: {exc}",
        )

    # The bundle does not expose the data object directly; read the actual
    # architecture from the built model/dataset instead.
    parameters = int(bundle.model.num_parameters()) if hasattr(bundle.model, "num_parameters") else None
    vocab_size = int(getattr(bundle.model, "vocab_size", 0)) or None
    context_length = int(getattr(bundle.model, "context_length", 0)) or None
    batch_size = int(cfg["batch_size"])

    seed_everything(seed)
    try:
        run = create_run(cfg)
    except ArtifactError as exc:
        return BenchmarkResult(profile=profile, status="failed", note=str(exc))

    try:
        trainer = Trainer.from_bundle(
            bundle, config=cfg, run=run, device=device, amp=cfg["amp"]
        )
        summary = trainer.train()
        evaluation = trainer.evaluate()
        run.mark_finished("finished")
    except Exception as exc:  # noqa: BLE001 - report, never abort the suite
        run.mark_finished("failed")
        return BenchmarkResult(
            profile=profile,
            status="failed",
            parameters=parameters,
            vocab_size=vocab_size,
            context_length=context_length,
            batch_size=batch_size,
            run_dir=str(run.run_dir),
            note=f"{type(exc).__name__}: {exc}",
        )

    memory = summary.get("memory") or {}
    return BenchmarkResult(
        profile=profile,
        status="ok",
        parameters=parameters,
        vocab_size=vocab_size,
        context_length=context_length,
        batch_size=batch_size,
        steps=int(summary["steps"]),
        tokens=int(summary["tokens"]),
        tokens_per_sec=float(summary["tokens_per_sec"]),
        steps_per_sec=float(summary["steps_per_sec"]),
        elapsed_s=float(summary["elapsed_s"]),
        val_loss=evaluation.get("val_loss"),
        val_perplexity=evaluation.get("val_perplexity"),
        cpu_peak_mb=memory.get("cpu_peak_mb"),
        gpu_peak_mb=memory.get("gpu_peak_mb"),
        gpu_total_mb=memory.get("gpu_total_mb"),
        run_dir=str(run.run_dir),
    )


def _parse_profiles(values: Sequence[str]) -> list[str]:
    if any(value == "all" for value in values):
        return list(PROFILE_ORDER)
    unknown = [value for value in values if value not in PROFILE_ORDER]
    if unknown:
        raise SystemExit(
            f"unknown profile(s): {', '.join(unknown)}; "
            f"expected ones of {', '.join(PROFILE_ORDER)} or 'all'"
        )
    return list(dict.fromkeys(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark M5 scaling profiles (throughput, memory, val loss)."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["tiny_debug", "tiny_train"],
        help="Profiles to run, or 'all' (default: tiny_debug tiny_train).",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Optimizer steps per profile (default: 50)."
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Corpus path override (default: each profile's data/sample_de).",
    )
    parser.add_argument(
        "--tokenizer-bundle",
        default=None,
        help=(
            "Reuse a trained tokenizer bundle instead of training BPE on the "
            "benchmark corpus (recommended for full-corpus slices)."
        ),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"), help="Execution device."
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force AMP on/off; default follows the profile preset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Benchmark output root (default: artifacts/benchmarks).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.steps < 1:
        parser.error("--steps must be >= 1")

    profiles = _parse_profiles(args.profiles)
    try:
        cfg = validate_config({"device": args.device, "amp": bool(args.amp)})
        info = get_device_info(cfg)
    except (ConfigError, DeviceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output_dir) / stamp
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []
    for profile in profiles:
        print(f"benchmarking {profile} ({args.steps} steps) ...", flush=True)
        result = benchmark_profile(
            profile,
            steps=args.steps,
            output_root=output_root,
            device=info.device,
            corpus=args.corpus,
            tokenizer_bundle=args.tokenizer_bundle,
            amp=args.amp,
            seed=args.seed,
        )
        results.append(result)
        if result.status == "ok":
            print(
                f"  {result.tokens_per_sec:,.0f} tok/s | "
                f"{result.steps_per_sec:.1f} steps/s | "
                f"val_loss={result.val_loss:.4f} | "
                f"peak={_mib(result.gpu_peak_mb or result.cpu_peak_mb)}",
                flush=True,
            )
        else:
            print(f"  {result.status}: {result.note}", flush=True)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": describe_device(info),
        "corpus": args.corpus or "profile default (data/sample_de)",
        "tokenizer_bundle": args.tokenizer_bundle,
        "steps": args.steps,
        "torch": torch.__version__,
        "profiles": profiles,
        "results": [result.to_dict() for result in results],
        "model_scale": profile_table_rows(),
    }
    (output_root / "benchmark.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = format_report(results, metadata)
    report_path = output_root / "report.md"
    report_path.write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"report: {report_path}")
    return 0 if any(result.status == "ok" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
