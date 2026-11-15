"""End-to-end educational pipeline: preprocess → tokenizer → GPT → eval (T-603).

One command, one run folder, every stage persisted::

    python -m scripts.pipeline --config configs/gpt_tiny_debug.yaml

The stages deliberately reuse the same building blocks as the individual
CLIs instead of duplicating logic:

1. :func:`core.corpus.preprocess.preprocess_corpus` writes a reproducible
   ``corpus/`` (profile, cleaned text, per-file report, input/output
   hashes).
2. :func:`core.tokenizer.pipeline.train_bundle_from_config` trains the
   char+BPE bundle on exactly that text and saves it under
   ``tokenizers/``.
3. :func:`core.gpt.pipeline.prepare_gpt_data` encodes the same corpus,
   using the bundle from stage 2, and chunks train/validation windows.
4. The shared :class:`core.trainer.Trainer` trains the model; its
   ``on_end`` callback writes ``gpt/``.
5. Evaluation computes token-weighted validation perplexity and generates
   a deterministic sample, saved in ``eval/``.

The combined provenance lands in ``pipeline.json``; the normal run
artifacts (config snapshot, log, metrics, checkpoints) are unchanged.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.artifacts import ArtifactError, RunArtifacts
from core.corpus.preprocess import (
    PreprocessError,
    PreprocessResult,
    get_profile,
    preprocess_corpus,
)
from core.device import describe_device, get_device_info
from core.gpt import (
    GPTData,
    GPTPipelineError,
    build_gpt_phase_from_data,
    effective_phase_config,
    generate_text,
    prepare_gpt_data,
)
from core.tokenizer.common import TokenizerError
from core.tokenizer.pipeline import (
    REPO_ROOT,
    TokenizerBundle,
    resolve_corpus_path,
    save_bundle,
    train_bundle_from_config,
)
from core.trainer import Trainer, TrainerError, seed_everything

__all__ = [
    "PIPELINE_FORMAT",
    "PIPELINE_VERSION",
    "PipelineError",
    "PipelineOptions",
    "PipelineResult",
    "run_pipeline",
]

PIPELINE_FORMAT = "llm-from-scratch-pipeline"
PIPELINE_VERSION = 1


class PipelineError(Exception):
    """Raised when the end-to-end pipeline cannot complete."""


@dataclass(frozen=True)
class PipelineOptions:
    """User-facing pipeline settings not already in the YAML config."""

    profile: str = "moderate_clean"
    inputs: tuple[str, ...] = ()
    prompt: str = "Die Sprache ist"
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 1.0
    generation_seed: int | None = None

    def __post_init__(self) -> None:
        get_profile(self.profile)  # raises PreprocessError for unknown names
        if self.max_new_tokens < 0:
            raise PipelineError("max_new_tokens must be >= 0")
        if self.temperature < 0:
            raise PipelineError("temperature must be >= 0")
        if self.top_k < 0:
            raise PipelineError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise PipelineError("top_p must be in (0, 1]")


@dataclass
class PipelineResult:
    """Everything produced by :func:`run_pipeline`."""

    run: RunArtifacts
    processed: PreprocessResult
    tokenizer: TokenizerBundle
    data: GPTData
    train_summary: dict[str, Any]
    eval_metrics: dict[str, float]
    sample: str
    pipeline_path: Path

    @property
    def run_dir(self) -> Path:
        return Path(self.run.run_dir)


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _pipeline_report(
    cfg: dict,
    options: PipelineOptions,
    *,
    run_dir: Path,
    device: str,
    stages: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": PIPELINE_FORMAT,
        "version": PIPELINE_VERSION,
        "status": "finished",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": cfg.get("run_name"),
        "profile": cfg.get("profile"),
        "seed": cfg.get("seed"),
        "device": device,
        "options": {
            "preprocess_profile": options.profile,
            "inputs": list(options.inputs),
            "prompt": options.prompt,
            "max_new_tokens": options.max_new_tokens,
            "temperature": options.temperature,
            "top_k": options.top_k,
            "top_p": options.top_p,
            "generation_seed": options.generation_seed,
        },
        "stages": stages,
        "artifacts": {
            "corpus": "corpus/cleaned.txt",
            "corpus_manifest": "corpus/manifest.json",
            "tokenizer": "tokenizers/bundle.json",
            "model": "gpt/model.pt",
            "eval": "eval/report.json",
            "sample": "eval/sample.txt",
        },
        "run_dir": str(run_dir),
    }


def run_pipeline(
    cfg: dict,
    options: PipelineOptions | None = None,
    *,
    run: RunArtifacts,
) -> PipelineResult:
    """Run all pipeline stages inside an existing run folder.

    ``run`` is created by the caller so configuration/device failures are
    still recorded as artifacts.  The caller also owns top-level error
    handling (marking the run failed/interrupted).
    """
    options = options or PipelineOptions()
    if cfg.get("phase") != "gpt":
        raise PipelineError(
            f"pipeline requires phase 'gpt', got {cfg.get('phase')!r}; "
            "use a GPT config such as configs/gpt_tiny_debug.yaml"
        )
    run_dir = Path(run.run_dir)
    pc = effective_phase_config(cfg)

    try:
        info = get_device_info(cfg)
    except Exception as exc:  # DeviceError and friends
        raise PipelineError(f"cannot resolve training device: {exc}") from exc

    seed_everything(int(cfg["seed"]))
    run.log(f"pipeline: {describe_device(info)}")
    run.log(
        f"pipeline stages: profile={options.profile} "
        f"phase=gpt seed={cfg['seed']} device={info.device}"
    )

    stages: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Preprocess
    # ------------------------------------------------------------------
    started = time.perf_counter()
    inputs: list[str] = list(options.inputs)
    if not inputs:
        inputs = [str(resolve_corpus_path(pc["corpus_path"], REPO_ROOT))]
    run.log(f"pipeline/preprocess: {', '.join(inputs)} -> {options.profile}")
    try:
        processed = preprocess_corpus(
            inputs if len(inputs) > 1 else inputs[0],
            options.profile,
            run_dir / "corpus",
        )
    except PreprocessError as exc:
        raise PipelineError(f"preprocess stage failed: {exc}") from exc
    corpus_stats = processed.corpus.stats
    stages["preprocess"] = {
        "profile": options.profile,
        "inputs": inputs,
        "output": "corpus/cleaned.txt",
        "sha256": processed.output_sha256,
        "input_digest": processed.input_digest,
        "bytes": processed.output_bytes,
        "chars": len(processed.corpus.text),
        "lines": corpus_stats.get("lines_out", 0),
        "files": corpus_stats.get("files_ok", 0),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    run.log(
        f"pipeline/preprocess: {corpus_stats.get('files_ok', 0)} file(s), "
        f"{len(processed.corpus.text)} chars, sha256={processed.output_sha256[:12]}…"
    )
    run.log_metric({"stage": "preprocess", **stages["preprocess"]})

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    started = time.perf_counter()
    run.log(
        "pipeline/tokenizer: training char+BPE "
        f"(vocab_size={pc['vocab_size']}, min_frequency={pc['min_frequency']}, "
        f"max_token_length={pc['max_token_length']})"
    )
    try:
        tokenizer, _ = train_bundle_from_config(cfg, corpus=processed.corpus)
        save_bundle(tokenizer, run_dir / "tokenizers")
    except TokenizerError as exc:
        raise PipelineError(f"tokenizer stage failed: {exc}") from exc
    stages["tokenizer"] = {
        "path": "tokenizers",
        "source": tokenizer.source,
        "char_vocab_size": tokenizer.char.vocab_size,
        "vocab_size": tokenizer.bpe.vocab_size,
        "merges": len(tokenizer.bpe.merges),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    run.log(
        f"pipeline/tokenizer: char={tokenizer.char.vocab_size} "
        f"bpe={tokenizer.bpe.vocab_size} merges={len(tokenizer.bpe.merges)}"
    )
    run.log_metric({"stage": "tokenizer", **stages["tokenizer"]})

    # ------------------------------------------------------------------
    # 3. GPT data + model (shared Trainer)
    # ------------------------------------------------------------------
    started = time.perf_counter()
    try:
        data = prepare_gpt_data(
            cfg, corpus=processed.corpus, tokenizer=tokenizer, base_dir=REPO_ROOT
        )
        phase = build_gpt_phase_from_data(cfg, data)
    except (GPTPipelineError, TokenizerError) as exc:
        raise PipelineError(f"GPT stage failed: {exc}") from exc
    stages["gpt_setup"] = {
        "tokens": len(data.token_ids),
        "context_length": data.datasets.context_length,
        "stride": data.datasets.stride,
        "train_windows": data.datasets.train_windows,
        "val_windows": data.datasets.val_windows,
        "parameters": phase.model.num_parameters(),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    run.log(
        f"pipeline/gpt: {len(data.token_ids)} tokens, "
        f"{data.datasets.train_windows} train / {data.datasets.val_windows} val "
        f"windows, {phase.model.num_parameters()} params"
    )
    run.log_metric({"stage": "gpt_setup", **stages["gpt_setup"]})

    trainer = Trainer.from_bundle(
        phase, config=cfg, run=run, device=info.device, amp=cfg["amp"]
    )
    if cfg.get("gradcheck"):
        trainer.check_gradients()

    started = time.perf_counter()
    try:
        train_summary = trainer.train()
    except TrainerError as exc:
        raise PipelineError(f"training failed: {exc}") from exc
    stages["train"] = {
        "steps": train_summary.get("steps"),
        "epochs": train_summary.get("epochs"),
        "best_val": train_summary.get("best_val"),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    run.log(
        f"pipeline/train: steps={stages['train']['steps']} "
        f"epochs={stages['train']['epochs']} "
        f"best_val={stages['train']['best_val']}"
    )

    # ------------------------------------------------------------------
    # 4. Evaluation + deterministic sample
    # ------------------------------------------------------------------
    started = time.perf_counter()
    try:
        eval_metrics = trainer.evaluate()
    except Exception as exc:
        raise PipelineError(f"evaluation failed: {exc}") from exc
    if "val_perplexity" not in eval_metrics:
        loss = eval_metrics.get("val_loss", eval_metrics.get("train_loss"))
        if loss is not None:
            eval_metrics["val_perplexity"] = math.exp(min(float(loss), 50.0))

    generation_settings = {
        "prompt": options.prompt,
        "max_new_tokens": options.max_new_tokens,
        "temperature": options.temperature,
        "top_k": options.top_k,
        "top_p": options.top_p,
        "seed": options.generation_seed
        if options.generation_seed is not None
        else int(cfg["seed"]),
    }
    try:
        sample = generate_text(
            trainer.model,
            tokenizer,
            options.prompt,
            max_new_tokens=options.max_new_tokens,
            temperature=options.temperature,
            top_k=options.top_k,
            top_p=options.top_p,
            seed=generation_settings["seed"],
        )
    except (ValueError, RuntimeError) as exc:
        raise PipelineError(f"sample generation failed: {exc}") from exc

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_report = {
        "metrics": eval_metrics,
        "generation": generation_settings,
    }
    (eval_dir / "report.json").write_text(
        json.dumps(eval_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "sample.txt").write_text(sample + "\n", encoding="utf-8")
    stages["eval"] = {
        **{key: float(value) for key, value in eval_metrics.items()},
        "sample_chars": len(sample),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    run.log(
        f"pipeline/eval: val_loss={eval_metrics.get('val_loss')} "
        f"val_perplexity={eval_metrics.get('val_perplexity')}"
    )
    run.log(f"pipeline/eval sample: {sample[:160]!r}")
    run.log_metric({"stage": "eval", **stages["eval"]})

    pipeline_path = run_dir / "pipeline.json"
    pipeline_path.write_text(
        json.dumps(
            _pipeline_report(
                cfg,
                options,
                run_dir=run_dir,
                device=str(info.device),
                stages=stages,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    run.log(f"pipeline report: {pipeline_path}")
    return PipelineResult(
        run=run,
        processed=processed,
        tokenizer=tokenizer,
        data=data,
        train_summary=train_summary,
        eval_metrics=eval_metrics,
        sample=sample,
        pipeline_path=pipeline_path,
    )
