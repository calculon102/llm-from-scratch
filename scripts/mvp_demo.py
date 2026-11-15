"""One-command MVP demo: reproduce the M0 + M1 + M2 + core M4 flow (T-606).

Usage::

    python -m scripts.mvp_demo
    python -m scripts.mvp_demo --device cpu --max-new-tokens 20

The demo creates a fresh output folder under ``artifacts/mvp_demo/`` (or the
``--output-dir`` you pass), writes resolved copies of the tiny configs there,
and then runs:

1. the XOR training phase (M1),
2. the tokenizer phase (M2), and
3. the end-to-end preprocess + tokenizer + MiniGPT + evaluation pipeline
   (core M4, wrapped by M6).

Each stage is checked for the artifacts a newcomer should see (a final
checkpoint, a tokenizer bundle, ``pipeline.json``, an evaluation report and a
generated sample).  The command exits 0 only when the whole flow passed.

This is intentionally the *tiny* path: the bundled CC0 German sample corpus is
used, so the demo needs no network access and finishes in seconds on CPU (or
a few seconds on the 7900 XTX).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from core.artifacts import ArtifactError, RunSummary, list_runs
from core.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "xor": REPO_ROOT / "configs" / "xor_tiny_debug.yaml",
    "tokenizer": REPO_ROOT / "configs" / "tokenizer_tiny_debug.yaml",
    "gpt": REPO_ROOT / "configs" / "gpt_tiny_debug.yaml",
}
SAMPLE_CORPUS = REPO_ROOT / "data" / "sample_de"


class DemoError(RuntimeError):
    """A demo stage failed or its artifacts are missing."""


@dataclass(frozen=True)
class DemoStage:
    name: str
    detail: str
    run_dir: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DemoError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DemoError(f"expected a JSON object in {path}")
    return data


def _last_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return {}
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("kind") == "summary":
            return record
    return {}


def _find_run(root: Path, run_name: str) -> RunSummary:
    matches = [run for run in list_runs(root, limit=100) if run.run_name == run_name]
    _require(bool(matches), f"no run named {run_name!r} found under {root}")
    return matches[0]


def _write_config(
    cfg: dict[str, Any],
    path: Path,
    *,
    output_dir: Path,
    run_name: str,
    device: str | None,
) -> None:
    """Write a resolved config copy with a demo-local output folder."""
    resolved = dict(cfg)
    resolved["output_dir"] = str(output_dir)
    resolved["run_name"] = run_name
    if device is not None:
        resolved["device"] = device
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run_stage(label: str, main: Any, argv: list[str]) -> float:
    print(f"\n== {label} ==")
    started = time.time()
    rc = main(argv)
    elapsed = time.time() - started
    _require(rc == 0, f"{label} failed with exit code {rc}")
    print(f"== {label} passed in {elapsed:.1f}s ==")
    return elapsed


def _check_xor(run: RunSummary) -> DemoStage:
    _require(run.status == "finished", f"XOR run status is {run.status!r}")
    checkpoint = run.run_dir / "checkpoints" / "final.pt"
    _require(checkpoint.is_file(), f"XOR final checkpoint missing: {checkpoint}")
    _require(
        (run.run_dir / "gradcheck_report.txt").is_file(),
        "XOR gradcheck report missing (config sets gradcheck: true)",
    )
    summary = _last_summary(run.run_dir)
    steps = int(summary.get("steps") or 0)
    _require(steps > 0, "XOR run has no training steps")
    best = summary.get("best_val")
    detail = f"steps={steps}"
    if isinstance(best, (int, float)):
        detail += f", best_val={best:.4f}"
    return DemoStage("XOR phase (M1)", detail, run.run_dir)


def _check_tokenizer(run: RunSummary) -> DemoStage:
    _require(run.status == "finished", f"tokenizer run status is {run.status!r}")
    bundle_path = run.run_dir / "tokenizers" / "bundle.json"
    _require(bundle_path.is_file(), f"tokenizer bundle missing: {bundle_path}")
    _require(
        (run.run_dir / "corpus" / "cleaned.txt").is_file(),
        f"cleaned corpus missing under {run.run_dir / 'corpus'}",
    )
    bundle = _read_json(bundle_path)
    char_vocab = bundle.get("char_vocab_size")
    bpe_vocab = bundle.get("bpe_vocab_size")
    merges = bundle.get("bpe_merges")
    detail = f"char_vocab={char_vocab}, bpe_vocab={bpe_vocab}, merges={merges}"
    return DemoStage("Tokenizer phase (M2)", detail, run.run_dir)


def _check_pipeline(run: RunSummary) -> DemoStage:
    _require(run.status == "finished", f"pipeline run status is {run.status!r}")
    report_path = run.run_dir / "pipeline.json"
    _require(report_path.is_file(), f"pipeline report missing: {report_path}")
    report = _read_json(report_path)
    _require(report.get("status") == "finished", "pipeline report is not finished")

    stages = report.get("stages")
    _require(isinstance(stages, dict), "pipeline report has no stages")
    expected = {"preprocess", "tokenizer", "gpt_setup", "train", "eval"}
    missing = expected.difference(stages)
    _require(not missing, f"pipeline stages missing: {sorted(missing)}")

    model_path = run.run_dir / "gpt" / "model.pt"
    eval_path = run.run_dir / "eval" / "report.json"
    sample_path = run.run_dir / "eval" / "sample.txt"
    _require(model_path.is_file(), f"MiniGPT model missing: {model_path}")
    _require(eval_path.is_file(), f"evaluation report missing: {eval_path}")
    _require(sample_path.is_file(), f"generated sample missing: {sample_path}")

    evaluation = _read_json(eval_path)
    loss = evaluation.get("val_loss")
    perplexity = evaluation.get("val_perplexity")
    sample = sample_path.read_text(encoding="utf-8").strip()
    _require(bool(sample), "generated sample is empty")
    detail = "stages=preprocess→tokenizer→gpt→eval"
    if isinstance(loss, (int, float)):
        detail += f", val_loss={loss:.4f}"
    if isinstance(perplexity, (int, float)):
        detail += f", val_perplexity={perplexity:.2f}"
    detail += f', sample="{sample[:60]}"'
    return DemoStage("MiniGPT pipeline (core M4)", detail, run.run_dir)


def _environment(device: str | None) -> str:
    from core.device import describe_device, get_device_info

    cfg = load_config(CONFIGS["gpt"])
    if device is not None:
        cfg["device"] = device
    info = get_device_info(cfg)
    try:
        import torch

        version = torch.__version__
    except Exception:  # pragma: no cover - torch is a runtime dependency
        version = "unavailable"
    return f"Python {platform.python_version()}, torch {version}, {describe_device(info)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete tiny MVP flow (XOR, tokenizer, MiniGPT pipeline) "
            "and verify its artifacts."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/mvp_demo",
        help="parent folder for this demo; a timestamped subfolder is created "
        "(default: artifacts/mvp_demo)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="override the device in all demo configs (default: keep config value)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="tokens generated by the pipeline evaluation sample (default: 40)",
    )
    args = parser.parse_args(argv)

    from core.device import DeviceError

    started = time.time()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent = Path(args.output_dir)
    if not parent.is_absolute():
        parent = REPO_ROOT / parent
    root = parent / stamp
    runs_dir = root / "runs"
    pipeline_dir = root / "pipeline"
    config_dir = root / "configs"
    root.mkdir(parents=True, exist_ok=False)

    print("llm-from-scratch MVP demo")
    print(f"artifacts: {root}")

    try:
        # Register every phase schema before validating the config copies.
        import core.gpt  # noqa: F401
        import core.tokenizer  # noqa: F401
        import core.xor  # noqa: F401

        _require(SAMPLE_CORPUS.is_dir(), f"sample corpus missing: {SAMPLE_CORPUS}")
        environment = _environment(args.device)
        print(f"environment: {environment}")

        configs: dict[str, dict[str, Any]] = {}
        for name, path in CONFIGS.items():
            _require(path.is_file(), f"config missing: {path}")
            configs[name] = load_config(path)
        print(f"configs: {len(configs)} validated")

        _write_config(
            configs["xor"],
            config_dir / "xor.yaml",
            output_dir=runs_dir,
            run_name="mvp_xor",
            device=args.device,
        )
        _write_config(
            configs["tokenizer"],
            config_dir / "tokenizer.yaml",
            output_dir=runs_dir,
            run_name="mvp_tokenizer",
            device=args.device,
        )
        _write_config(
            configs["gpt"],
            config_dir / "gpt.yaml",
            output_dir=pipeline_dir,
            run_name="mvp_pipeline",
            device=args.device,
        )

        # Import the CLI entrypoints lazily so --help works without torch.
        from scripts import pipeline as pipeline_cli
        from scripts import tokenizer as tokenizer_cli
        from scripts import train as train_cli

        _run_stage(
            "XOR training (M1)",
            train_cli.main,
            ["--config", str(config_dir / "xor.yaml")],
        )
        xor = _check_xor(_find_run(runs_dir, "mvp_xor"))

        _run_stage(
            "Tokenizer training (M2)",
            tokenizer_cli.main,
            ["--config", str(config_dir / "tokenizer.yaml")],
        )
        tokenizer = _check_tokenizer(_find_run(runs_dir, "mvp_tokenizer"))

        _run_stage(
            "MiniGPT pipeline (M6 wrapper around core M4)",
            pipeline_cli.main,
            [
                "--config",
                str(config_dir / "gpt.yaml"),
                "--profile",
                "moderate_clean",
                "--max-new-tokens",
                str(args.max_new_tokens),
            ],
        )
        gpt = _check_pipeline(_find_run(pipeline_dir, "mvp_pipeline"))
    except (DemoError, ConfigError, ArtifactError, DeviceError) as exc:
        print(f"\nMVP demo failed: {exc}", file=sys.stderr)
        print(f"partial artifacts: {root}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nMVP demo interrupted", file=sys.stderr)
        return 130

    elapsed = time.time() - started
    print("\n" + "=" * 68)
    print("MVP demo complete")
    print("=" * 68)
    print(f"environment : {environment}")
    print(f"configs     : {len(configs)} validated")
    for stage in (xor, tokenizer, gpt):
        print(f"{stage.name:<28}: {stage.detail}")
        print(f"{'':<28}  {stage.run_dir}")
    print(f"artifacts   : {root}")
    print(f"wall time   : {elapsed:.1f}s")
    print("\nNext steps:")
    print("  python -m scripts.ui")
    print("  open http://127.0.0.1:8080/runs and inspect the new runs")
    print("  see docs/release_checklist.md for the full release checklist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
