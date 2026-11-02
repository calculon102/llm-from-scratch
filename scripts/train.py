"""CLI: run a training phase with the shared Trainer (T-005).

Usage:
    python -m scripts.train --config configs/xxx.yaml [--resume PATH]

Loads and validates the YAML config, builds the phase (model + datasets +
objective) via the phase registry, and runs the shared Trainer loop.
Every run is written to its own folder under the config's ``output_dir``
(config snapshot, logs, checkpoints).  ``--resume`` continues from a
checkpoint file or a previous run directory (latest checkpoint used).

Phase factories are registered by the phase modules at import time, e.g.
XOR (T-101)::

    from core.trainer import register_phase
    register_phase("xor", build_xor_phase)
"""

from __future__ import annotations

import argparse
import sys

import torch

from core.artifacts import ArtifactError, create_run
from core.config import ConfigError, add_config_arg, load_config
from core.trainer import (
    Trainer,
    TrainerError,
    build_phase,
    resolve_device,
    seed_everything,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a training phase with the shared Trainer."
    )
    add_config_arg(parser)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint file or previous run directory to resume from.",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        bundle = build_phase(cfg)
    except TrainerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    seed_everything(cfg["seed"])
    try:
        run = create_run(cfg)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    device = resolve_device(cfg)
    run.log(
        f"phase={cfg['phase']} profile={cfg['profile']} device={device} "
        f"amp={cfg['amp']} seed={cfg['seed']}"
    )

    trainer = Trainer.from_bundle(
        bundle, config=cfg, run=run, device=device, amp=cfg["amp"]
    )

    if args.resume:
        try:
            trainer.resume(args.resume)
        except TrainerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            run.mark_finished("failed")
            return 1

    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        run.mark_finished("interrupted")
        print(f"interrupted; checkpoint saved in {run.run_dir}", file=sys.stderr)
        return 130

    run.mark_finished("finished")
    print(f"run directory: {run.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
