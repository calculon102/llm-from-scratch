"""CLI: create a demo run and produce all artifact types (T-004).

Usage:
    python -m scripts.demo_run --config configs/tiny_debug.yaml

Creates a run folder under the config's ``output_dir`` and writes:

- a config snapshot (``config.yaml``)
- a few log lines (``run.log``)
- a tiny checkpoint (``checkpoints/step_000001.pt``)
- a simple loss-curve plot (``plots/loss.png``, requires the viz extra)

Prints the run directory on success; exits 1 on a bad config.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch

from core.artifacts import ArtifactError, create_run
from core.config import ConfigError, add_config_arg, load_config
from core.device import DeviceError, describe_device, get_device_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a demo run with config snapshot, logs, checkpoint, and plot."
    )
    add_config_arg(parser)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        run = create_run(cfg)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        info = get_device_info(cfg)
    except DeviceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        run.mark_finished("failed")
        return 1
    run.log(f"startup: {describe_device(info)}")
    run.log(
        f"demo run '{cfg['run_name']}' started "
        f"(profile={cfg['profile']}, device={info.device}, amp={cfg['amp']}, seed={cfg['seed']})"
    )

    # Checkpoint: a tiny state dict (the real trainer saves model weights).
    state = {"model": torch.zeros(2, 2), "step": 1, "loss": 0.693}
    run.save_checkpoint(state, step=1)

    # Plot: a fake decreasing loss curve.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.arange(1, 51)
        loss = 0.693 * np.exp(-steps / 10.0) + 0.01
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(steps, loss)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("demo loss curve")
        run.save_plot(fig, "loss")
        plt.close(fig)
    except ArtifactError as exc:
        print(f"warning: {exc}", file=sys.stderr)
        run.log(f"plot skipped: {exc}", level=logging.WARNING)

    run.mark_finished()
    print(f"run directory: {run.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
