"""Per-run artifact manager (T-004).

Every run gets its own folder under ``output_dir`` (default ``artifacts/``)
laid out as::

    artifacts/
    ├── runs.jsonl                              # index of all runs (T-604)
    └── 20261102-224000_xor_tiny/
        ├── config.yaml      # snapshot of the resolved config
        ├── run.log          # run log (also mirrored to the console)
        ├── run_meta.json    # run metadata (status, start/end times)
        ├── checkpoints/     # step_000123.pt, final.pt, ...
        └── plots/           # PNG figures

Usage::

    from core.artifacts import create_run

    run = create_run(cfg)          # cfg = load_config("configs/xxx.yaml")
    run.log("starting training")
    path = run.save_checkpoint(model.state_dict(), step=100)
    run.save_plot(fig, "loss")
    run.mark_finished()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ArtifactError", "RunArtifacts", "create_run"]


class ArtifactError(Exception):
    """Raised when an artifact operation fails (bad input, missing file,
    missing optional dependency)."""


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class RunArtifacts:
    """Manages the artifact folder of one run.

    Parameters
    ----------
    config:
        A validated config mapping (from :func:`core.config.load_config`).
    output_dir:
        Override for ``config["output_dir"]``.
    run_name:
        Override for ``config["run_name"]``.
    """

    def __init__(
        self,
        config: dict,
        output_dir: str | Path | None = None,
        run_name: str | None = None,
    ) -> None:
        if not isinstance(config, dict):
            raise ArtifactError(
                "config must be a mapping; use core.config.load_config to "
                "load and validate one"
            )
        self._config = config
        root = Path(output_dir or config.get("output_dir") or "artifacts")
        name = str(run_name or config.get("run_name") or "run")
        self.run_dir = self._unique_run_dir(root, name)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.log_path.touch()
        self._start_time = datetime.now()
        self._logger: logging.Logger | None = None
        self._write_config_snapshot()
        self._write_meta(status="running")
        self._append_index_line()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def meta_path(self) -> Path:
        return self.run_dir / "run_meta.json"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    @property
    def index_path(self) -> Path:
        return self.run_dir.parent / "runs.jsonl"

    @staticmethod
    def _unique_run_dir(root: Path, name: str) -> Path:
        """Create ``root/<timestamp>_<name>``, disambiguating collisions."""
        base = f"{_timestamp()}_{name}"
        candidate = root / base
        counter = 2
        while candidate.exists():
            candidate = root / f"{base}_{counter}"
            counter += 1
        candidate.mkdir(parents=True)
        return candidate

    # ------------------------------------------------------------------
    # Config snapshot
    # ------------------------------------------------------------------

    def _write_config_snapshot(self) -> None:
        try:
            self.config_path.write_text(
                yaml.safe_dump(self._config, sort_keys=True, allow_unicode=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactError(
                f"cannot write config snapshot {self.config_path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @property
    def logger(self) -> logging.Logger:
        """A logger that writes to ``run.log`` and mirrors to the console."""
        if self._logger is None:
            logger = logging.getLogger(self.run_dir.name)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if not logger.handlers:
                fmt = logging.Formatter(
                    "%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
                file_handler.setFormatter(fmt)
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(fmt)
                logger.addHandler(file_handler)
                logger.addHandler(console_handler)
            self._logger = logger
        return self._logger

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log ``message`` to ``run.log`` (and the console)."""
        self.logger.log(level, message)

    # ------------------------------------------------------------------
    # Metadata / index
    # ------------------------------------------------------------------

    def _meta(self, status: str, end_time: datetime | None) -> dict:
        cfg = self._config
        return {
            "run_dir": str(self.run_dir),
            "run_name": cfg.get("run_name"),
            "profile": cfg.get("profile"),
            "phase": cfg.get("phase"),
            "seed": cfg.get("seed"),
            "device": cfg.get("device"),
            "amp": cfg.get("amp"),
            "start_time": self._start_time.isoformat(timespec="seconds"),
            "end_time": end_time.isoformat(timespec="seconds") if end_time else None,
            "status": status,
        }

    def _write_meta(self, status: str, end_time: datetime | None = None) -> None:
        meta = self._meta(status, end_time)
        self.meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _append_index_line(self) -> None:
        """Append one JSON line per run to ``runs.jsonl`` in the output root."""
        cfg = self._config
        line = {
            "run_dir": str(self.run_dir),
            "run_name": cfg.get("run_name"),
            "profile": cfg.get("profile"),
            "phase": cfg.get("phase"),
            "start_time": self._start_time.isoformat(timespec="seconds"),
        }
        try:
            with self.index_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise ArtifactError(f"cannot update run index {self.index_path}: {exc}") from exc

    def mark_finished(self, status: str = "finished") -> None:
        """Record the run's end time and final status (``finished``/``failed``)."""
        self._write_meta(status, datetime.now())

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: Any, step: int | None = None) -> Path:
        """Save ``state`` (e.g. a torch state dict) to ``checkpoints/``.

        With ``step`` the file is named ``step_<step:06d>.pt``; without it,
        ``final.pt``.  Returns the checkpoint path.
        """
        import torch  # local import: keep this module importable without torch

        filename = f"step_{step:06d}.pt" if step is not None else "final.pt"
        path = self.checkpoints_dir / filename
        try:
            torch.save(state, path)
        except (OSError, RuntimeError) as exc:
            raise ArtifactError(f"cannot save checkpoint {path}: {exc}") from exc
        self.log(f"checkpoint saved: {path.name}")
        return path

    def latest_checkpoint(self) -> Path | None:
        """Return the newest ``step_*.pt``, or ``final.pt`` if that is all."""
        candidates = sorted(self.checkpoints_dir.glob("step_*.pt"))
        if candidates:
            return candidates[-1]
        final = self.checkpoints_dir / "final.pt"
        return final if final.is_file() else None

    def load_checkpoint(self, path: str | Path | None = None) -> Any:
        """Load a checkpoint (default: the latest one) onto CPU."""
        import torch  # local import: keep this module importable without torch

        if path is None:
            path = self.latest_checkpoint()
            if path is None:
                raise ArtifactError(f"no checkpoints found in {self.checkpoints_dir}")
        path = Path(path)
        if not path.is_file():
            raise ArtifactError(f"checkpoint not found: {path}")
        try:
            return torch.load(path, map_location="cpu")
        except Exception as exc:
            raise ArtifactError(f"cannot load checkpoint {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def save_plot(self, fig: Any, name: str) -> Path:
        """Save a matplotlib figure to ``plots/`` as PNG.

        ``name`` may omit the extension (``.png`` is added).  Requires the
        optional viz extra (``pip install -e ".[viz]"``).
        """
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as exc:
            raise ArtifactError(
                "save_plot requires matplotlib; install the viz extra: "
                "pip install -e \".[viz]\""
            ) from exc

        if not name.endswith((".png", ".svg")):
            name = f"{name}.png"
        path = self.plots_dir / name
        try:
            fig.savefig(path, dpi=140, bbox_inches="tight")
        except OSError as exc:
            raise ArtifactError(f"cannot save plot {path}: {exc}") from exc
        self.log(f"plot saved: {path.name}")
        return path


def create_run(
    config: dict,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
) -> RunArtifacts:
    """Create a new run folder from a validated config mapping.

    This is the entry point every training phase (T-005 onward) uses to
    open its run: the config snapshot, log file, metadata, checkpoint and
    plot directories are created up front.
    """
    return RunArtifacts(config, output_dir=output_dir, run_name=run_name)
