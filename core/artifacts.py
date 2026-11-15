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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ArtifactEntry",
    "ArtifactError",
    "ArtifactPreview",
    "DEFAULT_ARTIFACTS_ROOT",
    "RunArtifacts",
    "RunSummary",
    "artifact_entries",
    "artifact_preview",
    "create_run",
    "list_runs",
    "load_run_summary",
]

#: Repository-level default output root (``<repo>/artifacts``).
DEFAULT_ARTIFACTS_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


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
    def metrics_path(self) -> Path:
        """Append-only JSONL metrics stream (T-502)."""
        return self.run_dir / "metrics.jsonl"

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
            # The full run dir path keeps logger names unique across output
            # roots (basenames can collide for same-name runs in the same
            # second, which would make a run inherit another run's handlers).
            logger = logging.getLogger(f"llm_from_scratch.run.{self.run_dir}")
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

    def log_metric(self, record: dict) -> None:
        """Append one JSON metric record to ``metrics.jsonl`` (T-502).

        The Trainer writes throughput/memory snapshots here at every log
        interval plus a final summary.  JSONL keeps the file append-only and
        crash-safe: a failed or interrupted run still has every record
        written before it stopped.
        """
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self.metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            raise ArtifactError(
                f"cannot append metrics to {self.metrics_path}: {exc}"
            ) from exc

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


# ---------------------------------------------------------------------------
# Discovery / browsing helpers (T-604)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSummary:
    """One run as listed by the runs/artifacts browser."""

    run_dir: Path
    run_name: str
    phase: str | None
    profile: str | None
    status: str
    start_time: str | None
    end_time: str | None
    device: str | None = None
    amp: bool | None = None

    @property
    def duration_s(self) -> float | None:
        """Wall-clock duration in seconds, or ``None`` if still running."""
        if not self.start_time or not self.end_time:
            return None
        try:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
        except ValueError:
            return None
        return max((end - start).total_seconds(), 0.0)


def load_run_summary(run_dir: str | Path) -> RunSummary | None:
    """Read ``run_meta.json`` from ``run_dir`` (``None`` if not a run)."""
    path = Path(run_dir)
    meta_path = path / "run_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return RunSummary(
        run_dir=path,
        run_name=str(meta.get("run_name") or path.name),
        phase=meta.get("phase"),
        profile=meta.get("profile"),
        status=str(meta.get("status") or "unknown"),
        start_time=meta.get("start_time"),
        end_time=meta.get("end_time"),
        device=meta.get("device"),
        amp=meta.get("amp"),
    )


def _index_run_dirs(root: Path) -> list[Path]:
    """Read ``runs.jsonl`` (the T-004 index) and return existing run dirs."""
    index = root / "runs.jsonl"
    found: list[Path] = []
    if not index.is_file():
        return found
    try:
        lines = index.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_dir = record.get("run_dir") if isinstance(record, dict) else None
        if not run_dir:
            continue
        path = Path(run_dir)
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            found.append(path)
    return found


def list_runs(
    root: str | Path | None = None, *, limit: int | None = 50
) -> list[RunSummary]:
    """List recent runs, newest first.

    Reads the ``runs.jsonl`` index written by :class:`RunArtifacts` and
    also scans direct subdirectories for ``run_meta.json`` so runs copied
    in manually are still discoverable.  Missing run directories (deleted
    artifacts) are skipped.
    """
    root_path = Path(root) if root is not None else DEFAULT_ARTIFACTS_ROOT
    candidates: list[Path] = _index_run_dirs(root_path)
    if root_path.is_dir():
        seen = {path.resolve() for path in candidates}
        for child in root_path.iterdir():
            if not child.is_dir():
                continue
            try:
                resolved = child.resolve()
            except OSError:
                continue
            if resolved not in seen:
                candidates.append(child)
                seen.add(resolved)

    summaries: list[RunSummary] = []
    for candidate in candidates:
        summary = load_run_summary(candidate)
        if summary is not None:
            summaries.append(summary)
    summaries.sort(
        key=lambda summary: (
            summary.start_time or "",
            summary.run_dir.stat().st_mtime if summary.run_dir.exists() else 0.0,
        ),
        reverse=True,
    )
    return summaries[:limit] if limit is not None else summaries


@dataclass(frozen=True)
class ArtifactEntry:
    """One file inside a run folder."""

    relative_path: str
    path: Path
    size: int
    kind: str  # "text" | "json" | "image" | "binary"


@dataclass(frozen=True)
class ArtifactPreview:
    """Text preview of one artifact (``text`` is ``None`` for images/binaries)."""

    entry: ArtifactEntry
    text: str | None
    truncated: bool = False
    error: str | None = None


_TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".log",
    ".md",
    ".py",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in (".json", ".jsonl"):
        return "json"
    if suffix in _TEXT_SUFFIXES:
        return "text"
    return "binary"


def artifact_entries(run_dir: str | Path) -> list[ArtifactEntry]:
    """List every file under ``run_dir`` as a flat, sorted artifact table."""
    root = Path(run_dir)
    if not root.is_dir():
        raise ArtifactError(f"run directory not found: {root}")
    entries: list[ArtifactEntry] = []
    try:
        paths = sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    except OSError as exc:
        raise ArtifactError(f"cannot list artifacts in {root}: {exc}") from exc
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        entries.append(
            ArtifactEntry(
                relative_path=path.relative_to(root).as_posix(),
                path=path,
                size=size,
                kind=_artifact_kind(path),
            )
        )
    return entries


def artifact_preview(
    entry: ArtifactEntry, *, max_chars: int = 20_000
) -> ArtifactPreview:
    """Read a safe text preview of ``entry``.

    Binary files and images return ``text=None``.  Text is decoded with
    replacement characters and capped at ``max_chars``; JSON is re-indented
    when it parses cleanly.
    """
    if entry.kind in ("binary", "image"):
        return ArtifactPreview(entry=entry, text=None)
    try:
        # Read a bounded amount: UTF-8 uses at most four bytes per char.
        with entry.path.open("rb") as handle:
            raw = handle.read(max_chars * 4 + 1)
    except OSError as exc:
        return ArtifactPreview(entry=entry, text=None, error=str(exc))
    truncated = len(raw) > max_chars * 4
    text = raw[: max_chars * 4].decode("utf-8", errors="replace")
    if truncated or len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    if entry.kind == "json" and not truncated:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            text = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
    return ArtifactPreview(entry=entry, text=text, truncated=truncated)
