"""UI job runner: start / stop / status / log streaming (T-008).

Wraps the shared Trainer (T-005) so long training jobs can be controlled
from the web UI:

- :meth:`JobRunner.start` validates the YAML config, builds the phase,
  opens a run folder (T-004), and runs ``Trainer.train()`` in a background
  thread, so the UI event loop stays responsive.
- :meth:`JobRunner.stop` asks the Trainer to stop after the current step
  (``Trainer.request_stop``); the final checkpoint is still saved and the
  run is marked ``interrupted``.
- :meth:`JobRunner.status` returns a small thread-safe dict for UI
  polling: state, phase, run name/dir, step, epoch, elapsed time, error.
- Log lines written to the run log (T-004) are mirrored into a
  thread-safe queue that the UI page drains into a ``ui.log`` element on
  a timer.

Only one job is active at a time.  The runner is phase-agnostic: phases
register their factories with ``core.trainer.register_phase`` (T-005) and
the job runs whatever ``cfg["phase"]`` resolves to.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import core.xor  # noqa: F401  # registers the "xor" phase (T-101)
from core.artifacts import RunArtifacts, create_run
from core.config import load_config
from core.device import describe_device, get_device_info
from core.trainer import (
    PhaseBundle,
    Trainer,
    build_phase,
    seed_everything,
)

__all__ = [
    "IDLE",
    "STARTING",
    "RUNNING",
    "STOPPING",
    "FINISHED",
    "STOPPED",
    "FAILED",
    "JobError",
    "JobRunner",
    "job_runner",
]

#: Job states.
IDLE = "idle"
STARTING = "starting"
RUNNING = "running"
STOPPING = "stopping"
FINISHED = "finished"
STOPPED = "stopped"
FAILED = "failed"

#: States in which the runner is occupied by a job.
_ACTIVE_STATES = frozenset({STARTING, RUNNING, STOPPING})


class JobError(Exception):
    """Raised for invalid job operations (job already active, nothing to stop)."""


class _QueueLogHandler(logging.Handler):
    """Logging handler that pushes formatted lines into a queue."""

    def __init__(self, out: "queue.Queue[str]") -> None:
        super().__init__()
        self._out = out

    def emit(self, record: logging.LogRecord) -> None:
        self._out.put(self.format(record))


class JobRunner:
    """Runs one training job at a time, controllable from the UI.

    The job runs in a daemon thread; the UI reads state through
    :meth:`status` and :meth:`drain_logs`, which are thread-safe.  UI
    elements must only be touched from the UI event loop (the page's
    timer), never from the job thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._trainer: Trainer | None = None
        self._run: RunArtifacts | None = None
        self._log_handler: _QueueLogHandler | None = None
        self._cfg: dict | None = None
        self._bundle: PhaseBundle | None = None
        self._state = IDLE
        self._phase: str | None = None
        self._run_name: str | None = None
        self._run_dir: str | None = None
        self._step: int | None = None
        self._epoch: int | None = None
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._error: str | None = None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self, config_path: str | Path) -> None:
        """Start a training job for the YAML config at ``config_path``.

        Config loading and phase building happen synchronously so bad
        configs fail fast (``ConfigError`` / ``TrainerError``); the
        training itself runs in a background thread, and failures there
        are reported via :meth:`status` (state ``failed`` + ``error``).
        Raises :class:`JobError` when a job is already active.
        """
        with self._lock:
            if self._state in _ACTIVE_STATES:
                raise JobError(f"a job is already active (state={self._state})")
            cfg = load_config(config_path)
            bundle = build_phase(cfg)
            self._cfg = cfg
            self._bundle = bundle
            self._cancel.clear()
            self._log_queue = queue.Queue()
            self._trainer = None
            self._run = None
            self._log_handler = None
            self._phase = cfg.get("phase")
            self._run_name = cfg.get("run_name")
            self._run_dir = None
            self._step = None
            self._epoch = None
            self._error = None
            self._start_time = time.time()
            self._end_time = None
            self._state = STARTING
            thread = threading.Thread(
                target=self._run_job, name="job-runner", daemon=True
            )
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        """Ask the active job to stop after the current step.

        Raises :class:`JobError` when no job is active.
        """
        with self._lock:
            if self._state not in _ACTIVE_STATES:
                raise JobError(f"no active job to stop (state={self._state})")
            self._cancel.set()
            self._state = STOPPING
            trainer = self._trainer
        if trainer is not None:
            trainer.request_stop()

    def reset(self) -> None:
        """Clear the last job's state (only when no job is active)."""
        with self._lock:
            if self._state in _ACTIVE_STATES:
                raise JobError(f"cannot reset while a job is active (state={self._state})")
            self._cfg = None
            self._bundle = None
            self._trainer = None
            self._run = None
            self._log_handler = None
            self._log_queue = queue.Queue()
            self._phase = None
            self._run_name = None
            self._run_dir = None
            self._step = None
            self._epoch = None
            self._start_time = None
            self._end_time = None
            self._error = None
            self._state = IDLE

    # ------------------------------------------------------------------
    # Status / log stream
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Thread-safe snapshot of the current or last job (for UI polling)."""
        with self._lock:
            elapsed = None
            if self._start_time is not None:
                end = self._end_time if self._end_time is not None else time.time()
                elapsed = end - self._start_time
            return {
                "state": self._state,
                "phase": self._phase,
                "run_name": self._run_name,
                "run_dir": self._run_dir,
                "step": self._step,
                "epoch": self._epoch,
                "elapsed_s": elapsed,
                "error": self._error,
            }

    def drain_logs(self) -> list[str]:
        """Pop all queued run-log lines (for the UI log stream)."""
        lines: list[str] = []
        while True:
            try:
                lines.append(self._log_queue.get_nowait())
            except queue.Empty:
                return lines

    # ------------------------------------------------------------------
    # Job thread
    # ------------------------------------------------------------------

    def _run_job(self) -> None:
        cfg = self._cfg
        bundle = self._bundle
        assert cfg is not None and bundle is not None
        run: RunArtifacts | None = None
        try:
            seed_everything(cfg["seed"])
            run = create_run(cfg)
            handler = _QueueLogHandler(self._log_queue)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
                )
            )
            run.logger.addHandler(handler)
            with self._lock:
                self._run = run
                self._log_handler = handler
                self._run_dir = str(run.run_dir)
            run.log(f"job started: phase={cfg['phase']} profile={cfg['profile']}")
            info = get_device_info(cfg)
            run.log(f"startup: {describe_device(info)}")
            trainer = Trainer.from_bundle(
                bundle,
                config=cfg,
                run=run,
                device=info.device,
                amp=cfg["amp"],
                callbacks={
                    "on_step": self._on_step,
                    "on_epoch_end": self._on_epoch_end,
                },
            )
            with self._lock:
                self._trainer = trainer
                if self._state == STARTING:
                    self._state = RUNNING
            if self._cancel.is_set():
                run.mark_finished("interrupted")
                self._finish(STOPPED)
                return
            trainer.train()
            stopped = self._cancel.is_set()
            run.mark_finished("interrupted" if stopped else "finished")
            self._finish(STOPPED if stopped else FINISHED)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            if run is not None:
                run.mark_finished("failed")
            self._finish(FAILED)
        finally:
            self._detach_log_handler(run)

    def _finish(self, state: str) -> None:
        with self._lock:
            self._state = state
            self._end_time = time.time()

    def _detach_log_handler(self, run: RunArtifacts | None) -> None:
        with self._lock:
            handler = self._log_handler
            self._log_handler = None
        if handler is not None and run is not None:
            run.logger.removeHandler(handler)

    # ------------------------------------------------------------------
    # Trainer callbacks (run in the job thread)
    # ------------------------------------------------------------------

    def _on_step(self, trainer: Trainer, *, step: int, epoch: int, **kwargs: Any) -> None:
        with self._lock:
            self._step = step
            self._epoch = epoch

    def _on_epoch_end(self, trainer: Trainer, *, epoch: int, **kwargs: Any) -> None:
        with self._lock:
            self._epoch = epoch


#: The app-wide job runner used by the ``/jobs`` page.
job_runner = JobRunner()
