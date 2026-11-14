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
- :meth:`JobRunner.start` accepts optional ``overrides`` (T-104): config
  values typed into the UI that replace the YAML file's values.  The
  special key ``phase_config`` is merged into the file's section (so one
  UI control does not wipe the file's other phase settings), and the
  merged config is re-validated — bad UI values fail fast with the same
  ``ConfigError`` as a bad file.
- A ``JobRunner`` may be constructed with extra Trainer callbacks
  (T-104), e.g. the XOR page's ``on_end`` plot hook.  The runner's own
  ``on_step`` / ``on_epoch_end`` hooks win on name conflicts, so the
  training history below is never clobbered by a page callback.
- :meth:`JobRunner.history` returns the per-step and per-epoch metrics
  recorded from the Trainer callbacks — the raw material for the live
  and final loss curves on the XOR page (T-104).

The XOR page's dedicated runner (:data:`xor_job_runner`, T-104) lives in
this module rather than in ``apps.ui.xor_page``: NiceGUI's test harness
removes page-route modules from ``sys.modules`` between tests, so a
singleton defined in the page module would silently be re-created, while
this module stays loaded and the runner stays one object for the page and
its tests.

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
from typing import Any, Callable

import core.xor  # noqa: F401  # registers the "xor" phase (T-101)
import core.embeddings  # noqa: F401  # registers the "embeddings" phase (T-302)
from core.artifacts import RunArtifacts, create_run
from core.config import load_config, validate_config
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

    def __init__(
        self,
        callbacks: dict[str, Callable[..., None]] | None = None,
    ) -> None:
        """Create a job runner.

        ``callbacks`` (T-104) are extra Trainer callbacks merged in with
        every job, e.g. the XOR page's ``on_end`` hook that saves the
        decision-boundary and activation plots.  The runner's own
        ``on_step`` / ``on_epoch_end`` hooks are merged in *after* these,
        so the training history below is never clobbered by a page
        callback registered under the same hook name.
        """
        self._lock = threading.Lock()
        # Extra Trainer callbacks supplied by the page (see class docstring).
        self._extra_callbacks: dict[str, Callable[..., None]] = dict(callbacks or {})
        # Per-step / per-epoch metric records, appended by the callbacks
        # below while the job thread runs; read via :meth:`history`.
        self._history: list[dict[str, Any]] = []
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

    def start(
        self,
        config_path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        """Start a training job for the YAML config at ``config_path``.

        ``overrides`` (T-104) are config values typed into the UI that
        replace the file's values *before* the config is validated and
        the phase is built, so a bad UI value fails fast with the same
        ``ConfigError`` as a bad file.  The special key ``phase_config``
        is deep-merged into the file's section (one UI control must not
        wipe the file's other phase settings).  ``None`` values are
        ignored: a blank UI control means "keep the config file's value".
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
            self._apply_overrides(cfg, overrides)
            # Re-validate after the UI edits: the file alone was valid,
            # but a typed value (e.g. activation="relu-sigmoid") may not be.
            validate_config(cfg)
            bundle = build_phase(cfg)
            # Fresh history for this job (see :meth:`history`).
            self._history = []
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
            self._history = []
            self._state = IDLE

    @staticmethod
    def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> None:
        """Apply UI-typed ``overrides`` to ``cfg`` in place (T-104).

        Top-level keys (``epochs``, ``learning_rate``, ``seed``, ...) are
        simply overwritten.  The special key ``phase_config`` is merged
        into the file's ``phase_config`` section key by key, so the
        remaining phase settings from the file survive.  ``None`` entries
        are skipped — a blank UI control means "use the file's value",
        and ``None`` would not pass schema validation anyway.
        """
        if not overrides:
            return
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "phase_config":
                # value is a dict of phase-specific keys (hidden_size,
                # activation, noise_std, ...), merged into the section.
                section = cfg.setdefault("phase_config", {})
                for sub_key, sub_value in value.items():
                    if sub_value is not None:
                        section[sub_key] = sub_value
            else:
                cfg[key] = value

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

    def history(self) -> list[dict[str, Any]]:
        """Return a copy of the metric history (T-104, for UI plots).

        Each record is one of:

        - ``{"step", "epoch", "loss", ...}`` — appended by the runner's
          ``on_step`` hook (once per logged step); the extra keys are the
          Trainer's per-step metrics (e.g. ``acc``).
        - ``{"epoch", "step", ...}`` — appended by the ``on_epoch_end``
          hook (once per epoch); the extra keys are the epoch evaluation
          (e.g. ``val_loss`` / ``val_acc`` when a validation split exists,
          otherwise ``train_loss`` / ...).

        The records are the raw material for the XOR page's live and
        final loss curves (:mod:`core.xor.plots`).
        """
        with self._lock:
            return list(self._history)

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
            # Page-supplied callbacks first (e.g. the XOR page's on_end
            # plot hook); the runner's own hooks win on name conflicts so
            # the history recorded below always sees every step and epoch.
            trainer = Trainer.from_bundle(
                bundle,
                config=cfg,
                run=run,
                device=info.device,
                amp=cfg["amp"],
                callbacks={
                    **self._extra_callbacks,
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
            # Optional finite-difference gradient check (T-103), before any
            # training step (see scripts/train.py).
            if cfg["gradcheck"]:
                trainer.check_gradients()
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

    def _on_step(
        self,
        trainer: Trainer,
        *,
        step: int,
        epoch: int,
        loss: float | None = None,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the running step counter and one history row (T-104).

        Runs in the job thread.  ``loss`` is the Trainer's averaged
        per-step training loss over the last ``log_interval`` steps, and
        ``metrics`` holds the forward pass's extra metrics (e.g. ``acc``).
        """
        with self._lock:
            self._step = step
            self._epoch = epoch
            record: dict[str, Any] = {"step": step, "epoch": epoch}
            if loss is not None:
                record["loss"] = loss
            if metrics:
                record.update(metrics)
            self._history.append(record)

    def _on_epoch_end(
        self,
        trainer: Trainer,
        *,
        epoch: int,
        step: int | None = None,
        val: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the epoch counter and one history row (T-104).

        ``val`` (T-104) is the Trainer's per-epoch evaluation, e.g.
        ``{"val_loss", "val_acc"}`` with a validation split — the
        per-epoch curve behind the final loss plot.
        """
        with self._lock:
            self._epoch = epoch
            record = {"epoch": epoch}
            if step is not None:
                record["step"] = step
            if val:
                record.update(val)
            self._history.append(record)


#: The app-wide job runner used by the ``/jobs`` page.
job_runner = JobRunner()


def _xor_page_on_end(trainer: Trainer, **kwargs: Any) -> None:
    """Save XOR plots at the end of a UI XOR run (called by xor_job_runner).

    Defined here rather than in apps.ui.xor_page so that xor_job_runner can
    live in this module, which is NOT removed from sys.modules between NiceGUI
    tests (only page-route modules are purged), making the runner a stable
    singleton accessible to both page code and test code via a persistent import.
    Plotting errors are caught so a matplotlib bug never fails a successful run.
    """
    if trainer.run is None:
        return
    try:
        from core.xor import plots  # lazy: optional viz extra

        plots.save_xor_plots(trainer.run, trainer.model)
        plots.save_loss_curve(trainer.run, xor_job_runner.history())
    except Exception as exc:
        trainer.run.log(f"plotting failed: {exc}", level=logging.WARNING)


#: Dedicated job runner for the XOR page (T-104).  Lives here (not in
#: apps.ui.xor_page) so it is never removed from sys.modules by NiceGUI's
#: test teardown and stays the same object across per-test app re-runs.
xor_job_runner = JobRunner(callbacks={"on_end": _xor_page_on_end})
