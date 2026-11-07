"""XOR page (T-104): train a 2-layer MLP on XOR from the browser and watch it learn.

The story this page tells
-------------------------

XOR ("exclusive or") is the smallest problem a straight line cannot
solve.  Its four points sit at the corners of a unit square with the two
diagonal corners labeled 1 and the other two labeled 0 -- a checkerboard.
A single linear layer (or a perceptron without a hidden layer) can only
draw *one straight line* through the input plane, and no single line can
carve out a checkerboard: the best such line always misclassifies at
least one point.  That is why every "how do neural nets work" course
starts here.

A **hidden layer** removes the limitation.  Each hidden unit computes
one half-plane (a linear threshold of the input, bent by the activation
function), and the output layer then draws a straight line *in the space
of those half-plane features*, where the four points are separable.  Two
hidden units already suffice (``|x1 - x2|`` is XOR on {0,1}^2); the
configurable ``hidden units`` control starts at the minimum workable 2
and the profiles use 8, which gives the optimizer slack to find a
solution robustly across random seeds.  The *activations* plot below
makes this visible: after a few epochs, individual units light up on
different half-planes, and the *decision boundary* plot shows the output
layer combining them into the XOR quadrants.

**Training** is the machinery that finds such a wiring.  The weights
start random, so the network's first answers are garbage and the loss
(the average "surprise" ``-log p(correct)`` of the true labels) sits
near its maximum ``ln(2) ~ 0.69``.  For each batch, backpropagation
computes, for every single weight, the direction in which nudging it
reduces the loss; the optimizer (AdamW) takes a small step there.
Repeated thousands of times, this is a walk downhill through the loss
landscape -- and the *loss curve* below is that walk, one point per
logged step.  The page's controls change where the walk starts and how
it proceeds:

- **hidden units** -- how many half-plane features the network may use;
- **activation** -- the bend between the two linear layers (tanh/relu);
- **learning rate** -- step size of the optimizer; too small and the
  curve flattens almost immediately (the network barely moves), too big
  and it oscillates;
- **epochs** -- how many full passes over the dataset;
- **noise std** -- Gaussian input jitter; a little noise makes the
  decision boundary smoother (regularization), zero keeps the toy crisp;
- **seed** -- the random initialization; different seeds take different
  downhill paths and settle on (equally valid) different solutions.

How the page is wired
---------------------

The page reuses the *exact same* pipeline as the CLI: the controls are
merged as overrides into the selected YAML config, re-validated (a bad
UI value fails with the same ``ConfigError`` as a bad file), and handed
to the shared ``JobRunner``/``Trainer`` (T-005/T-008).  The job runs in
a background thread and writes to its own artifact folder (T-004); a
0.5 s timer on this page drains the runner's status, log stream and
history into the labels, the log box and the three plots:

- while a run is in flight, the **loss curve** is *live* -- re-rendered
  from the runner's growing metric history into a scratch PNG (only when
  the history actually grew, so the timer does not re-render a static
  image every tick);
- when the run ends, the runner's ``on_end`` hook writes the final
  ``loss_curve.png`` plus the decision-boundary and activation plots
  into the run's ``plots/`` folder, and the page's plot slots switch to
  those files -- so what the page shows is exactly what is on disk.

Only one XOR job is active at a time; the page's dedicated runner
(:data:`xor_job_runner`) is separate from the app-wide one used by
``/jobs``, because it carries the XOR ``on_end`` plot hook and records
the per-step/per-epoch history the plots are built from.  That runner is
defined in :mod:`apps.ui.jobs` (not here) so it is never unloaded by
NiceGUI's per-test module cleanup and stays the same object across page
re-imports.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from nicegui import ui

from apps.ui.jobs import (
    FAILED,
    FINISHED,
    IDLE,
    RUNNING,
    STARTING,
    STOPPED,
    JobError,
    xor_job_runner,
)
from core.config import ConfigError, load_config
from core.trainer import TrainerError
from core.xor.model import ACTIVATIONS

__all__ = ["xor_job_runner"]

#: Repo root (``apps/ui/xor_page.py`` -> parents[2]), the base for the
#: relative config paths produced by ``_config_options()``.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Scratch location for the *live* (still-running) loss curve.  The final
#: curve is written into the run's ``plots/`` folder by the on_end hook;
#: while training is in flight we re-render this scratch PNG instead so
#: the page can show progress without touching the run folder until the
#: run has ended and marked itself finished.
LIVE_LOSS_PNG = Path(tempfile.gettempdir()) / "xor_page_live_loss_curve.png"

# Per-element bookkeeping for :func:`_show_image`:
# id(element) -> (str(path), mtime_ns, size) of the last source actually
# set or reloaded.  Keyed by ``id(img)`` because the page holds exactly
# one live image element per plot slot.
_IMAGE_STATE: dict[int, tuple[str, int, int]] = {}


def _show_image(img: ui.image, path: Path | None) -> None:
    """Point an (initially empty) ``ui.image`` at ``path`` without breaking the page.

    NiceGUI raises ``FileNotFoundError`` when an image's source file does
    not exist, so a page built *before* any run has happened cannot simply
    point its plot slots at the (not-yet-existing) run folder.  The slots
    therefore start empty, and this helper only touches the element once
    the file really exists:

    - first time, or a *different* file than before: ``set_source(path)``
      (NiceGUI serves it as a static file at a stable URL per path);
    - *same* file but the content changed (a newer PNG was written to the
      same path): ``force_reload()`` bumps the URL's cache-buster so the
      browser re-requests the file instead of keeping the cached image.

    Missing paths are ignored, so this can be called unconditionally every
    timer tick regardless of the job state.
    """
    if path is None or not path.is_file():
        return
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    prev = _IMAGE_STATE.get(id(img))
    if prev is None or prev[0] != str(path):
        img.set_source(path)
    elif key != prev:
        img.force_reload()
    _IMAGE_STATE[id(img)] = key


def _default_base_config(options: list[str]) -> str | None:
    """First XOR profile among the config options (fallback: first option).

    The page is about XOR, so it opens on an XOR profile if one exists
    in ``configs/``; a repo without any XOR profile falls back to the
    first config file of any phase.
    """
    for option in options:
        if Path(option).name.startswith("xor_"):
            return option
    return options[0] if options else None


def xor_page() -> None:
    """Build the XOR page: controls, train/stop, status, log, three plots.

    The layout top-to-bottom mirrors the story: *what you configure*
    (controls), *what happens while it runs* (status + log), *what the
    learning looks like* (loss curve, decision boundary, activations).
    """
    # Imported here, not at module top: ``main.register_pages()`` imports
    # this module, so a top-level import back into ``main`` would hit a
    # partially initialized module whenever ``apps.ui.xor_page`` is imported
    # before ``apps.ui.main``.  By the time a page is actually built,
    # ``main`` is fully loaded, and this import is cached afterwards.
    from apps.ui.main import _config_options, _format_status, _shell

    with _shell("/xor"):
        ui.row().classes("items-center")
        ui.badge("M1", outline=True)
        ui.label("XOR Learning").classes("text-h5")
        ui.label(
            "A 2-layer MLP learns the XOR function.  Type settings below, hit "
            "Train, and watch the loss curve fall while the decision boundary "
            "snaps onto the four corner points."
        ).classes("q-my-sm")
        ui.separator()

        # ------------------------------------------------------------------
        # Controls
        # ------------------------------------------------------------------
        # The selected YAML config supplies everything the controls do not
        # override (output dir, batch size, device, val split, ...).  When
        # the selection changes, the controls are *prefilled* with the file's
        # effective values (file + schema defaults), so the page always shows
        # what the next run will actually use.  A control left on its prefilled
        # value still becomes an explicit override -- to "keep the file's
        # value" you simply do not clear anything; to change one setting you
        # edit exactly that control.
        options = _config_options()
        base_config = ui.select(
            options=options,
            value=_default_base_config(options),
            label="Config file",
        ).classes("w-96")

        # Flex-wrap row: on narrow screens the controls stack instead of
        # overflowing (the page must stay usable on a phone).
        with ui.row().classes("w-full flex items-end gap-4 flex-wrap"):
            # How many hidden units the MLP gets (minimum workable: 2 --
            # one per half-plane of |x1 - x2|; more gives robustness).
            hidden_size_el = ui.number("hidden units", min=1, step=1).classes("w-32")
            # The non-linear bend between the two linear layers.  Without
            # it the network collapses to one linear map and cannot learn
            # XOR at all -- switching to relu and retraining is a nice demo.
            activation_el = ui.select(
                options=list(ACTIVATIONS), label="activation", value=None
            ).classes("w-32")
            # Optimizer step size (AdamW).  The single most impactful knob:
            # watch what 0.001 vs 0.05 vs 1.0 do to the live loss curve.
            learning_rate_el = ui.number(
                "learning rate", min=0.0, step=0.001
            ).classes("w-32")
            # Full passes over the dataset (each pass = repeats/4 steps).
            epochs_el = ui.number("epochs", min=1, step=1).classes("w-28")
            # Std of the Gaussian input jitter applied to the four points;
            # 0 keeps the toy crisp, >0 regularizes the boundary.
            noise_std_el = ui.number("noise std", min=0.0, step=0.01).classes("w-28")
            # Seed of the random weight initialization (and the dataset's
            # noise): the same settings with different seeds take different
            # downhill paths and land on different (equally valid) solutions.
            seed_el = ui.number("seed", step=1).classes("w-28")

        # Train/Stop + one-line status (state, step, epoch, elapsed, error)
        # and the artifact folder of the current/last run.
        with ui.row().classes("items-center q-my-sm"):
            train_btn = (
                ui.button("Train", icon="play_arrow")
                .props("color=positive")
                .mark("xor-train")
            )
            stop_btn = (
                ui.button("Stop", icon="stop").props("color=negative").mark("xor-stop")
            )
        status_label = ui.label("idle").classes("text-subtitle1").mark("xor-status")
        run_dir_label = ui.label("").classes("text-caption").mark("xor-run-dir")

        # Streamed run log (startup line, per-epoch loss lines, checkpoint
        # notifications, the final summary) -- the same text that goes to
        # the run folder's run.log on the CLI.
        log_box = ui.log(max_lines=500).classes("w-full h-64").mark("xor-log")

        # ------------------------------------------------------------------
        # Plots
        # ------------------------------------------------------------------
        # Three answers to three questions the loss number cannot:
        #  * loss curve  -- *is it learning?* (falling toward 0)
        #  * boundary    -- *where* does it draw the decision line?
        #  * activations -- *how* does it decompose XOR into half-planes?
        # All three are PNG files written to the run's plots/ folder; the
        # image elements below start empty and are filled by the timer
        # (see :func:`_show_image`) once the files exist.
        ui.label("Plots").classes("text-h6")
        with ui.grid(columns="repeat(auto-fit, minmax(280px, 1fr))").classes(
            "w-full gap-4"
        ):
            loss_img = ui.image().mark("xor-loss-plot")
            boundary_img = ui.image().mark("xor-boundary-plot")
            activations_img = ui.image().mark("xor-activations-plot")

        # ------------------------------------------------------------------
        # Behavior (closures over the elements above)
        # ------------------------------------------------------------------

        # How much of the runner's history has been rendered into the live
        # loss curve already.  The live curve is only re-rendered when this
        # count grows, so the 0.5 s timer renders at most once per new
        # metric record instead of re-saving a static figure every tick.
        last_live_len = 0

        def _prefill_controls(cfg_path: str) -> None:
            """Fill the training controls from the selected config file.

            ``load_config`` returns the *effective* config (file values
            plus schema defaults), so afterwards every control shows the
            value the next run will use.  A file that fails to load keeps
            the current control values and only flashes a warning -- the
            page must stay usable (you can still pick a different file).
            """
            path = Path(cfg_path)
            if not path.is_absolute():
                # Config options are relative to the repo root (see
                # ``_config_options``), independent of the server CWD.
                path = _REPO_ROOT / path
            try:
                cfg = load_config(path)
            except ConfigError as exc:
                ui.notify(f"could not load {path.name}: {exc}", type="warning")
                return
            pc = cfg.get("phase_config", {})
            for element, value in (
                (learning_rate_el, cfg.get("learning_rate")),
                (epochs_el, cfg.get("epochs")),
                (seed_el, cfg.get("seed")),
                (hidden_size_el, pc.get("hidden_size")),
                (activation_el, pc.get("activation")),
                (noise_std_el, pc.get("noise_std")),
            ):
                if value is not None:
                    element.value = value

        def _build_overrides() -> dict[str, Any]:
            """Collect the UI controls into a config-override dict.

            Blank controls (``None``) are *omitted*, not sent as ``None``:
            the schema validator rejects explicit ``None`` for every key
            with a default, and an empty box conceptually means "keep the
            config file's value".  The XOR-specific controls are grouped
            under the special ``phase_config`` key so the JobRunner
            deep-merges them into the file's phase section instead of
            wiping the file's other phase settings (e.g. ``repeats``).
            """
            phase: dict[str, Any] = {}
            raw = hidden_size_el.value
            if raw is not None:
                phase["hidden_size"] = int(raw)
            raw = activation_el.value
            if raw:
                phase["activation"] = str(raw)
            raw = noise_std_el.value
            if raw is not None:
                phase["noise_std"] = float(raw)
            overrides: dict[str, Any] = {}
            raw = learning_rate_el.value
            if raw is not None:
                overrides["learning_rate"] = float(raw)
            raw = epochs_el.value
            if raw is not None:
                overrides["epochs"] = int(raw)
            raw = seed_el.value
            if raw is not None:
                overrides["seed"] = int(raw)
            if phase:
                overrides["phase_config"] = phase
            return overrides

        def train_clicked() -> None:
            """Start a fresh XOR training run from the current controls.

            This is the heart of the page: the same one-liner a CLI user
            would run, but with the typed settings merged in.  Config
            errors (bad file, bad UI value, job already active) are
            reported as a notification; training failures later on show
            up in the status line (state ``failed`` + error).
            """
            nonlocal last_live_len
            config_path = base_config.value
            if not config_path:
                ui.notify("no config file selected", type="negative")
                return
            # The live curve starts fresh for every run: forget how much
            # history has been rendered and delete the scratch PNG from
            # the previous run, so this run's first timer tick re-renders
            # from zero.
            last_live_len = 0
            LIVE_LOSS_PNG.unlink(missing_ok=True)
            try:
                xor_job_runner.start(config_path, overrides=_build_overrides())
            except (JobError, ConfigError, TrainerError) as exc:
                ui.notify(str(exc), type="negative")

        def stop_clicked() -> None:
            """Ask the active run to stop after the current step.

            The Trainer still saves its final checkpoint and marks the
            run ``interrupted`` (see ``JobRunner.stop``), and the plots
            for the partial run are still written by the on_end hook.
            """
            try:
                xor_job_runner.stop()
            except JobError as exc:
                ui.notify(str(exc), type="negative")

        def _refresh_plots(status: dict[str, Any]) -> None:
            """Update the three plot slots for the current job state.

            While a run is in flight, the loss curve is *live*: it is
            re-rendered from the runner's growing history into the scratch
            PNG whenever new metric records arrived.  Once the run ends
            (finished/stopped/failed), the slots switch to the run
            folder's final plots -- written by the runner's ``on_end``
            hook (:func:`apps.ui.jobs._xor_page_on_end`), the
            same files a CLI run produces, so what the page shows is
            exactly what is on disk.  A failed run may have no plots yet
            (it crashed before on_end); the missing files are simply not
            shown.
            """
            nonlocal last_live_len
            state = status["state"]
            if state in (STARTING, RUNNING):
                history = xor_job_runner.history()
                if len(history) > last_live_len:
                    last_live_len = len(history)
                    try:
                        import matplotlib.pyplot as plt

                        from core.xor import plots  # optional viz extra

                        fig = plots.loss_curve_fig(history)
                        fig.savefig(LIVE_LOSS_PNG)
                        plt.close(fig)
                        _show_image(loss_img, LIVE_LOSS_PNG)
                    except Exception:
                        # matplotlib is an optional extra and the timer
                        # must never crash the UI event loop.
                        pass
                return
            if state not in (FINISHED, STOPPED, FAILED):
                return
            run_dir = status["run_dir"]
            if not run_dir:
                return
            plots_dir = Path(run_dir) / "plots"
            _show_image(loss_img, plots_dir / "loss_curve.png")
            _show_image(boundary_img, plots_dir / "decision_boundary.png")
            _show_image(activations_img, plots_dir / "activations.png")

        def refresh() -> None:
            """Timer tick: status line, run dir, button states, log, plots.

            Runs on the UI event loop every 0.5 s (and once immediately);
            all runner access goes through its thread-safe methods -- the
            job thread itself never touches any UI element.
            """
            status = xor_job_runner.status()
            status_label.text = _format_status(status)
            run_dir_label.text = (
                f"run: {status['run_dir']}" if status["run_dir"] else ""
            )
            # Train is only offered between runs; Stop only while one is
            # starting or running (a failed/finished job has nothing to
            # stop, and starting a second job is rejected by the runner).
            train_btn.set_enabled(status["state"] == IDLE)
            stop_btn.set_enabled(status["state"] in (STARTING, RUNNING))
            for line in xor_job_runner.drain_logs():
                log_box.push(line)
            _refresh_plots(status)

        # Wiring: the select's change handler must be attached *after* the
        # element exists (the handler list does not exist during
        # __init__), and the initial value is prefilled with one direct
        # call, because the handler only fires on *changes*.
        base_config.on_value_change(lambda e: _prefill_controls(e.value))
        if base_config.value:
            _prefill_controls(base_config.value)
        train_btn.on_click(train_clicked)
        stop_btn.on_click(stop_clicked)

        # Poll the job thread, then run one first tick so a page opened
        # right after a run (reconnect / refresh) immediately shows the
        # run's outputs instead of "idle".
        ui.timer(0.5, refresh)
        refresh()


def register_xor_page() -> None:
    """Register the ``/xor`` route (called from ``main.register_pages``).

    The route must be registered inside :func:`register_pages` (not at
    module import time) because the headless UI tests re-execute
    ``apps/ui/main.py`` per test *after* resetting the global NiceGUI
    app state, and this module is only imported once (Python caches it
    in ``sys.modules``).  Registering from ``register_pages`` means the
    route is re-created in every fresh app instance, exactly like the
    other routes in ``main.py``.
    """
    @ui.page("/xor", title="XOR Learning – llm-from-scratch")
    def xor_page_route() -> None:
        xor_page()
