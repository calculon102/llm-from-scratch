"""Shared Trainer loop: train / eval / checkpoint / resume (T-005).

One phase-agnostic training loop reused by *all* training phases (XOR,
skip-gram, GPT, ...).  Phases plug in by registering a factory that builds
a :class:`PhaseBundle` (model + datasets + objective); the Trainer owns
the loop, logging, checkpointing, resuming, and evaluation::

    from core.trainer import Trainer, register_phase, PhaseBundle

    def build_xor(cfg):
        return PhaseBundle(model=..., train_dataset=..., val_dataset=...,
                           loss_fn=lambda model, batch: (loss, {"acc": ...}))

    register_phase("xor", build_xor)

Batches are dicts of tensors (e.g. ``{"x": ..., "y": ...}`` or
``{"input_ids": ...}``); the first tensor's ``shape[0]`` is treated as the
batch size.  ``loss_fn(model, batch)`` must return ``(loss, metrics)``
where ``metrics`` maps names to floats; if ``loss_fn`` is ``None`` the
model's forward output is used directly (a bare loss tensor, or a
``(loss, metrics)`` tuple).

The training cycle, step by step
--------------------------------

For each *batch* (a small sample of the dataset, e.g. 32 examples) the
Trainer does four things — this is the whole of "training", for XOR and
for GPT alike::

    1. forward pass    evaluate the model on the batch at its current
                       weights; compute the LOSS: one number measuring how
                       wrong the predictions are (lower = better; a
                       perfectly fitting model has loss ~ 0).
    2. backward pass   loss.backward() asks autograd for the GRADIENT:
                       for every trainable weight, how much the loss would
                       change if that weight were nudged.  (You can verify
                       these numbers any time with the finite-difference
                       gradient check, T-103: Trainer.check_gradients.)
    3. optimizer step  move every weight a little bit opposite to its
                       gradient,  W -= lr * grad  (AdamW additionally
                       adapts the step size per weight over time).  The
                       learning rate lr controls how far each step goes:
                       too large overshoots the minimum, too small crawls.
    4. repeat          over many steps the weights drift downhill through
                       the loss landscape.  An EPOCH is one full pass over
                       the training data; the validation set — which is
                       never used for weight updates — tells us whether the
                       model is actually improving or just memorizing.

After enough steps on a small task like XOR, the loss reaches ~ 0: the
weights have settled in a region where the network answers correctly on
all training points.

Every run goes through the artifact manager (T-004): checkpoints are
written to ``checkpoints/`` as ``step_*.pt`` (every ``save_interval``
steps) plus a ``final.pt`` at the end.  A checkpoint stores model weights,
optimizer state, step/epoch counters, best val loss, and the RNG state, so
:meth:`Trainer.resume` continues exactly where a previous run stopped
(resuming at the start of the next epoch).
"""

from __future__ import annotations

import contextlib
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from core.artifacts import RunArtifacts
from core.config import PHASES, validate_config
from core.device import detect_backend, resolve_device, resolve_precision
from core.gradcheck import DEFAULT_EPS, DEFAULT_RTOL, GradientCheckReport, format_report, gradient_check

__all__ = [
    "TrainerError",
    "PhaseBundle",
    "Trainer",
    "register_phase",
    "build_phase",
    "seed_everything",
    "resolve_device",
    "split_dataset",
]


class TrainerError(Exception):
    """Raised for missing/unregistered phases or invalid training setups."""


# ---------------------------------------------------------------------------
# Seed / device helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
    dataset: Dataset, val_split: float, seed: int = 0
) -> tuple[Dataset, Dataset | None]:
    """Split ``dataset`` into (train, val) using ``val_split`` of the data.

    Returns ``(dataset, None)`` when the split would leave no validation
    samples.  The split is deterministic for a given ``seed``.
    """
    n = len(dataset)
    n_val = int(round(n * val_split))
    if n_val <= 0:
        return dataset, None
    if n_val >= n:
        raise TrainerError(
            f"val_split={val_split} leaves no training samples (n={n})"
        )
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n - n_val, n_val], generator=gen)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Phase registry
# ---------------------------------------------------------------------------

@dataclass
class PhaseBundle:
    """Everything the Trainer needs for one phase, built by a phase factory.

    Attributes
    ----------
    model:
        The ``nn.Module`` to train.
    train_dataset / val_dataset:
        ``torch.utils.data.Dataset`` instances yielding dicts of tensors.
    loss_fn:
        ``loss_fn(model, batch) -> (loss, metrics)``; ``None`` means the
        model's forward output is the loss (or a ``(loss, metrics)`` tuple).
    metric_fn:
        Optional ``metric_fn(model, val_loader) -> dict[str, float]`` that
        replaces the Trainer's default evaluation (e.g. perplexity for GPT).
    optimizer_factory:
        Optional ``optimizer_factory(model) -> Optimizer``; defaults to
        AdamW with the config's ``learning_rate``.
    callbacks:
        Optional Trainer callbacks (``on_start``, ``on_step``,
        ``on_epoch_end``, ``on_end``) attached by the phase itself, e.g.
        XOR (T-102) saves its plots in ``on_end``.  Merged with the
        callbacks passed to :meth:`Trainer.from_bundle` (caller wins on
        hook-name conflicts).
    """

    model: nn.Module
    train_dataset: Dataset
    val_dataset: Dataset | None = None
    loss_fn: Callable[[nn.Module, dict], tuple[torch.Tensor, dict[str, float]]] | None = None
    metric_fn: Callable[[nn.Module, DataLoader], dict[str, float]] | None = None
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None
    callbacks: dict[str, Callable] | None = None


_PHASE_FACTORIES: dict[str, Callable[[dict], PhaseBundle]] = {}


def register_phase(name: str, factory: Callable[[dict], PhaseBundle]) -> None:
    """Register the phase factory for ``name`` (one of :data:`PHASES`).

    Phase modules call this at import time, e.g. XOR (T-101) registers
    ``"xor"``.  ``factory(cfg)`` receives the validated config and returns
    a :class:`PhaseBundle`.
    """
    if name not in PHASES:
        raise TrainerError(f"unknown phase {name!r}; expected one of {list(PHASES)}")
    _PHASE_FACTORIES[name] = factory


def build_phase(cfg: dict) -> PhaseBundle:
    """Build the :class:`PhaseBundle` for ``cfg["phase"]``.

    Raises :class:`TrainerError` when no phase is set or the phase has no
    registered factory.
    """
    phase = cfg.get("phase")
    if phase is None:
        raise TrainerError(
            "config has no 'phase' set; expected one of: " + ", ".join(PHASES)
        )
    factory = _PHASE_FACTORIES.get(phase)
    if factory is None:
        registered = ", ".join(sorted(_PHASE_FACTORIES)) or "(none)"
        raise TrainerError(
            f"phase {phase!r} has no registered factory (registered: {registered})"
        )
    return factory(cfg)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def _batch_size(batch: dict) -> int:
    """Infer the batch size from the first tensor in the batch dict."""
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return value.shape[0]
    return 1


class Trainer:
    """Phase-agnostic training loop shared by all training phases.

    Parameters
    ----------
    model, optimizer:
        The module and its optimizer.
    train_dataset, val_dataset:
        Datasets yielding dicts of tensors; ``val_dataset`` may be ``None``.
    config:
        A validated config mapping (defaults are filled when omitted).
    run:
        Optional :class:`~core.artifacts.RunArtifacts`; when given, logs go
        to the run log and checkpoints are written to the run folder.
    device, amp:
        Override the config's ``device`` / ``amp`` settings.
    batch_size:
        Override the config's ``batch_size``.
    loss_fn, metric_fn, callbacks, grad_clip:
        See :class:`PhaseBundle`; ``callbacks`` maps hook names
        (``on_start``, ``on_step``, ``on_epoch_end``, ``on_end``) to
        callables invoked as ``callback(trainer, **kwargs)``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        *,
        config: dict | None = None,
        run: RunArtifacts | None = None,
        device: torch.device | None = None,
        amp: bool | None = None,
        batch_size: int | None = None,
        loss_fn: Callable[[nn.Module, dict], tuple[torch.Tensor, dict[str, float]]] | None = None,
        metric_fn: Callable[[nn.Module, DataLoader], dict[str, float]] | None = None,
        callbacks: dict[str, Callable] | None = None,
        grad_clip: float = 1.0,
    ) -> None:
        self.config = config if config is not None else validate_config({})
        self.run = run
        self.device = device if device is not None else resolve_device(self.config)
        self.amp = bool(self.config["amp"]) if amp is None else bool(amp)
        backend = detect_backend() if self.device.type == "cuda" else "cpu"
        self.precision, self._amp_dtype = resolve_precision(backend, self.amp)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.grad_clip = grad_clip
        self.callbacks = callbacks or {}

        bs = batch_size if batch_size is not None else self.config["batch_size"]
        self.train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=True, num_workers=0
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0)
            if val_dataset is not None
            else None
        )

        self.start_step = 0
        self._last_step = 0
        self._epoch = 0  # completed epochs
        self.best_val: float | None = None
        self._stop = False
        # GradScaler is only needed for fp16 autocast (NVIDIA CUDA); bf16
        # (ROCm/CPU) has fp32's exponent range and scales without one.
        self._scaler = (
            torch.amp.GradScaler("cuda", dtype=torch.float16)
            if self.amp
            and self.device.type == "cuda"
            and self._amp_dtype == torch.float16
            else None
        )
        self._fallback_logger = logging.getLogger("llm_from_scratch.trainer")
        if not self._fallback_logger.handlers:
            self._fallback_logger.setLevel(logging.INFO)
            self._fallback_logger.addHandler(logging.StreamHandler())

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------

    @classmethod
    def from_bundle(
        cls,
        bundle: PhaseBundle,
        *,
        config: dict | None = None,
        run: RunArtifacts | None = None,
        device: torch.device | None = None,
        amp: bool | None = None,
        callbacks: dict[str, Callable] | None = None,
        grad_clip: float = 1.0,
    ) -> "Trainer":
        """Build a Trainer from a :class:`PhaseBundle` (default optimizer:
        AdamW with the config's ``learning_rate``)."""
        cfg = config if config is not None else validate_config({})
        optimizer = (
            bundle.optimizer_factory(bundle.model)
            if bundle.optimizer_factory is not None
            else torch.optim.AdamW(bundle.model.parameters(), lr=cfg["learning_rate"])
        )
        # Phase-attached callbacks first; caller-supplied ones win on
        # hook-name conflicts.
        merged = {**(bundle.callbacks or {}), **(callbacks or {})}
        return cls(
            bundle.model,
            optimizer,
            bundle.train_dataset,
            bundle.val_dataset,
            config=cfg,
            run=run,
            device=device,
            amp=amp,
            loss_fn=bundle.loss_fn,
            metric_fn=bundle.metric_fn,
            callbacks=merged,
            grad_clip=grad_clip,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str, level: int = logging.INFO) -> None:
        if self.run is not None:
            self.run.log(message, level=level)
        else:
            self._fallback_logger.log(level, message)

    def _fire(self, hook: str, **kwargs: Any) -> None:
        callback = self.callbacks.get(hook)
        if callback is not None:
            callback(self, **kwargs)

    # ------------------------------------------------------------------
    # Forward / step
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _autocast(self):
        if not self.amp:
            yield
            return
        # dtype resolved by the device manager (T-006): bf16 on ROCm/CPU,
        # fp16 on NVIDIA CUDA.
        with torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype):
            yield

    def _forward(
        self, model: nn.Module, batch: dict
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.loss_fn is not None:
            loss, metrics = self.loss_fn(model, batch)
        else:
            out = model(batch)
            if isinstance(out, tuple):
                loss, metrics = out[0], dict(out[1])
            else:
                loss, metrics = out, {}
        return loss, metrics

    def _train_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        # One training step = the cycle in the module docstring:
        # forward + loss, backward, optimizer update.
        self.model.train()  # training mode: enables dropout/BatchNorm semantics
        self.optimizer.zero_grad()  # grads accumulate — clear last step's first
        with self._autocast():
            loss, metrics = self._forward(self.model, batch)  # 1. forward pass + loss
        if self._scaler is not None:
            self._scaler.scale(loss).backward()  # 2. backprop (scaled, fp16 path)
            if self.grad_clip > 0:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()  # 2. backprop: fills param.grad for every weight
            if self.grad_clip > 0:
                # Safety net: cap the total gradient norm so one pathological
                # batch cannot send the weights flying (matters for noisy data).
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()  # 3. W -= lr * grad (AdamW: adaptive per weight)
        return loss.detach(), metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, float]:
        """Evaluate on the val set (or the train set when no val is given).

        Returns sample-weighted metrics: ``val_loss`` (or ``train_loss``)
        plus any metrics reported by the forward pass.  A custom
        ``metric_fn`` replaces this entirely and must return a dict of
        floats.
        """
        if self.metric_fn is not None:
            loader = self.val_loader if self.val_loader is not None else self.train_loader
            return self.metric_fn(self.model, loader)

        loader = self.val_loader if self.val_loader is not None else self.train_loader
        key = "val_loss" if self.val_loader is not None else "train_loss"
        self.model.eval()
        total = 0.0
        n_total = 0
        acc: dict[str, float] = {}
        with torch.no_grad(), self._autocast():
            for batch in loader:
                loss, metrics = self._forward(self.model, batch)
                n = _batch_size(batch)
                total += loss.item() * n
                n_total += n
                for name, value in metrics.items():
                    acc[name] = acc.get(name, 0.0) + value * n
        self.model.train()
        result = {key: total / max(n_total, 1)}
        for name, value in acc.items():
            result[name] = value / max(n_total, 1)
        return result

    # ------------------------------------------------------------------
    # Gradient check (T-103)
    # ------------------------------------------------------------------

    def check_gradients(
        self, eps: float = DEFAULT_EPS, rtol: float = DEFAULT_RTOL
    ) -> GradientCheckReport:
        """Verify the model's gradients against finite differences (T-103).

        Compares the analytic gradients from ``loss.backward()`` (what the
        optimizer actually uses) with central finite-difference estimates of
        the same derivatives on one training batch, and builds a per-
        parameter mismatch report — see :mod:`core.gradcheck` for the math
        and for how to read a FAIL.

        The check runs on a deep-copied CPU float32 copy of the model, so
        the real model (device, dtype, weights) is untouched.  When the Trainer
        has a run attached, the report is written to
        ``<run_dir>/gradcheck_report.txt`` and the verdict is logged
        (WARNING on FAIL — this is a diagnostic, not a gate: training
        continues, so you can still see what wrong gradients do to learning).

        Returns the :class:`~core.gradcheck.GradientCheckReport`.
        """
        batch = next(iter(self.train_loader))
        report = gradient_check(self.model, self.loss_fn, batch, eps=eps, rtol=rtol)
        if self.run is not None:
            report_path = self.run.run_dir / "gradcheck_report.txt"
            report_path.write_text(format_report(report, model_name=type(self.model).__name__))
            self._log(f"gradient check report written to {report_path.name}")
        status = "PASS" if report.passed else "FAIL"
        level = logging.INFO if report.passed else logging.WARNING
        self._log(
            f"gradient check: {status} (params={report.num_params}, "
            f"elements={report.total_elements}, "
            f"max_rel_err={report.max_rel_err:.3e}, eps={report.eps:g}, rtol={report.rtol:g})"
        )
        if not report.passed:
            self._log(
                f"gradient check: worst parameter is {report.worst_param!r} — "
                "inspect the backward path that flows through it",
                level=level,
            )
        return report

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int | None = None) -> Path | None:
        """Save model + optimizer + progress state.

        With ``step`` the file is ``checkpoints/step_<step:06d>.pt``;
        without it, ``checkpoints/final.pt``.  Returns the path, or ``None``
        when the Trainer has no run attached.
        """
        if self.run is None:
            return None
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._last_step if step is None else step,
            "epoch": self._epoch,
            "best_val": self.best_val,
            "rng": torch.get_rng_state(),
        }
        return self.run.save_checkpoint(state, step=step)

    def resume(self, path: str | Path | None = None) -> None:
        """Restore model, optimizer, and progress from a checkpoint.

        ``path`` may be a checkpoint file or a previous run directory (the
        newest ``step_*.pt`` is used, else ``final.pt``).  With
        ``path=None`` the latest checkpoint of this Trainer's run is used.
        Training continues at the start of the epoch that was in progress
        when the checkpoint was saved, so batches already processed in that
        epoch may be re-processed.
        """
        if path is None:
            if self.run is None:
                raise TrainerError("cannot resume: no run and no checkpoint path given")
            path = self.run.latest_checkpoint()
        path = Path(path)
        if path.is_dir():
            steps = sorted(path.glob("checkpoints/step_*.pt"))
            final = path / "checkpoints" / "final.pt"
            path = steps[-1] if steps else (final if final.is_file() else None)
            if path is None:
                raise TrainerError(f"no checkpoints found in run dir {path}")
        if not path.is_file():
            raise TrainerError(f"checkpoint not found: {path}")

        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.start_step = int(state["step"])
        self._last_step = self.start_step
        self._epoch = int(state.get("epoch", 0))
        self.best_val = state.get("best_val")
        if "rng" in state:
            torch.set_rng_state(state["rng"])
        self._log(f"resumed from {path} (step={self.start_step}, epoch={self._epoch})")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the training loop to stop after the current step (used by the
        UI job runner, T-008)."""
        self._stop = True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the training loop and return a summary dict.

        Summary keys: ``steps`` (optimizer steps taken), ``epochs``
        (completed epochs), ``best_val`` (best val loss, or ``None``),
        ``elapsed_s``.
        """
        cfg = self.config
        epochs = cfg["epochs"]
        max_steps = cfg["max_steps"]
        log_interval = cfg["log_interval"]
        save_interval = cfg["save_interval"]

        self._log(
            f"training started: phase={cfg.get('phase') or '?'} "
            f"profile={cfg['profile']} device={self.device} "
            f"precision={self.precision} amp={self.amp} "
            f"seed={cfg['seed']} epochs={epochs} batch_size={cfg['batch_size']} "
            f"lr={cfg['learning_rate']}"
            + (f" max_steps={max_steps}" if max_steps is not None else "")
            + (f" (resuming at step {self.start_step})" if self.start_step else "")
        )
        self._fire("on_start")

        start = time.time()
        step = self.start_step
        window: list[float] = []
        while self._epoch < epochs and not self._stop:
            for batch in self.train_loader:
                if self._stop or (max_steps is not None and step >= max_steps):
                    break
                loss, metrics = self._train_step(batch)
                step += 1
                self._last_step = step
                window.append(loss.item())
                if len(window) > log_interval:
                    window.pop(0)
                if step % log_interval == 0:
                    avg = sum(window) / len(window)
                    lr = self.optimizer.param_groups[0]["lr"]
                    line = (
                        f"step {step} | epoch {self._epoch + 1}/{epochs} | "
                        f"loss {avg:.4f} | lr {lr:.6g}"
                    )
                    for name, value in metrics.items():
                        line += f" | {name} {value:.4f}"
                    self._log(line)
                    self._fire("on_step", step=step, loss=avg, metrics=metrics,
                              epoch=self._epoch)
                if save_interval and step % save_interval == 0:
                    self.save_checkpoint(step)
            self._epoch += 1
            if self.val_loader is not None:
                val = self.evaluate()
                vloss = val.get("val_loss")
                if vloss is not None and (self.best_val is None or vloss < self.best_val):
                    self.best_val = vloss
                self._log(
                    f"epoch {self._epoch}/{epochs} done | "
                    + " | ".join(f"{k} {v:.4f}" for k, v in val.items())
                )
            else:
                self._log(f"epoch {self._epoch}/{epochs} done")
            self._fire("on_epoch_end", epoch=self._epoch, step=step)
            if max_steps is not None and step >= max_steps:
                break

        self.save_checkpoint()
        elapsed = time.time() - start
        summary = {
            "steps": step,
            "epochs": self._epoch,
            "best_val": self.best_val,
            "elapsed_s": elapsed,
        }
        self._log(
            f"training finished: steps={step} epochs={self._epoch} "
            f"best_val={self.best_val if self.best_val is not None else 'n/a'} "
            f"elapsed={elapsed:.1f}s"
        )
        self._fire("on_end", summary=summary)
        return summary
