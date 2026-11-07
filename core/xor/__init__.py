"""M1 – XOR learning module (T-101…T-105).

The XOR task is the classic "not linearly separable" toy problem: the
smallest problem a single linear layer provably cannot solve, and the
smallest problem that still exercises the full training machinery
(forward pass, loss, backprop, optimizer, checkpointing).  It is the
template phase for everything that follows (tokenizer, GPT).

This module registers the ``"xor"`` phase with the shared Trainer
(T-005)::

    register_phase("xor", build_xor_phase)

and its ``phase_config`` schema with the config loader (T-003)::

    register_phase_schema("xor", validate_xor_phase_config)

Phase config keys (all optional; defaults shown):

    phase_config:
      hidden_size: 8        # hidden units (minimum workable: 2)
      activation: tanh      # "tanh" or "relu"
      repeats: 100          # repeats of the 4 canonical points
      noise_std: 0.0        # Gaussian input noise std
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import register_phase_schema
from core.trainer import PhaseBundle, Trainer, register_phase, split_dataset
from core.xor.dataset import XORDataset
from core.xor.model import ACTIVATIONS, XorMLP

__all__ = [
    "XorMLP",
    "XORDataset",
    "xor_loss",
    "build_xor_phase",
    "validate_xor_phase_config",
]


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def xor_loss(
    model: nn.Module, batch: dict
) -> tuple[torch.Tensor, dict[str, float]]:
    """Binary cross-entropy on the logits, plus a training accuracy metric.

    The loss is the per-example "surprise" of the true label, averaged
    over the batch.  If ``p`` is the probability the network assigns to
    the true label (``sigmoid(logit)`` for y=1, ``1 - sigmoid(logit)``
    for y=0), the loss is::

        L = -log(p)

    so a model that is certain and *right* pays ~0, one that is certain
    and *wrong* pays a huge penalty, and one that is unsure (p = 0.5)
    pays a moderate, constant amount.  That asymmetry — being wrong
    costs far more than being unsure — is what pushes the network from
    "guessing 50/50" to confident, correct predictions; minimizing the
    average surprise over the data *is* "fitting the model".

    ``model(batch["x"])`` yields logits ``(B, 1)``; ``batch["y"]`` is the
    0/1 label ``(B, 1)``.  We use ``BCEWithLogitsLoss`` (the
    ``_with_logits`` variant) rather than a separate sigmoid + loss: it
    fuses the two into one numerically stable operation (the plain form
    can overflow/underflow for large |logit|, and the fused form's
    backward pass is slightly more accurate).

    The accuracy metric (fraction of examples answered correctly) is
    computed under ``torch.no_grad`` so it does not build a computation
    graph — it is only for logging, never for the gradient — and is
    reported as a float so the Trainer can log and epoch-average it.
    """
    logits = model(batch["x"])
    loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= 0.5).to(batch["y"].dtype)
        acc = (pred == batch["y"]).to(torch.float32).mean()
    return loss, {"acc": acc.item()}


# ---------------------------------------------------------------------------
# Phase config schema
# ---------------------------------------------------------------------------

#: phase_config schema: key -> spec (type / default / choices / min).
_XOR_PHASE_SCHEMA: dict[str, dict[str, Any]] = {
    "hidden_size": {"type": int, "default": 8, "min": 1},
    "activation": {"type": str, "default": "tanh", "choices": tuple(ACTIVATIONS)},
    "repeats": {"type": int, "default": 100, "min": 1},
    "noise_std": {"type": float, "default": 0.0, "min": 0.0},
}


def _xor_defaults() -> dict[str, Any]:
    """Default value for every phase_config key."""
    return {key: spec["default"] for key, spec in _XOR_PHASE_SCHEMA.items()}


def validate_xor_phase_config(phase_config: dict, source: str) -> list[str]:
    """Validate the ``phase_config`` section of an XOR config.

    Returns a list of error strings (empty = valid).  Unknown keys are
    rejected so typos surface at config load time, not mid-training.
    """
    errors: list[str] = []
    for key, value in phase_config.items():
        if key not in _XOR_PHASE_SCHEMA:
            errors.append(
                f"phase_config: unknown key '{key}' "
                f"(allowed: {', '.join(_XOR_PHASE_SCHEMA)})"
            )
            continue
        spec = _XOR_PHASE_SCHEMA[key]
        if spec["type"] is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif spec["type"] is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        else:
            ok = isinstance(value, spec["type"])
        if not ok:
            errors.append(
                f"phase_config.{key}: expected {spec['type'].__name__}, got {value!r}"
            )
            continue
        if "choices" in spec and value not in spec["choices"]:
            errors.append(
                f"phase_config.{key}: {value!r} is not one of {list(spec['choices'])}"
            )
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and "min" in spec
            and value < spec["min"]
        ):
            errors.append(f"phase_config.{key}: {value} is below minimum {spec['min']}")
    return errors


# ---------------------------------------------------------------------------
# Phase factory
# ---------------------------------------------------------------------------

def _xor_on_end(trainer: "Trainer", **kwargs: Any) -> None:
    """Save the decision-boundary and activation plots at the end of a run.

    Attached via ``PhaseBundle.callbacks`` so both the CLI (T-005) and the
    UI job runner (T-008) produce the plots without phase-specific code.
    """
    if trainer.run is None:
        return
    from core.xor import plots  # lazy: matplotlib is an optional viz extra

    plots.save_xor_plots(trainer.run, trainer.model)


def build_xor_phase(cfg: dict) -> PhaseBundle:
    """Build the XOR PhaseBundle (model + datasets + objective) from a
    validated config."""
    pc = cfg.get("phase_config", {})
    defaults = _xor_defaults()
    hidden_size = pc.get("hidden_size", defaults["hidden_size"])
    activation = pc.get("activation", defaults["activation"])
    repeats = pc.get("repeats", defaults["repeats"])
    noise_std = pc.get("noise_std", defaults["noise_std"])

    model = XorMLP(in_dim=2, hidden_size=hidden_size, activation=activation)
    dataset = XORDataset(repeats=repeats, noise_std=noise_std, seed=cfg["seed"])
    train_ds, val_ds = split_dataset(dataset, cfg["val_split"], seed=cfg["seed"])
    return PhaseBundle(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=xor_loss,
        callbacks={"on_end": _xor_on_end},
    )


# Registration happens at import time: entry points (scripts/train.py,
# apps/ui/jobs.py) import this module before building any phase.
register_phase("xor", build_xor_phase)
register_phase_schema("xor", validate_xor_phase_config)
