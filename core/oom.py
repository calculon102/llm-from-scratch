"""OOM detection and actionable fallback hints (T-503).

Scaling runs fail in one characteristic way: the GPU runs out of memory.
A bare ``torch.OutOfMemoryError`` traceback is not actionable for a learner,
so this module turns the failure into a short list of next steps and the
Trainer checkpoints the run before re-raising:

1. halve the batch size (or use gradient accumulation later),
2. shorten the context window,
3. turn the AMP toggle on (bf16 on ROCm/CPU, fp16 on NVIDIA),
4. drop one rung on the profile ladder or shrink ``n_layer``/``n_embd``,
5. free other GPU processes,
6. resume the saved checkpoint with ``--resume``.

GPU memory is a *high-water mark*: an OOM can happen even when the current
allocation looks small, because a transient activation (e.g. attention
scores) peaked above the limit.  That is why the hints target the largest
transient (context length) as well as the persistent footprint (model
width/depth and optimizer state).

The same module also documents the interrupt path: Ctrl+C / SIGTERM and
the UI's Stop button all leave a resumable checkpoint via
:meth:`core.trainer.Trainer.train`, so a long run never loses progress.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from core.scaling import PROFILE_ORDER

__all__ = [
    "OOM_MARKERS",
    "TrainingOOMError",
    "fallback_suggestions",
    "format_oom_help",
    "is_oom_error",
    "next_smaller_profile",
]

#: Lower-case substrings seen in CUDA/ROCm allocator failures.
OOM_MARKERS: tuple[str, ...] = (
    "out of memory",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cuda error: out of memory",
    "hip out of memory",
    "insufficient memory",
    "failed to allocate",
)


class TrainingOOMError(RuntimeError):
    """Raised when training fails with an out-of-memory error.

    ``hints`` contains the numbered fallback steps (also embedded in the
    message), and ``checkpoint`` points at the checkpoint the Trainer saved
    before raising (``None`` when no run folder was attached).
    """

    def __init__(
        self,
        message: str,
        *,
        hints: Iterable[str] | None = None,
        checkpoint: Any = None,
    ) -> None:
        super().__init__(message)
        self.hints: list[str] = list(hints or [])
        self.checkpoint = checkpoint


def _oom_type_names() -> tuple[type[BaseException], ...]:
    """OOM exception classes available in this torch build."""
    types: list[type[BaseException]] = []
    for name in ("OutOfMemoryError",):
        candidate = getattr(torch, name, None)
        if isinstance(candidate, type):
            types.append(candidate)
    cuda_candidate = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if isinstance(cuda_candidate, type):
        types.append(cuda_candidate)
    return tuple(types)


def is_oom_error(exc: BaseException) -> bool:
    """Return true when ``exc`` is a CUDA/ROCm out-of-memory failure."""
    if isinstance(exc, _oom_type_names()):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in OOM_MARKERS)


def next_smaller_profile(profile: str | None) -> str | None:
    """Return the next cheaper rung on the ladder, or ``None`` at the bottom."""
    if profile not in PROFILE_ORDER:
        return None
    index = PROFILE_ORDER.index(profile)
    return PROFILE_ORDER[index - 1] if index > 0 else None


def _phase_value(cfg: dict, key: str) -> Any:
    phase_config = cfg.get("phase_config")
    if isinstance(phase_config, dict):
        return phase_config.get(key)
    return None


def fallback_suggestions(
    cfg: dict,
    *,
    checkpoint: Any = None,
    config_path: str | None = None,
) -> list[str]:
    """Build numbered fallback steps tailored to the failed config.

    ``cfg`` is a validated config mapping.  The suggestions are ordered from
    cheapest/most-targeted change to larger structural changes, and the
    final step is always a concrete resume command when a checkpoint exists.
    """
    suggestions: list[str] = []

    batch_size = cfg.get("batch_size")
    if isinstance(batch_size, int) and batch_size > 1:
        suggestions.append(
            f"Halve the batch size: batch_size: {batch_size} -> "
            f"{max(1, batch_size // 2)} (the fastest fix; re-check throughput)."
        )
    else:
        suggestions.append(
            "Keep batch_size at 1 and shorten the context window instead."
        )

    context = _phase_value(cfg, "context_length")
    if isinstance(context, int) and context > 8:
        suggestions.append(
            f"Halve phase_config.context_length: {context} -> {max(8, context // 2)} "
            "(attention activations grow quadratically with context)."
        )

    if not cfg.get("amp", False):
        suggestions.append(
            "Enable AMP: set amp: true (bf16 on ROCm/CPU, fp16 on NVIDIA) "
            "to roughly halve activation and model memory."
        )
    else:
        suggestions.append(
            "AMP is already on; move on to a smaller batch/context or profile."
        )

    profile = cfg.get("profile")
    smaller = next_smaller_profile(profile)
    if smaller is not None:
        suggestions.append(
            f"Drop one rung on the profile ladder: {profile} -> {smaller} "
            f"(set profile: {smaller} with profile_defaults: true), or shrink "
            "phase_config.n_layer / n_embd directly."
        )
    else:
        suggestions.append(
            f"Already at the smallest profile ({profile}); reduce n_layer/n_embd "
            "or batch_size instead."
        )

    if _phase_value(cfg, "n_embd") is not None:
        n_embd = _phase_value(cfg, "n_embd")
        n_layer = _phase_value(cfg, "n_layer")
        suggestions.append(
            f"Shrink the model: n_embd {n_embd} -> {max(32, int(n_embd) // 2)} "
            f"and/or n_layer {n_layer} -> {max(1, int(n_layer or 1) - 1)}."
        )

    suggestions.append(
        "Free VRAM before retrying: stop other GPU processes (or notebooks) "
        "and check `rocm-smi` / `nvidia-smi`; the Trainer calls "
        "torch.cuda.empty_cache() before saving the checkpoint."
    )

    if checkpoint:
        target = str(checkpoint)
        if config_path:
            resume = (
                f"python -m scripts.train --config {config_path} --resume {target}"
            )
        else:
            resume = f"python -m scripts.train --config <your config> --resume {target}"
        suggestions.append(f"Resume from the saved checkpoint: {resume}")
    return suggestions


def format_oom_help(
    exc: BaseException,
    cfg: dict,
    *,
    checkpoint: Any = None,
    config_path: str | None = None,
) -> str:
    """Format an OOM failure plus its numbered fallback suggestions."""
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    hints = fallback_suggestions(cfg, checkpoint=checkpoint, config_path=config_path)
    lines = [
        f"out of memory while training (profile={cfg.get('profile', '?')}): {first_line}",
        "Fallback suggestions:",
    ]
    lines.extend(f"  {i}. {hint}" for i, hint in enumerate(hints, start=1))
    if checkpoint:
        lines.append(f"A checkpoint was saved to: {checkpoint}")
    return "\n".join(lines)
