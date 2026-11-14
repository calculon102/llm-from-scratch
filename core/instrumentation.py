"""Throughput and memory instrumentation for training runs (T-502).

The scaling milestone needs more than a loss curve: to compare profiles we
must know **how fast** a run processes tokens and **how much memory** it
uses.  This module provides the small, dependency-free pieces the shared
Trainer records at every logged step:

* :class:`ThroughputMeter` -- tokens and optimizer steps per wall-clock
  second.  It can be paused around evaluation/checkpointing so validation
  time is not charged to training throughput.
* :func:`memory_stats` -- a :class:`MemoryStats` snapshot with the process
  peak RSS (CPU) and, on CUDA/ROCm, allocated/reserved/peak/total VRAM.
* :func:`format_memory` -- one compact, log-friendly line.

Why peak RSS rather than current RSS?  Current RSS drops when Python frees
objects, which hides the real high-water mark of a run.  ``ru_maxrss`` is
monotonic and is the number an OOM post-mortem cares about.  On the GPU,
``max_memory_allocated`` is the equivalent high-water mark; it is what to
compare against the 24 GB budget of the 7900 XTX.

Example log line::

    step 250 | epoch 1/8 | loss 3.8120 | lr 0.0002 | tok/s 18432 | mem cpu=912MiB gpu=6.1/7.4GiB peak=6.3GiB/23.9GiB
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "MemoryStats",
    "ThroughputMeter",
    "format_memory",
    "memory_stats",
]


def _bytes_to_mib(value: float) -> float:
    return float(value) / 1024**2


@dataclass(frozen=True)
class MemoryStats:
    """Memory high-water marks for one training step / run.

    CPU values are always available.  GPU values are ``None`` on CPU-only
    runs (and on a build without CUDA/ROCm support).
    """

    cpu_peak_mb: float | None
    gpu_allocated_mb: float | None = None
    gpu_reserved_mb: float | None = None
    gpu_peak_mb: float | None = None
    gpu_total_mb: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Flat mapping for ``metrics.jsonl`` / ``run_meta.json``."""
        return {
            "cpu_peak_mb": self.cpu_peak_mb,
            "gpu_allocated_mb": self.gpu_allocated_mb,
            "gpu_reserved_mb": self.gpu_reserved_mb,
            "gpu_peak_mb": self.gpu_peak_mb,
            "gpu_total_mb": self.gpu_total_mb,
        }


def _cpu_peak_mb() -> float | None:
    """Process peak RSS in MiB.

    ``resource`` is POSIX-only; on Windows the value is unavailable rather
    than fatal.  ``ru_maxrss`` is KiB on Linux and bytes on macOS.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return _bytes_to_mib(peak)
    return peak / 1024.0


def memory_stats(device: torch.device | str | None = None) -> MemoryStats:
    """Snapshot CPU peak RSS and (when available) GPU memory counters."""
    cpu = _cpu_peak_mb()
    if isinstance(device, str):
        device = torch.device(device)

    gpu_device: torch.device | None = None
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        gpu_device = device
    elif device is None and torch.cuda.is_available():
        gpu_device = torch.device("cuda", torch.cuda.current_device())

    if gpu_device is None:
        return MemoryStats(cpu_peak_mb=cpu)

    index = gpu_device.index
    if index is None:
        index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        total = _bytes_to_mib(total_bytes)
    except (RuntimeError, AttributeError):  # pragma: no cover - old/odd builds
        total = _bytes_to_mib(props.total_memory)
    return MemoryStats(
        cpu_peak_mb=cpu,
        gpu_allocated_mb=_bytes_to_mib(torch.cuda.memory_allocated(index)),
        gpu_reserved_mb=_bytes_to_mib(torch.cuda.memory_reserved(index)),
        gpu_peak_mb=_bytes_to_mib(torch.cuda.max_memory_allocated(index)),
        gpu_total_mb=total,
    )


def _format_mib(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1024:
        return f"{value / 1024:.1f}GiB"
    return f"{value:.0f}MiB"


def format_memory(stats: MemoryStats) -> str:
    """Format a :class:`MemoryStats` as one compact log fragment."""
    text = f"cpu={_format_mib(stats.cpu_peak_mb)}"
    if stats.gpu_peak_mb is not None:
        text += (
            f" gpu={_format_mib(stats.gpu_allocated_mb)}/"
            f"{_format_mib(stats.gpu_reserved_mb)}"
            f" peak={_format_mib(stats.gpu_peak_mb)}"
            f"/{_format_mib(stats.gpu_total_mb)}"
        )
    return text


class ThroughputMeter:
    """Wall-clock tokens/steps counter with pause/resume support.

    The meter is *active* from construction until :meth:`pause` is called.
    Pausing accumulates the active interval, so wrapping evaluation or
    checkpointing in ``pause``/``resume`` reports the throughput of
    training work only.  All times are monotonic and cheap to read.
    """

    def __init__(self) -> None:
        self.tokens: int = 0
        self.steps: int = 0
        self._elapsed_s: float = 0.0
        self._active_since: float | None = time.perf_counter()

    def reset(self) -> None:
        """Zero the counters and restart the clock."""
        self.tokens = 0
        self.steps = 0
        self._elapsed_s = 0.0
        self._active_since = time.perf_counter()

    def update(self, tokens: int = 0) -> None:
        """Record one completed optimizer step and its token count."""
        self.tokens += max(int(tokens), 0)
        self.steps += 1

    def pause(self) -> None:
        """Stop the clock (idempotent)."""
        if self._active_since is not None:
            self._elapsed_s += time.perf_counter() - self._active_since
            self._active_since = None

    def resume(self) -> None:
        """Restart the clock after :meth:`pause` (idempotent)."""
        if self._active_since is None:
            self._active_since = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        """Active (non-paused) elapsed seconds."""
        active = (
            time.perf_counter() - self._active_since
            if self._active_since is not None
            else 0.0
        )
        return self._elapsed_s + active

    @property
    def tokens_per_sec(self) -> float:
        elapsed = self.elapsed_s
        return self.tokens / elapsed if elapsed > 0 else 0.0

    @property
    def steps_per_sec(self) -> float:
        elapsed = self.elapsed_s
        return self.steps / elapsed if elapsed > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Summary fields for logs, metrics files and benchmark reports."""
        return {
            "tokens": self.tokens,
            "steps": self.steps,
            "elapsed_s": self.elapsed_s,
            "tokens_per_sec": self.tokens_per_sec,
            "steps_per_sec": self.steps_per_sec,
        }
