"""Device manager: ROCm/CPU detection + AMP toggle (T-006).

Resolves the config's ``device`` setting (``auto``/``cpu``/``cuda``),
detects the execution backend, decides the precision mode for the AMP
toggle, and reports device/VRAM info clearly at startup::

    from core.device import get_device_info, describe_device

    info = get_device_info(cfg)
    print(describe_device(info))
    # device=cuda(0) backend=rocm name='AMD Radeon RX 7900 XTX'
    # vram=24564 MiB total / 24312 MiB free precision=bf16 (AMP)

Backends
--------
``rocm`` -- torch built against HIP (``torch.version.hip`` set) with a GPU
           available.  ROCm shows up as ``cuda`` inside torch; this module
           distinguishes it so precision defaults can differ.
``cuda`` -- NVIDIA CUDA build with a GPU available.
``cpu``  -- CPU fallback (always available).

Precision
---------
- ``amp: false`` -> ``fp32`` (no autocast).
- ``amp: true``  -> ``bf16`` on ROCm and CPU, ``fp16`` on NVIDIA CUDA
  (ROCm prefers bf16; fp16 on CPU is poorly supported).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

__all__ = [
    "DeviceError",
    "DeviceInfo",
    "detect_backend",
    "resolve_device",
    "resolve_precision",
    "get_device_info",
    "describe_device",
    "log_device_startup",
]


class DeviceError(Exception):
    """Raised when the requested device is unavailable or invalid."""


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved execution target and precision mode for one run.

    Attributes
    ----------
    device:
        The ``torch.device`` to run on.
    backend:
        One of ``"rocm"``, ``"cuda"``, ``"cpu"``.
    name:
        Human-readable device name (e.g. ``"AMD Radeon RX 7900 XTX"``).
    vram_total_mb / vram_free_mb:
        VRAM in MiB, or ``None`` on CPU.
    amp:
        Whether the AMP toggle is on.
    precision:
        Human-readable precision label (``"fp32"``, ``"bf16 (AMP)"``,
        ``"fp16 (AMP)"``).
    amp_dtype:
        The autocast dtype, or ``None`` when AMP is off.
    """

    device: torch.device
    backend: str
    name: str
    vram_total_mb: float | None
    vram_free_mb: float | None
    amp: bool
    precision: str
    amp_dtype: torch.dtype | None


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    """Return the execution backend: ``"rocm"``, ``"cuda"``, or ``"cpu"``.

    ROCm is a HIP build of torch and reports through the CUDA API, so it
    is identified via ``torch.version.hip``.
    """
    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "rocm"
    return "cuda"


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(cfg: dict) -> torch.device:
    """Resolve the config's ``device`` setting (``auto``/``cpu``/``cuda``).

    ``auto`` picks the GPU when available (ROCm shows up as CUDA in torch)
    and falls back to CPU otherwise.  Requesting ``cuda`` when no GPU is
    available raises :class:`DeviceError` with an actionable message.
    """
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch_device = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise DeviceError(f"invalid device {device!r}: {exc}") from exc
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise DeviceError(
            "device='cuda' requested but no CUDA/ROCm device is available; "
            "use device: auto (falls back to CPU) or device: cpu"
        )
    return torch_device


# ---------------------------------------------------------------------------
# Precision (AMP toggle)
# ---------------------------------------------------------------------------

def resolve_precision(backend: str, amp: bool) -> tuple[str, torch.dtype | None]:
    """Return ``(precision_label, autocast_dtype)`` for a backend + AMP flag.

    AMP off is always fp32.  AMP on uses bf16 on ROCm and CPU (ROCm prefers
    bf16; fp16 on CPU is poorly supported) and fp16 on NVIDIA CUDA.
    """
    if not amp:
        return "fp32", None
    if backend == "cuda":
        return "fp16 (AMP)", torch.float16
    return "bf16 (AMP)", torch.bfloat16


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

def _device_name_and_vram(device: torch.device) -> tuple[str, float | None, float | None]:
    """Return ``(name, vram_total_mib, vram_free_mib)`` for a device."""
    if device.type != "cuda":
        return "CPU", None, None
    index = device.index if device.index is not None else 0
    props = torch.cuda.get_device_properties(index)
    free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
    return props.name, props.total_memory / 1024**2, free_bytes / 1024**2


def get_device_info(cfg: dict) -> DeviceInfo:
    """Resolve device, backend, and precision for a (validated) config.

    Raises :class:`DeviceError` when the requested device is unavailable.
    """
    device = resolve_device(cfg)
    backend = detect_backend() if device.type == "cuda" else "cpu"
    amp = bool(cfg.get("amp", False))
    precision, amp_dtype = resolve_precision(backend, amp)
    name, vram_total, vram_free = _device_name_and_vram(device)
    return DeviceInfo(
        device=device,
        backend=backend,
        name=name,
        vram_total_mb=vram_total,
        vram_free_mb=vram_free,
        amp=amp,
        precision=precision,
        amp_dtype=amp_dtype,
    )


def describe_device(info: DeviceInfo) -> str:
    """Format a :class:`DeviceInfo` as one clear startup line."""
    parts = [
        f"device={info.device}",
        f"backend={info.backend}",
        f"name={info.name!r}",
    ]
    if info.vram_total_mb is not None:
        parts.append(
            f"vram={info.vram_total_mb:.0f} MiB total / "
            f"{info.vram_free_mb:.0f} MiB free"
        )
    parts.append(f"precision={info.precision}")
    return " ".join(parts)


def log_device_startup(cfg: dict, logger: logging.Logger | None = None) -> DeviceInfo:
    """Log the resolved device/precision at startup and return the info.

    Uses ``logger`` when given, otherwise the ``llm_from_scratch.device``
    logger.  The run log (T-004) can be passed in so the line lands in
    ``run.log`` as well.
    """
    info = get_device_info(cfg)
    log = logger if logger is not None else logging.getLogger("llm_from_scratch.device")
    log.info("startup: %s", describe_device(info))
    return info
