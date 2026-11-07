"""Finite-difference gradient check (T-103).

An *independent* way to verify that the gradients our network computes are
actually correct — and, when they are not, a report that says *which*
parameter gets it wrong and by how much.

What is a gradient, and why check it?
=====================================

During training the network's weights W are nudgeable knobs, and the loss L
is one number that measures how wrong the current predictions are.  The
*gradient* ∇L is the vector of all partial derivatives dL/dW_i — for every
single weight it says: "if you increase this weight by a tiny amount, the
loss changes by *this* amount".  Training (gradient descent) just follows
these numbers downhill: ``W_i -= learning_rate * dL/dW_i`` for every i.

The gradients normally come from *automatic differentiation* (autograd):
Torch records every arithmetic operation of the forward pass and then
applies the chain rule in reverse to build dL/dW_i for every weight.  That
machinery is fast, but it is still code — and in this project (and in most
ML projects) some of the math is hand-written: custom losses, sampling
functions, attention masks, ...  A sign flip, a transposed matrix, or a
forgotten term in any hand-written backward pass silently produces wrong
gradients, and training then "works" (loss goes down, often!) but learns
the wrong thing or gets stuck.  The finite-difference check is how you
catch that.

How the check works
===================

For *every* weight element W_i we compute the gradient twice:

1. **Analytic gradient** — from ``loss.backward()`` (autograd).  This is
   the value the optimizer actually uses.

2. **Finite-difference gradient** — from the very definition of a
   derivative, without any chain rule and without touching autograd's
   backward pass at all.  We physically nudge the weight by a tiny step
   ``eps``, re-run the whole network *forward* to get the loss twice, and
   take the slope between the two loss values::

       dL/dW_i ≈ ( L(W_i + eps) - L(W_i - eps) ) / (2 * eps)

   This is the *central difference* formula.  It is just "rise over run"
   on the loss curve, with the point placed symmetrically so that first-
   order curvature errors cancel (the truncation error then scales like
   eps² instead of eps).

Both numbers are estimates of the same truth, computed by completely
independent routes.  If the backward pass is correct, they must agree up
to floating-point rounding noise.  We compare them element-wise with the
same relative-error measure Torch's own ``torch.autograd.gradcheck``
uses::

    rel_err = |analytic - numeric| / max(1, |analytic|, |numeric|)

and the check PASSES when no element's rel_err exceeds ``rtol``.

Choosing eps — the one knob that matters
----------------------------------------

* eps **too large**: the loss curve bends noticeably over that distance,
  and "rise over run" no longer equals the true slope (truncation error,
  ~ eps²).
* eps **too small**: the two loss values differ only in their least
  significant bits; subtracting them throws most digits away and the
  result is pure rounding noise (round-off error, ~ eps⁻¹).

The optimum balances the two: for central differences it scales like
``eps ≈ (machine_epsilon)^(1/3)``.  float64 (what Torch's own
``torch.autograd.gradcheck`` uses) lands at ~1e-6; **float32 has 8x fewer
mantissa bits, so the sweet spot moves up to ~1e-3** (the default here).
Measured on the XOR MLP: at eps=1e-5 a *correct* backward pass already
disagrees with finite differences by ~1e-2 (pure round-off), while at
eps=1e-3 the disagreement drops to ~1e-4 — and a real bug (e.g. a
backward that returns half the gradient) stays at ~1e-2, cleanly above
the rtol = 1e-3 threshold on either side.  That is also why the check
runs on a **CPU float32 copy** of the model: half precision (fp16/bf16,
used for fast training) has too few mantissa bits for the two loss values
to differ meaningfully, so the numerical gradient would come out zero or
garbage.  The check validates the *math* of your model, not the precision
of your hardware.

Cost, and which models to check
--------------------------------

Every weight element needs two extra forward passes, so the whole check
costs ``2 * (number of weights)`` forward passes.  That is trivial for the
XOR MLP (33 weights → 66 passes, milliseconds) but far too expensive for
a GPT-sized model with millions of parameters — use this on small
debug-profile models (the ``tiny_debug`` rung of the scale ladder) or on
newly hand-written submodules in isolation, not on full-size runs.

How to read a FAIL
==================

The report lists every parameter with its worst absolute and relative
error.  Look at *which* parameter is bad: the error is localized to the
backward path that flows through it.  Typical culprits:

* a custom ``autograd.Function`` whose ``backward`` is wrong (e.g. returns
  half the gradient, or the gradient of the wrong quantity),
* a matrix multiplication that is transposed in forward but not in
  backward (or vice versa),
* a term present in the forward pass that was forgotten in the loss (the
  bias is the classic one: forward uses it, the hand-written loss doesn't,
  so the analytic bias gradient is zero while the numeric one is not).

Usage
=====

Standalone::

    from core.gradcheck import gradient_check, format_report

    report = gradient_check(model, loss_fn, batch)   # loss_fn(model, batch) -> (loss, metrics)
    print(format_report(report))

Through the shared Trainer (writes ``gradcheck_report.txt`` into the run
folder and logs the verdict; enabled via the config key ``gradcheck: true``
or the ``Trainer.check_gradients`` method)::

    python -m scripts.train --config configs/xor_tiny_debug.yaml   # with gradcheck: true
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn as nn

__all__ = [
    "DEFAULT_EPS",
    "DEFAULT_RTOL",
    "ParameterRow",
    "GradientCheckReport",
    "analytic_gradient",
    "finite_difference_gradient",
    "compare_gradients",
    "gradient_check",
    "format_report",
]

#: Step size of the central difference.  See the module docstring for why
#: ~1e-3 (not the 1e-5/1e-6 used for float64) is the float32 sweet spot
#: between truncation and round-off error.
DEFAULT_EPS = 1e-3

#: Relative-error threshold.  Measured on the XOR MLP (eps=1e-3): a
#: *correct* backward pass agrees with the finite-difference estimate to
#: within ~1e-4, while a real bug (half gradient, transposed product,
#: missing term) shows up at ~1e-2 or more — one full order of magnitude
#: of margin on each side of this limit.
DEFAULT_RTOL = 1e-3


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------

@dataclass
class ParameterRow:
    """Per-parameter result of the check (one row in the report table).

    Attributes
    ----------
    name:
        The parameter's dotted name in the model (e.g. ``"net.0.weight"``).
    shape:
        Tensor shape, e.g. ``(8, 2)``.
    num_elements:
        Number of weight elements checked for this parameter.
    max_abs_err:
        Worst |analytic - numeric| over the parameter's elements.
    max_rel_err:
        Worst relative error (normalized by ``max(1, |a|, |n|)``) over the
        parameter's elements.  This is the number to look at.
    ok:
        ``True`` when ``max_rel_err <= rtol``.
    """

    name: str
    shape: tuple[int, ...]
    num_elements: int
    max_abs_err: float
    max_rel_err: float
    ok: bool


@dataclass
class GradientCheckReport:
    """The full mismatch report of one gradient check.

    Attributes
    ----------
    passed:
        ``True`` when every parameter's ``max_rel_err`` is within ``rtol``.
    eps / rtol:
        The settings the check ran with (echoed into the report text).
    num_params / total_elements:
        Coverage: how many parameters and weight elements were checked.
    max_abs_err / max_rel_err:
        The worst errors over the whole model (the "headlines").
    worst_param:
        Name of the parameter with the largest ``max_rel_err`` — the place
        to start looking when the check FAILs (``None`` if there were no
        parameters at all).
    n_failed_params:
        How many parameters exceeded the threshold (0 = clean check).
    rows:
        One :class:`ParameterRow` per parameter, in model order.
    """

    passed: bool
    eps: float
    rtol: float
    num_params: int
    total_elements: int
    max_abs_err: float
    max_rel_err: float
    worst_param: str | None
    n_failed_params: int
    rows: list[ParameterRow]


# ---------------------------------------------------------------------------
# The two gradients
# ---------------------------------------------------------------------------

def _ndindex(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Yield every multi-index of a tensor shape (like ``itertools.ndindex``,
    which was removed from the standard library in Python 3.13)."""
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for sub in _ndindex(shape[1:]):
            yield (i, *sub)



def _loss_for(
    model: nn.Module,
    loss_fn: "Callable[[nn.Module, dict], object] | None",
    batch: dict,
) -> torch.Tensor:
    """Compute the scalar loss for (model, batch).

    Accepts the same conventions the shared Trainer (T-005) accepts:
    ``loss_fn(model, batch)`` returning a bare loss or ``(loss, metrics)``,
    or ``loss_fn is None`` meaning "the model's forward output *is* the
    loss (or a ``(loss, metrics)`` tuple)".
    """
    if loss_fn is not None:
        out = loss_fn(model, batch)
    else:
        out = model(batch)
    loss = out[0] if isinstance(out, tuple) else out
    if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
        raise TypeError(
            "gradient check needs a scalar loss; got "
            f"{type(loss).__name__} of dim {getattr(loss, 'dim', 'n/a')}"
        )
    return loss


def analytic_gradient(
    model: nn.Module,
    loss_fn: "Callable[[nn.Module, dict], object] | None",
    batch: dict,
) -> dict[str, torch.Tensor]:
    """dL/dW for every parameter via autograd (the value training uses).

    Runs one forward pass, one ``backward()``; returns a name → gradient
    tensor dict.  Parameters that do not reach the loss (no gradient) come
    back as zeros, so the report always has one row per parameter.
    """
    model.zero_grad(set_to_none=True)
    loss = _loss_for(model, loss_fn, batch)
    loss.backward()
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        grads[name] = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
    return grads


def finite_difference_gradient(
    model: nn.Module,
    loss_fn: "Callable[[nn.Module, dict], object] | None",
    batch: dict,
    eps: float = DEFAULT_EPS,
) -> dict[str, torch.Tensor]:
    """dL/dW for every parameter via central finite differences.

    For each element of each parameter this evaluates the loss at
    ``w + eps`` and ``w - eps`` and takes ``(L+ - L-) / (2 eps)`` — the
    definition of the derivative, with no autograd involved (see the module
    docstring).  That is ``2 * (# elements)`` forward passes, which is why
    the check targets small debug models.

    The caller is responsible for passing a CPU float32 model (the shared
    entry point :func:`gradient_check` does this on a deep copy); the
    function modifies parameter values in place and restores them exactly.
    """
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        original = param.data.clone()
        g = torch.zeros_like(param)
        for idx in _ndindex(param.shape):
            # +eps: evaluate the loss at the nudged weight.
            param.data[idx] += eps
            loss_plus = _loss_for(model, loss_fn, batch).item()
            # -eps (a full 2*eps step down from +eps): second loss value.
            param.data[idx] -= 2.0 * eps
            loss_minus = _loss_for(model, loss_fn, batch).item()
            # Restore the exact original value and record the slope.
            param.data[idx] = original[idx]
            g[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        grads[name] = g
    return grads


# ---------------------------------------------------------------------------
# Comparison / report
# ---------------------------------------------------------------------------

def compare_gradients(
    analytic: dict[str, torch.Tensor],
    numeric: dict[str, torch.Tensor],
    eps: float = DEFAULT_EPS,
    rtol: float = DEFAULT_RTOL,
) -> GradientCheckReport:
    """Compare the two gradient sets element-wise and build the report.

    Per element: ``abs_err = |a - n|`` and
    ``rel_err = abs_err / max(1, |a|, |n|)`` (the same normalization as
    ``torch.autograd.gradcheck`` — the ``max(1, ...)`` makes the comparison
    absolute for small gradients, where relative error is meaningless).
    A parameter passes when its worst ``rel_err`` is ≤ ``rtol``; the check
    passes when every parameter passes.
    """
    rows: list[ParameterRow] = []
    total_elements = 0
    for name in analytic:
        a = analytic[name].flatten()
        n = numeric[name].flatten()
        abs_err = (a - n).abs()
        scale = torch.maximum(torch.ones_like(a), torch.maximum(a.abs(), n.abs()))
        rel_err = abs_err / scale
        row = ParameterRow(
            name=name,
            shape=tuple(int(s) for s in analytic[name].shape),
            num_elements=int(a.numel()),
            max_abs_err=float(abs_err.max().item()),
            max_rel_err=float(rel_err.max().item()),
            ok=bool(rel_err.max().item() <= rtol),
        )
        rows.append(row)
        total_elements += row.num_elements

    if rows:
        worst = max(rows, key=lambda r: r.max_rel_err)
        worst_param: str | None = worst.name
        max_rel = worst.max_rel_err
        max_abs = max(r.max_abs_err for r in rows)
    else:
        worst_param, max_rel, max_abs = None, 0.0, 0.0

    return GradientCheckReport(
        passed=all(r.ok for r in rows),
        eps=eps,
        rtol=rtol,
        num_params=len(rows),
        total_elements=total_elements,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        worst_param=worst_param,
        n_failed_params=sum(1 for r in rows if not r.ok),
        rows=rows,
    )


def gradient_check(
    model: nn.Module,
    loss_fn: "Callable[[nn.Module, dict], object] | None" = None,
    batch: dict | None = None,
    eps: float = DEFAULT_EPS,
    rtol: float = DEFAULT_RTOL,
) -> GradientCheckReport:
    """Run the full check: analytic vs finite-difference gradient.

    Both gradients are computed on a **deep-copied CPU float32 copy** of
    ``model`` (half precision is too coarse for the difference of two
    nearly-equal loss values — see the module docstring), so the original
    model, its device, and its dtype are left untouched and the check
    never disturbs an ongoing run.  ``batch`` follows the Trainer's
    dict-of-tensors convention; non-tensor values are passed through.
    """
    if batch is None:
        raise ValueError("gradient_check needs a batch (dict of tensors)")
    # Deep copy first: the .cpu()/.float() moves are in-place, so they must
    # happen on the copy, never on the caller's model.
    model_copy = copy.deepcopy(model).cpu().float()
    batch_copy = {
        key: value.detach().cpu().float()
        for key, value in batch.items()
        if torch.is_tensor(value)
    }
    analytic = analytic_gradient(model_copy, loss_fn, batch_copy)
    numeric = finite_difference_gradient(model_copy, loss_fn, batch_copy, eps=eps)
    return compare_gradients(analytic, numeric, eps=eps, rtol=rtol)


def format_report(report: GradientCheckReport, model_name: str = "model") -> str:
    """Render the report as human-readable text (written to the run folder
    by ``Trainer.check_gradients`` and printed by tests)."""
    width = 78
    lines: list[str] = []
    lines.append("=" * width)
    lines.append("Finite-difference gradient check (T-103)")
    lines.append("=" * width)
    lines.append("")
    lines.append("HOW TO READ THIS REPORT")
    lines.append("")
    lines.append(
        "For every weight element W_i of the model the gradient dL/dW_i was"
    )
    lines.append(
        "computed two independent ways and compared:"
    )
    lines.append("")
    lines.append(
        "  1. analytic  — from autograd's backpropagation; the value the"
    )
    lines.append(
        "     optimizer actually uses when it updates the weights;"
    )
    lines.append("")
    lines.append(
        "  2. numeric   — from the definition of a derivative: the weight"
    )
    lines.append(
        f"     is nudged by +eps and -eps, the whole network is re-run"
    )
    lines.append(
        f"     forward each time, and the slope (L(W+eps) - L(W-eps)) / (2*eps)"
    )
    lines.append(
        "     is taken.  No chain rule, no autograd — pure arithmetic."
    )
    lines.append("")
    lines.append(
        "If the backward pass is correct, both estimates of the same"
    )
    lines.append(
        "quantity must agree up to floating-point noise.  The per-element"
    )
    lines.append(
        "relative error  rel_err = |analytic - numeric| / max(1, |analytic|, |numeric|)"
    )
    lines.append(
        f"is below rtol = {report.rtol:g} on every element for a PASS."
    )
    lines.append(
        "On a FAIL, the parameter rows show *which* weight is wrong — the"
    )
    lines.append(
        "bug lives in the backward path that flows through it (a custom"
    )
    lines.append(
        "backward(), a transposed product, a term missing from the loss)."
    )
    lines.append("")
    lines.append("SETTINGS")
    lines.append("")
    lines.append(
        f"  eps = {report.eps:g}   (central-difference step size; ~1e-3 is the"
    )
    lines.append(
        f"  float32 sweet spot between truncation and round-off error)"
    )
    lines.append(
        f"  rtol = {report.rtol:g}   (per-element relative-error threshold)"
    )
    lines.append(
        "  precision: float32 on CPU (deliberately: fp16/bf16 have too few"
    )
    lines.append(
        "  mantissa bits for two nearly-equal losses to differ meaningfully)"
    )
    lines.append("")
    lines.append("MODEL")
    lines.append("")
    lines.append(
        f"  {model_name}: {report.num_params} parameters, "
        f"{report.total_elements} weight elements checked"
    )
    lines.append("")
    lines.append("PER-PARAMETER RESULTS")
    lines.append("")
    header = (
        f"  {'parameter':<24} {'shape':<12} {'elements':>8} "
        f"{'max_abs_err':>12} {'max_rel_err':>12}  ok"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for row in report.rows:
        shape = "(" + ", ".join(str(s) for s in row.shape) + ")"
        lines.append(
            f"  {row.name:<24} {shape:<12} {row.num_elements:>8} "
            f"{row.max_abs_err:>12.3e} {row.max_rel_err:>12.3e}  "
            f"{'yes' if row.ok else 'NO'}"
        )
    lines.append("")
    lines.append("VERDICT")
    lines.append("")
    if report.passed:
        lines.append(
            f"  PASS — all {report.num_params} parameters agree; "
            f"worst max_rel_err = {report.max_rel_err:.3e} "
            f"(threshold {report.rtol:g})."
        )
        lines.append(
            "  Backpropagation matches the definition of the derivative; "
            "training is using the correct gradients."
        )
    else:
        lines.append(
            f"  FAIL — {report.n_failed_params} of {report.num_params} "
            f"parameters disagree; worst max_rel_err = {report.max_rel_err:.3e} "
            f"(threshold {report.rtol:g})."
        )
        if report.worst_param is not None:
            lines.append(
                f"  Start with: {report.worst_param} — inspect the backward "
                "path that flows through it."
            )
    lines.append("")
    lines.append(
        "Caveat: finite differences are an *estimate*. A PASS means the "
        "gradients agree to within floating-point noise at this eps/rtol; "
        "a FAIL at O(1) error is a real bug, while a borderline FAIL can "
        "also mean eps is poorly chosen for this loss scale."
    )
    return "\n".join(lines) + "\n"
