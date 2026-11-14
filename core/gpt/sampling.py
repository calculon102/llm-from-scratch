"""Token sampling and text generation for the Mini GPT (T-405).

A language model outputs one logit per vocabulary entry.  Turning those
logits into the next token is a separate decision with four standard
controls:

* **greedy** — take ``argmax(logits)``.  Deterministic, often repetitive.
* **temperature** — divide logits by ``T`` before the softmax.  ``T < 1``
  sharpens the distribution (more likely tokens dominate); ``T > 1``
  flattens it (more surprises); ``T = 0`` is greedy.
* **top-k** — keep the ``k`` highest logits, set the rest to ``-inf``.
  A hard cap on how many candidates can be sampled.
* **top-p (nucleus)** — sort by probability and keep the smallest set
  whose cumulative probability reaches ``p``.  The candidate count adapts
  to how confident the model is: a peaked distribution keeps one token,
  a flat one keeps many.

Filters are applied in that order (temperature, top-k, top-p) and the
result is renormalised with a softmax.  ``generate`` then appends one
sampled token at a time, cropping the context to the model's
``context_length`` so generation can continue past the training window.

Example::

    from core.gpt.sampling import generate_text

    text = generate_text(
        model, tokenizer, "Die Straße",
        max_new_tokens=60, temperature=0.8, top_k=40, top_p=0.95,
    )
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = [
    "generate",
    "generate_text",
    "sample_next_token",
    "top_k_filter",
    "top_p_filter",
]


def _validate_sampling(
    *, temperature: float, top_k: int, top_p: float
) -> None:
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k must be a non-negative int (0 disables it)")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the ``k`` largest logits, replacing every other entry by ``-inf``.

    ``k <= 0`` disables the filter.  Works on the last dimension, so both
    ``(V,)`` and ``(B, V)`` logits are accepted.  Ties are resolved by
    ``torch.topk`` (the lower index wins), which keeps the operation
    deterministic.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0 or k >= logits.shape[-1]:
        return logits.clone()
    values, indices = torch.topk(logits, k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, indices, values)
    return filtered


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest nucleus whose cumulative probability is at least ``p``.

    The nucleus is computed from the softmax of ``logits`` (after any
    temperature/top-k filtering).  The highest-probability token is always
    kept, even when ``p`` is extremely small, so the distribution never
    becomes empty.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    if p >= 1.0:
        return logits.clone()
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # cumulative - prob is the mass *before* each token.  Keeping tokens
    # whose preceding mass is below p keeps the first token and the smallest
    # prefix that reaches p.
    remove = (cumulative - sorted_probs) >= p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


def _filtered_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
) -> torch.Tensor:
    _validate_sampling(temperature=temperature, top_k=top_k, top_p=top_p)
    if greedy or temperature == 0.0:
        return logits
    filtered = logits / temperature
    filtered = top_k_filter(filtered, top_k)
    filtered = top_p_filter(filtered, top_p)
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token from ``logits`` of shape ``(V,)`` or ``(B, V)``.

    Returns ``(1,)`` for a single distribution and ``(B, 1)`` for a batch,
    matching ``torch.multinomial``.  ``greedy=True`` (or
    ``temperature=0``) returns the argmax without consuming randomness.
    """
    _validate_sampling(temperature=temperature, top_k=top_k, top_p=top_p)
    if greedy or temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    filtered = _filtered_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=False,
    )
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def _as_prompt_ids(prompt_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(prompt_ids, torch.Tensor):
        if prompt_ids.dim() != 1:
            raise ValueError("prompt_ids must be 1-D")
        ids = prompt_ids.detach().to("cpu", torch.long)
    else:
        ids = torch.as_tensor(list(prompt_ids), dtype=torch.long)
        if ids.dim() != 1:
            raise ValueError("prompt_ids must be 1-D")
    if ids.numel() == 0:
        raise ValueError("prompt_ids must not be empty")
    if int(ids.min()) < 0:
        raise ValueError("prompt_ids must be non-negative")
    return ids


@torch.no_grad()
def generate(
    model,
    prompt_ids: Sequence[int] | torch.Tensor,
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    eos_id: int | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Autoregressively generate token ids after ``prompt_ids``.

    Parameters
    ----------
    model:
        A :class:`~core.gpt.model.MiniGPT` (anything with
        ``forward(idx) -> (B, T, V)`` and a ``context_length`` works).
    prompt_ids:
        The conditioning tokens; must be non-empty.  If longer than the
        model's context it is cropped to the newest tokens.
    max_new_tokens:
        Hard upper bound on generated tokens.  Zero returns a copy of the
        prompt.
    temperature / top_k / top_p / greedy:
        See the module docstring.  Invalid combinations raise
        :class:`ValueError` before any work is done.
    eos_id:
        When given, generation stops *after* appending this token.
    seed / generator:
        Control sampling.  Passing ``seed`` creates a fresh generator on
        the model's device, so repeated calls with the same seed and prompt
        are reproducible.  ``generator`` is used as-is when supplied.

    Returns
    -------
    list[int]
        ``prompt_ids`` followed by the generated tokens.
    """
    _validate_sampling(temperature=temperature, top_k=top_k, top_p=top_p)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    prompt = _as_prompt_ids(prompt_ids)
    if seed is not None and generator is not None:
        raise ValueError("pass either seed or generator, not both")

    device = next(model.parameters()).device
    input_ids = prompt.to(device).unsqueeze(0)  # (1, T)
    context_length = int(getattr(model, "context_length", input_ids.shape[1]))
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    for _ in range(int(max_new_tokens)):
        window = input_ids[:, -context_length:]
        logits = model(window)[0, -1]  # (V,)
        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            generator=generator,
        ).reshape(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        if eos_id is not None and int(next_token) == int(eos_id):
            break
    return input_ids[0].tolist()


def generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """Encode ``prompt``, generate ids, and decode the continuation.

    The prompt is encoded without adding special tokens (the model should
    continue the text, not restart after ``<eos>``).  An empty prompt starts
    from ``<bos>`` so the model can generate a fresh document.
    """
    tokenizer_bpe = getattr(tokenizer, "bpe", tokenizer)
    prompt_ids = tokenizer_bpe.encode(prompt)
    if not prompt_ids:
        prompt_ids = [int(tokenizer_bpe.bos_id)]
    ids = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        eos_id=int(tokenizer_bpe.eos_id),
        seed=seed,
        generator=generator,
    )
    return tokenizer_bpe.decode(ids, skip_special_tokens=True)
