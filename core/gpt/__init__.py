"""M4 – Mini GPT module (T-401…T-407).

A decoder-only transformer language model built from scratch and trained
with the shared Trainer loop (T-005).  The package is split the same way
as the earlier phases:

* :mod:`core.gpt.model` — causal self-attention, transformer blocks and the
  MiniGPT language model (T-401, T-402)
* :mod:`core.gpt.data` — LM chunking and train/validation split (T-403)
* :mod:`core.gpt.pipeline` — config → tokenizer → chunks → artifacts
  (T-404)
* :mod:`core.gpt.sampling` — greedy / top-k / top-p / temperature sampling
  and text generation (T-405)

The phase is registered with the config loader and the Trainer registry
when this package is imported::

    import core.gpt
    python -m scripts.train --config configs/gpt_tiny_debug.yaml
"""

from __future__ import annotations

from core.gpt.model import CausalSelfAttention

__all__ = ["CausalSelfAttention"]
