"""Skip-gram with negative sampling: the model and its objective (T-302).

The model has two embedding tables, exactly as in word2vec:

* ``center_embeddings`` — $v_w$, the vector for a token when it is the
  centre of a context window.  These are the vectors we keep and query;
* ``context_embeddings`` — $u_w$, a second vector per token used only while
  training to score context and negative tokens.

For one positive pair $(w, c)$ and $K$ negatives $n_1,\dots,n_K$ drawn from
the noise distribution, the loss is

$$
\ell = -\log\sigma(v_w^\top u_c)
      - \sum_{k=1}^{K} \log\sigma(-v_w^\top u_{n_k}),
$$

where $\sigma(x)=1/(1+e^{-x})$.  Minimizing it raises the dot product with
the observed context and lowers it for random distractors.  Gradients flow
into *both* tables, but only the centre table is exported: a token's
context role is a training scaffold, not the thing we want to look up.

Two implementation details matter for stable training:

* **Small initial weights.**  ``nn.Embedding``'s default $N(0,1)$ makes
  initial dot products enormous, so every sigmoid saturates and gradients
  vanish.  We use the word2vec-style uniform
  $[-0.5/d, 0.5/d]$, which keeps initial logits near zero.
* **Mean over positives and negatives.**  Summing $K$ negative terms makes
  the loss scale with ``negative_samples`` and changes the effective
  learning rate.  The mean keeps the objective comparable across profiles.

``skipgram_loss`` is the thin adapter the shared Trainer calls.  It moves
the batch to the model's device (the Trainer leaves batching on CPU) and
returns the scalar loss plus accuracy metrics for the run log.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SkipGramModel",
    "skipgram_loss",
]


class SkipGramModel(nn.Module):
    """Word2vec skip-gram with negative sampling.

    Parameters
    ----------
    vocab_size:
        Number of token ids covered by the tokenizer.
    embedding_dim:
        Width of the learned vectors.  ``32`` is plenty for a tiny German
        corpus; 128–300 is the usual range for real word2vec runs.
    init_scale:
        Optional override for the uniform initialisation bound.  The
        default ``0.5 / embedding_dim`` follows word2vec.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        init_scale: float | None = None,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.center_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.context_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        bound = float(init_scale) if init_scale is not None else 0.5 / self.embedding_dim
        self.reset_parameters(bound)

    def reset_parameters(self, bound: float | None = None) -> None:
        """Initialise both tables uniformly in ``[-bound, bound]``."""
        if bound is None:
            bound = 0.5 / self.embedding_dim
        if bound <= 0:
            raise ValueError("init scale must be > 0")
        for table in (self.center_embeddings, self.context_embeddings):
            nn.init.uniform_(table.weight, -bound, bound)

    @property
    def vectors(self) -> torch.Tensor:
        """The exported centre vectors, shape ``(vocab_size, embedding_dim)``."""
        return self.center_embeddings.weight

    def forward(
        self,
        center: torch.Tensor,
        context: torch.Tensor,
        negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(loss, metrics)`` for one batch.

        ``center`` and ``context`` are ``(B,)`` id tensors, ``negatives``
        is ``(B, K)``.  The metrics are detached floats, safe for logging:
        ``acc`` (fraction of positive and negative scores on the correct
        side of zero), ``pos_acc`` and ``neg_acc`` separately.
        """
        if center.dim() != 1 or context.dim() != 1:
            raise ValueError("center and context must be 1-D id tensors")
        if negatives.dim() != 2 or negatives.shape[0] != center.shape[0]:
            raise ValueError(
                "negatives must have shape (batch, num_negatives) matching center"
            )

        center_vectors = self.center_embeddings(center)             # (B, D)
        context_vectors = self.context_embeddings(context)          # (B, D)
        negative_vectors = self.context_embeddings(negatives)       # (B, K, D)

        positive_scores = (center_vectors * context_vectors).sum(dim=-1)  # (B,)
        negative_scores = torch.bmm(
            negative_vectors, center_vectors.unsqueeze(-1)
        ).squeeze(-1)                                                     # (B, K)

        positive_loss = -F.logsigmoid(positive_scores).mean()
        negative_loss = -F.logsigmoid(-negative_scores).mean()
        loss = positive_loss + negative_loss

        with torch.no_grad():
            positive_hits = (positive_scores > 0).float().mean()
            negative_hits = (negative_scores < 0).float().mean()
            accuracy = (positive_hits + negative_hits) / 2
        metrics = {
            "acc": float(accuracy.item()),
            "pos_acc": float(positive_hits.item()),
            "neg_acc": float(negative_hits.item()),
        }
        return loss, metrics


def skipgram_loss(
    model: SkipGramModel, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Trainer adapter: move the batch to the model's device and score it.

    The shared Trainer deliberately keeps batch tensors on CPU so phases
    stay device-agnostic; the model may live on CUDA/ROCm.  This function
    bridges that gap in one place.
    """
    device = next(model.parameters()).device
    center = batch["center"].to(device=device, dtype=torch.long)
    context = batch["context"].to(device=device, dtype=torch.long)
    negatives = batch["negatives"].to(device=device, dtype=torch.long)
    return model(center, context, negatives)
