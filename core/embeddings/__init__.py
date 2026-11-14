"""M3 – Embeddings module (T-301…T-306).

A from-scratch skip-gram word2vec trainer that reuses the shared Trainer
loop (T-005), plus cosine nearest-neighbor search, optional UMAP/t-SNE
projection and a NiceGUI explorer.

The module is built ticket by ticket; this package currently exports the
data pipeline (T-301) and grows the model, tooling and phase factory in
the following tickets.
"""

from core.embeddings.data import (
    NegativeSampler,
    PairOptions,
    SkipGramDataset,
    build_skipgram_dataset,
    iter_skipgram_pairs,
    pairs_to_dataset,
)

__all__ = [
    "NegativeSampler",
    "PairOptions",
    "SkipGramDataset",
    "build_skipgram_dataset",
    "iter_skipgram_pairs",
    "pairs_to_dataset",
]
