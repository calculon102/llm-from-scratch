# Phase 3 — Embeddings: skip-gram, negative sampling, and what the vectors mean

This note explains the math and engineering behind the M3 embeddings phase:
how a token sequence becomes training pairs, how the skip-gram objective
turns co-occurrence into vectors, why negative sampling makes that objective
cheap, and how to read the resulting space with cosine neighbours and 2-D
projections. It is the conceptual companion to:

- `core/embeddings/data.py` — pair generation, the negative sampler and the
  deterministic pair dataset (T-301).
- `core/embeddings/model.py` — the two embedding tables, the negative-
  sampling loss and its accuracy diagnostics (T-302).
- `core/embeddings/pipeline.py` — corpus/tokenizer resolution, pair
  construction, and `vectors.npy`/`vocab.json`/`meta.json` artifacts.
- `core/embeddings/__init__.py` — the `embeddings` phase schema, factory and
  Trainer-registered save callback.
- `core/embeddings/neighbors.py` — cosine top-k search and word-vector
  arithmetic (T-303).
- `core/embeddings/projection.py` — t-SNE, UMAP and PCA layouts (T-304).
- `apps/ui/embeddings_page.py` — the interactive explorer (T-305).
- `configs/embeddings_tiny_debug.yaml` and `configs/embeddings_small.yaml` —
  the two bundled training profiles.

The phase is offline and CPU-friendly. It is also deliberately small: the
bundled German corpus has only a few hundred BPE tokens, which is enough to
demonstrate the objective and the tooling but not enough to produce a
production-quality word-vector space. That limitation is discussed in
§8 rather than hidden.

---

## 1. Why embeddings at all?

A tokenizer maps text to integer ids. A neural language model could treat
those ids as arbitrary labels, but then it would have to learn separately
that `Katze` and `Kater` behave similarly, and it could not generalise from
one word to a related one it has rarely seen. Embeddings replace the sparse
id with a dense vector:

$$
w \in \{1,\dots,V\}
\;\longmapsto\;
\mathbf{v}_w \in \mathbb{R}^{d},
$$

where $V$ is the vocabulary size and $d$ is the embedding dimension
(32 in `tiny_debug`, 64 in `small`). The vectors are parameters: they are
learned from data, not hand-written.

The guiding hypothesis is the **distributional hypothesis**: tokens that
occur in similar contexts tend to have similar meanings. In a large corpus,
`Katze` and `Hund` both appear near `füttern`, `schläft`, `Tier` and
`Haustier`; `Paris` and `Berlin` both appear near `Hauptstadt`, `Frankreich`
or `Deutschland`. If we force tokens with similar contexts to have similar
vectors, the geometry of the vector space becomes a rough map of usage.

A useful intuition is that the embedding matrix is a lookup table:

$$
\mathbf{V} \in \mathbb{R}^{V \times d},
\qquad
\mathbf{v}_w = \mathbf{V}[w,:].
$$

The skip-gram phase learns $\mathbf{V}$ plus a second table $\mathbf{U}$ that
is used only during training (§3.1). After training we keep $\mathbf{V}$.

---

## 2. The skip-gram objective

### 2.1 From a corpus to context pairs

Let the corpus be a sequence of token ids

$$
w_1, w_2, \dots, w_T.
$$

For a centre position $t$ and a window radius $m$, the observed context is

$$
\{w_{t-m}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+m}\},
$$

clipped at the sequence boundaries. Skip-gram asks: given the centre $w_t$,
how likely is each context token $w_{t+j}$?

The model defines that probability with a softmax over the vocabulary:

$$
p(w_o \mid w_c)
=
\frac{
  \exp\left(\mathbf{v}_{w_c}^{\top}\mathbf{u}_{w_o}\right)
}{
  \sum_{w'=1}^{V}
  \exp\left(\mathbf{v}_{w_c}^{\top}\mathbf{u}_{w'}\right)
},
$$

where $w_c$ is the centre, $w_o$ is an observed context ("output") token,
$\mathbf{v}$ is the centre table and $\mathbf{u}$ is the context table.

The skip-gram likelihood wants observed context tokens to have high
probability. Maximising it is equivalent to minimising the negative
log-likelihood:

$$
J(\theta)
=
-\frac{1}{T}
\sum_{t=1}^{T}
\sum_{\substack{-m \le j \le m \\ j \ne 0}}
\log p(w_{t+j} \mid w_t).
$$

The inner sum runs over every observed pair in the window; the outer sum
runs over every centre position. In code, this is exactly the list of
`(centre, context)` pairs produced by `iter_skipgram_pairs`.

### 2.2 Why two tables?

Word2vec uses two vectors per token:

- $\mathbf{v}_w$ — the **centre** (input) vector, used when $w$ is the token
  whose context we are predicting;
- $\mathbf{u}_w$ — the **context** (output) vector, used when $w$ is a
  positive or negative candidate.

The asymmetry makes optimisation easier: the same token can play two roles
with different statistics, and the model does not force one vector to be
good at both. At export time only $\mathbf{V}$ is kept. The context table is
a training scaffold; querying it would give a second, differently shaped
view of the same vocabulary for no clear benefit.

`SkipGramModel` therefore has `center_embeddings` and `context_embeddings`,
and its `vectors` property returns the centre table. `vectors.npy` contains
only that table.

### 2.3 Dynamic windows

With a fixed radius $m$, every context at distance $1,\dots,m$ is treated
equally. Word2vec commonly samples a smaller effective radius for each
centre:

$$
m_t \sim \text{Uniform}\{1,\dots,m\}.
$$

A context at distance $d$ is then included with probability

$$
\Pr(d \le m_t) = \frac{m-d+1}{m}
\qquad (1 \le d \le m),
$$

so close neighbours are seen more often than distant ones. This is a soft,
data-driven distance weighting: the model spends more capacity on the
tight syntactic neighbourhood and still sees longer-range topical context
sometimes. The bundled configs enable it with `dynamic_window: true`.

The generator in `core/embeddings/data.py` draws $m_t$ once per centre,
clips the window at both ends of the sequence, and never emits a
`(centre, centre)` pair. The random draw uses a seeded `random.Random`, so
the pair list is reproducible.

### 2.4 Pair caps and validation

The number of pairs grows linearly with sequence length and window size:

$$
\#\text{pairs} \approx 2mT - m(m+1)
$$

for a fixed window, or less when a dynamic window is used. `max_pairs`
(default 20 000) uniformly samples a reproducible subset when the corpus is
larger. This keeps the tiny profiles fast and makes an epoch's cost
predictable. Setting `max_pairs: 0` disables the cap.

The Trainer then splits the **pair dataset**, not the token stream, into
train and validation sets (`val_split`). This is convenient and cheap, but
it has a leakage caveat: the same centre token can occur in pairs on both
sides of the split, and neighbouring positions are highly correlated. The
validation loss therefore measures "can the model score held-out pairs from
the same text?" better than "does it generalise to new text?". Treat it as
a training diagnostic, not a benchmark.

---

## 3. Negative sampling

### 3.1 Why the softmax is too expensive

The denominator of the softmax sums over the entire vocabulary for every
pair. With $V=300$ and tens of thousands of pairs that is tolerable, but
with a realistic $V=50{,}000$ and millions of pairs it dominates training.
It also gives a tiny gradient to almost every vocabulary entry: improving
the score for one positive context requires renormalising against every
other token.

Negative sampling sidesteps the full softmax. Instead of predicting which
token is the context among all $V$ tokens, the model solves $1+K$ binary
problems for each positive pair:

- one true pair $(w_c, w_o)$ should be labelled **1**;
- $K$ random "negative" tokens $n_1,\dots,n_K$ should be labelled **0**.

The approximation is

$$
\log p(w_o \mid w_c)
\approx
\log \sigma\!\left(\mathbf{v}_{w_c}^{\top}\mathbf{u}_{w_o}\right)
+
\sum_{k=1}^{K}
\log \sigma\!\left(-\mathbf{v}_{w_c}^{\top}\mathbf{u}_{n_k}\right),
$$

with the logistic sigmoid

$$
\sigma(x) = \frac{1}{1+e^{-x}}.
$$

The implementation averages the negative terms instead of summing them:

$$
\ell
=
-\log \sigma\!\left(\mathbf{v}_{c}^{\top}\mathbf{u}_{o}\right)
-
\frac{1}{K}
\sum_{k=1}^{K}
\log \sigma\!\left(-\mathbf{v}_{c}^{\top}\mathbf{u}_{n_k}\right).
$$

That keeps the loss scale comparable when `negative_samples` changes. With
$K=5$ the starting loss is still about $2\ln 2 \approx 1.386$ (one positive
term plus the mean of five negative terms), not $(K+1)\ln 2$. The exact
scale is less important than watching it fall.

### 3.2 Gradients and intuition

For one positive pair and one negative term, write
$s_o = \mathbf{v}_c^{\top}\mathbf{u}_o$ and
$s_k = \mathbf{v}_c^{\top}\mathbf{u}_{n_k}$. Differentiating the loss gives

$$
\frac{\partial \ell}{\partial \mathbf{v}_c}
=
-\left(1-\sigma(s_o)\right)\mathbf{u}_o
+
\sum_{k=1}^{K}
\sigma(s_k)\,\mathbf{u}_{n_k}
$$

for the summed form (divide the negative part by $K$ for the implemented
mean). The positive term pulls the centre vector toward the observed
context vector; every negative term pushes it away from a random token's
context vector. The same updates flow back into $\mathbf{U}$:

$$
\frac{\partial \ell}{\partial \mathbf{u}_o}
=
-\left(1-\sigma(s_o)\right)\mathbf{v}_c,
\qquad
\frac{\partial \ell}{\partial \mathbf{u}_{n_k}}
=
\sigma(s_k)\,\mathbf{v}_c .
$$

A well-trained model has $\sigma(s_o)$ close to 1 for observed pairs and
$\sigma(s_k)$ close to 0 for random tokens. The logged metrics expose this
directly:

| metric | meaning |
|---|---|
| `pos_acc` | fraction of positive pairs with score $> 0$ (sigmoid $> 0.5$) |
| `neg_acc` | fraction of negative pairs with score $< 0$ |
| `acc` | mean of the two |

At initialisation both are near 0.5 because the dot products are near zero.
`pos_acc` rising while `neg_acc` also rises is the healthy pattern.

### 3.3 The noise distribution

Negatives are not drawn uniformly. They follow the smoothed unigram
distribution

$$
P_n(w) =
\frac{f(w)^{0.75}}
     {\sum_{w'} f(w')^{0.75}},
$$

where $f(w)$ is the token's count in the training stream. The exponent
$0.75$ is the word2vec default and has three useful effects:

- **between unigram and uniform:** exponent $1$ would make `der`, `die`,
  `und` almost the only negatives; exponent $0$ would waste most negatives
  on very rare tokens;
- **more informative negatives:** frequent tokens still appear often, but
  rare and mid-frequency tokens get more probability mass than they would
  under the raw unigram distribution;
- **stable estimates:** raising counts to a fractional power shrinks the
  dynamic range, which makes gradients less dominated by a few function
  words.

`NegativeSampler` builds the cumulative weights once and uses
`torch.multinomial` to draw batches. It can exclude ids per draw, and the
dataset excludes the centre and the positive context of the pair so the
model is not asked to separate a token from itself. This exclusion does not
remove *other* occurrences of the same context token from the negative set,
and on a tiny vocabulary the chance of a "false negative" remains
non-negligible. That is a known approximation, not a bug.

### 3.4 Initialisation

`nn.Embedding` initialises weights from $N(0,1)$. With $d=32$, a dot product
of two such vectors has standard deviation roughly $\sqrt{d}$, so most
initial sigmoids saturate and gradients nearly vanish. The model instead
uses the word2vec-style uniform initialisation

$$
\mathbf{v}_w, \mathbf{u}_w \sim
\text{Uniform}\left(-\frac{0.5}{d}, \frac{0.5}{d}\right).
$$

Initial dot products are then small, the sigmoid stays in its responsive
region, and the loss starts near $2\ln 2$.

---

## 4. From corpus to batches

The pipeline is intentionally explicit:

```text
data/sample_de/*.txt
   │  ingest_corpus: decode, clean, filter, deduplicate (M2)
   ▼
cleaned German text
   │  BPETokenizer.encode: text → ids
   ▼
content token ids (special tokens removed)
   │  iter_skipgram_pairs + dynamic windows
   ▼
(centre, context) pairs
   │  NegativeSampler + per-pair seeded RNG
   ▼
{center, context, negatives[B,K]} batches → shared Trainer
```

Two details are worth calling out.

**Special tokens are excluded.** `<pad>`, `<unk>`, `<bos>` and `<eos>` are
structural markers. Training an embedding for `<pad>` would mostly learn
padding statistics, and exporting an untrained row would let the UI query a
meaningless vector. `prepare_embedding_data` drops special ids from the
token stream, and the save callback exports only the non-special rows.

**Pair-level randomness is seeded per item.** Each dataset item derives its
negative-sampling generator from the global seed and the pair index, rather
than from a shared mutable RNG. The same pair therefore gets the same
negatives regardless of DataLoader iteration order, which makes tests and
repeated runs reproducible. The pair list itself is also deterministic for
a given seed and cap.

The tokenizer can be trained on the configured corpus or reused from a
previous M2 run:

```yaml
phase_config:
  tokenizer_bundle: ""            # empty: train a fresh BPE bundle
  # tokenizer_bundle: artifacts/<run>/tokenizers
  corpus_path: data/sample_de
```

When a bundle is supplied, it is loaded and copied into the embeddings run;
when it is not, `train_bundle` builds character and BPE tokenizers from the
cleaned corpus. Either way the run is self-contained: it records the
tokenizer and the corpus text it used.

---

## 5. Reading the space: cosine similarity

The learned vectors are not normalised during training, so their lengths
vary. For semantic comparison, word2vec uses **cosine similarity**:

$$
\cos(\mathbf{a},\mathbf{b})
=
\frac{\mathbf{a}^{\top}\mathbf{b}}
     {\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}.
$$

It measures the angle between vectors and ignores their length. For a query
token $q$, the neighbour list is

$$
\operatorname{topk}_{w \ne q}\ \cos(\mathbf{v}_q, \mathbf{v}_w).
$$

`EmbeddingIndex` computes the row norms once, stores an L2-normalised copy
of the matrix, and answers a query with one matrix-vector product. The query
token itself is excluded. Ties are broken by vocabulary id so the output is
stable. A zero query vector (which can happen for a completely untrained or
degenerate row) yields all-zero scores instead of a division by zero.

Two cautions:

- **Direction is the semantic signal; norm is not a probability.** The norm
  often correlates with frequency and with how strongly a token was
  optimised, but it is not a calibrated frequency count. `save_projection_plot`
  uses norm only as a colour/salience cue.
- **Top-k is only as good as the corpus.** On a few hundred tokens, many
  "neighbours" are spelling fragments such as `bl` or `ße`, because German
  BPE splits rare words and those fragments share contexts. This is a data
  limitation, not a failure of cosine similarity.

For a quick sanity check, the index also supports word-vector arithmetic:

```python
index.most_similar(positive=["könig", "frau"], negative=["mann"], k=5)
```

The result is the mean of the positive vectors minus the mean of the
negative vectors, ranked by cosine. On a tiny corpus it is illustrative
only; it needs a large, clean corpus to be meaningful.

---

## 6. Two-dimensional projections

A 32- or 64-dimensional vector cannot be plotted directly. The explorer
offers three layouts, each preserving a different thing.

### 6.1 PCA

Principal component analysis centres the matrix and computes its singular
value decomposition

$$
\mathbf{X} - \bar{\mathbf{x}}
=
\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}.
$$

The first two columns of $\mathbf{V}$ are the directions of largest
variance, so the 2-D coordinates are

$$
\mathbf{Z} = (\mathbf{X}-\bar{\mathbf{x}})\,\mathbf{V}_{:,:2}.
$$

PCA is deterministic, fast and dependency-free (NumPy only). It preserves
global variance and shows the dominant axes of the space, but it is a
linear projection: a curved manifold gets flattened, and local clusters can
overlap even when they are distinct in the original space. The explained
variance ratio of the two components is recorded, so it is clear how much
of the total variance the plot actually shows.

### 6.2 t-SNE

t-SNE converts pairwise distances into neighbour probabilities. In the
high-dimensional space it fits a Gaussian around each point, with a
per-point bandwidth controlled indirectly by the perplexity (roughly "how
many effective neighbours to consider"). In the 2-D space it uses a
heavy-tailed Student-$t$ kernel:

$$
q_{ij}
=
\frac{
  \left(1+\lVert\mathbf{z}_i-\mathbf{z}_j\rVert^2\right)^{-1}
}{
  \sum_{k\ne l}
  \left(1+\lVert\mathbf{z}_k-\mathbf{z}_l\rVert^2\right)^{-1}
}.
$$

It then minimises the Kullback–Leibler divergence

$$
\mathrm{KL}(P\,\Vert\,Q)
=
\sum_{i\ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}},
$$

which strongly rewards keeping high-probability neighbours close. The
result is excellent at revealing local clusters. It is **not** a metric
embedding: distances between clusters, cluster sizes and axis directions
are not meaningful. It is also stochastic and sensitive to perplexity,
initialisation and seed.

### 6.3 UMAP

UMAP builds a weighted nearest-neighbour graph in the high-dimensional
space, then optimises a low-dimensional graph layout with a cross-entropy
objective. `n_neighbors` controls the balance between local and global
structure (larger values preserve more global organisation), and `min_dist`
controls how tightly points may pack. This implementation passes
`metric="cosine"` and a fixed random seed, and uses UMAP's
`compatibility_layout` flag when the installed version offers it so results
stay reproducible across releases.

UMAP is usually faster than t-SNE on larger vocabularies and preserves more
global structure, but it adds `umap-learn`/numba as a dependency and is
still a stochastic, non-metric visualisation.

### 6.4 What the plot is and is not

The projections are **exploration tools**, not measurements. If two tokens
look close in a t-SNE plot, that is a hint to query their cosine neighbours;
it is not evidence that their cosine similarity is high. The code therefore
L2-normalises the vectors before projecting (cosine geometry) and keeps the
neighbour list as the authoritative view:

```bash
python -m scripts.embeddings project --run artifacts/<run> --method tsne
```

writes a PNG under `<run>/plots/`, while

```bash
python -m scripts.embeddings neighbors --run artifacts/<run> --token Katze
```

prints the actual ranked similarities.

---

## 7. Training recipe, configuration and artifacts

### 7.1 Profiles

| setting | `embeddings_tiny_debug` | `embeddings_small` |
|---|---:|---:|
| `embedding_dim` | 32 | 64 |
| `vocab_size` | 300 | 400 |
| `window_size` | 3 | 4 |
| `negative_samples` | 5 | 7 |
| `max_pairs` | 20 000 | 0 (no cap) |
| `batch_size` | 32 | 64 |
| `learning_rate` | 0.01 | 0.01 |
| `epochs` | 15 | 25 |
| `val_split` | 0.1 | 0.1 |

Both use `negative_power: 0.75`, `dynamic_window: true` and the same
German-first corpus cleaning options as M2. The tiny profile is the default
for tests and quick iteration; the small profile is for a slightly richer
vector space. Neither is a large-scale word2vec run.

### 7.2 Command line

```bash
python -m scripts.train --config configs/embeddings_tiny_debug.yaml
```

The shared Trainer handles the loop, logging, validation, checkpoints and
resume. The phase factory resolves the tokenizer and corpus, builds the
pair dataset, and registers an `on_end` callback that exports the learned
vectors.

### 7.3 Artifact layout

```text
artifacts/<timestamp>_<run_name>/
  config.yaml
  run.log
  run_meta.json
  checkpoints/
  tokenizers/              # copy of the bundle used for this run
  embeddings/
    vectors.npy            # float32 (vocab_size, embedding_dim), centre table
    vocab.json             # JSON list; row i ↔ token i
    meta.json              # format/version + training metadata
    corpus.txt             # cleaned corpus the vectors were trained on
  plots/                   # projection PNGs when requested
```

`vocab.json` is a list, not a mapping, so the row alignment with
`vectors.npy` is unambiguous. `meta.json` records the format marker,
dimensions and the settings needed to interpret a run (tokenizer source,
corpus path, window, negatives, pair/token counts, training summary). The
run's checkpoints contain the full model, including the context table that
is deliberately not exported.

### 7.4 The UI explorer

Start the app with `python -m scripts.ui` and open `/embeddings`. The page:

- discovers run folders with `find_embedding_dirs` and loads the newest by
  default;
- shows the vocabulary size and embedding dimension;
- accepts a token and a top-k value and renders a ranked neighbour table
  with cosine scores;
- projects the space with t-SNE, UMAP or PCA on demand, showing the image
  inline and saving a PNG in the run's `plots/` directory.

The page is a thin client over the same `EmbeddingIndex` and
`project_embeddings` functions used by the CLI and tests, so there is no
second implementation to drift.

---

## 8. Failure modes and diagnostics

### 8.1 The corpus is tiny

The bundled corpus is about 1.9 KB (roughly 250 content tokens after BPE).
That is enough to check that loss falls, that pairs and negatives have the
right shapes, and that artifact round-trips work. It is not enough to learn
a useful German embedding space. Expect neighbour lists dominated by
function words and BPE fragments, and do not over-interpret them.

**Diagnostic:** compare `vocab_size` and pair count in `meta.json`; if the
vocabulary is a few hundred tokens, the geometry is anecdotal.

### 8.2 BPE fragmentation

BPE may represent `Straße` as several pieces. Each piece gets its own
embedding, so a query for a whole word may fail (`token not in vocabulary`)
or return morphological relatives rather than semantic ones. Inspect
`vocab.json` and query the actual pieces. A larger, more diverse corpus (or
word-level tokenisation) reduces fragmentation.

### 8.3 Frequent-token dominance without subsampling

This implementation uses negative-sampling power $0.75$ but does not
implement word2vec's frequent-word **subsampling** (dropping frequent tokens
with a probability based on their frequency). Frequent function words can
therefore dominate the pair stream and the neighbour lists. `negative_power`
partly compensates by giving rare tokens more negative-sampling mass, but
subsampling remains the natural next extension for larger corpora.

### 8.4 Leakage in the pair-level validation split

Train and validation are random pairs from the same token stream. The same
centre can appear in both, and adjacent pairs share context. A falling
validation loss is evidence that the objective is being optimised, not that
the vectors generalise to unseen text. For an honest held-out estimate,
encode a separate document/file and evaluate on its pairs.

### 8.5 Overfitting and seed variance

On a tiny corpus a high-capacity model can memorise local co-occurrences.
Watch `best_val` versus the final training loss; the run records both. The
pair order, initialisation, negative draws and stochastic projections all
depend on seeds, so two runs with different seeds can produce visibly
different neighbour lists. Compare a few seeds before drawing conclusions.

### 8.6 False negatives on a small vocabulary

Negatives are drawn from the smoothed unigram distribution, excluding the
pair's own centre and positive context. Other occurrences of the same token
type can still appear as negatives, and with $V$ in the hundreds this is not
rare. The model then receives contradictory labels for the same token.
Larger vocabularies and the usual $K=5$–$15$ range make this effect small;
on a tiny corpus it is one more reason not to trust fine-grained geometry.

### 8.7 Projection artefacts

t-SNE and UMAP can produce impressive-looking clusters from noise, and their
inter-cluster distances are not metric. A cluster in a 2-D plot does not by
itself mean the tokens are cosine neighbours. Always confirm with the
neighbour table, and vary the seed/perplexity/`n_neighbors` to see what is
stable.

### 8.8 Untrained or rare rows

Special tokens are not trained and are filtered from the exported matrix.
Rare BPE pieces may receive very few updates, so their vectors are close to
their random initial values. A small norm or a neighbour list of unrelated
tokens is a signal to check the token's count before blaming the model.

---

## 9. What to verify in a run

A healthy tiny run shows:

1. **Initial loss near $2\ln 2 \approx 1.386$** and initial `acc` near 0.5.
2. **Falling loss** over the first few hundred steps, with `pos_acc` and
   `neg_acc` both rising above 0.5.
3. **A best validation loss** recorded in the run log; later epochs may
   overfit.
4. **Artifacts that load**: `vectors.npy` shape matches
   `(len(vocab.json), embedding_dim)`, and a neighbour query returns
   finite scores.
5. **A projection PNG** with as many points as the exported vocabulary,
   generated without errors.

The full pipeline is intentionally inspectable: every quantity in this note
maps to a function in `core/embeddings/`, and every training run records the
configuration and corpus that produced its vectors.
