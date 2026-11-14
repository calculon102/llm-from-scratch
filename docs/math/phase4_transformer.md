# Phase 4 — Mini GPT: causal attention and the language-model objective

This note explains the math behind the M4 decoder-only transformer and the
next-token training objective. It is the conceptual companion to:

- `core/gpt/model.py` — causal self-attention, the pre-LN transformer block,
  `MiniGPT`, and the Trainer loss adapter (T-401, T-402).
- `core/gpt/data.py` — token-stream chunking, the contiguous train/validation
  split, and batch construction (T-403).
- `core/gpt/pipeline.py` — corpus/tokenizer resolution, perplexity evaluation,
  and model artifacts (T-404).
- `core/gpt/sampling.py` — greedy, temperature, top-k and top-p generation
  (T-405).
- `apps/ui/gpt_page.py` — the interactive generation page (T-406).
- `configs/gpt_tiny_debug.yaml` — the fast CPU profile used for tests.
- `docs/math/phase2_tokenizer.md` and `docs/math/phase3_embeddings.md` — the
  tokenizer and embedding background this phase builds on.

The model is deliberately small and readable rather than state of the art.
The tiny profile (2 layers, 2 heads, width 32, context 16, ~34 k parameters)
trains on a laptop CPU in seconds. That is enough to verify the mathematics,
the masks and the training loop; it is not enough to produce fluent German.
The gap between "the loss falls" and "the samples are good" is discussed in
§10.

---

## 1. The task: predict the next token

A tokenizer turns text into a sequence of integer ids

$$
x_0, x_1, \dots, x_{N-1}, \qquad x_t \in \{0,\dots,V-1\}.
$$

A causal language model defines a probability distribution over the next
token given everything before it:

$$
p_\theta(x_t \mid x_0,\dots,x_{t-1}).
$$

Applying the chain rule to a whole sequence gives

$$
p_\theta(x_0,\dots,x_{N-1})
=
\prod_{t=0}^{N-1} p_\theta(x_t \mid x_0,\dots,x_{t-1}).
$$

Training maximises the likelihood of the observed corpus, which is the same
as minimising the mean negative log-likelihood:

$$
\mathcal{L}(\theta)
=
-\frac{1}{N}\sum_{t=0}^{N-1}
\log p_\theta(x_t \mid x_0,\dots,x_{t-1}).
$$

The important structural constraint is the direction of the conditioning:
position $t$ may use positions $0,\dots,t-1$ and nothing later. Without that
constraint the model could copy the answer from the future, and the loss
would collapse for the wrong reason.

A transformer computes all $T$ next-token predictions in one forward pass.
The input window is

$$
\mathbf{x} = (x_0,\dots,x_{T-1}),
$$

and the targets are the same sequence shifted by one:

$$
\mathbf{y} = (x_1,\dots,x_T).
$$

The model output at position $t$ is a vector of $V$ logits, and its target is
$y_t = x_{t+1}$. This is why `LMChunkDataset` stores `input_ids` and
`targets` of the same shape `(T,)`, and why a batch is shaped `(B, T)`.

---

## 2. From token ids to a residual stream

### 2.1 Token and position embeddings

The model learns two lookup tables:

- a token embedding $\mathbf{E}\in\mathbb{R}^{V\times d}$;
- a position embedding $\mathbf{P}\in\mathbb{R}^{T_{\max}\times d}$.

For a token $x_t$ at position $t$, the initial hidden state is

$$
\mathbf{h}^{(0)}_t
=
\mathbf{E}[x_t] + \mathbf{P}[t].
$$

Both rows are learned. The addition, rather than concatenation, keeps the
width fixed at $d$ and lets the model use the same residual stream
throughout. Dropout is applied to the sum during training.

The positions are absolute and learned, not sinusoidal or rotary. That is
the simplest option and is sufficient for a small educational model. The
limit is explicit: the model has never seen position $T_{\max}$ and cannot
represent it, so `MiniGPT.forward` rejects sequences longer than
`context_length`.

### 2.2 Shapes for the tiny profile

With batch size $B=8$, context $T=16$, width $d=32$, $H=2$ heads and
$V=259$ BPE tokens:

| tensor | shape |
|---|---|
| input ids | `(8, 16)` |
| hidden state | `(8, 16, 32)` |
| fused QKV projection | `(8, 16, 96)` |
| per-head queries/keys/values | `(8, 2, 16, 16)` |
| attention scores / weights | `(8, 2, 16, 16)` |
| attention output | `(8, 16, 32)` |
| logits | `(8, 16, 259)` |

The head width is $d_h = d/H = 16$. Every head has its own $d_h$-dimensional
view of the same residual stream; the heads are concatenated back to width
$d$ after attention.

---

## 3. Causal self-attention

### 3.1 Queries, keys and values

Attention is a learned, content-dependent weighted average. Each position
produces three vectors by linear projection:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\qquad
\mathbf{K} = \mathbf{X}\mathbf{W}_K,\qquad
\mathbf{V} = \mathbf{X}\mathbf{W}_V,
$$

with $\mathbf{X}\in\mathbb{R}^{T\times d}$ and each
$\mathbf{W}\in\mathbb{R}^{d\times d}$. The implementation fuses the three
projections into one `qkv_proj` layer for speed, then splits the result into
three chunks.

For one head, the raw compatibility score between a query at position $i$
and a key at position $j$ is the dot product

$$
s_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}}.
$$

The scaling by $1/\sqrt{d_h}$ matters. If the components of $\mathbf{q}$ and
$\mathbf{k}$ have roughly unit variance and are independent, their dot
product has variance $d_h$. Without scaling, larger heads produce very large
scores, the softmax saturates on one position, and gradients through the
other positions almost vanish. Dividing by $\sqrt{d_h}$ keeps the score
variance roughly constant as the head width changes.

### 3.2 The causal mask

Let $\mathbf{S}\in\mathbb{R}^{T\times T}$ be the matrix of scaled scores.
The causal mask is

$$
M_{ij} =
\begin{cases}
1 & j \le i,\\
0 & j > i.
\end{cases}
$$

Before the softmax, forbidden entries are replaced by negative infinity:

$$
\tilde{\mathbf{S}}_{ij} =
\begin{cases}
s_{ij} & j \le i,\\
-\infty & j > i.
\end{cases}
$$

Then each row is normalised:

$$
a_{ij}
=
\frac{\exp(\tilde s_{ij})}
     {\sum_{k=0}^{T-1}\exp(\tilde s_{ik})}.
$$

Because $\exp(-\infty)=0$, position $i$ puts exactly zero probability on
every future position. The resulting matrix is lower triangular, and each
row sums to one. In code the mask is a boolean lower-triangular buffer
(`causal_mask`), and the line

```python
scores = scores.masked_fill(~mask, float("-inf"))
```

is the entire difference between a causal language model and a
bidirectional encoder. `tests/test_gpt_attention.py` checks both the mask
itself and the stronger property that changing a future token does not
change any past output.

### 3.3 Multi-head attention

One attention head can only average values according to one similarity
pattern. Multi-head attention runs $H$ independent heads in parallel:

$$
\mathbf{Q}_h = \mathbf{X}\mathbf{W}_{Q,h},\quad
\mathbf{K}_h = \mathbf{X}\mathbf{W}_{K,h},\quad
\mathbf{V}_h = \mathbf{X}\mathbf{W}_{V,h},
$$

$$
\mathbf{A}_h = \operatorname{softmax}
\left(
\frac{\mathbf{Q}_h\mathbf{K}_h^\top}{\sqrt{d_h}}
+ \mathbf{M}_{\text{causal}}
\right),
\qquad
\mathbf{O}_h = \mathbf{A}_h\mathbf{V}_h.
$$

The heads are concatenated and mixed by an output projection:

$$
\mathbf{O} = \operatorname{Concat}(\mathbf{O}_1,\dots,\mathbf{O}_H)\mathbf{W}_O.
$$

Each head is computed as a batched matrix product of shape
`(B, H, T, T) @ (B, H, T, d_h) -> (B, H, T, d_h)`. After transposing back
and concatenating, the result is `(B, T, d)`.

The cost is quadratic in the context length:

$$
\mathcal{O}(B\,H\,T^2\,d_h) = \mathcal{O}(B\,T^2\,d)
$$

for the score and weighted-sum products, plus $\mathcal{O}(B\,T^2\,H)$
memory for the attention matrix. With $T=16$ this is negligible; it becomes
the central scaling concern for long contexts.

### 3.4 What the attention weights mean

`CausalSelfAttention.attention_weights(x)` returns the normalised
probabilities without dropout, shape `(B, H, T, T)`. Row $i$ answers: "when
forming the representation at position $i$, how much mass did this head put
on each earlier position?" The lower-triangular structure is the causal
constraint. It is a diagnostic, not a explanation of the model's decision:
attention weights are only one part of a residual network, and a high weight
does not by itself prove that a token caused an output.

---

## 4. The transformer block

### 4.1 Pre-LayerNorm residual blocks

Each block has two sublayers, attention and a position-wise MLP. The
implementation uses pre-LayerNorm:

$$
\mathbf{a} = \mathbf{x} + \operatorname{Attn}(\operatorname{LN}_1(\mathbf{x})),
$$

$$
\mathbf{y} = \mathbf{a} + \operatorname{MLP}(\operatorname{LN}_2(\mathbf{a})).
$$

The residual connection is an identity path around each sublayer. It lets
gradients flow directly from the loss to earlier layers and lets a block
learn a small correction rather than a complete replacement for its input.

Normalising *before* the sublayer keeps that identity path clean. The
original GPT used post-LN, $\operatorname{LN}(\mathbf{x}+\operatorname{Attn}(\mathbf{x}))$,
which can be harder to train without warmup. Pre-LN is the modern default
and is what this model uses.

### 4.2 The MLP

Attention mixes information across positions. The MLP transforms each
position independently:

$$
\operatorname{MLP}(\mathbf{x})
=
\mathbf{W}_2\,\operatorname{GELU}(\mathbf{W}_1\mathbf{x}),
$$

with

$$
\mathbf{W}_1\in\mathbb{R}^{4d\times d},
\qquad
\mathbf{W}_2\in\mathbb{R}^{d\times 4d}.
$$

The 4x expansion is the usual transformer ratio. Most of the parameters and
most of the per-token nonlinear computation live here. GELU is a smooth
version of ReLU:

$$
\operatorname{GELU}(x) \approx x\,\Phi(x),
$$

where $\Phi$ is the standard normal CDF. The exact PyTorch implementation is
used, not this approximation.

### 4.3 Final normalisation and logits

After $L$ blocks, a final LayerNorm produces
$\mathbf{H}\in\mathbb{R}^{B\times T\times d}$. The language-model head maps
each hidden state to vocabulary logits:

$$
\mathbf{Z} = \mathbf{H}\mathbf{W}_U^\top,
\qquad
\mathbf{Z}\in\mathbb{R}^{B\times T\times V}.
$$

There is no bias in the head. With `tie_weights: true` (the default), the
output matrix is the token embedding matrix itself:

$$
\mathbf{W}_U = \mathbf{E}.
$$

Weight tying saves $Vd$ parameters and gives the input and output token
representations a shared geometry: a token's embedding is also the direction
used to score it as the next token. GPT-2 popularised this trick. The tiny
model enables it by default; the config can turn it off for experiments.

The total parameter count is roughly

$$
\underbrace{Vd + Td}_{\text{embeddings}}
+
L\underbrace{\left(4d^2 + 8d^2\right)}_{\text{attention + MLP}}
+
\mathcal{O}(Ld),
$$

where the $4d^2$ attention term counts the four $d\times d$ projections and
the $8d^2$ MLP term counts the two $d\leftrightarrow4d$ matrices. Weight
tying removes a second $Vd$ output matrix.

---

## 5. The language-model loss

### 5.1 Cross-entropy at every position

At position $t$ the model produces logits $\mathbf{z}_t\in\mathbb{R}^V$.
The predicted probability of token $w$ is

$$
p_\theta(w\mid x_{<t})
=
\frac{\exp(z_{t,w})}
     {\sum_{v=0}^{V-1}\exp(z_{t,v})}.
$$

For a batch of $B$ windows of length $T$, the training loss is the mean
cross-entropy over all $B\cdot T$ predictions:

$$
\mathcal{L}
=
-\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=0}^{T-1}
\log p_\theta\!\left(y_{b,t}\mid x_{b,<t}\right),
$$

with $y_{b,t}=x_{b,t+1}$. Flattening the logits to `(B*T, V)` and the
targets to `(B*T,)` is exactly what `language_model_loss` does before calling
`torch.nn.functional.cross_entropy`.

Every position contributes. There is no padding in the chunk dataset: the
last partial window is dropped, so no loss is computed on pad tokens and no
attention mask for padding is needed.

### 5.2 Perplexity

Perplexity is the exponentiated mean negative log-likelihood:

$$
\operatorname{PPL}
=
\exp\left(
-\frac{1}{N_{\text{tok}}}
\sum_{i=1}^{N_{\text{tok}}}
\log p_\theta(y_i\mid x_{<i})
\right)
=
\exp(\text{mean NLL}).
$$

It has a useful interpretation: a uniform model over $V$ tokens has
perplexity $V$, and lower perplexity means the model is, on average, as
surprised as a uniform chooser over that many equally likely tokens. A
perplexity of 10 is not "10% accuracy"; it is an exponential average of
probabilities.

Two implementation details matter:

1. **Token weighting.** `evaluate_perplexity` sums the negative
   log-likelihood over every target token and divides by the total token
   count. It does not average per-batch losses, because the last batch can
   be smaller and would otherwise get equal weight. This is the same
   weighting the training loss uses.
2. **Numerical clamp.** The reported value uses
   `exp(min(mean_loss, 50))`. A badly initialised model can have a huge loss,
   and exponentiating it would overflow to `inf`. The clamp keeps logs and
   plots finite; it does not change training.

Perplexity is only comparable across runs that use the same tokenizer and
vocabulary. A BPE model with 259 tokens and a word-level model with 50 000
tokens are answering different multiple-choice questions, so their
perplexities are not on the same scale.

A useful sanity check is a deliberately zeroed model: all logits equal, so
the predicted distribution is uniform and the measured perplexity is exactly
$V$. `tests/test_gpt_training.py` verifies that identity.

---

## 6. Turning a corpus into training windows

### 6.1 Windows and targets

Let the encoded corpus contain $N$ tokens. A window of length $T$ starting
at offset $s$ uses

$$
\text{input} = x_s,\dots,x_{s+T-1},
\qquad
\text{target} = x_{s+1},\dots,x_{s+T}.
$$

It therefore needs $T+1$ tokens. The possible starts are

$$
s \in \{0,\ \text{stride},\ 2\cdot\text{stride},\dots\},
\qquad
s \le N-T-1.
$$

`chunk_starts` implements this as `range(0, N - T, stride)`: the stop value
$N-T$ is exclusive, so the last start is at most $N-T-1$. A partial final
window is dropped rather than padded. The number of full windows for
$N>T$ is

$$
\left\lfloor\frac{N-T-1}{\text{stride}}\right\rfloor + 1.
$$

With `stride = T` (the default when `stride: 0` in the config) the windows
do not overlap. With a smaller stride they overlap and the effective number
of training examples grows, at the cost of more correlated updates. A
stride larger than $T$ skips tokens.

For the tiny QA run the cleaned sample corpus encoded to 878 BPE tokens.
With `context_length: 16` and a 10% validation split, the train prefix had
790 tokens and the validation suffix 88 tokens, producing 49 train windows
and 5 validation windows.

### 6.2 Split before chunking

The split is a contiguous suffix of the token stream:

$$
\underbrace{x_0,\dots,x_{s-1}}_{\text{train}},
\qquad
\underbrace{x_s,\dots,x_{N-1}}_{\text{validation}}.
$$

Chunking then happens independently on each side. This ordering is
essential. If we chunked first and split the windows randomly, two adjacent
windows would share almost all their tokens and the validation loss would be
measuring memorisation of a neighbouring window. The contiguous suffix
makes the validation tokens genuinely unseen, at least within the corpus.

The split fraction is applied to tokens, not windows. `val_split: 0.1`
reserves about 10% of the token stream. `build_lm_datasets` raises a clear
error if either side is too short to produce one full window, since a
language-model split needs at least $2(T+1)$ tokens in total.

One caveat remains: a contiguous suffix from the same document is not a
random sample of German. It is a held-out *text region*. For a stronger
generalisation estimate, hold out whole documents or files and report the
per-document results. The tiny corpus is too small for that distinction to
matter, but the code keeps the split deterministic and honest.

### 6.3 Special tokens

With `add_special_tokens: true`, the BPE encoder adds `<bos>` at the start
and `<eos>` at the end of the encoded corpus. The model therefore sees a
beginning-of-sequence signal once and an end-of-sequence signal once, not
one pair per line. The EOS token can appear in targets near the end of the
stream, and `generate_text` stops if it samples EOS. On the tiny corpus EOS
is rare during generation; that is expected, not a bug.

---

## 7. Training and initialisation

### 7.1 Parameter initialisation

All linear and embedding weights start from

$$
\mathcal{N}(0,\sigma^2), \qquad \sigma = 0.02,
$$

and all biases start at zero. This is the GPT-2 recipe. The residual output
projections (`out_proj` in attention, `fc_out` in the MLP) use a smaller
standard deviation:

$$
\sigma_{\text{res}} = \frac{\sigma}{\sqrt{2L}},
$$

where $L$ is the number of layers. The factor compensates for the fact that
each residual branch adds variance to the stream; without it, deeper models
start with very large activations and logits. Both choices are in
`MiniGPT._init_weights`.

### 7.2 The shared Trainer

The M4 phase does not implement its own training loop. It builds a
`PhaseBundle` and hands it to the shared Trainer (T-005), which owns:

- epoch and step iteration, with `max_steps` support;
- AdamW with the configured learning rate;
- logging, validation, best-checkpoint tracking and checkpoint saving;
- callbacks, including the `on_end` callback that writes GPT artifacts.

The phase factory supplies the model, the train/validation chunk datasets,
`language_model_loss` and the perplexity metric. `metric_fn` replaces the
Trainer's default evaluation only when a validation split exists; without
one, the Trainer's ordinary training-loss evaluation is used.

The tiny profile runs 400 steps with batch size 8, learning rate $3\times
10^{-3}$ and no learning-rate schedule. On CPU it finishes in roughly two
seconds. A real run should use a larger corpus, a larger model, gradient
accumulation and a warmup/decay schedule; those are M5 concerns, not
requirements for verifying the M4 objective.

### 7.3 Dropout and evaluation modes

Dropout is disabled by default in the tiny profile. When enabled it is
applied to attention weights, residual outputs, the MLP output and the
embedding sum. `evaluate_perplexity` switches the model to `eval()` and
restores its previous mode afterwards, so evaluation and generation are
deterministic (given the same weights). Forgetting that switch is a classic
source of noisy validation numbers.

---

## 8. Sampling: turning logits into text

The model produces one logit per vocabulary token. A sampler decides which
token to append, then the new sequence is fed back in. M4 supports four
controls; they can be combined, and the implementation applies them in the
order temperature → top-k → top-p.

### 8.1 Greedy decoding

Greedy decoding chooses

$$
w_t = \arg\max_w z_{t,w}.
$$

It is deterministic and cheap, and it is the limit of temperature sampling
as $T\to0$: the softmax becomes arbitrarily peaked on the argmax. The code
treats `temperature == 0` and `greedy=True` as the same path. Greedy text
often repeats itself because a small model's most likely continuation can
fall into a loop.

### 8.2 Temperature

Temperature divides the logits before the softmax:

$$
p_w(T) =
\frac{\exp(z_w/T)}
     {\sum_{v}\exp(z_v/T)}.
$$

- $T<1$ sharpens the distribution: likely tokens become more likely.
- $T=1$ is the model's raw distribution.
- $T>1$ flattens it: low-probability tokens get more mass, which increases
  diversity and also the chance of nonsense.
- $T\to0$ approaches greedy; $T\to\infty$ approaches uniform.

Temperature is applied before top-k and top-p, so those filters operate on
the tempered distribution.

### 8.3 Top-k

Top-k keeps the $k$ largest logits and sets every other logit to $-\infty$:

$$
\tilde z_w =
\begin{cases}
z_w & w \in \operatorname{topk}_k(z),\\
-\infty & \text{otherwise}.
\end{cases}
$$

After renormalisation, exactly $k$ tokens (or fewer if $k\ge V$) can be
sampled. `top_k=0` disables the filter, `top_k=1` is equivalent to greedy,
and a large $k$ is a no-op. Ties are resolved by `torch.topk`, which keeps
the result deterministic.

### 8.4 Top-p (nucleus) sampling

Top-p sorts tokens by probability and keeps the smallest set whose
cumulative probability reaches $p$:

$$
\mathcal{N}(p)
=
\min\left\{
S \subseteq \mathcal{V} :
\sum_{w\in S} p_w \ge p
\right\},
$$

where the minimum is over nested prefixes of the descending-probability
order. In implementation, a token is removed when the probability mass
*before* it is already at least $p$:

```python
remove = (cumulative - probabilities) >= p
```

This keeps the first token even when $p$ is extremely small, so the
distribution is never empty. `top_p=1` keeps everything. Unlike top-k, the
candidate count adapts: a confident model keeps one or two tokens, a flat
model keeps many.

### 8.5 The generation loop

After filtering, the remaining distribution is renormalised and one token is
drawn with `torch.multinomial`. The new token is appended to the sequence,
and the process repeats. Two engineering details:

- **Context cropping.** The model has a fixed `context_length`. At every
  step the code feeds only the newest `context_length` tokens:
  `input_ids[:, -context_length:]`. Generation can therefore continue past
  the training window, but the model cannot remember anything older than
  that window.
- **No KV cache.** This educational implementation recomputes attention for
  the whole window at every step, which is simple and obviously correct but
  costs $\mathcal{O}(T^2)$ per generated token. Production decoders cache
  keys and values and only compute the new row. That optimisation belongs
  to a later scaling step.

If `eos_id` is supplied, generation stops after appending that token. If
`seed` is supplied, a fresh `torch.Generator` is seeded on the model's
device, so repeated calls with the same prompt, weights and settings are
reproducible. Passing a generator instead lets a caller share one random
stream across calls.

Sampling is a design choice, not a correctness check: greedy is the most
faithful to the model's probability ranking, while temperature and nucleus
sampling trade likelihood for diversity. When evaluating whether a model
has learned anything, look at validation perplexity and greedy output first;
sampling settings can make an undertrained model look more or less plausible
without changing its underlying knowledge.

---

## 9. Configuration, artifacts and the UI

### 9.1 Tiny profile

`configs/gpt_tiny_debug.yaml` is the fast CPU profile used by tests and the
walkthrough:

| setting | value |
|---|---:|
| `context_length` | 16 |
| `n_layer` | 2 |
| `n_head` | 2 |
| `n_embd` | 32 |
| `dropout` | 0.0 |
| `tie_weights` | true |
| `vocab_size` | 300 (trained BPE reaches 259 on the sample corpus) |
| `batch_size` | 8 |
| `learning_rate` | 0.003 |
| `max_steps` | 400 |
| `val_split` | 0.1 |

The model's vocabulary is always taken from the tokenizer bundle, not from
the config's `vocab_size` directly: `build_gpt_model` uses
`tokenizer.bpe.vocab_size`. If a bundle is reused, its vocabulary wins; if a
new tokenizer is trained, `vocab_size` is a target for the BPE trainer.

### 9.2 Command line

```bash
python -m scripts.train --config configs/gpt_tiny_debug.yaml
```

The shared Trainer writes a run folder under `artifacts/`:

```text
artifacts/<timestamp>_<run_name>/
  config.yaml
  run.log
  run_meta.json
  checkpoints/
    step_000100.pt
    ...
    final.pt
  gpt/
    model.pt            # state dict
    model_config.json   # GPTConfig, including vocab/context/architecture
    meta.json           # format/version, counts, best val, perplexity
    corpus.txt          # cleaned corpus used for this run
  tokenizers/           # copy of the BPE/char bundle used for this run
```

`load_gpt_artifacts` accepts the run folder or its `gpt/` subfolder and
locates the copied tokenizer, so a run is self-contained for generation.
`find_gpt_dirs` returns discovered runs newest-first for the UI.

### 9.3 Generation UI

Start the app with `python -m scripts.ui` and open `/gpt`. The page:

- discovers every GPT run and loads the newest by default;
- shows vocabulary, context, parameter count, best validation loss and
  perplexity when recorded;
- accepts a prompt and exposes greedy, temperature, top-k, top-p and seed
  controls;
- displays the decoded continuation and the number of generated tokens;
- explains how to train a run when no artifact exists.

The page calls the same `generate` function as `core/gpt/sampling.py`; there
is no second sampler to drift out of sync.

---

## 10. Failure modes and diagnostics

### 10.1 The corpus is tiny

The bundled sample corpus is about 1.9 KB. After cleaning and BPE it yields
878 tokens, so the model has only 49 training windows at context 16. The
loss can fall and the model can memorise local patterns, but it cannot learn
German. Samples from the tiny run are word salad even when training is
numerically healthy. That is a data limitation, not evidence that attention
or the loss is broken.

**Diagnostic:** check the token/window counts in `meta.json` or the run log.
If the corpus is a few kilobytes, judge the code by loss curves, causality
tests and artifact round-trips, not by sample quality.

### 10.2 Perplexity is not comparable across tokenizers

Perplexity is per token. Changing the tokenizer changes both the number of
prediction problems and the size of the multiple-choice set. A character
model and a BPE model can have very different perplexities on the same text.
Compare runs only when the tokenizer, vocabulary and validation split agree.

### 10.3 Validation leakage from chunking

If windows are split randomly after chunking, adjacent windows share almost
all tokens and validation perplexity looks unrealistically good. This
implementation splits the token stream first and chunks each side
independently. If you modify the pipeline, preserve that ordering. A
contiguous suffix still shares document-level style with the training text,
so for a real generalisation estimate hold out whole documents.

### 10.4 Causal mask bugs

A missing or mis-sized mask lets position $t$ see the future. Training loss
can then become suspiciously low and samples can look like copies of the
prompt. The test suite checks three things: the mask is exactly lower
triangular, attention rows sum to one with zero mass on the future, and
changing a future token leaves earlier outputs unchanged. If you change the
attention code, rerun those tests before trusting a loss curve.

### 10.5 Context truncation and memory

Generation crops to the newest `context_length` tokens. The model literally
cannot condition on anything older, no matter how long the prompt is. A
prompt longer than the context is not an error; it is silently cropped to
the suffix. If long-range coherence matters, increase the context and
retrain; at inference time cropping cannot recover information the model
never had.

### 10.6 Sampling extremes

- Temperature near zero or `greedy=True` can loop.
- Very high temperature approaches uniform sampling and destroys fluency.
- `top_k=1` is greedy; `top_k` larger than the vocabulary is a no-op.
- `top_p` near zero keeps only the most likely token; `top_p=1` disables the
  nucleus.
- Top-k and top-p interact: if top-k already leaves one token, top-p has
  nothing left to remove.

The sampler validates its parameters and raises `ValueError` for negative
temperature, negative `top_k`, or `top_p` outside $(0,1]$. The UI turns
those errors into a notification instead of crashing the page.

### 10.7 Overfitting and the tiny validation set

Five validation windows are enough to log a number, not enough to estimate
generalisation. The Trainer records the best validation loss, and the save
callback records `best_val` and `val_perplexity`. On a tiny corpus the
training loss can continue falling while validation loss rises; that is
overfitting. Use `best_val` for comparisons and treat the final training
loss as an optimisation diagnostic only.

### 10.8 Numerical issues

- Causal masking uses $-\infty$ before the softmax. If every entry in a row
  were masked, the row would be undefined; the lower-triangular mask always
  keeps the diagonal, so this cannot happen.
- Perplexity is clamped at $\exp(50)$ to keep logs finite.
- Very small temperatures can make tempered logits huge and the softmax
  numerically sharp; in the limiting case the code takes the greedy path at
  exactly zero. Avoid extreme non-zero temperatures in experiments.
- Loading a checkpoint relies on `model_config.json`; if the architecture
  config and the state dict disagree, loading fails loudly rather than
  silently producing wrong shapes.

### 10.9 Weight tying and saved configuration

`tie_weights` changes the parameter count and the state-dict keys. The
artifact stores the full `GPTConfig`, and `load_gpt_artifacts` rebuilds the
model from it before loading weights, so a tied and an untied run remain
distinguishable. Do not edit `model_config.json` by hand.

### 10.10 No KV cache

Generation recomputes the full attention window for every new token. This is
correct but scales poorly with context and output length. It is fine for the
tiny model and for interactive demos; a production sampler should cache keys
and values, which changes performance but not the mathematical distribution.

---

## 11. What to verify in a run

A healthy tiny M4 run shows:

1. **Correct shapes and finite values.** A forward pass on `(B, T)` ids
   returns `(B, T, V)` logits with no NaNs.
2. **Causality.** Future tokens do not affect earlier logits, and attention
   rows have zero mass above the diagonal.
3. **A reasonable starting loss.** Near $\ln V$ (about 5.5 for the 259-token
   vocabulary) before training, then a clear fall. Exact values depend on
   initialisation and token frequencies.
4. **Validation perplexity logged each epoch.** The token-weighted value is
   computed over the held-out suffix, not by averaging batch perplexities.
5. **Artifacts that reload.** `model.pt`, `model_config.json`, `meta.json`
   and the copied tokenizer reconstruct a model that generates text.
6. **All sampling modes available.** Greedy, temperature, top-k and top-p
   produce tokens, and a fixed seed reproduces a sample.
7. **A working `/gpt` page.** A run can be selected, a prompt entered, and a
   continuation generated without network access.

The tiny run used for QA met these checks: 34,272 parameters, 49 train and 5
validation windows, 400 steps in about 1.6 s on CPU, final training loss
0.433, best validation loss 4.5193 (perplexity 91.77), and reloadable
artifacts. The low training loss and much higher validation loss are exactly
the overfitting pattern expected from a tiny corpus; they are not a claim of
fluent generation.

The full pipeline is intentionally inspectable: every equation in this note
maps to a function in `core/gpt/`, and every run records the config, corpus
and tokenizer that produced its weights.
