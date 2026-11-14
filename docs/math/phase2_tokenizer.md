# Phase 2 — Tokenization: from noisy German bytes to BPE ids

This note explains the math and engineering behind the M2 tokenizer phase:
how raw corpus files become clean text, how a character tokenizer assigns
ids, how the from-scratch BPE trainer chooses merges, and which trade-offs
those choices buy. It is the conceptual companion to:

- `core/tokenizer/corpus.py` — decoding, Unicode cleanup, line filters,
  deduplication, per-file ingestion reports (T-201).
- `core/tokenizer/char.py` — deterministic character vocabulary and
  exact encode/decode (T-202).
- `core/tokenizer/bpe.py` — BPE training, merge persistence and encoding
  (T-203).
- `core/tokenizer/metrics.py` — compression, tokens/word, OOV and
  round-trip metrics (T-204).
- `core/tokenizer/pipeline.py` — bundle save/load and config glue.
- `scripts/tokenizer.py` — the CLI that writes corpus, tokenizers and
  `metrics.json` into one artifact run.
- `apps/ui/tokenizer_page.py` — the side-by-side compare page (T-205).

The phase is CPU-only and offline. A tokenizer is not a neural network:
there is no gradient descent here, only counting, sorting and merge
bookkeeping. That makes every result reproducible and every failure mode
inspectable.

---

## 1. Why tokenization is a modelling decision

A language model consumes a sequence of integer ids. The tokenizer is the
fixed map

$$
\text{text} \;\xrightarrow{\;\text{encode}\;}\; (i_1,\dots,i_T)
\;\xrightarrow{\;\text{decode}\;}\; \text{text}.
$$

The model never sees characters or bytes. If the tokenizer cuts German
`Straße` into six character tokens, the model must learn spelling and
word identity separately. If it cuts the word into one token, the model
gets a direct symbol for the concept but needs a much larger vocabulary to
cover the language. Tokenization chooses where on that spectrum we sit.

Three quantities drive the choice:

- **sequence length** $T$: attention costs grow with $T^2$ during
  training, so fewer tokens per document means cheaper training and a
  longer effective context;
- **vocabulary size** $|V|$: embedding and output matrices grow linearly
  with $|V|$, and rare tokens receive few gradient updates;
- **coverage**: text at inference time may contain characters or words the
  tokenizer never saw. Those become `<unk>`, and information is lost.

Character tokenization optimizes coverage and a tiny vocabulary, BPE
optimizes sequence length by learning frequent multi-character pieces.
This phase implements both so the trade-off can be measured rather than
assumed.

---

## 2. Corpus ingestion: making noisy bytes safe to tokenize

Real German corpora are not clean UTF-8 strings. They contain BOMs, mixed
line endings, CP1252 exports, zero-width characters, duplicated boilerplate
and binary files with a `.txt` suffix. Feeding those directly to a
tokenizer creates phantom vocabulary entries (`Ã¤` instead of `ä`) and
inflates the corpus with duplicates.

### 2.1 Decoding

For each file, `decode_bytes` tries the configured encodings in order:

| Order | Codec | Purpose |
|---:|---|---|
| 1 | `utf-8-sig` | UTF-8 and UTF-8-with-BOM; the BOM is stripped |
| 2 | `utf-8` | plain UTF-8 without BOM |
| 3 | `cp1252` | common Windows German export (umlauts, `ß`, curly quotes) |
| 4 | `latin-1` | total byte fallback; every byte maps to a character |

A strict decode is attempted for each. If all strict attempts fail, the
file is decoded as UTF-8 with `errors="replace"` and the number of U+FFFD
replacement characters is recorded. The chosen codec, the replacement count
and whether the decoder fell back are all written to the per-file report.

The order matters: `utf-8-sig` first means a BOM never becomes a token.
`latin-1` last is a deliberate safety net, but it can mis-decode a broken
UTF-8 file into plausible-looking mojibake. That is why the report keeps
the encoding and the CLI logs every non-UTF-8 file.

Binary detection runs before decoding: a NUL byte or more than 10% C0
control bytes in the first 8 KiB marks the file as binary and it is
skipped. This prevents compiled files, images or archives from polluting
the vocabulary.

### 2.2 Normalization

After decoding, the text is normalized:

1. **NFC** (`unicodedata.normalize("NFC", text)`). Canonically equivalent
   sequences such as `a` + combining diaeresis and precomposed `ä` become
   one code point. Without this, the same visible character could occupy
   two vocabulary slots.
2. **Line endings** `\r\n` and `\r` become `\n`.
3. **Format characters** (`Unicode category Cf`): BOM and soft hyphen are
   deleted; zero-width spaces and bidi controls become a space, so
   `A\u200bB` does not silently turn into `AB`.
4. **Control characters** (`Cc`) and space-like characters (NBSP, narrow
   NBSP, ideographic space, …) become a plain space.
5. **Horizontal whitespace runs** collapse to one space; newlines are
   preserved.

The space normalization is not cosmetic: BPE's chunks are separated by
whitespace, and two visually identical spaces must have the same id.

### 2.3 Line filters

Each line is stripped and then filtered by `CorpusOptions`:

- empty line → dropped (`blank_dropped`);
- length below `min_line_chars` → dropped (`short_dropped`);
- longer than `max_line_chars` → truncated, then re-checked
  (`truncated_lines`);
- more than 5% U+FFFD replacement characters → dropped as junk
  (`junk_dropped`);
- alphabetic fraction below `min_alpha_ratio` → dropped
  (`low_alpha_dropped`), where

$$
\alpha(\ell) = \frac{\#\{\,c \in \ell : c \text{ is alphabetic}\,\}}{|\ell|}.
$$

- duplicate line → dropped (`duplicate_dropped`).

Duplicate detection uses `line.casefold()` as the key, so `Hallo` and
`HALLO` count as the same line. If `lowercase: true`, the kept line is
lowercased; otherwise the first occurrence's casing survives. Dedupe is
applied **within a file and across files**, using one global `seen` set:
boilerplate copied into every file is counted once.

Line order is preserved. Paragraph breaks disappear as blank lines, but
line boundaries remain, which is enough structure for a first tokenizer.
The cleaned text ends with a newline when non-empty, and
`chars_out` includes that newline.

A file that is binary, unreadable, empty, or cleans down to nothing is
recorded as `skipped` with a reason; ingestion continues. The run fails
only if no file produces usable text — an empty corpus cannot train a
tokenizer.

---

## 3. Character tokenizer

### 3.1 Vocabulary construction

The character tokenizer reserves four special ids:

| Id | Token | Meaning |
|---:|---|---|
| 0 | `<pad>` | padding / batch alignment |
| 1 | `<unk>` | unknown character |
| 2 | `<bos>` | beginning of sequence |
| 3 | `<eos>` | end of sequence |

All remaining ids are assigned to characters **sorted by Unicode code
point**:

$$
V = (\,\texttt{<pad>},\texttt{<unk>},\texttt{<bos>},\texttt{<eos>},
c_1,c_2,\dots,c_K\,), \qquad c_1 < c_2 < \dots < c_K.
$$

Sorting by code point, not by locale or first appearance, makes the
vocabulary deterministic across machines, Python hash seeds and corpus
file order. `min_frequency` can drop rare characters, and `extra_chars`
can force characters into the vocabulary even if the training text did not
contain them.

### 3.2 Encoding and decoding

For a character $c$ with id $v(c)$,

$$
\text{encode}(s) = \big(v(s_1),\dots,v(s_n)\big),
\qquad
\text{decode}(i_1,\dots,i_n) = s_{i_1}\cdots s_{i_n}.
$$

Encoding an unseen character yields `<unk>` unless `strict=True`, in which
case it raises `TokenizerError`. Decoding an `<unk>` yields U+FFFD by
default; `errors="strict"` raises, `errors="ignore"` drops it.

The round-trip guarantee is immediate for text whose characters are all in
the vocabulary:

$$
\text{decode}(\text{encode}(s)) = s.
$$

For text with unknown characters, the guarantee is intentionally broken:
the tokenizer reports `roundtrip_ok = False` rather than pretending that
`<unk>` can be recovered.

### 3.3 What a character tokenizer buys and costs

- **Buys**: smallest possible vocabulary, no word-boundary assumptions,
  robust to typos and novel compounds, exact coverage of every character
  seen during training.
- **Costs**: one token per character, so a 100-character sentence is 100
  tokens. The model must learn word identity from long sequences, and
  attention/memory costs are paid per character.

On the bundled sample corpus the character vocabulary is 74 tokens
(4 specials + 70 characters), and every character of the sample text costs
exactly one token: **1.00 chars/token**.

---

## 4. BPE: learning frequent pieces

Byte pair encoding (BPE) starts from the character alphabet and repeatedly
merges the most frequent adjacent pair into a new token. The result is a
vocabulary of characters plus frequently occurring character sequences.

### 4.1 Pre-tokenization

BPE does not run over the raw string. The text is split into chunks with

```python
CHUNK_RE = re.compile(r"\s+|\w+|[^\w\s]+", re.UNICODE)
```

Three alternatives cover every character:

- `\s+` — whitespace runs (spaces, tabs, newlines);
- `\w+` — Unicode word runs; German umlauts and `ß` are word characters;
- `[^\w\s]+` — punctuation runs.

The chunks concatenate back to the original text exactly. Merges are
learned **within chunks only**; a pair that spans two chunks is never
merged. This is deliberate:

- spaces stay separate tokens, so the model can see word boundaries and
  the tokenizer never invents `word␣word` tokens that mix two words;
- the number of pair candidates stays manageable;
- encoding is stable: a word's tokenization does not depend on the
  sentence around it.

### 4.2 Training algorithm

Let the corpus be a multiset of chunks. After pre-tokenization, each unique
chunk is stored once with its frequency $f_w$:

$$
\text{words} = \{(w, f_w)\}.
$$

The initial segmentation of each chunk is its character sequence. The
training loop is:

1. **Count adjacent pairs** in every chunk, weighted by chunk frequency:

   $$
   c(a,b) = \sum_{w} f_w \cdot \#\{\,i : w_i = a,\ w_{i+1} = b\,\}.
   $$

   The count includes overlapping occurrences (`aaa` contributes two
   `aa` pairs). The merge itself is applied left to right and does not
   overlap.

2. **Filter candidates**:
   - count at least `min_frequency`;
   - merged length at most `max_token_length`;
   - the merged string is not already in the vocabulary.

3. **Select the best pair** by highest count, with a deterministic
   lexicographic tie-break:

   $$
   (a,b)^* = \operatorname*{arg\,min}_{(a,b)}
   \big(-c(a,b),\; (a,b)\big).
   $$

   Sorting by the pair tuple means the same corpus always produces the
   same merges, regardless of dictionary iteration order or hash seed.

4. **Merge** every non-overlapping occurrence of $(a,b)^*$, left to right,
   and add $a\!+\!b$ to the vocabulary with the next id.

5. **Repeat** until one of three stop conditions:
   - vocabulary reaches `vocab_size` (`stop_reason=target_vocab_size`);
   - no pair reaches `min_frequency` (`stop_reason=min_frequency`);
   - all frequent candidates exceed `max_token_length`
     (`stop_reason=max_token_length`).

The vocabulary size after $M$ merges is

$$
|V| = \underbrace{|S|}_{4 \text{ specials}} + \underbrace{|A|}_{\text{character alphabet}} + M.
$$

`vocab_size` includes specials and the alphabet. If it is smaller than
$|S|+|A|$, training raises instead of silently dropping characters.

### 4.3 Worked example

On the bundled German sample with `vocab_size: 300`, `min_frequency: 2`,
`max_token_length: 16`, the trainer starts from 74 characters + 4 specials
and learns 185 merges before no pair reaches frequency 2. The first ten
merges are:

```text
e+n → en        e+r → er        c+h → ch        e+i → ei
i+e → ie        u+n → un        s+t → st        t+e → te
i+n → in        un+d → und
```

The last one is the payoff: `und`, one of the most frequent German words,
becomes a single token after nine earlier merges built its pieces. The
BPE vocabulary ends at 259 tokens (target 300 was not reached because the
corpus is small; `stop_reason=min_frequency`).

### 4.4 Encoding with learned merges

Training records an ordered list of merge rules. Encoding applies them
greedily by **rank**, the position in that list:

1. split the text into chunks;
2. start with the character symbols of each chunk;
3. find the adjacent pair whose merge has the smallest rank;
4. replace it with the concatenated token;
5. repeat until no learned pair is adjacent.

Because every merge rule adds exactly one new token, the merge list plus
the character alphabet reconstructs the whole vocabulary. The rank order
is what makes decoding unambiguous: the same rule order that built a token
during training is used to build it at inference time.

Decoding is the simplest part:

$$
\text{decode}(i_1,\dots,i_T) = \text{concat}(v^{-1}(i_1),\dots,v^{-1}(i_T)).
$$

There is no separator to insert because every token already contains the
whitespace it was trained with — or, more precisely, whitespace chunks are
their own tokens and are concatenated like any other token.

### 4.5 Persistence: vocab + merges must round-trip

A BPE tokenizer is two ordered artifacts:

- `vocab.json` — `token -> id`, ids contiguous from 0, specials first;
- `merges.json` — ordered `(left, right)` pairs, exactly the training
  order;
- `meta.json` — format name/version, special tokens, vocabulary size,
  merge count, `max_token_length`, training statistics.

Loading validates the invariants: ids are contiguous, specials occupy
0–3, merges are unique, both merge parts and the merged token exist in the
vocabulary, and `max_token_length >= 2`. A corrupted or hand-edited bundle
fails loudly instead of producing silently different tokenizations. The
saved merge order is preserved, which is required because rank determines
encoding.

`TokenizerBundle` stores the character tokenizer and the BPE tokenizer
together under `char/` and `bpe/`, plus a `bundle.json` that points at
both. The CLI writes this bundle into the run's `tokenizers/` folder; the
UI page can load any bundle found under `artifacts/*/tokenizers/`.

### 4.6 Complexity

Let $N$ be the number of characters, $C$ the number of unique chunks, and
$M$ the number of merges. A straightforward implementation (the one here)
recomputes pair counts and resegments the chunk multiset after every
merge:

- each training iteration costs roughly $O(N)$ in the worst case,
- total training is $O(MN)$,
- memory is $O(N + |V|)$.

That is fine for an educational corpus of a few megabytes and even for
hundreds of megabytes with patience. Production tokenizers use incremental
pair-count updates and priority queues, but the algorithm and the resulting
merge order are the same.

Encoding scans each chunk repeatedly for the best-ranked adjacent pair; a
chunk of length $k$ costs $O(k^2)$ in the worst case. Since chunks are
short (words and punctuation runs), this is negligible in practice.

---

## 5. Metrics: what "good" means

The metrics module compares both tokenizers on the same text. Let $C$ be
the number of characters, $W$ the number of word matches (`\w+`), and $T$
the number of tokens produced.

| Metric | Formula | Reading |
|---|---|---|
| compression ratio | $C/T$ | characters carried by one token; higher is better |
| tokens per word | $T/W$ | fertility; lower is better |
| average token length | $\frac{1}{T}\sum_t |t|$ | how much text a typical token holds |
| unique-token ratio | $\#\{\text{unique ids}\}/T$ | lexical diversity of the encoded sequence |
| OOV characters | $\#\{c : c \notin V_{\text{chars}}\}$ | characters the tokenizer cannot represent |
| unk rate | $n_{\texttt{<unk>}}/T$ | share of tokens that lost information |
| round-trip | `decode(encode(x)) == x` | exactness of the pipeline |

For the sample run:

| Tokenizer | Vocabulary | Tokens | Chars/token | Tokens/word | OOV | Round-trip |
|---|---:|---:|---:|---:|---:|---|
| Character | 74 | 1760 | 1.00 | 7.21 | 0 | ok |
| BPE | 259 | 878 | 2.00 | 3.60 | 0 | ok |

BPE produces **2.00× fewer tokens** (50.1% fewer), so the same text fits
in about half the context. The cost is a vocabulary 3.5× larger and a
dependency on the training corpus: a character the BPE alphabet never saw
is still an OOV, and a word absent from training is split into whatever
pieces the merges provide.

**Caveat.** BPE merges are learned on the training corpus, so measuring
compression on that same corpus is mildly optimistic. The CLI measures on
the cleaned corpus and logs the sample size; the UI measures on arbitrary
user text, which is an honest held-out estimate. A rigorous evaluation
would train on a train split and report on a disjoint validation split.

---

## 6. Trade-offs and alternatives

| Choice | Character | BPE (this phase) | Word-level |
|---|---|---|---|
| Vocabulary | tiny (alphabet) | medium (alphabet + merges) | huge (all words) |
| Sequence length | longest | medium | shortest |
| OOV behaviour | only unseen characters | unseen chars + split words | every unseen word |
| Morphology | none | frequent subwords | none |
| German compounds | fine (characters) | good (shared pieces) | poor |
| Implementation | trivial | counting + merges | dictionary lookup |
| Typical use | debugging, tiny models | LLM default | classic NLP |

BPE sits between characters and words. It handles German compounding
(`Fußgängerüberweg` can reuse `Fuß`, `gänger`, `weg`) and inflections
without a morphology engine, while keeping the vocabulary bounded. The
main knobs are:

- **`vocab_size`** — larger means fewer tokens per document but more
  embedding parameters and rarer updates per token. For a tiny debug
  corpus 300 is already larger than the data can support; real runs use
  thousands.
- **`min_frequency`** — 1 lets a pair seen once become a token (overfits
  the corpus), 2 is a conservative default; higher values keep only
  genuinely frequent pieces.
- **`max_token_length`** — caps absurdly long merged tokens (a whole
  repeated line could otherwise become one token). It is measured in
  Unicode code points.
- **pre-tokenization** — merging across word boundaries would shorten
  sequences further but destroys the word-level structure the model can
  use, and makes the tokenizer sensitive to spacing.

---

## 7. Failure modes

| Symptom | Likely cause | Fix / signal |
|---|---|---|
| `Ã¤`, `Ã¶`, `ÃŸ` in the vocabulary | UTF-8 decoded as CP1252/latin-1 | check `encoding` in `report.json`; the CLI logs non-UTF-8 decodes |
| Two ids for the same visible `ä` | no NFC before vocabulary build | NFC is applied in `clean_text`; never bypass it |
| BPE vocabulary smaller than requested | corpus too small / `min_frequency` too high | `training_stats.stop_reason` says `min_frequency` |
| BPE vocabulary larger than expected | merges add exactly one token each; `vocab_size` includes specials + alphabet | compare `base_vocab_size` and `merges` in stats |
| Round-trip fails on user text | character not in the training alphabet | `oov_chars` lists it; add data or accept `<unk>` |
| Duplicate boilerplate dominates | dedupe disabled or line differs by case/whitespace | `drop_duplicates: true`; inspect `duplicate_dropped` |
| Huge memory during training | corpus too large for the naive $O(MN)$ loop | use a smaller sample for iteration; production needs incremental counts |
| Train/validation leakage | tokenizer trained on the full corpus including validation text | train on the train split only; metrics then become honest |
| Whitespace tokens look empty in the UI | spaces are real tokens | the UI renders `␣`, `\n`, `\t` escaped |
| `vocab_size` rejected | target below specials + alphabet | error reports the required base size |

The last point is worth repeating: BPE can only encode characters that
were in its training alphabet. `roundtrip_ok` is therefore a **coverage**
test, not a formality.

---

## 8. Configuration and commands

The tokenizer phase config (validated by `TOKENIZER_PHASE_SCHEMA`):

| Key | Default | Meaning |
|---|---|---|
| `corpus_path` | `data/sample_de` | file or directory, relative to the repo root |
| `vocab_size` | 500 | specials + alphabet + merges |
| `min_frequency` | 2 | minimum pair count to merge |
| `max_token_length` | 32 | maximum merged token length (code points) |
| `min_line_chars` | 2 | drop shorter lines |
| `max_line_chars` | 2000 | truncate longer lines |
| `min_alpha_ratio` | 0.0 | minimum alphabetic fraction per line |
| `drop_duplicates` | true | dedupe lines within and across files |
| `lowercase` | false | lowercase kept lines |
| `encodings` | utf-8-sig, utf-8, cp1252, latin-1 | decode fallback order |
| `extensions` | `.txt` | directory scan suffixes |
| `metrics_sample_chars` | 4000 | cap for metric text (0 = all) |

Run the full pipeline:

```bash
python -m scripts.tokenizer --config configs/tokenizer_tiny_debug.yaml
python -m scripts.tokenizer --config configs/tokenizer_tiny_debug.yaml \
    --input data/sample_de --vocab-size 800
```

The run folder contains:

```text
<run>/
├── corpus/cleaned.txt      # cleaned, deduplicated text
├── corpus/report.json      # per-file encoding + filter counters
├── tokenizers/bundle.json  # char + BPE bundle pointer
├── tokenizers/char/        # vocab.json, meta.json
├── tokenizers/bpe/         # vocab.json, merges.json, meta.json
├── metrics.json            # char vs BPE comparison
└── run.log                 # startup, stats, first merges, metrics
```

The web UI at `/tokenizer` exposes the same comparison interactively:
choose the bundled sample or an artifact bundle, edit the text, and press
**Tokenize**.

---

## 9. Summary

The phase turns bytes into ids in four deterministic stages:

$$
\text{bytes}
\;\xrightarrow{\text{decode + NFC + filters}}\;
\text{clean text}
\;\xrightarrow{\text{char/Train BPE}}\;
(V, \text{merges})
\;\xrightarrow{\text{encode}}\;
(i_1,\dots,i_T).
$$

The character tokenizer proves the round-trip property with the smallest
possible vocabulary. The BPE trainer adds the merge rule

$$
(a,b)^* = \operatorname*{arg\,min}_{(a,b)}
\big(-c(a,b),\,(a,b)\big),
\qquad
V \leftarrow V \cup \{a\!+\!b\},
$$

which is just "merge the most frequent adjacent pair, break ties
lexicographically". Repeating that rule until the vocabulary target or the
frequency floor is reached gives a compact, coverage-aware tokenizer whose
merge order can be saved, reloaded and applied exactly. The measured
result on the bundled corpus — 2.00× fewer tokens than characters with
zero OOV — is the empirical case for using BPE as the M3/M4 input stage.
