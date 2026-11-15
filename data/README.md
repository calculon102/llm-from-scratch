# Corpora

Small, self-contained text collections used by examples, tests and the
tokenizer UI live in this directory. The larger local German corpus used
for real training runs is kept under `data/corpus_de/` and is deliberately
**not checked into git** (it is too large and has its own upstream
licences).

## `sample_de/` — noisy German-first sample

Everything in `sample_de/` is **original text written for this
repository** and released under CC0-1.0 (public domain dedication), so it
can be copied, modified and redistributed without attribution. The point
of this corpus is not literary quality; it is to exercise the ingestion
pipeline's failure modes in miniature:

| File | Encoding / endings | Noise it demonstrates |
| --- | --- | --- |
| `sprache_und_schrift.txt` | UTF-8, LF | clean prose, umlauts, `ß`, long compounds |
| `stadtnotizen.txt` | UTF-8, LF | duplicate boilerplate line, blank lines, a URL, a one-word page number |
| `archiv_cp1252.txt` | Windows-1252, CRLF | legacy encoding, German quotation marks `„…“` |
| `rechtschreibung_bom.txt` | UTF-8 with BOM, CRLF | BOM and Windows line endings |

Run ingestion on it with:

```sh
.venv/bin/python scripts/tokenizer.py --config configs/tokenizer_tiny_debug.yaml
```

The run reports which encoding each file used and how many lines were
dropped as blank/short/duplicate; see `corpus/report.json` in the run
folder. Expected: all four files decode and the duplicate line in
`stadtnotizen.txt` is dropped. `Seite 14` is only 8 characters, so it
survives the default `min_line_chars: 2`; raise the threshold (e.g. to
12) to watch the short-line filter remove it.

## Fetching sources — `scripts.corpus` (T-601)

Corpus sources are registered in `core/corpus/sources.py` and fetched
into a local, gitignored cache (`data/raw/cache/`). One command both
downloads/copies the files and records them in
`data/corpus_manifest.json` under a `cache` section:

```sh
# Fetch the tiny bundled German sample (offline, default)
.venv/bin/python -m scripts.corpus fetch

# Fetch a specific source, or every registered source including the
# large opt-in dumps
.venv/bin/python -m scripts.corpus fetch --source gutenberg_de_catalog
.venv/bin/python -m scripts.corpus fetch --all

# Cache-only: fail if a source was never fetched
.venv/bin/python -m scripts.corpus fetch --offline

# Inspect the registry / what the manifest currently records
.venv/bin/python -m scripts.corpus list
.venv/bin/python -m scripts.corpus status
```

Every file is downloaded to a `.part` entry and renamed atomically;
when a source declares a size or SHA-256 it is verified before the cache
is updated, and a corrupt/truncated download is refused. The registry is
German-first and records the licence of each source. `wikipedia_de_dump`
(~8 GiB, CC BY-SA 4.0) and `gutenberg_de_catalog` are opt-in because they
are large; the default fetch uses only the CC0 sample.

## `corpus_de/` — local German training corpus

This directory is created locally by the corpus preparation step and is
ignored by git. It holds a German-first mixture large enough for the M5
profile ladder without turning the repository into a data dump:

```
corpus_de/
├── wikipedia/   cleaned article text from the German Wikipedia dump
├── gutenberg/   cleaned German Project Gutenberg books
└── tokenizer/   reusable char+BPE bundle trained on a small sample
```

The local copy contains **251 text files, ~266 MiB / 274.7M characters**:

| Part | Files | Size | Content |
| --- | ---: | ---: | --- |
| `wikipedia/` | 3 shards | ~205 MiB | 87,894 cleaned articles |
| `gutenberg/` | 248 books | ~61 MiB | German prose and drama |

The sources are:

- **German Wikipedia** — the `pages-articles5` part of the 2026-11-06
  dump (`dewiki-latest-pages-articles5.xml-p6115465p7615464.bz2`,
  Last-Modified 2026-11-06), cleaned with
  [WikiExtractor](https://github.com/attardi/wikiextractor). Every third
  article (deterministic CRC32 sampling) was staged until the Wikipedia
  share reached ~205 MiB. Wikipedia text is licensed **CC BY-SA 4.0**;
  attribution and share-alike apply to redistributed text.
- **Project Gutenberg** — the 2,416 German-language `Text` entries in
  `https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv`. 2,393 were
  downloaded as `pg<id>.txt` (23 returned 404 or a non-text page); the
  first 248 cleaned books were staged until the share reached ~61 MiB.
  Gutenberg texts are public domain in the United States; German copyright
  terms can differ, so check the author's death date (70 years p.m.a.)
  before redistributing.

`data/raw/` keeps the downloaded dumps, the catalogue and the per-book
files for rebuilding. Both `data/raw/` and `data/corpus_de/` are ignored
by git. `data/corpus_manifest.json` is tracked and records the exact
sizes, file counts, source URLs and licences.

### `corpus_de_quick/`

A ~8 MiB, two-file slice (`wikipedia/wiki_quick.txt` and
`gutenberg/gutenberg_quick.txt`) used by the CPU smoke-test config. It is
also gitignored.

### Tokenizer and training

`data/corpus_de/tokenizer/` is a 4096-token BPE bundle trained once on a
1 MiB sample (`data/raw/tokenizer_sample.txt`); both GPT configs reuse it.
Retraining it is only necessary when the corpus changes:

```bash
.venv/bin/python -m scripts.tokenizer --config configs/tokenizer_de.yaml
cp -r artifacts/<tokenizer-run>/tokenizers data/corpus_de/tokenizer
```

Corpus ingestion is deterministic: `core.tokenizer.corpus` handles UTF-8,
CP1252, BOMs, CRLF and duplicate lines, and reports what it dropped. Point
`phase_config.corpus_path` at `data/corpus_de` (or at the `wikipedia/` and
`gutenberg/` subdirectories separately) to train on it.

The ready-to-run configs are `configs/gpt_de_quick.yaml` (CPU smoke test)
and `configs/gpt_de_small.yaml` (the `small` rung; a GPU is strongly
recommended). See the top-level README for the commands and artifact
layout.
