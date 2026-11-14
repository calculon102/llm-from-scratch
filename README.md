# llm-from-scratch

Educational, Python-only, offline-capable LLM pipeline built step by step, with a minimal NiceGUI web UI.

Target hardware: a single AMD 7900 XTX (24 GB) via ROCm, with CPU fallback and an AMP toggle. The corpus is German-first (DE-first) and noisy; everything runs offline from a local cache.

## Milestones

- **M0** – Bootstrap & infrastructure (packaging, config, trainer, artifacts, UI shell)
- **M1** – XOR learning module (first end-to-end training loop)
- **M2** – Tokenizer (corpus ingestion, character + BPE from scratch)
- **M3** – Embeddings (skip-gram, negative sampling, explorer)
- **M4** – Mini GPT (decoder transformer, training, sampling, generation UI)
- **M5** – Scaling on the 7900 XTX (profiles, throughput, OOM fallbacks)
- **M6** – Capstone & docs (end-to-end pipeline, scaling guide, release checklist)

MVP = M0 + M1 + M2 + core M4. See [docs/roadmap.md](docs/roadmap.md) for details, [docs/tickets.md](docs/tickets.md) for the issue-sized backlog, and [docs/model-complexity.md](docs/model-complexity.md) for capability expectations.

## Project layout

```
core/          shared library (config, trainer, device, artifacts, tokenizer, ...)
apps/ui/       NiceGUI web app (dark, responsive)
configs/       YAML config profiles (tiny_debug → max_7900xtx)
data/          bundled sample corpora (German-first, offline)
scripts/       CLI entrypoints
artifacts/     per-run outputs (config snapshot, logs, checkpoints, plots)
docs/math/     per-phase math notes
tests/         test suite
```

## Getting started

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[viz]"   # plotting / projection (matplotlib, scikit-learn, umap-learn)
pip install -e ".[dev]"   # development (pytest)
```

### AMD ROCm (7900 XTX)

The default PyPI `torch` wheels ship CUDA builds. For ROCm, install the matching ROCm wheel instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

The device manager (T-006) auto-detects ROCm/CPU at startup and logs the chosen device and precision.

## Configuration

Runs are driven by YAML profiles passed via `--config`. Every config is
validated against a schema (unknown keys, wrong types, out-of-range values,
duplicate keys, and malformed YAML all produce a clear, aggregated error).
Missing optional keys are filled with defaults.

```bash
python -m scripts.check_config --config configs/tiny_debug.yaml
```

Common fields: `run_name`, `seed`, `device` (`auto`/`cpu`/`cuda`), `amp`,
`profile` (ladder `tiny_debug → tiny_train → small → medium → max_7900xtx`),
`output_dir`, `log_interval`, `batch_size`, `learning_rate`, `epochs`,
`max_steps`, `val_split`, `save_interval`, `phase` (`xor`/`tokenizer`/
`embeddings`/`gpt`), and `phase_config` (phase-specific settings, validated
by the phase module). Setting `profile_defaults: true` fills missing keys
from the named profile preset (explicit YAML values still win). See
`configs/tiny_debug.yaml` for a sample.

## Running

Training runs are driven by YAML profiles and the shared Trainer loop
(T-005):

```bash
python -m scripts.train --config configs/xxx.yaml [--resume PATH]
```

The config's `phase` (`xor`/`tokenizer`/`embeddings`/`gpt`) selects the
phase module, which registers a factory building model + datasets +
objective (`core.trainer.register_phase`). The Trainer owns the loop,
logging, evaluation, and checkpointing; `--resume` continues from a
checkpoint file or a previous run directory (latest checkpoint used).

Every run is written to a dedicated folder under `artifacts/` (see
`core/artifacts.py`):

```
artifacts/
├── runs.jsonl                              # index of all runs
└── 20261102-224000_<run_name>/
    ├── config.yaml      # snapshot of the resolved config
    ├── run.log          # run log (also mirrored to the console)
    ├── run_meta.json    # status, start/end times
    ├── checkpoints/     # step_000123.pt, final.pt
    └── plots/           # PNG figures
```

A demo run that exercises the artifact manager end to end:

```bash
python -m scripts.demo_run --config configs/tiny_debug.yaml
```

### Tokenizer (M2)

Ingest a noisy German-first corpus, train a character tokenizer and a BPE
tokenizer from scratch, persist `vocab.json`/`merges.json`, and log
compression/OOV metrics:

```bash
python -m scripts.tokenizer --config configs/tokenizer_tiny_debug.yaml
python -m scripts.tokenizer --config configs/tokenizer_tiny_debug.yaml \
    --input data/sample_de --vocab-size 800
```

The run folder gets `corpus/cleaned.txt`, `corpus/report.json`,
`tokenizers/{char,bpe}/`, and `metrics.json`. The bundled
`data/sample_de/` files are original CC0 German text with deliberate
noise (mixed encodings, BOM, CRLF, duplicates); see `data/README.md`.
The math and trade-offs are written up in
[docs/math/phase2_tokenizer.md](docs/math/phase2_tokenizer.md).

### Embeddings (M3)

Train skip-gram word vectors with negative sampling on the cleaned corpus.
The run reuses a saved tokenizer bundle when one is configured, exports the
centre embedding table, and keeps the tokenizer/corpus for reproducibility:

```bash
python -m scripts.train --config configs/embeddings_tiny_debug.yaml
python -m scripts.train --config configs/embeddings_small.yaml
```

The run folder gets `embeddings/vectors.npy`, `embeddings/vocab.json`,
`embeddings/meta.json`, and a copy of the tokenizer under `tokenizers/`.
Query the learned space or draw a 2-D projection from the command line:

```bash
python -m scripts.embeddings neighbors --token Katze --top-k 10
python -m scripts.embeddings project --method tsne
```

Both commands use the newest embedding run by default; pass `--run PATH` to
select another. The objective, cosine geometry, projection trade-offs and
failure modes are documented in
[docs/math/phase3_embeddings.md](docs/math/phase3_embeddings.md).

### Mini GPT (M4)

Train the decoder-only transformer on contiguous LM chunks, with causal
self-attention, next-token cross-entropy and validation perplexity:

```bash
python -m scripts.train --config configs/gpt_tiny_debug.yaml
```

The tiny profile runs on CPU in seconds. The run folder gets
`gpt/model.pt`, `gpt/model_config.json`, `gpt/meta.json`, `gpt/corpus.txt`,
and a copy of the tokenizer under `tokenizers/`. Load an artifact and
generate text with greedy, temperature, top-k or top-p sampling:

```python
from core.gpt import generate_text, load_gpt_artifacts

art = load_gpt_artifacts("artifacts/<run>")
print(generate_text(art.model, art.tokenizer, "Die", max_new_tokens=40))
```

The architecture, causal mask, LM objective, perplexity and sampling math
are documented in
[docs/math/phase4_transformer.md](docs/math/phase4_transformer.md).

### Scaling on the 7900 XTX (M5)

Profiles follow the ladder `tiny_debug → tiny_train → small → medium →
max_7900xtx`. The preset values live in `core/scaling.py`; changing the
single `profile:` line in `configs/gpt_scaling.yaml` switches architecture,
batch size, learning rate, context length and run length together. Thin
per-rung configs are available as `configs/gpt_tiny_train.yaml`,
`gpt_small.yaml`, `gpt_medium.yaml` and `gpt_max_7900xtx.yaml`:

```bash
python -m scripts.train --config configs/gpt_scaling.yaml
# edit profile: tiny_debug -> small, then rerun the same command
```

Every logged step reports tokens/sec and memory; the run folder additionally
gets an append-only `metrics.jsonl` with throughput and CPU/GPU memory
snapshots. Benchmark a profile (or the whole ladder) with:

```bash
python -m scripts.benchmark --profiles tiny_debug tiny_train --steps 100
python -m scripts.benchmark --profiles all --corpus /path/to/full_de --steps 20
```

The benchmark writes `benchmark.json` and a Markdown speed/quality table
under `artifacts/benchmarks/<timestamp>/`. See
[docs/benchmark_7900xtx.md](docs/benchmark_7900xtx.md) for the report and
the 7900 XTX reproduction commands. OOM failures checkpoint the run and print
tailored fallback steps (smaller batch/context, AMP, next profile, resume);
Ctrl+C, SIGTERM and the UI Stop button also leave a resumable checkpoint.

## Web UI

The NiceGUI app shell (T-007) is dark by default and responsive (the left
drawer collapses to an off-canvas overlay on mobile, opened via the menu
button). It has one route per pipeline phase:

```
/            home / dashboard (M0)
/xor         M1 – XOR learning module (real page)
/tokenizer   M2 – character vs. BPE compare page (real page)
/embeddings  M3 – embeddings explorer (real page)
/gpt         M4 – mini GPT generation playground (real page)
/scaling     M5 – scaling on the 7900 XTX
/capstone    M6 – capstone & docs
```

Start it with:

```bash
python -m scripts.ui [--host 127.0.0.1] [--port 8080] [--reload]
```

The XOR, tokenizer, embeddings and GPT pages are real; the remaining phase
pages are shells for now (phase overview + ticket list). Phase-specific UI
lands in the later tickets (T-604).

## License

MIT — see [LICENSE](LICENSE).
