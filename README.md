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
core/          shared library (config, trainer, device, artifacts, ...)
apps/ui/       NiceGUI web app (dark, responsive)
configs/       YAML config profiles (tiny_debug → max_7900xtx)
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
by the phase module). See `configs/tiny_debug.yaml` for a sample.

## Running

Once the trainer (T-005) lands, runs are driven by YAML profiles:

```bash
python -m scripts.train --config configs/xxx.yaml
```

Every run is written to a dedicated folder under `artifacts/`.

## License

MIT — see [LICENSE](LICENSE).
