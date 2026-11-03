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

## Web UI

The NiceGUI app shell (T-007) is dark by default and responsive (the left
drawer collapses to an off-canvas overlay on mobile, opened via the menu
button). It has one route per pipeline phase:

```
/            home / dashboard (M0)
/xor         M1 – XOR learning module
/tokenizer   M2 – tokenizer module
/embeddings  M3 – embeddings module
/gpt         M4 – mini GPT module
/scaling     M5 – scaling on the 7900 XTX
/capstone    M6 – capstone & docs
```

Start it with:

```bash
python -m scripts.ui [--host 127.0.0.1] [--port 8080] [--reload]
```

The phase pages are shells for now (phase overview + ticket list); the
phase-specific UI lands in the later tickets (T-104, T-205, T-305, T-406,
T-604).

## License

MIT — see [LICENSE](LICENSE).
