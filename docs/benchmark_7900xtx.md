# M5 benchmark: scaling profiles on the 7900 XTX

This is the speed/quality comparison required by T-504. It has two layers:

1. **Model scale** for every rung of the ladder, computed analytically from
   the profile presets (`core/scaling.py`) — no GPU needed.
2. **Measured slices** from `scripts/benchmark.py`, including throughput and
   memory. The numbers below were captured on the development machine
   (CPU-only; no ROCm device was available). They are a reproducibility
   baseline, **not** 7900 XTX numbers. The exact GPU command is in
   [Running on the 7900 XTX](#running-on-the-7900-xtx). For a practical
   tuning walkthrough — VRAM budget, measured RX 7900 XTX numbers,
   batch/context trade-offs and the OOM playbook — see
   [`scaling_7900xtx.md`](scaling_7900xtx.md).

## Model scale (all profiles)

Target values from `core/scaling.py`; parameter counts assume the target BPE
vocabulary is reached and tied input/output embeddings.

| Profile | Parameters (target) | Layers | Width | Context | Vocab target | Batch | Max steps | Runtime budget | Capability |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| tiny_debug | 35,776 | 2 | 32 | 16 | 300 | 8 | 400 | seconds–minutes | shapes/smoke tests only |
| tiny_train | 934,144 | 4 | 128 | 64 | 1,024 | 16 | 2,000 | minutes | short toy completions |
| small | 12,325,632 | 6 | 384 | 256 | 4,096 | 16 | 10,000 | tens of minutes–1.5 h | narrow-domain baseline |
| medium | 29,688,832 | 8 | 512 | 512 | 8,192 | 8 | 20,000 | 1–4 h+ | better coherence |
| max_7900xtx | 98,452,992 | 12 | 768 | 1,024 | 16,384 | 4 | 50,000 | long run | best single-GPU attempt |

The 24 GB budget is comfortably sufficient for `max_7900xtx`: ~98 M
parameters need roughly 1.2 GB for fp32 weights plus two AdamW moments,
leaving room for the 1024-token attention activations. The binding
constraint is corpus size/quality (M6), not VRAM.

## Measured slices (development CPU)

Command (reproducible; writes `artifacts/benchmarks/<timestamp>/`):

```bash
python -m scripts.benchmark --profiles tiny_debug tiny_train --steps 100 --device cpu
```

Environment: `torch 2.15.0+cu132` (CPU execution), Python 3.11, AMD/ROCm
device **not available**. Corpus: bundled `data/sample_de` (1,760 cleaned
characters → 880 BPE tokens; the tokenizer therefore produces 259 tokens,
not the 300/1,024 targets). Throughput excludes evaluation/checkpoint time.

| Profile | Actual params | Context | Batch | Tokens/s | Steps/s | Val loss | Perplexity | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny_debug | 34,272 | 16 | 8 | 28,709 | 255.6 | 4.6103 | 100.52 | 1,001 MiB |
| tiny_train | 834,688 | 64 | 16 | 16,503 | 21.5 | 5.9323 | 377.01 | 1.04 GiB |

Reading the table:

- Throughput falls as context/width grow even at 100 steps; the useful
  comparison is **relative** (`tiny_train` is ~0.57× the tokens/s of
  `tiny_debug` on CPU at ~26× the parameters).
- Validation loss/perplexity after 100 steps is a sanity signal only. The
  bundled corpus is ~880 tokens and the run is ~100 optimizer steps — this
  cannot produce a quality claim for any profile.
- `small`, `medium` and `max_7900xtx` are reported as `skipped` against the
  bundled corpus because a 256/512/1024-token validation window does not
  fit. They need the larger M6 corpus (or `--corpus PATH`).

## Running on the 7900 XTX

Install the ROCm torch wheel (see the README), then run the same script on
the GPU. `--device auto`/`cuda` uses ROCm through torch's CUDA API, and
`--amp` is on by default for `small` and above.

```bash
# Smoke-fit every rung (needs a corpus large enough for 1024-token windows)
python -m scripts.benchmark --profiles all --corpus /path/to/full_de \
    --tokenizer-bundle /path/to/bundle --steps 20

# Longer, steadier measurement for the two profiles that fit VRAM comfortably
python -m scripts.benchmark --profiles small medium --corpus /path/to/full_de \
    --tokenizer-bundle /path/to/bundle --steps 200 --device cuda

# The largest rung; watch memory, then run its full max_steps for quality
python -m scripts.benchmark --profiles max_7900xtx --corpus /path/to/full_de \
    --tokenizer-bundle /path/to/bundle --steps 50
```

Pass `--tokenizer-bundle` on a full corpus: without it the benchmark trains
BPE from scratch in pure Python, which can dominate the measurement time.

Each run writes:

```
artifacts/benchmarks/<timestamp>/
├── benchmark.json   # exact results + device line + per-profile scale
├── report.md        # the table above
└── runs/<run>/      # normal artifact runs: config, run.log, metrics.jsonl, final.pt
```

Record the `Device:` line and the `gpu_peak_mb` / `gpu_total_mb` columns in
`benchmark.json` when publishing 7900 XTX numbers; those fields are null on
CPU.

## Using the table

- **Pick the rung by budget, not by hope.** `tiny_debug` is for code paths;
  `tiny_train` for watching loss move; `small` for a first real baseline.
- **Scale one axis at a time** (see `docs/model-complexity.md`): context
  length → batch size → model size. The benchmark makes the throughput cost
  of each change visible.
- **Quality needs full runs.** A benchmark slice validates speed/memory and
  that the pipeline runs; compare profile quality only after each profile's
  configured `max_steps`.
- **OOM?** The Trainer checkpoints first and prints the tailored fallback
  steps (T-503), including the next smaller profile and the resume command.
