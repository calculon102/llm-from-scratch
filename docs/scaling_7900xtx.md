# Scaling on the AMD Radeon RX 7900 XTX (24 GB)

This is the practical companion to the M5 benchmark report
([`benchmark_7900xtx.md`](benchmark_7900xtx.md)). The benchmark answers
*"what do the profiles cost?"*; this guide answers *"what do I change when
the GPU is full, slow, or idle?"* It is written for this repository's
`MiniGPT` + shared `Trainer` exactly as implemented, not for a generic
transformer stack.

> **Measurement status.** The M5 benchmark report's slices were captured on
> CPU; no ROCm device was available during that ticket. The `max_7900xtx`
> run summarized below did execute on a Radeon RX 7900 XTX, and the
> quick GPU commands in this guide were re-run on the same machine. Record
> your own numbers with `scripts/benchmark.py` before quoting throughput;
> thermals, driver versions and background GPU load all move the result.

## 1. Confirm the GPU is actually being used

Install the ROCm torch wheel described in the README, then check the
runtime rather than trusting the package name:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("device:", torch.cuda.get_device_name(0))
    print("VRAM:", round(props.total_memory / 1024**3, 2), "GiB")
PY
```

Expected: a HIP version, `GPU available: True`, about `23.98 GiB`, and a
Radeon device. ROCm may report the generic `AMD Radeon Graphics` as the
torch device name even when `rocm-smi --showproductname` identifies the card
as a Radeon RX 7900 XTX (`gfx1100`). The Trainer logs the same information
at startup (T-006), for example:

```
device=cuda:0 backend=rocm name='AMD Radeon Graphics'
vram=24560 MiB total / 24486 MiB free precision=bf16 (AMP)
```

If `torch.cuda.is_available()` is `False`:

1. Match the wheel to the installed ROCm version (`rocminfo | grep -i rocm`
   and the wheel's `rocmX.Y` tag). Mixing versions is the most common cause.
2. Check that your user is in the `video` and `render` groups and re-login.
3. Run `rocm-smi` and `rocminfo`; if they cannot see the card, fix the
   driver before touching Python.
4. The RX 7900 XTX is `gfx1100`, which official ROCm builds support. Do not
   set `HSA_OVERRIDE_GFX_VERSION` as a first move; it is a workaround for
   unsupported cards and can hide the real mismatch.
5. To keep working while the GPU is unavailable, set `device: cpu` in the
   YAML (or pass `--device cpu` to `scripts.benchmark`). The Trainer then
   uses fp32/bf16 CPU paths and the same configs.

## 2. Know the budget: what consumes the 24 GB

For a model with `P` parameters, AdamW training in fp32 keeps roughly:

| Item | Bytes per parameter | `max_7900xtx` (~98.5 M params) |
|---|---:|---:|
| Weights | 4 | ~0.37 GiB |
| Gradients | 4 | ~0.37 GiB |
| AdamW first moment | 4 | ~0.37 GiB |
| AdamW second moment | 4 | ~0.37 GiB |
| **Persistent subtotal** | **16** | **~1.47 GiB** |

The preset target is 98.5 M parameters; the measured run below used a
4096-entry BPE bundle and reported 88,988,160 parameters, because the
16,384-token vocabulary target was not reached. The optimizer math scales
with the actual parameter count either way.

AMP does not shrink the fp32 master weights or optimizer state in this
implementation; it shrinks the *transient* activations and matmuls. On ROCm
the Trainer uses **bf16 autocast** (no `GradScaler` needed); NVIDIA CUDA uses
fp16 with a scaler. See `core/device.py::resolve_precision`.

The real consumer is activations. `CausalSelfAttention` materializes the
full score matrix `(batch, heads, context, context)` rather than using
Flash/SDPA kernels, so its cost grows **quadratically with context length**.
Per layer, the main transients are:

```
attention scores/weights : B * H * T * T
Q, K, V projections      : 3 * B * T * D
MLP hidden (4D) + GELU   : ~2 * B * T * 4D
final logits             : B * T * V
```

With the preset values (`B=batch`, `H=heads`, `T=context`, `D=width`,
`V=vocab`), the `max_7900xtx` rung at `B=4, H=12, T=1024, D=768` spends
roughly `0.09 GiB` on one score matrix per layer and `~0.16 GiB` on the
main transients per layer at bf16. Twelve layers plus logits therefore
land in the **single-digit GiB** range, not near 24 GB — before backward
saved tensors, allocator fragmentation, and anything else using the
display GPU.

Practical takeaway: the 24 GB card has room for `max_7900xtx`, but **do not
fill it to the last gigabyte**. Keep 2–3 GB free for the desktop/compositor,
the allocator cache, and transient peaks. The persistent model state is
small; batch and context are the knobs that move peak memory.

## 3. Pick the rung before tuning the knobs

| Rung | Params (target) | Context | Batch | AMP | Use it for |
|---|---:|---:|---:|---|---|
| `tiny_debug` | 35,776 | 16 | 8 | off | shapes/smoke tests, UI demos |
| `tiny_train` | 934,144 | 64 | 16 | off | watching loss move, CPU runs |
| `small` | 12,325,632 | 256 | 16 | on | first real German baseline |
| `medium` | 29,688,832 | 512 | 8 | on | better coherence if time allows |
| `max_7900xtx` | 98,452,992 | 1024 | 4 | on | upper end of one-GPU budget |

The presets live in `core/scaling.py`; a config selects one with
`profile:` + `profile_defaults: true`. Change one line to move rungs:

```yaml
profile: small        # tiny_debug -> tiny_train -> small -> medium -> max_7900xtx
profile_defaults: true
```

`configs/gpt_scaling.yaml` is the switchable example. `configs/gpt_de_small.yaml`
shows the full-corpus wiring (`phase: gpt`, `phase_config.corpus_path,
tokenizer_bundle`). Copy that `phase_config` into a max-profile config
rather than pointing the 1024-token rung at the bundled `data/sample_de`,
which is too short for even one window.

**Data comes first.** At these model sizes the binding constraint is corpus
size/quality, not VRAM. A profile that cannot fill its context windows will
report `skipped` in the benchmark or fail with a short-corpus error. Build
the local corpus first (see `data/README.md`), then scale the model.

### Measured baseline: `max_7900xtx` on the development 7900 XTX

This is a real full run, not a projection:

| Item | Value |
|---|---|
| Device | AMD Radeon RX 7900 XTX (`gfx1100`), 24,560 MiB |
| Software | torch `2.14.1+rocm7.2`, ROCm 7.2 |
| Profile | `max_7900xtx`, 12 layers, width 768, 12 heads |
| Context / batch / AMP | 1,024 / 4 / bf16 autocast |
| Actual parameters | 88,988,160 (4096-entry BPE, not the 16,384 target) |
| Corpus | `data/corpus_de`, 116,807,266 tokens |
| Windows | 112,928 train / 1,140 val |
| Steps / epochs | 50,000 / 2 |
| Wall clock | 7,897.7 s (~2 h 11 m) |
| Throughput | **26,107 tokens/s** (about 6.4 steps/s) |
| GPU peak / total | **6.9 GiB / 24.0 GiB** |
| GPU allocated / reserved | 1.4 GiB / 7.3 GiB |
| Final val loss / perplexity | 2.0062 / **7.43** |

The 6.9 GiB high-water mark matches the memory model: persistent state is
~1.4 GiB and the rest is activation/allocator headroom. There is plenty of
room left on the card, which is why batch size is the first knob to raise if
you want more throughput. The run folder is
`artifacts/20261115-114028_gpt_max_7900xtx/` (local artifact, gitignored).

A quick 20-step GPU check with the tiny rungs shows the other end of the
range: `tiny_debug` reached ~2,545 tokens/s at 65 MiB peak and `tiny_train`
~88,250 tokens/s at 108 MiB peak, both in fp32 because their presets leave
AMP off. **Do not use the tiny rungs to judge GPU throughput**: at batch
8×16 and 16×64 they are dominated by per-step Python/launch overhead, not
by matrix math.

A 5-step `small` slice on the same full corpus and bundle (context 256,
batch 16, bf16) measured **19,073 tokens/s at 1.0 GiB peak** with
12,318,720 parameters. It is a speed/memory check only — five steps say
nothing about quality — but it confirms the expected shape: going from
`small` to `max_7900xtx` raises peak memory from ~1 GiB to ~7 GiB and the
parameter count ~7×, while throughput did **not** drop. The full max run
averaged 26,107 tokens/s, about 1.4× the small slice, because its larger
matmuls amortize Python and kernel-launch overhead better. Measure before
assuming that a bigger rung is slower.

## 4. Tune in this order

The order matters because each step changes both speed and memory; changing
several at once makes the result unattributable.

### Step 0 — establish a reproducible baseline

```bash
# 5-step smoke test: does the whole path run on the GPU?
python -m scripts.benchmark --profiles tiny_debug --steps 5

# Real slice: small profile on the local corpus
python -m scripts.benchmark --profiles small --corpus data/corpus_de \
    --tokenizer-bundle data/corpus_de/tokenizer --steps 50

# Largest rung: short slice, then inspect peak memory
python -m scripts.benchmark --profiles max_7900xtx --corpus data/corpus_de \
    --tokenizer-bundle data/corpus_de/tokenizer --steps 20
```

`--tokenizer-bundle` is important on the full corpus: without it the
benchmark trains BPE from scratch in pure Python before the first step,
which can take far longer than the slice you wanted to measure. The bundle
must match (or be compatible with) the corpus; `data/corpus_de/tokenizer`
is the 4096-entry bundle used by `configs/gpt_de_small.yaml`.

The benchmark writes `benchmark.json`, `report.md` and normal run folders
under `artifacts/benchmarks/<timestamp>/`. Keep the device line and the
`gpu_peak_mb` value from `benchmark.json`; they are the baseline for every
later change.

### Step 1 — context length

Choose the shortest context that covers the structure you care about
(sentence, paragraph, or document). Halving `context_length` cuts the
attention score matrix by 4× and usually buys back more memory than halving
the model width. It also reduces the number of training windows available
from a fixed corpus, so check `train_windows`/`val_windows` in the run log.

### Step 2 — batch size

Increase `batch_size` only while tokens/sec improves and peak memory stays
below your headroom target (for example, `gpu_peak_mb` under ~20 GiB on a
24 GB card). Very small batches can leave the GPU waiting on Python and the
data loader; very large ones raise memory linearly and may hurt
generalization on a small corpus. Batch size is per-device here — gradient
accumulation is **not** implemented in the shared Trainer, so "larger
effective batch" means raising `batch_size`, not adding accumulation steps.

If you change batch size substantially, re-check the learning rate. The
presets use `3e-3 → 1e-3 → 6e-4 → 3e-4 → 2e-4` down the ladder; as a rule of
thumb, scale the learning rate with the square root of the batch change
(and re-run validation to confirm).

### Step 3 — model width/depth

Only after context and batch are settled. `n_embd` has the largest effect
on parameter count and per-token compute; `n_layer` multiplies activation
memory. Keep `n_embd % n_head == 0` (the schema validates this). A cheaper
intermediate experiment is to keep the rung and reduce `n_layer` by one or
`n_embd` by 25%, rather than dropping a whole rung.

### Step 4 — run length and checkpoint cadence

More steps help only if validation loss is still falling. Set
`save_interval` so an interruption costs minutes, not hours, and watch
`best_val` rather than the last training loss. Long runs on a noisy German
corpus should use the validation split (`val_split`) and early stopping by
judgement — the Trainer does not implement automatic early stopping.

## 5. Read the right numbers

Every logged step appends a record to `metrics.jsonl` in the run folder:

```json
{"kind": "step", "step": 250, "epoch": 1, "tokens": 1024000,
 "steps": 250, "elapsed_s": 40.0, "tokens_per_sec": 25600.0,
 "steps_per_sec": 6.25, "cpu_peak_mb": 10340.0,
 "gpu_allocated_mb": 1433.0, "gpu_reserved_mb": 7475.0,
 "gpu_peak_mb": 7068.0, "gpu_total_mb": 24560.0, "loss": 4.89}
```

Use `tokens_per_sec` for cross-profile comparisons (steps/sec depends on
batch × context). `gpu_peak_mb` is a high-water mark and is the number to
compare with the card's total; `gpu_reserved_mb` includes the allocator's
cached blocks. The benchmark excludes evaluation/checkpoint time from the
throughput clock (T-502), so its `tokens_per_sec` is training throughput.

A quick summary without extra tooling:

```bash
python - <<'PY'
import json, pathlib
path = sorted(pathlib.Path("artifacts").glob("*/metrics.jsonl"))[-1]
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
summary = [row for row in rows if row.get("kind") == "summary"][-1]
print(path.parent.name)
print({k: summary.get(k) for k in ("steps", "tokens_per_sec", "best_val", "gpu_peak_mb", "gpu_total_mb")})
PY
```

For benchmark comparison, trust **relative** differences measured in the same
session. CPU and GPU numbers are not comparable, and a 5–20-step slice is a
speed/memory measurement, not a quality claim.

## 6. OOM playbook

The Trainer catches CUDA/ROCm out-of-memory errors, saves a checkpoint,
calls `torch.cuda.empty_cache()`, and raises `TrainingOOMError` with
config-specific hints (T-503). `scripts/train.py` exits with status 2 for
OOM. Apply the fixes in this order:

1. **Halve the batch size.** `batch_size: 4 → 2 → 1`. This is the fastest
   and least disruptive change.
2. **Shorten the context.** `context_length: 1024 → 512` (or 768). Attention
   memory falls quadratically; this is the biggest lever at long context.
3. **Confirm AMP is on.** `amp: true`. On ROCm this is bf16 autocast, which
   is stable and needs no scaler. If AMP is already on, do not toggle it
   repeatedly; move to the next step.
4. **Drop a rung or shrink the model.** `max_7900xtx → medium`, or reduce
   `n_layer`/`n_embd`. This changes quality as well as memory, so record it.
5. **Free the card.** Close browsers, notebooks, other training jobs and
   compositor-heavy apps; inspect with `rocm-smi --showmeminfo vram` and
   `rocm-smi --showpids`.
6. **Resume, don't restart.** The failure message contains the checkpoint
   path and a command like:

   ```bash
   # substitute the config you actually ran; the hint prints <your config>
   python -m scripts.train --config configs/my_max.yaml \
       --resume artifacts/<run>/checkpoints/step_XXXXXX.pt
   ```

OOM is a high-water-mark event: current allocation can look small while a
single attention transient peaks above the limit. That is why context length
is a first-class fallback, not an afterthought. `PYTORCH_HIP_ALLOC_CONF` /
expandable-segment allocator options are experimental on ROCm; treat them as
a last resort and re-measure rather than relying on them.

## 7. Throughput troubleshooting (GPU not saturated)

| Symptom | Likely cause | First check |
|---|---|---|
| `gpu_peak_mb` far below 24 GB and tokens/sec flat as batch grows | Python/data-loader bound | batch size, `num_workers=0` in this Trainer, corpus size |
| tokens/sec falls sharply when context doubles | eager attention is quadratic | context length; consider a smaller rung |
| memory grows every epoch | optimizer/activation leak or growing cache | one run per process; inspect `gpu_reserved_mb` |
| loss becomes `nan`/`inf` | LR too high or bad batch | lower `learning_rate`, keep gradient clip at 1.0, check corpus for garbage |
| validation much worse than training | tiny/noisy corpus or overfitting | more data, dropout, shorter run |
| first steps slow, later steady | kernel/allocator warm-up | ignore first ~10 steps in manual measurements; the benchmark averages all requested steps |

The dataloader uses `num_workers=0`, and the GPT dataset is held in memory.
On a fast GPU, tiny batches can therefore be CPU-bound; increasing batch is
often better than adding workers that do not exist in this code path.

## 8. Full-run recipes

Once a benchmark slice fits, run the profile for its configured
`max_steps`:

```bash
# First real baseline (small, 256-token context)
python -m scripts.train --config configs/gpt_de_small.yaml

# Largest rung: start from the committed small config and switch the profile
cp configs/gpt_de_small.yaml configs/my_max.yaml
#   profile: max_7900xtx
#   profile_defaults: true
# keep the phase_config corpus/tokenizer block from gpt_de_small.yaml
python -m scripts.train --config configs/my_max.yaml
```

The one-command M6 pipeline runs preprocess → tokenizer → GPT training →
evaluation and leaves every artifact in one folder:

```bash
# Fast end-to-end demonstration (no prebuilt tokenizer needed)
python -m scripts.pipeline --config configs/gpt_tiny_debug.yaml \
    --profile moderate_clean
```

The pipeline's tokenizer stage trains char+BPE from scratch in pure Python.
That is fine for the tiny/quick configs but slow on the full ~266 MiB
corpus, so the full-corpus recipe above uses `scripts.train` with the
prebuilt `data/corpus_de/tokenizer` bundle instead.

Inspect any result at `/runs` in the UI (T-604) or read
`pipeline.json`/`metrics.jsonl` directly. A full run is successful when
validation perplexity is finite and trending down, `gpu_peak_mb` stayed
inside your headroom target, and the run folder contains the checkpoint
you expect.

## 9. Record your numbers

When publishing a 7900 XTX result, include:

- the device line from `run.log` / `benchmark.json`;
- torch and ROCm versions;
- profile, actual parameter count and actual vocabulary (BPE may produce
  fewer merges than the target on a small corpus);
- context, batch, AMP setting and steps;
- `tokens_per_sec`, `gpu_peak_mb`, `gpu_total_mb`;
- corpus path/size and whether it is the bundled sample or the full German
  mix.

That is enough for someone else to reproduce the slice and tell whether a
difference is real or just a different measurement.
