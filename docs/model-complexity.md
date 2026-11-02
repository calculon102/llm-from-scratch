# Model Complexity & Expected Capability (7900 XTX, 24GB)

## Abstraction Tiers
- Tier A: Small assistant-like (roughly “Haiku-like” class)
- Tier B: Balanced assistant-like (roughly “Sonnet-like” class)
- Tier C: Large assistant-like (roughly “Opus-like” class)

## Realistic expectation for this project
Single-GPU, from-scratch educational training is typically:
- strong for pedagogy and pipeline understanding
- capable of toy-to-narrow practical text generation
- generally below robust Tier B/C quality without much larger compute/data

Most realistic target:
- **low Tier A** behavior in constrained domains after scaling/tuning.

## Profile ladder (project configs)

### tiny_debug (minutes)
- Purpose: verify code paths and shapes
- Capability: none/very limited generation quality

### tiny_train (minutes)
- Purpose: learn training loop behavior
- Capability: short toy completions

### small (tens of minutes to ~1.5h)
- Purpose: useful baseline
- Capability: basic local assistant behavior in narrow scope

### medium (1–4h+)
- Purpose: better fluency/consistency
- Capability: improved coherence, still constrained robustness

### max_7900xtx (long run)
- Purpose: best single-GPU attempt
- Capability: upper-end of project constraints; still likely under true Tier B

## Why this ceiling exists
- Compute budget (single consumer GPU)
- Dataset scale/quality
- Training time budget
- Optimization maturity (kernels, precision, throughput)
- Architecture/data curation limits vs frontier systems

## Practical recommendations
1. Optimize for learning and reproducibility first.
2. Keep tiny profiles fast for iteration.
3. Track tokens/sec, val loss, and wall-clock.
4. Scale one axis at a time: context length → batch/accumulation → model size.
5. Use checkpoints and resume aggressively.
