# LLM From Scratch – Roadmap

## Goal
Educational, Python-only, offline-capable project to build an LLM pipeline step-by-step with minimal web UI.

## Milestones

### M0 – Bootstrap & Infrastructure
- Project skeleton, MIT license, README
- Python packaging, dependencies
- YAML config loader + validation
- Shared trainer loop + artifacts/checkpoints
- NiceGUI shell (dark mode, responsive)
- Job runner (start/stop/status/logs)

### M1 – XOR Learning Module
- XOR dataset + 2-layer MLP
- Loss curve + decision boundary + activations
- Optional finite-difference gradient checks
- Math notes in markdown

### M2 – Tokenizer Module
- Noisy DE-first corpus ingestion
- Character tokenizer
- BPE tokenizer from scratch
- Tokenizer compare UI + quality metrics
- Math notes in markdown

### M3 – Embeddings Module
- Skip-gram + negative sampling
- Nearest-neighbor search
- Optional UMAP/t-SNE projection
- Embeddings explorer UI
- Math notes in markdown

### M4 – Mini GPT Module
- Decoder-only transformer (causal attention)
- LM dataset chunking
- Training + perplexity evaluation
- Sampling (greedy/top-k/top-p/temp)
- Generation UI
- Math notes in markdown

### M5 – Scaling on 7900 XTX
- Profiles: tiny_debug → max_7900xtx
- Throughput + memory instrumentation
- OOM fallback guidance + resume robustness
- Benchmark report

### M6 – Capstone & Docs
- Data downloader + offline cache manifest
- Preprocess profiles
- End-to-end pipeline command
- Artifacts/runs browser in UI
- Scaling guide + MVP release checklist

## Suggested board workflow
Backlog → Ready → In Progress → Blocked → Review/Verify → Done

## MVP vs Stretch
- MVP: M0 + M1 + M2 + core M4
- Stretch: full M3 + M5 + M6
