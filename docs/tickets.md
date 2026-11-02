# Ticket Backlog (Issue-Sized)

Legend:
- Priority: P0/P1/P2
- Size: XS (~0.5d), S (~1d), M (~2–3d), L (~4–6d)

## M0 – Bootstrap & Infrastructure

### T-001 Create project skeleton and MIT license
- Priority: P0, Size: S
- Done when: folder structure + MIT + base README exist.

### T-002 Add Python packaging and dependency groups
- Priority: P0, Size: S
- Depends on: T-001
- Done when: clean install works on Python 3.11.

### T-003 Implement YAML config loader and schema validation
- Priority: P0, Size: M
- Depends on: T-002
- Done when: `--config` loads/validates with clear errors.

### T-004 Build run artifact manager
- Priority: P0, Size: M
- Depends on: T-003
- Done when: run outputs config snapshot/logs/checkpoints/plots.

### T-005 Implement shared Trainer loop (train/eval/checkpoint/resume)
- Priority: P0, Size: L
- Depends on: T-004
- Done when: reused by XOR + at least one later module.

### T-006 Add device manager for ROCm/CPU + AMP toggle
- Priority: P0, Size: S
- Depends on: T-002
- Done when: startup logs device + precision clearly.

### T-007 Build NiceGUI app shell (dark, responsive)
- Priority: P0, Size: M
- Depends on: T-001
- Done when: all phase routes reachable.

### T-008 Add UI job runner (start/stop/status/log stream)
- Priority: P0, Size: L
- Depends on: T-007, T-005
- Done when: long training jobs controllable in UI.

## M1 – XOR

### T-101 Implement XOR dataset + 2-layer MLP baseline
- Priority: P0, Size: S
- Depends on: T-005
- Done when: default config learns XOR reliably.

### T-102 Add XOR decision boundary and activation plots
- Priority: P0, Size: M
- Depends on: T-101
- Done when: plots generated and visible in UI.

### T-103 Add finite-difference gradient check mode
- Priority: P1, Size: S
- Depends on: T-101
- Done when: gradient mismatch report available.

### T-104 Build XOR UI controls and outputs
- Priority: P0, Size: S
- Depends on: T-007, T-101, T-102
- Done when: retrain from UI updates outputs.

### T-105 Write docs/math/phase1_xor.md
- Priority: P0, Size: S
- Depends on: T-101
- Done when: equations + intuition + failure modes documented.

## M2 – Tokenizer

### T-201 Implement corpus ingestion (noisy DE-first)
- Priority: P0, Size: M
- Depends on: T-003
- Done when: robust ingestion on noisy data.

### T-202 Build character tokenizer with tests
- Priority: P0, Size: S
- Depends on: T-201
- Done when: deterministic encode/decode round-trip passes.

### T-203 Build BPE tokenizer trainer from scratch
- Priority: P0, Size: L
- Depends on: T-202
- Done when: vocab+merges persist and reload.

### T-204 Add tokenizer metrics
- Priority: P1, Size: M
- Depends on: T-203
- Done when: compression/token-length/OOV metrics logged.

### T-205 Build tokenizer UI compare page
- Priority: P0, Size: M
- Depends on: T-007, T-202, T-203
- Done when: side-by-side tokenizer view works.

### T-206 Write docs/math/phase2_tokenizer.md
- Priority: P0, Size: S
- Depends on: T-203
- Done when: merge algorithm and tradeoffs documented.

## M3 – Embeddings

### T-301 Implement skip-gram pair generator + negative sampler
- Priority: P1, Size: M
- Depends on: T-203
- Done when: stable configurable batch pipeline exists.

### T-302 Implement skip-gram model + trainer integration
- Priority: P1, Size: M
- Depends on: T-301, T-005
- Done when: loss decreases on tiny/small profiles.

### T-303 Add nearest-neighbor cosine tooling
- Priority: P1, Size: S
- Depends on: T-302
- Done when: token top-k neighbor query works.

### T-304 Add UMAP/t-SNE projection utility
- Priority: P2, Size: M
- Depends on: T-302
- Done when: 2D projection plots generated.

### T-305 Build embeddings UI explorer
- Priority: P1, Size: M
- Depends on: T-007, T-303, T-304
- Done when: query + neighbors + optional scatter available.

### T-306 Write docs/math/phase3_embeddings.md
- Priority: P1, Size: S
- Depends on: T-302
- Done when: objective and intuition documented.

## M4 – Mini GPT

### T-401 Implement causal self-attention block with tests
- Priority: P0, Size: L
- Depends on: T-005
- Done when: mask correctness + shape tests pass.

### T-402 Implement decoder transformer stack + LM head
- Priority: P0, Size: L
- Depends on: T-401
- Done when: stable tiny forward pass.

### T-403 Implement LM dataset chunking + train/val split
- Priority: P0, Size: M
- Depends on: T-203
- Done when: configurable context batches produced.

### T-404 Add GPT training + perplexity evaluation
- Priority: P0, Size: M
- Depends on: T-402, T-403
- Done when: tiny overfit sanity + val perplexity logs.

### T-405 Implement sampling modes
- Priority: P1, Size: S
- Depends on: T-404
- Done when: greedy/top-k/top-p/temp all available.

### T-406 Build GPT generation UI page
- Priority: P1, Size: M
- Depends on: T-007, T-405
- Done when: interactive generation works.

### T-407 Write docs/math/phase4_transformer.md
- Priority: P0, Size: M
- Depends on: T-402
- Done when: attention + LM objective documented.

## M5 – Scaling on 7900 XTX

### T-501 Define scaling profile YAMLs
- Priority: P1, Size: S
- Depends on: T-404
- Done when: one config switch changes scale profile.

### T-502 Add throughput and memory instrumentation
- Priority: P1, Size: M
- Depends on: T-404
- Done when: tokens/sec + memory indicators logged.

### T-503 Add OOM-safe fallback hints + auto-save on interrupt
- Priority: P1, Size: M
- Depends on: T-502
- Done when: failures output actionable fallback steps.

### T-504 Benchmark profile report on 7900 XTX
- Priority: P1, Size: M
- Depends on: T-501, T-502
- Done when: speed/quality comparison table documented.

## M6 – Capstone & Docs

### T-601 Add corpus downloader + local cache manifest (DE-first)
- Priority: P1, Size: M
- Depends on: T-201
- Done when: one command pulls and registers sources.

### T-602 Add preprocess profiles
- Priority: P1, Size: M
- Depends on: T-601
- Done when: reproducible processed corpora generated.

### T-603 Build end-to-end pipeline command
- Priority: P1, Size: M
- Depends on: T-203, T-404, T-602
- Done when: tokenizer→gpt→eval runs from one command.

### T-604 Build UI runs/artifacts browser page
- Priority: P2, Size: S
- Depends on: T-008, T-004
- Done when: recent runs/artifacts browsable in UI.

### T-605 Write 7900 XTX scaling guide
- Priority: P1, Size: S
- Depends on: T-504
- Done when: practical tuning guide exists.

### T-606 Create MVP release checklist + demo script
- Priority: P1, Size: S
- Depends on: T-603
- Done when: newcomer can reproduce MVP flow.
