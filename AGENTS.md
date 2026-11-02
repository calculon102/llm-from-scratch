# AGENTS.md

## Project state
- Pre-code: the repo contains only `README.md` and `docs/` (roadmap, tickets, model-complexity). No source, manifests, or CI yet — don't assume `pytest`, `ruff`, or CI commands exist.
- Work is driven by GitHub issues: https://github.com/calculon102/llm-from-scratch — issues #2–#43 mirror `docs/tickets.md` (T-001…T-606); issue #1 is the roadmap board.
- Pick a ticket whose "Depends on" items are done; the "Done when" line in each issue is the acceptance criterion.

## What we're building
- Educational, Python-only, offline-capable LLM pipeline with a minimal NiceGUI web UI (dark, responsive).
- Target hardware: single AMD 7900 XTX (24 GB) via ROCm with CPU fallback and AMP toggle.
- Corpus is German-first (DE-first) and noisy; everything must run offline (local cache, no external API calls).

## Planned structure (per T-001)
- `core/` shared library, `apps/ui/` NiceGUI app, `configs/` YAML profiles, `scripts/` entrypoints, `artifacts/` run outputs, `docs/math/` per-phase math notes.
- Python 3.11, `pyproject.toml` with grouped dependencies (runtime + optional viz).
- Every phase ships a math note in `docs/math/` (e.g. `phase1_xor.md`).

## Architecture rules
- Reuse the shared Trainer loop (T-005) for all training phases — don't write per-phase training loops.
- Config is YAML loaded via `--config` with schema validation (T-003). Scale profiles follow the ladder `tiny_debug → tiny_train → small → medium → max_7900xtx`; keep tiny profiles fast for iteration.
- Runs go through the artifact manager (T-004): per-run folder with config snapshot, logs, checkpoints, plots.

## Milestones
- M0 bootstrap → M1 XOR → M2 tokenizer → M3 embeddings → M4 mini GPT → M5 scaling → M6 capstone.
- MVP = M0 + M1 + M2 + core M4; M3/M5/M6 are stretch. Details in `docs/roadmap.md`; capability expectations in `docs/model-complexity.md`.
