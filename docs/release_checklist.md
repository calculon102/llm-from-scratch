# MVP release checklist and demo (T-606)

This is the newcomer path through the **MVP**: M0 bootstrap + M1 XOR +
M2 tokenizer + core M4 MiniGPT. Everything uses the bundled CC0 German
sample corpus and runs offline. M3 embeddings, M5 scaling and the M6
extras (corpus downloader, full pipeline, runs browser) are demonstrated
separately and are not required for MVP acceptance.

## 1. Prerequisites

- Linux, macOS or WSL; Python 3.11+.
- No GPU required. The tiny configs run on CPU; on the 7900 XTX they use
  ROCm automatically through `device: auto`.
- No network required for the demo. `data/sample_de/` is checked in.

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # runtime + pytest; add ".[viz]" for plots
```

## 2. Reproduce the MVP with one command

```bash
python -m scripts.mvp_demo
```

The demo creates a fresh `artifacts/mvp_demo/<timestamp>/` folder, writes
resolved config copies under `configs/`, and runs:

| Stage | What it proves | Expected artifact |
| --- | --- | --- |
| XOR phase (M1) | A real training loop, gradient check, checkpoints | `runs/*_mvp_xor/checkpoints/final.pt`, `gradcheck_report.txt`, plots |
| Tokenizer phase (M2) | Noisy German ingestion, char + BPE from scratch | `runs/*_mvp_tokenizer/corpus/cleaned.txt`, `tokenizers/bundle.json`, `metrics.json` |
| MiniGPT pipeline (core M4, M6 wrapper) | Preprocess → tokenizer → transformer training → evaluation | `pipeline/*_mvp_pipeline/pipeline.json`, `gpt/model.pt`, `eval/report.json`, `eval/sample.txt` |

A successful run ends with:

```text
MVP demo complete
environment : Python 3.11... device=cpu ... precision=fp32
configs     : 3 validated
XOR phase (M1)              : steps=500, best_val=...
Tokenizer phase (M2)        : char_vocab=74, bpe_vocab=259, merges=185
MiniGPT pipeline (core M4)  : stages=preprocess→tokenizer→gpt→eval, val_loss=...
artifacts   : .../artifacts/mvp_demo/<timestamp>
```

The command exits `0` only if every stage finishes and every artifact above
exists. The CPU path takes a few seconds; GPU is faster.

Useful options:

```bash
python -m scripts.mvp_demo --device cpu          # force CPU fallback
python -m scripts.mvp_demo --device cuda         # require GPU
python -m scripts.mvp_demo --max-new-tokens 80  # longer generated sample
python -m scripts.mvp_demo --output-dir /tmp/mvp_demo
```

## 3. Reproduce the flow manually

Run these from the repository root with the venv active. Each command
creates its own run folder under `artifacts/`.

```bash
# 0. Sanity-check one config and the interpreter
python -m scripts.check_config --config configs/xor_tiny_debug.yaml

# 1. M1: XOR (fast; prints accuracy/loss, writes checkpoints and plots)
python -m scripts.train --config configs/xor_tiny_debug.yaml

# 2. M2: tokenizer (char + BPE, corpus report, compression/OOV metrics)
python -m scripts.tokenizer --config configs/tokenizer_tiny_debug.yaml

# 3. Core M4: MiniGPT training by itself
python -m scripts.train --config configs/gpt_tiny_debug.yaml

# 4. M6 convenience wrapper: preprocess + tokenizer + GPT + eval in one run
python -m scripts.pipeline --config configs/gpt_tiny_debug.yaml --max-new-tokens 40
```

To inspect the results in the browser:

```bash
python -m scripts.ui
# open http://127.0.0.1:8080/runs
```

The `/runs` page lists recent runs, shows status/timing/pipeline metrics,
and previews configs, logs, JSON metrics and plots. Point it at another
folder with `LLM_ARTIFACTS_ROOT=/path/to/artifacts python -m scripts.ui`.

## 4. Acceptance checklist

Copy this into the release issue/PR and tick each item:

- [ ] `pip install -e ".[dev]"` succeeds in a clean Python 3.11 venv.
- [ ] `pytest -q` passes (currently 385+ passed, 2 skipped).
- [ ] `python -m scripts.mvp_demo` exits `0` and prints `MVP demo complete`.
- [ ] The XOR run contains `final.pt`, `gradcheck_report.txt` and plots.
- [ ] The tokenizer run contains `corpus/cleaned.txt`,
      `tokenizers/{char,bpe}/` and `tokenizers/bundle.json`.
- [ ] The pipeline run contains `pipeline.json`, `gpt/model.pt`,
      `eval/report.json` and a non-empty `eval/sample.txt`.
- [ ] `pipeline.json` reports `"status": "finished"` and all five stages
      (`preprocess`, `tokenizer`, `gpt_setup`, `train`, `eval`).
- [ ] `python -m scripts.ui` serves `/runs` and the new runs are visible.
- [ ] CPU fallback works (`--device cpu` or a machine without CUDA/ROCm).
- [ ] No external API/network call is needed for the demo.
- [ ] `README.md`, `docs/roadmap.md` and `docs/release_checklist.md`
      agree on the MVP scope.
- [ ] `git status` is clean and the release tag/commit is recorded.

## 5. Release steps

1. Freeze scope: MVP means M0 + M1 + M2 + core M4; M3/M5/M6 are stretch
   or follow-up work.
2. Run the acceptance checklist above from a clean checkout.
3. Run the full test suite: `pytest -q`.
4. Optionally record a hardware slice:
   `python -m scripts.benchmark --profiles tiny_debug --steps 20`.
5. Update the changelog/README with the version and known limitations.
6. Commit, tag (`git tag -a v0.x.0 -m "MVP"`), and publish the release.
7. Close the M6 issues and mark the roadmap board's MVP milestone done.

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `python: command not found` | Use `python3.11`, and activate `.venv`. |
| `ModuleNotFoundError: torch` | `pip install -e .` inside the activated venv. |
| Plots missing | `pip install -e ".[viz]"`; training itself does not need matplotlib. |
| `device=cuda` unavailable | Let `device: auto` fall back to CPU, or pass `--device cpu` to the demo. |
| Demo says an artifact is missing | Read the stage's `run.log`; the demo reports the partial artifacts folder. |
| `(null): No such file or directory` from ROCm/PyTorch on startup | Benign library-loader noise on some ROCm installs; check that the demo's environment line reports the expected device. |
| Port 8080 busy | `python -m scripts.ui --port 8081`. |

## 7. Sign-off

| Field | Value |
| --- | --- |
| Release / tag | |
| Commit | |
| Date | |
| Python / torch | |
| Device / backend | |
| `pytest -q` | |
| `scripts.mvp_demo` | pass / fail |
| Reviewed by | |
