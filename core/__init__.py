"""Shared library for the llm-from-scratch pipeline.

Submodules (added by their tickets):
- config:    YAML config loading + schema validation (T-003)
- artifacts: per-run artifact manager (T-004)
- trainer:   shared train/eval/checkpoint/resume loop (T-005)
- device:    ROCm/CPU device manager + AMP toggle (T-006)
"""

__version__ = "0.1.0"
