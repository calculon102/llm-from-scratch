"""YAML config loading and schema validation (T-003).

Configs are plain YAML files passed via ``--config``.  Every config is
validated against a small declarative schema before use:

- unknown keys are rejected (typos surface immediately),
- types, allowed values (choices) and numeric ranges are checked,
- missing optional keys are filled with defaults,
- duplicate YAML keys and malformed YAML are reported,
- *all* problems found in a file are reported in one :class:`ConfigError`.

Phase-specific settings live in the ``phase_config`` mapping.  Later
tickets register per-phase validators with :func:`register_phase_schema`,
e.g. XOR (T-101) or GPT (T-404) will validate their own sections without
changing this loader.

Example config (``configs/tiny_debug.yaml``)::

    run_name: xor_tiny_debug
    phase: xor
    seed: 42
    device: auto
    amp: false
    profile: tiny_debug
    learning_rate: 0.05
    epochs: 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import yaml

__all__ = [
    "ConfigError",
    "SCALE_PROFILES",
    "PHASES",
    "add_config_arg",
    "load_config",
    "register_phase_schema",
    "validate_config",
]

#: Scale profile ladder (see AGENTS.md).
SCALE_PROFILES = ("tiny_debug", "tiny_train", "small", "medium", "max_7900xtx")

#: Known pipeline phases.
PHASES = ("xor", "tokenizer", "embeddings", "gpt")


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or fails validation."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

#: Base schema: field name -> spec.
#:
#: Spec keys:
#:   type      -- one of "str", "int", "float", "bool", "dict"
#:   default   -- value used when the key is absent (required fields
#:                have no default)
#:   choices   -- optional tuple of allowed values
#:   min/max   -- optional inclusive numeric bounds
_BASE_SCHEMA: dict[str, dict[str, Any]] = {
    "run_name": {"type": "str", "default": "run"},
    "seed": {"type": "int", "default": 42},
    "device": {"type": "str", "default": "auto", "choices": ("auto", "cpu", "cuda")},
    "amp": {"type": "bool", "default": False},
    "profile": {"type": "str", "default": "tiny_debug", "choices": SCALE_PROFILES},
    "output_dir": {"type": "str", "default": "artifacts"},
    "log_interval": {"type": "int", "default": 10, "min": 1},
    "batch_size": {"type": "int", "default": 32, "min": 1},
    "learning_rate": {"type": "float", "default": 1e-3, "min": 0.0},
    "epochs": {"type": "int", "default": 10, "min": 1},
    "max_steps": {"type": "int", "default": None, "min": 1},
    "val_split": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
    "save_interval": {"type": "int", "default": 100, "min": 1},
    "phase": {"type": "str", "default": None, "choices": PHASES},
    "phase_config": {"type": "dict", "default": {}},
}

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "str": lambda v: isinstance(v, str),
    # bool is a subclass of int in Python; reject it for int/float fields.
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "dict": lambda v: isinstance(v, dict),
}

#: phase -> validator(phase_config, source) -> list of error strings.
_PHASE_SCHEMAS: dict[str, Callable[[dict, str], list[str]]] = {}


def register_phase_schema(phase: str, validator: Callable[[dict, str], list[str]]) -> None:
    """Register a validator for the ``phase_config`` of ``phase``.

    ``validator`` receives the ``phase_config`` mapping and a source label
    (the config file path) and returns a list of error strings; an empty
    list means the section is valid.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {list(PHASES)}")
    _PHASE_SCHEMAS[phase] = validator


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_value(key: str, value: Any, spec: dict[str, Any]) -> list[str]:
    """Return error strings for one field value."""
    errors: list[str] = []
    if value is None:
        if spec.get("default") is not None:
            errors.append(f"'{key}': must not be null")
        return errors
    if not _TYPE_CHECKS[spec["type"]](value):
        errors.append(
            f"'{key}': expected {spec['type']}, got {type(value).__name__} ({value!r})"
        )
        return errors
    if "choices" in spec and value not in spec["choices"]:
        errors.append(f"'{key}': {value!r} is not one of {list(spec['choices'])}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "min" in spec and value < spec["min"]:
            errors.append(f"'{key}': {value} is below minimum {spec['min']}")
        if "max" in spec and value > spec["max"]:
            errors.append(f"'{key}': {value} is above maximum {spec['max']}")
    return errors


def validate_config(data: Any, source: str = "<dict>") -> dict:
    """Validate a parsed config mapping and return it with defaults filled.

    Raises :class:`ConfigError` listing *all* problems found.
    """
    if not isinstance(data, dict):
        got = type(data).__name__ if data is not None else "null (empty file?)"
        raise ConfigError(f"{source}: config must be a YAML mapping, got {got}")

    errors: list[str] = []
    resolved: dict[str, Any] = {}
    for key, spec in _BASE_SCHEMA.items():
        if key in data:
            value = data[key]
        elif "default" in spec:
            value = spec["default"]
        else:
            errors.append(f"missing required key '{key}'")
            continue
        errors.extend(_check_value(key, value, spec))
        resolved[key] = value

    for key in data:
        if key not in _BASE_SCHEMA:
            errors.append(
                f"unknown key '{key}' (allowed keys: {', '.join(_BASE_SCHEMA)})"
            )

    # Phase-specific section, validated by the registered schema if any.
    phase = resolved["phase"]
    phase_config = resolved["phase_config"]
    if isinstance(phase_config, dict) and phase in _PHASE_SCHEMAS:
        errors.extend(_PHASE_SCHEMAS[phase](phase_config, source))

    if errors:
        raise ConfigError(f"{source}: invalid config:\n  - " + "\n  - ".join(errors))
    return resolved


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _key_text(node: yaml.Node) -> str:
    if isinstance(node, yaml.ScalarNode):
        return str(node.value)
    return f"<{node.__class__.__name__}>"


def _duplicate_key_errors(node: yaml.Node, path: str = "$") -> list[str]:
    """Walk the YAML node tree and report duplicate mapping keys."""
    errors: list[str] = []
    if isinstance(node, yaml.MappingNode):
        seen: dict[str, str] = {}
        for key_node, value_node in node.value:
            key = _key_text(key_node)
            where = f"{path}.{key}"
            if key in seen:
                errors.append(f"duplicate key '{key}' (first at {seen[key]})")
            else:
                seen[key] = where
            errors.extend(_duplicate_key_errors(value_node, where))
    elif isinstance(node, yaml.SequenceNode):
        for i, item in enumerate(node.value):
            errors.extend(_duplicate_key_errors(item, f"{path}[{i}]"))
    return errors


def load_config(path: str | Path) -> dict:
    """Load a YAML config file, validate it, and return it with defaults.

    Raises :class:`ConfigError` with a clear message on any problem:
    missing file, unreadable file, malformed YAML, duplicate keys, or
    schema violations.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc

    try:
        root = yaml.compose(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    dup_errors = _duplicate_key_errors(root) if root is not None else []
    if dup_errors:
        raise ConfigError(f"{path}: invalid config:\n  - " + "\n  - ".join(dup_errors))

    data = yaml.safe_load(text)
    return validate_config(data, source=str(path))


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def add_config_arg(parser: argparse.ArgumentParser, required: bool = True) -> None:
    """Add the standard ``--config`` argument to an argparse parser."""
    parser.add_argument(
        "--config",
        type=str,
        required=required,
        help="Path to the YAML config file to load and validate.",
    )
