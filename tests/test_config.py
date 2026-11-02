"""Tests for the YAML config loader and schema validation (T-003)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import core.config as config
from core.config import ConfigError, add_config_arg, load_config, validate_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_valid_config_fills_defaults():
    cfg = validate_config({"run_name": "x"})
    assert cfg["run_name"] == "x"
    assert cfg["seed"] == 42
    assert cfg["profile"] == "tiny_debug"
    assert cfg["device"] == "auto"
    assert cfg["amp"] is False
    assert cfg["phase"] is None
    assert cfg["phase_config"] == {}
    assert cfg["learning_rate"] == 1e-3


def test_int_accepted_for_float_field():
    cfg = validate_config({"learning_rate": 1})
    assert cfg["learning_rate"] == 1


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml_raises_clear_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("run_name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(p)


def test_non_mapping_raises_clear_error(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_config(p)


def test_empty_file_raises_clear_error(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_config(p)


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("runnam: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key 'runnam'"):
        load_config(p)


def test_wrong_type_reported(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("learning_rate: fast\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'learning_rate': expected float, got str"):
        load_config(p)


def test_bool_rejected_for_int_field(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'seed': expected int, got bool"):
        load_config(p)


def test_bad_choice_reported(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("profile: huge\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'profile': 'huge' is not one of"):
        load_config(p)


def test_out_of_range_reported(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("val_split: 2.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="above maximum 1.0"):
        load_config(p)


def test_null_reported_for_non_optional_field(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("epochs: null\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'epochs': must not be null"):
        load_config(p)


def test_null_ok_for_optional_field():
    cfg = validate_config({"max_steps": None})
    assert cfg["max_steps"] is None


def test_all_errors_reported_together(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("epochs: -1\nprofile: huge\nbatch_size: x\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(p)
    msg = str(excinfo.value)
    assert "below minimum 1" in msg
    assert "'huge' is not one of" in msg
    assert "'batch_size': expected int" in msg


def test_duplicate_keys_reported(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\nseed: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key 'seed'"):
        load_config(p)


def test_phase_config_validated_by_registered_schema():
    def xor_validator(phase_config, source):
        errors = []
        if "hidden_units" not in phase_config:
            errors.append("phase_config: 'hidden_units' is required for phase 'xor'")
        return errors

    config.register_phase_schema("xor", xor_validator)
    try:
        with pytest.raises(ConfigError, match="'hidden_units' is required"):
            validate_config({"phase": "xor", "phase_config": {}})
        cfg = validate_config({"phase": "xor", "phase_config": {"hidden_units": 8}})
        assert cfg["phase_config"] == {"hidden_units": 8}
    finally:
        # Don't leak the mock schema into other tests' processes.
        config._PHASE_SCHEMAS.pop("xor", None)


def test_register_unknown_phase_rejected():
    with pytest.raises(ValueError, match="unknown phase"):
        config.register_phase_schema("nope", lambda pc, src: [])


def test_add_config_arg_parses():
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    args = parser.parse_args(["--config", "configs/tiny_debug.yaml"])
    assert args.config == "configs/tiny_debug.yaml"


def test_sample_profile_in_repo_loads():
    cfg = load_config(REPO_ROOT / "configs" / "tiny_debug.yaml")
    assert cfg["profile"] == "tiny_debug"
    assert cfg["run_name"] == "tiny_debug"
    assert cfg["seed"] == 42
