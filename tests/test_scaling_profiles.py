"""Tests for the M5 scale-profile presets and YAML switch (T-501).

Acceptance: "one config switch changes scale profile".  The switch is the
single ``profile:`` line in ``configs/gpt_scaling.yaml``; the preset layer
must fill architecture/optimizer values while explicit YAML values still
win.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import SCALE_PROFILES, ConfigError, load_config, validate_config
from core.scaling import (
    PROFILE_ORDER,
    PROFILE_PRESETS,
    apply_profile_defaults,
    get_profile,
    list_profiles,
    profile_config,
    profile_table_rows,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_profile_order_matches_config_ladder() -> None:
    assert PROFILE_ORDER == SCALE_PROFILES
    assert list(PROFILE_PRESETS) == list(PROFILE_ORDER)


def test_every_profile_is_complete_and_grows_with_the_ladder() -> None:
    presets = list_profiles()
    assert len(presets) == 5

    parameters = [preset.estimate_parameters() for preset in presets]
    contexts = [preset.phase_config["context_length"] for preset in presets]
    depths = [preset.phase_config["n_layer"] for preset in presets]
    assert parameters == sorted(parameters) and len(set(parameters)) == 5
    assert contexts == sorted(contexts) and len(set(contexts)) == 5
    assert depths == sorted(depths) and len(set(depths)) == 5

    for preset in presets:
        assert preset.config["batch_size"] >= 1
        assert preset.config["learning_rate"] > 0
        assert preset.phase_config["n_embd"] % preset.phase_config["n_head"] == 0
        assert preset.config["max_steps"] >= preset.config["save_interval"]
        assert preset.runtime and preset.capability


def test_single_profile_line_switches_the_whole_scale(tmp_path: Path) -> None:
    """The ticket's acceptance criterion, exercised on the real YAML."""
    source = REPO_ROOT / "configs" / "gpt_scaling.yaml"
    text = source.read_text(encoding="utf-8")
    assert "profile: tiny_debug" in text

    tiny = load_config(source)
    assert tiny["phase_config"]["n_embd"] == 32
    assert tiny["phase_config"]["context_length"] == 16
    assert tiny["batch_size"] == 8

    switched_path = tmp_path / "switched.yaml"
    switched_path.write_text(
        text.replace("profile: tiny_debug", "profile: small"), encoding="utf-8"
    )
    small = load_config(switched_path)
    assert small["profile"] == "small"
    assert small["phase_config"]["n_embd"] == 384
    assert small["phase_config"]["context_length"] == 256
    assert small["phase_config"]["n_layer"] == 6
    assert small["batch_size"] == 16
    assert small["amp"] is True
    # One line changed, the rest of the scale followed.
    assert small["learning_rate"] != tiny["learning_rate"]
    assert small["max_steps"] != tiny["max_steps"]


def test_explicit_yaml_values_win_over_profile_defaults() -> None:
    cfg = validate_config(
        {
            "profile": "small",
            "profile_defaults": True,
            "phase": "gpt",
            "batch_size": 3,
            "phase_config": {"n_embd": 64, "n_head": 4},
        }
    )
    assert cfg["batch_size"] == 3  # explicit
    assert cfg["learning_rate"] == 6e-4  # from the profile
    assert cfg["phase_config"]["n_embd"] == 64  # explicit
    assert cfg["phase_config"]["context_length"] == 256  # from the profile
    assert cfg["phase_config"]["vocab_size"] == 4096


def test_profile_defaults_is_opt_in() -> None:
    # Without the switch, "profile" stays a label (existing behaviour).
    plain = validate_config({"profile": "small"})
    assert plain["batch_size"] == 32
    assert plain["phase_config"] == {}
    assert plain["profile_defaults"] is False


def test_phase_preset_only_applies_to_gpt() -> None:
    data = apply_profile_defaults(
        {"phase": "xor", "profile": "small", "profile_defaults": True}
    )
    assert data.get("phase_config", {}) == {}
    assert data["batch_size"] == 16  # top-level keys still apply

    cfg = validate_config(
        {
            "phase": "xor",
            "profile": "small",
            "profile_defaults": True,
            "phase_config": {"hidden_size": 8},
        }
    )
    assert cfg["phase_config"] == {"hidden_size": 8}


def test_unknown_profile_still_reports_the_schema_error() -> None:
    with pytest.raises(ConfigError, match="'huge' is not one of"):
        validate_config({"profile": "huge", "profile_defaults": True})
    assert get_profile("huge") is None


def test_all_rung_yaml_files_load() -> None:
    files = {
        "gpt_tiny_debug_profile.yaml": "tiny_debug",
        "gpt_tiny_train.yaml": "tiny_train",
        "gpt_small.yaml": "small",
        "gpt_medium.yaml": "medium",
        "gpt_max_7900xtx.yaml": "max_7900xtx",
    }
    for filename, profile in files.items():
        cfg = load_config(REPO_ROOT / "configs" / filename)
        assert cfg["profile"] == profile
        assert cfg["phase"] == "gpt"
        assert cfg["phase_config"]["n_embd"] == PROFILE_PRESETS[profile].phase_config["n_embd"]
        assert cfg["batch_size"] == PROFILE_PRESETS[profile].config["batch_size"]


def test_profile_config_helper_and_report_rows() -> None:
    cfg = profile_config("tiny_train", base={"run_name": "override", "seed": 7})
    assert cfg["run_name"] == "override"
    assert cfg["seed"] == 7
    assert cfg["phase"] == "gpt"
    assert cfg["phase_config"]["context_length"] == 64

    rows = profile_table_rows()
    assert [row["profile"] for row in rows] == list(PROFILE_ORDER)
    assert rows[0]["parameters"] < rows[-1]["parameters"]
    assert rows[-1]["context_length"] == 1024
