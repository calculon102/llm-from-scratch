"""Scale-profile presets for the M5 ladder (T-501).

The project ladder is::

    tiny_debug -> tiny_train -> small -> medium -> max_7900xtx

A profile is a named bundle of values for the shared config (optimizer,
batch size, run length) *and* the GPT phase (context length, depth, width,
vocabulary target).  Keeping them in one place means the YAML profiles are
thin: one ``profile:`` line selects the whole scale.

How profiles interact with YAML
-------------------------------
The presets are a *layer between* the schema defaults and explicit YAML
values, but they are opt-in.  A config enables the layer with::

    profile: small
    profile_defaults: true

Then every key the YAML does not set is filled from the ``small`` preset;
explicit YAML values always win.  Without ``profile_defaults: true`` the
``profile`` field stays a label (as it was before M5), so existing configs
and tests are unaffected.  A minimal switchable config is
``configs/gpt_scaling.yaml``: change its ``profile`` line to move up or
down the ladder.

The presets target the GPT phase; ``phase_config`` values are only applied
when ``phase: gpt``.  The top-level training values (batch size, learning
rate, ...) apply to any phase, so the same switch also works for future
training phases.

Sizing notes
------------
* ``vocab_size`` is the *target* BPE vocabulary.  On a small corpus BPE
  simply produces fewer merges, so the actual model is smaller; that is
  expected and is reported in the run metadata.
* Parameter counts are analytic estimates from the MiniGPT architecture
  (see :meth:`ScaleProfile.estimate_parameters`), so generating a table
  never has to allocate a model.
* ``max_7900xtx`` is sized to leave headroom on 24 GB for AdamW states and
  activations.  The practical bottleneck is data (M6), not VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "PROFILE_ORDER",
    "PROFILE_PRESETS",
    "ScaleProfile",
    "apply_profile_defaults",
    "get_profile",
    "list_profiles",
    "profile_config",
    "profile_table_rows",
]

#: Ladder order, cheapest first.  Must match ``core.config.SCALE_PROFILES``.
PROFILE_ORDER: tuple[str, ...] = (
    "tiny_debug",
    "tiny_train",
    "small",
    "medium",
    "max_7900xtx",
)


@dataclass(frozen=True)
class ScaleProfile:
    """One rung of the scaling ladder.

    Attributes
    ----------
    name:
        Profile id used in ``profile:`` in a YAML file.
    title:
        Human-readable one-line label.
    runtime:
        Rough wall-clock budget from ``docs/model-complexity.md``.
    capability:
        What the profile is useful for.
    description:
        What changes relative to the previous rung.
    config:
        Top-level config values (``batch_size``, ``learning_rate``, ...).
    phase_config:
        GPT ``phase_config`` values (architecture, data, tokenizer target).
    """

    name: str
    title: str
    runtime: str
    capability: str
    description: str
    config: Mapping[str, Any] = field(default_factory=dict)
    phase_config: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def estimate_parameters(self) -> int:
        """Analytic MiniGPT parameter count for this profile's target vocab.

        MiniGPT consists of token + position embeddings, ``n_layer``
        pre-LayerNorm transformer blocks and a final LayerNorm.  With tied
        weights the LM head shares the token embedding.  Per block the
        parameters are ``12 * n_embd**2 + 13 * n_embd`` (QKV = 3D², output
        projection = D², MLP = 8D², LayerNorms = 4D, biases = 9D).
        """
        d = int(self.phase_config["n_embd"])
        layers = int(self.phase_config["n_layer"])
        vocab = int(self.phase_config["vocab_size"])
        context = int(self.phase_config["context_length"])
        bias = bool(self.phase_config.get("bias", True))
        tied = bool(self.phase_config.get("tie_weights", True))

        norm = 2 * d if bias else d
        per_block = 12 * d * d + (12 if bias else 9) * d + 2 * norm
        embeddings = (vocab + context) * d
        head = 0 if tied else d * vocab
        return int(embeddings + layers * per_block + norm + head)

    def as_dict(self) -> dict[str, Any]:
        """Plain mapping (useful for reports and the UI)."""
        return {
            "name": self.name,
            "title": self.title,
            "runtime": self.runtime,
            "capability": self.capability,
            "description": self.description,
            "parameters": self.estimate_parameters(),
            "config": dict(self.config),
            "phase_config": dict(self.phase_config),
        }


#: Shared corpus/tokenizer settings used by every rung.
_CORPUS: dict[str, Any] = {
    "corpus_path": "data/sample_de",
    "min_frequency": 2,
    "add_special_tokens": True,
    "drop_duplicates": True,
    "lowercase": False,
    "encodings": ["utf-8-sig", "utf-8", "cp1252", "latin-1"],
    "extensions": [".txt"],
}


PROFILE_PRESETS: dict[str, ScaleProfile] = {
    "tiny_debug": ScaleProfile(
        name="tiny_debug",
        title="tiny_debug - shapes and smoke tests",
        runtime="seconds to minutes",
        capability="None/very limited generation; verifies code paths.",
        description="The fastest rung: 2 layers, 32-wide, 16-token context.",
        config={
            "batch_size": 8,
            "learning_rate": 3e-3,
            "epochs": 100,
            "max_steps": 400,
            "val_split": 0.1,
            "save_interval": 100,
            "log_interval": 10,
            "amp": False,
        },
        phase_config={
            **_CORPUS,
            "vocab_size": 300,
            "max_token_length": 16,
            "context_length": 16,
            "stride": 0,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 32,
            "dropout": 0.0,
            "bias": True,
            "tie_weights": True,
            "init_std": 0.02,
        },
    ),
    "tiny_train": ScaleProfile(
        name="tiny_train",
        title="tiny_train - learn training-loop behaviour",
        runtime="minutes",
        capability="Short toy completions; enough to watch loss/perplexity move.",
        description="4 layers, 128-wide, 64-token context, ~1k vocab target.",
        config={
            "batch_size": 16,
            "learning_rate": 1e-3,
            "epochs": 50,
            "max_steps": 2_000,
            "val_split": 0.1,
            "save_interval": 250,
            "log_interval": 25,
            "amp": False,
        },
        phase_config={
            **_CORPUS,
            "vocab_size": 1024,
            "max_token_length": 24,
            "context_length": 64,
            "stride": 0,
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 128,
            "dropout": 0.1,
            "bias": True,
            "tie_weights": True,
            "init_std": 0.02,
        },
    ),
    "small": ScaleProfile(
        name="small",
        title="small - useful baseline",
        runtime="tens of minutes to ~1.5 h",
        capability="Basic local assistant behaviour in a narrow domain.",
        description="6 layers, 384-wide, 256-token context, ~4k vocab target.",
        config={
            "batch_size": 16,
            "learning_rate": 6e-4,
            "epochs": 30,
            "max_steps": 10_000,
            "val_split": 0.05,
            "save_interval": 500,
            "log_interval": 50,
            "amp": True,
        },
        phase_config={
            **_CORPUS,
            "vocab_size": 4096,
            "max_token_length": 32,
            "context_length": 256,
            "stride": 0,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "dropout": 0.1,
            "bias": True,
            "tie_weights": True,
            "init_std": 0.02,
        },
    ),
    "medium": ScaleProfile(
        name="medium",
        title="medium - better fluency/consistency",
        runtime="1-4 h+",
        capability="Improved coherence; still constrained robustness.",
        description="8 layers, 512-wide, 512-token context, ~8k vocab target.",
        config={
            "batch_size": 8,
            "learning_rate": 3e-4,
            "epochs": 15,
            "max_steps": 20_000,
            "val_split": 0.02,
            "save_interval": 1_000,
            "log_interval": 100,
            "amp": True,
        },
        phase_config={
            **_CORPUS,
            "vocab_size": 8192,
            "max_token_length": 40,
            "context_length": 512,
            "stride": 0,
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "dropout": 0.1,
            "bias": True,
            "tie_weights": True,
            "init_std": 0.02,
        },
    ),
    "max_7900xtx": ScaleProfile(
        name="max_7900xtx",
        title="max_7900xtx - best single-GPU attempt",
        runtime="long run",
        capability="Upper end of the project's single-GPU constraints.",
        description=(
            "12 layers, 768-wide, 1024-token context, ~16k vocab target; "
            "fits 24 GB with AMP, but needs the full M6 corpus to train."
        ),
        config={
            "batch_size": 4,
            "learning_rate": 2e-4,
            "epochs": 8,
            "max_steps": 50_000,
            "val_split": 0.01,
            "save_interval": 2_500,
            "log_interval": 250,
            "amp": True,
        },
        phase_config={
            **_CORPUS,
            "vocab_size": 16384,
            "max_token_length": 48,
            "context_length": 1024,
            "stride": 0,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "dropout": 0.1,
            "bias": True,
            "tie_weights": True,
            "init_std": 0.02,
        },
    ),
}


def get_profile(name: str | None) -> ScaleProfile | None:
    """Return the preset for ``name``, or ``None`` for an unknown/missing name."""
    return PROFILE_PRESETS.get(name) if isinstance(name, str) else None


def list_profiles() -> list[ScaleProfile]:
    """Return the presets in ladder order (cheapest first)."""
    return [PROFILE_PRESETS[name] for name in PROFILE_ORDER]


def profile_config(
    name: str, *, base: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build a complete config mapping for ``name``.

    ``base`` values win over the preset (the same precedence as YAML
    overrides).  The result is not schema-validated; callers that need a
    validated config should pass it to :func:`core.config.validate_config`.
    """
    preset = get_profile(name)
    if preset is None:
        raise KeyError(
            f"unknown profile {name!r}; expected one of {', '.join(PROFILE_ORDER)}"
        )
    cfg: dict[str, Any] = {
        "run_name": f"gpt_{name}",
        "profile": name,
        "profile_defaults": True,
        "phase": "gpt",
        **preset.config,
        "phase_config": dict(preset.phase_config),
    }
    if base:
        overrides = dict(base)
        phase_overrides = overrides.pop("phase_config", None)
        cfg.update(overrides)
        if phase_overrides:
            cfg["phase_config"] = {**cfg["phase_config"], **phase_overrides}
    return cfg


def apply_profile_defaults(data: Mapping[str, Any]) -> dict[str, Any]:
    """Fill missing keys from ``data["profile"]``'s preset.

    Explicit values in ``data`` always win.  ``phase_config`` is merged
    key-by-key and only for the GPT phase (the presets describe GPT
    architectures).  Unknown profiles are left untouched so the normal
    schema validation can report the bad value with the usual message.

    The input mapping is not mutated; a new dict is returned.
    """
    resolved = dict(data)
    if not resolved.get("profile_defaults"):
        return resolved
    preset = get_profile(resolved.get("profile"))
    if preset is None:
        return resolved

    for key, value in preset.config.items():
        resolved.setdefault(key, value)

    if resolved.get("phase") == "gpt":
        section = dict(preset.phase_config)
        explicit = resolved.get("phase_config")
        if isinstance(explicit, Mapping):
            section.update(explicit)
        resolved["phase_config"] = section

    return resolved


def profile_table_rows() -> list[dict[str, Any]]:
    """Rows for a profile comparison table (README/UI/benchmark report)."""
    return [
        {
            "profile": preset.name,
            "parameters": preset.estimate_parameters(),
            "n_layer": int(preset.phase_config["n_layer"]),
            "n_embd": int(preset.phase_config["n_embd"]),
            "context_length": int(preset.phase_config["context_length"]),
            "vocab_size": int(preset.phase_config["vocab_size"]),
            "batch_size": int(preset.config["batch_size"]),
            "runtime": preset.runtime,
            "capability": preset.capability,
        }
        for preset in list_profiles()
    ]
