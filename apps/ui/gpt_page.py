"""Mini GPT generation page (T-406).

The page is the interactive front-end for the M4 stack:

* a run selector discovers every ``<run>/gpt`` artifact written by the
  training phase (T-404, :func:`core.gpt.pipeline.find_gpt_dirs`);
* loading a run restores the model and its BPE tokenizer, so generation is
  entirely local (no API calls, no network);
* the prompt plus greedy / temperature / top-k / top-p controls call the
  T-405 sampler (:func:`core.gpt.sampling.generate`) — the same code the
  command line uses, so the page and scripts agree;
* the generated continuation and its token count are shown inline.

If no GPT run exists yet, the page explains how to train one instead of
showing an empty playground.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from core.gpt.pipeline import (
    GPTArtifacts,
    GPTPipelineError,
    find_gpt_dirs,
    load_gpt_artifacts,
)
from core.gpt.sampling import generate

__all__ = ["register_gpt_page", "gpt_page"]

#: Loaded artifacts are cached by path + mtime.  A newly trained run has a
#: new directory, and re-training into the same folder invalidates the mtime.
_ARTIFACT_CACHE: dict[tuple[str, int], GPTArtifacts] = {}


def _source_options() -> dict[str, str]:
    """Return ``{gpt_dir: label}`` for all discovered runs."""
    options: dict[str, str] = {}
    for gpt_dir in find_gpt_dirs():
        run_name = gpt_dir.parent.name
        label = f"{run_name}  ({gpt_dir})"
        if label in options.values():
            label = f"{label} [{len(options)}]"
        options[str(gpt_dir)] = label
    return options


def _cached_artifacts(source: str) -> GPTArtifacts:
    path = Path(source)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    key = (str(path), mtime)
    if key not in _ARTIFACT_CACHE:
        _ARTIFACT_CACHE[key] = load_gpt_artifacts(path)
    return _ARTIFACT_CACHE[key]


def _meta_text(artifacts: GPTArtifacts) -> str:
    """One-line summary of the loaded run."""
    metadata = artifacts.metadata
    parts = [
        f"{artifacts.vocab_size} vocab",
        f"context {artifacts.context_length}",
        f"{artifacts.model.num_parameters():,} params",
    ]
    run_name = metadata.get("run_name")
    if run_name:
        parts.insert(0, f"run: {run_name}")
    if metadata.get("best_val") is not None:
        parts.append(f"best val {float(metadata['best_val']):.4f}")
    if metadata.get("val_perplexity") is not None:
        parts.append(f"perplexity {float(metadata['val_perplexity']):.2f}")
    return " · ".join(parts)


def _no_artifacts_message() -> None:
    """Explain the missing prerequisite instead of rendering an empty page."""
    ui.label("No GPT runs found yet").classes("text-h6 q-mt-md")
    ui.label(
        "Train the tiny GPT first (the Jobs page can start any config), "
        "then come back; the run appears automatically."
    ).classes("q-my-sm")
    ui.code("python -m scripts.train --config configs/gpt_tiny_debug.yaml").classes(
        "w-full"
    )


def gpt_page() -> None:
    """Build the ``/gpt`` generation playground."""
    # Imported lazily: apps/ui/main.py imports this module inside
    # register_pages(), so a module-level import would be circular.
    from apps.ui.main import _shell

    options = _source_options()
    with _shell("/gpt"):
        ui.label("Mini GPT generation").classes("text-h4")
        ui.label(
            "Continue a German prompt with the decoder-only transformer trained "
            "in M4. Pick a run, then explore greedy, temperature, top-k and "
            "top-p sampling."
        ).classes("text-subtitle1")
        ui.separator()

        if not options:
            _no_artifacts_message()
            return

        source_values = list(options)
        with ui.row().classes("w-full items-end gap-4 flex-wrap"):
            source_el = (
                ui.select(
                    options=options,
                    value=source_values[0],
                    label="GPT run",
                )
                .classes("w-[34rem] max-w-full")
                .mark("gpt-source")
            )
            meta_label = ui.label("").classes("text-caption").mark("gpt-meta")

        prompt_el = (
            ui.textarea(
                label="Prompt",
                value="Die",
            )
            .props("outlined autogrow")
            .classes("w-full")
            .mark("gpt-prompt")
        )

        with ui.row().classes("w-full items-end gap-4 flex-wrap"):
            max_tokens_el = (
                ui.number("max new tokens", value=32, min=0, max=512, step=1)
                .classes("w-36")
                .mark("gpt-max-new-tokens")
            )
            temperature_el = (
                ui.number("temperature", value=1.0, min=0.0, max=5.0, step=0.1)
                .classes("w-36")
                .mark("gpt-temperature")
            )
            top_k_el = (
                ui.number("top-k (0 = off)", value=0, min=0, max=1000, step=1)
                .classes("w-36")
                .mark("gpt-top-k")
            )
            top_p_el = (
                ui.number("top-p", value=1.0, min=0.0, max=1.0, step=0.01)
                .classes("w-36")
                .mark("gpt-top-p")
            )
            seed_el = (
                ui.number("seed", value=42, min=0, step=1)
                .classes("w-32")
                .mark("gpt-seed")
            )
            greedy_el = ui.switch("greedy").mark("gpt-greedy")
            generate_btn = (
                ui.button("Generate", icon="auto_awesome")
                .props("color=primary")
                .mark("gpt-generate")
            )

        status_label = ui.label("").classes("text-body2 q-my-sm").mark("gpt-status")
        output_label = (
            ui.label("")
            .classes("text-body1 whitespace-pre-wrap w-full q-pa-md bg-grey-10 rounded")
            .mark("gpt-output")
        )

        def load_source(value: str | None) -> None:
            source = value or source_values[0]
            try:
                artifacts = _cached_artifacts(source)
            except (GPTPipelineError, OSError, ValueError) as exc:
                meta_label.text = f"error: {exc}"
                ui.notify(str(exc), type="negative")
                generate_btn.set_enabled(False)
                return
            meta_label.text = _meta_text(artifacts)
            if artifacts.tokenizer is None:
                status_label.text = (
                    "this artifact has no tokenizer; retrain it with T-404"
                )
                generate_btn.set_enabled(False)
            else:
                status_label.text = ""
                generate_btn.set_enabled(True)

        def generate_clicked() -> None:
            source = source_el.value or source_values[0]
            try:
                artifacts = _cached_artifacts(source)
            except (GPTPipelineError, OSError, ValueError) as exc:
                status_label.text = f"error: {exc}"
                ui.notify(str(exc), type="negative")
                return
            tokenizer = artifacts.tokenizer
            if tokenizer is None:
                status_label.text = "the selected run has no tokenizer"
                return

            bpe = tokenizer.bpe
            prompt = prompt_el.value or ""
            prompt_ids = bpe.encode(prompt)
            if not prompt_ids:
                prompt_ids = [int(bpe.bos_id)]
            try:
                ids = generate(
                    artifacts.model,
                    prompt_ids,
                    max_new_tokens=int(max_tokens_el.value or 0),
                    temperature=float(
                        0.0 if temperature_el.value is None else temperature_el.value
                    ),
                    top_k=int(top_k_el.value or 0),
                    top_p=float(1.0 if top_p_el.value is None else top_p_el.value),
                    greedy=bool(greedy_el.value),
                    eos_id=int(bpe.eos_id),
                    seed=None if seed_el.value is None else int(seed_el.value),
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                status_label.text = f"sampling error: {exc}"
                ui.notify(str(exc), type="negative")
                return

            text = bpe.decode(ids, skip_special_tokens=True)
            new_tokens = len(ids) - len(prompt_ids)
            output_label.text = text
            mode = "greedy" if greedy_el.value else "sampled"
            status_label.text = (
                f"generated {new_tokens} tokens ({mode}) · "
                f"{len(text)} characters"
            )

        generate_btn.on_click(generate_clicked)
        source_el.on_value_change(lambda event: load_source(event.value))
        load_source(source_el.value)


def register_gpt_page() -> None:
    """Register the ``/gpt`` route (called from ``main.register_pages``).

    Registration happens from ``register_pages`` rather than at module
    import because the headless UI tests reset the app and re-execute
    ``main.py`` per test while this module stays cached in ``sys.modules``.
    """

    @ui.page("/gpt", title="Mini GPT – llm-from-scratch")
    def gpt_page_route() -> None:
        gpt_page()
