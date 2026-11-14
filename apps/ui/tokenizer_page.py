"""Tokenizer compare page (T-205): character tokenizer vs. BPE, side by side.

Why this page exists
--------------------

A language model never sees text -- it sees **tokens**, the small pieces a
tokenizer cuts the text into before the model turns them into vectors (M3).
The tokenizer therefore decides what the model can learn cheaply: if
common German words like ``und`` or ``die`` are single tokens, the model
sees them often and can learn their behaviour; if the tokenizer splits
them into characters, every word becomes a long, sparse sequence and the
model has to spend many attention steps re-discovering spelling.

Two tokenizers make that trade-off visible:

* the **character tokenizer** (T-202) maps every Unicode character to one
  id.  It is tiny, exact and never produces an unknown token -- but a
  100-character sentence is 100 tokens, so sequences are long and each
  token carries almost no meaning.
* the **BPE tokenizer** (T-203, "byte pair encoding", trained from
  scratch here) starts from characters and repeatedly merges the most
  frequent adjacent pair into a new token.  After enough merges, frequent
  syllables and words (``en``, ``er``, ``und``) become single ids.  The
  same sentence now takes roughly half the tokens on a small German
  corpus, which means longer effective context and cheaper training.

The page runs the exact same objects the CLI persists (T-203's
``vocab.json``/``merges.json``): pick the bundled sample corpus (trained
on the fly at the chosen vocab size) or a saved artifact run from the
dropdown, type text, and press **Tokenize**.  Both tokenizations are
computed by :func:`core.tokenizer.metrics.compare_tokenizers`, the same
function the CLI uses for its ``metrics.json`` -- so the numbers here and
in an artifact folder cannot disagree.

What to look at
---------------

The two token tables show each token with escaped whitespace (``␣`` for a
space, ``\\n`` for a newline) and its id.  Hovering over a whitespace
token makes the BPE merge order visible; BPE never merges across those
chunk boundaries, which is why words stay whole while spaces remain
separate.  The metric lines report **chars/token** (how much text one id
carries), **tokens/word**, **OOV** (characters the tokenizer has never
seen), and a round-trip badge proving that decoding the ids returns the
input exactly.

The page deliberately needs no network and no model checkpoint: it loads
or trains a tokenizer locally, so it also works after a fresh clone with
only the sample files in ``data/sample_de``.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from core.tokenizer import (
    DEFAULT_SAMPLE_CORPUS,
    DEFAULT_SAMPLE_TEXT,
    CorpusOptions,
    TokenizerBundle,
    TokenizerError,
    compare_tokenizers,
    find_bundle_dirs,
    ingest_corpus,
    load_bundle,
    train_bundle,
)

__all__ = ["register_tokenizer_page"]

#: Sentinel select value for the on-the-fly sample-corpus tokenizer.  A
#: dedicated string keeps the option map simple (NiceGUI select values are
#: either sample or an artifact bundle directory path).
SAMPLE_SOURCE = "sample"

#: How many tokens each table renders.  A 4000-character text can produce
#: thousands of character tokens; the tables are for reading, not for
#: displaying an unbounded export.
TABLE_LIMIT = 300

#: Bundle cache: ``(source, vocab_size)`` -> trained/loaded bundle.  A page
#: can be re-rendered many times (tests, reconnects) and the sample bundle
#: is cheap to train but not free; the key includes the vocab size because
#: changing it asks for a different tokenizer.  Artifact bundles ignore the
#: vocab size (their vocabulary is fixed on disk), hence ``0`` in that key.
_BUNDLE_CACHE: dict[tuple[str, int], TokenizerBundle] = {}


def _source_options() -> dict[str, str]:
    """Return select options ``{value: label}`` for sample + artifact bundles.

    Artifact runs are discovered with :func:`find_bundle_dirs` (newest
    first), so a CLI run made a moment ago is immediately selectable here.
    Labels include the run folder name, which is unique and timestamped.
    """
    options: dict[str, str] = {SAMPLE_SOURCE: "Bundled sample corpus (data/sample_de)"}
    for bundle_dir in find_bundle_dirs():
        # ``find_bundle_dirs`` returns ``<run>/tokenizers``; the run folder
        # (the timestamped artifact directory) is its parent.
        run_name = bundle_dir.parent.name
        label = f"artifact: {run_name}"
        # Defensive: two runs with the same name (e.g. a copied artifact
        # folder) must still get distinct select values.
        if label in options:
            label = f"{label} ({bundle_dir})"
        options[str(bundle_dir)] = label
    return options


def _cached_bundle(source: str, vocab_size: int) -> TokenizerBundle:
    """Train the sample bundle or load an artifact bundle, cached per key."""
    key = (source, vocab_size if source == SAMPLE_SOURCE else 0)
    if key not in _BUNDLE_CACHE:
        if source == SAMPLE_SOURCE:
            corpus = ingest_corpus(DEFAULT_SAMPLE_CORPUS, CorpusOptions())
            _BUNDLE_CACHE[key] = train_bundle(
                corpus.text,
                vocab_size=vocab_size,
                min_frequency=2,
                max_token_length=32,
                source="data/sample_de",
            )
        else:
            _BUNDLE_CACHE[key] = load_bundle(Path(source))
    return _BUNDLE_CACHE[key]


def _visible(token: str) -> str:
    """Escape a token so whitespace and backslashes are readable in a table."""
    escaped = (
        token.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return escaped.replace(" ", "␣")


def _token_rows(tokenizer: object, text: str) -> tuple[list[dict[str, object]], int]:
    """Build table rows and the total token count for one tokenizer."""
    ids = tokenizer.encode(text)
    strings = tokenizer.token_strings(ids)
    rows = [
        {
            "index": index,
            "token": _visible(token),
            "token_id": token_id,
            "chars": len(token),
        }
        for index, (token_id, token) in enumerate(zip(ids, strings))
        if index < TABLE_LIMIT
    ]
    return rows, len(ids)


def _metrics_text(metrics: dict[str, object]) -> str:
    """One compact metric line for a tokenizer card."""
    return (
        f"{metrics['tokens']} tokens · "
        f"{metrics['compression_ratio']:.2f} chars/token · "
        f"{metrics['tokens_per_word']:.2f} tokens/word · "
        f"OOV {metrics['oov_chars_total']}"
    )


def tokenizer_page() -> None:
    """Build the ``/tokenizer`` compare page."""
    # Imported lazily: apps/ui/main.py imports this module inside
    # register_pages(), so a module-level import would be circular.
    from apps.ui.main import _shell

    with _shell("/tokenizer"):
        ui.label("Tokenizer compare").classes("text-h4")
        ui.label(
            "The same German text through a character tokenizer and a BPE "
            "tokenizer trained from scratch. Compare token counts, "
            "compression, OOV characters and exact round-trips."
        ).classes("text-subtitle1")
        ui.separator()

        source_options = _source_options()
        with ui.row().classes("w-full items-end gap-4 flex-wrap"):
            source_el = ui.select(
                options=source_options,
                value=SAMPLE_SOURCE,
                label="Tokenizer source",
            ).classes("w-96").mark("tok-source")
            # Only the sample tokenizer is trained here; artifact bundles
            # already contain a fixed vocabulary.  The control is disabled
            # for artifacts so the page cannot imply a retrain.
            vocab_el = (
                ui.number("sample vocab size", value=300, min=8, step=50)
                .classes("w-40")
                .mark("tok-vocab-size")
            )

        text_el = (
            ui.textarea(label="Text to tokenize", value=DEFAULT_SAMPLE_TEXT)
            .classes("w-full")
            .props("rows=5")
            .mark("tok-text")
        )

        with ui.row().classes("items-center q-my-sm"):
            compare_btn = (
                ui.button("Tokenize", icon="compare_arrows")
                .props("color=primary")
                .mark("tok-compare")
            )
        status_label = ui.label("").classes("text-caption").mark("tok-status")

        # Side-by-side cards: the grid auto-fits, so on a phone the two
        # tokenizers stack instead of squeezing each table to unreadable
        # width.
        with ui.grid(columns="repeat(auto-fit, minmax(340px, 1fr))").classes(
            "w-full gap-4"
        ):
            with ui.card().classes("w-full"):
                ui.label("Character tokenizer").classes("text-h6")
                char_metrics = ui.label("").classes("text-body2").mark(
                    "tok-char-metrics"
                )
                char_roundtrip = ui.badge("").mark("tok-char-roundtrip")
                char_count = ui.label("").classes("text-caption").mark("tok-char-count")
                char_table = ui.table(
                    rows=[],
                    columns=[
                        {"name": "index", "label": "#", "field": "index", "align": "right"},
                        {"name": "token", "label": "token", "field": "token", "align": "left"},
                        {"name": "token_id", "label": "id", "field": "token_id", "align": "right"},
                        {"name": "chars", "label": "chars", "field": "chars", "align": "right"},
                    ],
                    row_key="index",
                ).classes("w-full").mark("tok-char-table")

            with ui.card().classes("w-full"):
                ui.label("BPE tokenizer").classes("text-h6")
                bpe_metrics = ui.label("").classes("text-body2").mark(
                    "tok-bpe-metrics"
                )
                bpe_roundtrip = ui.badge("").mark("tok-bpe-roundtrip")
                bpe_count = ui.label("").classes("text-caption").mark("tok-bpe-count")
                bpe_table = ui.table(
                    rows=[],
                    columns=[
                        {"name": "index", "label": "#", "field": "index", "align": "right"},
                        {"name": "token", "label": "token", "field": "token", "align": "left"},
                        {"name": "token_id", "label": "id", "field": "token_id", "align": "right"},
                        {"name": "chars", "label": "chars", "field": "chars", "align": "right"},
                    ],
                    row_key="index",
                ).classes("w-full").mark("tok-bpe-table")

        def _set_roundtrip(badge: ui.badge, ok: bool) -> None:
            badge.text = "round-trip ok" if ok else "round-trip FAILED"
            badge.props(f"color={'positive' if ok else 'negative'}")

        def _fill_table(
            table: ui.table,
            count_label: ui.label,
            tokenizer: object,
            text: str,
        ) -> None:
            rows, total = _token_rows(tokenizer, text)
            table.rows = rows
            table.update()
            shown = min(total, TABLE_LIMIT)
            count_label.text = f"showing {shown} of {total} tokens"

        def compare() -> None:
            """Tokenize the textarea with both tokenizers and refresh the UI."""
            text = text_el.value or ""
            if not text.strip():
                status_label.text = "enter some text first"
                return
            source = source_el.value or SAMPLE_SOURCE
            try:
                vocab_size = int(vocab_el.value or 300)
                bundle = _cached_bundle(source, vocab_size)
                comparison = compare_tokenizers(bundle.char, bundle.bpe, text)
            except (TokenizerError, OSError, ValueError) as exc:
                status_label.text = f"error: {exc}"
                ui.notify(str(exc), type="negative")
                return

            char_metrics.text = _metrics_text(comparison["char"])
            bpe_metrics.text = _metrics_text(comparison["bpe"])
            _set_roundtrip(char_roundtrip, bool(comparison["char"]["roundtrip_ok"]))
            _set_roundtrip(bpe_roundtrip, bool(comparison["bpe"]["roundtrip_ok"]))
            _fill_table(char_table, char_count, bundle.char, text)
            _fill_table(bpe_table, bpe_count, bundle.bpe, text)

            text_info = comparison["text"]
            status_label.text = (
                f"{text_info['chars']} characters, {text_info['words']} words · "
                f"BPE uses {comparison['compression_gain']:.2f}× fewer tokens "
                f"than char ({comparison['token_reduction']:.1f}% reduction)"
            )

        def source_changed(value: str | None) -> None:
            """Disable the vocab control for fixed-vocabulary artifact bundles."""
            vocab_el.set_enabled((value or SAMPLE_SOURCE) == SAMPLE_SOURCE)

        compare_btn.on_click(compare)
        source_el.on_value_change(lambda event: source_changed(event.value))
        source_changed(source_el.value)
        # Populate the page immediately: an empty compare view on first
        # visit would make the page look broken.  The default source is the
        # sample corpus, so this works on a fresh clone.
        compare()


def register_tokenizer_page() -> None:
    """Register the ``/tokenizer`` route (called from ``main.register_pages``).

    Like :func:`apps.ui.xor_page.register_xor_page`, registration happens
    from ``register_pages`` rather than at module import: the headless UI
    tests reset the app and re-execute ``main.py`` per test, while this
    module is cached in ``sys.modules`` and its decorators would not run
    again.
    """
    @ui.page("/tokenizer", title="Tokenizer – llm-from-scratch")
    def tokenizer_page_route() -> None:
        tokenizer_page()
