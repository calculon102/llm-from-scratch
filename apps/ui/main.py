"""NiceGUI app shell: dark, responsive, one route per pipeline phase (T-007).

Start the app with:

    python -m scripts.ui [--host 127.0.0.1] [--port 8080]

Routes
------
``/``            home / dashboard (M0)
``/xor``         M1 -- XOR learning module
``/tokenizer``   M2 -- tokenizer module
``/embeddings``  M3 -- embeddings module
``/gpt``         M4 -- mini GPT module
``/scaling``     M5 -- scaling on the 7900 XTX
``/capstone``    M6 -- capstone & docs

The phase pages are shells for now: they show the phase overview and its
ticket list.  Phase-specific UI (controls, plots, generation) lands in the
later tickets (T-104, T-205, T-305, T-406, T-604).
"""

from __future__ import annotations

from dataclasses import dataclass

from nicegui import ui
from nicegui.binding import bind

from core.device import DeviceError, describe_device, get_device_info

__all__ = ["Phase", "PHASE_PAGES", "register_pages", "run_app"]


@dataclass(frozen=True)
class Phase:
    """One pipeline phase with its UI route and ticket list.

    ``tickets`` is a tuple of ``(ticket id, title, priority)`` triples.
    """

    route: str
    milestone: str
    title: str
    blurb: str
    ui_ticket: str | None
    tickets: tuple[tuple[str, str, str], ...]


PHASE_PAGES: tuple[Phase, ...] = (
    Phase(
        route="/xor",
        milestone="M1",
        title="XOR Learning",
        blurb=(
            "First end-to-end training loop: a 2-layer MLP learns XOR, "
            "with loss curves, decision boundary and activation plots."
        ),
        ui_ticket="T-104",
        tickets=(
            ("T-101", "Implement XOR dataset + 2-layer MLP baseline", "P0"),
            ("T-102", "Add XOR decision boundary and activation plots", "P0"),
            ("T-103", "Add finite-difference gradient check mode", "P1"),
            ("T-104", "Build XOR UI controls and outputs", "P0"),
            ("T-105", "Write docs/math/phase1_xor.md", "P0"),
        ),
    ),
    Phase(
        route="/tokenizer",
        milestone="M2",
        title="Tokenizer",
        blurb=(
            "Noisy DE-first corpus ingestion, a character tokenizer and a "
            "BPE tokenizer trained from scratch, with a compare page."
        ),
        ui_ticket="T-205",
        tickets=(
            ("T-201", "Implement corpus ingestion (noisy DE-first)", "P0"),
            ("T-202", "Build character tokenizer with tests", "P0"),
            ("T-203", "Build BPE tokenizer trainer from scratch", "P0"),
            ("T-204", "Add tokenizer metrics", "P1"),
            ("T-205", "Build tokenizer UI compare page", "P0"),
            ("T-206", "Write docs/math/phase2_tokenizer.md", "P0"),
        ),
    ),
    Phase(
        route="/embeddings",
        milestone="M3",
        title="Embeddings",
        blurb=(
            "Skip-gram with negative sampling, nearest-neighbor search and "
            "optional UMAP/t-SNE projection in an explorer UI."
        ),
        ui_ticket="T-305",
        tickets=(
            ("T-301", "Implement skip-gram pair generator + negative sampler", "P1"),
            ("T-302", "Implement skip-gram model + trainer integration", "P1"),
            ("T-303", "Add nearest-neighbor cosine tooling", "P1"),
            ("T-304", "Add UMAP/t-SNE projection utility", "P2"),
            ("T-305", "Build embeddings UI explorer", "P1"),
            ("T-306", "Write docs/math/phase3_embeddings.md", "P1"),
        ),
    ),
    Phase(
        route="/gpt",
        milestone="M4",
        title="Mini GPT",
        blurb=(
            "Decoder-only transformer with causal attention: training, "
            "perplexity evaluation and interactive text generation."
        ),
        ui_ticket="T-406",
        tickets=(
            ("T-401", "Implement causal self-attention block with tests", "P0"),
            ("T-402", "Implement decoder transformer stack + LM head", "P0"),
            ("T-403", "Implement LM dataset chunking + train/val split", "P0"),
            ("T-404", "Add GPT training + perplexity evaluation", "P0"),
            ("T-405", "Implement sampling modes", "P1"),
            ("T-406", "Build GPT generation UI page", "P1"),
            ("T-407", "Write docs/math/phase4_transformer.md", "P0"),
        ),
    ),
    Phase(
        route="/scaling",
        milestone="M5",
        title="Scaling (7900 XTX)",
        blurb=(
            "Scale profiles from tiny_debug to max_7900xtx, throughput and "
            "memory instrumentation, OOM fallbacks and a benchmark report."
        ),
        ui_ticket=None,
        tickets=(
            ("T-501", "Define scaling profile YAMLs", "P1"),
            ("T-502", "Add throughput and memory instrumentation", "P1"),
            ("T-503", "Add OOM-safe fallback hints + auto-save on interrupt", "P1"),
            ("T-504", "Benchmark profile report on 7900 XTX", "P1"),
        ),
    ),
    Phase(
        route="/capstone",
        milestone="M6",
        title="Capstone & Docs",
        blurb=(
            "Offline corpus cache, preprocess profiles, one-command "
            "end-to-end pipeline, runs browser and the scaling guide."
        ),
        ui_ticket="T-604",
        tickets=(
            ("T-601", "Add corpus downloader + local cache manifest (DE-first)", "P1"),
            ("T-602", "Add preprocess profiles", "P1"),
            ("T-603", "Build end-to-end pipeline command", "P1"),
            ("T-604", "Build UI runs/artifacts browser page", "P2"),
            ("T-605", "Write 7900 XTX scaling guide", "P1"),
            ("T-606", "Create MVP release checklist + demo script", "P1"),
        ),
    ),
)


def _device_line() -> str:
    """One-line device/precision summary for the home page (T-006)."""
    try:
        info = get_device_info({"device": "auto", "amp": False})
        return describe_device(info)
    except DeviceError as exc:
        return f"device unavailable: {exc}"


# ---------------------------------------------------------------------------
# Shared shell (dark, responsive)
# ---------------------------------------------------------------------------

def _shell(active_route: str) -> ui.column:
    """Build the shared shell: dark mode, header, left drawer, content column.

    ``active_route`` marks the current page in the drawer navigation.
    """
    dark = ui.dark_mode(True)
    drawer = ui.left_drawer().classes("w-56")
    with ui.header().classes("items-center"):
        # Menu button toggles the drawer (its only opener on mobile, where
        # the drawer is an off-canvas overlay).
        ui.button(icon="menu", on_click=lambda: drawer.toggle()).props(
            "flat dense text-color=white"
        ).mark("app-menu-button")
        ui.icon("psychology")
        ui.label("llm-from-scratch").classes("text-h6")
        ui.space()
        toggle = ui.switch().props("dense icon=dark_mode")
        bind(toggle, "value", dark, "value")
    with drawer:
        ui.label("Phases").classes("text-subtitle1 q-mx-md q-mt-md")
        ui.link("Home", target="/").classes(
            "q-mx-md q-my-xs" + (" q-text-bold" if active_route == "/" else "")
        )
        for phase in PHASE_PAGES:
            ui.link(phase.title, target=phase.route).classes(
                "q-mx-md q-my-xs"
                + (" q-text-bold" if active_route == phase.route else "")
            )
    return ui.column().classes("w-full q-pa-md")


def _phase_card(phase: Phase) -> None:
    """Card for one phase on the home page."""
    with ui.card().classes("w-full"):
        ui.row().classes("items-center w-full")
        ui.badge(phase.milestone, outline=True)
        ui.label(phase.title).classes("text-subtitle1")
        ui.label(phase.blurb).classes("text-caption q-my-sm")
        ui.row().classes("items-center w-full")
        ui.space()
        ui.button("Open", icon="open_in_new", on_click=lambda: ui.navigate.to(phase.route))


def _ticket_table(phase: Phase) -> None:
    """Ticket list for one phase page."""
    ui.table(
        rows=[
            {"id": ticket_id, "ticket": ticket_id, "title": title, "priority": priority}
            for ticket_id, title, priority in phase.tickets
        ],
        columns=[
            {"name": "ticket", "label": "Ticket", "field": "ticket"},
            {"name": "title", "label": "Title", "field": "title"},
            {"name": "priority", "label": "Priority", "field": "priority"},
        ],
        row_key="id",
    ).classes("w-full")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _home_page() -> None:
    """Home / dashboard page (M0)."""
    with _shell("/"):
        ui.label("llm-from-scratch").classes("text-h4")
        ui.label(
            "Educational, Python-only, offline-capable LLM pipeline. "
            "German-first corpus, ROCm with CPU fallback."
        ).classes("text-subtitle1")
        ui.label(_device_line()).classes("text-caption")
        ui.separator()
        ui.label("Pipeline phases").classes("text-h6")
        # Responsive grid: as many columns as fit (min 250px each),
        # so it collapses to one column on mobile.
        with ui.grid(columns="repeat(auto-fit, minmax(250px, 1fr))").classes("w-full gap-4"):
            for phase in PHASE_PAGES:
                _phase_card(phase)


def _phase_page(phase: Phase) -> None:
    """Phase shell page: overview, status badge, ticket list."""
    with _shell(phase.route):
        ui.row().classes("items-center")
        ui.badge(phase.milestone, outline=True)
        ui.label(phase.title).classes("text-h5")
        ui.label(phase.blurb).classes("q-my-sm")
        if phase.ui_ticket:
            ui.badge(f"shell only – phase UI lands in {phase.ui_ticket}", color="warning")
        else:
            ui.badge("shell only – no dedicated phase UI planned", color="warning")
        ui.separator()
        ui.label(f"Tickets ({phase.milestone})").classes("text-h6")
        _ticket_table(phase)


_PAGES_REGISTERED = False


def register_pages() -> None:
    """Register all app routes: home plus one route per pipeline phase."""
    global _PAGES_REGISTERED
    if _PAGES_REGISTERED:
        return
    _PAGES_REGISTERED = True

    @ui.page("/", title="Home – llm-from-scratch")
    def home_page() -> None:
        _home_page()

    for phase in PHASE_PAGES:
        @ui.page(phase.route, title=f"{phase.title} – llm-from-scratch")
        def phase_page(_phase: Phase = phase) -> None:
            _phase_page(_phase)


def run_app(
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    reload: bool = False,
    show: bool = False,
) -> None:
    """Start the NiceGUI server for the app shell."""
    ui.run(
        title="llm-from-scratch",
        dark=True,
        host=host,
        port=port,
        reload=reload,
        show=show,
    )


register_pages()


if __name__ == "__main__":
    run_app()
