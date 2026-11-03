"""App shell tests (T-007): every phase route is reachable.

Uses the NiceGUI headless user simulation (``user`` fixture from
``nicegui.testing.user_plugin``, loaded in ``conftest.py``).  The ``user``
fixture executes ``apps/ui/main.py`` (``main_file`` ini option) so all
routes are registered, then each test opens a route and checks the page
content.

The ``user`` fixture and the ``User`` methods are async; they are run by
the anyio pytest plugin (a hard dependency of NiceGUI).  Requesting
``anyio_backend`` is what makes the plugin wrap the async fixture.
"""

from __future__ import annotations

import pytest

from nicegui import ui
from nicegui.testing import User

from apps.ui.main import PHASE_PAGES

# (route, text that must be visible on the page)
ROUTES: list[tuple[str, str]] = [("/", "llm-from-scratch")] + [
    (phase.route, phase.title) for phase in PHASE_PAGES
]


@pytest.mark.anyio
@pytest.mark.parametrize("route,expected", ROUTES)
async def test_route_reachable(
    user: User, anyio_backend: str, route: str, expected: str
) -> None:
    await user.open(route)
    await user.should_see(expected)


@pytest.mark.anyio
@pytest.mark.parametrize("phase", PHASE_PAGES, ids=[p.route for p in PHASE_PAGES])
async def test_phase_page_lists_tickets(user: User, anyio_backend: str, phase) -> None:
    await user.open(phase.route)
    # Quasar renders table rows client-side, so check the row data on the
    # Python-side table element instead of looking for row text.
    table = next(iter(user.find(kind=ui.table).elements))
    assert [row["ticket"] for row in table.rows] == [
        ticket_id for ticket_id, _title, _priority in phase.tickets
    ]


@pytest.mark.anyio
async def test_home_links_to_all_phases(user: User, anyio_backend: str) -> None:
    await user.open("/")
    for phase in PHASE_PAGES:
        await user.should_see(phase.title)


@pytest.mark.anyio
async def test_shell_has_dark_mode_toggle(user: User, anyio_backend: str) -> None:
    await user.open("/")
    await user.should_see(kind=ui.switch)


@pytest.mark.anyio
async def test_header_has_menu_button(user: User, anyio_backend: str) -> None:
    # The menu button is the drawer's only opener on mobile (off-canvas).
    await user.open("/")
    await user.should_see(marker="app-menu-button")
