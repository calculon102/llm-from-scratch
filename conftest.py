"""Pytest configuration for the project test suite.

Loads the NiceGUI user-simulation plugin (headless, no browser required)
so tests can open app pages via the ``user`` fixture.  The main file is
configured in ``pyproject.toml`` (``main_file`` ini option).
"""

pytest_plugins = ["nicegui.testing.user_plugin"]
