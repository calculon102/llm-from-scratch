"""Pytest configuration for the project test suite.

Loads the NiceGUI user-simulation plugin (headless, no browser required)
so tests can open app pages via the ``user`` fixture.  The main file is
configured in ``pyproject.toml`` (``main_file`` ini option).
"""

# Force matplotlib's non-interactive backend before any test imports it:
# with a DISPLAY set but no usable X connection (e.g. CI), auto-detection
# would pick TkAgg and figure creation would raise TclError.
import matplotlib

matplotlib.use("Agg")

pytest_plugins = ["nicegui.testing.user_plugin"]
