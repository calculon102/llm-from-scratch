"""Smoke test: the package installs and imports cleanly."""

import core


def test_import_and_version():
    assert hasattr(core, "__version__")
    assert core.__version__
