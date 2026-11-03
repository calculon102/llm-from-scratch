"""CLI: start the NiceGUI web app shell (T-007).

Usage:
    python -m scripts.ui [--host 127.0.0.1] [--port 8080] [--reload]

Importing ``apps.ui.main`` registers all phase routes; ``run_app`` then
starts the server.  The app is dark by default and responsive (the left
drawer collapses to an overlay on mobile).
"""

from __future__ import annotations

import argparse
import sys

from apps.ui.main import run_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the NiceGUI web app.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (development)."
    )
    args = parser.parse_args(argv)
    run_app(host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
