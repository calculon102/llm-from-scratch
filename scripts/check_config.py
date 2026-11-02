"""CLI: load and validate a YAML config file, print the resolved config.

Usage:
    python -m scripts.check_config --config configs/tiny_debug.yaml

Exits 0 and prints the validated config (with defaults filled) as JSON on
success; exits 1 with a clear error message on failure.
"""

from __future__ import annotations

import argparse
import json
import sys

from core.config import ConfigError, add_config_arg, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load and validate a YAML config file."
    )
    add_config_arg(parser)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(cfg, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
