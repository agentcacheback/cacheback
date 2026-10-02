"""Run the package's setup diagnostic without loading model weights."""

import argparse
import json
import sys

from rclc.diagnostics import check


def main() -> None:
    """Write a JSON diagnostic and exit nonzero when setup needs attention."""
    parser = argparse.ArgumentParser(prog="python -m rclc", description=__doc__)
    parser.add_argument("command", choices=("doctor",))
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--allow-unstable", action="store_true")
    options = parser.parse_args()
    report = check(backend=options.backend, allow_unstable=options.allow_unstable)
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
