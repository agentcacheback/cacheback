"""Command line for building the FanOutQA source audit and prepared panel.

``audit-source`` writes the panel's ``source_audit`` and ``prepare`` writes its
``items.pkl`` and manifest, both from the natural text bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, FANOUTQA_NATURAL_DEV50_LATENT
from rcc.benchmarks.fanoutqa.source_audit import validate_source_commit
from rcc.benchmarks.fanoutqa.source_build import audit_source_panel, prepare_panel
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.run.plan import route_family_for_lane

PROFILES = {
    "fanoutqa-natural-dev50": FANOUTQA_NATURAL_DEV50,
    "fanoutqa-natural-dev50-latent": FANOUTQA_NATURAL_DEV50_LATENT,
}


def resolve_profile(name: str) -> BenchmarkProfile:
    """Return the profile one ``--profile`` name selects."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unregistered benchmark profile {name!r}; choose from {sorted(PROFILES)}"
        ) from exc


def _emit(payload: Mapping[str, object]) -> int:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def _family():
    """Return the route family named by RCC_FANOUT_FAMILY, Qwen when it is unset."""
    return route_family_for_lane(os.environ.get("RCC_FANOUT_FAMILY", "qwen"))


def _natural_bundle(args: argparse.Namespace, profile: BenchmarkProfile) -> Path:
    """Return the natural bundle path, raising when it was not given."""
    bundle = cast(Path | None, args.natural_bundle)
    if bundle is None:
        raise ValueError(
            f"{profile.profile_id} builds from a natural bundle; pass --natural-bundle"
        )
    return bundle


def _audit_source(args: argparse.Namespace, profile: BenchmarkProfile) -> int:
    audit = audit_source_panel(
        args.run_root,
        source_commit=validate_source_commit(args.source_commit),
        profile=profile,
        family=_family(),
        natural_bundle=_natural_bundle(args, profile),
    )
    return _emit(
        {
            "audit_fingerprint": audit["audit_fingerprint"],
            "construction_sha256": audit["construction_sha256"],
            "panel": audit["panel"],
            "schema": audit["schema"],
        }
    )


def _prepare(args: argparse.Namespace, profile: BenchmarkProfile) -> int:
    manifest = prepare_panel(
        args.run_root,
        source_commit=validate_source_commit(args.source_commit),
        profile=profile,
        family=_family(),
        natural_bundle=_natural_bundle(args, profile),
    )
    return _emit({key: value for key, value in manifest.items() if key != "qids"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rcc-fanoutqa-build", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit-source", "prepare"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--profile", required=True, choices=sorted(PROFILES))
        sub.add_argument("--run-root", type=Path, required=True)
        sub.add_argument("--source-commit", required=True)
        sub.add_argument("--natural-bundle", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one build command against the profile named on the command line."""
    args = _parser().parse_args(argv)
    profile = resolve_profile(args.profile)
    handler = {
        "audit-source": _audit_source,
        "prepare": _prepare,
    }[args.command]
    return handler(args, profile)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"FATAL: {exc}\n")
        raise SystemExit(2) from exc


__all__ = ("PROFILES", "main", "resolve_profile")
