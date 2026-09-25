"""Validate the shared FanOutQA source bundle against one profile's pins.

The bundle is rehashed whichever family is going to tokenize it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa.natural_panel import validate_natural_bundle
from rcc.benchmarks.protocol import BenchmarkProfile


def validate_shared_source_bundle(
    bundle_root: Path,
    *,
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    """Rehash one profile's source bundle and return its manifest."""
    return validate_natural_bundle(Path(bundle_root), profile=profile)


__all__ = ("validate_shared_source_bundle",)
