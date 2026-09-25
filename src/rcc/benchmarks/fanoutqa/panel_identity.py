"""The registration a FanOutQA panel's prepared bytes carry.

The builder writes it into the panel and its fingerprint into the manifest; the
loader recomputes both from the same family registration before accepting a panel.
"""

from __future__ import annotations

from typing import Any

from rcc.benchmarks.fanoutqa.qwen_serving import serving_registration
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen import QWEN_FAMILY
from rcc.models.route import RouteFamily
from rcc.run import identity


def panel_registration(
    *, profile: BenchmarkProfile, family: RouteFamily = QWEN_FAMILY
) -> dict[str, Any]:
    """Return the registration one panel's prepared bytes carry."""
    return serving_registration(profile, family=family)


def panel_registration_sha(*, profile: BenchmarkProfile, family: RouteFamily = QWEN_FAMILY) -> str:
    """Return the fingerprint of that registration."""
    registration = panel_registration(profile=profile, family=family)
    return identity.fingerprint(registration, identity.json_compact_legacy, digest_chars=16)


__all__ = ("panel_registration", "panel_registration_sha")
