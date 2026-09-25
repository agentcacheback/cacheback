"""Per-panel construction pins: what a panel's build is required to produce.

Digests are registered per qid and per lane, bound by ``profile_id`` and
``PANEL_PINS_SHA256``, so a panel load that reconstructs different bytes raises.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

_PINS_PATH = Path(__file__).parent / "panel_pins.json"
PANEL_PINS_SHA256 = "36da0c90f0b0fae9ffae4f9cf2e913717215924b9230ff6bd81da6fe320435a6"
PANEL_PINS_SCHEMA = "fanoutqa-panel-pins-v2"
_LANE_FIELDS = {
    "gemma": "gemma_item_construction_sha256",
    "ministral": "ministral_item_construction_sha256",
}


def _load() -> dict[str, Any]:
    raw = _PINS_PATH.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != PANEL_PINS_SHA256:
        raise RuntimeError(
            f"panel_pins.json sha {observed[:16]} != pinned {PANEL_PINS_SHA256[:16]}; "
            "refusing to construct a registered panel from unpinned bytes"
        )
    body = cast(dict[str, Any], json.loads(raw))
    if body.get("schema") != PANEL_PINS_SCHEMA:
        raise RuntimeError("panel_pins.json carries an unregistered schema")
    return body


def _panel(panel_id: str) -> dict[str, Any]:
    panels = cast(dict[str, Any], _load()["panels"])
    try:
        return cast(dict[str, Any], panels[panel_id])
    except KeyError as exc:
        raise RuntimeError(
            f"{panel_id!r} has no pinned construction identity; a panel must register one "
            f"before it can be constructed (registered: {sorted(panels)})"
        ) from exc


def registered_item_construction_sha256(panel_id: str, lane: str) -> dict[str, str]:
    """Return one lane's per-qid construction digests for one panel."""
    try:
        field = _LANE_FIELDS[lane]
    except KeyError as exc:
        raise ValueError(f"unknown construction lane {lane!r}") from exc
    table = cast(Mapping[str, str], _panel(panel_id).get(field) or {})
    return {str(qid): str(value) for qid, value in table.items()}


def require_registered_identity(
    panel_id: str,
    lane: str,
    rebuilt: Mapping[str, str],
    *,
    required: bool,
) -> None:
    """Raise when a panel's reconstruction differs from its pinned identity."""
    registered = registered_item_construction_sha256(panel_id, lane)
    if not registered:
        if required:
            raise RuntimeError(
                f"{panel_id!r} registers no reconstructed-item identity for the {lane} "
                "lane, so a drifted construction could not be detected; refusing to "
                "serve it"
            )
        return
    drift = [
        f"{qid}: rebuilt {digest[:16]} != registered "
        f"{registered.get(str(qid), '<unregistered>')[:16]}"
        for qid, digest in rebuilt.items()
        if digest != registered.get(str(qid))
    ]
    if drift:
        raise RuntimeError(
            f"{panel_id}: reconstructed {lane} items differ from the registered panel "
            f"identity ({'; '.join(drift[:3])}); the construction inputs or the "
            "construction code moved under a sealed panel"
        )


__all__ = (
    "PANEL_PINS_SCHEMA",
    "PANEL_PINS_SHA256",
    "registered_item_construction_sha256",
    "require_registered_identity",
)
