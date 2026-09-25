"""Move a layer bank to and from plain tensors on disk.

A stored payload is checked field by field before any row dataclass is built
from it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch

from rcc.transforms.select.core.types import LayerDescriptor

LAYER_BANK_SCHEMA = "query-support-layer-bank-v4"
BANK_SCOPE_GLOBAL_ONLY = "global-only"
BANK_SCOPE_ALL_LAYER = "all-layer"


def _tag(order: float) -> str:
    return f"{order:g}".replace(".", "p")


def _tensor_ref(
    tensors: dict[str, torch.Tensor],
    name: str,
    value: torch.Tensor | None,
) -> str | None:
    if value is None:
        return None
    if name in tensors:
        raise RuntimeError(f"layer bank tensor name is duplicated: {name}")
    tensors[name] = value.detach().to(dtype=torch.float32).cpu().clone()
    return name


def _row_payload(row: Any, row_index: int, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    prefix = f"row{row_index}"
    refs: dict[str, Any] = {
        "snap": _tensor_ref(tensors, f"{prefix}.snap", row.snap),
        "snap_chunks": [
            _tensor_ref(tensors, f"{prefix}.snap_chunk{chunk_index}", chunk)
            for chunk_index, chunk in enumerate(row.snap_chunks)
        ],
        "energy": _tensor_ref(tensors, f"{prefix}.energy", row.energy),
        "row_energy": _tensor_ref(tensors, f"{prefix}.row_energy", row.row_energy),
        "support_e": {
            _tag(order): _tensor_ref(tensors, f"{prefix}.support_e.p{_tag(order)}", value)
            for order, value in sorted(row.support_e.items())
        },
        "support_eprime": {
            _tag(order): _tensor_ref(tensors, f"{prefix}.support_eprime.p{_tag(order)}", value)
            for order, value in sorted(row.support_eprime.items())
        },
        "support_r": {
            _tag(order): _tensor_ref(tensors, f"{prefix}.support_r.p{_tag(order)}", value)
            for order, value in sorted(row.support_r.items())
        },
        "support_einf": _tensor_ref(tensors, f"{prefix}.support_einf", row.support_einf),
        "support_rinf": _tensor_ref(tensors, f"{prefix}.support_rinf", row.support_rinf),
        "prepool_e2": _tensor_ref(tensors, f"{prefix}.prepool_e2", row.prepool_e2),
        "prepool_r2": _tensor_ref(tensors, f"{prefix}.prepool_r2", row.prepool_r2),
    }
    descriptor = row.descriptor
    return {
        "schema": LAYER_BANK_SCHEMA,
        "descriptor": {
            "layer_idx": descriptor.layer_idx,
            "layer_type": descriptor.layer_type,
            "logical_memory_length": descriptor.logical_memory_length,
            "physical_key_length": descriptor.physical_key_length,
            "absolute_key_offset": descriptor.absolute_key_offset,
            "window": descriptor.window,
            "query_length": descriptor.query_length,
        },
        "heads": row.heads,
        "query_rows": row.query_rows,
        "kv_groups": row.kv_groups,
        "tensors": refs,
    }


def payload_from_bank(bank: Any) -> dict[str, Any]:
    """Encode a bank as dictionaries, scalars, lists, and host tensors only."""
    tensors: dict[str, torch.Tensor] = {}
    rows = [_row_payload(row, index, tensors) for index, row in enumerate(bank.rows)]
    if not rows:
        raise RuntimeError("cannot serialize an empty layer bank")
    first = bank.rows[0]
    return {
        "schema": LAYER_BANK_SCHEMA,
        "scope": bank.scope,
        "pool_kernel": bank.pool_kernel,
        "energy_pool_kernel": bank.energy_pool_kernel,
        "row_energy_pool_kernel": bank.row_energy_pool_kernel,
        "support_orders": list(bank.support_orders),
        # kv_groups is per row rather than part of the shared geometry, because
        # a hybrid model mixes GQA factors per layer type.
        "geometry": {
            "logical_memory_length": first.descriptor.logical_memory_length,
            "heads": first.heads,
            "query_rows": first.query_rows,
            "layer_count": len(rows),
        },
        "tensor_roster": list(tensors),
        "tensors": tensors,
        "rows": rows,
    }


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise RuntimeError(f"layer bank {label} schema is invalid")


def _int(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"layer bank {label} is invalid")
    return value


def _vector(tensors: Mapping[str, Any], name: Any, label: str, length: int) -> torch.Tensor:
    if not isinstance(name, str) or name not in tensors:
        raise RuntimeError(f"layer bank {label} tensor reference is invalid")
    value = tensors[name]
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 1
        or int(value.shape[0]) != length
        or not bool(torch.isfinite(value).all())
    ):
        raise RuntimeError(f"layer bank {label} tensor is malformed")
    return value


def validate_payload(payload: object) -> dict[str, Any]:
    """Check every field of a stored bank payload before anything scores it."""
    if not isinstance(payload, dict):
        raise RuntimeError("layer bank payload is not a dictionary")
    payload = cast(dict[str, Any], payload)
    _require_keys(
        payload,
        {
            "schema",
            "scope",
            "pool_kernel",
            "energy_pool_kernel",
            "row_energy_pool_kernel",
            "support_orders",
            "geometry",
            "tensor_roster",
            "tensors",
            "rows",
        },
        "top-level",
    )
    if payload["schema"] != LAYER_BANK_SCHEMA:
        raise RuntimeError(f"unknown layer bank schema {payload.get('schema')!r}")
    if payload["scope"] not in {BANK_SCOPE_GLOBAL_ONLY, BANK_SCOPE_ALL_LAYER}:
        raise RuntimeError("layer bank scope is invalid")
    pool_kernel = _int(payload["pool_kernel"], "pool kernel")
    if pool_kernel % 2 == 0:
        raise RuntimeError("layer bank pool kernel is not odd")
    for name in ("energy_pool_kernel", "row_energy_pool_kernel"):
        value = payload[name]
        if value is not None and (_int(value, name) % 2 == 0):
            raise RuntimeError(f"layer bank {name} is not odd")
    raw_orders = payload["support_orders"]
    if not isinstance(raw_orders, list):
        raise RuntimeError("layer bank support orders are invalid")
    orders: list[float] = []
    for raw_order in cast(list[object], raw_orders):
        if isinstance(raw_order, bool) or not isinstance(raw_order, (int, float)):
            raise RuntimeError("layer bank support orders are invalid")
        order = float(raw_order)
        if not math.isfinite(order) or order <= 1:
            raise RuntimeError("layer bank support orders are invalid")
        orders.append(order)
    if tuple(orders) != tuple(sorted(set(orders))):
        raise RuntimeError("layer bank support orders are not canonical")
    geometry = payload["geometry"]
    if not isinstance(geometry, dict):
        raise RuntimeError("layer bank geometry is invalid")
    geometry = cast(dict[str, Any], geometry)
    _require_keys(
        geometry,
        {"logical_memory_length", "heads", "query_rows", "layer_count"},
        "geometry",
    )
    logical_length = _int(geometry["logical_memory_length"], "logical memory length")
    heads = _int(geometry["heads"], "heads")
    query_rows = _int(geometry["query_rows"], "query rows")
    layer_count = _int(geometry["layer_count"], "layer count")
    raw_roster = payload["tensor_roster"]
    raw_tensors = payload["tensors"]
    if not isinstance(raw_roster, list) or not isinstance(raw_tensors, dict):
        raise RuntimeError("layer bank tensor roster is invalid")
    roster_values = cast(list[object], raw_roster)
    if any(not isinstance(name, str) for name in roster_values):
        raise RuntimeError("layer bank tensor roster is invalid")
    roster = cast(list[str], roster_values)
    if len(set(roster)) != len(roster):
        raise RuntimeError("layer bank tensor roster is invalid")
    tensors = cast(dict[str, Any], raw_tensors)
    if set(roster) != set(tensors) or roster != list(tensors):
        raise RuntimeError("layer bank tensor roster is invalid")
    for name in roster:
        _vector(tensors, name, name, logical_length)
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list):
        raise RuntimeError("layer bank row roster is invalid")
    rows = cast(list[Any], raw_rows)
    if len(rows) != layer_count or not rows:
        raise RuntimeError("layer bank row roster is invalid")
    referenced: list[str] = []
    layer_indices: list[int] = []
    decoded_rows: list[dict[str, Any]] = []
    for _row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError("layer bank row is not a dictionary")
        row = cast(dict[str, Any], row)
        _require_keys(
            row, {"schema", "descriptor", "heads", "query_rows", "kv_groups", "tensors"}, "row"
        )
        if row["schema"] != LAYER_BANK_SCHEMA:
            raise RuntimeError("layer bank row schema is invalid")
        descriptor = row["descriptor"]
        if not isinstance(descriptor, dict):
            raise RuntimeError("layer bank descriptor is invalid")
        descriptor = cast(dict[str, Any], descriptor)
        _require_keys(
            descriptor,
            {
                "layer_idx",
                "layer_type",
                "logical_memory_length",
                "physical_key_length",
                "absolute_key_offset",
                "window",
                "query_length",
            },
            "descriptor",
        )
        layer_idx = _int(descriptor["layer_idx"], "layer index", minimum=0)
        layer_indices.append(layer_idx)
        if descriptor["layer_type"] not in {"global", "sliding"}:
            raise RuntimeError("layer bank layer type is invalid")
        if payload["scope"] == BANK_SCOPE_GLOBAL_ONLY and descriptor["layer_type"] != "global":
            raise RuntimeError("global-only layer bank contains a sliding row")
        if _int(descriptor["logical_memory_length"], "descriptor logical length") != logical_length:
            raise RuntimeError("layer bank descriptor geometry differs")
        physical = _int(descriptor["physical_key_length"], "physical key length")
        query_length = _int(descriptor["query_length"], "descriptor query length")
        if (
            physical < query_length
            or _int(descriptor["absolute_key_offset"], "absolute offset", minimum=0) < 0
        ):
            raise RuntimeError("layer bank descriptor geometry is invalid")
        window = descriptor["window"]
        if window is not None and _int(window, "window") < 1:
            raise RuntimeError("layer bank descriptor window is invalid")
        row_heads = _int(row["heads"], "row heads")
        row_query_rows = _int(row["query_rows"], "row query rows")
        row_kv_groups = _int(row["kv_groups"], "row kv groups")
        if row_heads % row_kv_groups:
            raise RuntimeError("layer bank row kv groups do not divide its heads")
        if (row_heads, row_query_rows) != (heads, query_rows):
            raise RuntimeError("layer bank row geometry differs")
        refs = row["tensors"]
        if not isinstance(refs, dict):
            raise RuntimeError("layer bank row tensor roster is invalid")
        refs = cast(dict[str, Any], refs)
        _require_keys(
            refs,
            {
                "snap",
                "snap_chunks",
                "energy",
                "row_energy",
                "support_e",
                "support_eprime",
                "support_r",
                "support_einf",
                "support_rinf",
                "prepool_e2",
                "prepool_r2",
            },
            "row tensor roster",
        )
        snap = _vector(tensors, refs["snap"], "snap", logical_length)
        chunks = refs["snap_chunks"]
        if not isinstance(chunks, list) or not chunks:
            raise RuntimeError("layer bank snap chunks are invalid")
        chunks = cast(list[Any], chunks)
        chunk_values = [_vector(tensors, name, "snap chunk", logical_length) for name in chunks]
        folded = chunk_values[0].clone()
        for chunk in chunk_values[1:]:
            folded = folded + chunk
        if not torch.equal(folded, snap):
            raise RuntimeError("layer bank snap chunks do not reproduce snap")
        for name in (
            "energy",
            "row_energy",
            "support_einf",
            "support_rinf",
            "prepool_e2",
            "prepool_r2",
        ):
            if refs[name] is not None:
                _vector(tensors, refs[name], name, logical_length)
        for name in ("support_e", "support_eprime", "support_r"):
            raw_value = refs[name]
            if not isinstance(raw_value, dict):
                raise RuntimeError(f"layer bank {name} roster is invalid")
            value = cast(dict[str, str], raw_value)
            if set(value) != {_tag(float(order)) for order in orders}:
                raise RuntimeError(f"layer bank {name} roster is invalid")
            for tensor_name in value.values():
                _vector(tensors, tensor_name, name, logical_length)
        row_refs: list[Any] = [refs["snap"], *chunks]
        row_refs.extend(
            refs[name]
            for name in (
                "energy",
                "row_energy",
                "support_einf",
                "support_rinf",
                "prepool_e2",
                "prepool_r2",
            )
            if refs[name] is not None
        )
        row_refs.extend(
            tensor_name
            for name in ("support_e", "support_eprime", "support_r")
            for tensor_name in cast(dict[str, str], refs[name]).values()
        )
        if any(not isinstance(name, str) for name in row_refs):
            raise RuntimeError("layer bank tensor reference is invalid")
        referenced.extend(cast(list[str], row_refs))
        decoded_rows.append(row)
    if layer_indices != sorted(layer_indices) or len(set(layer_indices)) != len(layer_indices):
        raise RuntimeError("layer bank layer indices are not ordered and unique")
    if set(referenced) != set(roster) or len(referenced) != len(set(referenced)):
        raise RuntimeError("layer bank tensor roster does not match row references")
    return payload


def rows_from_payload(payload: dict[str, Any]) -> tuple[Any, ...]:
    """Build the in-memory row dataclasses from a payload that has been checked."""
    from rcc.transforms.select.query_support.capture_bank import LayerBankRow

    tensors = cast(dict[str, torch.Tensor], payload["tensors"])
    rows: list[LayerBankRow] = []
    for raw_row in cast(list[Any], payload["rows"]):
        row = cast(dict[str, Any], raw_row)
        descriptor = cast(dict[str, Any], row["descriptor"])
        refs = cast(dict[str, Any], row["tensors"])

        def value(name: str, refs: dict[str, Any] = refs) -> torch.Tensor | None:
            ref = refs[name]
            return None if ref is None else tensors[ref].clone()

        def mapping(name: str, refs: dict[str, Any] = refs) -> dict[float, torch.Tensor]:
            return {
                float(tag.removeprefix("p").replace("p", ".")): tensors[ref].clone()
                for tag, ref in refs[name].items()
            }

        snap = value("snap")
        if snap is None:
            raise RuntimeError("layer bank snap tensor is missing")

        rows.append(
            LayerBankRow(
                descriptor=LayerDescriptor(**descriptor),
                snap=snap,
                energy=value("energy"),
                row_energy=value("row_energy"),
                support_e=mapping("support_e"),
                support_eprime=mapping("support_eprime"),
                support_r=mapping("support_r"),
                support_einf=value("support_einf"),
                support_rinf=value("support_rinf"),
                prepool_e2=value("prepool_e2"),
                prepool_r2=value("prepool_r2"),
                heads=row["heads"],
                query_rows=row["query_rows"],
                kv_groups=row["kv_groups"],
                snap_chunks=tuple(tensors[ref].clone() for ref in refs["snap_chunks"]),
            )
        )
    return tuple(rows)


__all__ = [
    "BANK_SCOPE_ALL_LAYER",
    "BANK_SCOPE_GLOBAL_ONLY",
    "LAYER_BANK_SCHEMA",
    "payload_from_bank",
    "rows_from_payload",
    "validate_payload",
]
