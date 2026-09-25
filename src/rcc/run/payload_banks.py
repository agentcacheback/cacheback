"""What one banked latent result row must find on disk, read by its layout.

A latent row is a claim about a payload: the rows the receiver was handed, one
selector vector per worker or per hop, and a manifest that signs both.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch

from rcc.benchmarks.fanoutqa.payload import PAYLOAD_SCHEMA, read_payload
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.protocol import PhysicalArm
from rcc.models.qwen.capture import (
    QWEN_PAYLOAD_LAYOUT,
    QwenFlatPayload,
    payload_plan_sha256,
    qwen_w16_keep,
    selected_indices_sha256,
    tensor_content_sha256,
)
from rcc.models.route import RouteFamily
from rcc.models.selection import ScoreAdapter
from rcc.run.io import is_sha256_hex
from rcc.topologies.chain import hop_budget
from rcc.topologies.chain.layout import CHAIN_TERMINAL_LAYOUT


def route_arms(family: RouteFamily) -> dict[str, PhysicalArm]:
    """Return the physical arms of one route family, keyed by policy."""
    return {arm.policy: arm for arm in family.profile.physical_arms}


class _LayoutManifest(NamedTuple):
    """How one payload layout's producer seat names and fills its manifest."""

    score_stem: str
    first_index: int
    meta_fields: tuple[str, ...]


#: The manifest each layout writes, keyed by that layout. The fan-out seat banks
#: one selector vector per worker, the chain seat one per hop; the table keeps
#: either roster from being read under the other name.
_LAYOUT_MANIFESTS: dict[str, _LayoutManifest] = {
    QWEN_PAYLOAD_LAYOUT: _LayoutManifest(
        score_stem="w",
        first_index=0,
        meta_fields=("worker_prompt_sha256", "manager_prompt_sha256"),
    ),
    CHAIN_TERMINAL_LAYOUT: _LayoutManifest(
        score_stem="h",
        first_index=1,
        meta_fields=(
            "budget_law",
            "kept_rows_by_hop",
            "hop_prompt_sha256",
            "retention_prompt_sha256",
            "hop_prefix_rows",
            "hop_prompt_rows",
            "hop_rows_pre_cut",
            "hop_budgets",
            "terminal_origins",
            "hop_extract_s",
            "hop_latent_roll_s",
            "hop_selector_capture_s",
        ),
    ),
}


def _layout_manifest(layout: str, *, label: str) -> _LayoutManifest:
    """Return how the seat writing this layout names and fills its manifest."""
    reader = _LAYOUT_MANIFESTS.get(layout)
    if reader is None:
        raise RuntimeError(
            f"{label}: no payload manifest reader is registered for layout {layout!r}"
        )
    return reader


def _row_int_list(value: object, *, count: int, label: str) -> tuple[int, ...]:
    """Read one banked list of exactly ``count`` nonnegative integers."""
    if not isinstance(value, list) or len(cast(list[object], value)) != count:
        raise RuntimeError(f"{label}: hydrated payload lineage is corrupt")
    values: list[int] = []
    for entry in cast(list[object], value):
        if type(entry) is not int or entry < 0:
            raise RuntimeError(f"{label}: hydrated payload lineage is corrupt")
        values.append(entry)
    return tuple(values)


def _row_keeps(raw_keeps: Sequence[object], *, label: str) -> list[tuple[int, ...]]:
    """Read one row's keep roster as sorted, unique, nonnegative positions."""
    keeps: list[tuple[int, ...]] = []
    for raw_keep in raw_keeps:
        if not isinstance(raw_keep, list):
            raise RuntimeError(f"{label}: hydrated payload lineage is corrupt")
        keep_values: list[int] = []
        for index in cast(list[object], raw_keep):
            if type(index) is not int:
                raise RuntimeError(f"{label}: hydrated payload lineage is corrupt")
            keep_values.append(index)
        keep = tuple(keep_values)
        if tuple(sorted(set(keep))) != keep or any(index < 0 for index in keep):
            raise RuntimeError(f"{label}: hydrated payload lineage is corrupt")
        keeps.append(keep)
    return keeps


def _latent_row_payload(
    row: Mapping[str, Any], *, policy: str, family: RouteFamily, profile: BenchmarkProfile
) -> tuple[str, tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Validate the payload identity one result row carries.

    ``keeps_by_worker`` is one keep per worker or hop, ``latent_rows_by_worker``
    the blocks the receiver was handed, and ``latent_tokens`` their sum.
    """
    arm = route_arms(family)[policy]
    semantic_arm = arm.semantic_arm
    layout = row.get("payload_layout")
    if layout != profile.payload_layout:
        raise RuntimeError(
            f"{row.get('qid')}/{policy}: hydrated payload layout {layout!r} is not the "
            f"{profile.payload_layout!r} {profile.benchmark_key} registers"
        )
    raw_keeps_value = row.get("keeps_by_worker")
    raw_counts_value = row.get("latent_rows_by_worker")
    if (
        row.get("payload_semantic_arm") != semantic_arm
        or row.get("semantic_arm") != semantic_arm
        or row.get("payload_plan_sha256")
        != payload_plan_sha256(semantic_arm, family=family, profile=profile)
        or not isinstance(raw_keeps_value, list)
        or len(cast(list[object], raw_keeps_value)) != profile.workers_per_item
        or not isinstance(raw_counts_value, list)
    ):
        raise RuntimeError(f"{row.get('qid')}/{policy}: hydrated payload lineage is corrupt")
    label = f"{row.get('qid')}/{policy}"
    keeps = _row_keeps(cast(list[object], raw_keeps_value), label=label)
    worker_rows = tuple(len(keep) for keep in keeps)
    shipped = (worker_rows[-1],) if layout == CHAIN_TERMINAL_LAYOUT else worker_rows
    counts = _row_int_list(cast(list[object], raw_counts_value), count=len(shipped), label=label)
    if (
        counts != shipped
        or row.get("latent_tokens") != sum(counts)
        or row.get("selected_indices_sha256") != selected_indices_sha256(keeps)
        or not isinstance(row.get("payload_tensor_sha256"), str)
        or not is_sha256_hex(row["payload_tensor_sha256"])
    ):
        raise RuntimeError(f"{label}: hydrated payload lineage is corrupt")
    return semantic_arm, tuple(keeps), worker_rows


def _recorded_widths(row: Mapping[str, Any], profile: BenchmarkProfile) -> list[int]:
    """Return the worker prompt widths the row records, one per selector score."""
    raw = row.get("worker_prompt_tokens")
    if not isinstance(raw, list):
        raise ValueError("selector evidence identity differs")
    widths = cast(list[object], raw)
    if len(widths) != profile.workers_per_item or any(type(width) is not int for width in widths):
        raise ValueError("selector evidence identity differs")
    return cast(list[int], widths)


def _selector_vector_rows(
    row: Mapping[str, Any],
    keeps: tuple[tuple[int, ...], ...],
    *,
    profile: BenchmarkProfile,
    ratio: int | None,
    law: str | None,
    budget_rows: int | None,
    label: str,
) -> tuple[int, ...]:
    """Return the length every banked selector vector must have.

    A fan-out worker scores one fixed prompt and its roll, so all three vectors
    are one length; a chain hop's vector grows with the rows it carried in.
    """
    if profile.payload_layout != CHAIN_TERMINAL_LAYOUT:
        return tuple(profile.rolled_rows(width) for width in _recorded_widths(row, profile))
    count = profile.workers_per_item
    pre_cut = _row_int_list(row.get("hop_rows_pre_cut"), count=count, label=label)
    prompts = _row_int_list(row.get("hop_prompt_rows"), count=count, label=label)
    budgets = _row_int_list(row.get("hop_budgets"), count=count, label=label)
    prefixes = (0, *(len(keep) for keep in keeps[:-1]))
    if any(
        rows != prefix + prompt + profile.latent_steps
        for rows, prefix, prompt in zip(pre_cut, prefixes, prompts, strict=True)
    ):
        raise RuntimeError(f"{label}: hydrated payload lineage is corrupt")
    if law is None:
        raise RuntimeError(f"{label}: a chain row names no budget law")
    for budget, rows, prefix, keep in zip(budgets, pre_cut, prefixes, keeps, strict=True):
        allowed = hop_budget(law, rows, ratio=ratio, prefix_rows=prefix, budget_rows=budget_rows)
        if budget != allowed or len(keep) != allowed:
            raise RuntimeError(
                f"{label}: a banked hop keeps {len(keep)} of {rows} rows under a claimed "
                f"{budget}-row budget, not the {allowed} rows the chain law allows"
            )
    return pre_cut


def _qwen_expected_payload_meta(
    policy: str,
    row: Mapping[str, Any],
    *,
    semantic_arm: str,
    family: RouteFamily,
    reader: _LayoutManifest,
) -> dict[str, Any]:
    """Rebuild the producer's manifest metadata from its result row.

    The fields both seats bank come first and the layout's own follow, so a
    manifest is compared against every field its producer wrote and no other.
    """
    expected = {
        name: row.get(name)
        for name in (
            "producer_backend",
            "producer_route",
            "payload_layout",
            "payload_semantic_arm",
            "payload_plan_sha256",
            "payload_tensor_sha256",
            "selected_indices_sha256",
            "selector_score_name",
            "selector_score_tensor_sha256",
            "keeps_by_worker",
            "latent_rows_by_worker",
            "latent_tokens",
            "extract_s",
            "latent_roll_s",
            "selector_capture_s",
            "selector_capture_peak_bytes",
            *reader.meta_fields,
        )
    }
    expected.update(
        {
            "family": family.model_id,
            "policy": policy,
            "semantic_arm": semantic_arm,
        }
    )
    if route_arms(family)[policy].selector == "support":
        expected["support_alpha"] = row.get("support_alpha")
    adapter = family.score_adapter(semantic_arm)
    if adapter is not None:
        expected["selector_recipe"] = row.get("selector_recipe")
        expected["selector_score_schema"] = row.get("selector_score_schema")
        expected["selector_capture_peak_scope"] = row.get("selector_capture_peak_scope")
    return expected


def _payload_manifest_body(
    manifest_path: Path, *, qid: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one payload manifest and return its file roster and its metadata."""
    try:
        decoded: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label}: hydrated payload manifest is corrupt") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{label}: hydrated payload manifest is corrupt")
    manifest = cast(dict[str, Any], decoded)
    raw_files = manifest.get("files")
    raw_meta = manifest.get("meta")
    if (
        manifest.get("schema") != PAYLOAD_SCHEMA
        or manifest.get("qid") != qid
        or manifest.get("report_bundle") is not None
        or not isinstance(raw_files, dict)
        or not isinstance(raw_meta, dict)
    ):
        raise RuntimeError(f"{label}: hydrated payload manifest is corrupt")
    return cast(dict[str, Any], raw_files), cast(dict[str, Any], raw_meta)


def _qwen_payload_dependency(
    arm_root: Path,
    policy: str,
    row: Mapping[str, Any],
    *,
    semantic_arm: str,
    family: RouteFamily,
    profile: BenchmarkProfile,
) -> tuple[Path, Path, tuple[Path, ...]] | None:
    """Validate one latent manifest and return its manifest and file paths.

    The roster and the metadata are read under the layout the benchmark declares,
    so each seat's score keys are checked against the seat that writes them.
    """
    qid = str(row["qid"])
    manifest_path = arm_root / "payloads" / f"{qid}.payload.json"
    if not manifest_path.is_file():
        return None
    label = f"{qid}/{policy}"
    reader = _layout_manifest(str(profile.payload_layout), label=label)
    score_names = tuple(
        f"score_{reader.score_stem}{index}"
        for index in range(reader.first_index, reader.first_index + profile.workers_per_item)
    )
    files, meta = _payload_manifest_body(manifest_path, qid=qid, label=label)
    expected_paths = {
        "embedding_rows": arm_root / "payloads" / "rows" / f"{qid}.pt",
        **{
            name: arm_root
            / "payloads"
            / "selector_scores"
            / f"{qid}_{name.removeprefix('score_')}.pt"
            for name in score_names
        },
    }
    if set(files) != set(expected_paths):
        raise RuntimeError(f"{qid}/{policy}: hydrated payload manifest is corrupt")
    if meta != _qwen_expected_payload_meta(
        policy, row, semantic_arm=semantic_arm, family=family, reader=reader
    ):
        raise RuntimeError(f"{qid}/{policy}: hydrated payload lineage is corrupt")
    for name, expected_path in expected_paths.items():
        raw_entry = files[name]
        if not isinstance(raw_entry, dict):
            raise RuntimeError(f"{qid}/{policy}: hydrated payload manifest is corrupt")
        entry = cast(dict[str, Any], raw_entry)
        if (
            set(entry) != {"path", "sha256"}
            or Path(str(entry.get("path") or "")) != expected_path
            or not is_sha256_hex(entry.get("sha256"))
        ):
            raise RuntimeError(f"{qid}/{policy}: hydrated payload manifest is corrupt")
    return (
        manifest_path,
        expected_paths["embedding_rows"],
        tuple(expected_paths[name] for name in score_names),
    )


def _expected_keep(
    adapter: ScoreAdapter | None,
    score: torch.Tensor,
    *,
    chain: bool,
    ratio: int | None,
    law: str | None,
    budget_rows: int | None,
    rows_pre_cut: int,
    prefix: int,
    latent_steps: int,
) -> tuple[int, ...] | None:
    """Re-derive the keep a banked selector vector explains, or None if nothing does.

    A family adapter keeps at the ratio; a chain hop keeps at its law's own
    budget, where every carried row competes inside the cut.
    """
    if adapter:
        adapter.validate(score)
        if ratio is None:
            raise ValueError("a score adapter keeps at a ratio")
        return adapter.keep(score, ratio)
    if not chain:
        return None
    if law is None:
        raise ValueError("a chain row names no budget law")
    budget = hop_budget(law, rows_pre_cut, ratio=ratio, prefix_rows=prefix, budget_rows=budget_rows)
    return qwen_w16_keep(score, ratio, latent_steps=latent_steps, budget=budget)


def _validate_qwen_payload_rows(
    manifest_path: Path,
    path: Path,
    row: Mapping[str, Any],
    *,
    policy: str,
    semantic_arm: str,
    keeps: tuple[tuple[int, ...], ...],
    row_counts: tuple[int, ...],
    profile: BenchmarkProfile,
    score_paths: tuple[Path, ...],
    score_rows: tuple[int, ...],
    ratio: int | None,
    law: str | None,
    budget_rows: int | None,
    family: RouteFamily,
) -> None:
    """Validate the embedding bytes on disk against the payload and row contracts.

    A chain hop's vector is the evidence of the cut it made, so the cut is
    recomputed from it and a keep the vector does not explain is refused.
    """
    qid = str(row["qid"])
    try:
        import torch

        read_payload(manifest_path, qid=qid)
        rows = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(rows, torch.Tensor):
            raise TypeError("payload is not a tensor")
        QwenFlatPayload(
            rows=rows,
            semantic_arm=semantic_arm,
            latent_plan_sha256=str(row["payload_plan_sha256"]),
            keeps=keeps,
            rows_by_worker=row_counts,
            selected_indices_sha256=str(row["selected_indices_sha256"]),
            tensor_sha256=str(row["payload_tensor_sha256"]),
            family=family,
            profile=profile,
            layout=str(row["payload_layout"]),
        )
        score_hashes = row.get("selector_score_tensor_sha256")
        if (
            row.get("selector_score_name") != route_arms(family)[policy].selector
            or not isinstance(score_hashes, list)
            or len(cast(list[object], score_hashes)) != profile.workers_per_item
        ):
            raise ValueError("selector evidence identity differs")
        chain = profile.payload_layout == CHAIN_TERMINAL_LAYOUT
        adapter = family.score_adapter(semantic_arm)
        prefixes = (0, *(len(keep) for keep in keeps[:-1]))
        for score_path, expected_hash, rows_pre_cut, prefix, keep in zip(
            score_paths,
            cast(list[object], score_hashes),
            score_rows,
            prefixes,
            keeps,
            strict=True,
        ):
            score = torch.load(score_path, map_location="cpu", weights_only=True)
            if (
                not isinstance(score, torch.Tensor)
                or score.shape != (adapter.shape(rows_pre_cut) if adapter else (rows_pre_cut,))
                or not bool(score.isfinite().all())
                or tensor_content_sha256(score) != expected_hash
            ):
                raise ValueError("selector evidence differs")
            expected_keep = _expected_keep(
                adapter,
                score,
                chain=chain,
                ratio=ratio,
                law=law,
                budget_rows=budget_rows,
                rows_pre_cut=rows_pre_cut,
                prefix=prefix,
                latent_steps=profile.latent_steps,
            )
            if expected_keep is not None and expected_keep != keep:
                raise ValueError("banked selector evidence does not explain its cut")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{qid}/{policy}: hydrated payload file is corrupt") from exc


__all__ = ("route_arms",)
