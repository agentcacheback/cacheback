"""Writer exclusivity and lifecycle validation for route worker banks.

Every row is held to its seat, and one item's producer and receiver stages each
stay on one GPU. Payload rules on disk are `rcc.run.payload_banks`.
"""

from __future__ import annotations

import fcntl
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rcc.hardware.fleet import FleetPlacement
from rcc.models.route import RouteFamily
from rcc.run.fleet.ledger import read_bank_rows
from rcc.run.payload_banks import route_arms
from rcc.transforms.select.query_support.methods.compose import SUPPORT_DEFAULT_ALPHA


def bank_paths(arm_root: Path) -> tuple[Path, ...]:
    """Return the per-GPU worker bank paths of one arm."""
    return tuple(arm_root / "workers" / f"gpu{index}" / "raw.jsonl" for index in range(8))


def _validate_route_row_policy_identity(
    policy: str, row: Mapping[str, Any], family: RouteFamily
) -> None:
    arms = route_arms(family)
    if policy not in arms:
        raise RuntimeError(f"unregistered {family.lane} policy {policy!r}")
    adapter = family.score_adapter(arms[policy].semantic_arm)
    if adapter is not None and (
        row.get("selector_recipe") != adapter.recipe
        or row.get("selector_score_schema") != adapter.score_schema
        or row.get("selector_capture_peak_scope") != adapter.peak_scope
    ):
        raise RuntimeError("result row Mamba selector recipe or component schema differs")
    declared = row.get("policy")
    if declared is not None and declared != policy:
        raise RuntimeError(
            f"result row policy {declared!r} does not match requested policy {policy!r}"
        )
    arm = row.get("arm")
    if isinstance(arm, str) and arm.startswith(family.policy_prefix) and arm != policy:
        raise RuntimeError("result row policy and arm identities disagree")
    if arms[policy].selector == "support":
        dose = row.get("support_alpha")
        if isinstance(dose, bool) or not isinstance(dose, (int, float)):
            raise RuntimeError(
                "result row support alpha is missing; registered correction strength is "
                f"{SUPPORT_DEFAULT_ALPHA:g}"
            )
        if float(dose) != SUPPORT_DEFAULT_ALPHA:
            raise RuntimeError(
                f"result row support alpha {float(dose):g} does not match "
                "registered correction strength "
                f"{SUPPORT_DEFAULT_ALPHA:g}"
            )


@contextmanager
def worker_bank_lock(run_root: Path, worker_index: int) -> Generator[None]:
    """Refuse two processes writing the same GPU bank concurrently."""
    lock_path = run_root / "workers" / f"gpu{worker_index}" / ".worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"FanOutQA worker {worker_index} already has an active launcher"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validated_worker_bank_rows(
    paths: Sequence[Path],
    *,
    policy: str,
    placement: FleetPlacement,
    family: RouteFamily,
) -> list[dict[str, Any]]:
    """Require one complete lifecycle bank from every GPU of the placement."""
    combined = validated_present_worker_bank_rows(
        paths,
        policy=policy,
        placement=placement,
        family=family,
    )
    for worker_index, path in enumerate(paths):
        rows = read_bank_rows((path,))
        role = placement.role(worker_index)
        if not rows:
            raise RuntimeError(f"{policy}: gpu{worker_index} bank is missing or empty")
        runtime_roles = {str(row.get("role")) for row in rows if row.get("kind") == "runtime"}
        complete_roles = {
            str(row.get("role")) for row in rows if row.get("kind") == "worker_complete"
        }
        if runtime_roles != {role} or complete_roles != {role}:
            raise RuntimeError(f"{policy}: gpu{worker_index} lifecycle role is incomplete")
    return combined


def validated_present_worker_bank_rows(
    paths: Sequence[Path],
    *,
    policy: str,
    placement: FleetPlacement,
    family: RouteFamily,
    identity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate every row present, without requiring a complete bank."""
    if len(paths) != placement.workers:
        raise RuntimeError(f"{policy}: worker bank roster has the wrong size")
    combined: list[dict[str, Any]] = []
    result_workers: dict[tuple[str, str], int] = {}
    producer_workers: dict[tuple[str, str], set[int]] = {}
    receiver_workers: dict[tuple[str, str], set[int]] = {}
    for worker_index, path in enumerate(paths):
        rows = read_bank_rows((path,))
        role = placement.role(worker_index)
        runtime_roles = {str(row.get("role")) for row in rows if row.get("kind") == "runtime"}
        complete_roles = {
            str(row.get("role")) for row in rows if row.get("kind") == "worker_complete"
        }
        if runtime_roles and runtime_roles != {role}:
            raise RuntimeError(f"{policy}: gpu{worker_index} runtime role differs")
        if complete_roles and complete_roles != {role}:
            raise RuntimeError(f"{policy}: gpu{worker_index} completion role differs")
        decoded = {
            (str(row.get("qid") or ""), str(row.get("attempt_id") or ""))
            for row in rows
            if row.get("kind") == "stage" and row.get("stage") == "decode_end"
        }
        for row in rows:
            if identity and any(row.get(name) != expected for name, expected in identity.items()):
                raise RuntimeError(f"{policy}: gpu{worker_index} bank identity differs")
            if row.get("kind") == "stage" and (
                row.get("gpu_index") != worker_index or row.get("role") != role
            ):
                raise RuntimeError(f"{policy}: gpu{worker_index} stage provenance mismatch")
            if row.get("kind") == "stage":
                lineage = (str(row.get("qid") or ""), str(row.get("attempt_id") or ""))
                stage = row.get("stage")
                if stage == "submit" and worker_index != 0:
                    raise RuntimeError(f"{policy}: submit stage is not on coordinator gpu0")
                if stage in {"producer_claim", "producer_start", "producer_end", "handoff_ready"}:
                    producer_workers.setdefault(lineage, set()).add(worker_index)
                if stage in {
                    "receiver_claim",
                    "payload_loaded",
                    "admitted",
                    "first_token",
                    "decode_end",
                }:
                    receiver_workers.setdefault(lineage, set()).add(worker_index)
            if row.get("kind") == "result" and role == "producer":
                raise RuntimeError(f"{policy}: producer gpu{worker_index} banked a result")
            if row.get("kind") == "result":
                _validate_route_row_policy_identity(policy, row, family)
                lineage = (str(row.get("qid") or ""), str(row.get("attempt_id") or ""))
                if not all(lineage) or lineage not in decoded:
                    raise RuntimeError(
                        f"{policy}: gpu{worker_index} result has no same-attempt decode"
                    )
                result_workers[lineage] = worker_index
        combined.extend(rows)
    lineages = set(result_workers) | set(producer_workers) | set(receiver_workers)
    for lineage in lineages:
        producers = producer_workers.get(lineage, set())
        receivers = receiver_workers.get(lineage, set())
        if len(producers) > 1 or len(receivers) > 1:
            raise RuntimeError(f"{policy}/{lineage[0]}: stage family changes physical bank")
        if lineage in result_workers and receivers != {result_workers[lineage]}:
            raise RuntimeError(f"{policy}/{lineage[0]}: result differs from receiver bank")
        if placement.fused and producers and receivers and producers != receivers:
            raise RuntimeError(f"{policy}/{lineage[0]}: fused stages change physical bank")
    return combined
