"""The gold-blind structural audit of one item's official decomposition.

Measures the fan-out width, the cited pages, and how the dependencies fall
across the worker shards, without reading any gold answer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

MULTIHOP_AUDIT_SCHEMA = "fanoutqa-multihop-audit-v2"


@dataclass(frozen=True)
class _Node:
    node_id: str
    pageid: int | None
    depends_on: tuple[str, ...]
    children: tuple[_Node, ...]
    depth: int


@dataclass(frozen=True)
class DependencyShardAudit:
    """One decomposition's dependency structure over the workers of an item."""

    schema: str
    decomposition_nodes: int
    evidence_mentions: int
    cited_pageids: frozenset[int]
    max_branch_width: int
    max_decomposition_depth: int
    dependency_edges: int
    dependency_chain_nodes: int
    dependency_edges_spanning_workers: int
    workers_per_item: int
    workers_with_evidence: int
    workers_with_dependency_incidence: int
    max_dependency_worker_span: int
    worker_evidence_mentions: tuple[int, ...]
    worker_dependency_incidence: tuple[int, ...]
    unowned_evidence_mentions: int
    reasoning_shape: str

    def as_row(self) -> dict[str, object]:
        """Return the JSON-compatible fields the preparation ledger records."""
        return {
            "multihop_audit_schema": self.schema,
            "official_decomposition_nodes": self.decomposition_nodes,
            "official_evidence_mentions": self.evidence_mentions,
            "official_unique_pages": len(self.cited_pageids),
            "max_branch_width": self.max_branch_width,
            "max_decomposition_depth": self.max_decomposition_depth,
            "dependency_edges": self.dependency_edges,
            "dependency_chain_nodes": self.dependency_chain_nodes,
            "dependency_edges_spanning_workers": self.dependency_edges_spanning_workers,
            "workers_per_item": self.workers_per_item,
            "workers_with_evidence": self.workers_with_evidence,
            "workers_with_dependency_incidence": self.workers_with_dependency_incidence,
            "max_dependency_worker_span": self.max_dependency_worker_span,
            "worker_evidence_mentions": list(self.worker_evidence_mentions),
            "worker_dependency_incidence": list(self.worker_dependency_incidence),
            "unowned_evidence_mentions": self.unowned_evidence_mentions,
            "reasoning_shape": self.reasoning_shape,
        }


def _as_sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeError(f"FanOutQA {field} must be a sequence")
    return cast(Sequence[object], value)


def _parse_decomposition(decomposition: object) -> tuple[tuple[_Node, ...], dict[str, _Node]]:
    flat: dict[str, _Node] = {}
    seen: set[str] = set()

    def parse(raw: object, depth: int) -> _Node:
        if not isinstance(raw, Mapping):
            raise RuntimeError("FanOutQA decomposition node must be an object")
        raw_map = cast(Mapping[str, object], raw)
        node_id = str(raw_map.get("id") or "")
        if not node_id:
            raise RuntimeError("FanOutQA decomposition node has no id")
        if node_id in seen:
            raise RuntimeError(f"duplicate decomposition node id {node_id!r}")
        seen.add(node_id)

        evidence = raw_map.get("evidence")
        pageid: int | None = None
        if evidence is not None:
            if not isinstance(evidence, Mapping) or "pageid" not in evidence:
                raise RuntimeError(f"FanOutQA node {node_id!r} has malformed evidence")
            evidence_map = cast(Mapping[str, object], evidence)
            try:
                pageid = int(cast(Any, evidence_map["pageid"]))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"FanOutQA node {node_id!r} has invalid pageid") from exc

        raw_dependencies = raw_map.get("depends_on") or ()
        dependencies = tuple(
            str(value) for value in _as_sequence(raw_dependencies, field="depends_on")
        )
        if any(not dependency for dependency in dependencies) or len(set(dependencies)) != len(
            dependencies
        ):
            raise RuntimeError(f"FanOutQA node {node_id!r} has invalid dependencies")
        raw_children = raw_map.get("decomposition") or ()
        children = tuple(
            parse(child, depth + 1) for child in _as_sequence(raw_children, field="decomposition")
        )
        node = _Node(node_id, pageid, dependencies, children, depth)
        flat[node_id] = node
        return node

    roots = tuple(parse(raw, 1) for raw in _as_sequence(decomposition or (), field="decomposition"))
    if not roots:
        raise RuntimeError("FanOutQA decomposition is empty")
    return roots, flat


def decomposition_pageids(decomposition: object) -> tuple[int, ...]:
    """Return the cited evidence page ids, recursively, in first-mention order."""
    roots, _flat = _parse_decomposition(decomposition)
    ordered: list[int] = []

    def visit(node: _Node) -> None:
        if node.pageid is not None:
            ordered.append(node.pageid)
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)
    return tuple(dict.fromkeys(ordered))


def _page_owners(shards: Sequence[Sequence[int]]) -> dict[int, int]:
    if len(shards) < 2:
        raise RuntimeError("FanOutQA dependency audit requires at least two shards")
    owners: dict[int, int] = {}
    for worker, shard in enumerate(shards):
        for value in shard:
            pageid = int(value)
            if pageid in owners:
                raise RuntimeError(f"FanOutQA worker shards share a page: {pageid}")
            owners[pageid] = worker
    return owners


def _dependency_chain(flat: Mapping[str, _Node]) -> int:
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def chain(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            raise RuntimeError("FanOutQA dependency graph contains a cycle")
        visiting.add(node_id)
        node = flat[node_id]
        for dependency in node.depends_on:
            if dependency not in flat:
                raise RuntimeError(
                    f"FanOutQA node {node_id!r} depends on unknown node {dependency!r}"
                )
        value = 1 + max((chain(dependency) for dependency in node.depends_on), default=0)
        visiting.remove(node_id)
        memo[node_id] = value
        return value

    return max(chain(node_id) for node_id in flat)


def audit_dependency_shards(
    decomposition: object,
    shards: Sequence[Sequence[int]],
) -> DependencyShardAudit:
    """Check and measure one decomposition, without reading any gold answer."""
    roots, flat = _parse_decomposition(decomposition)
    owners = _page_owners(shards)
    chain_nodes = _dependency_chain(flat)
    cited = frozenset(node.pageid for node in flat.values() if node.pageid is not None)

    owner_memo: dict[str, frozenset[int]] = {}

    def support_owners(node: _Node) -> frozenset[int]:
        if node.node_id in owner_memo:
            return owner_memo[node.node_id]
        found: set[int] = {owners[node.pageid]} if node.pageid in owners else set()
        for child in node.children:
            found.update(support_owners(child))
        result = frozenset(found)
        owner_memo[node.node_id] = result
        return result

    for node in roots:
        support_owners(node)

    edges = [(node, flat[dependency]) for node in flat.values() for dependency in node.depends_on]
    spanning = 0
    incidence = [0] * len(shards)
    max_worker_span = 0
    for dependent, dependency in edges:
        participants = support_owners(dependent) | support_owners(dependency)
        if len(participants) > 1:
            spanning += 1
        max_worker_span = max(max_worker_span, len(participants))
        for worker in participants:
            incidence[worker] += 1

    evidence_mentions = [0] * len(shards)
    unowned = 0
    for node in flat.values():
        if node.pageid is None:
            continue
        worker = owners.get(node.pageid)
        if worker is None:
            unowned += 1
        else:
            evidence_mentions[worker] += 1

    if not edges:
        shape = "parallel_fanout"
    elif spanning:
        shape = "explicit_cross_worker_dependency"
    else:
        shape = "explicit_dependency_within_worker"
    branch_width = max([len(roots), *(len(node.children) for node in flat.values())])
    return DependencyShardAudit(
        schema=MULTIHOP_AUDIT_SCHEMA,
        decomposition_nodes=len(flat),
        evidence_mentions=sum(node.pageid is not None for node in flat.values()),
        cited_pageids=cited,
        max_branch_width=branch_width,
        max_decomposition_depth=max(node.depth for node in flat.values()),
        dependency_edges=len(edges),
        dependency_chain_nodes=chain_nodes,
        dependency_edges_spanning_workers=spanning,
        workers_per_item=len(shards),
        workers_with_evidence=sum(count > 0 for count in evidence_mentions),
        workers_with_dependency_incidence=sum(count > 0 for count in incidence),
        max_dependency_worker_span=max_worker_span,
        worker_evidence_mentions=tuple(evidence_mentions),
        worker_dependency_incidence=tuple(incidence),
        unowned_evidence_mentions=unowned,
        reasoning_shape=shape,
    )


def summarize_dependency_audits(
    audits: Sequence[DependencyShardAudit],
) -> dict[str, object]:
    """Aggregate per-item dependency audits into panel-level counts."""
    if not audits:
        raise ValueError("cannot summarize an empty FanOutQA audit panel")
    shapes = Counter(audit.reasoning_shape for audit in audits)
    return {
        "schema": MULTIHOP_AUDIT_SCHEMA,
        "items": len(audits),
        "explicit_dependency_items": sum(audit.dependency_edges > 0 for audit in audits),
        "parallel_fanout_items": shapes["parallel_fanout"],
        "cross_worker_dependency_items": sum(
            audit.dependency_edges_spanning_workers > 0 for audit in audits
        ),
        "all_worker_evidence_items": sum(
            audit.workers_with_evidence == audit.workers_per_item for audit in audits
        ),
        "all_worker_dependency_incidence_items": sum(
            audit.workers_with_dependency_incidence == audit.workers_per_item for audit in audits
        ),
        "max_dependency_worker_span": max(audit.max_dependency_worker_span for audit in audits),
        "reasoning_shapes": dict(sorted(shapes.items())),
        "min_unique_pages": min(len(audit.cited_pageids) for audit in audits),
        "max_unique_pages": max(len(audit.cited_pageids) for audit in audits),
        "min_branch_width": min(audit.max_branch_width for audit in audits),
        "max_branch_width": max(audit.max_branch_width for audit in audits),
        "min_dependency_chain_nodes": min(audit.dependency_chain_nodes for audit in audits),
        "max_dependency_chain_nodes": max(audit.dependency_chain_nodes for audit in audits),
        "max_decomposition_depth": max(audit.max_decomposition_depth for audit in audits),
        "unowned_evidence_mentions": sum(audit.unowned_evidence_mentions for audit in audits),
    }


__all__ = (
    "MULTIHOP_AUDIT_SCHEMA",
    "DependencyShardAudit",
    "audit_dependency_shards",
    "decomposition_pageids",
    "summarize_dependency_audits",
)
