"""The vLLM KV-connector seam that injects and extracts selected caches.

Every decision lives in the pure functions here. `RCCConnector`, the class vLLM
loads by module path, is built on first attribute access, so no vllm import.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rcc.injector.extraction import (
    EXTRACTED,
    REQUESTED,
    ExtractionPlanner,
    ExtractSpec,
    extract_request,
)
from rcc.injector.layout import Geometry, pack_cache, slot_mapping
from rcc.injector.staging import PENDING, StagedCache, StagingError

_LOG = logging.getLogger(__name__)

_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")

_VLLM_INSTALL_HINT = (
    "vllm is not installed in this environment, so the RCCConnector class cannot be built.\n"
    "Install the serving pins, in this order:\n"
    "    pip install vllm==0.11.1\n"
    "    pip install transformers==4.57.1\n"
    "vllm first: it pulls in a matching torch/cuda build."
)


@dataclass(frozen=True)
class InjectionSpec:
    """One request's page-write order: the physical blocks and the staged length k."""

    block_ids: tuple[int, ...]
    k: int


def parse_layer_index(layer_name: str) -> int:
    """Return the integer i from a registered layer name like `model.layers.{i}.self_attn.attn`.

    The layer map is built from parsed indices because dict order is not guaranteed to
    follow the layers, and a dict-order zip would misplace layers with no shape error.
    """
    match = _LAYER_INDEX_RE.search(layer_name)
    if match is None:
        raise ValueError(f"no layer index in {layer_name!r}; expected 'model.layers.<i>.'")
    return int(match.group(1))


def read_layer_map(
    kv_caches: Mapping[str, torch.Tensor], *, block_size: int
) -> tuple[Geometry, list[torch.Tensor]]:
    """Read the engine's registered pages into (geometry, per-layer tensors by index).

    Runs once at `register_kv_caches` and checks what the scatter trusts: indices
    forming `range(layers)`, one uniform page shape and dtype, contiguity, block grain.
    """
    if not kv_caches:
        raise ValueError("kv_caches is empty; nothing was registered")
    parsed = sorted((parse_layer_index(name), name) for name in kv_caches)
    indices = [index for index, _ in parsed]
    if indices != list(range(len(indices))):
        raise ValueError(
            f"parsed layer indices {indices} do not form range(0, {len(indices)}); "
            "refusing to map layers by dict order"
        )
    first = kv_caches[parsed[0][1]]
    shape = tuple(first.shape)
    ordered: list[torch.Tensor] = []
    for _, name in parsed:
        tensor = kv_caches[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} drifts from {shape}")
        if tensor.dtype != first.dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} drifts from {first.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} is not contiguous (HND layout?); the scatter is NHD-only")
        ordered.append(tensor)
    if len(shape) != 5 or shape[0] != 2:
        raise ValueError(
            f"pages must be [2, num_blocks, block_size, kv_heads, head_dim], got {shape}"
        )
    if int(shape[2]) != block_size:
        raise ValueError(f"page grain {int(shape[2])} != engine cache block_size {block_size}")
    geometry = Geometry(
        layers=len(ordered),
        kv_heads=int(shape[3]),
        head_dim=int(shape[4]),
        block_size=block_size,
    )
    return geometry, ordered


def credit_tokens(staged_k: int, *, num_request_tokens: int, block_size: int) -> int:
    """Return the token count credited to the scheduler for one staged request: exactly k.

    The staged length must be block aligned and the request must carry one token past
    the prefix; flooring the credit would recompute KV over the injected content.
    """
    if staged_k < 1:
        raise ValueError(f"staged length must be positive, got {staged_k}")
    if staged_k % block_size != 0:
        raise ValueError(
            f"staged length {staged_k} is not block aligned (block_size {block_size}); "
            "align the Select budget before the cut"
        )
    if num_request_tokens < staged_k + 1:
        raise ValueError(
            f"request has {num_request_tokens} tokens for a {staged_k} token prefix; "
            "at least one real tail token must remain to be forwarded"
        )
    return staged_k


def prompt_token_count(new_request: Any) -> int:
    """Return one newly scheduled request's prompt length.

    ``prompt_is_token_ids`` says which of ``prompt_token_ids`` and ``prompt_embeds`` is the
    prompt; an embeddings prompt's length is its row count, and neither present is zero.
    """
    ids = getattr(new_request, "prompt_token_ids", None)
    embeds = getattr(new_request, "prompt_embeds", None)
    ids_are_the_prompt = bool(getattr(new_request, "prompt_is_token_ids", True))
    if ids is not None and (ids_are_the_prompt or embeds is None):
        return len(ids)
    if embeds is not None:
        return int(embeds.shape[0])
    return 0


def capture_block_ids(
    block_ids: Sequence[int], num_external_tokens: int, block_size: int
) -> tuple[int, ...]:
    """Keep the leading blocks that hold the externally computed tokens.

    The allocator hands over the request's full block list; the injected
    prefix owns the first ceil(num_external / block_size) of them.
    """
    needed = (num_external_tokens + block_size - 1) // block_size
    if len(block_ids) < needed:
        raise ValueError(
            f"{len(block_ids)} allocated block ids cannot hold {num_external_tokens} "
            f"external tokens ({needed} blocks needed)"
        )
    return tuple(int(block) for block in block_ids[:needed])


def _validate_entry(
    request_id: str, entry: StagedCache, spec: InjectionSpec, geometry: Geometry
) -> None:
    """Raise if the engine's pages cannot hold this staged cache."""
    expected = (geometry.layers, geometry.kv_heads, spec.k, geometry.head_dim)
    if tuple(entry.keys.shape) != expected:
        raise ValueError(
            f"staged cache for {request_id!r} is {tuple(entry.keys.shape)}, engine pages "
            f"expect [layers, kv_heads, k, head_dim] == {expected}"
        )
    if spec.k % geometry.block_size != 0:
        raise ValueError(
            f"injection length {spec.k} for {request_id!r} is not block aligned "
            f"(block_size {geometry.block_size})"
        )


def write_staged_requests(
    layer_tensors: Sequence[torch.Tensor],
    plan: Mapping[str, InjectionSpec],
    registry: Any,
    geometry: Geometry,
) -> None:
    """Scatter every staged request in the step's plan onto the pages' device.

    Staged entries arrive on the host or already co-resident, so they are moved first.
    A runtime failure records the request's blocks on the load-error surface.
    """
    device = layer_tensors[0].device if layer_tensors else torch.device("cpu")
    for request_id, spec in plan.items():
        entry = registry.load(request_id)
        _validate_entry(request_id, entry, spec, geometry)
        slots = slot_mapping(spec.block_ids, spec.k, geometry.block_size).to(device)
        try:
            pack_cache(layer_tensors, entry.keys.to(device), entry.values.to(device), slots)
        except RuntimeError as err:
            registry.record_load_error(spec.block_ids)
            _LOG.error(
                "page write failed for request %r; its %d blocks are reported as load errors",
                request_id,
                len(spec.block_ids),
            )
            raise StagingError(f"page write failed for request {request_id!r}") from err


_lazy_connector_class: Any = None


def __getattr__(name: str) -> Any:
    """Build the vllm-facing connector class on first access (PEP 562)."""
    if name == "RCCConnector":
        global _lazy_connector_class
        if _lazy_connector_class is None:
            _lazy_connector_class = _build_connector_class()
        return _lazy_connector_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_connector_class() -> Any:
    """Subclass the live vllm's KVConnectorBase_V1; raise with pins when vllm is absent."""
    try:
        base_module = importlib.import_module("vllm.distributed.kv_transfer.kv_connector.v1.base")
        envs = importlib.import_module("vllm.envs")
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(_VLLM_INSTALL_HINT) from err
    connector_base: Any = base_module.KVConnectorBase_V1
    metadata_base: Any = base_module.KVConnectorMetadata
    # The bases exist only at runtime, so a super() call is an unknown member to
    # the type checker; these aliases have the same single-inheritance semantics
    # under a checked signature.
    connector_base_init: Callable[..., None] = connector_base.__init__
    metadata_base_init: Callable[..., None] = metadata_base.__init__

    class RCCMetadata(metadata_base):
        """The per-step plan the scheduler ships to the worker: inject and extract."""

        def __init__(
            self,
            plan: dict[str, InjectionSpec],
            extract_specs: list[ExtractSpec] | None = None,
        ) -> None:
            """Wrap one step's injections and its completed extraction orders."""
            metadata_base_init(self)
            self.plan = plan
            self.extract_specs = extract_specs or []

    class RCCConnector(connector_base):
        """Inject and extract selected caches through the engine's paged KV.

        One class serves both roles: the scheduler credits staged lengths and ships
        extraction orders, the worker scatters staged caches and gathers completed ones.
        """

        def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any = None) -> None:
            """Bind the engine config and refuse a configuration the page write cannot serve."""
            connector_base_init(self, vllm_config, role, kv_cache_config)
            if getattr(envs, "VLLM_ENABLE_V1_MULTIPROCESSING", True):
                raise StagingError(
                    "RCCConnector stage 1 requires VLLM_ENABLE_V1_MULTIPROCESSING=0: the "
                    "staging registry is in-process, and a background engine core would "
                    "silently miss every staged cache"
                )
            if bool(getattr(vllm_config.cache_config, "enable_prefix_caching", False)):
                raise StagingError(
                    "RCCConnector requires enable_prefix_caching=False: block hashes key "
                    "on token ids, so a text request could silently hit injected pages "
                    "and vice versa (two-way poisoning)"
                )
            tensor_parallel = int(
                getattr(getattr(vllm_config, "parallel_config", None), "tensor_parallel_size", 1)
            )
            if tensor_parallel != 1:
                raise StagingError(
                    f"RCCConnector stage 1 requires TP == 1, got {tensor_parallel}: the "
                    "page write assumes all KV heads live on one device"
                )
            self._block_size = int(vllm_config.cache_config.block_size)
            self._credited: dict[str, int] = {}
            self._specs: dict[str, InjectionSpec] = {}
            self._geometry: Geometry | None = None
            self._layers: list[torch.Tensor] = []
            self._extract_planner = ExtractionPlanner()

        def get_num_new_matched_tokens(
            self, request: Any, num_computed_tokens: int
        ) -> tuple[int, bool]:
            """Credit exactly the staged length for staged requests, zero otherwise."""
            request_id = str(request.request_id)
            if request_id not in PENDING:
                return 0, False
            if num_computed_tokens != 0:
                raise StagingError(
                    f"request {request_id!r} has {num_computed_tokens} locally computed "
                    "tokens; run the engine with enable_prefix_caching=False"
                )
            entry = PENDING.load(request_id)
            staged_k = int(entry.keys.shape[2])
            credit = credit_tokens(
                staged_k,
                num_request_tokens=int(request.num_tokens),
                block_size=self._block_size,
            )
            self._credited[request_id] = credit
            return credit, False

        def update_state_after_alloc(
            self, request: Any, blocks: Any, num_external_tokens: int
        ) -> None:
            """Capture the allocated physical blocks for a credited request."""
            request_id = str(request.request_id)
            credit = self._credited.pop(request_id, None)
            if credit is None:
                return
            if num_external_tokens != credit:
                raise StagingError(
                    f"engine allocated {num_external_tokens} external tokens for "
                    f"{request_id!r} but {credit} were credited"
                )
            group0 = list(blocks.get_block_ids()[0])
            self._specs[request_id] = InjectionSpec(
                block_ids=capture_block_ids(group0, num_external_tokens, self._block_size),
                k=credit,
            )

        def build_connector_meta(self, scheduler_output: Any) -> Any:
            """Drain this step's captured specs into the worker-bound metadata.

            Extraction orders are picked up on a request's first schedule, with block ownership
            from `scheduled_new_reqs`; later steps ship the scheduled-token count.
            """
            plan = dict(self._specs)
            self._specs.clear()
            for new_request in getattr(scheduler_output, "scheduled_new_reqs", ()):
                request_id = str(new_request.req_id)
                if request_id not in REQUESTED:
                    continue
                REQUESTED.discard(request_id)
                if request_id in PENDING:
                    raise StagingError(
                        f"request {request_id!r} is both staged for injection and marked "
                        "for extraction; unsupported at this stage"
                    )
                self._extract_planner.start(
                    request_id,
                    prompt_token_count(new_request),
                    (int(b) for b in new_request.block_ids[0]),
                )
            cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
            if cached is not None:
                resumed = {str(r) for r in getattr(cached, "resumed_req_ids", ()) or ()}
                new_block_ids = list(getattr(cached, "new_block_ids", ()) or ())
                for position, raw_id in enumerate(getattr(cached, "req_ids", ()) or ()):
                    request_id = str(raw_id)
                    if not self._extract_planner.tracks(request_id):
                        continue
                    entry = new_block_ids[position] if position < len(new_block_ids) else None
                    blocks = tuple(int(b) for b in (entry[0] if entry else ()))
                    if request_id in resumed:
                        # After a preemption vLLM replaces the block list and
                        # recomputes the prompt from zero into fresh blocks.
                        self._extract_planner.reset(request_id, blocks)
                    elif blocks:
                        self._extract_planner.extend(request_id, blocks)
            progress = {
                str(request_id): int(count)
                for request_id, count in getattr(
                    scheduler_output, "num_scheduled_tokens", {}
                ).items()
            }
            extract_specs = self._extract_planner.advance(progress)
            return RCCMetadata(plan, extract_specs)

        def request_finished(self, request: Any, block_ids: Any) -> tuple[bool, Any]:
            """Drop the staged entry, stale credit, and extraction tracking; blocks free."""
            request_id = str(request.request_id)
            PENDING.finish(request_id)
            self._credited.pop(request_id, None)
            self._extract_planner.discard(request_id)
            REQUESTED.discard(request_id)
            return False, None

        def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
            """Read and assert the engine's page map once, before any write."""
            geometry, ordered = read_layer_map(kv_caches, block_size=self._block_size)
            self._geometry = geometry
            self._layers = ordered

        def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
            """Write every staged cache in the bound plan before the forward runs."""
            metadata = self._get_connector_metadata()
            plan = getattr(metadata, "plan", None)
            if not plan:
                return
            if self._geometry is None:
                raise StagingError("start_load_kv before register_kv_caches")
            write_staged_requests(self._layers, plan, PENDING, self._geometry)

        def wait_for_layer_load(self, layer_name: str) -> None:
            """No-op: the load is synchronous and complete before the forward."""
            return None

        def save_kv_layer(
            self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any
        ) -> None:
            """No-op: extraction reads whole pages in `wait_for_save`, per completed prompt."""
            return None

        def wait_for_save(self) -> None:
            """Execute this step's completed extraction orders; stateless on this role.

            The planner emits a spec only on the step whose forward completes the prompt, and
            this hook runs after that forward, so every layer's pages are written.
            """
            metadata = self._get_connector_metadata()
            specs: list[ExtractSpec] = list(getattr(metadata, "extract_specs", None) or [])
            if not specs:
                return
            if self._geometry is None:
                raise StagingError("extraction before register_kv_caches")
            for spec in specs:
                extract_request(self._layers, spec, EXTRACTED, self._geometry)

        def get_block_ids_with_load_errors(self) -> set[int]:
            """Surface failed page writes so the scheduler truncates and reschedules."""
            return PENDING.take_load_errors()

    return RCCConnector
