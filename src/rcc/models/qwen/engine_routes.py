"""The registered Qwen capture routes and the settings one engine open must match.

Three tuples are registered: padded FanOutQA capture, the same at the natural
window, and chain capture. The tables below spell their own values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from rcc.models.route import RouteFamily

QWEN_ENGINE_ROUTE = "vllm-prefill-zero-copy-alias-roll-v1"
QWEN_ENGINE_GPU_MEMORY_UTILIZATION = 0.40
QWEN_ENGINE_MAX_MODEL_LEN = 50_000
QWEN_ENGINE_MAX_BATCHED_TOKENS = 50_000
#: The capture window of a natural-content panel. A padded geometry renders
#: every worker to exactly 50,000 tokens and natural content up to 50,128, so
#: its producer opens this wider window; the run layer picks by profile.
QWEN_NATURAL_ENGINE_MAX_MODEL_LEN = 53_248
QWEN_CAPTURE_WINDOWS = (QWEN_ENGINE_MAX_MODEL_LEN, QWEN_NATURAL_ENGINE_MAX_MODEL_LEN)
QWEN_ENGINE_MAX_NUM_SEQS = 16
QWEN_CHAIN_ENGINE_ROUTE = "vllm-prefill-embeds-zero-copy-alias-roll-v1"
QWEN_CHAIN_ENGINE_MAX_MODEL_LEN = 131_072
#: Chunked prefill at 16,384 batched tokens shrinks the activation reservation
#: the engine profiles, so the share below can drop while the pool still holds
#: the ratio 2 hop, and the side pass gains room for its two live KV copies.
QWEN_CHAIN_ENGINE_MAX_BATCHED_TOKENS = 16_384
#: The KV-pool share of the chain capture engine, keeping the pool above the ratio 2
#: prefill while leaving the side pass its two live KV copies; seats also run with
#: expandable segments (`rcc.run.fleet.schedule`). It moves with the table below.
QWEN_CHAIN_ENGINE_GPU_MEMORY_UTILIZATION = 0.45
#: The share the one-step-prefill backend opens the chain engine at, so the
#: cost chunked prefill adds can be measured. It is not a registered route.
QWEN_CHAIN_ONE_STEP_PREFILL_GPU_MEMORY_UTILIZATION = 0.60

_SHARED_ROUTE: Mapping[str, object] = MappingProxyType(
    {
        "max_num_seqs": 16,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "disable_hybrid_kv_cache_manager": True,
        "disable_log_stats": True,
        "kv_connector": "RCCConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "rcc.injector.connector",
        "v1_multiprocessing": False,
    }
)
#: The capture tuples a live engine may serve, spelled out rather than derived
#: so the table is an independent check.
_FANOUTQA_ROUTE: Mapping[str, object] = MappingProxyType(
    {
        "route": "vllm-prefill-zero-copy-alias-roll-v1",
        "gpu_memory_utilization": 0.40,
        "max_model_len": 50_000,
        "max_num_batched_tokens": 50_000,
        "enable_prompt_embeds": False,
        **_SHARED_ROUTE,
    }
)
_CHAIN_ROUTE: Mapping[str, object] = MappingProxyType(
    {
        "route": "vllm-prefill-embeds-zero-copy-alias-roll-v1",
        "gpu_memory_utilization": 0.45,
        "max_model_len": 131_072,
        "max_num_batched_tokens": 16_384,
        "enable_prompt_embeds": True,
        **_SHARED_ROUTE,
    }
)
#: The natural-content FanOutQA panel opens the same route at the 53,248
#: window, registered independently like the tuples above.
_FANOUTQA_NATURAL_ROUTE: Mapping[str, object] = MappingProxyType(
    {
        "route": "vllm-prefill-zero-copy-alias-roll-v1",
        "gpu_memory_utilization": 0.40,
        "max_model_len": 53_248,
        "max_num_batched_tokens": 53_248,
        "enable_prompt_embeds": False,
        **_SHARED_ROUTE,
    }
)
_REGISTERED_ROUTES: tuple[Mapping[str, object], ...] = (
    _FANOUTQA_ROUTE,
    _FANOUTQA_NATURAL_ROUTE,
    _CHAIN_ROUTE,
)
_ROUTE_FIELDS = tuple(_FANOUTQA_ROUTE)


def _first_route_difference(
    observed: Mapping[str, object], registered: Mapping[str, object]
) -> tuple[str, object] | None:
    """Return the first field that leaves one registered route, and its value."""
    for name in _ROUTE_FIELDS:
        if observed[name] != registered[name]:
            return name, observed[name]
    return None


@dataclass(frozen=True)
class QwenEngineSettings:
    """Exact live-engine settings required by a registered capture route.

    A tuple matching none of the three registered routes is refused; the check
    reads two independent sources, the constants and the tables.
    """

    route: str = QWEN_ENGINE_ROUTE
    gpu_memory_utilization: float = QWEN_ENGINE_GPU_MEMORY_UTILIZATION
    max_model_len: int = QWEN_ENGINE_MAX_MODEL_LEN
    max_num_batched_tokens: int = QWEN_ENGINE_MAX_BATCHED_TOKENS
    max_num_seqs: int = QWEN_ENGINE_MAX_NUM_SEQS
    enforce_eager: bool = True
    enable_prompt_embeds: bool = False
    enable_prefix_caching: bool = False
    disable_hybrid_kv_cache_manager: bool = True
    disable_log_stats: bool = True
    kv_connector: str = "RCCConnector"
    kv_role: str = "kv_both"
    kv_connector_module_path: str = "rcc.injector.connector"
    v1_multiprocessing: bool = False

    def __post_init__(self) -> None:
        """Refuse settings that leave every registered capture route, by field.

        The refusal names, for each registered route, the first field that left
        it and the value that left it.
        """
        observed: dict[str, object] = {name: getattr(self, name) for name in _ROUTE_FIELDS}
        named: list[str] = []
        for registered in _REGISTERED_ROUTES:
            difference = _first_route_difference(observed, registered)
            if difference is None:
                return
            field, value = difference
            named.append(f"{registered['route']} differs first at {field}={value!r}")
        raise ValueError(
            f"Qwen engine settings differ from the registered vLLM route: {'; '.join(named)}"
        )

    @classmethod
    def for_fanoutqa(cls, max_model_len: int | None = None) -> QwenEngineSettings:
        """Return the registered FanOutQA capture tuple at one registered window.

        With no window named, the padded 50,000 route is read off the module
        constant; the natural panel names its 53,248 window explicitly.
        """
        window = QWEN_ENGINE_MAX_MODEL_LEN if max_model_len is None else max_model_len
        return cls(
            route=QWEN_ENGINE_ROUTE,
            gpu_memory_utilization=QWEN_ENGINE_GPU_MEMORY_UTILIZATION,
            max_model_len=window,
            max_num_batched_tokens=window,
            enable_prompt_embeds=False,
        )

    @classmethod
    def for_chain(cls) -> QwenEngineSettings:
        """Return the registered chain capture tuple: the 131,072 embeds route."""
        return cls(
            route=QWEN_CHAIN_ENGINE_ROUTE,
            gpu_memory_utilization=QWEN_CHAIN_ENGINE_GPU_MEMORY_UTILIZATION,
            max_model_len=QWEN_CHAIN_ENGINE_MAX_MODEL_LEN,
            max_num_batched_tokens=QWEN_CHAIN_ENGINE_MAX_BATCHED_TOKENS,
            enable_prompt_embeds=True,
        )

    def engine_overrides(self, *, family: RouteFamily) -> dict[str, object]:
        """Return the fields an orchestration adapter passes to vLLM."""
        return {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "enforce_eager": self.enforce_eager,
            "enable_prompt_embeds": self.enable_prompt_embeds,
            "enable_prefix_caching": self.enable_prefix_caching,
            "disable_hybrid_kv_cache_manager": self.disable_hybrid_kv_cache_manager,
            "disable_log_stats": self.disable_log_stats,
            "hf_overrides": dict(family.hf_overrides),
            "kv_transfer_config": (
                ("kv_connector", self.kv_connector),
                ("kv_role", self.kv_role),
                ("kv_connector_module_path", self.kv_connector_module_path),
            ),
        }
