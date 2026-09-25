"""Single-rank vLLM executor with atomically assigned rendezvous ports."""

from __future__ import annotations

import importlib
import os
from functools import cache
from typing import Any


def distributed_args(config: Any) -> tuple[str, int, int]:
    """Let the sole TCPStore server allocate and retain its own ephemeral port."""
    parallel = config.parallel_config
    if parallel.world_size != 1 or parallel.data_parallel_size != 1:
        raise RuntimeError("Gemma atomic rendezvous requires a single-rank executor")
    if os.environ.get("TORCHELASTIC_USE_AGENT_STORE", "").lower() in {"true", "1"}:
        raise RuntimeError("Gemma atomic rendezvous cannot use an external agent store")
    device = str(config.device_config.device).split(":")
    local_rank = int(device[1]) if len(device) > 1 else 0
    return "tcp://127.0.0.1:0", 0, local_rank


@cache
def executor_class() -> Any:
    """Load the pinned executor lazily so CPU artifact readers need no vLLM."""
    module: Any = importlib.import_module("vllm.v1.executor.uniproc_executor")

    class GemmaSingleRankExecutor(module.UniProcExecutor):
        def _distributed_args(self) -> tuple[str, int, int]:
            return distributed_args(self.vllm_config)

    GemmaSingleRankExecutor.__qualname__ = "GemmaSingleRankExecutor"
    globals()["GemmaSingleRankExecutor"] = GemmaSingleRankExecutor
    return GemmaSingleRankExecutor


def __getattr__(name: str) -> Any:
    """Resolve the importable executor name for vLLM configuration serialization."""
    if name == "GemmaSingleRankExecutor":
        return executor_class()
    raise AttributeError(name)
