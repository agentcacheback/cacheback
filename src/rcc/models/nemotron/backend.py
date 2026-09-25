"""Nemotron-specific engine settings on the shared vLLM decode adapter."""

import importlib
from typing import Any

from rcc.models.nemotron import CAPTURE_MAX_MODEL_LEN
from rcc.models.nemotron.bridge import native_view
from rcc.models.nemotron.runtime_policy import after_engine, attest
from rcc.models.qwen.backend import QwenBackendSettings, QwenVllmBackend
from rcc.models.qwen.engine_routes import QWEN_CHAIN_ENGINE_MAX_MODEL_LEN
from rcc.models.route import RouteFamily


def engine_kwargs(
    kwargs: dict[str, object],
    *,
    capture: bool,
    chain: bool = False,
    one_step_prefill: bool = False,
) -> dict[str, object]:
    """Use native hybrid cache management and leave room for the producer side pass.

    The one-step diagnostic keeps the prefill tuple the shared chain overrides
    set, so the 4,096-token chunking and 0.40 share apply only when it is off.
    """
    kwargs.pop("kv_transfer_config", None)
    kwargs.pop("disable_hybrid_kv_cache_manager", None)
    kwargs.update(
        trust_remote_code=False,
        mamba_ssm_cache_dtype="float32",
        enable_prefix_caching=False,
        generation_config="auto",
        tensor_parallel_size=1,
        enable_prompt_embeds=True,
        compilation_config={"inductor_compile_config": {"deterministic": True}},
    )
    if capture:
        kwargs.update(
            max_model_len=CAPTURE_MAX_MODEL_LEN,
            max_num_seqs=1,
            mamba_cache_mode="none",
            attention_config={"backend": "FLASH_ATTN"},
            enforce_eager=True,
        )
        if not (chain and one_step_prefill):
            kwargs.update(max_num_batched_tokens=4096, gpu_memory_utilization=0.40)
        if chain:
            kwargs["max_model_len"] = QWEN_CHAIN_ENGINE_MAX_MODEL_LEN
    return kwargs


class NemotronBackend(QwenVllmBackend):
    """Expose the native weight view while keeping the shared decode seams."""

    def __init__(
        self,
        llm: Any,
        settings: QwenBackendSettings,
        family: RouteFamily,
        *,
        policy_before: dict[str, Any],
    ) -> None:
        """Restore and attest the declared policy after native engine construction."""
        super().__init__(llm, settings, family)
        try:
            self._compiler_policy = after_engine(llm, policy_before, capture=settings.capture)
        except BaseException:
            super().close()
            raise

    def runtime_policy_record(self) -> dict[str, Any]:
        """Read the current compiler state for the role's durable phase bank."""
        return attest(self.raw_llm, self._compiler_policy, capture=self.settings.capture)

    def close(self) -> None:
        """Check post-workload policy and release resources even when it drifted."""
        try:
            if self._llm is not None:
                self.runtime_policy_record()
        finally:
            super().close()

    def resident_model(self) -> Any:
        """Return the native embedding and hybrid-model view."""
        return native_view(self.raw_llm)

    def kv_pool_tokens(self) -> int | None:
        """Read the attention-block pool size from the vLLM 0.26 runner."""
        engine = self.raw_llm.llm_engine
        runner = engine.model_executor.driver_worker.model_runner
        cache = getattr(runner, "kv_cache_config", None)
        if cache is None:
            raise RuntimeError("Nemotron receiver cannot read the actual hybrid KV allocation")
        utility: Any = importlib.import_module("vllm.v1.core.kv_cache_utils")
        tokens, _concurrency = utility.get_kv_cache_capacity(engine.vllm_config, cache)
        if not isinstance(tokens, int) or tokens < 1:
            raise RuntimeError("Nemotron hybrid token capacity is invalid")
        return tokens
