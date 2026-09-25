"""Copy vLLM 0.26 hybrid state before its scheduler releases the request pages."""

import importlib
import re
from collections.abc import Callable, Sequence
from typing import Any

import torch

from rcc.models.nemotron.cache import new_cache


def _validate_token_request(request: Any, tokens: list[int]) -> None:
    if request.prompt_token_ids != tokens:
        raise RuntimeError("Nemotron capture engine received a different token prompt")


def _embedding_validator(expected: torch.Tensor) -> Callable[[Any], None]:
    def validate(request: Any) -> None:
        actual = getattr(request, "prompt_embeds", None)
        if request.prompt_token_ids is not None or not isinstance(actual, torch.Tensor):
            raise RuntimeError("Nemotron capture engine received a non-embedding prompt")
        if actual.shape != expected.shape or not torch.equal(actual.detach().cpu(), expected):
            raise RuntimeError("Nemotron capture engine received different embedding rows")

    return validate


def _attention(
    pool: torch.Tensor, blocks: list[int], block_size: int, length: int, config: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        pool.ndim != 4
        or pool.shape[1] != config.num_key_value_heads
        or pool.shape[-1] != 2 * config.head_dim
    ):
        raise RuntimeError("Nemotron requires vLLM 0.26 packed FlashAttention KV")
    kernel_size = int(pool.shape[2])
    if block_size % kernel_size or len(blocks) * block_size < length:
        raise RuntimeError("Nemotron attention block geometry is incomplete")
    factor = block_size // kernel_size
    needed = (length + kernel_size - 1) // kernel_size
    pages = [block * factor + offset for block in blocks for offset in range(factor)][:needed]
    if min(pages) < 0 or max(pages) >= pool.shape[0] or len(set(pages)) != len(pages):
        raise RuntimeError("Nemotron attention page table is invalid")
    indices = torch.tensor(pages, device=pool.device, dtype=torch.long)
    packed = (
        pool.index_select(0, indices)
        .transpose(1, 2)
        .reshape(-1, config.num_key_value_heads, 2 * config.head_dim)
    )
    ordered = packed[:length].transpose(0, 1).unsqueeze(0)
    return ordered[..., : config.head_dim].contiguous(), ordered[
        ..., config.head_dim :
    ].contiguous()


def _mamba(
    cache: Any, index: int, pools: Any, blocks: list[int], config: Any, dim_first: bool
) -> None:
    if len(blocks) != 1 or len(pools) != 2:
        raise RuntimeError("Nemotron extraction requires mamba_cache_mode=none and no speculation")
    conv_pool, state_pool = pools
    block = blocks[0]
    if not 0 <= block < min(conv_pool.shape[0], state_pool.shape[0]):
        raise RuntimeError("Nemotron Mamba state block is invalid")
    conv = conv_pool[block].unsqueeze(0)
    if not dim_first:
        conv = conv.transpose(-1, -2)
    state = state_pool[block].unsqueeze(0)
    width = (
        config.mamba_num_heads * config.mamba_head_dim + 2 * config.n_groups * config.ssm_state_size
    )
    if (
        conv.shape != (1, width, config.conv_kernel - 1)
        or state.shape != (1, config.mamba_num_heads, config.mamba_head_dim, config.ssm_state_size)
        or state.dtype != torch.float32
    ):
        raise RuntimeError("Nemotron Mamba state shape or FP32 precision is invalid")
    # vLLM stores K-1 history; HF rolls K slots before consuming the next token.
    cache.update_conv_state(torch.nn.functional.pad(conv, (1, 0)), index)
    cache.update_recurrent_state(state, index)


@torch.inference_mode(False)
@torch.no_grad()
def snapshot_cache(runner: Any, request: Any, config: Any, length: int, *, dim_first: bool) -> Any:
    """Gather each actual cache group into an independently owned native cache."""
    cache = new_cache(config)
    seen: set[int] = set()
    groups = runner.kv_cache_config.kv_cache_groups
    context = runner.compilation_config.static_forward_context
    for group, blocks in zip(groups, request.block_ids, strict=True):
        for name in group.layer_names:
            match = re.fullmatch(r"model\.layers\.(\d+)\.mixer(?:\.attn)?", name)
            if match is None:
                raise RuntimeError(f"Unrecognized Nemotron cache layer {name}")
            index = int(match[1])
            if index in seen or index >= len(config.layers_block_type):
                raise RuntimeError("Duplicate or invalid Nemotron cache layer")
            seen.add(index)
            pool = context[name].kv_cache
            kind = config.layers_block_type[index]
            if kind == "full_attention":
                keys, values = _attention(
                    pool, blocks, group.kv_cache_spec.block_size, length, config
                )
                cache.update(keys, values, index)
            elif kind == "linear_attention":
                _mamba(cache, index, pool, blocks, config, dim_first)
            else:
                raise RuntimeError("Unexpected Nemotron cache layer type")
    expected = {i for i, kind in enumerate(config.layers_block_type) if kind != "mlp"}
    if seen != expected or cache.get_seq_length() != length:
        raise RuntimeError("Nemotron hybrid cache is incomplete")
    return cache


class VllmHybridPrefill:
    """Use the isolated, in-process TP1 capture engine for every worker prefill."""

    def __init__(self, llm: Any, config: Any) -> None:
        """Bind the resident capture engine and its native model configuration."""
        self.llm = llm
        self.config = config
        self.runner = llm.llm_engine.model_executor.driver_worker.model_runner

    def _sampling(self) -> Any:
        vllm: Any = importlib.import_module("vllm")
        return vllm.SamplingParams(temperature=0, max_tokens=1, detokenize=False)

    def _dim_first(self) -> bool:
        utils: Any = importlib.import_module("vllm.model_executor.layers.mamba.mamba_utils")
        return bool(utils.is_conv_state_dim_first())

    def _extract(
        self,
        prompt: Any,
        length: int,
        validate_request: Callable[[Any], None],
    ) -> Any:
        """Snapshot one isolated request before vLLM releases its pages."""
        if length < 1:
            raise ValueError("Nemotron prefill requires a nonempty prefix")
        original = self.runner.execute_model
        captured: list[Any] = []

        def execute(scheduled: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(scheduled, *args, **kwargs)
            for request_id, count in scheduled.num_scheduled_tokens.items():
                request = self.runner.requests[request_id]
                if len(scheduled.num_scheduled_tokens) != 1:
                    raise RuntimeError("Nemotron capture engine must serve one isolated worker")
                validate_request(request)
                if request.num_computed_tokens + count == length:
                    if captured:
                        raise RuntimeError("Nemotron prefill completed more than once")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    captured.append(
                        snapshot_cache(
                            self.runner,
                            request,
                            self.config,
                            length,
                            dim_first=self._dim_first(),
                        )
                    )
            return result

        self.runner.execute_model = execute
        try:
            self.llm.generate([prompt], self._sampling(), use_tqdm=False)
        finally:
            self.runner.execute_model = original
        if len(captured) != 1:
            raise RuntimeError("Nemotron vLLM prefill did not complete its hybrid snapshot")
        return captured[0]

    def extract(self, token_ids: Sequence[int]) -> Any:
        """Snapshot a token prefill before sampling frees its blocks."""
        tokens = list(token_ids)
        return self._extract(
            {"prompt_token_ids": tokens},
            len(tokens),
            lambda request: _validate_token_request(request, tokens),
        )

    def extract_embeds(self, rows: torch.Tensor) -> Any:
        """Snapshot a continuous embedding prefill with the same page guards."""
        if rows.ndim != 2 or rows.shape[0] < 1:
            raise ValueError("Nemotron embedding prefill requires rows shaped [N, D]")
        expected = rows.detach().to(device="cpu", dtype=torch.bfloat16)
        return self._extract(
            {"prompt_embeds": expected}, int(expected.shape[0]), _embedding_validator(expected)
        )
