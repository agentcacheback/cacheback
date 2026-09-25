"""Fake vLLM engine seams shared by the engine producer and chain engine tests.

One fake serves both prefill paths, token ids and prompt embeddings, so the
chain hop and the FanOutQA worker run through the same seam.
"""

from __future__ import annotations

from typing import Any

import torch

from rcc.injector.extraction import EXTRACTED
from rcc.injector.staging import StagedCache
from rcc.models.qwen.capture import QWEN_LATENT_STEPS
from rcc.models.qwen.engine import EngineHandle, EngineProducer


class FakeCompletionOutput:
    """The one throwaway sampled token a capture request asks for."""

    def __init__(self, token_ids: list[int]) -> None:
        """Hold the sampled ids of one completion."""
        self.token_ids = token_ids


class FakeStepOutput:
    """One finished request as the engine reports it to the drive loop."""

    def __init__(self, request_id: str, *, num_cached_tokens: int = 0) -> None:
        """Report a finished request and its prefix-cache evidence."""
        self.request_id = request_id
        self.finished = True
        self.num_cached_tokens = num_cached_tokens
        self.outputs = [FakeCompletionOutput([0])]


def _staged(past: Any, rows: int) -> StagedCache:
    layers = [(layer.keys, layer.values) for layer in past.layers]
    return StagedCache(
        keys=torch.stack([keys.squeeze(0) for keys, _ in layers]),
        values=torch.stack([values.squeeze(0) for _, values in layers]),
        positions=torch.arange(rows, dtype=torch.int64),
    )


def staged_from_prefill(model: Any, token_ids: torch.Tensor) -> StagedCache:
    """Prefill the tiny model and shape its KV the way extraction stages it."""
    with torch.no_grad():
        past = model(input_ids=token_ids, use_cache=True, return_dict=True).past_key_values
    return _staged(past, int(token_ids.shape[1]))


def staged_from_embeds(model: Any, rows: torch.Tensor) -> StagedCache:
    """Prefill from embedding rows the way the engine prefills prompt embeds."""
    with torch.no_grad():
        past = model(
            inputs_embeds=rows.unsqueeze(0),
            use_cache=True,
            return_dict=True,
        ).past_key_values
    return _staged(past, int(rows.shape[0]))


class FakeEngine:
    """Admits a prefill and, on the completing step, stages its KV under the request id.

    A prompt is either a list of token ids or a `{"prompt_embeds": rows}`
    mapping, the two shapes the registered capture routes submit.
    """

    def __init__(
        self,
        model: Any,
        *,
        reject: bool = False,
        drop: bool = False,
        num_cached_tokens: int = 0,
    ) -> None:
        """Bind one tiny model and the failure mode this engine stands for."""
        self._model = model
        self._reject = reject
        self._drop = drop
        self._num_cached_tokens = num_cached_tokens
        self._live: dict[str, list[int] | torch.Tensor] = {}
        self.aborted: list[str] = []

    def add_request(self, request_id: str, prompt: Any, sampling: Any) -> None:
        """Admit one token or prompt-embeddings prefill."""
        del sampling
        if self._reject:
            raise ValueError(f"prompt too long for {request_id!r}")
        if self._drop:
            return
        if isinstance(prompt, dict):
            rows = torch.as_tensor(prompt["prompt_embeds"])
            self._live[request_id] = rows
            return
        self._live[request_id] = list(prompt)

    def step(self) -> list[FakeStepOutput]:
        """Finish every live request and stage the KV its prefill produced."""
        outputs: list[FakeStepOutput] = []
        for request_id, prompt in list(self._live.items()):
            if isinstance(prompt, torch.Tensor):
                staged = staged_from_embeds(self._model, prompt)
            else:
                staged = staged_from_prefill(self._model, torch.tensor([prompt], dtype=torch.long))
            EXTRACTED.put(request_id, staged)
            outputs.append(FakeStepOutput(request_id, num_cached_tokens=self._num_cached_tokens))
        self._live.clear()
        return outputs

    def has_unfinished_requests(self) -> bool:
        """Report whether any admitted request is still resident."""
        return bool(self._live)

    def abort_request(self, request_ids: list[str]) -> None:
        """Record the ids the producer aborted."""
        self.aborted.extend(request_ids)


def request_factory(request_id: str, token_ids: list[int]) -> tuple[Any, Any]:
    """Submit token ids with no vLLM import."""
    del request_id
    return token_ids, None


def embeds_request_factory(request_id: str, rows: torch.Tensor) -> tuple[Any, Any]:
    """Submit embedding rows with no vLLM import and no dtype cast.

    The production cast to bfloat16 is free on served weights but would
    quantize the float32 tiny model, so rows stay in the model's own dtype.
    """
    del request_id
    return {"prompt_embeds": rows}, None


def fake_producer(
    model: Any,
    *,
    latent_steps: int = QWEN_LATENT_STEPS,
    **kwargs: Any,
) -> EngineProducer:
    """Build the production producer over one fake engine and both factories."""
    handle = EngineHandle(
        engine=FakeEngine(model, **kwargs),
        model=model,
        request_factory=request_factory,
        embeds_request_factory=embeds_request_factory,
    )
    return EngineProducer(handle, latent_steps=latent_steps)


__all__ = (
    "FakeEngine",
    "embeds_request_factory",
    "fake_producer",
    "request_factory",
)
