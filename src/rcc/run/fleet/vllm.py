"""Adapter from a resident decode engine to the fleet's add-request/step loop.

It also translates the caller's request ids to and from the engine's own, and
keeps the fleet code from importing vLLM itself.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, cast


class _LogicalOutput:
    """Carry the caller's request id while forwarding one engine output."""

    def __init__(self, source: Any, request_id: str) -> None:
        self._source = source
        self.request_id = request_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


def _single_request_id(raw: object) -> str:
    if not isinstance(raw, list):
        raise RuntimeError(f"vLLM returned invalid request ids {raw!r}")
    request_ids = cast(list[object], raw)
    if len(request_ids) != 1 or not isinstance(request_ids[0], str) or not request_ids[0]:
        raise RuntimeError(f"vLLM returned invalid request ids {raw!r}")
    return request_ids[0]


class VllmEngineHandle:
    """Expose add-request and step over a resident decode engine."""

    def __init__(
        self,
        decode_engine: Any,
        *,
        require_public_enqueue: bool = False,
        detokenize: bool | None = None,
    ) -> None:
        """Bind a resident decode engine and check the enqueue seam it offers."""
        self._sampling_params = importlib.import_module("vllm").SamplingParams
        self._request_output_kind: Any | None = None
        llm = decode_engine._llm
        if llm is None:
            raise RuntimeError("cannot stream through a closed DecodeEngine")
        self._llm = llm
        self._core = llm.llm_engine
        self._detokenize = detokenize
        self._logical_to_internal: dict[str, str] = {}
        self._internal_to_logical: dict[str, str] = {}
        self._output_to_logical: dict[str, str] = {}
        enqueue = cast(Any, getattr(llm, "enqueue", None))
        self._use_public_enqueue = require_public_enqueue
        if require_public_enqueue:
            self._request_output_kind = importlib.import_module(
                "vllm.sampling_params"
            ).RequestOutputKind
            if not callable(enqueue):
                raise RuntimeError("vLLM LLM.enqueue is absent")
            llm_type = cast(Any, type(llm))
            public_enqueue: Any = llm_type.enqueue
            signature = list(inspect.signature(public_enqueue).parameters)
            if signature[:3] != ["self", "prompts", "sampling_params"]:
                raise RuntimeError(f"vLLM LLM.enqueue signature drifted: {signature}")

    def sampling(self, sampling: Any, seed: int) -> Any:
        """Translate a family sampling contract into vLLM parameters."""
        values = {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
        }
        # A family with top-k off renders no top_k field at all, which reads
        # here as off rather than as a missing attribute.
        top_k = getattr(sampling, "top_k", None)
        if top_k and top_k > 0:
            values["top_k"] = top_k
        if sampling.presence_penalty != 0:
            values["presence_penalty"] = sampling.presence_penalty
        if sampling.stop_token_ids is not None:
            values["stop_token_ids"] = list(sampling.stop_token_ids)
        bad_words = getattr(sampling, "bad_words", None)
        if bad_words is not None:
            values["bad_words"] = list(bad_words)
        if self._detokenize is not None:
            values["detokenize"] = self._detokenize
        return self._sampling_params(**values, seed=seed)

    def add_request(self, request_id: str, prompt: Any, sampling_params: Any) -> None:
        """Add one request through whichever enqueue seam was selected."""
        if not self._use_public_enqueue:
            self._core.add_request(request_id, prompt, sampling_params)
            return
        if request_id in self._logical_to_internal:
            raise ValueError(f"request {request_id!r} is already admitted")
        raw = cast(object, self._llm.enqueue([prompt], sampling_params, use_tqdm=False))
        internal_id = _single_request_id(raw)
        if internal_id in self._internal_to_logical:
            raise RuntimeError(f"vLLM reused live request id {internal_id!r}")
        processor = getattr(self._core, "output_processor", None)
        states_object = cast(object, getattr(processor, "request_states", None))
        states = cast(dict[str, Any], states_object) if isinstance(states_object, dict) else None
        state: Any | None = states.get(internal_id) if states is not None else None
        if state is None:
            raise RuntimeError(f"vLLM omitted request state for {internal_id!r}")
        output_id = str(getattr(state, "external_req_id", ""))
        if not output_id:
            raise RuntimeError(f"vLLM omitted output request id for {internal_id!r}")
        output_kind = self._request_output_kind
        if output_kind is None:
            raise RuntimeError("vLLM public output kind is unavailable")
        if getattr(state, "output_kind", None) != output_kind.FINAL_ONLY:
            raise RuntimeError(f"vLLM enqueue output mode drifted for {internal_id!r}")
        state.output_kind = output_kind.CUMULATIVE
        if output_id in self._output_to_logical:
            raise RuntimeError(f"vLLM reused live output request id {output_id!r}")
        self._logical_to_internal[request_id] = internal_id
        self._internal_to_logical[internal_id] = request_id
        self._output_to_logical[output_id] = request_id

    def step(self) -> Any:
        """Step the engine and restore the caller's request ids."""
        outputs: Any = self._core.step()
        if not self._use_public_enqueue:
            return outputs
        translated: list[Any] = []
        for output in cast(list[Any], outputs):
            output_id = str(getattr(output, "request_id", ""))
            logical_id = self._output_to_logical.get(output_id)
            if logical_id is None:
                raise RuntimeError(f"vLLM returned unknown request id {output_id!r}")
            translated.append(_LogicalOutput(output, logical_id))
            if bool(getattr(output, "finished", False)):
                internal_id = self._logical_to_internal[logical_id]
                del self._internal_to_logical[internal_id]
                del self._logical_to_internal[logical_id]
                del self._output_to_logical[output_id]
        return translated

    def has_unfinished_requests(self) -> bool:
        """Return whether the engine still holds unfinished work."""
        return bool(self._core.has_unfinished_requests())

    def abort_request(self, request_ids: list[str]) -> None:
        """Abort these requests and release their id mappings."""
        if not self._use_public_enqueue:
            self._core.abort_request(request_ids)
            return
        internal_ids = [
            self._logical_to_internal[request_id]
            for request_id in request_ids
            if request_id in self._logical_to_internal
        ]
        if internal_ids:
            self._core.abort_request(internal_ids, internal=True)
        for request_id in request_ids:
            internal_id = self._logical_to_internal.pop(request_id, None)
            if internal_id is not None:
                self._internal_to_logical.pop(internal_id, None)
                output_ids = [
                    output_id
                    for output_id, logical_id in self._output_to_logical.items()
                    if logical_id == request_id
                ]
                for output_id in output_ids:
                    del self._output_to_logical[output_id]
