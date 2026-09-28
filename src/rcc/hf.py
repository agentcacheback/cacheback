"""Small Hugging Face bindings for existing dense Qwen3 models."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput

from rcc import vllm
from rcc._async import run_in_worker
from rcc._cache import clone_cache
from rcc.latent import forward_rows
from rcc.transport import Delivery, SenderState, token_ids, validate_model

Chat = Sequence[dict[str, Any]]


@dataclass(frozen=True)
class PreparedRequest:
    """A receiver prompt snapshot and its communication/output allowances."""

    input_ids: torch.Tensor
    max_new_tokens: int
    available_positions: int


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _chat(tokenizer: Any, messages: Chat, *, generation_prompt: bool) -> str:
    if tokenizer is None:
        raise ValueError("chat input needs a tokenizer; bind one or pass token IDs")
    if not tokenizer.chat_template:
        raise ValueError("tokenizer has no chat template; supply a rendered prompt or exact IDs")
    if not messages:
        raise ValueError("chat messages must not be empty")
    return cast(
        str,
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=generation_prompt,
            enable_thinking=False,
        ),
    )


def _complete_cache(model: Any, rows: torch.Tensor, past: Any) -> Any:
    if rows.shape[0] > model.config.max_position_embeddings:
        raise ValueError("sender prompt exceeds the model's context limit")
    length = 0 if past is None else past.get_seq_length()
    if not 0 <= length <= rows.shape[0]:
        raise ValueError("sender cache length exceeds its recorded input tokens")
    if length < rows.shape[0]:
        past = forward_rows(model, rows[length:][None], clone_cache(past), length).past_key_values
    return past


@torch.inference_mode()
def sender_from_hf(
    model: Any,
    tokenizer: Any,
    prompt: str | torch.Tensor | Chat | None = None,
    *,
    past_key_values: Any = None,
    generation: object = None,
) -> SenderState:
    """Bind chat, exact tokens or a native generation result, prefilling only uncached tokens."""
    validate_model(model)
    if generation is not None:
        if prompt is not None or past_key_values is not None:
            raise ValueError("choose generation or prompt/past_key_values, not both")
        if not isinstance(generation, GenerateDecoderOnlyOutput):
            raise ValueError("generation result must be a native non-beam decoder-only output")
        prompt = generation.sequences
        past_key_values = generation.past_key_values
    if isinstance(prompt, Sequence) and not isinstance(prompt, str):
        prompt = _chat(tokenizer, prompt, generation_prompt=False)
    if not isinstance(prompt, (str, torch.Tensor)):
        raise ValueError("provide a prompt, chat messages or a native generation result")
    ids = token_ids(model, tokenizer, prompt, "prompt")
    rows = model.get_input_embeddings()(ids)[0]
    state = SenderState(
        model,
        _complete_cache(model, rows, past_key_values),
        rows,
        tokenizer=tokenizer,
        token_ids=ids[0],
    )
    state.validate()
    return state


class HFReceiver:
    """Queue handoffs for a matching Qwen3 checkpoint; leave generation to the agent."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        messages: Chat | None = None,
        max_new_tokens: int | None = None,
        context_limit: int | None = None,
    ) -> None:
        """Bind an already-loaded receiver; sender and receiver must use matching weights."""
        validate_model(model)
        self.model = model
        self.tokenizer = tokenizer
        self.messages: Chat = [] if messages is None else messages
        self.max_new_tokens = (
            None if max_new_tokens is None else _positive(max_new_tokens, "max_new_tokens")
        )
        self.context_limit = (
            model.config.max_position_embeddings
            if context_limit is None
            else _positive(context_limit, "context_limit")
        )
        if self.context_limit > model.config.max_position_embeddings:
            raise ValueError("context_limit cannot exceed the model's context limit")
        self._deliveries: deque[tuple[Delivery, PreparedRequest]] = deque()

    def __call__(self, delivery: Delivery) -> None:
        """Enqueue one delivery without running the receiver model."""
        self.enqueue(delivery, self.prepare_request(delivery.request))

    def _prompt_ids(self, request: str | torch.Tensor) -> torch.Tensor:
        if isinstance(request, str):
            request = _chat(
                self.tokenizer,
                [*self.messages, {"role": "user", "content": request}],
                generation_prompt=True,
            )
        elif self.messages:
            raise ValueError("a receiver with chat history requires a text request")
        return token_ids(self.model, self.tokenizer, request, "request")

    def prepare_request(self, request: str | torch.Tensor) -> PreparedRequest:
        """Snapshot a configured receiver prompt before transport allocates its budget."""
        ids = self._prompt_ids(request)
        output = self.max_new_tokens or 1
        return PreparedRequest(ids.clone(), output, self.context_limit - ids.shape[1] - output)

    def enqueue(self, delivery: Delivery, prepared: PreparedRequest) -> None:
        """Queue a delivery with the prompt already prepared by transport."""
        self._deliveries.append((delivery, prepared))

    def __len__(self) -> int:
        """Return the number of pending receiver requests."""
        return len(self._deliveries)

    def selection_html(self, request_index: int = 0) -> str:
        """Render a queued request's recorded selections without consuming its delivery."""
        if type(request_index) is not int or not 0 <= request_index < len(self._deliveries):
            raise ValueError("request_index must identify a queued request")
        return self._deliveries[request_index][0].selection_html()

    def validate_sender(self, sender: SenderState) -> None:
        """Reject known checkpoint or vocabulary mismatches before selection."""
        validate_model(self.model)
        source, target = sender.model.config, self.model.config
        fields = (
            "model_type",
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
        if any(getattr(source, field, None) != getattr(target, field, None) for field in fields):
            raise ValueError("sender and receiver must use the same checkpoint architecture")
        for field in ("_name_or_path", "_commit_hash"):
            left, right = getattr(source, field, None), getattr(target, field, None)
            if left and right and left != right:
                raise ValueError("sender and receiver checkpoint names and revisions must match")
        other = sender.tokenizer
        if other is None or self.tokenizer is None or other is self.tokenizer:
            return
        if other.get_vocab() != self.tokenizer.get_vocab():
            raise ValueError("sender and receiver must use the same tokenizer vocabulary")

    def _next(self) -> tuple[Delivery, PreparedRequest]:
        if not self._deliveries:
            raise IndexError(
                "no queued handoff; await rcc.transfer (or call transfer_sync) before pop"
            )
        return self._deliveries[0]

    def pop(
        self, *, prompt: str | torch.Tensor | None = None, max_new_tokens: int | None = None
    ) -> dict[str, Any]:
        """Pop native generation arguments using the queued prompt and reserved output space."""
        return self._pop(prompt, max_new_tokens)[0]

    @torch.inference_mode()
    def _pop(
        self, prompt: str | torch.Tensor | None, max_new_tokens: int | None
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Return generation inputs and their prefix IDs, with -1 for continuous rows."""
        validate_model(self.model)
        delivery, prepared = self._next()
        output = (
            prepared.max_new_tokens
            if max_new_tokens is None
            else _positive(max_new_tokens, "max_new_tokens")
        )
        ids = (
            prepared.input_ids
            if prompt is None
            else token_ids(self.model, self.tokenizer, prompt, "prompt")
        )[0]
        positions = sum(message.positions for message in delivery.messages) + ids.shape[0]
        if positions + output > self.context_limit:
            raise ValueError(
                f"receiver has no context space for {output} output tokens: "
                f"{positions} input positions + {output} exceed {self.context_limit}; "
                "increase ratio, reduce budget or shorten the receiver prompt"
            )
        embeddings = self.model.get_input_embeddings()
        rows = [message.materialize(embeddings.weight) for message in delivery.messages]
        known = [
            ids.new_full((message.positions,), -1)
            if message.token_ids is None
            else message.token_ids.to(ids.device)
            for message in delivery.messages
        ]
        inputs = self._generation_inputs(torch.cat((*rows, embeddings(ids)))[None], output)
        self._deliveries.popleft()
        return inputs, torch.cat((*known, ids))

    def _generation_inputs(self, rows: torch.Tensor, output: int) -> dict[str, Any]:
        mask = torch.ones(rows.shape[:2], dtype=torch.long, device=rows.device)
        return {"inputs_embeds": rows, "attention_mask": mask, "max_new_tokens": output}


class Agent(HFReceiver):
    """Bind an existing agent that can send state and receive queued handoffs."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        *,
        backend: str = "hf",
        messages: Chat | None = None,
        prompt: str | torch.Tensor | None = None,
        past_key_values: Any = None,
        generation: object = None,
        max_new_tokens: int = 128,
        context_limit: int | None = None,
        request_id: str | None = None,
        allow_unstable: bool = False,
    ) -> None:
        """Attach to a dense Qwen3 model; state is prepared lazily from chat unless supplied."""
        if backend not in ("hf", "vllm"):
            raise ValueError("backend must be 'hf' or 'vllm'")
        self.backend = backend
        self.allow_unstable = allow_unstable
        self.engine = model if backend == "vllm" else None
        if self.engine is not None:
            model, tokenizer, context_limit = vllm.binding_model(
                model, tokenizer, context_limit, allow_unstable
            )
        elif request_id is not None or allow_unstable:
            raise ValueError("request_id and allow_unstable require backend='vllm'")
        super().__init__(
            model,
            tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
            context_limit=context_limit,
        )
        self._state: SenderState | None = None
        self._from_history = True
        self._pending_inputs: dict[str, Any] | None = None
        self._pending_ids = torch.empty(0, dtype=torch.long)
        self._pending_inherited = 0
        if any(value is not None for value in (prompt, past_key_values, generation, request_id)):
            self.update(
                prompt,
                past_key_values=past_key_values,
                generation=generation,
                request_id=request_id,
            )

    def _history(self) -> str:
        if not self.messages:
            raise ValueError(
                "agent has no state: bind it with messages, prompt or generation, "
                "or record state with agent.update"
            )
        return _chat(self.tokenizer, self.messages, generation_prompt=False)

    @torch.inference_mode()
    def sender_state(self) -> SenderState:
        """Resolve current sender state, reusing a matching chat prefix when history grows."""
        if self.engine is not None:
            vllm.require_idle(self.engine)
        if self._pending_inputs is not None:
            raise ValueError("record the prepared continuation with agent.update before sending")
        state = self._state
        if not self._from_history and state is not None:
            return state
        ids = token_ids(self.model, self.tokenizer, self._history(), "prompt")
        past = None
        if state is not None and state.token_ids is not None:
            previous = state.token_ids
            if torch.equal(ids[0, : previous.numel()], previous):
                if ids.shape[1] == previous.numel():
                    return state
                past = state.past_key_values
        self._state = (
            sender_from_hf(self.model, self.tokenizer, ids, past_key_values=past)
            if self.engine is None
            else vllm.prefill_state(self.engine, self.model, self.tokenizer, ids[0])
        )
        return self._state

    def _generation_inputs(self, rows: torch.Tensor, output: int) -> dict[str, Any]:
        if self.engine is None:
            return super()._generation_inputs(rows, output)
        vllm.require_idle(self.engine)
        return vllm.generation_inputs(rows[0], output)

    def _set_pending(self, inputs: dict[str, Any], ids: torch.Tensor, inherited: int) -> None:
        self._pending_inputs, self._pending_ids, self._pending_inherited = inputs, ids, inherited

    async def append(self, prompt: str | torch.Tensor) -> None:
        """Append rendered text or exact token IDs while preserving received continuous state."""
        state = await run_in_worker(self._appended_state, prompt)
        self._state, self._from_history = state, False

    async def inputs(self, *, max_new_tokens: int | None = None) -> dict[str, Any]:
        """Prepare native generation from retained state without consuming queued handoffs."""
        pending = await run_in_worker(self._continuation_inputs, max_new_tokens)
        self._set_pending(*pending)
        return pending[0]

    def _continuation_inputs(self, maximum: int | None) -> tuple[dict[str, Any], torch.Tensor, int]:
        state = self.sender_state()
        rows = state.input_embeds
        output = (
            (self.max_new_tokens or 1) if maximum is None else _positive(maximum, "max_new_tokens")
        )
        if len(rows) + output > self.context_limit:
            raise ValueError("retained state leaves insufficient context space for output tokens")
        known = state.token_ids
        if known is None:
            known = torch.full((len(rows),), -1, dtype=torch.long, device=rows.device)
        inputs = self._generation_inputs(
            (rows if self.engine is not None else rows.clone())[None], output
        )
        return inputs, known, state.inherited_positions

    def inspect(self, *, selection: bool = False) -> dict[str, Any]:
        """Describe cached state and queued handoffs without generation or consuming the queue."""
        deliveries: list[dict[str, Any]] = []
        for delivery, prepared in self._deliveries:
            positions = [message.positions for message in delivery.messages]
            deliveries.append(
                {
                    "senders": len(positions),
                    "retained_positions": positions,
                    "payload_bytes": sum(message.nbytes for message in delivery.messages),
                    "prompt_positions": prepared.input_ids.shape[1],
                    "reserved_output_positions": prepared.max_new_tokens,
                    "remaining_positions": prepared.available_positions - sum(positions),
                }
            )
            if selection:
                deliveries[-1]["selection"] = [
                    None if message.selection is None else message.selection.inspect()
                    for message in delivery.messages
                ]
        return {
            "backend": self.backend,
            "state_source": "messages" if self._from_history else "retained",
            "cached_positions": 0 if self._state is None else len(self._state.input_embeds),
            "pending_generation": self._pending_inputs is not None,
            "queued_requests": len(self),
            "deliveries": deliveries,
        }

    def _require_settled(self) -> None:
        if self._pending_inputs is not None or self._deliveries:
            raise ValueError(
                "record pending generation and consume queued handoffs before save/load"
            )

    async def save(self, path: str | Path) -> None:
        """Atomically save retained input state, checking that no pending work would be lost."""
        from rcc.checkpoint import write_state

        self._require_settled()
        state = await run_in_worker(self.sender_state)
        await run_in_worker(write_state, path, state)

    async def load(self, path: str | Path) -> None:
        """Check a saved state's checkpoint identity and re-prefill it on this bound model."""
        from rcc.checkpoint import read_state

        self._require_settled()
        state = await run_in_worker(read_state, path, self)
        self._state, self._from_history = state, False

    @torch.inference_mode()
    def _appended_state(self, prompt: str | torch.Tensor) -> SenderState:
        state = self.sender_state()
        added = token_ids(self.model, self.tokenizer, prompt, "prompt")[0]
        rows = torch.cat((state.input_embeds, self.model.get_input_embeddings()(added)))
        if rows.shape[0] > self.context_limit:
            raise ValueError("appended input exceeds the agent's context limit")
        previous = state.token_ids
        if previous is None:
            previous = added.new_full((state.input_embeds.shape[0],), -1)
        ids = torch.cat((previous, added))
        if self.engine is not None:
            return vllm.prefill_state(
                self.engine,
                self.model,
                self.tokenizer,
                ids,
                input_embeds=rows,
                inherited_positions=state.inherited_positions,
            )
        extended = replace(
            state,
            input_embeds=rows,
            token_ids=ids,
            past_key_values=_complete_cache(self.model, rows, state.past_key_values),
            latent_steps=0,
        )
        extended.validate()
        return extended

    def pop(
        self,
        *,
        prompt: str | torch.Tensor | None = None,
        max_new_tokens: int | None = None,
        discard_pending: bool = False,
    ) -> dict[str, Any]:
        """Prepare native generation inputs and remember their received prefix for update."""
        if self._pending_inputs is not None and not discard_pending:
            raise ValueError(
                "record the previous continuation with agent.update before popping again; "
                "use pop(discard_pending=True) to explicitly discard it"
            )
        inherited = sum(message.positions for message in self._next()[0].messages)
        inputs, ids = self._pop(prompt, max_new_tokens)
        self._set_pending(inputs, ids, inherited)
        return inputs

    def _continuation(
        self, inputs: dict[str, Any], completion: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rows and known IDs for the pending prefix followed by generated tokens."""
        embeddings = self.model.get_input_embeddings()
        prefix = (
            inputs["inputs_embeds"][0]
            if self.engine is None
            else inputs["prompts"]["prompt_embeds"].to(embeddings.weight)
        )
        ids = token_ids(self.model, self.tokenizer, completion, "generation")[0]
        return torch.cat((prefix, embeddings(ids))), torch.cat((self._pending_ids, ids))

    @torch.inference_mode()
    def update(
        self,
        prompt: str | torch.Tensor | None = None,
        *,
        past_key_values: Any = None,
        generation: object = None,
        inputs: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        """Record native generation, a token prompt/cache, or the current chat history."""
        validate_model(self.model)
        if self.engine is not None:
            if past_key_values is not None:
                raise ValueError(
                    "use request_id for captured vLLM prompt state, not past_key_values"
                )
            self._update_vllm(prompt, generation, inputs, request_id)
            return
        if request_id is not None:
            raise ValueError("request_id requires backend='vllm'")
        if inputs is None:
            if self._pending_inputs is not None and generation is not None:
                raise ValueError(
                    "pass inputs=the_pop_result when recording a received continuation"
                )
            from_history = prompt is None and generation is None
            state = sender_from_hf(
                self.model,
                self.tokenizer,
                self._history() if from_history else prompt,
                past_key_values=past_key_values,
                generation=generation,
            )
        else:
            if (
                inputs is not self._pending_inputs
                or prompt is not None
                or past_key_values is not None
            ):
                raise ValueError(
                    "inputs must be this agent's most recent pop/inputs result, "
                    "without prompt/cache"
                )
            if not isinstance(generation, GenerateDecoderOnlyOutput):
                raise ValueError("use native non-beam generation with return_dict_in_generate=True")
            rows, ids = self._continuation(inputs, generation.sequences)
            state = SenderState(
                self.model,
                _complete_cache(self.model, rows, generation.past_key_values),
                rows,
                tokenizer=self.tokenizer,
                token_ids=ids,
                inherited_positions=self._pending_inherited,
            )
            state.validate()
            from_history = False
        self._state, self._from_history = state, from_history
        self._pending_inputs = None

    def _update_vllm(
        self,
        prompt: str | torch.Tensor | None,
        generation: object,
        inputs: dict[str, Any] | None,
        request_id: str | None,
    ) -> None:
        if inputs is not None:
            if inputs is not self._pending_inputs or prompt is not None or request_id is not None:
                raise ValueError(
                    "inputs must be this agent's most recent pop/inputs result, "
                    "without prompt/request_id"
                )
        elif generation is not None:
            raise ValueError(
                "vLLM generation updates need inputs=the_pop_result; "
                "otherwise supply the complete prompt"
            )
        elif request_id is not None and prompt is None:
            raise ValueError("provide the exact captured prompt with request_id")
        vllm.require_idle(self.engine)
        rows, inherited = None, 0
        if inputs is not None:
            rows, ids = self._continuation(inputs, vllm.completion_ids(generation))
            inherited = self._pending_inherited
        else:
            rendered = self._history() if prompt is None else prompt
            ids = token_ids(self.model, self.tokenizer, rendered, "prompt")[0]
        state = (
            vllm.prefill_state(
                self.engine,
                self.model,
                self.tokenizer,
                ids,
                input_embeds=rows,
                inherited_positions=inherited,
            )
            if request_id is None
            else vllm.state_from_capture(self.model, self.tokenizer, request_id, ids)
        )
        self._state = state
        self._from_history = prompt is None and inputs is None
        self._pending_inputs = None
