"""Shared resident-weight view and engine-bridge seam for capture families.

Reach the resident language model, alias its fused tensors into a meta-device
HF skeleton, then drive one greedy request whose prompt pages capture gathers.
"""

from __future__ import annotations

import importlib
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from typing import Any

import torch

RequestFactory = Callable[[str, list[int]], tuple[Any, Any]]


def required_tensor(named: Mapping[str, torch.Tensor], name: str, family: str) -> torch.Tensor:
    """Return one detached resident tensor, refusing a missing name."""
    tensor = named.get(name)
    if tensor is None:
        raise ValueError(f"resident {family} tensors lack {name!r}")
    return tensor.detach()


def shaped(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Return the tensor after checking it against its declared shape."""
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    return tensor


def map_passthrough(
    views: dict[str, torch.Tensor],
    named: Mapping[str, torch.Tensor],
    base: str,
    shapes: Mapping[str, tuple[int, ...]],
    family: str,
) -> None:
    """Copy one layer's unfused tensors straight across, checking each shape."""
    for suffix, shape in shapes.items():
        name = f"{base}.{suffix}"
        views[name] = shaped(name, required_tensor(named, name, family), shape)


def alias_hf_model(
    named: Mapping[str, torch.Tensor],
    views: dict[str, torch.Tensor],
    *,
    family: str,
    skeleton: Callable[[], Any],
    materialize: Callable[[Any, torch.device], None],
    generation_config: Any = None,
) -> Any:
    """Build an eval-mode HF model whose every parameter aliases engine storage."""
    grad_flags = {name: tensor.requires_grad for name, tensor in named.items()}
    device = views["model.embed_tokens.weight"].device
    with torch.device("meta"):
        model = skeleton()
    model.load_state_dict(views, assign=True, strict=True)
    materialize(model, device)
    model.requires_grad_(False)

    source_objects = {id(tensor) for tensor in named.values()}
    source_storage = {
        tensor.untyped_storage().data_ptr()
        for tensor in named.values()
        if tensor.device.type != "meta"
    }
    for name, parameter in model.named_parameters():
        if parameter.device.type == "meta":
            raise ValueError(f"HF {family} parameter {name} remained on meta")
        if id(parameter) in source_objects:
            raise ValueError(f"HF {family} parameter {name} reused an engine object by identity")
        if parameter.untyped_storage().data_ptr() not in source_storage:
            raise ValueError(f"HF {family} parameter {name} does not alias engine storage")
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            raise ValueError(f"HF {family} buffer {name} remained on meta")
    for name, tensor in named.items():
        if tensor.requires_grad != grad_flags[name]:
            raise ValueError(f"aliasing changed requires_grad on engine tensor {name}")
    if generation_config is not None:
        model.generation_config = generation_config
    return model.eval()


def resident_language_model(llm: Any, family: str) -> Any:
    """Follow vLLM 0.26's own in-process cleanup path to the live model.

    The registered architecture is the multimodal wrapper; its ``language_model``
    child is the text stack, under the ``model.layers.*`` names HF expects.
    """
    installed = version("vllm")
    if installed != "0.26.0":
        raise RuntimeError(f"{family} capture requires vllm 0.26.0, found {installed}")
    engine = getattr(llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)
    driver = getattr(executor, "driver_worker", None)
    runner = getattr(driver, "model_runner", None)
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError(
            "cannot reach vLLM 0.26 driver_worker.model_runner.model; "
            "VLLM_ENABLE_V1_MULTIPROCESSING must be 0"
        )
    return getattr(model, "language_model", model)


def resident_named_tensors(model: Any, family: str) -> dict[str, torch.Tensor]:
    """Merge the resident parameters and buffers, refusing a name collision."""
    named = dict(model.named_parameters())
    for name, buffer in model.named_buffers():
        if name in named:
            raise RuntimeError(f"resident {family} tensor name collision at {name!r}")
        named[name] = buffer
    return named


def check_qkv_splits(model: Any, layer_count: int, rows: Callable[[int], tuple[int, int]]) -> None:
    """Refuse a resident QKV fusion whose declared split is not Q, K, V."""
    for layer in range(layer_count):
        q_rows, kv_rows = rows(layer)
        module = model.model.layers[layer].self_attn.qkv_proj
        declared = [int(value) for value in getattr(module, "output_sizes", ())]
        if declared != [q_rows, kv_rows, kv_rows]:
            raise RuntimeError(
                f"vLLM layer {layer} QKV split {declared} != {[q_rows, kv_rows, kv_rows]}"
            )


def vllm_request_factory(request_id: str, token_ids: list[int]) -> tuple[Any, Any]:
    """Build the one-token greedy request whose prompt pages are extracted."""
    del request_id
    vllm: Any = importlib.import_module("vllm")
    inputs: Any = importlib.import_module("vllm.inputs")

    return (
        inputs.TokensPrompt(prompt_token_ids=token_ids),
        vllm.SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
    )


class EngineCaptureBridge:
    """Borrow one live vLLM engine for extraction and the HF side pass."""

    family = ""
    request_prefix = ""

    def __init__(
        self,
        llm: Any,
        model: Any,
        tokenizer: Any,
        *,
        request_factory: RequestFactory,
    ) -> None:
        """Borrow an engine, its aliased HF view, and a token-id request factory."""
        self.llm = llm
        self._model: Any | None = model
        self.tokenizer = tokenizer
        self._request_factory = request_factory

    @property
    def model(self) -> Any:
        """Return the live aliased HF view until it has been released."""
        if self._model is None:
            raise RuntimeError(f"the {self.family} HF view has been released")
        return self._model

    def release_view(self) -> None:
        """Drop only the HF wrapper; the borrowed vLLM engine remains live."""
        self._model = None

    def _mark(self, internal_id: str) -> None:
        raise NotImplementedError

    def _take(self, internal_id: str) -> Any:
        raise NotImplementedError

    def _discard(self, internal_id: str) -> None:
        raise NotImplementedError

    def _rebuild(
        self, artifact: Any, config: Any, *, device: torch.device, dtype: torch.dtype
    ) -> Any:
        raise NotImplementedError

    def _abort(self, request_id: str, *, internal: bool = True) -> None:
        try:
            self.llm.llm_engine.abort_request([request_id], internal=internal)
        except Exception:
            pass

    def _drive(self, request_id: str, token_ids: list[int]) -> str:
        """Drive one logical request and return vLLM's randomized internal id."""
        prompt, sampling = self._request_factory(request_id, token_ids)
        internal_id = self.llm.llm_engine.add_request(request_id, prompt, sampling)
        if not isinstance(internal_id, str) or not internal_id:
            self._abort(request_id, internal=False)
            raise RuntimeError(f"vLLM returned invalid extraction id {internal_id!r}")
        try:
            self._mark(internal_id)
            deadline = time.monotonic() + 600.0
            while self.llm.llm_engine.has_unfinished_requests():
                for output in self.llm.llm_engine.step():
                    if output.request_id == request_id and bool(output.finished):
                        return internal_id
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"{self.family} extraction {request_id!r}/{internal_id!r} "
                        "exceeded 600 seconds"
                    )
            raise RuntimeError(
                f"{self.family} extraction request {request_id!r}/{internal_id!r} never finished"
            )
        except Exception:
            self._abort(internal_id)
            self._discard(internal_id)
            raise

    def extract(self, token_ids: Sequence[int]) -> Any:
        """Prefill the token ids on vLLM and return their rebuilt HF cache."""
        ids = [int(token) for token in token_ids]
        if not ids:
            raise ValueError(f"{self.family} extraction requires at least one prompt token")
        request_id = f"{self.request_prefix}-{uuid.uuid4().hex}"
        internal_id = self._drive(request_id, ids)
        try:
            artifact = self._take(internal_id)
        finally:
            self._discard(internal_id)
        parameter = next(self.model.parameters())
        return self._rebuild(
            artifact,
            self.model.config,
            device=parameter.device,
            dtype=parameter.dtype,
        )


__all__ = (
    "EngineCaptureBridge",
    "RequestFactory",
    "alias_hf_model",
    "check_qkv_splits",
    "map_passthrough",
    "required_tensor",
    "resident_language_model",
    "resident_named_tensors",
    "shaped",
    "vllm_request_factory",
)
