"""Scoped observers for the three Nemotron Mamba execution paths."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch

from rcc.models.nemotron.mamba_history import History
from rcc.models.nemotron.selection_contract import LAYERS


def argument(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, index: int) -> Any:
    """Read a pinned kernel's positional or keyword argument."""
    return args[index] if len(args) > index else kwargs[name]


class ObserveMamba(AbstractContextManager["ObserveMamba"]):
    """Delegate every kernel unchanged and copy only selected-layer intermediates."""

    def __init__(self, llm: Any, model: Any, length: int, *, trace: bool = False) -> None:
        """Bind isolated native and engine models without changing any weights."""
        self.llm, self.model = llm, model
        self.histories = {layer: History(length, trace) for layer in LAYERS}
        self.restore: list[tuple[Any, str, Any]] = []
        self.active: int | None = None

    def replace(self, owner: Any, name: str, replacement: Callable[..., Any]) -> None:
        """Record restoration before installing an observation wrapper."""
        self.restore.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def native_forward(self, layer: int, original: Callable[..., Any]) -> Callable[..., Any]:
        """Associate native scan/update calls with their actual layer."""

        def forward(*args: Any, **kwargs: Any) -> Any:
            previous, self.active = self.active, layer
            try:
                return original(*args, **kwargs)
            finally:
                self.active = previous

        return forward

    def kernel(
        self, original: Callable[..., Any], route: str, pointers: dict[int, int]
    ) -> Callable[..., Any]:
        """Observe actual x/B/C/delta and the returned pre-normalization output."""

        def call(*args: Any, **kwargs: Any) -> Any:
            update = route == "native_update"
            offset = int(update)
            a = argument(args, kwargs, "A", 2 + offset)
            layer = pointers.get(a.data_ptr()) if route == "vllm_prefill" else self.active
            result = original(*args, **kwargs)
            if layer not in self.histories:
                return result
            x, dt, b, c = [
                argument(args, kwargs, name, index + offset)
                for name, index in (("x", 0), ("dt", 1), ("B", 3), ("C", 4))
            ]
            bias = kwargs["dt_bias"]
            d = argument(args, kwargs, "D", 6) if update else kwargs["D"]
            if update:
                if x.shape[0] != 1:
                    raise ValueError("native observation requires one worker")
                dt, bias, a, d = dt[:, :, 0], bias[:, 0], a[:, 0, 0], d[:, 0]
                reference = result
            elif route == "native_scan":
                if x.shape[0] != 1:
                    raise ValueError("native scan observation requires one worker")
                x, dt, b, c = x[0], dt[0], b[0], c[0]
                reference = result[0][0]
            else:
                reference = kwargs["out"]
            processed_dt = dt.float() + bias.float()
            if not kwargs.get("dt_softplus", False):
                raise ValueError("unexpected Mamba discretization")
            processed_dt = torch.nn.functional.softplus(processed_dt)
            limits = kwargs.get("dt_limit", (0.0, float("inf")))
            processed_dt = processed_dt.clamp(min=limits[0], max=limits[1])
            self.histories[layer].append(
                x,
                b,
                c,
                processed_dt,
                a,
                d,
                reference,
                route,
                chunk_size=int(kwargs.get("chunk_size", 1 if update else 128)),
            )
            return result

        return call

    def __enter__(self) -> ObserveMamba:
        """Install wrappers only after the engine and native kernels have initialized."""
        vllm = importlib.import_module("vllm.model_executor.layers.mamba.mamba_mixer2")
        native = importlib.import_module("transformers.models.nemotron_h.modeling_nemotron_h")
        runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
        context = runner.compilation_config.static_forward_context
        pointers = {context[f"model.layers.{layer}.mixer"].A.data_ptr(): layer for layer in LAYERS}
        try:
            self.replace(
                vllm,
                "mamba_chunk_scan_combined_varlen",
                self.kernel(vllm.mamba_chunk_scan_combined_varlen, "vllm_prefill", pointers),
            )
            self.replace(
                native,
                "mamba_chunk_scan_combined",
                self.kernel(native.mamba_chunk_scan_combined, "native_scan", {}),
            )
            self.replace(
                native,
                "selective_state_update",
                self.kernel(native.selective_state_update, "native_update", {}),
            )
            for layer in LAYERS:
                mixer = self.model.model.layers[layer].mixer
                self.replace(mixer, "forward", self.native_forward(layer, mixer.forward))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Restore wrappers even on a capture failure."""
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)
        self.restore.clear()
