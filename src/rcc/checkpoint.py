"""Portable input-state snapshots, restored by re-prefill on matching weights."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import weakref
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors import safe_open
from safetensors import torch as tensor_io

from rcc.latent import forward_rows
from rcc.transport import SenderState, validate_model, validate_rows

if TYPE_CHECKING:
    from rcc.hf import Agent

save_file: Callable[[dict[str, torch.Tensor], str, dict[str, str]], None] = cast(
    Any, tensor_io
).save_file
_WEIGHT_DIGESTS: weakref.WeakKeyDictionary[Any, tuple[object, str]] = weakref.WeakKeyDictionary()


def _fingerprint(parameters: Mapping[str, torch.Tensor]) -> object:
    """Return storage and version counters that change when a weight is edited or replaced."""
    try:
        return tuple((name, t.data_ptr(), t._version) for name, t in parameters.items())
    except RuntimeError:
        return None


def _hash_weights(parameters: Mapping[str, torch.Tensor]) -> str:
    """Hash every weight's name, shape, dtype and bytes."""
    digest = hashlib.sha256()
    for name, parameter in sorted(parameters.items()):
        digest.update(json.dumps([name, list(parameter.shape), str(parameter.dtype)]).encode())
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), 4 * 1024 * 1024):
            chunk = flat[start : start + 4 * 1024 * 1024]
            raw = cast(Any, chunk.cpu().contiguous().view(torch.uint8)).numpy()
            digest.update(memoryview(raw))
    return digest.hexdigest()


def _weight_digest(model: Any) -> str:
    """Hash the weights, reusing the hash while no weight has been edited or replaced."""
    parameters = cast(Mapping[str, torch.Tensor], model.state_dict())
    fingerprint = _fingerprint(parameters)
    cached = _WEIGHT_DIGESTS.get(model)
    if fingerprint is not None and cached is not None and cached[0] == fingerprint:
        return cached[1]
    result = _hash_weights(parameters)
    if fingerprint is not None:
        _WEIGHT_DIGESTS[model] = (fingerprint, result)
    return result


def _identity(model: Any, tokenizer: Any) -> str:
    """Hash the weights, forward configuration and tokenizer vocabulary.

    The weight hash is reused until a weight's storage or version counter changes.
    """
    fields = (
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "hidden_act",
        "sliding_window",
        "layer_types",
    )
    config = {name: getattr(model.config, name, None) for name in fields}
    rope = dict(
        getattr(model.config, "rope_parameters", None)
        or getattr(model.config, "rope_scaling", None)
        or {}
    )
    rope.setdefault("rope_type", rope.pop("type", "default"))
    rope.setdefault("rope_theta", getattr(model.config, "rope_theta", None))
    config["rope"] = rope
    vocabulary = None if tokenizer is None else tokenizer.get_vocab()
    identity = [config, vocabulary, _weight_digest(model)]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def write_state(path: str | Path, state: SenderState) -> None:
    """Write tensors and plain metadata with atomic replacement, without serializing a model."""
    state.validate()
    tensors = {"rows": state.input_embeds.detach().cpu().contiguous()}
    if state.token_ids is not None:
        tensors["ids"] = state.token_ids.detach().cpu().contiguous()
    metadata = {
        "rcc": json.dumps(
            {
                "format": 1,
                "identity": _identity(state.model, state.tokenizer),
                "latent_steps": state.latent_steps,
                "inherited_positions": state.inherited_positions,
            }
        )
    }
    destination = Path(path)
    descriptor, name = tempfile.mkstemp(dir=destination.parent)
    os.close(descriptor)
    try:
        save_file(tensors, name, metadata)
        with open(name, "rb") as temporary:
            os.fsync(temporary.fileno())
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def _read_payload(path: str | Path) -> dict[str, Any]:
    with cast(
        AbstractContextManager[Any], safe_open(path, framework="pt", device="cpu")
    ) as archive:
        metadata = cast(dict[str, str], archive.metadata() or {})
        raw: Any = json.loads(metadata.get("rcc", "{}"))
        if not isinstance(raw, dict):
            raise ValueError("not an RCC state snapshot")
        payload = cast(dict[str, Any], raw)
        fields = {"format", "identity", "latent_steps", "inherited_positions"}
        if set(payload) != fields or type(payload["format"]) is not int or payload["format"] != 1:
            raise ValueError("unsupported RCC state snapshot format")
        keys = set(cast(list[str], archive.keys()))
        if "rows" not in keys or keys - {"rows", "ids"}:
            raise ValueError("snapshot must contain rows and optional token IDs")
        payload["rows"] = archive.get_tensor("rows")
        payload["ids"] = archive.get_tensor("ids") if "ids" in keys else None
    return payload


@torch.inference_mode()
def read_state(path: str | Path, agent: Agent) -> SenderState:
    """Validate an input snapshot before rebuilding its cache on the caller's engine."""
    from rcc import vllm

    validate_model(agent.model)
    if agent.engine is not None:
        vllm.require_idle(agent.engine)
    payload = _read_payload(path)
    rows, ids = payload["rows"], payload["ids"]
    weight = agent.model.get_input_embeddings().weight
    if rows.ndim != 2 or rows.dtype != weight.dtype or rows.shape[1] != weight.shape[1]:
        raise ValueError(
            f"snapshot rows have shape {tuple(rows.shape)} and dtype {rows.dtype}; "
            f"bound model has width {weight.shape[1]} and dtype {weight.dtype}"
        )
    if not 0 < len(rows) <= agent.context_limit or not bool(torch.isfinite(rows).all()):
        raise ValueError("snapshot rows are nonfinite, empty or exceed the agent's context limit")
    rows = rows.to(weight.device)
    ids = (torch.full((len(rows),), -1, dtype=torch.long) if ids is None else ids).to(rows.device)
    state = SenderState(
        agent.model,
        None,
        rows,
        tokenizer=agent.tokenizer,
        token_ids=ids,
        latent_steps=payload["latent_steps"],
        inherited_positions=payload["inherited_positions"],
    )
    validate_rows(state)
    if payload["identity"] != _identity(agent.model, agent.tokenizer):
        raise ValueError(
            "snapshot requires identical weights, configuration and tokenizer vocabulary"
        )
    discrete = ids >= 0
    if not torch.equal(rows[discrete], agent.model.get_input_embeddings()(ids[discrete])):
        raise ValueError("snapshot token IDs do not reproduce their input rows")
    if agent.engine is None:
        past = forward_rows(agent.model, rows[None], None, 0).past_key_values
    else:
        prefilled = vllm.prefill_state(
            agent.engine, agent.model, agent.tokenizer, ids, input_embeds=rows
        )
        past = prefilled.past_key_values
    state.past_key_values = past
    state.validate()
    return state
