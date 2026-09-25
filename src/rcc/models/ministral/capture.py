"""Dense Ministral FanOutQA capture through the resident vLLM bridge.

One worker's framed prompt is prefilled on the borrowed engine, rolled for the
registered latent steps, and scored with query-support.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from rcc.latent.rollout import Realign, latent_rollout
from rcc.topologies.fanout import FANOUT_M3
from rcc.transforms.select.core.types import CaptureSpec
from rcc.transforms.select.query_support.capture_bank import (
    BANK_SCOPE_GLOBAL_ONLY,
    LayerBank,
)
from rcc.transforms.select.query_support.methods.compose import (
    SUPPORT_DEFAULT_ALPHA,
    SUPPORT_DEFAULT_ORDER,
    _pin_sink,
    selector_score,
)
from rcc.transforms.select.query_support.methods.single import capture_query_support

LATENT_STEPS = 40
POOL_KERNEL = 7
SUPPORT_SELECTOR = f"support-p{SUPPORT_DEFAULT_ORDER:g}-a{SUPPORT_DEFAULT_ALPHA:g}"
#: Every Ministral latent arm ships query-support and no mean query attention arm is
#: registered, so the capture scores that one selector.
CAPTURE_SELECTORS = ("support",)


class ResidentCaptureBridge(Protocol):
    """Borrowed HF-shaped view over one live TP=1 Ministral engine."""

    model: Any
    tokenizer: Any

    def extract(self, token_ids: Sequence[int]) -> Any:
        """Return the dense engine-prefilled cache for direct token ids."""
        ...

    def release_view(self) -> None:
        """Drop the HF view without closing the borrowed engine."""
        ...


class RollFrame(Protocol):
    """Minimal framed-prompt seam consumed by a worker capture."""

    def prompt(self, memory_ids: torch.Tensor) -> torch.Tensor:
        """Return the exact registered worker prompt token ids."""
        ...


@dataclass(frozen=True)
class MinistralWorkerCapture:
    """One worker's complete prompt-plus-roll embedding and selector artifact."""

    scores: Mapping[str, torch.Tensor]
    layer_bank: LayerBank
    embedding_rows: torch.Tensor
    prompt_token_ids: tuple[int, ...]
    prompt_tokens: int

    def __post_init__(self) -> None:
        """Validate the complete capture artifact before it can be selected."""
        if LATENT_STEPS != 40:
            raise RuntimeError("Ministral capture requires the registered 40-step roll")
        if set(self.scores) != set(CAPTURE_SELECTORS):
            raise ValueError("Ministral capture must contain exactly the support score")
        payload = self.prompt_tokens + LATENT_STEPS
        if any(score.shape != (payload,) for score in self.scores.values()):
            raise ValueError("Ministral selector scores do not cover the rolled payload")
        if any(
            not bool(torch.isfinite(score).all()) or bool((score < 0).any())
            for score in self.scores.values()
        ):
            raise ValueError("Ministral selector scores must be finite and nonnegative")
        if self.prompt_tokens != len(self.prompt_token_ids):
            raise ValueError("Ministral prompt token identity differs from its geometry")
        if self.embedding_rows.ndim != 2 or tuple(self.embedding_rows.shape[:1]) != (payload,):
            raise ValueError("Ministral capture rows do not cover prompt plus 40-step roll")
        if self.embedding_rows.dtype != torch.bfloat16 or self.embedding_rows.device.type != "cpu":
            raise ValueError("Ministral handoff rows must be CPU bfloat16")

    @property
    def rolled(self) -> torch.Tensor:
        """Return the protected 40-row latent tail of the complete artifact."""
        return self.embedding_rows[-LATENT_STEPS:]


def _cache_length(past: Any) -> int:
    getter = getattr(past, "get_seq_length", None)
    if not callable(getter):
        raise RuntimeError("Ministral capture bridge returned no HF cache length")
    return int(cast(Any, getter)())


def _require_dense_cache(past: Any, *, expected_length: int, expected_layers: int) -> None:
    """Require every post-roll dense layer to retain every logical row."""
    if _cache_length(past) != expected_length:
        raise RuntimeError(
            f"Ministral rolled cache length {_cache_length(past)} != {expected_length}"
        )
    layers = list(getattr(past, "layers", ()) or ())
    if len(layers) != expected_layers:
        raise RuntimeError(
            f"Ministral rolled cache has {len(layers)} layers, expected {expected_layers}"
        )
    for index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise RuntimeError(f"Ministral rolled cache layer {index} has no KV tensors")
        if keys.shape != values.shape or int(keys.shape[-2]) != expected_length:
            raise RuntimeError(f"Ministral rolled cache layer {index} dropped logical rows")
        if not bool(torch.isfinite(keys).all()) or not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"Ministral rolled cache layer {index} is non-finite")


def _require_dense_bank_geometry(tag: str, bank: LayerBank, payload: int) -> None:
    """Reject sliding, windowed, incomplete, or wrong-length capture rows."""
    if bank.scope != BANK_SCOPE_GLOBAL_ONLY:
        raise RuntimeError(f"{tag}: Ministral layer bank is not global-only")
    rows = bank.selected_rows(BANK_SCOPE_GLOBAL_ONLY)
    if not rows:
        raise RuntimeError(f"{tag}: Ministral layer bank contains no rows")
    for row in rows:
        descriptor = row.descriptor
        if not descriptor.is_global or descriptor.window is not None:
            raise RuntimeError(f"{tag}: Ministral layer bank contains a non-dense row")
        if descriptor.logical_memory_length != payload:
            raise RuntimeError(f"{tag}: Ministral layer bank row has the wrong payload length")


def _capture_scores(
    tag: str,
    model: Any,
    past: Any,
    judger_ids: torch.Tensor,
    question_ids: torch.Tensor,
    *,
    payload: int,
) -> tuple[dict[str, torch.Tensor], LayerBank]:
    """Capture the shipped p2-a2 support score and replay it from the bank.

    The replay is same-device and must equal the live score bit for bit. What
    ships is the bank's CPU replay; bit exactness across devices is not claimed.
    """
    if SUPPORT_DEFAULT_ORDER != 2.0 or SUPPORT_DEFAULT_ALPHA != 2.0:
        raise RuntimeError("Ministral capture requires support-p2-a2")
    result = capture_query_support(
        model,
        past,
        judger_ids,
        torch.ones_like(judger_ids),
        question_ids,
        pool_kernel=POOL_KERNEL,
        spec=CaptureSpec(
            energy_pool_kernel=POOL_KERNEL,
            row_energy_pool_kernel=POOL_KERNEL,
            support_orders=(SUPPORT_DEFAULT_ORDER,),
            collect_statistics=True,
            collect_layer_bank=True,
        ),
    )
    if not result.found or result.moments is None or result.layer_bank is None:
        raise RuntimeError(f"{tag}: Ministral Query-Support capture is incomplete")
    bank = result.layer_bank
    _require_dense_bank_geometry(tag, bank, payload)

    device = judger_ids.device
    live_support = selector_score(result.moments, SUPPORT_SELECTOR).float()
    replay_support = selector_score(
        bank.replay_moments(scope=BANK_SCOPE_GLOBAL_ONLY, device=device),
        SUPPORT_SELECTOR,
    ).float()
    if not torch.equal(replay_support, live_support):
        raise RuntimeError(f"{tag}: durable p2-a2 replay differs from live capture")

    scores = {
        "support": _pin_sink(
            bank.replay_score(SUPPORT_SELECTOR, scope=BANK_SCOPE_GLOBAL_ONLY).float().cpu(),
            1,
        ),
    }
    if any(score.shape != (payload,) for score in scores.values()):
        raise RuntimeError(f"{tag}: selector score length differs from the rolled payload")
    return scores, bank


def capture_worker(
    bridge: ResidentCaptureBridge,
    item: Mapping[str, Any],
    worker: int,
    realign: Realign,
    frame: RollFrame,
) -> MinistralWorkerCapture:
    """Extract prefix KV, land the final token, and roll exactly 40 steps."""
    if LATENT_STEPS != 40:
        raise RuntimeError("Ministral capture requires the registered 40-step roll")
    if realign.enabled:
        raise RuntimeError("Ministral capture requires the registered identity realign")
    if not 0 <= worker < FANOUT_M3.workers_per_item:
        raise ValueError(f"Ministral worker index {worker} is outside M=3")

    memory_rows = cast(Sequence[torch.Tensor], item["memory_ids"])
    prompt_rows = cast(Sequence[torch.Tensor], item["prompt_ids"])
    memory_ids = memory_rows[worker]
    prompt_ids = prompt_rows[worker]
    if not torch.equal(prompt_ids, frame.prompt(memory_ids)):
        raise RuntimeError(f"{item['qid']}/w{worker}: framed prompt differs from its source")
    prompt_tokens = int(prompt_ids.numel())
    if prompt_tokens < 2:
        raise RuntimeError(f"{item['qid']}/w{worker}: framed prompt has no extractable prefix")

    model = bridge.model
    parameter = next(model.parameters())
    device = parameter.device
    prompt_ids = prompt_ids.to(device=device)
    prefix_length = prompt_tokens - 1
    prefix_ids = cast(
        list[int],
        prompt_ids[:-1].tolist(),  # pyright: ignore[reportUnknownMemberType]
    )
    past = bridge.extract(prefix_ids)
    if _cache_length(past) != prefix_length:
        raise RuntimeError(
            f"{item['qid']}/w{worker}: extracted cache has {_cache_length(past)} rows, "
            f"expected {prefix_length}"
        )
    roll = latent_rollout(
        model,
        prompt_ids[-1:].unsqueeze(0),
        latent_steps=LATENT_STEPS,
        realign=realign,
        past=past,
        record_embeds=True,
    )
    payload = prompt_tokens + LATENT_STEPS
    expected_layers = int(model.config.num_hidden_layers)
    _require_dense_cache(
        roll.past,
        expected_length=payload,
        expected_layers=expected_layers,
    )
    if roll.embeds is None or tuple(roll.embeds.shape[:2]) != (1, LATENT_STEPS + 1):
        raise RuntimeError(f"{item['qid']}/w{worker}: latent roll recorded the wrong rows")
    embedding = model.get_input_embeddings()
    hidden = int(embedding.weight.shape[1])
    rolled = roll.embeds[0, -LATENT_STEPS:].detach().to(dtype=torch.bfloat16).cpu()
    if rolled.shape != (LATENT_STEPS, hidden):
        raise RuntimeError(f"{item['qid']}/w{worker}: latent cargo has the wrong shape")
    with torch.no_grad():
        prompt_rows = embedding(prompt_ids).detach().to(dtype=torch.bfloat16).cpu()
    embedding_rows = torch.cat((prompt_rows, rolled), dim=0)
    if embedding_rows.shape != (payload, hidden):
        raise RuntimeError(f"{item['qid']}/w{worker}: handoff rows have the wrong shape")

    judger_ids = cast(torch.Tensor, item["judger_ids"]).to(device=device)
    question_ids = cast(torch.Tensor, item["question_ids"]).to(device=device)
    tag = f"{item['qid']}/w{worker}"
    scores, bank = _capture_scores(
        tag,
        model,
        roll.past,
        judger_ids,
        question_ids,
        payload=payload,
    )
    prompt_token_ids = cast(
        list[int],
        prompt_ids.detach().cpu().tolist(),  # pyright: ignore[reportUnknownMemberType]
    )
    return MinistralWorkerCapture(
        scores=scores,
        layer_bank=bank,
        embedding_rows=embedding_rows,
        prompt_token_ids=tuple(prompt_token_ids),
        prompt_tokens=prompt_tokens,
    )


__all__ = (
    "CAPTURE_SELECTORS",
    "LATENT_STEPS",
    "POOL_KERNEL",
    "SUPPORT_SELECTOR",
    "MinistralWorkerCapture",
    "ResidentCaptureBridge",
    "RollFrame",
    "capture_worker",
)
