"""The Qwen three-worker resident latent producer."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import Any

import torch

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.nemotron.producer import NemotronProducer
from rcc.models.qwen.capture import (
    QwenFlatPayload,
    QwenWorkerCapture,
    build_flat_payload,
    tensor_content_sha256,
)
from rcc.models.qwen.engine import QWEN_ENGINE_ROUTE, EngineProducer, WorkerProduct
from rcc.models.qwen.prompts import manager_prompt_record, worker_prompts
from rcc.models.route import RouteFamily
from rcc.transforms.select.query_support.methods.compose import SUPPORT_DEFAULT_ALPHA


@dataclass(frozen=True)
class QwenLatentProduction:
    """One embedding-only payload and its resident capture measurements."""

    qid: str
    semantic_arm: str
    payload: QwenFlatPayload
    selector_score_name: str
    selector_scores: tuple[torch.Tensor, ...]
    prompt_sha256: tuple[str, ...]
    manager_prompt_sha256: str
    extract_s: float
    roll_s: float
    capture_s: float
    capture_peak_bytes: int
    #: Rendered worker prompt widths.
    prompt_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate payload identity and nonnegative capture measurements."""
        if self.payload.semantic_arm != self.semantic_arm:
            raise ValueError("Qwen production and payload semantic arms differ")
        if self.selector_score_name not in {"snap", "support"}:
            raise ValueError("Qwen production has an unregistered selector score")
        widths = self.prompt_tokens
        if len(self.selector_scores) != 3 or len(widths) != 3:
            raise ValueError("Qwen production must retain three finite selector vectors")
        adapter = self.payload.family.score_adapter(self.semantic_arm)
        if any(
            score.shape
            != (
                adapter.shape(width + FANOUTQA_NATURAL_DEV50.latent_steps)
                if adapter
                else (width + FANOUTQA_NATURAL_DEV50.latent_steps,)
            )
            or not bool(score.isfinite().all())
            for score, width in zip(self.selector_scores, widths, strict=True)
        ):
            raise ValueError("Qwen production must retain three finite selector vectors")
        if adapter:
            for score in self.selector_scores:
                adapter.validate(score)
        if len(self.prompt_sha256) != 3:
            raise ValueError("Qwen production must carry three prompt records")
        if any(value < 0 for value in (self.extract_s, self.roll_s, self.capture_s)):
            raise ValueError("Qwen production timings must be nonnegative")
        if self.capture_peak_bytes < 0:
            raise ValueError("Qwen production peak allocation must be nonnegative")

    def result_fields(self) -> dict[str, object]:
        """Return payload lineage and producer measurements for a result row."""
        fields: dict[str, object] = {
            "producer_backend": "engine",
            "producer_route": dict(self.payload.family.profile.runtime.engine_flags).get(
                "producer_backend"
            )
            if self.payload.family.lane == "nemotron"
            else QWEN_ENGINE_ROUTE,
            "payload_layout": self.payload.layout,
            "payload_semantic_arm": self.payload.semantic_arm,
            "payload_plan_sha256": self.payload.latent_plan_sha256,
            "payload_tensor_sha256": self.payload.tensor_sha256,
            "selected_indices_sha256": self.payload.selected_indices_sha256,
            "selector_score_name": self.selector_score_name,
            "selector_score_tensor_sha256": [
                tensor_content_sha256(score) for score in self.selector_scores
            ],
            "keeps_by_worker": [list(keep) for keep in self.payload.keeps],
            "latent_rows_by_worker": list(self.payload.rows_by_worker),
            "latent_tokens": int(self.payload.rows.shape[0]),
            "worker_prompt_sha256": list(self.prompt_sha256),
            "manager_prompt_sha256": self.manager_prompt_sha256,
            "extract_s": round(self.extract_s, 4),
            "latent_roll_s": round(self.roll_s, 4),
            "selector_capture_s": round(self.capture_s, 4),
            "selector_capture_peak_bytes": self.capture_peak_bytes,
        }
        if self.semantic_arm.startswith("latent_query_support_"):
            fields["support_alpha"] = SUPPORT_DEFAULT_ALPHA
        adapter = self.payload.family.score_adapter(self.semantic_arm)
        if adapter:
            fields.update(
                selector_recipe=adapter.recipe,
                selector_score_schema=adapter.score_schema,
                selector_capture_peak_scope=adapter.peak_scope,
            )
        return fields


def produce_latent_payload(
    producer: EngineProducer | NemotronProducer,
    tokenizer: Any,
    item: ProbeItem,
    *,
    semantic_arm: str,
    family: RouteFamily,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> QwenLatentProduction:
    """Capture three native workers, apply their registered selector and emit flat rows."""
    adapter = family.score_adapter(semantic_arm)
    if adapter is not None:
        adapter.validate_tokenizer(tokenizer)
    rendered = worker_prompts(item, tokenizer, family=family, profile=profile)
    prompt_ids = tuple(
        tuple(int(token) for token in tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in rendered
    )
    # The judger is the receiver prompt whose question rows score the rolled
    # cache. The ablation replaces only that question: the workers above read
    # the item's own, and so does the receiver that finally answers.
    judger_item = item
    if profile.judger_question is not None:
        judger_item = replace(item, question=profile.judger_question)
    manager = manager_prompt_record(
        judger_item, tokenizer, family=family, profile=FANOUTQA_NATURAL_DEV50, channel="text"
    )
    judger = torch.tensor(
        (manager.token_ids,),
        dtype=torch.long,
        device=producer.device,
    )
    judger_mask = torch.ones_like(judger)
    captures: list[QwenWorkerCapture] = []
    products: list[WorkerProduct] = []
    scoring_extra_s = 0.0
    for ids, prompt in zip(prompt_ids, rendered, strict=True):
        product = producer.produce_worker(
            torch.tensor((ids,), dtype=torch.long),
            judger_ids=judger,
            judger_mask=judger_mask,
            question_ids=manager.question_token_ids,
        )
        products.append(product)
        started = time.perf_counter()
        bank = None
        if adapter is not None:
            bank = adapter.prepare(product, tokenizer, prompt, item.question, ids)
        scoring_extra_s += time.perf_counter() - started
        captures.append(
            QwenWorkerCapture(
                embedding_rows=product.embeds,
                scores=product.scores,
                question_found=product.question_found,
                prompt_token_ids=ids,
                selector_components=bank,
            )
        )

    def embed(token_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor(token_ids, dtype=torch.long, device=producer.device)
        with torch.no_grad():
            return producer.model.get_input_embeddings()(ids).detach()

    payload = build_flat_payload(captures, semantic_arm=semantic_arm, embed=embed, family=family)
    selector = next(
        arm.selector for arm in FANOUTQA_NATURAL_DEV50.arms if arm.arm_id == semantic_arm
    )
    if selector not in {"snap", "support"}:
        raise RuntimeError(f"{semantic_arm}: Qwen latent producer has no selector score")
    return QwenLatentProduction(
        qid=item.qid,
        semantic_arm=semantic_arm,
        payload=payload,
        selector_score_name=selector,
        selector_scores=tuple(
            capture.selector_components
            if capture.selector_components is not None
            else capture.scores[selector].detach().float().cpu()
            for capture in captures
        ),
        prompt_sha256=tuple(hashlib.sha256(prompt.encode()).hexdigest() for prompt in rendered),
        manager_prompt_sha256=manager.sha256,
        extract_s=sum(product.extract_s for product in products),
        roll_s=sum(product.roll_s for product in products),
        capture_s=sum(product.capture_s for product in products) + scoring_extra_s,
        capture_peak_bytes=max(product.capture_peak_bytes for product in products),
        prompt_tokens=tuple(len(ids) for ids in prompt_ids),
    )


__all__ = ("QwenLatentProduction", "produce_latent_payload")
