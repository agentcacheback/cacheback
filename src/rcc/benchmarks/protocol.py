"""The benchmark registration values, with no dependencies on the rest of the tree.

One arm and one benchmark's panel, topology, arms, ceilings, and seed rules. A
profile's identity dict and its hash are what every run of it is checked against.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass


@dataclass(frozen=True)
class ArmSpec:
    """One model-independent arm from the shared FanOutQA roster."""

    arm_id: str
    channel: str
    selector: str | None = None
    retention_ratio: int | None = None
    sender_role: str | None = None
    #: The resident-budget law a chain arm runs. A fan-out arm names none, and
    #: the key enters the identity only when set.
    budget_law: str | None = None
    #: The fixed carried row count of a ``bounded`` chain arm, entered only when set.
    budget_rows: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return this arm's identity dict."""
        return {
            "arm_id": self.arm_id,
            "channel": self.channel,
            "selector": self.selector,
            "retention_ratio": self.retention_ratio,
            "sender_role": self.sender_role,
            **({} if self.budget_law is None else {"budget_law": self.budget_law}),
            **({} if self.budget_rows is None else {"budget_rows": self.budget_rows}),
        }


@dataclass(frozen=True)
class BenchmarkProfile:
    """One benchmark: its source panel, topology, arms, ceilings, and seed rules."""

    profile_id: str
    source_commit: str
    source_logical_fingerprint: str
    source_manifest_sha256: str
    source_archive_sha256: str
    question_index_sha256: str
    prepared_artifact_sha256: str
    prepared_config_fingerprint: str
    panel_registration_sha256: str
    source_audit_fingerprint: str
    question_ids: tuple[str, ...]
    workers_per_item: int
    worker_prompt_tokens: int
    latent_steps: int
    span_width: int
    ratios: tuple[int, ...]
    selectors: tuple[str, ...]
    arms: tuple[ArmSpec, ...]
    sealed_qwen_policies: tuple[tuple[str, str], ...]
    sample_tags: tuple[str, ...]
    answer_seed_namespace: str
    report_seed_base: int
    answer_ceiling: int
    report_ceiling: int
    max_model_len: int
    #: The sampling presence penalty one report draw decodes under, bounding the same
    #: draw as `report_ceiling`. It sits last because `max_model_len` has no default,
    #: and enters the identity only for a profile that names a penalty.
    report_presence_penalty: float = 0.0
    # Execution identity, outside `to_dict()` and so outside
    # `scientific_identity_hash`: these fields name how a benchmark runs on the
    # shared pipeline, not what it measures.
    _: KW_ONLY
    #: The config key this profile is registered under.
    benchmark_key: str
    #: The arm profiles a config may name for this benchmark.
    arm_profiles: tuple[str, ...]
    #: The topology this benchmark runs on.
    topology_key: str
    #: Run id stem of the route lanes only (Qwen and the lanes sharing its
    #: route): `{lane}-{run_id_stem}-{label}`. The resident Gemma and Ministral
    #: lanes create their own run ids and never read this field.
    run_id_stem: str
    #: Output prefix of the route lanes only: `{root}/{output_prefix}/{run_id}`.
    #: Gemma and Ministral keep their own prefixes.
    output_prefix: str
    #: The declared passes of one shared bank, in order.
    declared_passes: tuple[tuple[int, int], ...]
    #: The embedding payload layout the receiver accepts.
    payload_layout: str
    #: The scorer and the per-sample score fields it emits.
    scorer: str
    score_fields: tuple[str, ...]
    #: The receiver prompt builder.
    prompt_builder: str
    #: How worker prompts relate to ``worker_prompt_tokens``: ``"exact"`` renders
    #: exactly that many tokens, ``"natural"`` renders real content only, at most that
    #: many. The field enters the scientific identity only when it is not ``"exact"``.
    prompt_geometry: str = "exact"
    #: The question the selector's judger carries, when an ablation replaces it.
    #: ``None`` is the registered rule, the item's own question: the worker prompt
    #: and the final receiver prompt always keep that one. The field enters the
    #: scientific identity only when set, so no sealed profile moves.
    judger_question: str | None = None

    def __post_init__(self) -> None:
        """Raise for a prompt geometry or a judger question this profile refuses."""
        if self.prompt_geometry not in {"exact", "natural"}:
            raise ValueError(f"unregistered prompt geometry {self.prompt_geometry!r}")
        if self.judger_question is not None and not self.judger_question.strip():
            raise ValueError("judger_question must carry text when it is set")

    @property
    def exact_prompts(self) -> bool:
        """Return whether every worker prompt must be exactly ``worker_prompt_tokens``."""
        return self.prompt_geometry == "exact"

    def admits_worker_prompt_tokens(self, counts: Sequence[int]) -> bool:
        """Return whether one item's rendered worker prompt widths fit this profile."""
        values = tuple(counts)
        if len(values) != self.workers_per_item or any(
            type(value) is not int or value <= 0 for value in values
        ):
            return False
        if self.exact_prompts:
            return values == (self.worker_prompt_tokens,) * self.workers_per_item
        return all(value <= self.worker_prompt_tokens for value in values)

    def rolled_rows(self, prompt_tokens: int) -> int:
        """Return the producer roll length one worker of this width yields."""
        return int(prompt_tokens) + self.latent_steps

    @property
    def scientific_identity_hash(self) -> str:
        """Return a stable sha256 over this profile's identity dict."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def sealed_qwen_policy(self, arm_id: str) -> str:
        """Return the Qwen sampling policy bound to one semantic arm."""
        try:
            return dict(self.sealed_qwen_policies)[arm_id]
        except KeyError as exc:
            raise ValueError(f"unregistered FanOutQA arm {arm_id!r}") from exc

    def answer_seeds(self, qid: str, arm_id: str) -> tuple[int, ...]:
        """Return one item and arm's receiver seeds, one per sample tag."""
        if qid not in self.question_ids:
            raise ValueError(f"unregistered FanOutQA question {qid!r}")
        policy = self.sealed_qwen_policy(arm_id)
        return tuple(
            int(
                hashlib.sha256(
                    "|".join(
                        (self.answer_seed_namespace, qid, policy, "answer", sample_tag)
                    ).encode()
                ).hexdigest()[:8],
                16,
            )
            for sample_tag in self.sample_tags
        )

    def report_seeds(self, qid: str, sample_tag: str) -> tuple[int, ...]:
        """Return one item and sample tag's report seeds, one per worker."""
        if qid not in self.question_ids:
            raise ValueError(f"unregistered FanOutQA question {qid!r}")
        if sample_tag not in self.sample_tags:
            raise ValueError(f"unregistered FanOutQA sample tag {sample_tag!r}")
        return tuple(
            zlib.crc32(
                "|".join(
                    (str(self.report_seed_base), qid, "report", str(worker), sample_tag)
                ).encode()
            )
            & 0x7FFFFFFF
            for worker in range(self.workers_per_item)
        )

    def to_dict(self) -> dict[str, object]:
        """Return the full registration dict, fingerprints included."""
        return {
            "profile_id": self.profile_id,
            "source_commit": self.source_commit,
            "source_logical_fingerprint": self.source_logical_fingerprint,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "question_index_sha256": self.question_index_sha256,
            "prepared_artifact_sha256": self.prepared_artifact_sha256,
            "prepared_config_fingerprint": self.prepared_config_fingerprint,
            "panel_registration_sha256": self.panel_registration_sha256,
            "source_audit_fingerprint": self.source_audit_fingerprint,
            "question_ids": list(self.question_ids),
            "workers_per_item": self.workers_per_item,
            "worker_prompt_tokens": self.worker_prompt_tokens,
            "latent_steps": self.latent_steps,
            "span_width": self.span_width,
            "ratios": list(self.ratios),
            "selectors": list(self.selectors),
            "arms": [arm.to_dict() for arm in self.arms],
            "sealed_qwen_policies": dict(self.sealed_qwen_policies),
            "sample_tags": list(self.sample_tags),
            "answer_seed_namespace": self.answer_seed_namespace,
            "answer_seed_rule": ("sha256(namespace|qid|sealed_qwen_policy|answer|sample_tag)[:8]"),
            "report_seed_base": self.report_seed_base,
            "report_seed_rule": "crc32(base|qid|report|worker|sample_tag)",
            "answer_ceiling": self.answer_ceiling,
            "report_ceiling": self.report_ceiling,
            "max_model_len": self.max_model_len,
            **(
                {}
                if self.report_presence_penalty == 0.0
                else {"report_presence_penalty": self.report_presence_penalty}
            ),
            **({} if self.exact_prompts else {"prompt_geometry": self.prompt_geometry}),
            **({} if self.judger_question is None else {"judger_question": self.judger_question}),
        }
