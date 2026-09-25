"""Route-native Qwen decode requests and engine protocol.

One decode spec renders a family's sampling contract, and a request binds one
such decode to an item, an arm, a sample tag, and its registered seed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Literal, Protocol, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.text_output import TextCompletion
from rcc.models.route import RouteFamily
from rcc.topologies.fanout import FANOUT_M3

#: The route's report redraw ladder, one draw per tag in order.
QWEN_REPORT_SEED_TAGS = ("s0", "s1", "s2")
QWEN_TEXT_ARMS = tuple(arm.arm_id for arm in FANOUTQA_NATURAL_DEV50.arms if arm.channel == "text")
QwenDecodePurpose = Literal["answer", "report"]

if len(QWEN_REPORT_SEED_TAGS) != FANOUT_M3.workers_per_item:
    raise RuntimeError("Qwen sample tags and canonical M=3 topology differ")


@dataclass(frozen=True)
class QwenDecodeSpec:
    """Backend-neutral view of one registered family's sampling contract."""

    purpose: QwenDecodePurpose
    family: RouteFamily
    closing: bool = False
    _: KW_ONLY
    #: The benchmark whose registered output ceilings this request decodes
    #: under. The FanOutQA lanes read the default; every other benchmark names
    #: its own at the call site.
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50

    def __post_init__(self) -> None:
        """Refuse invalid purposes and a closing continuation off the report side."""
        if self.purpose not in {"answer", "report"}:
            raise ValueError("Qwen decode purpose must be answer or report")
        if self.closing and self.purpose != "report":
            raise ValueError("Qwen closer injection continues a report draw only")

    @property
    def max_tokens(self) -> int:
        """Return the registered output ceiling for this request purpose.

        A closing continuation is the one exception: it decodes under the
        family's registered closing budget, never a second full report ceiling.
        """
        if self.closing:
            return self.family.closing_token_budget
        profile = self.family.benchmark_profile(self.profile)
        return profile.answer_ceiling if self.purpose == "answer" else profile.report_ceiling

    @property
    def enable_thinking(self) -> bool:
        """Return the family-native thinking switch."""
        return self.family.profile.decode.enable_thinking

    @property
    def presence_penalty(self) -> float:
        """Return the sampling penalty this request purpose decodes under.

        An answer keeps the family protocol's value, so the receiver draw is
        the same on every benchmark; a report reads the benchmark's own value.
        """
        if self.purpose == "answer":
            return self.family.profile.decode.presence_penalty
        return self.family.benchmark_profile(self.profile).report_presence_penalty

    def backend_sampling(self) -> dict[str, object]:
        """Render complete backend fields without exposing caller overrides."""
        return {
            **self.family.profile.decode.backend_sampling(),
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
            "stop_token_ids": list(self.family.stop_token_ids),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the model, benchmark, and output-ceiling identities together."""
        decode = self.family.profile.decode
        return {
            "purpose": self.purpose,
            "decode_profile": decode.profile_id,
            "decode_fingerprint": decode.identity_hash,
            "enable_thinking": self.enable_thinking,
            **self.backend_sampling(),
        }


@dataclass(frozen=True)
class QwenDecodeRequest:
    """One request whose seed is derived from the registered panel identity."""

    request_id: str
    qid: str
    semantic_arm: str
    sample_tag: str
    member_index: int
    seed: int
    decode: QwenDecodeSpec
    profile: BenchmarkProfile

    def __post_init__(self) -> None:
        """Reject caller-minted seeds, tags, questions, or physical arm aliases."""
        if not self.request_id:
            raise ValueError("Qwen decode request id must be nonempty")
        # The seeds come from this request's benchmark and the output ceiling
        # from the decode spec's, so the two must name the same one.
        if self.decode.profile != self.profile:
            raise ValueError("Qwen decode request and its decode spec name different benchmarks")
        if self.sample_tag not in QWEN_REPORT_SEED_TAGS:
            raise ValueError("Qwen decode request has an unregistered sample tag")
        if self.decode.purpose == "report":
            if self.semantic_arm not in QWEN_TEXT_ARMS:
                raise ValueError("Qwen report request has an unregistered text arm")
            expected = self.profile.report_seeds(self.qid, self.sample_tag)
        else:
            if self.semantic_arm not in {arm.arm_id for arm in self.profile.arms}:
                raise ValueError("Qwen answer request has an unregistered semantic arm")
            expected = self.profile.answer_seeds(self.qid, self.semantic_arm)
        if (
            self.member_index not in range(len(expected))
            or self.seed != expected[self.member_index]
        ):
            raise ValueError("Qwen decode request seed differs from sealed benchmark identity")


class TextEngine(Protocol):
    """Engine seam used by all three Qwen text sender sizes."""

    def decode_text_full(
        self,
        prompts: Sequence[str],
        requests: Sequence[QwenDecodeRequest],
    ) -> Sequence[TextCompletion]:
        """Decode one independent report per signed backend-neutral request."""
        ...

    def decode_token_ids_full(
        self, prompts: Sequence[Sequence[int]], requests: Sequence[QwenDecodeRequest]
    ) -> Sequence[TextCompletion]:
        """Decode caller-tokenized prompts without changing their token identity."""
        ...


def validate_report_ladder_fields(fields: Mapping[str, object]) -> None:
    """Refuse a report bundle whose seed ladder and closure roster disagree."""
    draws = fields.get("draws_n")
    redraw = fields.get("redraw_wall_s")
    generation = fields.get("report_generation_s")
    seed_tag = fields.get("report_seed_tag")
    injected_by = fields.get("report_injected_by_worker")
    closed = fields.get("report_thinking_closed")
    if type(draws) is not int or not 1 <= draws <= len(QWEN_REPORT_SEED_TAGS):
        raise ValueError("Qwen report draw count differs from the registered ladder")
    if seed_tag not in QWEN_REPORT_SEED_TAGS or draws != QWEN_REPORT_SEED_TAGS.index(seed_tag) + 1:
        raise ValueError("Qwen report draw count differs from its registered seed tag")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, (int, float))
        or not math.isfinite(generation)
        or generation < 0.0
    ):
        raise ValueError("Qwen report generation wall clock is invalid")
    if (
        isinstance(redraw, bool)
        or not isinstance(redraw, (int, float))
        or not math.isfinite(redraw)
        or redraw < 0.0
    ):
        raise ValueError("Qwen report redraw wall clock is invalid")
    if draws == 1 and redraw != 0.0:
        raise ValueError("Qwen single-draw report bundle cannot carry redraw wall clock")
    injected_values = cast(Sequence[object], injected_by)
    if (
        type(fields.get("injected")) is not bool
        or not isinstance(injected_by, (list, tuple))
        or len(injected_values) != FANOUT_M3.workers_per_item
        or any(type(value) is not bool for value in injected_values)
        or fields.get("injected") != any(injected_values)
    ):
        raise ValueError("Qwen report injection summary differs from its worker roster")
    closed_values = cast(Sequence[object], closed)
    if (
        not isinstance(closed, (list, tuple))
        or len(closed_values) != FANOUT_M3.workers_per_item
        or any(type(value) is not bool for value in closed_values)
    ):
        raise ValueError("Qwen report closure roster is malformed")
    if any(
        was_injected and not was_closed
        for was_injected, was_closed in zip(injected_values, closed_values, strict=True)
    ):
        raise ValueError("Qwen closer injection must leave its own worker closed")


@dataclass(frozen=True)
class QwenReportBundle:
    """Three accepted worker reports plus their exact raw-token evidence."""

    reports: tuple[str, ...]
    raw_outputs: tuple[str, ...]
    token_ids_by_worker: tuple[tuple[int, ...], ...]
    tokens_by_worker: tuple[int, ...]
    finish_reasons: tuple[str, ...]
    seeds: tuple[int, ...]
    seed_tag: str
    thinking_closed: tuple[bool, ...]
    decode: QwenDecodeSpec
    generation_s: float
    queue_s_mean: float | None
    ttft_s_mean: float | None
    report_failed_workers: tuple[int, ...] = ()
    draws_n: int = 1
    redraw_wall_s: float = 0.0
    injected_by_worker: tuple[bool, ...] = (False, False, False)
    prompt_token_sha256: tuple[str, ...] = ()
    prompt_tokens_by_worker: tuple[int, ...] = ()
    prompt_unpadded_tokens_by_worker: tuple[int, ...] = ()
    prompt_policy: str | None = None

    def __post_init__(self) -> None:
        """Validate one complete three-worker bundle, quarantined workers included."""
        fields = (
            self.reports,
            self.raw_outputs,
            self.token_ids_by_worker,
            self.tokens_by_worker,
            self.finish_reasons,
            self.seeds,
            self.thinking_closed,
            self.injected_by_worker,
        )
        if any(len(field) != 3 for field in fields):
            raise ValueError("Qwen report bundle must contain exactly three worker rows")
        validate_report_ladder_fields(
            {
                "draws_n": self.draws_n,
                "redraw_wall_s": self.redraw_wall_s,
                "report_generation_s": self.generation_s,
                "report_seed_tag": self.seed_tag,
                "injected": any(self.injected_by_worker),
                "report_injected_by_worker": self.injected_by_worker,
                "report_thinking_closed": self.thinking_closed,
            }
        )
        failed = self.report_failed_workers
        if list(failed) != sorted(set(failed)) or any(worker not in range(3) for worker in failed):
            raise ValueError("Qwen report bundle names unregistered quarantined workers")
        for worker, report in enumerate(self.reports):
            if worker in failed:
                if report.strip():
                    raise ValueError("Qwen quarantined workers must carry empty reports")
            elif not report.strip():
                raise ValueError("Qwen accepts only nonempty post-think reports")
        family = self.decode.family
        expected = tuple(
            family.post_think_handoff(raw, ended=reason == "stop")
            for raw, reason in zip(self.raw_outputs, self.finish_reasons, strict=True)
        )
        if expected != self.reports:
            raise ValueError("Qwen reports differ from their raw post-think reconstruction")
        expected_closed = tuple(
            not self.decode.enable_thinking or family.think_close in raw for raw in self.raw_outputs
        )
        if expected_closed != self.thinking_closed:
            raise ValueError("Qwen report closure fields differ from raw outputs")

    def result_fields(self) -> dict[str, object]:
        """Return the exact serializable report fields used by result rows."""
        fields: dict[str, object] = {
            "reports": list(self.reports),
            "report_raw_outputs": list(self.raw_outputs),
            "report_tokens": sum(self.tokens_by_worker),
            "report_tokens_by_worker": list(self.tokens_by_worker),
            "report_token_ids_by_worker": [list(ids) for ids in self.token_ids_by_worker],
            "report_generation_s": round(self.generation_s, 4),
            "draws_n": self.draws_n,
            "redraw_wall_s": round(self.redraw_wall_s, 4),
            "injected": any(self.injected_by_worker),
            "report_injected_by_worker": list(self.injected_by_worker),
            "report_seed_tag": self.seed_tag,
            "report_seeds": list(self.seeds),
            "report_thinking_closed": list(self.thinking_closed),
            "report_enable_thinking": self.decode.enable_thinking,
            "decode": self.decode.to_dict(),
            "report_finish_reasons": list(self.finish_reasons),
            "report_failed": bool(self.report_failed_workers),
            "report_failed_workers": list(self.report_failed_workers),
            "report_failure": (
                f"workers {list(self.report_failed_workers)} empty after "
                f"{len(QWEN_REPORT_SEED_TAGS)} registered draws"
                if self.report_failed_workers
                else None
            ),
            "report_queue_s_mean": self.queue_s_mean,
            "report_ttft_s_mean": self.ttft_s_mean,
        }
        if self.decode.family.native_sender_prompts:
            fields["report_decode"] = self.decode.to_dict()
            fields["report_prompt_token_sha256"] = list(self.prompt_token_sha256)
            fields["report_prompt_tokens_by_worker"] = list(self.prompt_tokens_by_worker)
            fields["report_prompt_unpadded_tokens_by_worker"] = list(
                self.prompt_unpadded_tokens_by_worker
            )
            fields["report_prompt_policy"] = self.prompt_policy
        return fields


__all__ = (
    "QWEN_REPORT_SEED_TAGS",
    "QWEN_TEXT_ARMS",
    "QwenDecodeRequest",
    "QwenDecodeSpec",
    "QwenReportBundle",
    "TextEngine",
    "validate_report_ladder_fields",
)
