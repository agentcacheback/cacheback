"""Qwen result rows and their independent raw-token rescore.

One row builder serves every benchmark on this lane; what a profile cannot say
lives in a row rule registered under its scorer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from rcc.benchmarks.fanoutqa.scoring import (
    DEFAULT_SCORER_VERSION,
    score_text,
    scorer_for_rows,
)
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.longbench_v2.geometry import (
    chain_wrapper_allowances,
    rewrite_prompt_ceiling,
)
from rcc.benchmarks.longbench_v2.scoring import EXTRACTION_RULE, extract_choice, score_choice
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.chain_text import validate_notes_chain_fields
from rcc.models.qwen.receiver import QwenReceiverPrompt, QwenVisibleAnswer
from rcc.models.qwen.text import validate_report_ladder_fields
from rcc.models.route import RouteFamily
from rcc.run.contract import PreparedItem
from rcc.run.fleet.answer_closer import (
    ANSWER_CLOSING_TOKEN_BUDGET,
    FINISH_REASONS,
    validate_banked_answer,
)
from rcc.run.fleet.latency import ItemLatency


def _physical_policy(semantic_arm: str, *, family: RouteFamily) -> str:
    try:
        return family.policy(semantic_arm)
    except ValueError:
        raise ValueError(f"unregistered Qwen semantic arm {semantic_arm!r}") from None


def _arm_channel(semantic_arm: str, *, profile: BenchmarkProfile) -> str:
    """Return one arm's registered channel, or refuse the arm by name."""
    for arm in profile.arms:
        if arm.arm_id == semantic_arm:
            return arm.channel
    raise ValueError(f"{profile.benchmark_key} registers no arm {semantic_arm!r}")


#: The key the FanOutQA proxy scorer's row rule registers under. A rule scores
#: exactly its own fields, so an unregistered scorer is refused by name here
#: rather than by a missing key inside the mean over the score fields.
_FANOUTQA_SCORER = "fanoutqa-string-proxy-loose-strict-v1"


class RowRule(Protocol):
    """One benchmark's scoring, worker-prompt counts, and extra row fields."""

    #: The one field a strata table and an interval are read on. It belongs to
    #: the rule, not to the profile's column order.
    headline_field: str

    def score(
        self, item: PreparedItem, visible: str, *, profile: BenchmarkProfile
    ) -> dict[str, float]:
        """Return one visible answer's scores, or refuse the item by what registered it."""
        ...

    def expected_worker_prompt_tokens(
        self,
        item: PreparedItem,
        prompt_counts: Sequence[int],
        *,
        semantic_arm: str,
        profile: BenchmarkProfile,
        family: RouteFamily,
        channel_fields: Mapping[str, object] | None,
    ) -> None:
        """Refuse worker-prompt counts this benchmark's geometry denies."""

    def validate_report_fields(self, row: Mapping[str, Any]) -> None:
        """Refuse a banked sender bundle this benchmark's ladder denies."""

    def row_fields(
        self, item: PreparedItem, visible: Sequence[str], *, profile: BenchmarkProfile
    ) -> dict[str, object]:
        """Return the result fields only this benchmark's row carries."""
        ...


class _FanoutqaRowRule:
    """The FanOutQA string proxy scored over one question's gold leaves."""

    headline_field = "loose"

    def score(
        self, item: PreparedItem, visible: str, *, profile: BenchmarkProfile
    ) -> dict[str, float]:
        """Score one visible answer against the question's gold leaves.

        A chain item reaching this rule means the profile's scorer and its
        prepared bytes disagree, so the refusal names both registered values.
        """
        if isinstance(item, ChainItem):
            raise ValueError(
                f"{profile.benchmark_key} registers the {profile.scorer} scorer, which "
                f"reads a FanOutQA question; {item.qid} arrived as a chain item"
            )
        return score_text(item.question_obj, visible)

    def expected_worker_prompt_tokens(
        self,
        item: PreparedItem,
        prompt_counts: Sequence[int],
        *,
        semantic_arm: str,
        profile: BenchmarkProfile,
        family: RouteFamily,
        channel_fields: Mapping[str, object] | None,
    ) -> None:
        """Refuse counts off the registered sender or the registered geometry.

        A padded geometry admits exactly the registered width per worker and a
        natural one any width up to it; a native sender, its banked counts.
        """
        del channel_fields
        if family.native_sender_prompts:
            from rcc.models.qwen.native_prompts import native_prompt_counts

            admitted = tuple(prompt_counts) == native_prompt_counts(
                _probe_item(item, profile=profile),
                arm=semantic_arm,
                family=family,
                profile=profile,
            )
        else:
            admitted = profile.admits_worker_prompt_tokens(prompt_counts)
        if not admitted:
            raise ValueError(
                f"result worker prompt counts differ from the sealed sender or the registered "
                f"{profile.workers_per_item} x {profile.prompt_geometry} "
                f"{profile.worker_prompt_tokens}"
            )

    def validate_report_fields(self, row: Mapping[str, Any]) -> None:
        """Apply the three-worker report ladder rule."""
        validate_report_ladder_fields(row)

    def row_fields(
        self, item: PreparedItem, visible: Sequence[str], *, profile: BenchmarkProfile
    ) -> dict[str, object]:
        """Return nothing: the FanOutQA row is the shared row."""
        return {}


def _probe_item(item: ProbeItem | ChainItem, *, profile: BenchmarkProfile) -> ProbeItem:
    """Refuse, by the values that registered it, an item the fan-out path cannot read.

    A native sender roster is registered on the fan-out topology alone; without
    this guard the failure is an attribute error inside a prompt-count helper.
    """
    if isinstance(item, ChainItem):
        raise ValueError(
            f"{profile.benchmark_key} registers the {profile.scorer} scorer over a native "
            f"sender roster, which reads a FanOutQA question; {item.qid} is a chain item"
        )
    return item


def _chain_item(item: PreparedItem, *, profile: BenchmarkProfile) -> ChainItem:
    """Refuse an item this scorer cannot read, by the values that registered it.

    The refusal names the benchmark and the scorer, which is what has to change.
    """
    if not isinstance(item, ChainItem):
        raise ValueError(
            f"{profile.benchmark_key} registers the {profile.scorer} scorer, which "
            f"reads a chain item; this prepared item is not one"
        )
    return item


#: The chain channels whose worker count is a hop prompt. ``hop_prompt_ids``
#: splices the registered chunk ids between a prefix and a suffix, so the chunk
#: count is an exact floor and the hop wrapper is the whole slack above it.
_CHAIN_HOP_PROMPT_CHANNELS = ("floor", "latent")

#: The per-hop count of the part as the text seat re-encoded it, banked beside
#: that seat's count in ``QwenNotesChainBundle.result_fields``.
_CHAIN_PART_TOKENS = "chunk_text_tokens"


def _rewritten_part_tokens(
    item: ChainItem,
    channel_fields: Mapping[str, object] | None,
    *,
    semantic_arm: str,
    profile: BenchmarkProfile,
) -> tuple[int, ...]:
    """Return the part each hop re-encoded, or refuse the seat that banked none.

    A rewrite prompt carries the part as decoded text, so the registered count
    is not a floor for it; the seat's own per-hop measurement is.
    """
    banked = (channel_fields or {}).get(_CHAIN_PART_TOKENS)
    counts = (
        tuple(int(count) for count in cast(list[Any], banked)) if isinstance(banked, list) else ()
    )
    if len(counts) != len(item.chunk_tokens):
        raise ValueError(
            f"{profile.benchmark_key} prices the {semantic_arm} rewrite prompt from the part "
            f"each hop re-encoded, so its seat banks one {_CHAIN_PART_TOKENS} per hop; "
            f"{item.qid} carries {banked!r}"
        )
    return counts


def _require_ledger_vocabulary(semantic_arm: str, *, family: RouteFamily) -> None:
    """Refuse a sender whose tokenizer is not one the lane pins beside the ledger.

    The rewrite band adds a ledger count and a sender budget together, so both
    must be in one vocabulary; the runtime profile pins the tokenizers.
    """
    registered = {arm.semantic_arm: arm for arm in family.profile.physical_arms}
    arm = registered.get(semantic_arm)
    pinned = dict(family.profile.runtime.tokenizer_revisions)
    tokenizer = arm.sender_tokenizer if arm is not None else None
    revision = arm.sender_tokenizer_revision if arm is not None else None
    if tokenizer is None or pinned.get(tokenizer) != revision:
        raise ValueError(
            f"{family.lane} prices the {semantic_arm} rewrite prompt against the ledger's "
            f"vocabulary, so its sender tokenizer is one the lane pins; {semantic_arm} "
            f"registers {tokenizer!r}"
        )


class _ChainRowRule:
    """The official LongBench v2 extraction over the four-hop chain."""

    headline_field = "correct"

    def score(
        self, item: PreparedItem, visible: str, *, profile: BenchmarkProfile
    ) -> dict[str, float]:
        """Score one visible answer under the official extraction rule."""
        return score_choice(_chain_item(item, profile=profile).gold, visible)

    def expected_worker_prompt_tokens(
        self,
        item: PreparedItem,
        prompt_counts: Sequence[int],
        *,
        semantic_arm: str,
        profile: BenchmarkProfile,
        family: RouteFamily,
        channel_fields: Mapping[str, object] | None,
    ) -> None:
        """Refuse counts outside the band this arm's own prompt is priced in.

        Both bands are one count per hop: a hop prompt is floored by its chunk
        count, a text arm is capped by ``rewrite_prompt_ceiling``.
        """
        chain = _chain_item(item, profile=profile)
        counts = tuple(prompt_counts)
        hop_wrapper, wrapper = chain_wrapper_allowances(profile)
        if len(counts) != len(chain.chunk_tokens):
            raise ValueError(
                f"{profile.benchmark_key} {semantic_arm} result worker prompt counts "
                f"{list(counts)} are not one count per hop over {list(chain.chunk_tokens)}"
            )
        if _arm_channel(semantic_arm, profile=profile) in _CHAIN_HOP_PROMPT_CHANNELS:
            floors = chain.chunk_tokens
            ceilings = tuple(chunk + hop_wrapper for chunk in chain.chunk_tokens)
        else:
            _require_ledger_vocabulary(semantic_arm, family=family)
            floors = _rewritten_part_tokens(
                chain, channel_fields, semantic_arm=semantic_arm, profile=profile
            )
            ceilings = tuple(
                rewrite_prompt_ceiling(
                    chunk, report_ceiling=profile.report_ceiling, wrapper=wrapper
                )
                for chunk in chain.chunk_tokens
            )
        if any(
            count < floor or count > ceiling
            for floor, ceiling, count in zip(floors, ceilings, counts, strict=True)
        ):
            raise ValueError(
                f"{profile.benchmark_key} {semantic_arm} result worker prompt counts "
                f"{list(counts)} are not one count per hop from {list(floors)} to {list(ceilings)}"
            )

    def validate_report_fields(self, row: Mapping[str, Any]) -> None:
        """Apply the four-hop notes ladder rule."""
        validate_notes_chain_fields(row)

    def row_fields(
        self, item: PreparedItem, visible: Sequence[str], *, profile: BenchmarkProfile
    ) -> dict[str, object]:
        """Return the gold letter, the item's strata, and every extracted choice."""
        chain = _chain_item(item, profile=profile)
        return {
            "gold": chain.gold,
            "stratum": chain.stratum,
            "domain": chain.domain,
            "difficulty": chain.difficulty,
            "extracted_choice_by_sample": [extract_choice(text) for text in visible],
        }


_ROW_RULES: dict[str, RowRule] = {
    _FANOUTQA_SCORER: _FanoutqaRowRule(),
    EXTRACTION_RULE: _ChainRowRule(),
}


def headline_score_field(profile: BenchmarkProfile) -> str:
    """Return the one score this benchmark's strata and intervals are read on."""
    return _require_registered_scorer(profile).headline_field


def _require_registered_scorer(profile: BenchmarkProfile) -> RowRule:
    """Return this benchmark's row rule, or refuse its scorer by name."""
    try:
        return _ROW_RULES[profile.scorer]
    except KeyError:
        raise ValueError(
            f"Qwen result rows implement the {', '.join(_ROW_RULES)} scorers; "
            f"{profile.benchmark_key} registers {profile.scorer}"
        ) from None


def _mean(fields: Sequence[Mapping[str, float]], name: str) -> float:
    return round(sum(float(row[name]) for row in fields) / len(fields), 4)


def build_result_row(
    item: ProbeItem | ChainItem,
    semantic_arm: str,
    answers: Sequence[QwenVisibleAnswer],
    prompt: QwenReceiverPrompt,
    *,
    worker_prompt_tokens: Sequence[int],
    latency: ItemLatency,
    family: RouteFamily,
    channel_fields: Mapping[str, object] | None = None,
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    """Bank three independent samples, mean within item/arm, and never vote."""
    profile = family.benchmark_profile(profile)
    rule = _require_registered_scorer(profile)
    roster = tuple(answers)
    if tuple(answer.tag for answer in roster) != profile.sample_tags:
        raise ValueError("Qwen result answers differ from the registered sample tag order")
    expected_seeds = profile.answer_seeds(item.qid, semantic_arm)
    if tuple(answer.seed for answer in roster) != expected_seeds:
        raise ValueError("Qwen result answer seeds differ from the sealed item-arm grid")
    rule.expected_worker_prompt_tokens(
        item,
        worker_prompt_tokens,
        semantic_arm=semantic_arm,
        profile=profile,
        family=family,
        channel_fields=channel_fields,
    )
    if len(latency.queued_offsets) != len(roster):
        raise ValueError("Qwen result latency offsets differ from the three-sample roster")
    visible_texts = [answer.visible_text for answer in roster]
    sample_fields = tuple(rule.score(item, text, profile=profile) for text in visible_texts)
    thinking_closed = tuple(answer.thinking_closed for answer in roster)
    finish_reasons = tuple(answer.finish_reason for answer in roster)
    generated_tokens = tuple(len(answer.token_ids) for answer in roster)
    decode = family.profile.decode
    policy = _physical_policy(semantic_arm, family=family)
    row: dict[str, Any] = {
        "kind": "result",
        **(
            {"scoring_version": DEFAULT_SCORER_VERSION}
            if profile.scorer == _FANOUTQA_SCORER
            else {}
        ),
        "family": family.model_id,
        "qid": item.qid,
        "question": item.question,
        "semantic_arm": semantic_arm,
        "policy": policy,
        "arm": policy,
        "channel": _arm_channel(semantic_arm, profile=profile),
        "decode_profile": decode.profile_id,
        "decode_fingerprint": decode.identity_hash,
        "decode": {
            **decode.to_dict(),
            "max_tokens": profile.answer_ceiling,
            "stop_token_ids": list(family.stop_token_ids),
        },
        "decode_tags": list(profile.sample_tags),
        "sample_seeds": list(expected_seeds),
        "answers": [answer.raw_text for answer in roster],
        "backend_answers": [answer.backend_text for answer in roster],
        "answer": roster[0].raw_text,
        "answer_texts": list(visible_texts),
        "sample_task_fields": [dict(fields) for fields in sample_fields],
        **{field: _mean(sample_fields, field) for field in profile.score_fields},
        "thinking_closed_by_sample": list(thinking_closed),
        "thinking_closed_rate": round(sum(thinking_closed) / len(roster), 4),
        "answer_injected_by_sample": [answer.answer_injected for answer in roster],
        # The banked reason is the head's, so these two vectors are equal by
        # construction and named apart: the rescore reads the head's reason
        # where it cannot re-derive one.
        "answer_head_finish_reasons": list(finish_reasons),
        "answer_closer_budget": ANSWER_CLOSING_TOKEN_BUDGET,
        "finish_reasons": list(finish_reasons),
        "finish_reason": finish_reasons[0],
        "length_finish_rate": round(
            sum(reason == "length" for reason in finish_reasons) / len(roster), 4
        ),
        "generated_token_ids_by_sample": [list(answer.token_ids) for answer in roster],
        "visible_decode_token_ids_by_sample": [
            list(answer.visible_decode_token_ids) for answer in roster
        ],
        "generated_tokens_by_sample": list(generated_tokens),
        "generated_tokens": round(sum(generated_tokens) / len(roster), 4),
        "num_cached_tokens_by_sample": [answer.num_cached_tokens for answer in roster],
        "num_cached_tokens": roster[0].num_cached_tokens,
        "first_token_emitted_by_sample": [bool(count) for count in generated_tokens],
        "worker_prompt_tokens": list(worker_prompt_tokens),
        "aggregate_worker_prompt_tokens": sum(worker_prompt_tokens),
        "receiver_prompt_tokens": prompt.prompt_rows,
        "base_prompt_tokens": prompt.manager_tokens,
        "latent_tokens": prompt.payload_rows,
        "payload_layout": prompt.payload_layout,
        "payload_semantic_arm": prompt.payload_semantic_arm,
        "payload_plan_sha256": prompt.payload_plan_sha256,
        "payload_tensor_sha256": prompt.payload_tensor_sha256,
        **rule.row_fields(item, visible_texts, profile=profile),
        **latency.result_fields(),
        "sample_aggregation": "mean_within_item_arm_then_macro_over_items_no_vote",
    }
    if channel_fields:
        overlap = set(row).intersection(channel_fields)
        if overlap:
            raise ValueError(f"Qwen channel fields overwrite signed result keys: {sorted(overlap)}")
        row.update(channel_fields)
    return row


def _visible_from_token_ids(
    tokenizer: Any, token_ids: object, *, family: RouteFamily
) -> tuple[str, str, list[int], list[int]]:
    if not isinstance(token_ids, list) or any(
        type(token) is not int or token < 0 for token in cast(list[object], token_ids)
    ):
        raise RuntimeError("Qwen result has malformed generated token ids")
    ids = cast(list[int], token_ids)
    stop = next(
        (index for index, token in enumerate(ids) if token in family.stop_token_ids),
        len(ids),
    )
    visible_ids = ids[:stop]
    raw = str(
        tokenizer.decode(visible_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    )
    return raw, family.post_think_report(raw), visible_ids, ids


def _bool_vector(value: object, label: str, name: str) -> list[bool]:
    if not isinstance(value, list) or len(cast(list[object], value)) != 3:
        raise RuntimeError(f"{label}: Qwen {name} evidence is incomplete")
    values = cast(list[object], value)
    if any(type(flag) is not bool for flag in values):
        raise RuntimeError(f"{label}: Qwen {name} evidence is malformed")
    return cast(list[bool], values)


def _answer_finish_reasons(
    row: Mapping[str, Any],
    generated_ids: Sequence[Sequence[int]],
    *,
    family: RouteFamily,
    label: str,
    ceiling: int,
) -> tuple[list[str], list[bool]]:
    """Derive each sample's head finish reason from its banked ids.

    A natural answer records its own termination, a stop id or the sampler cap.
    An injected answer dropped its head stop id, so its reason is read banked.
    """
    injected = _bool_vector(row.get("answer_injected_by_sample"), label, "injection")
    banked = row.get("answer_head_finish_reasons")
    if not isinstance(banked, list) or len(cast(list[object], banked)) != 3:
        raise RuntimeError(f"{label}: Qwen answer head termination evidence is incomplete")
    head_reasons = cast(list[object], banked)
    budget = row.get("answer_closer_budget")
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise RuntimeError(f"{label}: Qwen answer closing budget is malformed")
    reasons: list[str] = []
    for index, ids in enumerate(generated_ids):
        if not injected[index]:
            stopped = any(token in family.stop_token_ids for token in ids)
            reason = "stop" if stopped else "length" if len(ids) == ceiling else "invalid"
        elif head_reasons[index] not in FINISH_REASONS:
            raise RuntimeError(f"{label}: Qwen injected answer head reason is unregistered")
        else:
            reason = str(head_reasons[index])
        if reason != "invalid":
            validate_banked_answer(
                ids,
                injected=injected[index],
                finish_reason=reason,
                closing_budget=budget,
                answer_ceiling=ceiling,
                label=f"{label}/s{index}",
                closer_ids=family.think_close_token_ids,
            )
        reasons.append(reason)
    return reasons, injected


def validate_and_rescore_result(
    row: Mapping[str, Any],
    item: ProbeItem | ChainItem,
    tokenizer: Any,
    *,
    family: RouteFamily,
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    """Reconstruct every answer from raw ids and independently recompute scores."""
    profile = family.benchmark_profile(profile)
    rule = _require_registered_scorer(profile)
    semantic_arm = row.get("semantic_arm")
    if not isinstance(semantic_arm, str):
        raise RuntimeError(f"{item.qid}: Qwen result is missing its semantic arm")
    decode = family.profile.decode
    expected_seeds = list(profile.answer_seeds(item.qid, semantic_arm))
    expected_policy = _physical_policy(semantic_arm, family=family)
    expected_channel = _arm_channel(semantic_arm, profile=profile)
    if (
        row.get("qid") != item.qid
        or row.get("question") != item.question
        or row.get("family") != family.model_id
        or row.get("policy") != expected_policy
        or row.get("arm") != expected_policy
        or row.get("channel") != expected_channel
        or row.get("sample_seeds") != expected_seeds
        or row.get("decode_tags") != list(profile.sample_tags)
        or row.get("decode_profile") != decode.profile_id
        or row.get("decode_fingerprint") != decode.identity_hash
        or row.get("sample_aggregation") != "mean_within_item_arm_then_macro_over_items_no_vote"
    ):
        raise RuntimeError(f"{item.qid}/{semantic_arm}: Qwen result identity differs")
    if family.native_sender_prompts:
        from rcc.models.qwen.native_results import validate_native_result

        validate_native_result(
            item,
            row,
            arm=semantic_arm,
            family=family,
            profile=profile,
        )
    if expected_channel == "text":
        try:
            rule.validate_report_fields(row)
        except ValueError as exc:
            raise RuntimeError(f"{item.qid}/{semantic_arm}: {exc}") from exc
    raw_token_rows = row.get("generated_token_ids_by_sample")
    if not isinstance(raw_token_rows, list) or len(cast(list[object], raw_token_rows)) != 3:
        raise RuntimeError(f"{item.qid}/{semantic_arm}: Qwen result sample roster differs")
    token_rows = cast(list[object], raw_token_rows)
    reconstructed = tuple(
        _visible_from_token_ids(tokenizer, ids, family=family) for ids in token_rows
    )
    raw = [text for text, _visible, _visible_ids, _ids in reconstructed]
    visible = [text for _raw, text, _visible_ids, _ids in reconstructed]
    visible_ids = [ids for _raw, _visible, ids, _ids in reconstructed]
    generated_ids = [ids for _raw, _visible, _visible_ids, ids in reconstructed]
    closed = [family.think_close in text for text in raw]
    finish_reasons, injected = _answer_finish_reasons(
        row,
        generated_ids,
        family=family,
        label=f"{item.qid}/{semantic_arm}",
        ceiling=profile.answer_ceiling,
    )
    if "invalid" in finish_reasons:
        raise RuntimeError(f"{item.qid}/{semantic_arm}: Qwen raw ids do not witness a finish")
    generated = [len(ids) for ids in generated_ids]
    if profile.scorer == _FANOUTQA_SCORER:
        scorer = scorer_for_rows((row,))
        question = _probe_item(item, profile=profile).question_obj
        scores = [scorer(question, text) for text in visible]
    else:
        scores = [rule.score(item, text, profile=profile) for text in visible]
    expected = {
        "answers": raw,
        "answer": raw[0],
        "answer_texts": visible,
        "visible_decode_token_ids_by_sample": visible_ids,
        "thinking_closed_by_sample": closed,
        "thinking_closed_rate": round(sum(closed) / len(closed), 4),
        "answer_injected_by_sample": injected,
        "answer_head_finish_reasons": finish_reasons,
        "answer_closer_budget": ANSWER_CLOSING_TOKEN_BUDGET,
        "finish_reasons": finish_reasons,
        "finish_reason": finish_reasons[0],
        "length_finish_rate": round(
            sum(reason == "length" for reason in finish_reasons) / len(finish_reasons), 4
        ),
        "generated_tokens_by_sample": generated,
        "generated_tokens": round(sum(generated) / len(generated), 4),
        "first_token_emitted_by_sample": [bool(count) for count in generated],
        "sample_task_fields": scores,
        **{field: _mean(scores, field) for field in profile.score_fields},
        **rule.row_fields(item, visible, profile=profile),
    }
    if any(row.get(field) != value for field, value in expected.items()):
        raise RuntimeError(f"{item.qid}/{semantic_arm}: Qwen raw-token rescore differs")
    return {**dict(row), **expected}


__all__ = (
    "RowRule",
    "build_result_row",
    "headline_score_field",
    "validate_and_rescore_result",
)
