"""Gemma visible-report generation, replay, and banked bundle rows."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.gemma import GEMMA
from rcc.models.gemma.text_bank import TextReportBankProtocol
from rcc.models.gemma.text_closer import continue_unclosed
from rcc.models.gemma.text_codec import (
    GemmaPromptRecord,
    prompt_record,
    report_is_substantive,
    visible_report,
)
from rcc.models.gemma.text_contract import (
    GEMMA_REPORT_SEED_TAGS,
    REPORT_BACKEND,
    REPORT_DECODE_PATH,
    TEXT_REGISTRATION_FINGERPRINT,
    TEXT_REPORT_SCHEMA,
    TextCompletion,
    TextEngine,
    TextSenderSpec,
    report_sampling,
    report_seed,
)
from rcc.topologies.fanout import FANOUT_M3

_FINISH_REASONS = frozenset({"stop", "length"})
_REPORT_FIELDS = (
    "text",
    "ids",
    "semantic_tokens",
    "decoded_tokens",
    "early_turn_end",
    "finish_reason",
    "generation_s",
    "draws_n",
    "redraw_wall_s",
    "injected",
    "decode_path",
    "raw_token_ids",
    "raw_text",
    "thinking_closed",
    "seed",
    "report_seed_tag",
    "report_queue_s_mean",
    "report_ttft_s_mean",
    "report_backend",
)


@dataclass(frozen=True)
class _Ladder:
    """The banked span of one bundle's shipped draw, and what it excludes.

    Every worker row of one bundle carries the same ladder, so a bundle rebuilt
    from the bank reads its span off the rows.
    """

    generation_s: float
    draws_n: int
    redraw_wall_s: float

    def __post_init__(self) -> None:
        """Refuse a ladder that no sequence of draws could have produced."""
        if self.draws_n < 1 or self.redraw_wall_s < 0.0 or self.generation_s < 0.0:
            raise RuntimeError("Gemma report ladder needs one draw and no negative clock")
        if self.draws_n == 1 and self.redraw_wall_s != 0.0:
            raise RuntimeError("Gemma single-draw report bundle cannot carry redraw wall clock")

    def row_fields(self) -> dict[str, Any]:
        """Return the ladder exactly as it is banked on every worker row."""
        return {
            "generation_s": round(self.generation_s, 4),
            "draws_n": self.draws_n,
            "redraw_wall_s": round(self.redraw_wall_s, 4),
        }


def _mean_elapsed(results: Sequence[TextCompletion], *, later: str, earlier: str) -> float | None:
    values: list[float] = []
    for result in results:
        later_value = getattr(result, later)
        earlier_value = getattr(result, earlier)
        if later_value is not None and earlier_value is not None:
            values.append(float(later_value) - float(earlier_value))
    return round(sum(values) / len(values), 4) if values else None


def _decode_draw(
    engine: TextEngine,
    tokenizer: Any,
    record: GemmaPromptRecord,
    sender: TextSenderSpec,
    *,
    tag: str,
    profile: BenchmarkProfile,
) -> tuple[list[dict[str, Any]], list[int], float]:
    seeds = profile.report_seeds(record.qid, tag)
    # `report_sampling` stays inside the timed region: resolving the suppressed
    # ids round-trips them through the tokenizer, so moving that cost out would
    # change what the measurement covers.
    started = time.perf_counter()
    sampling = report_sampling(tokenizer, sender)
    results = tuple(
        engine.decode_token_ids_full(
            record.prompt_ids,
            sampling,
            seeds=seeds,
        )
    )
    if len(results) != FANOUT_M3.workers_per_item:
        raise RuntimeError(f"{record.qid}: vLLM did not return M=3 reports")
    if any(result.num_cached_tokens not in (None, 0) for result in results):
        raise RuntimeError(f"{record.qid}: prefix caching contaminated Gemma reports")
    # The closer injection continues this same draw, so it is inside this
    # draw's clock.
    results, injected = continue_unclosed(
        engine,
        tokenizer,
        record.prompt_ids,
        sampling,
        results,
        seeds=seeds,
        qid=record.qid,
        sender=sender,
        profile=profile,
    )
    elapsed = time.perf_counter() - started
    queue_s = _mean_elapsed(results, later="scheduled_ts", earlier="queued_ts")
    ttft_s = _mean_elapsed(results, later="first_token_ts", earlier="queued_ts")
    rows: list[dict[str, Any]] = []
    invalid: list[int] = []
    for worker, (result, seed) in enumerate(zip(results, seeds, strict=True)):
        report = visible_report(tokenizer, result)
        row = {
            "injected": injected[worker],
            "text": report.text,
            "ids": list(report.visible_token_ids),
            "semantic_tokens": len(report.visible_token_ids),
            "decoded_tokens": len(report.raw_token_ids),
            "early_turn_end": report.finish_reason == "stop",
            "finish_reason": report.finish_reason,
            "decode_path": REPORT_DECODE_PATH,
            "raw_token_ids": list(report.raw_token_ids),
            "raw_text": report.raw_text,
            "thinking_closed": report.thinking_closed,
            "seed": seed,
            "report_seed_tag": tag,
            "report_backend": REPORT_BACKEND,
            "report_queue_s_mean": queue_s,
            "report_ttft_s_mean": ttft_s,
        }
        if not report.accepted:
            invalid.append(worker)
        rows.append(row)
    return rows, invalid, elapsed


def _validate_banked(
    record: GemmaPromptRecord,
    sender: TextSenderSpec,
    worker: int,
    row: Mapping[str, Any],
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    try:
        report = {name: row[name] for name in _REPORT_FIELDS}
    except KeyError as exc:
        raise RuntimeError(f"banked Gemma report omits {exc.args[0]}") from exc
    tag = str(report["report_seed_tag"])
    raw_value, ids_value = report["raw_token_ids"], report["ids"]
    generation_s = report["generation_s"]
    if not isinstance(raw_value, list) or not isinstance(ids_value, list):
        raise RuntimeError(f"{record.qid}/{sender.semantic_arm}/w{worker}: bank invalid")
    raw = cast(list[object], raw_value)
    ids = cast(list[object], ids_value)
    tokens_invalid = any(type(token) is not int or token < 0 for token in raw)
    ids_invalid = any(type(token) is not int or token < 0 for token in ids)
    if (
        tag not in GEMMA_REPORT_SEED_TAGS
        or report["seed"] != report_seed(record.qid, worker, tag, profile=profile)
        or row.get("semantic_arm") != sender.semantic_arm
        or row.get("worker_checkpoint") != sender.checkpoint
        or row.get("worker_revision") != sender.revision
        or report["report_backend"] != REPORT_BACKEND
        or report["decode_path"] != REPORT_DECODE_PATH
        or report["finish_reason"] not in _FINISH_REASONS
        or not isinstance(report["thinking_closed"], bool)
        or tokens_invalid
        or ids_invalid
        or report["semantic_tokens"] != len(ids)
        or report["decoded_tokens"] != len(raw)
        or isinstance(generation_s, bool)
        or not isinstance(generation_s, (int, float))
        or generation_s < 0
    ):
        raise RuntimeError(f"{record.qid}/{sender.semantic_arm}/w{worker}: bank invalid")
    # After the type guard above, never before it: the ladder check reads
    # `generation_s` as a number, so a null clock fails by name here rather than
    # as a bare TypeError inside the ladder.
    _validate_banked_ladder(record, sender, worker, report)
    return report


def _validate_banked_ladder(
    record: GemmaPromptRecord,
    sender: TextSenderSpec,
    worker: int,
    report: Mapping[str, Any],
) -> None:
    """Refuse a banked row whose ladder or injection evidence is unusable.

    A row that claims injection and is still unclosed cannot have come from this
    protocol.
    """
    draws_n, redraw_wall_s = report["draws_n"], report["redraw_wall_s"]
    injected = report["injected"]
    if (
        isinstance(draws_n, bool)
        or not isinstance(draws_n, int)
        or isinstance(redraw_wall_s, bool)
        or not isinstance(redraw_wall_s, (int, float))
        or not isinstance(injected, bool)
        or (injected and not report["thinking_closed"])
    ):
        raise RuntimeError(f"{record.qid}/{sender.semantic_arm}/w{worker}: bank invalid")
    _Ladder(float(report["generation_s"]), draws_n, float(redraw_wall_s))


def _banked_ladder(rows: Sequence[Mapping[str, Any]], mixed: str) -> _Ladder:
    """Return the one ladder every row of a banked batch must agree on."""
    ladders = {
        (float(row["generation_s"]), int(row["draws_n"]), float(row["redraw_wall_s"]))
        for row in rows
    }
    if len(ladders) != 1:
        raise RuntimeError(mixed)
    return _Ladder(*ladders.pop())


def _row_accepted(row: Mapping[str, Any]) -> bool:
    """Apply the registered acceptance predicate to one banked report row."""
    return bool(row["thinking_closed"]) and report_is_substantive(str(row["text"]))


def _bundle(
    sender: TextSenderSpec,
    record: GemmaPromptRecord,
    reports: Sequence[Mapping[str, Any]],
    ladder: _Ladder,
) -> dict[str, Any]:
    tags = {str(row["report_seed_tag"]) for row in reports}
    if len(reports) != FANOUT_M3.workers_per_item or len(tags) != 1:
        raise RuntimeError(f"{sender.semantic_arm}: report batch is incomplete or mixed")
    failed = [worker for worker, row in enumerate(reports) if not _row_accepted(row)]
    closed = [bool(row["thinking_closed"]) for row in reports]
    injected = [bool(row["injected"]) for row in reports]
    return {
        "schema": TEXT_REPORT_SCHEMA,
        "semantic_arm": sender.semantic_arm,
        "policy": sender.policy,
        "worker_checkpoint": sender.checkpoint,
        "worker_revision": sender.revision,
        "worker_tokenizer": sender.checkpoint,
        "worker_tokenizer_revision": sender.revision,
        "worker_native_max_model_len": sender.native_max_model_len,
        "worker_context_extension": "none",
        "source_identity": record.source_identity,
        "sender_prepared_sha256": record.sender_prepared_sha256,
        "tokenizer_json_sha256": record.tokenizer_json_sha256,
        "chat_template_blob": record.chat_template_blob,
        "text_registration_fingerprint": TEXT_REGISTRATION_FINGERPRINT,
        "sampling_profile": GEMMA.decode.profile_id,
        "sampling_fingerprint": GEMMA.decode.identity_hash,
        "report_temperature": GEMMA.decode.temperature,
        "report_top_p": GEMMA.decode.top_p,
        "report_top_k": GEMMA.decode.top_k,
        "report_presence_penalty": GEMMA.decode.presence_penalty,
        "worker_prompt_tokens": [len(prompt) for prompt in record.prompt_ids],
        "worker_prompt_sha256": list(record.prompt_sha256),
        "reports": [str(row["text"]) for row in reports],
        "report_raw_outputs": [str(row["raw_text"]) for row in reports],
        "report_tokens": sum(int(row["decoded_tokens"]) for row in reports),
        "report_tokens_by_worker": [int(row["decoded_tokens"]) for row in reports],
        "report_token_ids_by_worker": [list(row["raw_token_ids"]) for row in reports],
        "report_visible_token_ids_by_worker": [list(row["ids"]) for row in reports],
        "report_visible_tokens_by_worker": [int(row["semantic_tokens"]) for row in reports],
        "report_generation_s": round(ladder.generation_s, 4),
        "draws_n": ladder.draws_n,
        "redraw_wall_s": round(ladder.redraw_wall_s, 4),
        "injected": any(injected),
        "report_injected_by_worker": injected,
        "report_seed_tag": tags.pop(),
        "report_seeds": [int(row["seed"]) for row in reports],
        "report_thinking_closed": closed,
        "report_thinking_closed_rate": round(sum(closed) / len(closed), 4),
        "report_failed": bool(failed),
        "report_failed_workers": failed,
        "report_failure": (
            f"workers {failed} unclosed or empty after {len(GEMMA_REPORT_SEED_TAGS)} draws"
            if failed
            else None
        ),
        "report_finish_reasons": [str(row["finish_reason"]) for row in reports],
        "report_decode_paths": [str(row["decode_path"]) for row in reports],
        "report_queue_s_mean": reports[0]["report_queue_s_mean"],
        "report_ttft_s_mean": reports[0]["report_ttft_s_mean"],
    }


def _complete_banked_bundle(
    sender: TextSenderSpec,
    record: GemmaPromptRecord,
    banked: Sequence[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Return a complete coherent banked bundle, or name it as incomplete."""
    present = [row for row in banked if row is not None]
    if len(present) != FANOUT_M3.workers_per_item:
        return None
    tags = {str(row["report_seed_tag"]) for row in present}
    if len(tags) != 1:
        raise RuntimeError(f"{record.qid}/{sender.semantic_arm}: banked batch mixed")
    mixed = f"{record.qid}/{sender.semantic_arm}: banked batch mixed"
    return _bundle(sender, record, present, _banked_ladder(present, mixed))


def _hydrate_reports(
    bank: TextReportBankProtocol,
    banked: Sequence[dict[str, Any] | None],
    accepted: Sequence[Mapping[str, Any]],
    record: GemmaPromptRecord,
    sender: TextSenderSpec,
    ladder: _Ladder,
) -> list[dict[str, Any]]:
    """Bank missing worker rows and retain already-validated replay rows."""
    hydrated: list[dict[str, Any]] = []
    for worker, report in enumerate(accepted):
        row = {
            **report,
            **ladder.row_fields(),
            "worker_checkpoint": sender.checkpoint,
            "worker_revision": sender.revision,
        }
        if banked[worker] is None:
            bank.bank_report(
                row,
                qid=record.qid,
                semantic_arm=sender.semantic_arm,
                worker=worker,
            )
            hydrated.append(row)
        else:
            hydrated.append(cast(dict[str, Any], banked[worker]))
    return hydrated


def generate_report_bundle(
    engine: TextEngine,
    tokenizer: Any,
    item: Mapping[str, Any],
    sender: TextSenderSpec,
    bank: TextReportBankProtocol,
    *,
    profile: BenchmarkProfile = FANOUTQA_NATURAL_DEV50,
) -> dict[str, Any]:
    """Generate or durably replay one exact all-worker report bundle."""
    record = prompt_record(tokenizer, item, sender, profile)
    existing = [
        bank.banked_report(record.qid, sender.semantic_arm, worker)
        for worker in range(FANOUT_M3.workers_per_item)
    ]
    banked = [
        _validate_banked(record, sender, worker, row, profile) if row is not None else None
        for worker, row in enumerate(existing)
    ]
    complete = _complete_banked_bundle(sender, record, banked)
    if complete is not None:
        return complete
    label = f"{record.qid}/{sender.semantic_arm}"
    present = [row for row in banked if row is not None]
    if present:
        tags = {str(row["report_seed_tag"]) for row in present}
        if len(tags) != 1:
            raise RuntimeError(f"{label}: partial batch mixed")
        ladder = _banked_ladder(present, f"{label}: partial batch mixed")
        accepted, invalid, _elapsed = _decode_draw(
            engine, tokenizer, record, sender, tag=tags.pop(), profile=profile
        )
        banked_failed = [
            worker
            for worker, row in enumerate(banked)
            if row is not None and not _row_accepted(row)
        ]
        if sorted(invalid) != banked_failed:
            raise RuntimeError(f"{record.qid}/{sender.semantic_arm}: replay failed at {invalid}")
        for worker, row in enumerate(banked):
            if row is not None and row["raw_token_ids"] != accepted[worker]["raw_token_ids"]:
                raise RuntimeError(f"{label}/w{worker}: replay drifted")
    else:
        # Draw exhaustion quarantines the cell rather than raising: the final
        # draw banks as evidence. Every draw is timed on its own and the banked
        # span is the last draw executed, accepted or exhausting.
        draw_elapsed: list[float] = []
        accepted = []
        for tag in GEMMA_REPORT_SEED_TAGS:
            accepted, invalid, elapsed = _decode_draw(
                engine, tokenizer, record, sender, tag=tag, profile=profile
            )
            draw_elapsed.append(elapsed)
            if not invalid:
                break
        ladder = _Ladder(
            generation_s=draw_elapsed[-1],
            draws_n=len(draw_elapsed),
            redraw_wall_s=sum(draw_elapsed[:-1]),
        )
    hydrated = _hydrate_reports(bank, banked, accepted, record, sender, ladder)
    return _bundle(sender, record, hydrated, ladder)


__all__ = ("generate_report_bundle",)
