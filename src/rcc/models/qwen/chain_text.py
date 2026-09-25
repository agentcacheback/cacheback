"""The chain's text arm: four note rewrites, in order, on one sender.

The latent arm hands rows forward, this arm hands words. Hop t reads the notes
hop t-1 shipped plus one new part, and the receiver answers from the fourth.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from rcc.benchmarks.fanoutqa.source_topology import token_sequence_sha256
from rcc.benchmarks.longbench_v2.data import ChainItem
from rcc.benchmarks.longbench_v2.prompts import PROMPT_BUILDERS, rewrite_prompt
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.qwen.text import (
    continue_unclosed,
    mean_elapsed,
    ordered_results,
    validate_draw,
)
from rcc.models.qwen.text_decode import (
    QWEN_REPORT_SEED_TAGS,
    QWEN_TEXT_ARMS,
    QwenDecodeRequest,
    QwenDecodeSpec,
    TextEngine,
)
from rcc.models.qwen.text_output import TextCompletion
from rcc.models.route import Decoder, RouteFamily
from rcc.topologies.chain import HOPS

#: How many places a wall clock banks at. The bundle's validator and the
#: banked row's reader both round to this.
WALL_CLOCK_PLACES = 4


def _hop_row(fields: Mapping[str, object], name: str, kind: type, label: str) -> list[Any]:
    """Return one four-hop row of one type, or refuse the field by its own name."""
    value = fields.get(name)
    row = cast(Sequence[object], value) if isinstance(value, (list, tuple)) else ()
    if len(row) != HOPS or any(type(item) is not kind for item in row):
        raise ValueError(f"Qwen notes chain {name} is not one {label} per hop")
    return list(row)


def _validate_chain_ladder(fields: Mapping[str, object]) -> list[int]:
    """Return the hop draws, refusing a ladder its total or its seed tags deny."""
    draws = cast(list[int], _hop_row(fields, "draws_by_hop", int, "draw count"))
    tags = cast(list[str], _hop_row(fields, "report_seed_tags_by_hop", str, "seed tag"))
    if any(not 1 <= count <= len(QWEN_REPORT_SEED_TAGS) for count in draws):
        raise ValueError("Qwen hop draw counts differ from the registered ladder")
    if fields.get("draws_n") != sum(draws):
        raise ValueError("Qwen notes chain draw count differs from its hop ladder")
    if any(tag != QWEN_REPORT_SEED_TAGS[count - 1] for tag, count in zip(tags, draws, strict=True)):
        raise ValueError("Qwen hop draw count differs from its registered seed tag")
    return draws


def _validate_chain_closure(fields: Mapping[str, object], *, draws: Sequence[int]) -> None:
    """Refuse an injection summary, a closure roster, or a wall clock that cannot be banked."""
    injected = cast(list[bool], _hop_row(fields, "report_injected_by_hop", bool, "flag"))
    closed = cast(list[bool], _hop_row(fields, "report_thinking_closed", bool, "flag"))
    if type(fields.get("injected")) is not bool or fields.get("injected") != any(injected):
        raise ValueError("Qwen report injection summary differs from its hop roster")
    if any(
        was_injected and not was_closed
        for was_injected, was_closed in zip(injected, closed, strict=True)
    ):
        raise ValueError("Qwen closer injection must leave its own hop closed")
    clocks = [fields.get("report_generation_s"), fields.get("redraw_wall_s")]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
        for value in clocks
    ):
        raise ValueError("Qwen notes chain wall clock is invalid")
    if (sum(draws) > HOPS) != (cast(float, fields.get("redraw_wall_s")) > 0.0):
        raise ValueError("Qwen notes chain redraw wall clock differs from its draw count")


def _validate_chain_notes(fields: Mapping[str, object]) -> None:
    """Refuse notes, survivors, or a failure roster the four notes do not witness."""
    notes = cast(list[str], _hop_row(fields, "notes_by_hop", str, "note"))
    read = cast(list[str], _hop_row(fields, "notes_read_by_hop", str, "note"))
    failed = [hop for hop, note in enumerate(notes, start=1) if not note]
    if fields.get("failed_hops") != failed:
        raise ValueError("Qwen chain exhausted hops differ from the notes they shipped")
    if type(fields.get("report_failed")) is not bool or fields.get("report_failed") != (
        bool(failed) or not notes[-1]
    ):
        raise ValueError("Qwen chain failure flag differs from the hops it shipped")
    carried = ""
    survivors: list[str] = []
    for note in notes:
        survivors.append(carried)
        carried = note or carried
    if read != survivors:
        raise ValueError("Qwen chain notes each hop read differ from the notes it shipped")
    if fields.get("reports") != [notes[-1]]:
        raise ValueError("Qwen notes chain ships one ticket, the fourth note")


def validate_notes_chain_fields(fields: Mapping[str, object]) -> None:
    """Refuse a banked notes chain whose own hop evidence disagrees.

    One redraw ladder per hop, one seed tag per hop, one wall clock over the
    discarded draws, and one ticket. The three parts run in order.
    """
    draws = _validate_chain_ladder(fields)
    _validate_chain_closure(fields, draws=draws)
    _validate_chain_notes(fields)


@dataclass(frozen=True)
class QwenNotesChainBundle:
    """Four accepted notes and the raw-token evidence of every hop.

    ``notes_read_by_hop`` is not ``notes`` shifted by one: after a hop that
    shipped empty, the next hop reads the last notes that shipped.
    """

    notes: tuple[str, ...]
    notes_read_by_hop: tuple[str, ...]
    raw_outputs: tuple[str, ...]
    token_ids_by_hop: tuple[tuple[int, ...], ...]
    tokens_by_hop: tuple[int, ...]
    finish_reasons: tuple[str, ...]
    seeds: tuple[int, ...]
    seed_tags: tuple[str, ...]
    thinking_closed: tuple[bool, ...]
    injected_by_hop: tuple[bool, ...]
    draws_by_hop: tuple[int, ...]
    prompt_token_sha256: tuple[str, ...]
    prompt_tokens_by_hop: tuple[int, ...]
    chunk_text_sha256: tuple[str, ...]
    chunk_text_tokens: tuple[int, ...]
    decode: QwenDecodeSpec
    generation_s: float
    redraw_wall_s: float
    queue_s_mean: float | None
    ttft_s_mean: float | None
    report_failed: bool
    failed_hops: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate one complete four-hop chain, an exhausted hop included."""
        rows = (
            self.notes,
            self.notes_read_by_hop,
            self.raw_outputs,
            self.token_ids_by_hop,
            self.tokens_by_hop,
            self.finish_reasons,
            self.seeds,
            self.seed_tags,
            self.thinking_closed,
            self.injected_by_hop,
            self.draws_by_hop,
            self.prompt_token_sha256,
            self.prompt_tokens_by_hop,
            self.chunk_text_sha256,
            self.chunk_text_tokens,
        )
        if any(len(row) != HOPS for row in rows):
            raise ValueError(f"a Qwen notes chain carries exactly {HOPS} hop rows")
        self._validate_ladder()
        self._validate_evidence()

    def _validate_ladder(self) -> None:
        """Refuse a ladder, a clock, or a closure roster that cannot be banked."""
        for tag, draws in zip(self.seed_tags, self.draws_by_hop, strict=True):
            if tag not in QWEN_REPORT_SEED_TAGS or draws != QWEN_REPORT_SEED_TAGS.index(tag) + 1:
                raise ValueError("Qwen hop draw count differs from its registered seed tag")
        for value in (self.generation_s, self.redraw_wall_s):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError("Qwen notes chain wall clock is invalid")
        # The banked value is the rounded one and the row's reader holds the
        # ladder to it, so a redraw faster than four places cannot bank a row
        # this rule refuses.
        redraw = round(self.redraw_wall_s, WALL_CLOCK_PLACES)
        redrew = sum(self.draws_by_hop) > HOPS
        if not redrew and redraw != 0.0:
            raise ValueError("Qwen single-draw hops cannot carry redraw wall clock")
        if redrew and redraw <= 0.0:
            raise ValueError("Qwen discarded hop draws must carry redraw wall clock")
        if any(
            was_injected and not was_closed
            for was_injected, was_closed in zip(
                self.injected_by_hop, self.thinking_closed, strict=True
            )
        ):
            raise ValueError("Qwen closer injection must leave its own hop closed")

    def _validate_evidence(self) -> None:
        """Refuse notes, closure flags, or counts that differ from the raw draws."""
        family = self.decode.family
        expected = tuple(
            family.post_think_handoff(raw, ended=reason == "stop")
            for raw, reason in zip(self.raw_outputs, self.finish_reasons, strict=True)
        )
        if expected != self.notes:
            raise ValueError("Qwen chain notes differ from their raw post-think reconstruction")
        closed = tuple(
            not self.decode.enable_thinking or family.think_close in raw for raw in self.raw_outputs
        )
        if closed != self.thinking_closed:
            raise ValueError("Qwen chain closure fields differ from raw outputs")
        if any(
            len(ids) != count
            for ids, count in zip(self.token_ids_by_hop, self.tokens_by_hop, strict=True)
        ):
            raise ValueError("Qwen chain token counts differ from their raw ids")
        if self.failed_hops != tuple(
            hop for hop, note in enumerate(self.notes, start=1) if not note
        ):
            raise ValueError("Qwen chain exhausted hops differ from the notes they shipped")
        if self.report_failed != (not self.notes[-1] or bool(self.failed_hops)):
            raise ValueError("Qwen chain failure flag differs from the hops it shipped")
        carried = ""
        read: list[str] = []
        for note in self.notes:
            read.append(carried)
            carried = note or carried
        if tuple(read) != self.notes_read_by_hop:
            raise ValueError("Qwen chain notes each hop read differ from the notes it shipped")

    @property
    def ticket(self) -> str:
        """Return the fourth note, the one text the receiver answers from."""
        return self.notes[-1]

    def result_fields(self) -> dict[str, object]:
        """Return the serializable per-hop fields a chain result row banks.

        ``reports`` holds the one ticket, the way the receiver reads it, and
        ``report_failed`` names any hop that shipped empty, not the ticket.
        """
        return {
            "reports": [self.ticket],
            "notes_by_hop": list(self.notes),
            "notes_read_by_hop": list(self.notes_read_by_hop),
            "report_raw_outputs": list(self.raw_outputs),
            "report_tokens": sum(self.tokens_by_hop),
            "report_tokens_by_hop": list(self.tokens_by_hop),
            "report_token_ids_by_hop": [list(ids) for ids in self.token_ids_by_hop],
            "report_generation_s": round(self.generation_s, WALL_CLOCK_PLACES),
            "draws_n": sum(self.draws_by_hop),
            "draws_by_hop": list(self.draws_by_hop),
            "redraw_wall_s": round(self.redraw_wall_s, WALL_CLOCK_PLACES),
            "injected": any(self.injected_by_hop),
            "report_injected_by_hop": list(self.injected_by_hop),
            "report_seed_tags_by_hop": list(self.seed_tags),
            "report_seeds": list(self.seeds),
            "report_thinking_closed": list(self.thinking_closed),
            "report_enable_thinking": self.decode.enable_thinking,
            "decode": self.decode.to_dict(),
            **(
                {"report_decode": self.decode.to_dict()}
                if self.decode.family.native_sender_prompts
                else {}
            ),
            "report_finish_reasons": list(self.finish_reasons),
            "report_failed": self.report_failed,
            "failed_hops": list(self.failed_hops),
            "report_failure": (
                f"hops {list(self.failed_hops)} shipped empty notes after "
                f"{len(QWEN_REPORT_SEED_TAGS)} draws"
                if self.report_failed
                else None
            ),
            "report_queue_s_mean": self.queue_s_mean,
            "report_ttft_s_mean": self.ttft_s_mean,
            "report_prompt_token_sha256_by_hop": list(self.prompt_token_sha256),
            "report_prompt_tokens_by_hop": list(self.prompt_tokens_by_hop),
            "chunk_text_sha256": list(self.chunk_text_sha256),
            "chunk_text_tokens": list(self.chunk_text_tokens),
        }


@dataclass(frozen=True)
class _HopDraw:
    """The shipped draw of one hop and the ladder that reached it."""

    note: str
    result: TextCompletion
    prompt_ids: tuple[int, ...]
    seed: int
    tag: str
    injected: bool
    draws: int
    shipped_s: float
    redraw_s: float


def _require_registered_chain_profile(profile: BenchmarkProfile) -> None:
    """Refuse by name, before any decode, a benchmark this rewriter cannot run."""
    if profile.workers_per_item != HOPS:
        raise ValueError(
            f"Qwen notes chains rewrite {HOPS} hops; {profile.benchmark_key} registers "
            f"workers_per_item {profile.workers_per_item}"
        )
    if profile.prompt_builder not in PROMPT_BUILDERS:
        raise ValueError(
            f"Qwen notes chains implement the {sorted(PROMPT_BUILDERS)} prompt builders; "
            f"{profile.benchmark_key} registers {profile.prompt_builder}"
        )


def _chunk_text(ledger_tokenizer: Any, item: ChainItem, hop: int) -> str:
    """Return the part hop ``hop`` reads, decoded from its registered ids.

    The decode is the profile-pinned tokenizer's, the one that cut the ledger,
    so the body a hop reads does not depend on the sender the arm opened.
    """
    return str(
        ledger_tokenizer.decode(
            [int(token) for token in item.chunks[hop - 1]], skip_special_tokens=False
        )
    )


def _hop_prompt(
    tokenizer: Any,
    item: ChainItem,
    *,
    chunk_text: str,
    notes: str,
    hop: int,
    qid: str,
    semantic_arm: str,
    enable_thinking: bool,
    builder: str,
    family: RouteFamily,
) -> str:
    """Render one hop's rewrite under the profile's builder, naming the cell in any refusal."""
    try:
        return rewrite_prompt(
            tokenizer,
            item.question,
            item.choices,
            notes=notes,
            chunk_text=chunk_text,
            hop=hop,
            enable_thinking=enable_thinking,
            builder=builder,
            family=family,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"{qid}/{semantic_arm}/hop {hop}: the rewrite prompt refused "
            f"{len(item.chunks[hop - 1])} chunk tokens and {len(notes)} note characters: {exc}"
        ) from exc


def _consumed_prompt_ids(
    tokenizer: Any,
    prompt: str,
    result: TextCompletion,
    *,
    qid: str,
    semantic_arm: str,
    hop: int,
) -> tuple[int, ...]:
    """Bind one draw to the exact prompt this run rendered.

    A backend that applies its own chat template, or truncates against its
    window, consumes other ids, so the draw is refused rather than banked.
    """
    consumed = tuple(int(token) for token in result.prompt_token_ids)
    rendered = tuple(
        int(token) for token in tokenizer(prompt, add_special_tokens=False)["input_ids"]
    )
    if consumed != rendered:
        raise RuntimeError(
            f"{qid}/{semantic_arm}/hop {hop}: the sender consumed {len(consumed)} prompt "
            f"tokens that differ from the {len(rendered)} tokens this run rendered"
        )
    return consumed


def _draw_hop(
    engine: TextEngine,
    tokenizer: Any,
    prompt: str,
    *,
    hop: int,
    qid: str,
    semantic_arm: str,
    family: RouteFamily,
    decoder: Decoder,
    decode: QwenDecodeSpec,
    profile: BenchmarkProfile,
) -> _HopDraw:
    """Draw one hop's notes, advancing the seed ladder only while it ships empty.

    The shipped draw is timed on its own and the draws before it bank as redraw
    wall clock; an exhausted ladder ships its last draw rather than failing.
    """
    elapsed: list[float] = []
    for tag in QWEN_REPORT_SEED_TAGS:
        seed = profile.report_seeds(qid, tag)[hop - 1]
        request = QwenDecodeRequest(
            request_id=f"{qid}:{semantic_arm}:rewrite:h{hop}:{tag}",
            qid=qid,
            semantic_arm=semantic_arm,
            sample_tag=tag,
            member_index=hop - 1,
            seed=seed,
            decode=decode,
            profile=profile,
        )
        started = time.perf_counter()
        results = ordered_results((request,), engine.decode_text_full((prompt,), (request,)))
        validate_draw(qid, results)
        prompt_ids = _consumed_prompt_ids(
            tokenizer, prompt, results[0], qid=qid, semantic_arm=semantic_arm, hop=hop
        )
        results, injected = continue_unclosed(
            engine,
            (request,),
            results,
            qid=qid,
            family=family,
            decoder=decoder,
            profile=profile,
        )
        elapsed.append(time.perf_counter() - started)
        # A sender whose prompt opened the thinking block writes its notes
        # inside it, so the handoff reading is used: the visible text alone
        # would call every such hop empty and burn the ladder.
        note = family.post_think_handoff(
            results[0].text, ended=str(results[0].finish_reason) == "stop"
        )
        if note or tag == QWEN_REPORT_SEED_TAGS[-1]:
            return _HopDraw(
                note=note,
                result=results[0],
                prompt_ids=prompt_ids,
                seed=seed,
                tag=tag,
                injected=injected[0],
                draws=len(elapsed),
                shipped_s=elapsed[-1],
                redraw_s=sum(elapsed[:-1]),
            )
    raise RuntimeError(f"{qid}/{semantic_arm}/hop {hop}: the registered redraw ladder is empty")


def generate_notes_chain(
    engine: TextEngine,
    tokenizer: Any,
    item: ChainItem,
    *,
    qid: str,
    semantic_arm: str,
    family: RouteFamily,
    decoder: Decoder,
    profile: BenchmarkProfile,
    ledger_tokenizer: Any,
) -> QwenNotesChainBundle:
    """Rewrite the running notes across the four registered hops, in order.

    ``ledger_tokenizer`` is the pinned one that cut the chunks, so every arm
    rewrites the same body. An exhausted hop ships an empty note.
    """
    profile = family.benchmark_profile(profile)
    _require_registered_chain_profile(profile)
    if semantic_arm not in QWEN_TEXT_ARMS:
        raise ValueError("Qwen text generation requires one registered sender arm")
    if item.qid != qid:
        raise RuntimeError(f"{qid}/{semantic_arm}: the chain item carries qid {item.qid!r}")
    if len(item.chunks) != HOPS:
        raise RuntimeError(
            f"{qid}/{semantic_arm}: {len(item.chunks)} chunks differ from the "
            f"registered {HOPS} hops"
        )
    family = family.sender_family(semantic_arm)
    decode = QwenDecodeSpec("report", family, profile=profile)
    hops: list[_HopDraw] = []
    parts: list[str] = []
    read: list[str] = []
    notes = ""
    for hop in range(1, HOPS + 1):
        parts.append(_chunk_text(ledger_tokenizer, item, hop))
        read.append(notes)
        prompt = _hop_prompt(
            tokenizer,
            item,
            chunk_text=parts[-1],
            notes=notes,
            hop=hop,
            qid=qid,
            semantic_arm=semantic_arm,
            enable_thinking=decode.enable_thinking,
            builder=profile.prompt_builder,
            family=family,
        )
        drawn = _draw_hop(
            engine,
            tokenizer,
            prompt,
            hop=hop,
            qid=qid,
            semantic_arm=semantic_arm,
            family=family,
            decoder=decoder,
            decode=decode,
            profile=profile,
        )
        hops.append(drawn)
        notes = drawn.note or notes
    failed_hops = tuple(hop for hop, drawn in enumerate(hops, start=1) if not drawn.note)
    shipped = tuple(drawn.result for drawn in hops)
    return QwenNotesChainBundle(
        notes=tuple(drawn.note for drawn in hops),
        notes_read_by_hop=tuple(read),
        raw_outputs=tuple(drawn.result.text for drawn in hops),
        token_ids_by_hop=tuple(
            tuple(int(token) for token in drawn.result.token_ids) for drawn in hops
        ),
        tokens_by_hop=tuple(int(drawn.result.n_tokens) for drawn in hops),
        finish_reasons=tuple(str(drawn.result.finish_reason) for drawn in hops),
        seeds=tuple(drawn.seed for drawn in hops),
        seed_tags=tuple(drawn.tag for drawn in hops),
        thinking_closed=tuple(
            not decode.enable_thinking or family.think_close in drawn.result.text for drawn in hops
        ),
        injected_by_hop=tuple(drawn.injected for drawn in hops),
        draws_by_hop=tuple(drawn.draws for drawn in hops),
        prompt_token_sha256=tuple(token_sequence_sha256(drawn.prompt_ids) for drawn in hops),
        prompt_tokens_by_hop=tuple(len(drawn.prompt_ids) for drawn in hops),
        chunk_text_sha256=tuple(hashlib.sha256(part.encode("utf-8")).hexdigest() for part in parts),
        chunk_text_tokens=tuple(
            len(ledger_tokenizer.encode(part, add_special_tokens=False)) for part in parts
        ),
        decode=decode,
        generation_s=sum(drawn.shipped_s for drawn in hops),
        redraw_wall_s=sum(drawn.redraw_s for drawn in hops),
        queue_s_mean=mean_elapsed(shipped, later="scheduled_ts", earlier="queued_ts"),
        ttft_s_mean=mean_elapsed(shipped, later="first_token_ts", earlier="queued_ts"),
        report_failed=not hops[-1].note or bool(failed_hops),
        failed_hops=failed_hops,
    )


__all__ = (
    "WALL_CLOCK_PLACES",
    "QwenNotesChainBundle",
    "generate_notes_chain",
    "validate_notes_chain_fields",
)
