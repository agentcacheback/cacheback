"""The chain engine route, the four-hop loop and its terminal block, and the note rewrites."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tests.conftest import random_ids
from tests.engine_fakes import fake_producer
from tests.longbench_coa_support import (
    CHAIN_PROFILE,
    REDRAW_SCRIPT,
    FakeTokenizer,
    TinyChainTokenizer,
    chain_item,
)

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.longbench_v2 import LONGBENCH_COA_EASY50_BOUNDED, LONGBENCH_COA_EASY50_TEXT
from rcc.benchmarks.longbench_v2.geometry import (
    HOP_WRAPPER_ALLOWANCE,
    chain_wrapper_allowances,
    rewrite_prompt_ceiling,
)
from rcc.benchmarks.longbench_v2.prompts import NO_NOTES, hop_prompt_ids
from rcc.models.qwen import QWEN_FAMILY, engine, engine_routes
from rcc.models.qwen import chain_text as qwen_chain_text
from rcc.models.qwen.backend import QwenBackendSettings, effective_engine_kwargs
from rcc.models.qwen.capture import (
    QwenFlatPayload,
    chain_plan_sha256,
    latent_plan_sha256,
    qwen_w16_keep,
    registered_selection,
    selected_indices_sha256,
    tensor_content_sha256,
)
from rcc.models.qwen.chain import _hop_prompt_sha256, produce_chain_payload
from rcc.models.qwen.chain_text import QwenNotesChainBundle, generate_notes_chain
from rcc.models.qwen.engine import (
    QWEN_CHAIN_ENGINE_GPU_MEMORY_UTILIZATION,
    QWEN_CHAIN_ENGINE_MAX_BATCHED_TOKENS,
    QWEN_CHAIN_ENGINE_MAX_MODEL_LEN,
    QWEN_CHAIN_ENGINE_ROUTE,
    QWEN_ENGINE_MAX_MODEL_LEN,
    QWEN_VLLM_VERSION,
    EngineProducer,
    HopProduct,
    QwenEngineSettings,
    vllm_embeds_request_factory,
)
from rcc.models.qwen.engine_routes import _CHAIN_ROUTE, _FANOUTQA_ROUTE
from rcc.models.qwen.results import (
    _ChainRowRule,
    _FanoutqaRowRule,
    validate_and_rescore_result,
)
from rcc.models.qwen.text import _closer_is_affordable
from rcc.models.qwen.text_decode import QWEN_REPORT_SEED_TAGS
from rcc.models.route import RouteFamily, tokenizer_decoder
from rcc.topologies.chain import HOPS, LATENT_STEPS, bounded_budget, hop_budget, rerank_budget
from rcc.topologies.chain.layout import CHAIN_TERMINAL_LAYOUT

# --- The chain capture route: a second registered engine tuple and one hop.

_LATENT_STEPS = 40


def test_chain_settings_are_a_second_registered_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two tuples are registered, nothing between them is, and the role carries.

    The chain route differs from the FanOutQA capture route in five fields and
    in nothing else, and a value off both tuples is refused by field name.
    """
    chain = QwenEngineSettings.for_chain()
    assert chain.route == QWEN_CHAIN_ENGINE_ROUTE
    assert chain.max_model_len == QWEN_CHAIN_ENGINE_MAX_MODEL_LEN == 131_072
    assert chain.max_num_batched_tokens == QWEN_CHAIN_ENGINE_MAX_BATCHED_TOKENS == 16_384
    assert chain.enable_prompt_embeds and chain.enforce_eager
    assert not chain.enable_prefix_caching
    assert chain.gpu_memory_utilization == QWEN_CHAIN_ENGINE_GPU_MEMORY_UTILIZATION
    assert QwenEngineSettings().max_model_len == QWEN_ENGINE_MAX_MODEL_LEN
    # The two tables spell their own values rather than read the constants the
    # settings are built from, so a constant moved in source leaves the table it
    # is checked against and the engine open refuses it by field.
    assert dict(_FANOUTQA_ROUTE) == {
        "route": "vllm-prefill-zero-copy-alias-roll-v1",
        "gpu_memory_utilization": 0.40,
        "max_model_len": 50_000,
        "max_num_batched_tokens": 50_000,
        "enable_prompt_embeds": False,
        "max_num_seqs": 16,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "disable_hybrid_kv_cache_manager": True,
        "disable_log_stats": True,
        "kv_connector": "RCCConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "rcc.injector.connector",
        "v1_multiprocessing": False,
    }
    assert dict(_CHAIN_ROUTE) == {
        **dict(_FANOUTQA_ROUTE),
        "route": "vllm-prefill-embeds-zero-copy-alias-roll-v1",
        "gpu_memory_utilization": 0.45,
        "max_model_len": 131_072,
        "max_num_batched_tokens": 16_384,
        "enable_prompt_embeds": True,
    }
    # The registered tuples are read-only, so no import can quietly move a
    # route out from under the settings that certify against it.
    with pytest.raises(TypeError):
        cast(Any, _CHAIN_ROUTE)["max_model_len"] = 1
    with pytest.raises(ValueError, match=r"the registered vLLM route.*max_model_len=100000"):
        dataclasses.replace(chain, max_model_len=100_000)
    with pytest.raises(ValueError, match=r"the registered vLLM route.*enable_prompt_embeds=True"):
        dataclasses.replace(QwenEngineSettings(), enable_prompt_embeds=True)

    kwargs = effective_engine_kwargs(
        "ckpt",
        "rev",
        settings=QwenBackendSettings(capture=True, chain=True),
        family=QWEN_FAMILY,
        tokenizer="tok",
        tokenizer_revision="rev",
    )
    assert kwargs["max_model_len"] == QWEN_CHAIN_ENGINE_MAX_MODEL_LEN
    assert kwargs["max_num_batched_tokens"] == QWEN_CHAIN_ENGINE_MAX_BATCHED_TOKENS
    assert kwargs["enable_prompt_embeds"] is True
    assert kwargs["gpu_memory_utilization"] == QWEN_CHAIN_ENGINE_GPU_MEMORY_UTILIZATION
    assert kwargs["enforce_eager"] is True
    # The one-step variant: the same tuple with the prefill in one step at the
    # pre-chunking memory share, applied after the registered values are read.
    one_step = effective_engine_kwargs(
        "ckpt",
        "rev",
        settings=QwenBackendSettings(capture=True, chain=True, chain_one_step_prefill=True),
        family=QWEN_FAMILY,
        tokenizer="tok",
        tokenizer_revision="rev",
    )
    assert one_step["max_num_batched_tokens"] == QWEN_CHAIN_ENGINE_MAX_MODEL_LEN
    assert one_step["gpu_memory_utilization"] == 0.60
    assert {
        k: v
        for k, v in one_step.items()
        if k not in ("max_num_batched_tokens", "gpu_memory_utilization")
    } == {
        k: v
        for k, v in kwargs.items()
        if k not in ("max_num_batched_tokens", "gpu_memory_utilization")
    }
    # The Nemotron capture tuple chunks at 4,096 tokens on a 0.40 share; its
    # one-step run takes the shared overrides instead of chunking.
    from rcc.models.nemotron import NEMOTRON, NEMOTRON_FAMILY

    nemotron_chunked, nemotron_one_step = (
        effective_engine_kwargs(
            NEMOTRON.checkpoint,
            NEMOTRON.revision,
            settings=QwenBackendSettings(capture=True, chain=True, chain_one_step_prefill=flag),
            family=NEMOTRON_FAMILY,
            tokenizer=NEMOTRON.tokenizer,
            tokenizer_revision=NEMOTRON.tokenizer_revision,
        )
        for flag in (False, True)
    )
    assert nemotron_chunked["max_num_batched_tokens"] == 4096
    assert nemotron_chunked["gpu_memory_utilization"] == 0.40
    assert nemotron_one_step["max_num_batched_tokens"] == QWEN_CHAIN_ENGINE_MAX_MODEL_LEN
    assert nemotron_one_step["gpu_memory_utilization"] == 0.60
    assert (
        nemotron_one_step["max_model_len"]
        == nemotron_chunked["max_model_len"]
        == QWEN_CHAIN_ENGINE_MAX_MODEL_LEN
        == 131_072
    )
    assert {
        k: v
        for k, v in nemotron_one_step.items()
        if k not in ("max_num_batched_tokens", "gpu_memory_utilization")
    } == {
        k: v
        for k, v in nemotron_chunked.items()
        if k not in ("max_num_batched_tokens", "gpu_memory_utilization")
    }
    with pytest.raises(ValueError, match="chain"):
        QwenBackendSettings(chain=True)
    with pytest.raises(ValueError, match="one-step"):
        QwenBackendSettings(capture=True, chain_one_step_prefill=True)

    # The settings class holds both capture routes, so a window moved out from
    # under a route reaches the engine as a refusal naming the field. Both
    # roles are checked, so neither route can move alone.
    for chain_role, window in (
        (False, "QWEN_ENGINE_MAX_MODEL_LEN"),
        (True, "QWEN_CHAIN_ENGINE_MAX_MODEL_LEN"),
    ):
        monkeypatch.setattr(engine_routes, window, 100_000)
        with pytest.raises(ValueError, match=r"the registered vLLM route.*max_model_len=100000"):
            effective_engine_kwargs(
                "ckpt",
                "rev",
                settings=QwenBackendSettings(capture=True, chain=chain_role),
                family=QWEN_FAMILY,
                tokenizer="tok",
                tokenizer_revision="rev",
            )
        monkeypatch.undo()

    # The receiver role is not a capture route and keeps the served window of
    # the lane's own runtime flags, so the certification above never reaches it.
    receiver = effective_engine_kwargs(
        "ckpt",
        "rev",
        settings=QwenBackendSettings(receiver=True),
        family=QWEN_FAMILY,
        tokenizer="tok",
        tokenizer_revision="rev",
    )
    assert receiver["max_model_len"] == int(
        dict(QWEN_FAMILY.profile.runtime.engine_flags)["max_model_len"]
    )
    assert receiver["enable_prompt_embeds"] is True
    assert receiver["gpu_memory_utilization"] == 0.80
    assert "kv_transfer_config" not in receiver


def test_hop_one_is_the_worker_path_and_hop_two_adds_the_prefix(
    tiny_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hop one is the FanOutQA worker path, and a prefix rides in front of hop two.

    Hop one matches `produce_worker` row for row, and hop two's sequence is
    prefix plus prompt plus the forty latent rows, the prefix unchanged.
    """
    producer = fake_producer(tiny_model)
    ids = random_ids(length=12, seed=3)
    judger = random_ids(length=7, seed=4)
    mask = torch.ones_like(judger)
    question = [int(token) for token in judger[0, 2:5].tolist()]

    worker = producer.produce_worker(
        ids, judger_ids=judger, judger_mask=mask, question_ids=question
    )
    hop = producer.produce_hop(
        None, ids, judger_ids=judger, judger_mask=mask, question_ids=question
    )

    assert torch.equal(worker.embeds, hop.embeds)
    assert hop.length == worker.length == 12 + _LATENT_STEPS
    assert hop.prefix_rows == 0 and hop.prompt_rows == 12
    for name in ("snap", "support"):
        assert torch.equal(worker.scores[name], hop.scores[name])

    # A hop that hands its cache to the scorer scores the same bytes. The chain
    # runs that way, because the copy does not fit at the smallest ratios.
    taken = producer.produce_hop(
        None, ids, judger_ids=judger, judger_mask=mask, question_ids=question, consume_past=True
    )
    assert taken.length == hop.length
    for name in ("snap", "support"):
        assert torch.equal(taken.scores[name], hop.scores[name])
    prefix = hop.embeds[:, :5]
    second = producer.produce_hop(
        prefix, ids, judger_ids=judger, judger_mask=mask, question_ids=question
    )

    assert second.length == 5 + 12 + _LATENT_STEPS
    assert second.prefix_rows == 5 and second.prompt_rows == 12
    assert torch.equal(second.embeds[:, :5], prefix)
    assert second.scores["support"].shape == (second.length,)
    with pytest.raises(ValueError, match=r"prefix_rows must be \[1, P, D\]"):
        producer.produce_hop(
            prefix[0], ids, judger_ids=judger, judger_mask=mask, question_ids=question
        )

    # The rows a hop submits are the one seam the fake engine stands in for,
    # so the production factory runs here over a stub of the pinned vLLM:
    # detached host rows in bfloat16, under the key the connector reads.
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **fields: SimpleNamespace(**fields)),
    )
    monkeypatch.setattr(engine, "version", lambda name: "0.10.0")
    with pytest.raises(RuntimeError, match=QWEN_VLLM_VERSION):
        vllm_embeds_request_factory("r", second.embeds[0])
    monkeypatch.setattr(engine, "version", lambda name: QWEN_VLLM_VERSION)
    prompt, sampling = vllm_embeds_request_factory("r", second.embeds[0])
    rows = cast(dict[str, torch.Tensor], prompt)["prompt_embeds"]
    assert set(cast(dict[str, Any], prompt)) == {"prompt_embeds"}
    assert rows.shape == second.embeds[0].shape
    assert rows.dtype is torch.bfloat16 and rows.device.type == "cpu"
    assert not rows.requires_grad
    assert (sampling.temperature, sampling.max_tokens, sampling.ignore_eos) == (0.0, 1, True)


# --- The four-hop loop, its provenance ledger, and the terminal block.

#: The re-ranked handoff arm the chain profile registers, and its ratio.
_ARM = "latent_query_support_rerank_r4"


_RATIO = 4


_TEXT_ARM = "text_medium"


#: A fan-out arm, for the recipe the chain does not sign.
_FANOUT_ARM = "latent_query_support_r2"


#: The smallest bounded arm and its cap. The tiny items never reach the cap,
#: so the walk below also runs the same arm at a cap the tiny hops exceed.
_BOUNDED_ARM = "latent_query_support_bounded_b4096"


_BUDGET_ROWS = 4096


_TIGHT_ROWS = 64


_TIGHT_PROFILE = replace(
    LONGBENCH_COA_EASY50_BOUNDED,
    arms=tuple(
        replace(arm, budget_rows=_TIGHT_ROWS) if arm.arm_id == _BOUNDED_ARM else arm
        for arm in LONGBENCH_COA_EASY50_BOUNDED.arms
    ),
)


def _chain_payload(keeps: tuple[tuple[int, ...], ...], rows: torch.Tensor) -> QwenFlatPayload:
    """Build one terminal payload under the chain profile and its layout."""
    return QwenFlatPayload(
        rows=rows,
        semantic_arm=_ARM,
        latent_plan_sha256=chain_plan_sha256(_ARM, family=QWEN_FAMILY, profile=CHAIN_PROFILE),
        keeps=keeps,
        rows_by_worker=tuple(len(keep) for keep in keeps),
        selected_indices_sha256=selected_indices_sha256(keeps),
        tensor_sha256=tensor_content_sha256(rows),
        family=QWEN_FAMILY,
        profile=CHAIN_PROFILE,
        layout=CHAIN_TERMINAL_LAYOUT,
    )


def _chain_text_row(item: Any) -> dict[str, Any]:
    """Return a chain text row whose identity is complete and whose ladder is not.

    Every field the identity check reads is the registered one, so the row
    reaches the report-ladder check, where the missing hop fields refuse it.
    """
    policy = QWEN_FAMILY.policy(_TEXT_ARM)
    decode = QWEN_FAMILY.profile.decode
    return {
        "kind": "result",
        "qid": item.qid,
        "question": item.question,
        "family": QWEN_FAMILY.model_id,
        "semantic_arm": _TEXT_ARM,
        "policy": policy,
        "arm": policy,
        "channel": "text",
        "sample_seeds": list(LONGBENCH_COA_EASY50_TEXT.answer_seeds(item.qid, _TEXT_ARM)),
        "decode_tags": list(LONGBENCH_COA_EASY50_TEXT.sample_tags),
        "decode_profile": decode.profile_id,
        "decode_fingerprint": decode.identity_hash,
        "sample_aggregation": "mean_within_item_arm_then_macro_over_items_no_vote",
    }


def _record_hops(
    monkeypatch: pytest.MonkeyPatch, producer: EngineProducer
) -> list[tuple[torch.Tensor | None, HopProduct]]:
    """Keep the prefix each hop was given and the product the engine returned.

    The loop reads a hop's rows once and gathers the kept ones into the next
    hop's prefix, so only this spy can hold them for a later comparison.
    """
    products: list[tuple[torch.Tensor | None, HopProduct]] = []
    produce_hop = producer.produce_hop

    def record(prefix_rows: torch.Tensor | None, *args: Any, **kwargs: Any) -> HopProduct:
        product = produce_hop(prefix_rows, *args, **kwargs)
        products.append((prefix_rows, product))
        return product

    monkeypatch.setattr(producer, "produce_hop", record)
    return products


def test_chain_loop_recuts_globally_and_ships_the_terminal_block(
    tiny_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four hops, one budget law per hop, and only the last block ships.

    Under the global re-cut a row reaches the terminal block only by winning
    every cut after the hop that made it, which the origins ledger names.
    """
    producer = fake_producer(tiny_model)
    recorded = _record_hops(monkeypatch, producer)
    # The narrower tokenizer renders a longer prompt, so the r4 hop-one budget
    # clears the forty protected latent rows the way every real panel item does.
    tokenizer = TinyChainTokenizer(width=2)
    item = chain_item("loop", 10)

    production = produce_chain_payload(
        producer,
        tokenizer,
        item,
        semantic_arm=_ARM,
        family=QWEN_FAMILY,
        profile=CHAIN_PROFILE,
    )

    assert [hop.hop for hop in production.hops] == [1, 2, 3, 4]
    for previous, scores, hop in zip(
        (None, *production.hops[:-1]), production.selector_scores, production.hops, strict=True
    ):
        prefix = 0 if previous is None else len(previous.keep)
        assert hop.prefix_rows == prefix
        assert hop.rows_pre_cut == prefix + hop.prompt_rows + LATENT_STEPS
        # The cut takes the law's own budget: the carried count plus the new
        # rows cut at the ratio, which is more than the ratio alone names at
        # every hop after the first.
        assert hop.budget == rerank_budget(hop.rows_pre_cut, prefix, _RATIO) == len(hop.keep)
        assert hop.keep == qwen_w16_keep(
            scores, _RATIO, latent_steps=LATENT_STEPS, budget=hop.budget
        )
        if prefix:
            assert hop.budget > math.ceil(hop.rows_pre_cut / _RATIO)
        assert 0 in hop.keep
        assert set(range(hop.rows_pre_cut - LATENT_STEPS, hop.rows_pre_cut)) <= set(hop.keep)
        assert len(hop.origins) == len(hop.keep)
        assert all(1 <= born <= hop.hop for born, _ in hop.origins)
        assert all(0 <= position < hop.rows_pre_cut for _, position in hop.origins)
    # Hop one carries nothing, so it is the global re-cut at the ratio.
    assert production.hops[0].budget == math.ceil(production.hops[0].rows_pre_cut / _RATIO)
    # A carried row can lose its seat: at hop two the keep is not the carried
    # prefix followed by new rows, which is the whole point of the law.
    assert production.hops[1].keep[: production.hops[1].prefix_rows] != tuple(
        range(production.hops[1].prefix_rows)
    )
    # The sink is protected at every cut, and the carried rows lead the next
    # hop's sequence in order, so position zero of the terminal block is the
    # very first row of hop one.
    last = production.hops[-1]
    assert last.origins[0] == (1, 0)
    # The forty rows the last hop rolled are the tail of the terminal block,
    # and they were born there, so no other hop can claim those positions.
    assert [born for born, _ in last.origins[-LATENT_STEPS:]] == [HOPS] * LATENT_STEPS

    assert production.payload.layout == CHAIN_TERMINAL_LAYOUT
    assert production.payload.rows.shape[0] == len(last.keep)
    # The shipped block is the gather of the last cut, row for row. A count
    # alone would also pass for the first rows of the hop, or for the right
    # rows in the wrong order.
    assert len(recorded) == HOPS
    last_product = recorded[-1][1]
    assert torch.equal(
        production.payload.rows,
        last_product.embeds[0][list(last.keep)].detach().cpu().to(torch.bfloat16),
    )
    # The prefix a hop reads is the previous hop's gather, so the ledger's
    # positions name the rows whose content actually rode into the next cut.
    first_prefix, first_product = recorded[0]
    assert first_prefix is None
    second_prefix = recorded[1][0]
    assert second_prefix is not None
    assert torch.equal(second_prefix[0], first_product.embeds[0][list(production.hops[0].keep)])
    assert production.payload.rows_by_worker[-1] == len(last.keep)
    assert production.payload.keeps == tuple(hop.keep for hop in production.hops)
    assert production.payload.rows.dtype is torch.bfloat16
    assert len(production.selector_scores) == HOPS
    for scores, hop in zip(production.selector_scores, production.hops, strict=True):
        assert scores.shape == (hop.rows_pre_cut,)

    fields = production.result_fields()
    assert fields["producer_route"] == QWEN_CHAIN_ENGINE_ROUTE
    assert fields["payload_layout"] == CHAIN_TERMINAL_LAYOUT
    assert fields["hop_budgets"] == [hop.budget for hop in production.hops]
    assert fields["hop_prefix_rows"] == [hop.prefix_rows for hop in production.hops]
    assert fields["hop_prompt_rows"] == [hop.prompt_rows for hop in production.hops]
    assert fields["hop_rows_pre_cut"] == [hop.rows_pre_cut for hop in production.hops]
    # The per-hop keeps bank under their own name and the shipped-row vector
    # holds the one block handed to the receiver, so
    # `sum(latent_rows_by_worker) == latent_tokens` holds on both topologies.
    assert fields["kept_rows_by_hop"] == [len(hop.keep) for hop in production.hops]
    assert fields["latent_rows_by_worker"] == [len(last.keep)]
    assert fields["latent_tokens"] == len(last.keep)
    assert sum(cast(list[int], fields["latent_rows_by_worker"])) == fields["latent_tokens"]
    assert fields["keeps_by_worker"] == [list(hop.keep) for hop in production.hops]
    # Each hop's digest is rebuilt from the item's own chunk for that hop, and
    # the four chunks have four lengths, so a loop that read one part of the
    # source four times cannot bank a roster that still reads right.
    assert len({len(chunk) for chunk in item.chunks}) == HOPS
    assert fields["hop_prompt_sha256"] == [
        _hop_prompt_sha256(
            hop_prompt_ids(
                tokenizer,
                item.question,
                item.choices,
                item.chunks[hop.hop - 1],
                hop.hop,
                enable_thinking=QWEN_FAMILY.profile.decode.enable_thinking,
            )
        )
        for hop in production.hops
    ]
    assert fields["retention_prompt_sha256"] == production.retention_prompt_sha256
    assert fields["selector_score_tensor_sha256"] == [
        tensor_content_sha256(score) for score in production.selector_scores
    ]
    assert fields["terminal_origins"] == [[born, position] for born, position in last.origins]
    # The loop signs the recipe it walked, not the fan-out one.
    assert production.payload.latent_plan_sha256 == chain_plan_sha256(
        _ARM, family=QWEN_FAMILY, profile=CHAIN_PROFILE
    )
    # The law this arm ran is banked, so a row names the rule that made it
    # rather than leaving a reader to infer it from the arm name.
    assert fields["budget_law"] == production.budget_law == "rerank"
    # The keep at an explicit count over the same real scores: the sink and the rolled
    # tail are protected and every other row competes. The rerank count reproduces the
    # hop's keep; with no budget the ratio is required and an oversize count is refused.
    scores = production.selector_scores[-1]
    length = int(scores.numel())
    prefix = production.hops[-1].prefix_rows
    assert prefix > 0
    budget = rerank_budget(length, prefix, _RATIO)
    assert qwen_w16_keep(scores, _RATIO, latent_steps=LATENT_STEPS, budget=budget) == last.keep
    capped = bounded_budget(length, _TIGHT_ROWS)
    bounded = qwen_w16_keep(scores, None, latent_steps=LATENT_STEPS, budget=capped)
    assert len(bounded) == capped == _TIGHT_ROWS < budget
    assert 0 in bounded and set(range(length - LATENT_STEPS, length)) <= set(bounded)
    assert bounded == tuple(sorted(set(bounded))) and max(bounded) < length
    with pytest.raises(ValueError, match="needs a ratio"):
        qwen_w16_keep(scores, None, latent_steps=LATENT_STEPS)
    with pytest.raises(ValueError, match="nonempty part"):
        qwen_w16_keep(scores, _RATIO, latent_steps=LATENT_STEPS, budget=length + 1)

    # The same loop under the bounded law: at most the arm's row budget
    # survives. The registered cap is far above these tiny hops, so the arm at
    # a cap the hops exceed is the one that binds, from hop one.
    for profile, cap in (
        (LONGBENCH_COA_EASY50_BOUNDED, _BUDGET_ROWS),
        (_TIGHT_PROFILE, _TIGHT_ROWS),
    ):
        walked = produce_chain_payload(
            producer,
            tokenizer,
            item,
            semantic_arm=_BOUNDED_ARM,
            family=QWEN_FAMILY,
            profile=profile,
        )
        assert walked.budget_law == walked.result_fields()["budget_law"] == "bounded"
        for previous, scores, hop in zip(
            (None, *walked.hops[:-1]), walked.selector_scores, walked.hops, strict=True
        ):
            carried = 0 if previous is None else len(previous.keep)
            assert hop.prefix_rows == carried
            assert hop.budget == bounded_budget(hop.rows_pre_cut, cap) == min(cap, hop.rows_pre_cut)
            assert hop.budget == hop_budget("bounded", hop.rows_pre_cut, budget_rows=cap)
            assert hop.keep == qwen_w16_keep(
                scores, None, latent_steps=LATENT_STEPS, budget=hop.budget
            )
            assert len(hop.keep) == hop.budget
        binds = [hop.rows_pre_cut > cap for hop in walked.hops]
        assert binds == [cap == _TIGHT_ROWS] * HOPS
        if cap == _TIGHT_ROWS:
            assert [len(hop.keep) for hop in walked.hops] == [cap] * HOPS
            assert walked.payload.rows.shape[0] == cap
        else:
            assert [len(hop.keep) for hop in walked.hops] == [
                hop.rows_pre_cut for hop in walked.hops
            ]
    # Hop one carries nothing under either law, so the two counts are the
    # laws' own rules over the same rows: the ratio's ceiling and the cap's
    # minimum.
    rows_one = production.hops[0].rows_pre_cut
    assert (
        production.hops[0].budget
        == hop_budget("rerank", rows_one, ratio=_RATIO)
        == math.ceil(rows_one / _RATIO)
    )
    assert hop_budget("bounded", rows_one, budget_rows=_BUDGET_ROWS) == min(_BUDGET_ROWS, rows_one)
    assert hop_budget("bounded", rows_one, budget_rows=_TIGHT_ROWS) == _TIGHT_ROWS < rows_one


def test_chain_loop_refuses_a_foreign_profile_and_a_sequence_past_the_window(
    tiny_model: Any,
) -> None:
    """The loop refuses by registered name, before it admits a single request.

    A foreign prompt builder is not this loop's to run, and a hop past the
    registered window never reaches the engine; both refusals name their value.
    """
    producer = fake_producer(tiny_model)
    tokenizer = TinyChainTokenizer(width=4)
    item = chain_item("refuse", 20)

    with pytest.raises(ValueError, match=FANOUTQA_NATURAL_DEV50.benchmark_key):
        produce_chain_payload(
            producer,
            tokenizer,
            item,
            semantic_arm=_ARM,
            family=QWEN_FAMILY,
            profile=FANOUTQA_NATURAL_DEV50,
        )
    with pytest.raises(RuntimeError, match=r"refuse/latent_query_support_rerank_r4/hop 1"):
        produce_chain_payload(
            producer,
            tokenizer,
            item,
            semantic_arm=_ARM,
            family=QWEN_FAMILY,
            profile=replace(CHAIN_PROFILE, max_model_len=64),
        )

    # A scorer and a prepared item that disagree are the same class of
    # mistake, so each row rule refuses by the values its profile registered
    # and never by the Python type that happened to arrive.
    with pytest.raises(ValueError, match=FANOUTQA_NATURAL_DEV50.scorer):
        _FanoutqaRowRule().score(item, "answer", profile=FANOUTQA_NATURAL_DEV50)
    with pytest.raises(ValueError, match=CHAIN_PROFILE.scorer):
        _ChainRowRule().score(object(), "answer", profile=CHAIN_PROFILE)

    # Each arm's band is the prompt that arm renders. A latent seat splices
    # sealed ids into a hop prompt, so the hop wrapper is the whole slack; a
    # text seat renders a wider rewrite prompt that carries the notes as well.
    rule = _ChainRowRule()
    hop_item = chain_item(CHAIN_PROFILE.question_ids[0], 20)
    chunks = hop_item.chunk_tokens
    hop_wrapper, rewrite_wrapper = chain_wrapper_allowances(LONGBENCH_COA_EASY50_TEXT)
    assert hop_wrapper == HOP_WRAPPER_ALLOWANCE
    ceiling = tuple(count + hop_wrapper for count in chunks)
    rule.expected_worker_prompt_tokens(
        hop_item,
        ceiling,
        semantic_arm=_ARM,
        profile=CHAIN_PROFILE,
        family=QWEN_FAMILY,
        channel_fields=None,
    )
    with pytest.raises(ValueError, match="one count per hop from"):
        rule.expected_worker_prompt_tokens(
            hop_item,
            tuple(count + 1 for count in ceiling),
            semantic_arm=_ARM,
            profile=CHAIN_PROFILE,
            family=QWEN_FAMILY,
            channel_fields=None,
        )
    banked = {"chunk_text_tokens": list(chunks)}
    rewrite_ceiling = tuple(
        rewrite_prompt_ceiling(
            count,
            report_ceiling=LONGBENCH_COA_EASY50_TEXT.report_ceiling,
            wrapper=rewrite_wrapper,
        )
        for count in chunks
    )
    rule.expected_worker_prompt_tokens(
        hop_item,
        rewrite_ceiling,
        semantic_arm=_TEXT_ARM,
        profile=LONGBENCH_COA_EASY50_TEXT,
        family=QWEN_FAMILY,
        channel_fields=banked,
    )
    with pytest.raises(ValueError, match="one count per hop from"):
        rule.expected_worker_prompt_tokens(
            hop_item,
            tuple(count + 1 for count in rewrite_ceiling),
            semantic_arm=_TEXT_ARM,
            profile=LONGBENCH_COA_EASY50_TEXT,
            family=QWEN_FAMILY,
            channel_fields=banked,
        )

    # The text band's floor is a ledger count and its ceiling adds a budget in
    # the sender's own tokens, so the two share a scale only while the sender
    # reads the ledger's vocabulary; an unpinned sender tokenizer is refused.
    foreign = replace(
        QWEN_FAMILY,
        profile=replace(
            QWEN_FAMILY.profile,
            physical_arms=tuple(
                replace(arm, sender_tokenizer="Qwen/Qwen3-32B")
                if arm.semantic_arm == _TEXT_ARM
                else arm
                for arm in QWEN_FAMILY.profile.physical_arms
            ),
        ),
    )
    with pytest.raises(ValueError, match="Qwen/Qwen3-32B"):
        rule.expected_worker_prompt_tokens(
            hop_item,
            rewrite_ceiling,
            semantic_arm=_TEXT_ARM,
            profile=LONGBENCH_COA_EASY50_TEXT,
            family=foreign,
            channel_fields=banked,
        )

    # A chain text row walks the whole identity check and is then held to the
    # chain's own four-hop ladder, so a row carrying no hop evidence is
    # refused by the hop roster it lacks and never by a worker roster.
    registered = chain_item(LONGBENCH_COA_EASY50_TEXT.question_ids[0], 20)
    with pytest.raises(RuntimeError, match="one draw count per hop"):
        validate_and_rescore_result(
            _chain_text_row(registered),
            registered,
            TinyChainTokenizer(width=4),
            family=QWEN_FAMILY,
            profile=LONGBENCH_COA_EASY50_TEXT,
        )


def test_payload_geometry_is_keyed_by_layout() -> None:
    """One payload class, two geometries, each named by the layout it serves.

    The flat layout concatenates every worker's block; the chain ships only the
    last hop's, the earlier keeps riding along as provenance of the cut.
    """
    keeps = ((0, 1, 2), (0, 1), (0, 2), (0, 1, 3, 4))
    rows = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    payload = _chain_payload(keeps, rows)
    assert payload.rows_by_worker == (3, 2, 2, 4)
    assert int(payload.rows.shape[0]) == payload.rows_by_worker[-1]

    # The two layouts sign two recipes: a chain terminal block is one hop's
    # rows, not three fan-out worker blocks, so the plan digest differs and the
    # fan-out digest may not move while it does.
    chain_plan = chain_plan_sha256(_ARM, family=QWEN_FAMILY, profile=CHAIN_PROFILE)
    fanout_plan = latent_plan_sha256(_FANOUT_ARM, family=QWEN_FAMILY)
    assert fanout_plan == "1d80f36e77317f9d438fc0602d7e9b283371a284f1a10c2e073e2b3f592731ca"
    assert chain_plan != fanout_plan
    with pytest.raises(ValueError, match=_ARM):
        latent_plan_sha256(_ARM, family=QWEN_FAMILY)
    # The chain body carries both layouts: `payload_layout` is the chain's,
    # the recipe this digest signs, and `physical_payload_layout` is the lane's.
    # The digest is pinned so neither moves in silence.
    assert chain_plan == "4583639b21b9adce536149d33a132447af910bdf436c9c96776f21bbf4a766c2"
    # The payload re-derives its digest by the layout it carries, so a chain
    # payload signed with the fan-out recipe is refused.
    with pytest.raises(ValueError, match="latent-plan signature differs"):
        QwenFlatPayload(
            rows=rows,
            semantic_arm=_ARM,
            latent_plan_sha256=fanout_plan,
            keeps=keeps,
            rows_by_worker=tuple(len(keep) for keep in keeps),
            selected_indices_sha256=selected_indices_sha256(keeps),
            tensor_sha256=tensor_content_sha256(rows),
            family=QWEN_FAMILY,
            profile=CHAIN_PROFILE,
            layout=CHAIN_TERMINAL_LAYOUT,
        )
    # The chain recipe refuses a profile registering another layout, and an arm
    # that profile never registered, each by the registered name.
    with pytest.raises(ValueError, match=FANOUTQA_NATURAL_DEV50.payload_layout):
        chain_plan_sha256(_ARM, family=QWEN_FAMILY, profile=FANOUTQA_NATURAL_DEV50)
    with pytest.raises(ValueError, match="never_registered"):
        chain_plan_sha256("never_registered", family=QWEN_FAMILY, profile=CHAIN_PROFILE)
    # Two rosters own the two halves of an arm. A benchmark that registers no
    # such scientific arm and a lane that implements no model for one are
    # different mistakes, so each is named for what is actually missing.
    thin = replace(
        QWEN_FAMILY,
        profile=replace(
            QWEN_FAMILY.profile,
            physical_arms=tuple(
                arm for arm in QWEN_FAMILY.profile.physical_arms if arm.semantic_arm != _ARM
            ),
        ),
    )
    with pytest.raises(ValueError, match="implements no physical arm"):
        chain_plan_sha256(_ARM, family=thin, profile=CHAIN_PROFILE)

    # A rerank arm carries its ratio and no row budget; a bounded arm the reverse.
    assert registered_selection(_ARM, profile=CHAIN_PROFILE) == ("support", _RATIO, "rerank", None)
    assert registered_selection("latent_query_support_rerank_r16", profile=CHAIN_PROFILE) == (
        "support",
        16,
        "rerank",
        None,
    )
    assert registered_selection(_BOUNDED_ARM, profile=LONGBENCH_COA_EASY50_BOUNDED) == (
        "support",
        None,
        "bounded",
        _BUDGET_ROWS,
    )
    # A fan-out arm names no chain law: on its own profile it resolves with no
    # law and no row budget (its one cut is the ratio's), and a chain profile
    # that carried it would refuse it by name.
    assert registered_selection(_FANOUT_ARM, profile=FANOUTQA_NATURAL_DEV50) == (
        "support",
        2,
        None,
        None,
    )
    fanout_arm = next(arm for arm in FANOUTQA_NATURAL_DEV50.arms if arm.arm_id == _FANOUT_ARM)
    with pytest.raises(ValueError, match="names no chain budget law"):
        registered_selection(
            _FANOUT_ARM, profile=replace(CHAIN_PROFILE, arms=(*CHAIN_PROFILE.arms, fanout_arm))
        )
    with pytest.raises(ValueError, match="never_registered"):
        registered_selection("never_registered", profile=CHAIN_PROFILE)
    with pytest.raises(ValueError, match="text_primary"):
        registered_selection("text_primary", profile=LONGBENCH_COA_EASY50_TEXT)

    # A FanOutQA-shaped block under the chain layout: the concatenation of
    # every hop's keep, which the chain never ships.
    concatenated = torch.arange(11 * 4, dtype=torch.bfloat16).reshape(11, 4)
    with pytest.raises(ValueError, match="chain terminal payload rows"):
        _chain_payload(keeps, concatenated)

    # A chain-shaped block under the flat layout: the last worker's rows only,
    # where the layout requires all three concatenated.
    flat_keeps = ((0, 1, 2), (0, 1), (0, 1, 3, 4))
    flat_rows = torch.arange(4 * 4, dtype=torch.bfloat16).reshape(4, 4)
    with pytest.raises(ValueError, match="row count differs from its worker geometry"):
        QwenFlatPayload(
            rows=flat_rows,
            semantic_arm=_FANOUT_ARM,
            latent_plan_sha256=latent_plan_sha256(_FANOUT_ARM, family=QWEN_FAMILY),
            keeps=flat_keeps,
            rows_by_worker=tuple(len(keep) for keep in flat_keeps),
            selected_indices_sha256=selected_indices_sha256(flat_keeps),
            tensor_sha256=tensor_content_sha256(flat_rows),
            family=QWEN_FAMILY,
            profile=FANOUTQA_NATURAL_DEV50,
            layout=FANOUTQA_NATURAL_DEV50.payload_layout,
        )


# --- The four sequential note rewrites of the text arms.

_QID = LONGBENCH_COA_EASY50_TEXT.question_ids[0]


_ARM_TEXT = "text_medium"


_ITEM = chain_item(_QID, 10)


class _LedgerTokenizer(FakeTokenizer):
    """The profile-pinned tokenizer, whose reading of the ledger is its own.

    A sender arm opens its own checkpoint's tokenizer, so this fake reads the
    sealed ids differently from the sender fake beside it.
    """

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        """Return the ids read as the ledger reads them."""
        del kwargs
        return " ".join(f"L{int(token)}" for token in tokens)


#: The pinned tokenizer the chain decodes its sealed parts through.
_LEDGER = _LedgerTokenizer(width=2)


class _ChainSender:
    """Replay one scripted draw per call and retain the prompt it consumed.

    ``prompt_ids`` scripts a backend that consumed other ids than the run
    rendered; ``closes`` off writes a draw that never spells the closer.
    """

    def __init__(
        self,
        script: Sequence[str],
        *,
        prompt_ids: Sequence[int] | None = None,
        closes: bool = True,
    ) -> None:
        self.script = list(script)
        self.prompt_ids = prompt_ids
        self.closes = closes
        self.tokenizer = FakeTokenizer()
        self.prompts: list[str] = []
        self.requests: list[Any] = []
        self.clock = 0.0

    def _completion(
        self, request: Any, prompt_ids: Sequence[int], text: str, ids: Sequence[int]
    ) -> Any:
        return SimpleNamespace(
            request_id=request.request_id,
            prompt_token_ids=tuple(int(token) for token in prompt_ids),
            text=text,
            n_tokens=len(ids),
            token_ids=tuple(ids),
            finish_reason="stop",
            num_cached_tokens=0,
            queued_ts=0.0,
            scheduled_ts=1.0,
            first_token_ts=2.0,
        )

    def decode_text_full(self, prompts: Sequence[str], requests: Sequence[Any]) -> list[Any]:
        """Answer the one rewrite request of one hop draw."""
        if len(prompts) != 1 or len(requests) != 1:
            raise AssertionError("a chain hop draws exactly one rewrite")
        self.prompts.append(prompts[0])
        self.requests.append(requests[0])
        self.clock += 1.0
        ids = (
            (
                QWEN_FAMILY.think_open_token_ids[0],
                700_001,
                QWEN_FAMILY.think_close_token_ids[0],
                700_002,
                QWEN_FAMILY.stop_token_ids[0],
            )
            if self.closes
            else (700_001, QWEN_FAMILY.stop_token_ids[0])
        )
        consumed = self.tokenizer.encode(prompts[0]) if self.prompt_ids is None else self.prompt_ids
        return [self._completion(requests[0], consumed, self.script.pop(0), ids)]

    def decode_token_ids_full(
        self, prompts: Sequence[Sequence[int]], requests: Sequence[Any]
    ) -> list[Any]:
        """Answer an injected closer with the empty continuation of the prefilled case."""
        if self.closes:
            raise AssertionError("the scripted chain closes every draw")
        self.clock += 1.0
        return [self._completion(requests[0], prompts[0], "", (QWEN_FAMILY.stop_token_ids[0],))]


def _sender(script: Sequence[str], monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> _ChainSender:
    """Build a scripted sender whose own draw counter is the module clock."""
    made = _ChainSender(script, **kwargs)
    monkeypatch.setattr(qwen_chain_text.time, "perf_counter", lambda: made.clock)
    return made


def _run(
    sender: _ChainSender,
    *,
    family: RouteFamily = QWEN_FAMILY,
    profile: Any = LONGBENCH_COA_EASY50_TEXT,
) -> QwenNotesChainBundle:
    """Rewrite the four hops on one scripted sender."""
    return generate_notes_chain(
        cast(Any, sender),
        sender.tokenizer,
        _ITEM,
        qid=_QID,
        semantic_arm=_ARM_TEXT,
        family=family,
        decoder=tokenizer_decoder(sender.tokenizer),
        profile=profile,
        ledger_tokenizer=_LEDGER,
    )


def _part(hop: int) -> str:
    """Return the body hop ``hop`` reads: the ledger's decode of its chunk."""
    return _LEDGER.decode(_ITEM.chunks[hop - 1])


def test_four_rewrites_carry_the_notes_forward_under_the_redraw_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _sender(REDRAW_SCRIPT, monkeypatch)
    bundle = _run(sender)
    assert bundle.notes == ("note 1", "note 2", "note 3", "note 4")
    assert bundle.ticket == "note 4"
    assert bundle.seed_tags == ("s0", "s1", "s0", "s0")
    assert bundle.draws_by_hop == (1, 2, 1, 1)
    assert bundle.seeds == tuple(
        LONGBENCH_COA_EASY50_TEXT.report_seeds(_QID, tag)[hop]
        for hop, tag in enumerate(bundle.seed_tags)
    )
    assert not bundle.report_failed and bundle.failed_hops == ()
    assert bundle.thinking_closed == (True,) * HOPS
    assert bundle.injected_by_hop == (False,) * HOPS
    assert bundle.tokens_by_hop == (5,) * HOPS
    # The count is over the ids the sender reported it consumed, one hop prompt
    # each: the second draw of hop two reads the same prompt as the first.
    hop_prompts = (sender.prompts[0], *sender.prompts[2:])
    assert bundle.prompt_tokens_by_hop == tuple(
        len(sender.tokenizer.encode(prompt)) for prompt in hop_prompts
    )
    # The part every hop read is banked as the text the sender saw, so a
    # normalizing decode is visible in the row. The reading is the pinned
    # tokenizer's on every arm, comparable to the ledger cut behind the chunk.
    parts = tuple(_part(hop) for hop in range(1, HOPS + 1))
    assert bundle.chunk_text_sha256 == tuple(
        hashlib.sha256(part.encode("utf-8")).hexdigest() for part in parts
    )
    assert bundle.chunk_text_tokens == tuple(len(_LEDGER.encode(part)) for part in parts)
    assert bundle.chunk_text_tokens != tuple(len(sender.tokenizer.encode(part)) for part in parts)
    # Hop t reads the notes hop t-1 shipped and part t, and no other part.
    assert NO_NOTES in sender.prompts[0] and _part(1) in sender.prompts[0]
    assert "note 1" in sender.prompts[1] and "note 1" in sender.prompts[2]
    assert _part(2) in sender.prompts[1] and _part(1) not in sender.prompts[1]
    assert "note 2" in sender.prompts[3] and _part(3) in sender.prompts[3]
    assert "note 3" in sender.prompts[4] and _part(4) in sender.prompts[4]
    ids = [request.request_id for request in sender.requests]
    assert ids[1] == f"{_QID}:{_ARM_TEXT}:rewrite:h2:s0"
    assert [request.member_index for request in sender.requests] == [0, 1, 1, 2, 3]
    assert bundle.generation_s == float(HOPS) and bundle.redraw_wall_s == 1.0


def test_an_exhausted_hop_carries_the_last_good_notes_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _sender(
        (
            "<think>x</think>note 1",
            *("<think>x</think>" for _ in QWEN_REPORT_SEED_TAGS),
            "<think>x</think>note 3",
            "<think>x</think>note 4",
        ),
        monkeypatch,
    )
    bundle = _run(sender)
    assert bundle.notes == ("note 1", "", "note 3", "note 4")
    assert bundle.draws_by_hop == (1, len(QWEN_REPORT_SEED_TAGS), 1, 1)
    assert bundle.seed_tags[1] == QWEN_REPORT_SEED_TAGS[-1]
    # The hop after an exhausted one reads the last notes that shipped, so the
    # chain loses one part and not every part before it.
    assert "note 1" in sender.prompts[4] and NO_NOTES not in sender.prompts[4]
    # A chain that lost a part is not a four-hop measurement, whatever hop
    # four wrote, so the receiver quarantines it on the same flag.
    assert bundle.failed_hops == (2,) and bundle.report_failed
    assert bundle.ticket == "note 4"
    # The notes each prompt actually carried, so the survivor a hop read after
    # a failed hop is auditable without rendering the four prompts again.
    assert bundle.notes_read_by_hop == ("", "note 1", "note 1", "note 3")


def test_the_closer_is_priced_against_the_consumed_prompt() -> None:
    # The continuation is priced against the ids the engine consumed, which is
    # the only count that bounds a rewrite prompt: the chain's own worker
    # budget is a chunk floor and says nothing about the notes above it.
    tail = len(QWEN_FAMILY.think_close_token_ids) + QWEN_FAMILY.closing_token_budget
    room = LONGBENCH_COA_EASY50_TEXT.max_model_len - LONGBENCH_COA_EASY50_TEXT.report_ceiling - tail
    for label, prompt_tokens, affordable in (
        ("inside the window", room, True),
        ("one token past it", room + 1, False),
    ):
        drawn = cast(
            Any,
            SimpleNamespace(
                prompt_token_ids=[1] * prompt_tokens,
                token_ids=[1] * LONGBENCH_COA_EASY50_TEXT.report_ceiling,
            ),
        )
        assert (
            _closer_is_affordable(drawn, family=QWEN_FAMILY, profile=LONGBENCH_COA_EASY50_TEXT)
            is affordable
        ), label


def test_result_fields_ship_one_ticket_and_a_failed_last_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _sender(
        (
            "<think>x</think>note 1",
            "<think>x</think>note 2",
            "<think>x</think>note 3",
            *("<think>x</think>" for _ in QWEN_REPORT_SEED_TAGS),
        ),
        monkeypatch,
    )
    bundle = _run(sender)
    assert bundle.notes[-1] == "" and bundle.report_failed
    assert bundle.draws_by_hop == (1, 1, 1, len(QWEN_REPORT_SEED_TAGS))
    fields = bundle.result_fields()
    assert fields["reports"] == [""]
    assert fields["notes_by_hop"] == ["note 1", "note 2", "note 3", ""]
    assert fields["draws_n"] == 6 and fields["draws_by_hop"] == [1, 1, 1, 3]
    assert fields["report_seed_tags_by_hop"] == ["s0", "s0", "s0", "s2"]
    assert fields["report_failed"] is True and fields["failed_hops"] == [HOPS]
    assert fields["report_failure"] == f"hops [{HOPS}] shipped empty notes after 3 draws"
    assert fields["report_tokens"] == 5 * HOPS
    assert len(cast(list[object], fields["report_prompt_token_sha256_by_hop"])) == HOPS
    assert len(cast(list[object], fields["chunk_text_sha256"])) == HOPS
    assert fields["chunk_text_tokens"] == [
        len(_LEDGER.encode(_part(hop))) for hop in range(1, HOPS + 1)
    ]
    assert fields["notes_read_by_hop"] == ["", "note 1", "note 2", "note 3"]
    # The two backend clocks the FanOutQA bundle banks, read over the four
    # shipped draws, so one report reader serves both topologies.
    assert fields["report_queue_s_mean"] == 1.0 and fields["report_ttft_s_mean"] == 2.0


def test_a_bundle_that_differs_from_its_own_evidence_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _run(
        _sender(tuple(f"<think>x</think>note {hop}" for hop in range(1, HOPS + 1)), monkeypatch)
    )
    for message, changes in (
        ("hop rows", {"notes": bundle.notes[:-1]}),
        ("hop rows", {"chunk_text_tokens": (1,)}),
        ("seed tag", {"seed_tags": ("s2",) * HOPS}),
        ("wall clock is invalid", {"redraw_wall_s": -1.0}),
        ("cannot carry redraw", {"redraw_wall_s": 3.0}),
        (
            "must carry redraw",
            {"draws_by_hop": (2, 1, 1, 1), "seed_tags": ("s1", "s0", "s0", "s0")},
        ),
        # A redraw faster than the banked precision is refused here, because
        # the row it would bank rounds to zero and its own reader refuses that.
        (
            "must carry redraw",
            {
                "draws_by_hop": (2, 1, 1, 1),
                "seed_tags": ("s1", "s0", "s0", "s0"),
                "redraw_wall_s": 0.00004,
            },
        ),
        ("post-think reconstruction", {"notes": ("other",) * HOPS}),
        ("closure fields", {"thinking_closed": (False,) * HOPS}),
        ("token counts", {"tokens_by_hop": (1,) * HOPS}),
        ("exhausted hops", {"failed_hops": (1,)}),
        ("notes each hop read", {"notes_read_by_hop": ("", "", "", "")}),
        ("failure flag", {"report_failed": True}),
    ):
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(bundle, **changes)


def test_a_profile_the_chain_cannot_run_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fan-out profile is refused by the same rule as a chain profile whose
    # hop count or brief moved: the registered value, named.
    for message, profile in (
        ("workers_per_item", dataclasses.replace(LONGBENCH_COA_EASY50_TEXT, workers_per_item=3)),
        (
            "other-builder-v1",
            dataclasses.replace(LONGBENCH_COA_EASY50_TEXT, prompt_builder="other-builder-v1"),
        ),
        (FANOUTQA_NATURAL_DEV50.benchmark_key, FANOUTQA_NATURAL_DEV50),
    ):
        sender = _sender(("<think>x</think>note 1",) * HOPS, monkeypatch)
        with pytest.raises(ValueError, match=message):
            _run(sender, profile=profile)
        # Refused by name before the first draw, never three hops in.
        assert sender.prompts == []


def test_a_backend_that_consumed_another_prompt_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _sender(("<think>x</think>note 1",), monkeypatch, prompt_ids=(11, 12, 13))
    with pytest.raises(RuntimeError, match="hop 1"):
        _run(sender)
