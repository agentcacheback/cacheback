"""The LongBench v2 chain benchmark: its panel, chunks, prompt bodies, sealers, and identity."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import inspect
import io
import json
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from tests.longbench_coa_support import (
    CHAIN_PROFILE,
    COMMIT,
    PROFILE_KEY,
    FakeTokenizer,
    chain_item,
    raw_row,
)

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, FANOUTQA_NATURAL_DEV50_LATENT
from rcc.benchmarks.fanoutqa import prompts as fanoutqa_prompts
from rcc.benchmarks.fanoutqa.panel import Question
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.longbench_v2 import (
    EASY50_ORDER,
    LONGBENCH_COA_EASY50_BOUNDED,
    LONGBENCH_COA_EASY50_RERANK,
    LONGBENCH_COA_EASY50_TEXT,
    UNMINTED_SENTINEL,
    build_cli,
    prompts,
    scoring,
)
from rcc.benchmarks.longbench_v2 import prompts as longbench_prompts
from rcc.benchmarks.longbench_v2.chunking import (
    LEDGER_FIELDS,
    SOURCE_CHUNK_CEILING,
    balanced_four_chunks,
    chunk_token_counts,
    read_chunk_ledger,
    validate_ledger,
)
from rcc.benchmarks.longbench_v2.data import build_item
from rcc.benchmarks.longbench_v2.geometry import (
    ANSWER_TEMPLATE_ALLOWANCE,
    HOP_WRAPPER_ALLOWANCE,
    NO_CONTEXT_PROMPT_ALLOWANCE,
    QUESTION_CHOICES_MAX_TOKENS,
    RATIO_TABLE_LAW,
    REWRITE_WRAPPER_ALLOWANCE,
    REWRITE_WRAPPER_ALLOWANCE_CONDENSE,
    chain_geometry,
    rewrite_prompt_ceiling,
    rewrite_wrapper_allowance,
    text_sender_prompt,
)
from rcc.benchmarks.longbench_v2.nemotron import (
    NEMOTRON_LONGBENCH_BOUNDED,
    NEMOTRON_LONGBENCH_RERANK,
    NEMOTRON_LONGBENCH_TEXT,
)
from rcc.benchmarks.longbench_v2.panel import (
    EASY50,
    PANELS,
    RAW_SOURCE_SHA256,
    PanelSelection,
    execution_order,
    hamilton_quotas,
    panel_records,
    read_manifest,
    sample_hash,
    select_panel,
    strata_counts,
    validate_selection,
    write_manifest,
)
from rcc.benchmarks.longbench_v2.prepare import (
    _PreparedUnpickler,
    _validate_items,
    load_prepared_panel,
    prepared_paths,
    seal_prepared_panel,
    seal_source_audit,
    validate_source_audit,
)
from rcc.benchmarks.longbench_v2.registration import (
    PAYLOAD_LAYOUT,
    answer_seed_namespace,
    bounded_rows,
    build_config_fingerprint,
    frozen_geometry,
    panel_registration_sha256,
    registration_body,
    rerank_ratios,
)
from rcc.benchmarks.longbench_v2.scoring import EXTRACTION_RULE
from rcc.benchmarks.longbench_v2.source_bundle import (
    build_source_bundle,
    bundle_rows,
    logical_fingerprint,
    unminted_digests,
    validate_source_bundle,
)
from rcc.hardware.fleet import FleetPlacement
from rcc.hardware.placements import (
    CHAIN_ARM_PLACEMENTS,
    LATENT_RATIOS,
    PLACEMENT_TABLES,
    SEMANTIC_ARM_PLACEMENTS,
    bind_policy_placements,
    placement_for_semantic_arm,
    placement_table,
)
from rcc.models.nemotron import NEMOTRON_FAMILY
from rcc.models.qwen import QWEN_FAMILY
from rcc.models.qwen import prompts as qwen_prompts
from rcc.models.qwen.prompts import manager_prompt_record
from rcc.run import plan as plan_module
from rcc.run import split_driver
from rcc.run.config import load_run_config
from rcc.run.plan import (
    ResolutionContext,
    load_and_resolve,
    resolve_plan,
)
from rcc.run.qwen.adapter import build_adapter
from rcc.topologies import chain
from rcc.topologies.chain import BOUNDED_ROW_BUDGETS, CHAIN_T4, CHAIN_TOPOLOGY_KEY
from rcc.topologies.chain.layout import CHAIN_TERMINAL_LAYOUT
from rcc.topologies.fanout import FANOUT_M3

# --- The panel: registration, chunks, prompts, geometry.

#: The source fields every profile over the easy fifty carries byte for byte.
_SOURCE_DIGESTS = (
    "source_logical_fingerprint",
    "source_manifest_sha256",
    "source_archive_sha256",
)


_BUNDLE_DIGESTS = (*_SOURCE_DIGESTS, "question_index_sha256")


_DIGEST_FIELDS = (*_SOURCE_DIGESTS, "prepared_artifact_sha256")


_RERANK_LADDER = (4, 8, 16, 32, 64, 128)


def test_profile_registration_reproduces_from_live_code():
    # The three Qwen profiles read one panel object under three config keys:
    # one seed namespace, one ledger, one set of four source fields.
    rerank, text = LONGBENCH_COA_EASY50_RERANK, LONGBENCH_COA_EASY50_TEXT
    bounded = LONGBENCH_COA_EASY50_BOUNDED
    for profile in (rerank, text, bounded):
        assert PANELS[profile.benchmark_key] is EASY50
        assert profile.workers_per_item == chain.CHAIN_T4.workers_per_item == 4
        assert profile.worker_prompt_tokens == SOURCE_CHUNK_CEILING
        assert (profile.latent_steps, profile.span_width) == (40, 16)
        assert profile.question_index_sha256 == RAW_SOURCE_SHA256
        assert profile.question_ids == EASY50_ORDER and len(profile.question_ids) == 50
        assert profile.answer_seed_namespace == answer_seed_namespace(EASY50)
        assert [getattr(profile, name) for name in _BUNDLE_DIGESTS] == [
            getattr(rerank, name) for name in _BUNDLE_DIGESTS
        ]
        # The registration digest is a pure function of the registration.
        assert panel_registration_sha256(profile) == profile.panel_registration_sha256
        assert profile.scientific_identity_hash != FANOUTQA_NATURAL_DEV50.scientific_identity_hash
        # The three receiver seeds of every item and arm are distinct on every
        # sealed profile, which the receiver relies on rather than rechecking.
        for qid in profile.question_ids:
            for arm in profile.arms:
                seeds = profile.answer_seeds(qid, arm.arm_id)
                assert len(set(seeds)) == len(profile.sample_tags) == 3, (profile.profile_id, qid)
        assert len(profile.report_seeds(profile.question_ids[0], "s0")) == 4, "one seed per hop"
    for sealed in (FANOUTQA_NATURAL_DEV50,):
        for qid in sealed.question_ids:
            for arm in sealed.arms:
                assert len(set(sealed.answer_seeds(qid, arm.arm_id))) == 3, (sealed.profile_id, qid)
    assert chain.CHAIN_T4.topology_id == "chain-t4-v1"
    assert len({p.scientific_identity_hash for p in (rerank, text, bounded)}) == 3
    # The two minted profiles: every digest is the shape its own validator
    # compares, and the build fingerprint reproduces the digest recorded in the profile.
    for sealed in (rerank, text, bounded):
        assert re.fullmatch(r"[0-9a-f]{40}", sealed.source_commit)
        assert build_cli.unminted_fields(sealed) == ()
        assert all(len(getattr(sealed, name)) == 64 for name in _DIGEST_FIELDS)
        assert len(sealed.source_audit_fingerprint) == 16
        assert build_config_fingerprint(sealed) == sealed.prepared_config_fingerprint
    # Six arms and nothing else, in ratio order, each naming the rerank law and
    # no row budget; the key enters an arm's dict only when it is set.
    assert rerank.benchmark_key == "longbench-v2-coa-easy50-rerank-sealed"
    assert rerank.report_seed_base == 20_260_912 and rerank.ratios == _RERANK_LADDER
    assert [
        (arm.arm_id, arm.retention_ratio, arm.to_dict()["budget_law"]) for arm in rerank.arms
    ] == [(f"latent_query_support_rerank_r{ratio}", ratio, "rerank") for ratio in _RERANK_LADDER]
    assert all(
        arm.budget_rows is None and "budget_rows" not in arm.to_dict() for arm in rerank.arms
    )
    assert rerank.sealed_qwen_policies == tuple(
        (f"latent_query_support_rerank_r{ratio}", f"qwen3_8b_rerank_r{ratio}_w16_support")
        for ratio in _RERANK_LADDER
    )
    assert rerank.scientific_identity_hash == (
        "a502f3660e9ddf1d130d7e9aa35d9c1d1d299df959051886b6ed0cc98b07ec31"
    )
    # The text profile: the floor and the three text arms in the shared order,
    # no latent arm and so no ratio ladder. The answer ceiling is where every
    # other profile has it, so a text row is read against the latent arms.
    assert text.benchmark_key == "longbench-v2-coa-easy50-text-sealed"
    assert text.report_seed_base == 20_260_913 and text.ratios == ()
    assert text.arms == FANOUTQA_NATURAL_DEV50.arms[:4]
    text_ids = ("issue_only", "text_primary", "text_medium", "text_small")
    assert tuple(arm.arm_id for arm in text.arms) == text_ids
    assert all("budget_law" not in arm.to_dict() for arm in text.arms)
    assert text.sealed_qwen_policies == FANOUTQA_NATURAL_DEV50.sealed_qwen_policies[:4]
    assert (text.report_ceiling, text.report_presence_penalty) == (16_000, 1.5)
    assert (rerank.report_ceiling, rerank.report_presence_penalty) == (24_000, 0.0)
    assert text.to_dict()["report_presence_penalty"] == 1.5
    assert "report_presence_penalty" not in rerank.to_dict()
    assert text.scientific_identity_hash == (
        "384de351386186c014666f3adef7fbf05b622feba3c8acbbb29ac65df9f8e6d7"
    )
    # The bounded profile: five arms, largest budget first, each naming the
    # bounded law and a row budget and no ratio, so the profile reports on no
    # ratio ladder.
    assert bounded.benchmark_key == "longbench-v2-coa-easy50-bounded-sealed"
    assert bounded.report_seed_base == 20_260_913 and bounded.ratios == ()
    assert (bounded.report_ceiling, bounded.report_presence_penalty) == (24_000, 0.0)
    assert bounded.arm_profiles == ("longbench-coa-bounded-6arm-v1",)
    assert [
        (arm.arm_id, arm.retention_ratio, arm.to_dict()["budget_law"], arm.to_dict()["budget_rows"])
        for arm in bounded.arms
    ] == [
        (f"latent_query_support_bounded_b{rows}", None, "bounded", rows)
        for rows in chain.BOUNDED_ROW_BUDGETS
    ]
    assert bounded.sealed_qwen_policies == tuple(
        (f"latent_query_support_bounded_b{rows}", f"qwen3_8b_bounded_b{rows}_w16_support")
        for rows in chain.BOUNDED_ROW_BUDGETS
    )
    # Every sealed field reproduces from live code.
    assert build_cli.unminted_fields(bounded) == ()
    assert build_config_fingerprint(bounded) == bounded.prepared_config_fingerprint
    assert bounded.scientific_identity_hash == (
        "b7590b1398e2298ba94edf5c15fc3afa6592ce7fd8475459fb1dcd905d05a382"
    )


def test_execution_order_is_the_proportional_interleave_of_the_frozen_manifest(tmp_path):
    panel = EASY50
    assert hashlib.sha256(panel.manifest_path.read_bytes()).hexdigest() == panel.manifest_sha256
    rows = read_manifest(panel.manifest_path)
    assert len(rows) == panel.size == 50
    assert [int(row["sample_order"]) for row in rows] == list(range(1, 51))
    assert strata_counts(rows) == panel.quotas == {3: 25, 4: 16, 5: 9}
    assert {row["difficulty"] for row in rows} == {"easy"}
    bands = {row["qid"]: int(row["selection_length_band_50k"]) for row in rows}
    order = execution_order(bands, [row["qid"] for row in rows])
    # The same interleave on every profile over the panel, 4/3/1 in the first eight.
    assert order == EASY50_ORDER == LONGBENCH_COA_EASY50_RERANK.question_ids
    assert (
        order == LONGBENCH_COA_EASY50_TEXT.question_ids == LONGBENCH_COA_EASY50_BOUNDED.question_ids
    )
    assert Counter(bands[qid] for qid in order[:8]) == {3: 4, 4: 3, 5: 1}
    assert Counter(bands[qid] for qid in order) == {3: 25, 4: 16, 5: 9}
    assert set(order).isdisjoint(panel.excluded_qids)
    for row in rows:
        expected = sample_hash(
            RAW_SOURCE_SHA256,
            salt=panel.sample_salt,
            band=bands[row["qid"]],
            domain=row["domain"],
            difficulty=row["difficulty"],
            qid=row["qid"],
        )
        assert row["selection_hash"] == expected, row["qid"]
    copy = tmp_path / "easy50.csv"
    write_manifest(copy, rows)
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == panel.manifest_sha256
    (tmp_path / "empty.csv").write_text("qid,other\n")
    with pytest.raises(RuntimeError, match="columns"):
        read_manifest(tmp_path / "empty.csv")


def test_selection_rule_picks_one_row_per_context_then_hamilton():
    def row(qid: str, context: str, difficulty: str = "easy") -> dict[str, str]:
        return {
            "_id": qid,
            "domain": "d",
            "sub_domain": "s",
            "difficulty": difficulty,
            "context": context,
        }

    rows = [
        row("q1", "shared"),
        row("q2", "shared"),
        row("q3", "shared"),
        row("q4", "solo", "hard"),
        row("q5", "other"),
        row("q6", "third"),
        row("q7", "band4"),
        row("q8", "band4"),
        row("q9", "band9"),
    ]
    tokens = {f"q{i}": 120_000 for i in range(1, 7)} | {"q7": 160_000, "q8": 160_000, "q9": 440_000}
    records = panel_records(rows, tokens, salt="test-salt")
    assert [record.band for record in records] == [3, 3, 3, 3, 3, 3, 4, 4, 9]
    # A panel admitting every difficulty.
    panel = dataclasses.replace(EASY50, quotas={3: 3, 4: 1}, difficulties=None)
    selection = select_panel(records, panel)
    shared = sorted((r.selection_hash, r.qid) for r in records[:3])
    band4 = sorted((r.selection_hash, r.qid) for r in records[6:8])
    assert set(selection.excluded) == {qid for _hash, qid in shared[1:]} | {band4[1][1]}
    assert shared[0][1] in {r.qid for r in selection.selected}
    assert selection.population == {3: 6, 4: 2}
    assert selection.survivors == {3: 4, 4: 1}
    assert [r.band for r in selection.selected] == [3, 3, 3, 4]
    hashes = [r.selection_hash for r in selection.selected[:3]]
    assert hashes == sorted(hashes), "registration order is band, then hash"
    assert len({r.context_sha256 for r in selection.selected}) == 4
    assert hamilton_quotas([("a", "e"), ("a", "e"), ("a", "e"), ("b", "h")], 2) == {
        ("a", "e"): 2,
        ("b", "h"): 0,
    }
    with pytest.raises(RuntimeError, match="population"):
        validate_selection(selection, panel)
    with pytest.raises(ValueError, match="repeat"):
        panel_records([row("q1", "x"), row("q1", "y")], {"q1": 120_000}, salt="test-salt")
    # An easy-only panel admits the easy rows before the one-per-context step,
    # so the hard row neither competes for its context nor counts as excluded.
    easy = select_panel(records, dataclasses.replace(panel, difficulties=("easy",)))
    assert "q4" not in {r.qid for r in easy.selected} and "q4" not in easy.excluded
    assert easy.population == {3: 5, 4: 2} and easy.survivors == {3: 3, 4: 1}


def test_chunker_cuts_at_whole_token_quartiles_and_reconstructs():
    tokenizer = FakeTokenizer(width=3)
    context = "".join(chr(97 + (i * 7) % 26) for i in range(998)) + "é" + "z"
    chunks, rows = balanced_four_chunks(context, tokenizer)
    assert "".join(chunks) == context
    assert len(rows) == 4 and all(chunk for chunk in chunks)
    tokens = len(context) // 3 + (len(context) % 3 > 0)
    boundaries = [(t * tokens) // 4 for t in (1, 2, 3)]
    assert [row["whole_token_end"] for row in rows[:3]] == boundaries
    assert [row["char_end"] for row in rows[:3]] == [3 * b for b in boundaries]
    assert rows[-1]["char_end"] == len(context)
    assert rows[-1]["byte_end"] == len(context.encode("utf-8")) > rows[-1]["char_end"]
    assert [row["char_start"] for row in rows] == [0, *(row["char_end"] for row in rows[:3])]
    assert [row["standalone_qwen_source_tokens"] for row in rows] == [
        len(tokenizer.encode(chunk)) for chunk in chunks
    ]
    assert [row["terminal"] for row in rows] == [False, False, False, True]
    assert [row["boundary_kind"] for row in rows] == ["qwen_token"] * 3 + ["terminal"]
    assert all(row["cumulative_whole_source_tokens"] == row["whole_token_end"] for row in rows)
    assert rows[0]["chunk_sha256"] == hashlib.sha256(chunks[0].encode("utf-8")).hexdigest()
    owned = {"qid", "sample_order", "source_tokens"}
    assert all(set(row) | owned == set(LEDGER_FIELDS) for row in rows)
    with pytest.raises(ValueError, match="exceeds"):
        balanced_four_chunks("a" * (4 * SOURCE_CHUNK_CEILING + 4), FakeTokenizer(width=1))
    with pytest.raises(ValueError, match="strictly increasing"):
        balanced_four_chunks("ab", FakeTokenizer(width=1))


def test_frozen_ledger_is_four_quartile_chunks_per_item():
    # The easy fifty's ledger is the one every profile reads.
    panel = EASY50
    assert hashlib.sha256(panel.ledger_path.read_bytes()).hexdigest() == panel.ledger_sha256
    rows = read_chunk_ledger(panel.ledger_path)
    manifest = read_manifest(panel.manifest_path)
    assert len(rows) == 200
    qids = [row["qid"] for row in manifest]
    validate_ledger(rows, qids)
    counts = chunk_token_counts(rows)
    flat = [count for chunk_tokens in counts.values() for count in chunk_tokens]
    assert (min(flat), max(flat)) == (25_042, 61_466)
    assert all(count <= SOURCE_CHUNK_CEILING for count in flat)
    source_tokens = {row["qid"]: int(row["source_tokens"]) for row in manifest}
    by_qid = {row["qid"]: row for row in rows if row["terminal"]}
    assert all(by_qid[qid]["whole_token_end"] == source_tokens[qid] for qid in counts)
    assert all(
        [int(row[f"worker_{k}_novel_tokens"]) for k in (1, 2, 3, 4)] == list(counts[row["qid"]])
        for row in manifest
    )
    contiguity = [dict(row) for row in rows]
    contiguity[1]["byte_start"] += 1
    with pytest.raises(RuntimeError, match="contiguous"):
        validate_ledger(contiguity, qids)
    # A contiguous ledger cut off the quartile is refused too.
    shifted = [dict(row) for row in rows]
    shifted[0]["whole_token_end"] += 1
    shifted[0]["cumulative_whole_source_tokens"] += 1
    shifted[1]["whole_token_start"] += 1
    with pytest.raises(RuntimeError, match="quartile"):
        validate_ledger(shifted, qids)


def test_extraction_and_scores_follow_the_official_rule():
    cases = {
        "The correct answer is (B).": "B",
        "**The correct answer is (C)**": "C",
        "So The correct answer is D": "D",
        "I think The correct answer is B. Later: The correct answer is (C)": "C",
        "The correct answer is (E)": None,
        "answer: A": None,
        "": None,
    }
    for text, expected in cases.items():
        assert scoring.extract_choice(text) == expected, text
    assert scoring.score_choice("B", "The correct answer is (B)") == {
        "correct": 1.0,
        "answered": 1.0,
    }
    assert scoring.score_choice("B", "The correct answer is (A)") == {
        "correct": 0.0,
        "answered": 1.0,
    }
    assert scoring.score_choice("B", "no idea") == {"correct": 0.0, "answered": 0.0}
    with pytest.raises(ValueError, match="gold"):
        scoring.score_choice("E", "x")
    samples = [{"correct": 1.0}, {"correct": 0.0}, {"correct": 1.0}]
    assert scoring.item_mean(samples) == 0.6667
    items = {"a": 1.0, "b": 0.0, "c": 0.5, "d": 1.0}
    assert scoring.macro_mean(items) == 0.625
    assert scoring.stratum_means(items, {"a": 3, "b": 3, "c": 4, "d": 5}) == {
        3: 0.5,
        4: 0.5,
        5: 1.0,
    }
    assert scoring.majority_floor("A" * 17 + "B" * 27 + "C" * 37 + "D" * 19) == 0.37
    assert scoring.CHANCE == 0.25


def test_geometry_table_on_the_frozen_ledger():
    # The re-ranked ladder adds the carried count whole, so r4 hop four is the
    # worst hop; no item is N/A.
    rerank = frozen_geometry(LONGBENCH_COA_EASY50_RERANK)
    assert rerank["rerank_ratios"] == {
        "4": {"worst_hop_rows": 109_320, "worst_terminal_request": 91_528, "na_items": 0},
        "8": {"worst_hop_rows": 85_896, "worst_terminal_request": 60_295, "na_items": 0},
        "16": {"worst_hop_rows": 74_184, "worst_terminal_request": 44_679, "na_items": 0},
        "32": {"worst_hop_rows": 68_328, "worst_terminal_request": 36_871, "na_items": 0},
        "64": {"worst_hop_rows": 65_400, "worst_terminal_request": 32_967, "na_items": 0},
        "128": {"worst_hop_rows": 63_936, "worst_terminal_request": 31_015, "na_items": 0},
    }
    # The sealed `ratios` table is priced on the plain per-hop re-cut,
    # `ceil(rows / ratio)`.
    assert RATIO_TABLE_LAW == "coa" and RATIO_TABLE_LAW not in chain.BUDGET_LAWS
    assert rerank["ratios"] == {
        "4": {"worst_hop_rows": 82_967, "worst_terminal_request": 49_801, "na_items": 0},
        "8": {"worst_hop_rows": 71_376, "worst_terminal_request": 37_981, "na_items": 0},
        "16": {"worst_hop_rows": 66_633, "worst_terminal_request": 33_224, "na_items": 0},
        "32": {"worst_hop_rows": 64_485, "worst_terminal_request": 31_075, "na_items": 0},
        "64": {"worst_hop_rows": 63_461, "worst_terminal_request": 30_051, "na_items": 0},
        "128": {"worst_hop_rows": 62_961, "worst_terminal_request": 29_551, "na_items": 0},
    }
    worst = max(max(row.values()) for row in rerank["rerank_ratios"].values())
    assert worst == 109_320 <= rerank["window"] == 131_072
    assert (rerank["hop_wrapper_allowance"], rerank["answer_prompt_allowance"]) == (963, 963)
    assert rerank["controls"] == {"full_fits": 0, "direct_fits": 1}
    assert rerank["chunk_tokens"] == {"min": 25_042, "max": 61_466}
    # The easy panel prices its text sender on the condensing brief's wider wrapper.
    assert rerank["text_sender_context"] == 114_685 == 61_466 + 24_000 + 1_123 + 24_000 + 4_096
    assert REWRITE_WRAPPER_ALLOWANCE_CONDENSE == 883 + 240 > REWRITE_WRAPPER_ALLOWANCE
    assert "bounded_rows" not in rerank
    # The bounded ladder on the same panel: the cap binds from hop two on every
    # item, so the worst hop and terminal request are the rung plus allowances.
    bounded = frozen_geometry(LONGBENCH_COA_EASY50_BOUNDED)
    assert bounded["ratios"] == {} and "rerank_ratios" not in bounded
    assert bounded["bounded_rows"] == {
        "65536": {"worst_hop_rows": 128_005, "worst_terminal_request": 94_595, "na_items": 0},
        "32768": {"worst_hop_rows": 95_237, "worst_terminal_request": 61_827, "na_items": 0},
        "16384": {"worst_hop_rows": 78_853, "worst_terminal_request": 45_443, "na_items": 0},
        "8192": {"worst_hop_rows": 70_661, "worst_terminal_request": 37_251, "na_items": 0},
        "4096": {"worst_hop_rows": 66_565, "worst_terminal_request": 33_155, "na_items": 0},
        "2048": {"worst_hop_rows": 64_517, "worst_terminal_request": 31_107, "na_items": 0},
    }
    for rows, row in bounded["bounded_rows"].items():
        assert row["worst_terminal_request"] == int(rows) + 963 + 24_000 + 4_096, rows
        assert row["worst_hop_rows"] == int(rows) + 61_466 + 963 + 40, rows
    for key in ("controls", "chunk_tokens", "text_sender_context"):
        assert bounded[key] == rerank[key], key
    # A roster with no latent arm registers no ladder: the ratio table is empty
    # and the sealed geometry is the window, the allowances, the controls, the
    # chunk bounds, and one sender context that falls with the note ceiling.
    text = frozen_geometry(LONGBENCH_COA_EASY50_TEXT)
    assert text["ratios"] == {} and "rerank_ratios" not in text and "bounded_rows" not in text
    assert text["controls"] == rerank["controls"] and text["chunk_tokens"] == rerank["chunk_tokens"]
    assert text["text_sender_context"] == 98_685 == 61_466 + 16_000 + 1_123 + 16_000 + 4_096
    # The allowance base is the registered constant; the easy panel's own
    # measured maximum sits under it.
    assert EASY50.question_choices_max_tokens == QUESTION_CHOICES_MAX_TOKENS == 883
    # The sealed table and the row rule both price a rewrite prompt on the
    # rewrite wrapper, the wider of the two.
    assert REWRITE_WRAPPER_ALLOWANCE > HOP_WRAPPER_ALLOWANCE
    shared = REWRITE_WRAPPER_ALLOWANCE
    for price in (rewrite_prompt_ceiling, text_sender_prompt):
        assert price(60_000, report_ceiling=24_000, wrapper=shared) == 60_000 + 24_000 + shared
    assert rewrite_wrapper_allowance("longbench-v2-official-0shot-v1") == shared
    with pytest.raises(ValueError, match="no rewrite wrapper allowance"):
        rewrite_wrapper_allowance("other-builder-v1")
    # The table's own pricing walks `ceil(rows / ratio)` over the whole sequence.
    walk = chain_geometry((60_000,) * 4, law=RATIO_TABLE_LAW, ratio=2, answer_ceiling=24_000)
    assert walk.hop_rows == (61_003, 91_505, 106_756, 114_381)
    assert walk.carried_rows == (30_502, 45_753, 53_378, 57_191)
    assert walk.terminal_request == 57_191 + 963 + 24_000 + 4_096 and walk.executable
    # The rerank walk adds the carried count whole and cuts the new rows.
    reranked = chain_geometry((60_000,) * 4, law="rerank", ratio=4, answer_ceiling=24_000)
    assert reranked.hop_rows == (61_003, 76_254, 91_505, 106_756)
    assert reranked.carried_rows == (15_251, 30_502, 45_753, 61_004)
    # The bounded walk carries everything until the cap binds, then the cap.
    capped_rows = chain_geometry(
        (60_000,) * 4, law="bounded", budget_rows=65_536, answer_ceiling=24_000
    )
    assert capped_rows.hop_rows == (61_003, 122_006, 126_539, 126_539)
    assert capped_rows.carried_rows == (61_003, 65_536, 65_536, 65_536)
    assert capped_rows.terminal_request == 65_536 + 963 + 24_000 + 4_096 and capped_rows.executable
    capped = chain_geometry(
        (60_000,) * 4,
        law="la",
        schedule="cap_65536",
        cumulative_source_tokens=(64_000, 128_000, 192_000, 256_000),
        answer_ceiling=24_000,
    )
    assert capped.carried_rows == (61_003, 65_536, 65_536, 65_536)
    assert capped.status == "executable" and capped.worst_hop == 126_539
    uncut = chain_geometry((70_000,) * 4, law="full", answer_ceiling=24_000)
    assert uncut.status == "N/A_context" and uncut.hop_rows[1] == 142_006


def test_chain_topology_and_budget_laws():
    assert chain.HOPS == 4 and LONGBENCH_COA_EASY50_RERANK.span_width == chain.SPAN_WIDTH
    assert chain.SCHEDULE_UNIT_TOKENS == SOURCE_CHUNK_CEILING
    assert chain.CHAIN_TOPOLOGY_KEY == "chain-t4" == LONGBENCH_COA_EASY50_RERANK.topology_key
    assert chain.CHAIN_T4.worker_assignment == "sequential-hop-chain-global-recut"
    nominal = {
        1: (2_048, 2_048, 2_048, 2_048),
        2: (2_048, 3_456, 3_744, 4_096),
        3: (2_048, 4_288, 5_040, 6_144),
        4: (2_048, 4_880, 6_144, 8_192),
    }
    for update, expected in nominal.items():
        stream = update * chain.SCHEDULE_UNIT_TOKENS
        observed = tuple(
            chain.schedule_rows(name, stream) for name in ("constant", "log", "sqrt", "linear")
        )
        assert observed == expected, update
        assert chain.schedule_rows("cap_65536", stream) == chain.CAP_ROWS
        assert all(rows % chain.SPAN_WIDTH == 0 for rows in observed), update
    assert chain.la_budget("cap_65536", 64_000, 1_000) == 1_000
    assert chain.hop_budget("la", 100, schedule="linear", cumulative_source_tokens=64_000) == 100
    assert chain.hop_budget("full", 7) == 7
    # Every chain arm names its law, and the law strings are these four and
    # nothing else.
    assert chain.BUDGET_LAWS == ("rerank", "bounded", "la", "full")
    assert chain.RERANK_BUDGET_LAW == "rerank" and chain.BOUNDED_BUDGET_LAW == "bounded"
    # The re-ranked handoff law: hop one is the plain re-cut, later hops add
    # the carried count whole and cut only the rows the hop just read.
    assert chain.hop_budget("rerank", 100, ratio=8) == math.ceil(100 / 8) == 13
    assert chain.rerank_budget(8_232, 0, 128) == 65 and chain.rerank_budget(10, 0, 1) == 10
    for rows, prefix, ratio in ((100, 40, 8), (8_232, 1_000, 128)):
        expected = prefix + math.ceil((rows - prefix) / ratio)
        assert chain.rerank_budget(rows, prefix, ratio) == expected > math.ceil(rows / ratio)
        assert chain.hop_budget("rerank", rows, ratio=ratio, prefix_rows=prefix) == expected
    # The bounded law: every row until the cap binds, then exactly the cap,
    # whatever the prefix; the ratio plays no part.
    assert chain.BOUNDED_ROW_BUDGETS == (65_536, 32_768, 16_384, 8_192, 4_096, 2_048)
    assert chain.bounded_budget(100, 4_096) == 100 and chain.bounded_budget(8_232, 4_096) == 4_096
    assert chain.hop_budget("bounded", 100, budget_rows=64) == 64
    assert chain.hop_budget("bounded", 100, budget_rows=64, prefix_rows=90, ratio=8) == 64
    assert chain.hop_budget("bounded", 100, budget_rows=65_536) == 100
    for bad in (
        lambda: chain.hop_budget("coa", 100, ratio=8),
        lambda: chain.hop_budget("rerank", 100),
        lambda: chain.hop_budget("rerank", 100, ratio=8, prefix_rows=100),
        lambda: chain.rerank_budget(0, 0, 8),
        lambda: chain.hop_budget("bounded", 100, ratio=8),
        lambda: chain.bounded_budget(0, 8),
        lambda: chain.hop_budget("la", 100, schedule="log"),
        lambda: chain.hop_budget("mystery", 100),
        lambda: chain.schedule_rows("square", 1),
    ):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError, match="needs a row budget"):
        chain.hop_budget("bounded", 100)
    # A body carries a law only where an arm registers it. The `coa` rule
    # string is recorded in every body; `ratios` is the ladder a profile
    # reports on.
    rerank_body = registration_body(LONGBENCH_COA_EASY50_RERANK)
    text_body = registration_body(LONGBENCH_COA_EASY50_TEXT)
    bounded_body = registration_body(LONGBENCH_COA_EASY50_BOUNDED)
    for body in (rerank_body, text_body, bounded_body):
        assert body["budget_laws"]["coa"] == "ceil(rows / ratio)"
        assert body["budget_laws"]["schedules"] == list(chain.SCHEDULE_NAMES)
        assert body["protected"].startswith("sink zero plus the current forty latent rows")
    assert rerank_ratios(LONGBENCH_COA_EASY50_RERANK) == _RERANK_LADDER
    assert rerank_body["budget_laws"]["rerank"] == "prefix + ceil((rows - prefix) / ratio)"
    assert rerank_body["budget_laws"]["rerank_ratios"] == list(_RERANK_LADDER)
    assert rerank_body["protected_rerank"] == (
        "sink zero plus the current forty latent rows; every carried row is "
        "re-scored against the rows the hop just read"
    )
    assert set(rerank_body["geometry"]["rerank_ratios"]) == {"4", "8", "16", "32", "64", "128"}
    assert rerank_body["ratios"] == list(_RERANK_LADDER)
    assert {"bounded", "bounded_rows"}.isdisjoint(rerank_body["budget_laws"])
    # The bounded body: its law and its rung list, largest first, beside the
    # recorded strings; no rerank rule, no rerank table, and no ratio ladder.
    assert bounded_rows(LONGBENCH_COA_EASY50_BOUNDED) == chain.BOUNDED_ROW_BUDGETS
    assert rerank_ratios(LONGBENCH_COA_EASY50_BOUNDED) == ()
    assert bounded_body["budget_laws"] == {
        "coa": "ceil(rows / ratio)",
        "bounded": "min(budget_rows, rows)",
        "bounded_rows": [65_536, 32_768, 16_384, 8_192, 4_096, 2_048],
        "la": "min(schedule(cumulative source tokens), rows)",
        "schedules": ["constant", "log", "sqrt", "linear", "cap_65536"],
        "kappa": 2_048,
        "cap_rows": 65_536,
        "span_width": 16,
    }
    assert bounded_body["ratios"] == [] and bounded_body["geometry"]["ratios"] == {}
    assert set(bounded_body["geometry"]["bounded_rows"]) == {
        "65536",
        "32768",
        "16384",
        "8192",
        "4096",
        "2048",
    }
    assert "protected_rerank" not in bounded_body
    assert "rerank_ratios" not in bounded_body["geometry"]
    # A profile with no latent arm carries no ladder and no law table on either
    # side; its budget-law block is the recorded strings alone. What it does
    # carry is the writer's two values, together under `sampling` and nowhere else.
    assert rerank_ratios(LONGBENCH_COA_EASY50_TEXT) == ()
    assert bounded_rows(LONGBENCH_COA_EASY50_TEXT) == ()
    assert text_body["ratios"] == [] and text_body["geometry"]["ratios"] == {}
    assert set(text_body["budget_laws"]) == {
        "coa",
        "la",
        "schedules",
        "kappa",
        "cap_rows",
        "span_width",
    }
    assert "protected_rerank" not in text_body
    assert "rerank_ratios" not in text_body["geometry"]
    assert "bounded_rows" not in text_body["geometry"]
    assert text_body["sampling"]["report_ceiling"] == 16_000
    assert text_body["sampling"]["report_presence_penalty"] == 1.5
    for other in (rerank_body, bounded_body):
        assert "report_presence_penalty" not in other["sampling"]
        assert other["sampling"]["report_ceiling"] == 24_000


# --- The sealers and the prompt bodies they measure.


def test_prepare_round_trip_and_refusals(tmp_path):
    profile = dataclasses.replace(CHAIN_PROFILE, question_ids=("q1", "q2", "q3"))
    items = (chain_item("q1", 10), chain_item("q2", 20, stratum=4), chain_item("q3", 30, stratum=5))
    with pytest.raises(RuntimeError, match="must run before"):
        validate_source_audit(tmp_path, source_commit=COMMIT, profile=profile)
    audit = seal_source_audit(tmp_path, source_commit=COMMIT, items=items, profile=profile)
    assert audit["strata"] == {"3": 1, "4": 1, "5": 1}
    assert validate_source_audit(tmp_path, source_commit=COMMIT, profile=profile) == audit
    manifest = seal_prepared_panel(
        tmp_path, source_commit=COMMIT, items=items, source_audit=audit, profile=profile
    )
    loaded, observed = load_prepared_panel(tmp_path, source_commit=COMMIT, profile=profile)
    assert loaded == items and observed == manifest
    assert manifest["panel_registration_sha"] == panel_registration_sha256(profile)
    assert manifest["config_fingerprint"] == profile.prepared_config_fingerprint
    with pytest.raises(RuntimeError, match="mismatch"):
        load_prepared_panel(tmp_path, source_commit="b" * 40, profile=profile)
    with pytest.raises(RuntimeError, match="wrong panel"):
        seal_source_audit(tmp_path, source_commit=COMMIT, items=items[:2], profile=profile)
    # A forged item restores through unpickling without its constructor; the
    # loader re-establishes every invariant, the gold letter included.
    for field, value in (("gold", "Z"), ("question", 7)):
        forged = copy.copy(items[0])
        object.__setattr__(forged, field, value)
        with pytest.raises(RuntimeError, match="malformed"):
            _validate_items((forged, *items[1:]), profile=profile, audit=audit)
    artifact, _manifest_path, _construction = prepared_paths(tmp_path, profile)
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="does not match"):
        load_prepared_panel(tmp_path, source_commit=COMMIT, profile=profile)
    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        _PreparedUnpickler(io.BytesIO(pickle.dumps(Path("x")))).load()
    with pytest.raises(ValueError, match="four chunks"):
        dataclasses.replace(items[0], chunks=items[0].chunks[:3])
    with pytest.raises(ValueError, match="gold"):
        dataclasses.replace(items[0], gold="E")


def test_source_bundle_binds_to_its_profile(tmp_path):
    rows = {"q1": raw_row("q1", "context one é"), "q2": raw_row("q2", "context two", "A")}
    # A stand-in whose source fields are still unminted; the roster and the
    # rules that read those fields are what is under test.
    profile = dataclasses.replace(
        CHAIN_PROFILE,
        question_ids=("q2", "q1"),
        source_logical_fingerprint=UNMINTED_SENTINEL,
        source_manifest_sha256=UNMINTED_SENTINEL,
        source_archive_sha256=UNMINTED_SENTINEL,
    )
    root = tmp_path / "bundle"
    manifest = build_source_bundle(root, rows, profile.question_ids)
    assert manifest["qids"] == ["q1", "q2"] and manifest["context_files"] == 2
    validated = validate_source_bundle(root, profile=profile, sentinel=UNMINTED_SENTINEL)
    assert validated["logical_fingerprint"] == manifest["logical_fingerprint"]
    assert validated["archive_sha256"] == manifest["archive_sha256"]
    assert unminted_digests(profile, UNMINTED_SENTINEL) == (
        "source_logical_fingerprint",
        "source_manifest_sha256",
        "source_archive_sha256",
    )
    assert bundle_rows(root) == rows
    roster = [
        (qid, hashlib.sha256(row["context"].encode()).hexdigest()) for qid, row in rows.items()
    ]
    assert manifest["logical_fingerprint"] == logical_fingerprint(
        raw_sha256=RAW_SOURCE_SHA256, qids=["q2", "q1"], roster=roster[::-1]
    )
    assert json.loads((root / "gold.json").read_text()) == {"q1": "C", "q2": "A"}
    assert "answer" not in (root / "items.json").read_text()
    # A minted profile whose manifest digest disagrees with the bundle is
    # refused; one whose digest agrees passes; the archive digest is recorded,
    # not compared; a different roster is refused outright.
    minted = dataclasses.replace(profile, source_manifest_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="differs from the sealed"):
        validate_source_bundle(root, profile=minted, sentinel=UNMINTED_SENTINEL)
    agreed = dataclasses.replace(
        profile,
        source_manifest_sha256=manifest["manifest_sha256"],
        source_archive_sha256="0" * 64,
    )
    validate_source_bundle(root, profile=agreed, sentinel=UNMINTED_SENTINEL)
    other = dataclasses.replace(profile, question_ids=("q1", "q2", "q3"))
    with pytest.raises(RuntimeError, match="roster"):
        validate_source_bundle(root, profile=other, sentinel=UNMINTED_SENTINEL)
    with pytest.raises(RuntimeError, match="unsafe"):
        build_source_bundle(tmp_path / "escape", {"../x": raw_row("../x", "c")}, ("../x",))
    (root / "contexts" / "q1.txt").write_text("drift")
    with pytest.raises(RuntimeError, match="drifted"):
        validate_source_bundle(root, profile=profile, sentinel=UNMINTED_SENTINEL)


def test_build_item_prompt_allowances_and_cli_profiles(monkeypatch):
    tokenizer = FakeTokenizer(width=2)
    context = "".join(chr(97 + (i * 5) % 26) for i in range(400))
    _chunks, chunk_rows = balanced_four_chunks(context, tokenizer)
    raw = raw_row("q9", context, "D")
    item = build_item(raw, chunk_rows, tokenizer, stratum=3)
    assert item.chunk_tokens == tuple(row["standalone_qwen_source_tokens"] for row in chunk_rows)
    assert item.source_tokens == chunk_rows[-1]["whole_token_end"] and item.gold == "D"
    drifted = [dict(row) for row in chunk_rows]
    drifted[0]["chunk_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="drifted"):
        build_item(raw, drifted, tokenizer, stratum=3)
    # The panel maximum must be the registered one exactly.
    pairs = [(item.question, item.choices)]
    with pytest.raises(RuntimeError, match="differs from the registered"):
        build_cli.assert_prompt_allowances(pairs, tokenizer, EASY50, prompts.PROMPT_BUILDER)
    observed = len(tokenizer.encode("Q q9?\n(A) a\n(B) b\n(C) c\n(D) d"))
    panel = dataclasses.replace(EASY50, question_choices_max_tokens=observed)
    # No panel may register a maximum over the registered allowance base.
    over = dataclasses.replace(panel, question_choices_max_tokens=10**6)
    with pytest.raises(RuntimeError, match="exceeds the allowance base"):
        build_cli.assert_prompt_allowances(pairs, tokenizer, over, prompts.PROMPT_BUILDER)
    monkeypatch.setattr(build_cli, "QUESTION_CHOICES_MAX_TOKENS", observed)
    maxima = build_cli.assert_prompt_allowances(pairs, tokenizer, panel, prompts.PROMPT_BUILDER)
    assert maxima["question_choices"] == observed and maxima["hop_wrapper"] > observed
    assert maxima["rewrite_wrapper"] > maxima["hop_wrapper"]
    # The floor body is gated too, on the answer prompt's own allowance: it
    # carries the same question and choices and no notes at all.
    assert NO_CONTEXT_PROMPT_ALLOWANCE == QUESTION_CHOICES_MAX_TOKENS + ANSWER_TEMPLATE_ALLOWANCE
    assert maxima["no_context_prompt"] > observed
    monkeypatch.setattr(build_cli, "NO_CONTEXT_PROMPT_ALLOWANCE", 1)
    with pytest.raises(RuntimeError, match="no context prompt"):
        build_cli.assert_prompt_allowances(pairs, tokenizer, panel, prompts.PROMPT_BUILDER)
    monkeypatch.setattr(build_cli, "ANSWER_PROMPT_ALLOWANCE", 1)
    with pytest.raises(RuntimeError, match="answer prompt"):
        build_cli.assert_prompt_allowances(pairs, tokenizer, panel, prompts.PROMPT_BUILDER)
    monkeypatch.setattr(build_cli, "rewrite_wrapper_allowance", lambda builder: 1)
    with pytest.raises(RuntimeError, match="rewrite wrapper"):
        build_cli.assert_prompt_allowances(pairs, tokenizer, panel, prompts.PROMPT_BUILDER)
    monkeypatch.setattr(build_cli, "HOP_WRAPPER_ALLOWANCE", 1)
    with pytest.raises(RuntimeError, match="hop wrapper"):
        build_cli.assert_prompt_allowances(pairs, tokenizer, panel, prompts.PROMPT_BUILDER)
    assert build_cli.resolve_profile(PROFILE_KEY) is CHAIN_PROFILE
    with pytest.raises(ValueError, match="unregistered"):
        build_cli.resolve_profile(FANOUTQA_NATURAL_DEV50.benchmark_key)
    # Every sealed chain profile is a build target, the three native ones too.
    assert sorted(build_cli.PROFILES) == [
        "longbench-v2-coa-easy50-bounded-sealed",
        "longbench-v2-coa-easy50-rerank-sealed",
        "longbench-v2-coa-easy50-text-sealed",
        "longbench-v2-coa-nemotron-easy50-bounded-sealed",
        "longbench-v2-coa-nemotron-easy50-rerank-sealed",
        "longbench-v2-coa-nemotron-easy50-text-sealed",
    ]
    for minted in (CHAIN_PROFILE, LONGBENCH_COA_EASY50_TEXT):
        assert build_cli.resolve_profile(minted.benchmark_key) is minted
        assert build_cli.unminted_fields(minted) == ()
        assert unminted_digests(minted, UNMINTED_SENTINEL) == ()
    # The bounded profile's bundle fields are the easy fifty's minted bytes, so
    # the validator has nothing to paste.
    bounded = build_cli.resolve_profile("longbench-v2-coa-easy50-bounded-sealed")
    assert bounded is LONGBENCH_COA_EASY50_BOUNDED
    assert unminted_digests(bounded, UNMINTED_SENTINEL) == ()
    assert build_cli.unminted_fields(bounded) == ()
    parser = build_cli._parser()
    for argv in (
        ["fetch-raw", "--profile", PROFILE_KEY, "--root", "r"],
        ["select", "--profile", PROFILE_KEY, "--raw", "d.json", "--panel-dir", "p"],
        ["build-bundle", "--profile", PROFILE_KEY, "--raw", "d", "--bundle", "b"],
        [
            "prepare",
            "--profile",
            PROFILE_KEY,
            "--bundle",
            "b",
            "--run-root",
            "r",
            "--source-commit",
            COMMIT,
        ],
    ):
        assert parser.parse_args(argv).command == argv[0]
    with pytest.raises(RuntimeError, match="population"):
        validate_selection(
            PanelSelection(selected=(), excluded=(), population={}, survivors={}), EASY50
        )


def test_prompts_are_the_official_templates_and_never_take_gold():
    sha = hashlib.sha256
    assert sha(prompts.ZERO_SHOT_TEMPLATE.encode()).hexdigest() == prompts.ZERO_SHOT_TEMPLATE_SHA256
    assert (
        sha(prompts.NO_CONTEXT_TEMPLATE.encode()).hexdigest() == prompts.NO_CONTEXT_TEMPLATE_SHA256
    )
    tokenizer = FakeTokenizer(width=2)
    # The checks below run on the fake, so the fake must carry the thinking
    # switch rather than flatten it: the real template closes an empty thinking
    # block when the switch is off.
    messages = [{"role": "user", "content": "q"}]
    thinking = tokenizer.apply_chat_template(messages, enable_thinking=True)
    closed = tokenizer.apply_chat_template(messages, enable_thinking=False)
    assert thinking != closed and closed.endswith("<think>\n\n</think>\n\n")
    assert tokenizer.apply_chat_template(messages) == thinking, "thinking is the default"
    question, choices = "  Who wrote it?  ", (" a ", "b", "c", "d")
    answer = prompts.answer_prompt(tokenizer, question, choices, enable_thinking=True)
    assert (
        "<text>\n\n</text>" in answer
        and "question: Who wrote it?\nChoices:\n(A) a\n(B) b" in answer
    )
    # A marker inside the notes is never substituted by a later pass.
    with_notes = prompts.answer_prompt(
        tokenizer, question, choices, notes="notes $Q$ $C_A$", enable_thinking=True
    )
    assert "<text>\nnotes $Q$ $C_A$\n</text>" in with_notes
    # The latent receiver body names the rows before the turn and has no slot.
    latent = prompts.answer_prompt(tokenizer, question, choices, enable_thinking=True, latent=True)
    assert "precede this message" in latent and "<text>" not in latent
    with pytest.raises(ValueError, match="never notes"):
        prompts.answer_prompt(
            tokenizer, question, choices, notes="n", enable_thinking=True, latent=True
        )
    item = chain_item("prompt-builder", 10)
    # The prompt record routes the channel: the latent arm reads the latent body.
    assert manager_prompt_record(
        item, tokenizer, family=QWEN_FAMILY, profile=CHAIN_PROFILE, reports=(), channel="latent"
    ).text == prompts.answer_prompt(
        tokenizer, item.question, item.choices, enable_thinking=True, latent=True
    )
    record = manager_prompt_record(
        item,
        tokenizer,
        family=QWEN_FAMILY,
        profile=CHAIN_PROFILE,
        reports=("running notes",),
        channel="text",
    )
    assert record.text == prompts.answer_prompt(
        tokenizer,
        item.question,
        item.choices,
        notes="running notes",
        enable_thinking=True,
    )
    # The builder strips the question, so a panel question with edge
    # whitespace renders the same prompt and its question rows are located
    # against that rendered text, never against the raw panel string.
    padded = dataclasses.replace(item, question=f"  {item.question}  ")
    assert (
        manager_prompt_record(
            padded,
            tokenizer,
            family=QWEN_FAMILY,
            profile=CHAIN_PROFILE,
            reports=("running notes",),
            channel="text",
        )
        == record
    )
    floor = prompts.no_context_prompt(tokenizer, question, choices, enable_thinking=True)
    assert "single, most likely" in floor and "<text>" not in floor
    chunk_ids = tuple(range(100, 140))
    ids = prompts.hop_prompt_ids(tokenizer, question, choices, chunk_ids, 2, enable_thinking=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts.hop_prompt_text(question, choices, 2)}]
    )
    prefix, suffix = rendered.split("<<RCC_DOC_SLOT>>")
    # A hop is a worker brief: the question and choices before the part, the
    # carried rows named from hop two on, and no answer directive anywhere.
    assert prefix.endswith("Passage:\n") and suffix.startswith("<|im_end|>")
    assert "part 2 of 4; what the earlier parts kept" in prefix
    assert prefix.index("Choices:") < prefix.index("Passage:")
    assert "Do not answer" in prefix and "correct answer is" not in prefix
    first = prompts.hop_prompt_text(question, choices, 1)
    assert "part 1 of 4." in first and "precedes" not in first
    assert ids == (*tokenizer.encode(prefix), *chunk_ids, *tokenizer.encode(suffix))
    counts = prompts.prompt_token_counts(
        tokenizer, question, choices, enable_thinking=True, builder=prompts.PROMPT_BUILDER
    )
    assert counts["hop_wrapper"] == len(ids) - len(chunk_ids)
    # The rewrite wrapper is the whole rendered rewrite prompt less the part
    # it carries, and it is the wider of the two wrappers. That is why a text
    # arm is priced on its own allowance and not on the hop's.
    rewritten = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompts.rewrite_prompt_text(
                    question,
                    choices,
                    notes="",
                    chunk_text="<<RCC_DOC_SLOT>>",
                    hop=1,
                    builder=prompts.PROMPT_BUILDER,
                ),
            }
        ]
    )
    rewrite_prefix, rewrite_suffix = rewritten.split("<<RCC_DOC_SLOT>>")
    assert prompts.NO_NOTES in rewrite_prefix
    assert rewrite_suffix.startswith("\n\nReturn the rewritten notes now.")
    assert counts["rewrite_wrapper"] == len(tokenizer.encode(rewrite_prefix)) + len(
        tokenizer.encode(rewrite_suffix)
    )
    assert counts["rewrite_wrapper"] > counts["hop_wrapper"]

    # The body the wrapper measures is the body the seat renders, part aside.
    def render(builder: str) -> str:
        return prompts.rewrite_prompt(
            tokenizer,
            question,
            choices,
            notes="",
            chunk_text="X",
            hop=1,
            enable_thinking=True,
            builder=builder,
        )

    assert render(prompts.PROMPT_BUILDER) == rewritten.replace("<<RCC_DOC_SLOT>>", "X")
    # The condensing builder renders the same wrapper around its own brief, with
    # the length target spelled out, and its sealed digest is its own.
    condense = render(prompts.PROMPT_BUILDER_CONDENSE)
    assert "under 8,000 words" in condense and "Do not copy passages" in condense
    assert "Return the rewritten notes now." in condense and "Part 1 of 4:\nX" in condense
    assert prompts.rewrite_bodies_sha256(prompts.PROMPT_BUILDER) != (
        prompts.rewrite_bodies_sha256(prompts.PROMPT_BUILDER_CONDENSE)
    )
    with pytest.raises(ValueError, match="no rewrite brief"):
        render("other-v1")
    assert counts["answer_prompt"] == len(tokenizer.encode(answer))
    assert counts["question_choices"] == len(
        tokenizer.encode(prompts.question_choices_text(question, choices))
    )
    assert (
        prompts.question_block(question, choices)
        == "Who wrote it?\nChoices:\n(A) a\n(B) b\n(C) c\n(D) d"
    )
    # A question with its own line breaks keeps them whole in both layouts.
    two_lines = "First line.\nSecond line?"
    assert prompts.question_choices_text(two_lines, choices) == (
        "First line.\nSecond line?\n(A) a\n(B) b\n(C) c\n(D) d"
    )
    assert prompts.question_block(two_lines, choices) == (
        "First line.\nSecond line?\nChoices:\n(A) a\n(B) b\n(C) c\n(D) d"
    )
    rewrite_two = prompts.rewrite_prompt(
        tokenizer,
        two_lines,
        choices,
        notes="n",
        chunk_text="p",
        hop=1,
        enable_thinking=True,
        builder=prompts.PROMPT_BUILDER,
    )
    assert "Question and choices:\nFirst line.\nSecond line?\nChoices:\n(A) a" in rewrite_two
    text, retention_ids, question_ids = prompts.retention_prompt(
        tokenizer, question, enable_thinking=True
    )
    assert text == tokenizer.apply_chat_template([{"role": "user", "content": "Who wrote it?"}])
    assert len(retention_ids) == len(tokenizer.encode(text)) and question_ids
    rewrite = prompts.rewrite_prompt(
        tokenizer,
        question,
        choices,
        notes="",
        chunk_text="part text",
        hop=3,
        enable_thinking=True,
        builder=prompts.PROMPT_BUILDER,
    )
    assert prompts.NO_NOTES in rewrite and "Part 3 of 4:\npart text" in rewrite
    for name in ("answer_prompt", "no_context_prompt", "hop_prompt_ids", "rewrite_prompt"):
        parameters = inspect.signature(getattr(prompts, name)).parameters
        assert not {"gold", "answer", "item"} & set(parameters), name
    with pytest.raises(ValueError, match="hop"):
        prompts.hop_prompt_text(question, choices, 5)


# --- The execution identity: profile fields, resolver, placements.

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


_COMMIT = "0" * 40


# The three shipped Qwen chain configs, one per ladder.
_RERANK_CONFIG = "qwen-longbench-coa-easy50-rerank-n50.toml"


_BOUNDED_CONFIG = "qwen-longbench-coa-easy50-bounded-n50.toml"


_TEXT_CONFIG = "qwen-longbench-coa-easy50-text-n50.toml"


_NEMOTRON = "nemotron-longbench-coa-"


_EXECUTION_FIELDS = (
    *("benchmark_key", "arm_profiles", "topology_key", "run_id_stem", "output_prefix"),
    *("declared_passes", "payload_layout"),
    *("scorer", "score_fields", "prompt_builder"),
)


_PROMPT_SEAMS = (
    *("answer_prompt", "no_context_prompt", "hop_prompt_text", "hop_prompt_ids"),
    *("question_block", "question_choices_text", "retention_prompt", "rewrite_prompt"),
    "prompt_token_counts",
)


def test_every_registered_benchmark_matches_its_topology_worker_count():
    for profile in plan_module._BENCHMARKS.values():
        assert (
            profile.workers_per_item
            == plan_module._TOPOLOGIES[profile.topology_key].workers_per_item
        ), profile.benchmark_key


def test_execution_identity_lives_beside_the_science_never_inside_it(
    monkeypatch: pytest.MonkeyPatch,
):
    for profile in plan_module._BENCHMARKS.values():
        registered = profile.to_dict()
        assert not set(_EXECUTION_FIELDS) & set(registered), profile.profile_id
        # Moving an execution field leaves the scientific identity in place.
        moved = dataclasses.replace(profile, output_prefix="elsewhere", declared_passes=())
        assert moved.scientific_identity_hash == profile.scientific_identity_hash
    # The resolver registers each profile under its declared key.
    assert {
        "fanoutqa-natural-dev50": FANOUTQA_NATURAL_DEV50,
        "fanoutqa-natural-dev50-latent": FANOUTQA_NATURAL_DEV50_LATENT,
        LONGBENCH_COA_EASY50_BOUNDED.benchmark_key: LONGBENCH_COA_EASY50_BOUNDED,
        PROFILE_KEY: CHAIN_PROFILE,
        LONGBENCH_COA_EASY50_TEXT.benchmark_key: LONGBENCH_COA_EASY50_TEXT,
        NEMOTRON_LONGBENCH_TEXT.benchmark_key: NEMOTRON_LONGBENCH_TEXT,
        NEMOTRON_LONGBENCH_RERANK.benchmark_key: NEMOTRON_LONGBENCH_RERANK,
        NEMOTRON_LONGBENCH_BOUNDED.benchmark_key: NEMOTRON_LONGBENCH_BOUNDED,
    } == plan_module._BENCHMARKS
    assert plan_module._TOPOLOGIES == {"fanout-m3": FANOUT_M3, "chain-t4": CHAIN_T4}

    for profile in (FANOUTQA_NATURAL_DEV50,):
        assert (profile.topology_key, profile.run_id_stem) == ("fanout-m3", "fanoutqa-m3-n15-v1")
        assert profile.output_prefix == "fanoutqa"
        assert profile.payload_layout == "flat-interleave-v1"
        assert profile.score_fields == ("loose", "strict", "n_leaves")
        assert profile.prompt_builder == "fanoutqa-coordinator-handoff-v1"
    # The chain's live latent profile: the six rerank arms, its own key, arm
    # profile, and stem, with the whole panel declared.
    chain = CHAIN_PROFILE
    assert chain.benchmark_key == PROFILE_KEY == "longbench-v2-coa-easy50-rerank-sealed"
    assert chain.arm_profiles == ("longbench-coa-rerank-6arm-v1",)
    assert (chain.topology_key, CHAIN_T4.topology_id) == (CHAIN_TOPOLOGY_KEY, "chain-t4-v1")
    assert chain.run_id_stem == "longbench-coa-easy50-rerank-t4-v1"
    assert chain.output_prefix == "longbench-v2-coa"
    assert chain.declared_passes == ((0, 50),)
    assert len(chain.question_ids) == 50
    assert chain.payload_layout == PAYLOAD_LAYOUT == "chain-terminal-v1"
    assert chain.scorer == EXTRACTION_RULE and chain.score_fields == ("correct", "answered")
    assert CHAIN_TERMINAL_LAYOUT == PAYLOAD_LAYOUT
    assert [arm.arm_id for arm in chain.arms] == [
        f"latent_query_support_rerank_r{ratio}" for ratio in _RERANK_LADDER
    ]
    assert chain.ratios == _RERANK_LADDER
    assert all(arm.budget_law == "rerank" for arm in chain.arms)
    # The text profile: the FanOutQA floor and text arms in the FanOutQA order,
    # none naming a law; it borrows every field but key, arm profile, and stem.
    text = LONGBENCH_COA_EASY50_TEXT
    assert text.benchmark_key == "longbench-v2-coa-easy50-text-sealed"
    assert text.arm_profiles == ("longbench-coa-text-4arm-v1",)
    assert text.run_id_stem == "longbench-coa-easy50-text-t4-v1"
    text_ids = ("issue_only", "text_primary", "text_medium", "text_small")
    assert tuple(arm.arm_id for arm in text.arms) == text_ids and text.ratios == ()
    assert all(arm.budget_law is None for arm in text.arms)
    assert text.arms == FANOUTQA_NATURAL_DEV50.arms[:4]
    assert text.sealed_qwen_policies == FANOUTQA_NATURAL_DEV50.sealed_qwen_policies[:4]
    borrowed = [
        field
        for field in _EXECUTION_FIELDS
        if field not in {"benchmark_key", "arm_profiles", "run_id_stem"}
    ]
    assert [getattr(text, f) for f in borrowed] == [getattr(chain, f) for f in borrowed]
    assert (text.report_ceiling, text.report_presence_penalty) == (16_000, 1.5)
    # The bounded profile: its own key, arm profile, stem, and five arms,
    # largest budget first; every other execution field is the rerank profile's.
    bounded = LONGBENCH_COA_EASY50_BOUNDED
    assert bounded.benchmark_key == "longbench-v2-coa-easy50-bounded-sealed"
    assert bounded.arm_profiles == ("longbench-coa-bounded-6arm-v1",)
    assert bounded.run_id_stem == "longbench-coa-easy50-bounded-t4-v1"
    assert [arm.arm_id for arm in bounded.arms] == [
        f"latent_query_support_bounded_b{rows}" for rows in BOUNDED_ROW_BUDGETS
    ]
    assert [(arm.budget_law, arm.budget_rows, arm.retention_ratio) for arm in bounded.arms] == [
        ("bounded", rows, None) for rows in BOUNDED_ROW_BUDGETS
    ]
    assert [getattr(bounded, f) for f in borrowed] == [getattr(chain, f) for f in borrowed]
    # Native profiles: the same fifty under their own keys, stems, and prefix.
    natives = (NEMOTRON_LONGBENCH_TEXT, NEMOTRON_LONGBENCH_RERANK, NEMOTRON_LONGBENCH_BOUNDED)
    for native, qwen in zip(natives, (text, chain, bounded), strict=True):
        assert native.benchmark_key == qwen.benchmark_key.replace("coa-", "coa-nemotron-")
        assert native.run_id_stem == qwen.run_id_stem.replace("coa-", "coa-nemotron-")
        assert native.output_prefix == "longbench-v2-coa-nemotron"
        n5 = native is NEMOTRON_LONGBENCH_BOUNDED  # no 64K rung natively
        assert native.arms == tuple(a for a in qwen.arms if not (n5 and a.budget_rows == 65_536))
        profiles = ("longbench-coa-nemotron-bounded-5arm-v1",) if n5 else qwen.arm_profiles
        assert native.arm_profiles == profiles
        assert native.declared_passes == ((0, 50),)
        assert (native.report_ceiling, native.report_presence_penalty) == (16_000, 0.0)
    # Every chain profile rewrites under the condensing brief; the official
    # brief is a second builder on the same module, and both render through it.
    condense = longbench_prompts.PROMPT_BUILDER_CONDENSE
    assert condense == "longbench-v2-official-0shot-condense-v1" != longbench_prompts.PROMPT_BUILDER
    for profile in (chain, text, bounded, NEMOTRON_LONGBENCH_TEXT):
        assert profile.prompt_builder == condense, profile.profile_id
        assert qwen_prompts._require_registered_prompt_builder(profile) is longbench_prompts
    assert (
        qwen_prompts._require_registered_prompt_builder(FANOUTQA_NATURAL_DEV50) is fanoutqa_prompts
    )
    official = dataclasses.replace(chain, prompt_builder=longbench_prompts.PROMPT_BUILDER)
    registered_chain = qwen_prompts._require_registered_prompt_builder(official)
    assert registered_chain is longbench_prompts
    for name in _PROMPT_SEAMS:
        assert getattr(registered_chain, name) is getattr(longbench_prompts, name), name
    missing_builder = dataclasses.replace(chain, prompt_builder="missing-prompt-builder")
    with pytest.raises(ValueError, match="missing-prompt-builder"):
        qwen_prompts._require_registered_prompt_builder(missing_builder)
    # A third registered builder is refused by name at both dispatches, never
    # run through the LongBench branch with LongBench's signature; the worker
    # roster refuses the chain itself, which has no fan-out workers.
    tokenizer = FakeTokenizer(width=2)
    third = dataclasses.replace(chain, prompt_builder="third-builder-v1")
    monkeypatch.setitem(
        qwen_prompts._REGISTERED_PROMPT_BUILDERS, "third-builder-v1", ModuleType("third_builder")
    )
    with pytest.raises(ValueError, match="third-builder-v1"):
        qwen_prompts.manager_prompt_record(
            chain_item("third", 10), tokenizer, family=QWEN_FAMILY, profile=third, channel="text"
        )
    question = Question(
        qid="fanout",
        question="What are alpha and bravo?",
        pages=((1, 0, "one"), (2, 0, "two")),
        raw={"answer": {"alpha": "1234", "bravo": "5678"}},
    )
    fanout_item = ProbeItem(
        qid=question.qid, question=question.question, shards=((65,),), question_obj=question
    )
    for refused in (chain, third):
        with pytest.raises(ValueError, match=refused.prompt_builder):
            qwen_prompts.worker_prompts(fanout_item, tokenizer, family=QWEN_FAMILY, profile=refused)
    # The FanOutQA lanes read the sealed default where a caller names no
    # profile; every chain seam names its profile. Each route family spells the
    # FanOutQA stem in its own run prefix.
    for family in (QWEN_FAMILY, NEMOTRON_FAMILY):
        assert family.run_prefix == f"{family.lane}-{FANOUTQA_NATURAL_DEV50.run_id_stem}-"
    for profile in (FANOUTQA_NATURAL_DEV50, chain, text, bounded):
        panel = set(range(len(profile.question_ids)))
        covered: set[int] = set()
        for start, count in profile.declared_passes:
            items = set(range(start, start + count))
            assert not covered & items and items <= panel
            covered |= items


def test_chain_configs_resolve_on_the_shared_resolver_under_their_own_prefix():
    rerank_plan = load_and_resolve(_CONFIGS / _RERANK_CONFIG, git_commit=_COMMIT)
    bounded_plan = load_and_resolve(_CONFIGS / _BOUNDED_CONFIG, git_commit=_COMMIT)
    # Each latent law resolves over its own arms under its own run id stem and
    # the shared output prefix. The physical roster is the arms the plan runs,
    # cut from the lane's seats, so a plan identity reads its own roster.
    for plan, profile, arms, law in (
        (rerank_plan, CHAIN_PROFILE, 6, "rerank"),
        (bounded_plan, LONGBENCH_COA_EASY50_BOUNDED, 6, "bounded"),
    ):
        assert plan.benchmark is profile and plan.model.model_id == "qwen3-8b"
        assert plan.arms == profile.arms and len(plan.arms) == arms
        assert [arm.semantic_arm for arm in plan.physical_arms] == [
            arm.arm_id for arm in profile.arms
        ]
        assert plan.output_uri == f"runs/longbench-v2-coa/{plan.run_id}"
        assert plan.run_id.startswith(f"qwen-longbench-coa-easy50-{law}-t4-v1-")
        assert plan.expected_rows.to_dict() == {
            "items": 50,
            "arms": arms,
            "seeds": 3,
            "total": 150 * arms,
        }
        assert plan.execution_question_ids == profile.question_ids
    assert bounded_plan.scientific_identity_hash != rerank_plan.scientific_identity_hash
    # The text profile resolves the same way over the floor and the three text
    # arms. It registers no latent arm.
    text_plan = load_and_resolve(_CONFIGS / _TEXT_CONFIG, git_commit=_COMMIT)
    assert text_plan.benchmark is LONGBENCH_COA_EASY50_TEXT and text_plan.topology is CHAIN_T4
    assert text_plan.arms == LONGBENCH_COA_EASY50_TEXT.arms and len(text_plan.arms) == 4
    assert text_plan.run_id.startswith("qwen-longbench-coa-easy50-text-t4-v1-")
    assert text_plan.output_uri == f"runs/longbench-v2-coa/{text_plan.run_id}"
    assert text_plan.expected_rows.to_dict() == {
        "items": 50,
        "arms": 4,
        "seeds": 3,
        "total": 600,
    }
    assert text_plan.execution_question_ids == LONGBENCH_COA_EASY50_TEXT.question_ids
    assert rerank_plan.scientific_identity_hash == CHAIN_PROFILE.scientific_identity_hash
    assert rerank_plan.to_dict()["execution_fingerprint"] == rerank_plan.execution_identity_hash
    assert rerank_plan.execution_expected_rows.to_dict() == rerank_plan.expected_rows.to_dict()
    assert rerank_plan.execution_identity_hash != rerank_plan.run_identity_hash
    # The native lane resolves its own profiles under its own prefixes.
    native_text, native_rerank = (
        load_and_resolve(_CONFIGS / f"{_NEMOTRON}{name}.toml", git_commit=_COMMIT)
        for name in ("easy50-text-n50", "easy50-rerank-n50")
    )
    for plan, profile, rows in (
        (native_text, NEMOTRON_LONGBENCH_TEXT, 50 * 4 * 3),
        (native_rerank, NEMOTRON_LONGBENCH_RERANK, 50 * 6 * 3),
    ):
        assert plan.benchmark is profile and plan.model.model_id == "nemotron-nano-12b-v2"
        assert plan.arms == profile.arms and plan.expected_rows.to_dict()["total"] == rows
        assert [arm.semantic_arm for arm in plan.physical_arms] == [arm.arm_id for arm in plan.arms]
        assert plan.run_id.startswith(f"nemotron-{profile.run_id_stem}-")
        assert plan.output_uri == f"runs/{profile.output_prefix}/{plan.run_id}"


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        (
            {"benchmark": LONGBENCH_COA_EASY50_BOUNDED.benchmark_key},
            r"unsupported arm profile 'longbench-coa-rerank-6arm-v1'",
        ),
        ({"item_start": 0, "item_count": 8, "model": "ministral3-14b"}, r"cannot run the"),
        ({"model": "nemotron-nano-12b-v2"}, r"cannot run the LongBench chain"),
        ({"topology": "fanout-m3"}, r"runs on topology 'chain-t4', not 'fanout-m3'"),
        ({"arm_profile": "fanoutqa-m3-11arm-sealed"}, r"unsupported arm profile"),
        (
            {"benchmark": "fanoutqa-natural-dev50", "arm_profile": "fanoutqa-m3-11arm-sealed"},
            r"runs on topology 'fanout-m3', not 'chain-t4'",
        ),
    ),
)
def test_chain_resolver_refuses_every_undeclared_shape(changes: dict[str, object], match: str):
    config = dataclasses.replace(load_run_config(_CONFIGS / _RERANK_CONFIG), **changes)
    with pytest.raises(ValueError, match=match):
        resolve_plan(config, ResolutionContext(git_commit=_COMMIT))


def test_chain_placements_are_a_second_table_keyed_by_topology():
    for lookup in (placement_for_semantic_arm, bind_policy_placements):
        parameter = inspect.signature(lookup).parameters["table"]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert PLACEMENT_TABLES == {
        "fanout-m3": SEMANTIC_ARM_PLACEMENTS,
        "chain-t4": CHAIN_ARM_PLACEMENTS,
    }
    assert placement_table("chain-t4") is CHAIN_ARM_PLACEMENTS
    assert placement_table(FANOUTQA_NATURAL_DEV50.topology_key) is SEMANTIC_ARM_PLACEMENTS
    expected = {
        "issue_only": FleetPlacement(8, 0, 8),
        "text_primary": FleetPlacement(8, 0, 8, fused=True),
        "text_medium": FleetPlacement(8, 6, 2),
        "text_small": FleetPlacement(8, 6, 2),
        # Every latent arm of both laws takes four producer seats and four
        # receiver seats. The smallest rerank ratio registers no row at all,
        # because its terminal request does not fit the window.
        **{
            f"latent_query_support_rerank_r{ratio}": FleetPlacement(8, 4, 4)
            for ratio in LATENT_RATIOS[1:]
        },
        **{
            f"latent_query_support_bounded_b{rows}": FleetPlacement(8, 4, 4)
            for rows in BOUNDED_ROW_BUDGETS
        },
    }
    assert expected == CHAIN_ARM_PLACEMENTS and len(CHAIN_ARM_PLACEMENTS) == 16
    assert "full" not in CHAIN_ARM_PLACEMENTS, "the uncut chain fits no item"
    assert set(CHAIN_ARM_PLACEMENTS) == {
        arm.arm_id
        for arm in (
            *LONGBENCH_COA_EASY50_TEXT.arms,
            *CHAIN_PROFILE.arms,
            *LONGBENCH_COA_EASY50_BOUNDED.arms,
        )
    }
    assert placement_for_semantic_arm(
        "latent_query_support_rerank_r4", table=placement_table(CHAIN_PROFILE.topology_key)
    ) == FleetPlacement(8, 4, 4)
    # FanOutQA keeps its own placements, and its arms are not chain arms.
    assert placement_for_semantic_arm(
        "latent_query_support_r2", table=placement_table(FANOUTQA_NATURAL_DEV50.topology_key)
    ) == FleetPlacement(8, 4, 4)
    with pytest.raises(ValueError, match="unregistered semantic arm"):
        placement_for_semantic_arm("latent_query_support_r2", table=CHAIN_ARM_PLACEMENTS)
    policies = {
        "qwen3_8b_rerank_r8_w16_support": "latent_query_support_rerank_r8",
        "qwen3_8b_bounded_b4096_w16_support": "latent_query_support_bounded_b4096",
    }
    bound = bind_policy_placements(policies, table=placement_table(CHAIN_PROFILE.topology_key))
    assert bound == dict.fromkeys(policies, FleetPlacement(8, 4, 4))
    with pytest.raises(ValueError, match="unregistered semantic arm"):
        placement_for_semantic_arm("full", table=CHAIN_ARM_PLACEMENTS)
    with pytest.raises(ValueError, match="unregistered placement topology"):
        placement_table("chain-t5")


def test_the_chain_reaches_the_fleet_through_its_topology_and_its_own_capture_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The adapter and capture route follow topology while model settings stay fixed."""
    from rcc.models.qwen.engine import QWEN_CHAIN_ENGINE_ROUTE
    from rcc.run.qwen.adapter import FanoutQAAdapter, execution_profile
    from rcc.run.qwen.chain_adapter import ChainAdapter
    from rcc.run.qwen.runtime import ENGINE_ROUTE, engine_route

    chain = CHAIN_PROFILE

    assert isinstance(build_adapter(QWEN_FAMILY, profile=chain), ChainAdapter)
    # The chain adapter extends the lane's, so the fan-out side is pinned by
    # its exact class and not by an isinstance every subclass would satisfy.
    assert type(build_adapter(QWEN_FAMILY, profile=FANOUTQA_NATURAL_DEV50)) is FanoutQAAdapter
    with pytest.raises(ValueError, match="chain-t5"):
        build_adapter(QWEN_FAMILY, profile=dataclasses.replace(chain, topology_key="chain-t5"))
    fanoutqa_route = engine_route(QWEN_FAMILY, FANOUTQA_NATURAL_DEV50)
    chain_route = engine_route(QWEN_FAMILY, chain)
    assert fanoutqa_route == ENGINE_ROUTE
    assert [key for key in chain_route if chain_route[key] != fanoutqa_route[key]] == [
        "route",
        "capture",
    ]
    assert chain_route["route"] == QWEN_CHAIN_ENGINE_ROUTE
    capture = cast(dict[str, Any], chain_route["capture"])
    assert capture["max_model_len"] == 131_072 and capture["enable_prompt_embeds"] is True
    native_fanout = cast(dict[str, Any], engine_route(NEMOTRON_FAMILY, FANOUTQA_NATURAL_DEV50))
    native_chain = cast(dict[str, Any], engine_route(NEMOTRON_FAMILY, chain))
    assert native_chain["capture"]["max_model_len"] == 131_072
    assert native_fanout["capture"]["max_model_len"] == 53_248
    assert {key: value for key, value in native_chain.items() if key != "capture"} == {
        key: value for key, value in native_fanout.items() if key != "capture"
    }
    assert {
        key: value for key, value in native_chain["capture"].items() if key != "max_model_len"
    } == {key: value for key, value in native_fanout["capture"].items() if key != "max_model_len"}
    with pytest.raises(ValueError, match="chain-t5"):
        engine_route(QWEN_FAMILY, dataclasses.replace(chain, topology_key="chain-t5"))
    # A worker resolves its benchmark from the environment, and with no
    # benchmark key set it refuses to guess one.
    for name, value in (
        ("RCC_FANOUT_FAMILY", "qwen"),
        ("RCC_FANOUT_ITEM_START", "0"),
        ("RCC_FANOUT_ITEM_COUNT", "15"),
    ):
        monkeypatch.setitem(os.environ, name, value)
    monkeypatch.delitem(os.environ, "RCC_FANOUT_BENCHMARK", raising=False)
    with pytest.raises(ValueError, match="RCC_FANOUT_BENCHMARK"):
        execution_profile()
    monkeypatch.setitem(os.environ, "RCC_FANOUT_BENCHMARK", FANOUTQA_NATURAL_DEV50.benchmark_key)
    assert execution_profile() is FANOUTQA_NATURAL_DEV50
    monkeypatch.setitem(os.environ, "RCC_FANOUT_BENCHMARK", PROFILE_KEY)
    monkeypatch.setitem(os.environ, "RCC_FANOUT_ITEM_COUNT", "8")
    assert execution_profile() is chain
    # The resident lanes are not benchmark parametric: they refuse a topology
    # they cannot run before they read a single argument of their own.
    namespace = argparse.Namespace(benchmark=PROFILE_KEY, item_offset=0, item_count=8)
    for binding in (split_driver._gemma_binding, split_driver._ministral_binding):
        with pytest.raises(RuntimeError, match=PROFILE_KEY):
            binding(namespace, _COMMIT)
    # Named by neither the command line nor the environment, the driver refuses
    # to run a node on a panel nobody asked for.
    monkeypatch.delitem(os.environ, "RCC_FANOUT_BENCHMARK", raising=False)
    unnamed = argparse.Namespace(benchmark=None, item_offset=0, item_count=8)
    with pytest.raises(RuntimeError, match="requires a benchmark"):
        split_driver._benchmark_profile(unnamed)
