"""The LongBench registration body, its two fingerprints, and the seed namespace.

Both fingerprints are functions of the profile and the two frozen package files
alone, so they can be recomputed without a built bundle.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rcc.benchmarks.longbench_v2.chunking import (
    SOURCE_UPDATES,
    chunk_token_counts,
    read_chunk_ledger,
)
from rcc.benchmarks.longbench_v2.geometry import (
    ANSWER_PROMPT_ALLOWANCE,
    HOP_WRAPPER_ALLOWANCE,
    WINDOW,
    bounded_table,
    control_fits,
    ratio_table,
    rewrite_wrapper_allowance,
    text_sender_context,
)
from rcc.benchmarks.longbench_v2.panel import (
    CLUSTER_KEY,
    DATASET_FILE,
    DATASET_ID,
    DATASET_REVISION,
    EXECUTION_ORDER_RULE,
    NEMOTRON_PANEL_KEYS,
    RAW_SOURCE_SHA256,
    SELECTION_BAND_TOKENS,
    TOKENIZER_ARTIFACT_SHA256,
    TOKENIZER_CHECKPOINT,
    TOKENIZER_REVISION,
    Panel,
    panel_for,
    read_manifest,
)
from rcc.benchmarks.longbench_v2.prompts import (
    NO_CONTEXT_TEMPLATE_SHA256,
    ZERO_SHOT_TEMPLATE_SHA256,
    answer_bodies_sha256,
    hop_bodies_sha256,
    rewrite_bodies_sha256,
)
from rcc.benchmarks.longbench_v2.scoring import EXTRACTION_RULE
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.run import identity
from rcc.topologies.chain import (
    BOUNDED_BUDGET_LAW,
    CAP_ROWS,
    KAPPA,
    RERANK_BUDGET_LAW,
    SCHEDULE_NAMES,
    SPAN_WIDTH,
)

REGISTRATION_VERSION = "longbench-v2-coa-serving-v1"
BUILD_CONFIG_VERSION = "longbench-v2-coa-build-v1"
PAYLOAD_LAYOUT = "chain-terminal-v1"
BOOTSTRAP_SEED = 20_260_824
BOOTSTRAP_DRAWS = 10_000
_CLAIM = (
    "descriptive LongBench v2 chain-of-agents run on the Qwen lane: exact A to D "
    "accuracy under the official extraction after a four-hop handoff; not "
    "leaderboard comparable, the chain replaces middle truncation"
)
_PROTECTED_RULE = "sink zero plus the current forty latent rows; carried rows compete"
_RERANK_RULE = "prefix + ceil((rows - prefix) / ratio)"
_BOUNDED_RULE = "min(budget_rows, rows)"
_PROTECTED_RERANK_RULE = (
    "sink zero plus the current forty latent rows; every carried row is "
    "re-scored against the rows the hop just read"
)


def law_ratios(profile: BenchmarkProfile, law: str) -> tuple[int, ...]:
    """Return the ratios this profile registers under one named budget law."""
    return tuple(
        sorted(
            {
                arm.retention_ratio
                for arm in profile.arms
                if arm.budget_law == law and arm.retention_ratio is not None
            }
        )
    )


def rerank_ratios(profile: BenchmarkProfile) -> tuple[int, ...]:
    """Return the ratios this profile registers under the re-ranked handoff law."""
    return law_ratios(profile, RERANK_BUDGET_LAW)


def bounded_rows(profile: BenchmarkProfile) -> tuple[int, ...]:
    """Return the row budgets this profile registers under the bounded law, largest first."""
    return tuple(
        sorted(
            {
                arm.budget_rows
                for arm in profile.arms
                if arm.budget_law == BOUNDED_BUDGET_LAW and arm.budget_rows is not None
            },
            reverse=True,
        )
    )


@dataclass(frozen=True)
class ChainBuildConfig:
    """The build settings the prepared bytes record."""

    version: str
    dataset_id: str
    dataset_revision: str
    dataset_file: str
    raw_sha256: str
    tokenizer_checkpoint: str
    tokenizer_revision: str
    tokenizer_artifacts: tuple[tuple[str, str], ...]
    sample_salt: str
    selection_rule: str
    cluster_key: str
    execution_order_rule: str
    band_tokens: int
    quotas: tuple[tuple[int, int], ...]
    excluded_qids: tuple[str, ...]
    source_updates: int
    chunk_ceiling: int
    span_width: int
    latent_steps: int
    workers_per_item: int
    answer_ceiling: int
    report_ceiling: int
    max_model_len: int
    report_seed_base: int
    sample_tags: tuple[str, ...]
    zero_shot_sha256: str
    no_context_sha256: str
    hop_bodies_sha256: str
    rewrite_bodies_sha256: str
    answer_bodies_sha256: str
    extraction_rule: str
    payload_layout: str


def build_config(profile: BenchmarkProfile) -> ChainBuildConfig:
    """Return one profile's build settings, with its lane's tokenizer."""
    panel = panel_for(profile)
    tokenizer_checkpoint = TOKENIZER_CHECKPOINT
    tokenizer_revision = TOKENIZER_REVISION
    tokenizer_artifacts = tuple(sorted(TOKENIZER_ARTIFACT_SHA256.items()))
    if profile.benchmark_key in NEMOTRON_PANEL_KEYS:
        from rcc.benchmarks.longbench_v2.nemotron import (
            NEMOTRON_LONGBENCH_TOKENIZER,
            NEMOTRON_TOKENIZER_ARTIFACTS,
        )

        tokenizer_checkpoint, tokenizer_revision = NEMOTRON_LONGBENCH_TOKENIZER
        tokenizer_artifacts = tuple(sorted(NEMOTRON_TOKENIZER_ARTIFACTS.items()))
    return ChainBuildConfig(
        version=BUILD_CONFIG_VERSION,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        dataset_file=DATASET_FILE,
        raw_sha256=RAW_SOURCE_SHA256,
        tokenizer_checkpoint=tokenizer_checkpoint,
        tokenizer_revision=tokenizer_revision,
        tokenizer_artifacts=tokenizer_artifacts,
        sample_salt=panel.sample_salt,
        selection_rule=panel.selection_rule,
        cluster_key=CLUSTER_KEY,
        execution_order_rule=EXECUTION_ORDER_RULE,
        band_tokens=SELECTION_BAND_TOKENS,
        quotas=tuple(sorted(panel.quotas.items())),
        excluded_qids=panel.excluded_qids,
        source_updates=SOURCE_UPDATES,
        chunk_ceiling=profile.worker_prompt_tokens,
        span_width=profile.span_width,
        latent_steps=profile.latent_steps,
        workers_per_item=profile.workers_per_item,
        answer_ceiling=profile.answer_ceiling,
        report_ceiling=profile.report_ceiling,
        max_model_len=profile.max_model_len,
        report_seed_base=profile.report_seed_base,
        sample_tags=tuple(profile.sample_tags),
        zero_shot_sha256=ZERO_SHOT_TEMPLATE_SHA256,
        no_context_sha256=NO_CONTEXT_TEMPLATE_SHA256,
        hop_bodies_sha256=hop_bodies_sha256(),
        rewrite_bodies_sha256=rewrite_bodies_sha256(profile.prompt_builder),
        answer_bodies_sha256=answer_bodies_sha256(),
        extraction_rule=EXTRACTION_RULE,
        payload_layout=PAYLOAD_LAYOUT,
    )


def build_config_fingerprint(profile: BenchmarkProfile) -> str:
    """Fingerprint the build settings the way the prepared manifest carries them."""
    return identity.fingerprint(
        asdict(build_config(profile)), identity.json_compact_legacy, digest_chars=16
    )


def _law_geometry(profile: BenchmarkProfile, counts: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """Return one geometry table per budget law the profile carries."""
    tables: dict[str, Any] = {}
    budgets = bounded_rows(profile)
    if budgets:
        tables["bounded_rows"] = {
            str(budget): row
            for budget, row in bounded_table(
                counts, budgets, answer_ceiling=profile.answer_ceiling
            ).items()
        }
    for key, law in (("rerank_ratios", RERANK_BUDGET_LAW),):
        ratios = law_ratios(profile, law)
        if not ratios:
            continue
        tables[key] = {
            str(ratio): row
            for ratio, row in ratio_table(
                counts, ratios, answer_ceiling=profile.answer_ceiling, law=law
            ).items()
        }
    return tables


def frozen_geometry(profile: BenchmarkProfile) -> dict[str, Any]:
    """Return the geometry table computed from the profile's frozen chunk ledger."""
    if profile.benchmark_key in NEMOTRON_PANEL_KEYS:
        from rcc.benchmarks.longbench_v2.native_geometry import frozen_geometry as native_geometry

        return native_geometry(profile)
    panel = panel_for(profile)
    counts = chunk_token_counts(read_chunk_ledger(panel.ledger_path))
    manifest = read_manifest(panel.manifest_path)
    source_tokens = {row["qid"]: int(row["source_tokens"]) for row in manifest}
    return {
        "window": WINDOW,
        "hop_wrapper_allowance": HOP_WRAPPER_ALLOWANCE,
        "answer_prompt_allowance": ANSWER_PROMPT_ALLOWANCE,
        "ratios": {
            str(ratio): row
            for ratio, row in ratio_table(
                counts, profile.ratios, answer_ceiling=profile.answer_ceiling
            ).items()
        },
        **_law_geometry(profile, counts),
        "controls": control_fits(counts, source_tokens, answer_ceiling=profile.answer_ceiling),
        "text_sender_context": max(
            text_sender_context(
                chunk_tokens,
                report_ceiling=profile.report_ceiling,
                wrapper=rewrite_wrapper_allowance(profile.prompt_builder),
            )
            for chunk_tokens in counts.values()
        ),
        "chunk_tokens": {
            "min": min(min(chunk_tokens) for chunk_tokens in counts.values()),
            "max": max(max(chunk_tokens) for chunk_tokens in counts.values()),
        },
    }


def registration_body(profile: BenchmarkProfile) -> dict[str, Any]:
    """Return the registration body the prepared bytes carry.

    A profile with no latent arm carries an empty ``ratios`` list and no law table. A
    named writer presence penalty sits beside the note ceiling under ``sampling``.
    """
    policies = [profile.sealed_qwen_policy(arm.arm_id) for arm in profile.arms]
    support = [
        profile.sealed_qwen_policy(arm.arm_id) for arm in profile.arms if arm.selector == "support"
    ]
    panel = panel_for(profile)
    rerank = rerank_ratios(profile)
    bounded = bounded_rows(profile)
    body: dict[str, Any] = {
        "version": REGISTRATION_VERSION,
        "claim": _CLAIM,
        "items": list(profile.question_ids),
        "item_order": EXECUTION_ORDER_RULE,
        "strata": {str(band): quota for band, quota in sorted(panel.quotas.items())},
        "band_tokens": SELECTION_BAND_TOKENS,
        "selection_rule": panel.selection_rule,
        "cluster_key": CLUSTER_KEY,
        "sample_salt": panel.sample_salt,
        "excluded_qids": list(panel.excluded_qids),
        "manifest_sha256": panel.manifest_sha256,
        "ledger_sha256": panel.ledger_sha256,
        "hops": profile.workers_per_item,
        "chunk_ceiling": profile.worker_prompt_tokens,
        "latent_steps": profile.latent_steps,
        "span_width": profile.span_width,
        "protected": _PROTECTED_RULE,
        **({} if not rerank else {"protected_rerank": _PROTECTED_RERANK_RULE}),
        "budget_laws": {
            "coa": "ceil(rows / ratio)",
            **({} if not rerank else {"rerank": _RERANK_RULE}),
            **({} if not rerank else {"rerank_ratios": list(rerank)}),
            **({} if not bounded else {"bounded": _BOUNDED_RULE}),
            **({} if not bounded else {"bounded_rows": list(bounded)}),
            "la": "min(schedule(cumulative source tokens), rows)",
            "schedules": list(SCHEDULE_NAMES),
            "kappa": KAPPA,
            "cap_rows": CAP_ROWS,
            "span_width": SPAN_WIDTH,
        },
        "policies": policies,
        "support_policies": support,
        "ratios": list(profile.ratios),
        "structural_na": ["full", "direct"],
        "payload_layout": PAYLOAD_LAYOUT,
        "geometry": frozen_geometry(profile),
        "prompts": {
            "zero_shot_sha256": ZERO_SHOT_TEMPLATE_SHA256,
            "no_context_sha256": NO_CONTEXT_TEMPLATE_SHA256,
            "hop_bodies_sha256": hop_bodies_sha256(),
            "rewrite_bodies_sha256": rewrite_bodies_sha256(profile.prompt_builder),
            "answer_bodies_sha256": answer_bodies_sha256(),
            "retention": "question only",
        },
        "scorer": EXTRACTION_RULE,
        "sampling": {
            "sample_tags": list(profile.sample_tags),
            "answer_ceiling": profile.answer_ceiling,
            "report_ceiling": profile.report_ceiling,
            # Ceiling and penalty together, because the two bound the same
            # draw. The key enters only for a profile that names a penalty.
            **(
                {}
                if not profile.report_presence_penalty
                else {"report_presence_penalty": profile.report_presence_penalty}
            ),
            "max_model_len": profile.max_model_len,
        },
        "passes": {
            "declared": [list(item_range) for item_range in panel.registration_passes],
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "unit": "qid within stratum",
        },
        "build_config_fingerprint": build_config_fingerprint(profile),
    }
    if profile.benchmark_key in NEMOTRON_PANEL_KEYS:
        from rcc.models.nemotron import NEMOTRON

        body["claim"] = _CLAIM.replace("Qwen lane", "Nemotron lane")
        body["native_model"] = NEMOTRON.to_dict()
    return body


def panel_registration_sha256(profile: BenchmarkProfile) -> str:
    """Fingerprint the registration body exactly as the prepared bytes carry it."""
    return identity.fingerprint(
        registration_body(profile), identity.json_compact_legacy, digest_chars=16
    )


def registration_string(panel: Panel) -> str:
    """Return the string one panel's seed namespace is the digest of."""
    return "|".join(
        (
            REGISTRATION_VERSION,
            DATASET_ID,
            DATASET_REVISION,
            RAW_SOURCE_SHA256,
            TOKENIZER_REVISION,
            panel.sample_salt,
            panel.selection_rule,
            EXECUTION_ORDER_RULE,
            panel.manifest_sha256,
            panel.ledger_sha256,
            f"hops{SOURCE_UPDATES}",
            f"w{SPAN_WIDTH}",
            PAYLOAD_LAYOUT,
        )
    )


def answer_seed_namespace(panel: Panel) -> str:
    """Return the seed namespace: the digest of the panel's registration string."""
    return hashlib.sha256(registration_string(panel).encode("utf-8")).hexdigest()


__all__ = (
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "BUILD_CONFIG_VERSION",
    "PAYLOAD_LAYOUT",
    "REGISTRATION_VERSION",
    "ChainBuildConfig",
    "answer_seed_namespace",
    "bounded_rows",
    "build_config",
    "build_config_fingerprint",
    "frozen_geometry",
    "law_ratios",
    "panel_registration_sha256",
    "registration_body",
    "registration_string",
    "rerank_ratios",
)
