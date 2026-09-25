"""The LongBench v2 chain-of-agents panel and its Qwen arm profiles.

One panel, the easy fifty, cut into four chunks per context and read under three
config keys. See docs/benchmarks.md, Arm rosters.
"""

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.longbench_v2.panel import (
    EASY50_BOUNDED_KEY,
    EASY50_RERANK_KEY,
    EASY50_TEXT_KEY,
    UNMINTED_SENTINEL,
)
from rcc.benchmarks.longbench_v2.prompts import PROMPT_BUILDER_CONDENSE
from rcc.benchmarks.protocol import ArmSpec, BenchmarkProfile
from rcc.topologies.chain import BOUNDED_ROW_BUDGETS, CHAIN_TOPOLOGY_KEY

# The easy fifty: the easy rows of three source-length strata (25/16/9), one
# question per context. The digests below are the source bytes every profile over
# this panel carries; the seed namespace is the digest of its registration string.
_EASY50_SOURCE_LOGICAL_FINGERPRINT = (
    "d4dc22adfa5f1e5a14f1f5a3041a3ba7af7106328dce47cf106a06eee80cdbb6"
)
_EASY50_SOURCE_MANIFEST_SHA256 = "abbd60e8f5d072ab50fed0873db17a0bba16b76f4dbf02459f837947a2b73ab1"
_EASY50_SOURCE_ARCHIVE_SHA256 = "a05840bd00156e0000094276d9df0ba499028c1b58e7a974368a84053375237d"
_EASY50_QUESTION_INDEX_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
_EASY50_ANSWER_SEED_NAMESPACE = "beee103e5c751c745275b22d117105e2815408ab035649e55e764eab5c808b17"
#: The execution order: the proportional interleave over the three strata.
EASY50_ORDER = (
    "6719c1febb02136c067d44db",
    "66ec2755821e116aacb1b6f2",
    "66ece545821e116aacb1dd77",
    "66fa700bbb02136c067c6c06",
    "6724694fbb02136c067d752b",
    "6703bed0bb02136c067cd381",
    "6725d977bb02136c067d8373",
    "66f4eff3821e116aacb31bc8",
    "6728586bbb02136c067d8f4f",
    "67191791bb02136c067d406c",
    "66f38969821e116aacb2d6ec",
    "66f3cca6821e116aacb2ef7c",
    "66ec4e43821e116aacb1cb54",
    "67239f49bb02136c067d6fc0",
    "6724938fbb02136c067d7657",
    "67285e1dbb02136c067d901b",
    "670646aebb02136c067cf85a",
    "66ec1054821e116aacb19f1a",
    "67192e81bb02136c067d427f",
    "6707b84cbb02136c067d10a5",
    "67286036bb02136c067d9092",
    "66fa415cbb02136c067c671c",
    "66f2b5ec821e116aacb2b1ec",
    "6724c1d4bb02136c067d77ec",
    "67239fd9bb02136c067d6ff7",
    "67041f08bb02136c067cdb52",
    "67009bd8bb02136c067caba9",
    "66ed23ce821e116aacb1ee84",
    "66f2616e821e116aacb28e0c",
    "6725d898bb02136c067d82d6",
    "66f97dc3bb02136c067c56c8",
    "67249705bb02136c067d76d1",
    "6724c788bb02136c067d792d",
    "6719ef50bb02136c067d4911",
    "672494e5bb02136c067d7697",
    "66fb6527bb02136c067c7b4f",
    "66f3f947821e116aacb3029b",
    "671b2e2dbb02136c067d515b",
    "671b530dbb02136c067d5454",
    "67246903bb02136c067d74f2",
    "6723a1ccbb02136c067d70b3",
    "66f3f095821e116aacb2fd28",
    "66f2d224821e116aacb2bb8a",
    "671b56cfbb02136c067d54e9",
    "6719167dbb02136c067d4025",
    "670bf1c4bb02136c067d2341",
    "67192057bb02136c067d41b4",
    "66fb8633bb02136c067c800b",
    "66f41108821e116aacb30c41",
    "671b5d49bb02136c067d55fb",
)

# The re-ranked handoff law on the same panel: the selector scores every row a hop
# holds, carried rows included, and keeps `kept[k-1] + ceil(chunk_k / ratio)`.
# See docs/benchmarks.md, Arm rosters, for why ratio 2 is absent.
_RERANK_RATIOS = (4, 8, 16, 32, 64, 128)
_RERANK_ARMS = tuple(
    ArmSpec(
        arm_id=f"latent_query_support_rerank_r{ratio}",
        channel="latent",
        selector="support",
        retention_ratio=ratio,
        sender_role="same-model",
        budget_law="rerank",
    )
    for ratio in _RERANK_RATIOS
)
_RERANK_POLICIES = tuple(
    (arm.arm_id, f"qwen3_8b_rerank_r{arm.retention_ratio}_w16_support") for arm in _RERANK_ARMS
)
# The re-ranked law alone, over those six arms. The panel is the easy fifty's own
# `EASY50` object under a second config key, so the salt, manifest, ledger, chunks,
# prompts, and seed namespace are shared and every seed pairs with its mates.
LONGBENCH_COA_EASY50_RERANK = BenchmarkProfile(
    profile_id="longbench-v2-coa-easy50-rerank-t4-n50-sealed-2026-09-12-v1",
    source_commit="83cff09055e7b9bfb17f03ffaa8e442608a74aaf",
    source_logical_fingerprint=_EASY50_SOURCE_LOGICAL_FINGERPRINT,
    source_manifest_sha256=_EASY50_SOURCE_MANIFEST_SHA256,
    source_archive_sha256=_EASY50_SOURCE_ARCHIVE_SHA256,
    question_index_sha256=_EASY50_QUESTION_INDEX_SHA256,
    prepared_artifact_sha256="3d508b5feeaeae5afad7ef786a56112ffcadb91a11c48ec91bae6a694b2af8e1",
    prepared_config_fingerprint="419438342fdb1b57",
    panel_registration_sha256="980cc61ab27bafd8",
    source_audit_fingerprint="25d8c728c61f6dc5",
    question_ids=EASY50_ORDER,
    workers_per_item=4,
    worker_prompt_tokens=64_000,
    latent_steps=40,
    span_width=16,
    ratios=_RERANK_RATIOS,
    selectors=("support",),
    arms=_RERANK_ARMS,
    sealed_qwen_policies=_RERANK_POLICIES,
    sample_tags=("s0", "s1", "s2"),
    answer_seed_namespace=_EASY50_ANSWER_SEED_NAMESPACE,
    report_seed_base=20_260_912,
    answer_ceiling=24_000,
    report_ceiling=24_000,
    max_model_len=131_072,
    benchmark_key=EASY50_RERANK_KEY,
    arm_profiles=("longbench-coa-rerank-6arm-v1",),
    topology_key=CHAIN_TOPOLOGY_KEY,
    run_id_stem="longbench-coa-easy50-rerank-t4-v1",
    output_prefix="longbench-v2-coa",
    declared_passes=((0, 50),),
    payload_layout="chain-terminal-v1",
    scorer="longbench-v2-official-extraction-strip-star-paren-then-bare-v1",
    score_fields=("correct", "answered"),
    prompt_builder=PROMPT_BUILDER_CONDENSE,
)

# The text arms alone on the same panel: no latent ladder, the floor plus the
# three text arms in the FanOutQA order, and a note draw that moves two values.
# See docs/benchmarks.md, Arm rosters.
_TEXT_ARMS = frozenset({"issue_only", "text_primary", "text_medium", "text_small"})
LONGBENCH_COA_EASY50_TEXT = BenchmarkProfile(
    profile_id="longbench-v2-coa-easy50-text-t4-n50-sealed-2026-09-12-v1",
    source_commit="250b4b52a587585461a7e55825d894538a9c0c57",
    source_logical_fingerprint=_EASY50_SOURCE_LOGICAL_FINGERPRINT,
    source_manifest_sha256=_EASY50_SOURCE_MANIFEST_SHA256,
    source_archive_sha256=_EASY50_SOURCE_ARCHIVE_SHA256,
    question_index_sha256=_EASY50_QUESTION_INDEX_SHA256,
    prepared_artifact_sha256="96145af931a2b468d2c2cddc12df2695d0a7354d7d72636eb75af8a0f5f3a968",
    prepared_config_fingerprint="74ec89a3206eecf3",
    panel_registration_sha256="38128cd2e6e88bf7",
    source_audit_fingerprint="d472eb82e461156c",
    question_ids=EASY50_ORDER,
    workers_per_item=4,
    worker_prompt_tokens=64_000,
    latent_steps=40,
    span_width=16,
    ratios=(),
    selectors=(),
    arms=tuple(arm for arm in FANOUTQA_NATURAL_DEV50.arms if arm.arm_id in _TEXT_ARMS),
    sealed_qwen_policies=tuple(
        pair for pair in FANOUTQA_NATURAL_DEV50.sealed_qwen_policies if pair[0] in _TEXT_ARMS
    ),
    sample_tags=("s0", "s1", "s2"),
    answer_seed_namespace=_EASY50_ANSWER_SEED_NAMESPACE,
    report_seed_base=20_260_913,
    answer_ceiling=24_000,
    # The two values this profile moves.
    report_ceiling=16_000,
    report_presence_penalty=1.5,
    max_model_len=131_072,
    benchmark_key=EASY50_TEXT_KEY,
    arm_profiles=("longbench-coa-text-4arm-v1",),
    topology_key=CHAIN_TOPOLOGY_KEY,
    run_id_stem="longbench-coa-easy50-text-t4-v1",
    output_prefix="longbench-v2-coa",
    declared_passes=((0, 50),),
    payload_layout="chain-terminal-v1",
    scorer="longbench-v2-official-extraction-strip-star-paren-then-bare-v1",
    score_fields=("correct", "answered"),
    prompt_builder=PROMPT_BUILDER_CONDENSE,
)

# The bounded law on the same panel. At every hop the whole sequence is re-ranked
# and at most the arm's row budget survives, one fixed total whatever the document
# and however long the chain. See docs/benchmarks.md, Arm rosters.
_BOUNDED_ARMS = tuple(
    ArmSpec(
        arm_id=f"latent_query_support_bounded_b{budget}",
        channel="latent",
        selector="support",
        sender_role="same-model",
        budget_law="bounded",
        budget_rows=budget,
    )
    for budget in BOUNDED_ROW_BUDGETS
)
_BOUNDED_POLICIES = tuple(
    (arm.arm_id, f"qwen3_8b_bounded_b{arm.budget_rows}_w16_support") for arm in _BOUNDED_ARMS
)
LONGBENCH_COA_EASY50_BOUNDED = BenchmarkProfile(
    profile_id="longbench-v2-coa-easy50-bounded-t4-n50-sealed-2026-09-13-v1",
    source_commit="63cf79ef012f56e61bffd61654fde20f1b7baf82",
    source_logical_fingerprint=_EASY50_SOURCE_LOGICAL_FINGERPRINT,
    source_manifest_sha256=_EASY50_SOURCE_MANIFEST_SHA256,
    source_archive_sha256=_EASY50_SOURCE_ARCHIVE_SHA256,
    question_index_sha256=_EASY50_QUESTION_INDEX_SHA256,
    prepared_artifact_sha256="fb0a0fe4fe7d6d552fe08e14dd6b3d011a77945accfaf86188672fdc1502e621",
    prepared_config_fingerprint="fc3bba02c44b45b0",
    panel_registration_sha256="6dc874a02ba8c36f",
    source_audit_fingerprint="f25544894dd30e96",
    question_ids=EASY50_ORDER,
    workers_per_item=4,
    worker_prompt_tokens=64_000,
    latent_steps=40,
    span_width=16,
    ratios=(),
    selectors=("support",),
    arms=_BOUNDED_ARMS,
    sealed_qwen_policies=_BOUNDED_POLICIES,
    sample_tags=("s0", "s1", "s2"),
    answer_seed_namespace=_EASY50_ANSWER_SEED_NAMESPACE,
    report_seed_base=20_260_913,
    answer_ceiling=24_000,
    report_ceiling=24_000,
    max_model_len=131_072,
    benchmark_key=EASY50_BOUNDED_KEY,
    arm_profiles=("longbench-coa-bounded-6arm-v1",),
    topology_key=CHAIN_TOPOLOGY_KEY,
    run_id_stem="longbench-coa-easy50-bounded-t4-v1",
    output_prefix="longbench-v2-coa",
    declared_passes=((0, 50),),
    payload_layout="chain-terminal-v1",
    scorer="longbench-v2-official-extraction-strip-star-paren-then-bare-v1",
    score_fields=("correct", "answered"),
    prompt_builder=PROMPT_BUILDER_CONDENSE,
)

__all__ = (
    "EASY50_ORDER",
    "LONGBENCH_COA_EASY50_BOUNDED",
    "LONGBENCH_COA_EASY50_RERANK",
    "LONGBENCH_COA_EASY50_TEXT",
    "UNMINTED_SENTINEL",
)
