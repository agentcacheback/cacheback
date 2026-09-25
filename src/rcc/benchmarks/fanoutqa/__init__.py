"""The FanOutQA natural panel and the eleven-arm roster it runs."""

import hashlib
from dataclasses import replace

from rcc.benchmarks.protocol import ArmSpec, BenchmarkProfile

_RATIOS = (2, 4, 8, 16, 32, 64, 128)
_ARMS = (
    ArmSpec(arm_id="issue_only", channel="floor"),
    ArmSpec(arm_id="text_primary", channel="text", sender_role="primary"),
    ArmSpec(
        arm_id="text_medium",
        channel="text",
        sender_role="registered-medium",
    ),
    ArmSpec(
        arm_id="text_small",
        channel="text",
        sender_role="registered-small",
    ),
    *(
        ArmSpec(
            arm_id=f"latent_query_support_r{ratio}",
            channel="latent",
            selector="support",
            retention_ratio=ratio,
            sender_role="same-model",
        )
        for ratio in _RATIOS
    ),
)

FULL_ARM = ArmSpec(arm_id="full", channel="full", sender_role="same-model")


def arms_with_full(profile: BenchmarkProfile) -> tuple[ArmSpec, ...]:
    """Return one profile's arms plus the full-memory control."""
    return (*profile.arms, FULL_ARM)


def shared_answer_seeds(
    profile: BenchmarkProfile,
    qid: str,
    arm_id: str,
) -> tuple[int, ...]:
    """Return one item and arm's receiver seeds, the full control included."""
    if arm_id != FULL_ARM.arm_id:
        return profile.answer_seeds(qid, arm_id)
    if qid not in profile.question_ids:
        raise ValueError(f"unregistered FanOutQA question {qid!r}")
    return tuple(
        int(
            hashlib.sha256(
                "|".join(
                    (profile.answer_seed_namespace, qid, arm_id, "answer", sample_tag)
                ).encode()
            ).hexdigest()[:8],
            16,
        )
        for sample_tag in profile.sample_tags
    )


#: The panel label the M=3 flat-interleave plan digests and the resident decode
#: records are signed under. It names the sealed fan-out recipe, not the panel a
#: run draws its items from, so a banked digest is re-derived from one value.
SEALED_M3_PLAN_LABEL = "fanoutqa-m3-n50-sealed-2026-08-13-v1"

_SEALED_QWEN_POLICIES = (
    ("issue_only", "none"),
    ("text_primary", "qwen3_8b_text"),
    ("text_medium", "qwen3_8b_calls_qwen3_4b_text"),
    ("text_small", "qwen3_8b_calls_qwen3_1_7b_text"),
    *((f"latent_query_support_r{ratio}", f"qwen3_8b_r{ratio}_w16_support") for ratio in _RATIOS),
)

# The natural-content panel: fifty official dev questions disjoint from the
# selector-development pool, each worker carrying real page text only. The roster rule
# and the token ceiling are under docs/benchmarks.md, FanOutQA, natural fifty-question panel.
_NATURAL_DEV50_QUESTION_IDS = (
    "95de313fdfef9d01",
    "c58ac65553f24640",
    "b3b15d0277166b1d",
    "832e7529292aa805",
    "6f9b579bc0a11d6d",
    "6533545d9f5eb7b6",
    "d098623a8b75ec0a",
    "a6545687ea8228bc",
    "a040ce757ab2fa7e",
    "347504ffb097442d",
    "efac681de642e27c",
    "aedaf67f40b712c7",
    "445e0f67be85856c",
    "f71b3ba07d3af7d6",
    "ca085ff48e269f74",
    "065b97c2df131209",
    "cc4e6774f97b0101",
    "66d8b25fb1cc7141",
    "a5f1f57ef5909b83",
    "4172435efe8bdd8b",
    "9eed0f868068dc12",
    "0a950dfa942cf3ad",
    "98ced84a1a9df028",
    "ff866ee3e2bf4820",
    "50fa232e3f44887a",
    "c4c57d0e2a79f7fc",
    "dabeedf8fc8758ee",
    "1244f9eb0096a5f2",
    "9dbfeb76f11979c1",
    "d37ba10545924859",
    "96f91d21a3270a89",
    "6c7873f9700736f7",
    "8e62c5c69ce8c58d",
    "21b5a69403eeda11",
    "542983d2cb1630e0",
    "440957e02d1c61da",
    "4a484ffa4676b4ce",
    "a3efebf047c6d6ec",
    "d001e4f3d27fac66",
    "d7ab22740996aea9",
    "085aa1592424a7be",
    "8047613f8e12e507",
    "8c658fbd12f23a02",
    "3f27dbe4ac7b2bc8",
    "291b1c294b1a69a7",
    "66d9d79aea844c0d",
    "1011a17b0afd13e9",
    "22892e5a38e781cc",
    "a0216c68b94f857e",
    "67fc04df63afb29c",
)

FANOUTQA_NATURAL_DEV50 = BenchmarkProfile(
    profile_id="fanoutqa-m3-natural-dev50-2026-09-06-v1",
    # The commit the prepared bytes are bound to.
    source_commit="0fb0b168249d0bebdb5c6ea2e9d3ea9820d6d0fd",
    # sha256 of the bundle's panel.json, the manifest digest, and the archive.
    source_logical_fingerprint="76f752acb2f3543115c93c95d4b06491295d40d8ef633dbcbd41cefebb71da7f",
    source_manifest_sha256="8d4397a8633b8f3c130f162299fccd1d2c83af3d26e70f8ac6ae57602a1d825e",
    source_archive_sha256="1874eafe4119bfd433cb535cb47d4075db771b6d25fad31a42873f80fcb2be58",
    question_index_sha256="359300b029c6891567816f351bf8786e9b018d7af8a1a44b7da9ba5ef4651288",
    # Built from the bundle with the Qwen3-8B pin; the prepared items carry no
    # wall clock, so a rebuild reproduces this digest. A native-sender family
    # builds its own artifact and pins its digest through
    # `RouteFamily.benchmark_profile`.
    prepared_artifact_sha256="e54146b1f7d972a48fa0f421540ef91c376c47e6f7e185cf5d529cf613f1dd6c",
    # Functions of the registration below, recomputable from this code.
    prepared_config_fingerprint="76a5e66c4f18d031",
    panel_registration_sha256="f09ed6843270566c",
    source_audit_fingerprint="acbfc55b7d2aad04",
    question_ids=_NATURAL_DEV50_QUESTION_IDS,
    workers_per_item=3,
    worker_prompt_tokens=53_248,
    latent_steps=40,
    span_width=16,
    ratios=_RATIOS,
    selectors=("support",),
    arms=_ARMS,
    sealed_qwen_policies=_SEALED_QWEN_POLICIES,
    sample_tags=("s0", "s1", "s2"),
    answer_seed_namespace="0aa9ddc81cd632ae77b4c08803fc6d269575e3bbdd97908539c637a896274dfd",
    report_seed_base=20_260_811,
    answer_ceiling=24_000,
    report_ceiling=24_000,
    max_model_len=131_072,
    benchmark_key="fanoutqa-natural-dev50",
    arm_profiles=(
        "fanoutqa-m3-11arm-sealed",
        "fanoutqa-m3-12arm-full-sealed",
        "gemma-natural-dev50-12arm-native-v1",
    ),
    topology_key="fanout-m3",
    run_id_stem="fanoutqa-m3-n15-v1",
    output_prefix="fanoutqa",
    declared_passes=((0, 50),),
    payload_layout="flat-interleave-v1",
    scorer="fanoutqa-string-proxy-loose-strict-v1",
    score_fields=("loose", "strict", "n_leaves"),
    prompt_builder="fanoutqa-coordinator-handoff-v1",
    prompt_geometry="natural",
)

_LATENT_ARM_IDS = frozenset(arm.arm_id for arm in _ARMS if arm.channel == "latent")

#: The natural panel cut to its seven support-latent arms. Same items, same
#: prompts, same seeds, same decode, same ceilings: only the roster moves, so
#: the config fingerprint and the source audit are the natural profile's, while
#: the prepared artifact and the serving registration digest (both carry the
#: policies) are its own.
FANOUTQA_NATURAL_DEV50_LATENT = replace(
    FANOUTQA_NATURAL_DEV50,
    profile_id="fanoutqa-m3-natural-dev50-latent-2026-09-19-v1",
    # The prepared items differ from the natural profile's only by the roster
    # they carry.
    prepared_artifact_sha256="9dac99a803938f0fa3da09eb7eca6472ed91fa1c3d7d83fb9ea41837cbbea7fa",
    panel_registration_sha256="3fdcfccc57ab18cf",
    arms=tuple(arm for arm in _ARMS if arm.arm_id in _LATENT_ARM_IDS),
    sealed_qwen_policies=tuple(
        pair for pair in _SEALED_QWEN_POLICIES if pair[0] in _LATENT_ARM_IDS
    ),
    benchmark_key="fanoutqa-natural-dev50-latent",
    arm_profiles=("fanoutqa-m3-7latent-sealed",),
)

# The natural profile precedes its latent cut: a construction receipt resolves
# to the first profile carrying its source fingerprint.
REGISTERED_PROFILES = (FANOUTQA_NATURAL_DEV50, FANOUTQA_NATURAL_DEV50_LATENT)


def profile_by_id(profile_id: str) -> BenchmarkProfile:
    """Return the FanOutQA profile with this id, raising when there is none."""
    for profile in REGISTERED_PROFILES:
        if profile.profile_id == profile_id:
            return profile
    raise ValueError(f"unregistered FanOutQA profile {profile_id!r}")


__all__ = (
    "FANOUTQA_NATURAL_DEV50",
    "FANOUTQA_NATURAL_DEV50_LATENT",
    "FULL_ARM",
    "REGISTERED_PROFILES",
    "SEALED_M3_PLAN_LABEL",
    "arms_with_full",
    "profile_by_id",
    "shared_answer_seeds",
)
