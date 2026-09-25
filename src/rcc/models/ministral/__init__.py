"""The pinned Ministral 3 model profile and its registered arm roster."""

from rcc.models.decode import decode_protocol
from rcc.models.protocol import ModelProfile, PhysicalArm, RuntimeProfile
from rcc.topologies.fanout.layout import FLAT_INTERLEAVE_LAYOUT

#: The registered Ministral policy vocabulary: the family name every policy
#: carries, its ratio ladder, and the one registered span width. The arm names
#: below are built from these.
MINISTRAL_FAMILY = "ministral3_14b"
MINISTRAL_RATIOS = (2, 4, 8, 16, 32, 64, 128)
MINISTRAL_SPAN_WIDTH = 16
# Reasoning-2512, not Instruct-2512-BF16: the lane runs with thinking on, and
# the thought channel exists only in the Reasoning repositories, whose
# tekken.json carries [THINK] and [/THINK] as special tokens.
_MINISTRAL_14B = (
    "mistralai/Ministral-3-14B-Reasoning-2512",
    "51f9210f3cd20f3452a80d5819d15dc61cc50630",
)
_MINISTRAL_8B = (
    "mistralai/Ministral-3-8B-Reasoning-2512",
    "81eaece1948f3875421d9a45bc55487d10e2d894",
)
_MINISTRAL_3B = (
    "mistralai/Ministral-3-3B-Reasoning-2512",
    "4a36357c811bf511a7b625d132e12f22408aac91",
)
# Read off the pinned revision and holding on all three senders: rope_type
# yarn, factor 16.0, original 16384, beta_fast 32.0, beta_slow 1.0, mscale
# 1.0, and the separate Llama 4 query-scale beta of 0.1.
_NATIVE_CONTEXT = (
    "checkpoint-native-yarn-factor16-original16384-betafast32-betaslow1-mscale1"
    "-llama4-query-scale-beta0.1"
)


def _arm(
    semantic_arm: str,
    policy: str,
    sender: tuple[str, str] | None,
    selector: str | None = None,
    *,
    payload_layout: str | None = None,
) -> PhysicalArm:
    checkpoint, revision = sender if sender is not None else (None, None)
    return PhysicalArm(
        semantic_arm=semantic_arm,
        policy=policy,
        sender_checkpoint=checkpoint,
        sender_revision=revision,
        sender_tokenizer=checkpoint,
        sender_tokenizer_revision=revision,
        selector=selector,
        sender_native_max_model_len=262_144 if sender is not None else None,
        sender_context_extension=_NATIVE_CONTEXT if sender is not None else None,
        payload_layout=payload_layout,
    )


_PHYSICAL_ARMS = (
    _arm("issue_only", "none", None),
    _arm("text_primary", f"{MINISTRAL_FAMILY}_text", _MINISTRAL_14B),
    _arm("text_medium", f"{MINISTRAL_FAMILY}_calls_ministral3_8b_text", _MINISTRAL_8B),
    _arm("text_small", f"{MINISTRAL_FAMILY}_calls_ministral3_3b_text", _MINISTRAL_3B),
    *(
        _arm(
            f"latent_query_support_r{ratio}",
            f"{MINISTRAL_FAMILY}_r{ratio}_w{MINISTRAL_SPAN_WIDTH}_support",
            _MINISTRAL_14B,
            "support",
            payload_layout=FLAT_INTERLEAVE_LAYOUT,
        )
        for ratio in MINISTRAL_RATIOS
    ),
    _arm(
        "full",
        f"{MINISTRAL_FAMILY}_full",
        _MINISTRAL_14B,
        payload_layout=FLAT_INTERLEAVE_LAYOUT,
    ),
)

MINISTRAL_RUNTIME = RuntimeProfile(
    profile_id=(
        "ministral3-reasoning-2512-vllm-0.26.0-transformers-5.13.0-torch-2.11.0-bf16-target-v3"
    ),
    python="3.10",
    torch="2.11.0",
    cuda="13.0",
    # 5.13.0, the same pin Gemma runs: one Transformers version across the
    # families.
    transformers="5.13.0",
    vllm="0.26.0",
    model_revisions=(_MINISTRAL_14B, _MINISTRAL_8B, _MINISTRAL_3B),
    tokenizer_revisions=(_MINISTRAL_14B, _MINISTRAL_8B, _MINISTRAL_3B),
    engine_flags=(
        ("dtype", "bfloat16"),
        # The pinned snapshot carries tekken.json but no tokenizer.json, so
        # there is no Transformers fast tokenizer: mistral-common against
        # tekken.json is the only path, and the prompt arrives as token ids.
        ("tokenizer_mode", "mistral"),
        ("config_format", "mistral"),
        ("load_format", "mistral"),
        ("language_model_only", "true"),
        ("tensor_parallel_size", "1"),
        ("v1_multiprocessing", "false"),
        # Serves the panel's largest request rather than the native 262144: the
        # full arm's three 50K contexts plus manager prompt and 24000 answer
        # ceiling stay under it, and the KV floor it implies fits an H100.
        ("max_model_len", "180224"),
        ("max_num_seqs", "16"),
        ("enable_prefix_caching", "false"),
        ("prompt_transport", "direct-mistral-common-token-ids"),
        ("hf_roll_cache", "dense-dynamic-cache"),
        ("context_extension", _NATIVE_CONTEXT),
        # Sampling and the thinking regime are not engine flags: they are the
        # family's decode protocol, registered under the key "ministral3" in
        # rcc.models.decode.
        ("producer_backend", "vllm-prefill-zero-copy-hf-roll-flat-payload"),
    ),
    auxiliary_packages=(("mistral-common[image]", "1.11.7"),),
)

MINISTRAL = ModelProfile(
    model_id="ministral3-14b",
    checkpoint=_MINISTRAL_14B[0],
    revision=_MINISTRAL_14B[1],
    tokenizer=_MINISTRAL_14B[0],
    tokenizer_revision=_MINISTRAL_14B[1],
    decode=decode_protocol("ministral3"),
    runtime=MINISTRAL_RUNTIME,
    lifecycle="split-fleet-controller",
    physical_arms=_PHYSICAL_ARMS,
)


def ministral_semantic_bindings() -> dict[str, str]:
    """Bind every executable Ministral policy name to its semantic arm.

    The split fleet is placed by semantic arm, so this is where the Ministral
    roster meets the shared table in :mod:`rcc.hardware.placements`.
    """
    bindings = {arm.policy: arm.semantic_arm for arm in _PHYSICAL_ARMS}
    if len(bindings) != len(_PHYSICAL_ARMS) or len(set(bindings.values())) != len(bindings):
        raise RuntimeError("Ministral placement binding does not cover the registered arm roster")
    return bindings


__all__ = (
    "MINISTRAL",
    "MINISTRAL_FAMILY",
    "MINISTRAL_RATIOS",
    "MINISTRAL_RUNTIME",
    "MINISTRAL_SPAN_WIDTH",
    "ministral_semantic_bindings",
)
