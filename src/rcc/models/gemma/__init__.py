"""The pinned Gemma 4 model profile and its registered arm roster."""

from rcc.models.decode import decode_protocol
from rcc.models.protocol import ModelProfile, PhysicalArm, RuntimeProfile

_RATIOS = (2, 4, 8, 16, 32, 64, 128)
_GEMMA_12B = (
    "google/gemma-4-12B-it",
    "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
)
_GEMMA_E4B = (
    "google/gemma-4-E4B-it",
    "ee0ef6023621cff504d758262d4e04895a5af4a2",
)
_GEMMA_E2B = (
    "google/gemma-4-E2B-it",
    "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
)


def _arm(
    semantic_arm: str,
    policy: str,
    sender: tuple[str, str] | None,
    selector: str | None = None,
    *,
    native_max_model_len: int | None = None,
    context_extension: str | None = None,
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
        sender_native_max_model_len=native_max_model_len,
        sender_context_extension=context_extension,
        payload_layout=payload_layout,
    )


_PHYSICAL_ARMS = (
    _arm("issue_only", "gemma4_12b_floor", None),
    _arm(
        "text_primary",
        "gemma4_12b_text",
        _GEMMA_12B,
        native_max_model_len=262_144,
        context_extension="none",
    ),
    _arm(
        "text_medium",
        "gemma4_12b_calls_gemma4_e4b_text",
        _GEMMA_E4B,
        native_max_model_len=131_072,
        context_extension="none",
    ),
    _arm(
        "text_small",
        "gemma4_12b_calls_gemma4_e2b_text",
        _GEMMA_E2B,
        native_max_model_len=131_072,
        context_extension="none",
    ),
    *(
        _arm(
            f"latent_query_support_r{ratio}",
            f"gemma4_12b_r{ratio}_w16_support",
            _GEMMA_12B,
            "support",
            payload_layout="flat-interleave-v1",
        )
        for ratio in _RATIOS
    ),
    _arm("full", "gemma4_12b_full", _GEMMA_12B),
)

GEMMA_RUNTIME = RuntimeProfile(
    profile_id=("gemma-v16-vllm-0.26.0-transformers-5.13.0-torch-2.11.0-text-12b-e4b-e2b-v4"),
    python="3.10",
    torch="2.11.0",
    cuda="13.0",
    transformers="5.13.0",
    vllm="0.26.0",
    model_revisions=(_GEMMA_12B, _GEMMA_E4B, _GEMMA_E2B),
    tokenizer_revisions=(_GEMMA_12B, _GEMMA_E4B, _GEMMA_E2B),
    engine_flags=(
        ("attention_backend", "FLASH_ATTN"),
        ("flash_attn_version", "4"),
        ("dtype", "bfloat16"),
        ("enable_prompt_embeds", "true"),
        ("enable_prefix_caching", "false"),
        ("enforce_eager", "false"),
        ("generation_config", "auto"),
        ("gpu_memory_utilization", "0.85"),
        ("kv_connector", "GemmaCaptureConnector"),
        ("kv_connector_module_path", "rcc.models.gemma.capture_connector"),
        ("kv_role", "kv_both"),
        ("language_model_only", "true"),
        ("logprobs_mode", "processed_logits"),
        ("max_logprobs", "-1"),
        ("max_model_len", "262144"),
        ("max_num_seqs", "16"),
        ("producer_backend", "resident-vllm-zero-copy-hf-roll"),
        ("selection_artifact_profile", "execution-arm-keyed-no-unused-audition-v1"),
        ("text_attention_backend", "FLASH_ATTN"),
        ("text_flash_attn_version", "4"),
        ("text_dtype", "bfloat16"),
        ("text_enable_prefix_caching", "false"),
        ("text_enforce_eager", "false"),
        ("text_generation_config", "auto"),
        ("text_gpu_memory_utilization", "0.85"),
        ("text_language_model_only", "true"),
        ("text_max_model_len", "131072"),
        ("text_max_num_seqs", "16"),
    ),
    auxiliary_packages=(("accelerate", "1.14.0"), ("tokenizers", "0.22.2")),
)

# The shared adapter runs the resident receiver and selector path and binds the
# source, the logical seeds, the text senders, and the arm projection.
GEMMA = ModelProfile(
    model_id="gemma4-12b-it",
    checkpoint=_GEMMA_12B[0],
    revision=_GEMMA_12B[1],
    tokenizer=_GEMMA_12B[0],
    tokenizer_revision=_GEMMA_12B[1],
    decode=decode_protocol("gemma4"),
    runtime=GEMMA_RUNTIME,
    lifecycle="split-fleet-controller",
    physical_arms=_PHYSICAL_ARMS,
)

__all__ = ("GEMMA", "GEMMA_RUNTIME")
