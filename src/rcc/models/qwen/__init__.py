"""The pinned Qwen 3 model profile and the route family built on it."""

from rcc.models.decode import decode_protocol
from rcc.models.protocol import ModelProfile, PhysicalArm, RuntimeProfile
from rcc.models.route import RouteFamily
from rcc.topologies.chain import BOUNDED_ROW_BUDGETS
from rcc.topologies.fanout.layout import FLAT_INTERLEAVE_LAYOUT

_RATIOS = (2, 4, 8, 16, 32, 64, 128)
#: The ratios the re-ranked handoff law runs. Ratio 2 is absent: under that law
#: the longest item of the easy panel exceeds the served window at the final
#: hop, so the lane implements no r2 rerank seat.
_RERANK_RATIOS = _RATIOS[1:]
_QWEN_8B = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
_QWEN_4B = ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c")
_QWEN_1P7B = ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")


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
        payload_layout=payload_layout,
    )


_PHYSICAL_ARMS = (
    _arm("issue_only", "none", None),
    _arm("text_primary", "qwen3_8b_text", _QWEN_8B),
    _arm("text_medium", "qwen3_8b_calls_qwen3_4b_text", _QWEN_4B),
    _arm("text_small", "qwen3_8b_calls_qwen3_1_7b_text", _QWEN_1P7B),
    *(
        _arm(
            f"latent_query_support_r{ratio}",
            f"qwen3_8b_r{ratio}_w16_support",
            _QWEN_8B,
            "support",
            payload_layout=FLAT_INTERLEAVE_LAYOUT,
        )
        for ratio in _RATIOS
    ),
    # The re-ranked handoff law on the chain topology: the same checkpoint,
    # selector, and layout with one different count inside the cut, so every
    # carried row is scored again at every later hop. Chain only.
    *(
        _arm(
            f"latent_query_support_rerank_r{ratio}",
            f"qwen3_8b_rerank_r{ratio}_w16_support",
            _QWEN_8B,
            "support",
            payload_layout=FLAT_INTERLEAVE_LAYOUT,
        )
        for ratio in _RERANK_RATIOS
    ),
    # The bounded law: one fixed carried row count per arm, whatever the
    # document. Chain only, like the rerank seats.
    *(
        _arm(
            f"latent_query_support_bounded_b{budget}",
            f"qwen3_8b_bounded_b{budget}_w16_support",
            _QWEN_8B,
            "support",
            payload_layout=FLAT_INTERLEAVE_LAYOUT,
        )
        for budget in BOUNDED_ROW_BUDGETS
    ),
)

QWEN_RUNTIME = RuntimeProfile(
    profile_id="qwen-vllm-0.11.1-transformers-4.57.1-torch-2.9.0-cu128-v4",
    python="3.10.12",
    torch="2.9.0+cu128",
    cuda="12.8",
    transformers="4.57.1",
    vllm="0.11.1",
    model_revisions=(_QWEN_8B, _QWEN_4B, _QWEN_1P7B),
    tokenizer_revisions=(_QWEN_8B, _QWEN_4B, _QWEN_1P7B),
    engine_flags=(
        ("dtype", "bfloat16"),
        ("max_model_len", "131072"),
        ("max_num_seqs", "16"),
        ("producer_gpu_memory_utilization", "0.40"),
        ("producer_max_model_len", "50000"),
        ("producer_max_num_batched_tokens", "50000"),
        ("producer_enforce_eager", "true"),
        ("producer_enable_prefix_caching", "false"),
        ("producer_disable_log_stats", "true"),
        ("producer_disable_hybrid_kv_cache_manager", "true"),
        ("producer_kv_connector", "RCCConnector"),
        ("producer_kv_role", "kv_both"),
        ("producer_kv_connector_module_path", "rcc.injector.connector"),
        ("v1_multiprocessing", "false"),
        ("rope_scaling", "yarn:factor=4.0:original=32768"),
        ("producer_backend", "vllm-prefill-zero-copy-hf-roll"),
        ("prepared_source_commit", "309fad8de6896d042fffbeed284e9384b6ade26f"),
        ("receiver_gpu_memory_utilization", "0.80"),
    ),
    auxiliary_packages=(("tokenizers", "0.22.0"),),
)

QWEN = ModelProfile(
    model_id="qwen3-8b",
    checkpoint=_QWEN_8B[0],
    revision=_QWEN_8B[1],
    tokenizer=_QWEN_8B[0],
    tokenizer_revision=_QWEN_8B[1],
    decode=decode_protocol("qwen3"),
    runtime=QWEN_RUNTIME,
    lifecycle="split-fleet-controller",
    physical_arms=_PHYSICAL_ARMS,
)

#: The served window: YaRN factor 4 over the 32768 native window, passed to
#: vLLM as hf_overrides by the engine and backend seams.
QWEN_HF_OVERRIDES = {
    "max_position_embeddings": 131_072,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32_768,
    },
}

QWEN_FAMILY = RouteFamily(
    profile=QWEN,
    lane="qwen",
    policy_prefix="qwen3_8b",
    hf_architecture="Qwen3ForCausalLM",
    # <|im_end|> and <|endoftext|> at the pinned revision.
    stop_token_ids=(151645, 151643),
    think_open="<think>",
    think_close="</think>",
    # The reserved Qwen3 ids for the reasoning delimiters, from
    # added_tokens_decoder of tokenizer_config.json at the pinned revision.
    think_open_token_ids=(151667,),
    think_close_token_ids=(151668,),
    literal_think_tags=False,
    hf_overrides=QWEN_HF_OVERRIDES,
)

__all__ = ("QWEN", "QWEN_FAMILY", "QWEN_HF_OVERRIDES", "QWEN_RUNTIME")
