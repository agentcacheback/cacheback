"""The pinned Nemotron Nano profile, arm ladder, and route family.

The route family carries what the shared Qwen route reads: native sender
prompts, a template-opened reasoning block, and payload rows in the user turn.
"""

from dataclasses import replace

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.decode import DecodeProtocol
from rcc.models.gemma import GEMMA_RUNTIME
from rcc.models.nemotron.kernels import DOWNLOAD_PINS, KERNEL_PINS
from rcc.models.nemotron.selection_contract import PRODUCTION_ARMS
from rcc.models.protocol import ModelProfile, PhysicalArm
from rcc.models.route import RouteFamily
from rcc.models.selection import ScoreAdapter
from rcc.topologies.chain import BOUNDED_ROW_BUDGETS
from rcc.topologies.fanout.layout import FLAT_INTERLEAVE_LAYOUT

NANO_12B = ("nvidia/NVIDIA-Nemotron-Nano-12B-v2", "f428df0ec725fed457b89cfca54dc26500fb88c1")
NANO_9B = ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "6533e8de2c68e4536bf7c411d7a3ce5734111476")
NANO_4B = ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f")
PRODUCER_ROUTE = "nemotron-vllm-prefill-aliased-hybrid-roll-v2"
_PREFIX = "nemotron_nano_12b_v2"


def _arm(name: str, suffix: str, sender: tuple[str, str] | None) -> PhysicalArm:
    latent = name.startswith("latent_")
    checkpoint, revision = sender or (None, None)
    return PhysicalArm(
        semantic_arm=name,
        policy=f"{_PREFIX}_{suffix}" if sender else "none",
        sender_checkpoint=checkpoint,
        sender_revision=revision,
        sender_tokenizer=checkpoint,
        sender_tokenizer_revision=revision,
        sender_native_max_model_len=(262_144 if sender == NANO_4B else 131_072) if sender else None,
        sender_context_extension="none" if sender else None,
        selector="support" if latent else None,
        payload_layout=FLAT_INTERLEAVE_LAYOUT if latent else None,
    )


#: Producer capture window. A padded geometry renders every worker to exactly
#: 50,000 prompt tokens plus 40 latent steps and natural content to at most
#: 51,658, so the window carries headroom rather than one exact width.
CAPTURE_MAX_MODEL_LEN = 53_248

#: sha256 of the Nemotron natural prepared artifact (`items.pkl`): the dev50
#: texts tokenized with NANO_12B plus the three native sender prompt records.
#: The prepared items pickle the serving registration, so the digest covers it.
NATURAL_PREPARED_SHA256 = "8f10771f7669c95696e4bfaaa31b0fdaa940544fa76d50f3128dee8beb68d1be"

NEMOTRON_RUNTIME = replace(
    GEMMA_RUNTIME,
    profile_id="nemotron-vllm-0.26.0-transformers-5.13.0-native-hybrid-deterministic-v3",
    python="3.10.12",
    torch="2.11.0+cu130",
    auxiliary_packages=(
        ("accelerate", "1.14.0"),
        ("tokenizers", "0.22.2"),
        *DOWNLOAD_PINS,
        ("kernels", "0.15.2"),
    ),
    model_revisions=(NANO_12B, NANO_9B, NANO_4B),
    tokenizer_revisions=(NANO_12B, NANO_9B, NANO_4B),
    engine_flags=(
        ("dtype", "bfloat16"),
        ("max_model_len", "131072"),
        ("max_num_seqs", "16"),
        ("mamba_ssm_cache_dtype", "float32"),
        ("producer_backend", PRODUCER_ROUTE),
        ("capture_max_model_len", str(CAPTURE_MAX_MODEL_LEN)),
        ("capture_max_num_batched_tokens", "4096"),
        ("capture_attention_backend", "FLASH_ATTN"),
        ("capture_mamba_cache_mode", "none"),
        ("causal_conv1d_revision", KERNEL_PINS[0][1]),
        ("mamba_ssm_revision", KERNEL_PINS[1][1]),
        ("native_sender_codec", "nemotron-native-thinking-v1"),
        ("text_small_temperature", "1.0"),
        ("attention_layers", "6"),
        ("enable_prefix_caching", "false"),
        ("latent_alignment", "output-to-input-norm-matched-v1"),
        ("latent_alignment_reg", "1e-5"),
        ("latent_budget", "original-total-sink-and-tail"),
        ("latent_score", "receiver-conditioned-attention-only-p2-a2"),
        ("latent_span_width", "16"),
        ("latent_steps", "40"),
        ("inductor_deterministic", "true"),
        ("TORCHINDUCTOR_DETERMINISTIC", "1"),
        ("VLLM_DISABLE_COMPILE_CACHE", "0"),
        ("VLLM_USE_AOT_COMPILE", "1"),
        ("VLLM_FORCE_AOT_LOAD", "0"),
        ("eager_deterministic_algorithms", "false"),
    ),
)
NEMOTRON = ModelProfile(
    model_id="nemotron-nano-12b-v2",
    checkpoint=NANO_12B[0],
    revision=NANO_12B[1],
    tokenizer=NANO_12B[0],
    tokenizer_revision=NANO_12B[1],
    decode=DecodeProtocol("nemotron", 0.6, 0.95, None, True),
    runtime=NEMOTRON_RUNTIME,
    lifecycle="split-fleet-controller",
    physical_arms=(
        _arm("issue_only", "", None),
        _arm("text_primary", "text", NANO_12B),
        _arm("text_medium", "calls_nano9b_text", NANO_9B),
        _arm("text_small", "calls_nano4b_text", NANO_4B),
        *(
            _arm(f"latent_query_support_r{r}", f"r{r}_w16_support", NANO_12B)
            for r in (2, 4, 8, 16, 32, 64, 128)
        ),
        # The chain's re-ranked handoff seats: the same checkpoint and
        # selector, with the chain's count inside the cut. There is no r2 seat,
        # because it does not fit the served window on this tokenizer.
        *(
            _arm(f"latent_query_support_rerank_r{r}", f"rerank_r{r}_w16_support", NANO_12B)
            for r in (4, 8, 16, 32, 64, 128)
        ),
        # The chain's bounded seats: one fixed carried memory, the same six
        # rungs the Qwen lane implements.
        *(
            _arm(f"latent_query_support_bounded_b{b}", f"bounded_b{b}_w16_support", NANO_12B)
            for b in BOUNDED_ROW_BUDGETS
        ),
    ),
    execution_order=PRODUCTION_ARMS,
)


class NemotronFamily(RouteFamily):
    """Read template-opened reasoning without rescuing unfinished traces."""

    def score_adapter(self, semantic_arm: str) -> ScoreAdapter | None:
        """Use the shared attention-only score and original W16 budget."""
        arm = next(a for a in self.profile.physical_arms if a.semantic_arm == semantic_arm)
        if arm.selector_recipe is not None:
            raise ValueError("active Nemotron arms require attention-only selection")
        return None

    @property
    def native_sender_prompts(self) -> bool:
        """Seal each sender's own tokenizer and chat template."""
        return True

    @property
    def prefilled_block(self) -> bool:
        """Both native templates open the reasoning block themselves."""
        return True

    def sender_family(self, semantic_arm: str) -> RouteFamily:
        """Use the 4B's distinct ChatML vocabulary and reserved think tokens."""
        if semantic_arm == "text_small":
            return replace(
                self,
                profile=replace(self.profile, decode=replace(self.profile.decode, temperature=1.0)),
                stop_token_ids=(2, 11),
                think_open_token_ids=(12,),
                think_close_token_ids=(13,),
                literal_think_tags=False,
                thinking_system_prompt=None,
            )
        return self

    def post_think_handoff(self, raw_output: str, *, ended: bool) -> str:
        """Require visible text after a closed reasoning block."""
        return self.post_think_report(raw_output)

    def benchmark_profile(self, profile: BenchmarkProfile) -> BenchmarkProfile:
        """Pin this family's own natural artifact digest.

        A native-sender family tokenizes the same texts with its own tokenizer
        and attaches its own sender prompt records, so its digest is its own.
        """
        if profile.prompt_geometry != "natural":
            return profile
        if profile.profile_id != FANOUTQA_NATURAL_DEV50.profile_id:
            raise ValueError(
                f"{profile.benchmark_key}: no Nemotron prepared artifact is registered "
                "for this natural profile"
            )
        return replace(profile, prepared_artifact_sha256=NATURAL_PREPARED_SHA256)


NEMOTRON_FAMILY = NemotronFamily(
    profile=NEMOTRON,
    lane="nemotron",
    policy_prefix=_PREFIX,
    hf_architecture="NemotronHForCausalLM",
    stop_token_ids=(2, 11, 12),
    think_open="<think>",
    think_close="</think>",
    think_open_token_ids=(49250, 2077, 1062),
    think_close_token_ids=(1885, 74045, 1062),
    literal_think_tags=True,
    thinking_system_prompt="/think",
    payload_in_user_turn=True,
)
