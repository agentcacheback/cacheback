"""Signed Nemotron selector and production scheduling identities."""

SELECTOR_RECIPE = "nemotron-support-mamba-write20-anchor90-read10-pool7-w16-v1"
COMPONENT_SCHEMA = "support-query-write-evidence-f64-v1"
LAYERS = (6, 33, 60)
PRODUCTION_ARMS = (
    "latent_query_support_r8",
    "text_primary",
    "issue_only",
    "text_medium",
    "text_small",
    "latent_query_support_r2",
    "latent_query_support_r4",
    "latent_query_support_r16",
    "latent_query_support_r32",
    "latent_query_support_r64",
    "latent_query_support_r128",
    # The chain's re-ranked handoff seats follow the fan-out roster; a chain
    # profile's own order is cut from this list to the arms it registers.
    "latent_query_support_rerank_r4",
    "latent_query_support_rerank_r8",
    "latent_query_support_rerank_r16",
    "latent_query_support_rerank_r32",
    "latent_query_support_rerank_r64",
    "latent_query_support_rerank_r128",
    # The chain's bounded seats, the same six rungs the Qwen lane implements.
    "latent_query_support_bounded_b65536",
    "latent_query_support_bounded_b32768",
    "latent_query_support_bounded_b16384",
    "latent_query_support_bounded_b8192",
    "latent_query_support_bounded_b4096",
    "latent_query_support_bounded_b2048",
)
