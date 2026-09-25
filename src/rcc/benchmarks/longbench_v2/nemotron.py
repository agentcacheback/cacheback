"""The Nemotron profiles over the LongBench easy fifty, on the native tokenizer.

They read the same panel object, chunks, prompts, and seed namespace as the Qwen
profiles, under the Nemotron sampling, thinking, and alignment settings.
"""

from __future__ import annotations

from dataclasses import replace

from rcc.benchmarks.longbench_v2 import (
    LONGBENCH_COA_EASY50_BOUNDED,
    LONGBENCH_COA_EASY50_RERANK,
    LONGBENCH_COA_EASY50_TEXT,
)
from rcc.benchmarks.longbench_v2.panel import (
    NEMOTRON_EASY50_BOUNDED_KEY,
    NEMOTRON_EASY50_RERANK_KEY,
    NEMOTRON_EASY50_TEXT_KEY,
)
from rcc.models.nemotron import NANO_12B

NEMOTRON_LONGBENCH_TOKENIZER = NANO_12B
NEMOTRON_LONGBENCH_CHUNK_CEILING = 70_000
NEMOTRON_TOKENIZER_ARTIFACTS = {
    "12b/tokenizer.json": "3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f",
    "12b/tokenizer_config.json": "a3e4d48a0f8b4ce6fd746464199299cbdefe6ee202e34048940da2d9838a9aa6",
    "12b/special_tokens_map.json": (
        "2a4d2e7403546286e5d75f5b6b3c197490be67fb1e2118e5c60ad5c26e6668b1"
    ),
    "9b/tokenizer.json": "3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f",
    "9b/tokenizer_config.json": "a3e4d48a0f8b4ce6fd746464199299cbdefe6ee202e34048940da2d9838a9aa6",
    "9b/special_tokens_map.json": (
        "2a4d2e7403546286e5d75f5b6b3c197490be67fb1e2118e5c60ad5c26e6668b1"
    ),
    "4b/tokenizer.json": "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
    "4b/tokenizer_config.json": "48de4056b0b17de26e03232fdc1f55b70595c9354ceb2ed061f724f45620aa41",
    "4b/special_tokens_map.json": (
        "e3a4f63da745f02317a45e00e6476c17fc66ac41faf14bb1b0be1f3211b0ca53"
    ),
    "4b/chat_template.jinja": "ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862",
}
# The text senders take a 16,000-token note ceiling and presence penalty 0; the
# receiver's answer ceiling stays 24,000. Each profile carries its own digests,
# so a panel prepared for one is not accepted for another.
_NATIVE_TEXT = {
    "worker_prompt_tokens": NEMOTRON_LONGBENCH_CHUNK_CEILING,
    "report_ceiling": 16_000,
    "report_presence_penalty": 0.0,
    "source_commit": "63cf79ef012f56e61bffd61654fde20f1b7baf82",
    "output_prefix": "longbench-v2-coa-nemotron",
}
NEMOTRON_LONGBENCH_TEXT = replace(
    LONGBENCH_COA_EASY50_TEXT,
    profile_id="longbench-v2-coa-nemotron-easy50-text-t4-n50-sealed-v1",
    panel_registration_sha256="ef8c72f21477dd86",
    prepared_artifact_sha256="07fd442709299a87fabadb68fd353691b09fbbdb47652bd46d6cd2225cbab961",
    prepared_config_fingerprint="024d3c5774ab1ec7",
    source_audit_fingerprint="f4576e94a41c953d",
    benchmark_key=NEMOTRON_EASY50_TEXT_KEY,
    run_id_stem="longbench-coa-nemotron-easy50-text-t4-v1",
    **_NATIVE_TEXT,
)
NEMOTRON_LONGBENCH_RERANK = replace(
    LONGBENCH_COA_EASY50_RERANK,
    profile_id="longbench-v2-coa-nemotron-easy50-rerank-t4-n50-sealed-v1",
    panel_registration_sha256="8cff66040fc6dd5a",
    prepared_artifact_sha256="fd2c43486420473fc2416e6dfe796cc14272601df162cfbdf597196b65c6b892",
    prepared_config_fingerprint="8dc614f3be497ade",
    source_audit_fingerprint="cde305a2c4afa475",
    benchmark_key=NEMOTRON_EASY50_RERANK_KEY,
    run_id_stem="longbench-coa-nemotron-easy50-rerank-t4-v1",
    **_NATIVE_TEXT,
)
# The bounded ladder: the same rungs as the Qwen bounded profile minus 65,536,
# which does not fit every item's worst hop inside the capture window on this
# tokenizer, and an over-window hop raises.
_NATIVE_BOUNDED_ARMS = tuple(
    arm for arm in LONGBENCH_COA_EASY50_BOUNDED.arms if arm.budget_rows != 65_536
)
_NATIVE_BOUNDED_POLICIES = tuple(
    pair
    for pair in LONGBENCH_COA_EASY50_BOUNDED.sealed_qwen_policies
    if pair[0] in {arm.arm_id for arm in _NATIVE_BOUNDED_ARMS}
)
NEMOTRON_LONGBENCH_BOUNDED = replace(
    LONGBENCH_COA_EASY50_BOUNDED,
    profile_id="longbench-v2-coa-nemotron-easy50-bounded-t4-n50-sealed-v1",
    panel_registration_sha256="f1f3e5925515fb21",
    prepared_artifact_sha256="845e1da4680cd9295459f764914e2b6b9b523310560e559c2df88e2ba8babf5b",
    prepared_config_fingerprint="024d3c5774ab1ec7",
    source_audit_fingerprint="f4576e94a41c953d",
    arms=_NATIVE_BOUNDED_ARMS,
    sealed_qwen_policies=_NATIVE_BOUNDED_POLICIES,
    arm_profiles=("longbench-coa-nemotron-bounded-5arm-v1",),
    benchmark_key=NEMOTRON_EASY50_BOUNDED_KEY,
    run_id_stem="longbench-coa-nemotron-easy50-bounded-t4-v1",
    **_NATIVE_TEXT,
)


__all__ = (
    "NEMOTRON_LONGBENCH_BOUNDED",
    "NEMOTRON_LONGBENCH_CHUNK_CEILING",
    "NEMOTRON_LONGBENCH_RERANK",
    "NEMOTRON_LONGBENCH_TEXT",
    "NEMOTRON_LONGBENCH_TOKENIZER",
    "NEMOTRON_TOKENIZER_ARTIFACTS",
)
