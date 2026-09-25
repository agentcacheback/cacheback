"""Command line for building the LongBench v2 chain inputs, raising on any drift.

``fetch-raw``, ``select``, ``build-bundle``, and ``prepare``, each printing one
sorted JSON summary. A profile still on the unminted sentinel prints its values.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rcc.benchmarks.longbench_v2 import (
    LONGBENCH_COA_EASY50_BOUNDED,
    LONGBENCH_COA_EASY50_RERANK,
    LONGBENCH_COA_EASY50_TEXT,
    UNMINTED_SENTINEL,
)
from rcc.benchmarks.longbench_v2.chunking import (
    balanced_four_chunks,
    read_chunk_ledger,
    write_chunk_ledger,
)
from rcc.benchmarks.longbench_v2.data import build_items, load_raw_rows, rows_by_qid
from rcc.benchmarks.longbench_v2.geometry import (
    ANSWER_PROMPT_ALLOWANCE,
    HOP_WRAPPER_ALLOWANCE,
    NO_CONTEXT_PROMPT_ALLOWANCE,
    QUESTION_CHOICES_MAX_TOKENS,
    rewrite_wrapper_allowance,
)
from rcc.benchmarks.longbench_v2.native_geometry import frozen_geometry, measure_geometry
from rcc.benchmarks.longbench_v2.nemotron import (
    NEMOTRON_LONGBENCH_BOUNDED,
    NEMOTRON_LONGBENCH_RERANK,
    NEMOTRON_LONGBENCH_TEXT,
    NEMOTRON_TOKENIZER_ARTIFACTS,
)
from rcc.benchmarks.longbench_v2.panel import (
    DATASET_FILE,
    DATASET_ID,
    DATASET_REVISION,
    NEMOTRON_PANEL_KEYS,
    RAW_SOURCE_SHA256,
    TOKENIZER_ARTIFACT_SHA256,
    TOKENIZER_CHECKPOINT,
    TOKENIZER_REVISION,
    Panel,
    execution_order,
    manifest_rows,
    panel_for,
    panel_records,
    read_manifest,
    select_panel,
    validate_selection,
    write_manifest,
)
from rcc.benchmarks.longbench_v2.prepare import (
    seal_prepared_panel,
    seal_source_audit,
    validate_source_commit,
)
from rcc.benchmarks.longbench_v2.prompts import prompt_token_counts
from rcc.benchmarks.longbench_v2.source_bundle import (
    build_source_bundle,
    bundle_rows,
    validate_source_bundle,
)
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.nemotron import NEMOTRON_FAMILY
from rcc.models.qwen import QWEN
from rcc.run.io import sha256_file

#: Every profile a build command may name. The profiles over one panel share its
#: bundle, so `prepare` runs each against the bundle already built and prints the
#: digests that profile carries next.
PROFILES = {
    profile.benchmark_key: profile
    for profile in (
        LONGBENCH_COA_EASY50_BOUNDED,
        LONGBENCH_COA_EASY50_RERANK,
        LONGBENCH_COA_EASY50_TEXT,
        NEMOTRON_LONGBENCH_TEXT,
        NEMOTRON_LONGBENCH_RERANK,
        NEMOTRON_LONGBENCH_BOUNDED,
    )
}


def _emit(payload: Mapping[str, object]) -> int:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def unminted_fields(profile: BenchmarkProfile) -> tuple[str, ...]:
    """Return the profile fields still carrying the unminted sentinel."""
    return tuple(
        sorted(
            field
            for field, value in profile.to_dict().items()
            if isinstance(value, str) and value == UNMINTED_SENTINEL
        )
    )


def resolve_profile(name: str) -> BenchmarkProfile:
    """Return the profile one ``--profile`` name selects."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unregistered benchmark profile {name!r}; choose from {sorted(PROFILES)}"
        ) from exc


def _hub_download() -> Any:
    """Return the hub download entry point without typing its optional stack."""
    import huggingface_hub

    return cast(Any, huggingface_hub).hf_hub_download


def load_pinned_tokenizer() -> Any:
    """Load the pinned tokenizer and verify its artifact digests."""
    import transformers

    download = _hub_download()
    for name, expected in TOKENIZER_ARTIFACT_SHA256.items():
        path = Path(str(download(TOKENIZER_CHECKPOINT, name, revision=TOKENIZER_REVISION)))
        observed = sha256_file(path.resolve())
        if observed != expected:
            raise RuntimeError(f"pinned tokenizer artifact {name} drifted: {observed}")
    tokenizer_class: Any = cast(Any, transformers).AutoTokenizer
    tokenizer: Any = tokenizer_class.from_pretrained(
        TOKENIZER_CHECKPOINT, revision=TOKENIZER_REVISION
    )
    tokenizer.model_max_length = 10**9
    return tokenizer


def load_native_tokenizers() -> dict[str, Any]:
    """Download and verify the three pinned tokenizer sets, and no model weights."""
    import transformers

    tokenizers: dict[str, Any] = {}
    for semantic_arm, label in (
        ("text_primary", "12b"),
        ("text_medium", "9b"),
        ("text_small", "4b"),
    ):
        arm = next(
            arm for arm in NEMOTRON_FAMILY.profile.physical_arms if arm.semantic_arm == semantic_arm
        )
        for key, digest in NEMOTRON_TOKENIZER_ARTIFACTS.items():
            if not key.startswith(label + "/"):
                continue
            name = key.split("/", 1)[1]
            path = Path(
                str(
                    _hub_download()(
                        arm.sender_tokenizer, name, revision=arm.sender_tokenizer_revision
                    )
                )
            )
            if sha256_file(path.resolve()) != digest:
                raise RuntimeError(f"native tokenizer artifact {key} drifted")
        tokenizer = cast(Any, transformers).AutoTokenizer.from_pretrained(
            arm.sender_tokenizer, revision=arm.sender_tokenizer_revision
        )
        tokenizer.model_max_length = 10**9
        tokenizers[semantic_arm] = tokenizer
    return tokenizers


def _fetch_raw(args: argparse.Namespace, profile: BenchmarkProfile) -> int:
    del profile
    path = Path(
        str(
            _hub_download()(
                DATASET_ID,
                DATASET_FILE,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=args.root / "source_cache",
            )
        )
    )
    digest = sha256_file(path)
    if digest != RAW_SOURCE_SHA256:
        raise RuntimeError(f"{path}: raw file digest {digest} differs from the pinned dataset")
    return _emit({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})


def _select(args: argparse.Namespace, profile: BenchmarkProfile) -> int:
    rows = load_raw_rows(args.raw)
    tokenizer = load_pinned_tokenizer()
    source_tokens = {
        str(row["_id"]): len(tokenizer.encode(str(row["context"]), add_special_tokens=False))
        for row in rows
    }
    panel = panel_for(profile)
    selection = select_panel(panel_records(rows, source_tokens, salt=panel.sample_salt), panel)
    validate_selection(selection, panel)
    ledger: list[dict[str, Any]] = []
    chunk_tokens: dict[str, list[int]] = {}
    for order, record in enumerate(selection.selected, start=1):
        _chunks, chunk_rows = balanced_four_chunks(
            str(rows[record.source_index]["context"]), tokenizer
        )
        for chunk_row in chunk_rows:
            chunk_row.update(
                {"qid": record.qid, "sample_order": order, "source_tokens": record.source_tokens}
            )
        ledger.extend(chunk_rows)
        chunk_tokens[record.qid] = [int(row["standalone_qwen_source_tokens"]) for row in chunk_rows]
    args.panel_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.panel_dir / panel.manifest_file
    ledger_path = args.panel_dir / panel.ledger_file
    write_manifest(manifest_path, manifest_rows(selection, chunk_tokens))
    write_chunk_ledger(ledger_path, ledger)
    digests = {"manifest": sha256_file(manifest_path), "ledger": sha256_file(ledger_path)}
    frozen = {"manifest": panel.manifest_sha256, "ledger": panel.ledger_sha256}
    if UNMINTED_SENTINEL not in frozen.values() and digests != frozen:
        raise RuntimeError(f"rebuilt panel drifted from the frozen digests: {digests}")
    bands = {record.qid: record.band for record in selection.selected}
    order = execution_order(bands, [record.qid for record in selection.selected])
    if order != profile.question_ids:
        raise RuntimeError("rebuilt execution order differs from the registered profile")
    return _emit({**digests, "excluded": list(selection.excluded), "items": len(order)})


def _build_bundle(args: argparse.Namespace, profile: BenchmarkProfile) -> int:
    rows = rows_by_qid(load_raw_rows(args.raw))
    manifest = build_source_bundle(args.bundle, rows, profile.question_ids)
    return _emit({key: value for key, value in manifest.items() if key != "files"})


def assert_prompt_allowances(
    items: Sequence[tuple[str, Sequence[str]]],
    tokenizer: Any,
    panel: Panel,
    builder: str,
    family: Any = None,
) -> dict[str, int]:
    """Check the wrapper allowances against every rendered prompt.

    The longest question plus choices must equal the panel's registered maximum, which may
    not exceed `QUESTION_CHOICES_MAX_TOKENS`; every rendered prompt must fit its allowance.
    """
    maxima = {
        "question_choices": 0,
        "hop_wrapper": 0,
        "rewrite_wrapper": 0,
        "answer_prompt": 0,
        "no_context_prompt": 0,
    }
    for question, choices in items:
        counts = prompt_token_counts(
            tokenizer,
            question,
            choices,
            enable_thinking=QWEN.decode.enable_thinking,
            builder=builder,
            family=family,
        )
        for name in maxima:
            maxima[name] = max(maxima[name], counts[name])
    if panel.question_choices_max_tokens > QUESTION_CHOICES_MAX_TOKENS:
        raise RuntimeError(
            f"{panel.benchmark_key}: question plus choices maximum "
            f"{panel.question_choices_max_tokens} exceeds the allowance base "
            f"{QUESTION_CHOICES_MAX_TOKENS}"
        )
    if maxima["question_choices"] != panel.question_choices_max_tokens:
        raise RuntimeError(
            f"question plus choices maximum {maxima['question_choices']} differs from the "
            f"registered {panel.question_choices_max_tokens}"
        )
    if maxima["hop_wrapper"] > HOP_WRAPPER_ALLOWANCE:
        raise RuntimeError(f"hop wrapper {maxima['hop_wrapper']} exceeds {HOP_WRAPPER_ALLOWANCE}")
    rewrite_allowance = rewrite_wrapper_allowance(builder)
    if maxima["rewrite_wrapper"] > rewrite_allowance:
        raise RuntimeError(
            f"rewrite wrapper {maxima['rewrite_wrapper']} exceeds {rewrite_allowance}"
        )
    if maxima["answer_prompt"] > ANSWER_PROMPT_ALLOWANCE:
        raise RuntimeError(
            f"answer prompt {maxima['answer_prompt']} exceeds {ANSWER_PROMPT_ALLOWANCE}"
        )
    if maxima["no_context_prompt"] > NO_CONTEXT_PROMPT_ALLOWANCE:
        raise RuntimeError(
            f"no context prompt {maxima['no_context_prompt']} exceeds {NO_CONTEXT_PROMPT_ALLOWANCE}"
        )
    return maxima


def _prepare(args: argparse.Namespace, profile: BenchmarkProfile) -> int:
    manifest = validate_source_bundle(args.bundle, profile=profile, sentinel=UNMINTED_SENTINEL)
    source_digests = {
        "source_logical_fingerprint": manifest["logical_fingerprint"],
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_archive_sha256": manifest["archive_sha256"],
    }
    source_commit = validate_source_commit(args.source_commit)
    rows = bundle_rows(args.bundle)
    panel = panel_for(profile)
    manifest_rows_ = read_manifest(panel.manifest_path)
    strata = {row["qid"]: int(row["selection_length_band_50k"]) for row in manifest_rows_}
    ledger = read_chunk_ledger(panel.ledger_path)
    if profile.benchmark_key in NEMOTRON_PANEL_KEYS:
        tokenizers = load_native_tokenizers()
        items = build_items(
            rows,
            ledger,
            strata,
            profile.question_ids,
            tokenizers["text_primary"],
            expected_count_field=None,
        )
        measured = measure_geometry(items, tokenizers, profile)
        if measured != frozen_geometry(profile):
            raise RuntimeError("rebuilt native context geometry differs from its digest")
        maxima = {
            name: measured[name] for name in ("hop_wrapper_allowance", "rewrite_wrapper_allowance")
        }
    else:
        tokenizer = load_pinned_tokenizer()
        items = build_items(rows, ledger, strata, profile.question_ids, tokenizer)
        maxima = assert_prompt_allowances(
            [(item.question, item.choices) for item in items],
            tokenizer,
            panel,
            profile.prompt_builder,
        )
    audit = seal_source_audit(
        args.run_root, source_commit=source_commit, items=items, profile=profile
    )
    prepared = seal_prepared_panel(
        args.run_root,
        source_commit=source_commit,
        items=items,
        source_audit=audit,
        profile=profile,
    )
    return _emit(
        {
            **source_digests,
            "source_commit": source_commit,
            "source_audit_fingerprint": audit["audit_fingerprint"],
            "prepared_artifact_sha256": prepared["artifact_sha256"],
            "prepared_config_fingerprint": prepared["config_fingerprint"],
            "panel_registration_sha256": prepared["panel_registration_sha"],
            "prompt_token_maxima": maxima,
            "unminted_before": list(unminted_fields(profile)),
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rcc-longbench-build", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch-raw", "select", "build-bundle", "prepare"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--profile", required=True, choices=sorted(PROFILES))
        if name == "fetch-raw":
            sub.add_argument("--root", type=Path, required=True)
        elif name == "select":
            sub.add_argument("--raw", type=Path, required=True)
            sub.add_argument("--panel-dir", type=Path, required=True)
        elif name == "build-bundle":
            sub.add_argument("--raw", type=Path, required=True)
            sub.add_argument("--bundle", type=Path, required=True)
        else:
            sub.add_argument("--bundle", type=Path, required=True)
            sub.add_argument("--run-root", type=Path, required=True)
            sub.add_argument("--source-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one build command against the profile named on the command line."""
    args = _parser().parse_args(argv)
    profile = resolve_profile(args.profile)
    handler = {
        "fetch-raw": _fetch_raw,
        "select": _select,
        "build-bundle": _build_bundle,
        "prepare": _prepare,
    }[args.command]
    return handler(args, profile)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"FATAL: {exc}\n")
        raise SystemExit(2) from exc


__all__ = (
    "PROFILES",
    "assert_prompt_allowances",
    "load_pinned_tokenizer",
    "main",
    "resolve_profile",
    "unminted_fields",
)
