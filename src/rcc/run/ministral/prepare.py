"""Build, or revalidate, the Ministral sender prompt panel for one item range.

The panel holds the exact prompt token rows each sender arm will be given.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50, profile_by_id
from rcc.benchmarks.fanoutqa.ministral_data import (
    load_ministral_prompt_panel,
    write_ministral_prompt_panel,
)
from rcc.benchmarks.fanoutqa.ministral_prepare import build_prepared_panel
from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.ministral.text_codec import (
    MINISTRAL_TEXT_CODEC_FILES,
    load_registered_text_codec,
)
from rcc.models.ministral.text_prompt import registered_text_senders


def prepared_panel_path(results_root: Path, item_offset: int, item_count: int) -> Path:
    """Return where one item range's prepared panel sits under the results root."""
    return (
        results_root / "prepared" / f"ministral3-fanoutqa-i{item_offset:03d}n{item_count:03d}.json"
    )


def tokenizer_snapshots(*, local_files_only: bool = False) -> dict[str, Path]:
    """Download or locate the sender tokenizer snapshots."""
    hub: Any = importlib.import_module("huggingface_hub")
    snapshots: dict[str, Path] = {}
    for sender in registered_text_senders():
        path = hub.snapshot_download(
            repo_id=sender.checkpoint,
            revision=sender.revision,
            allow_patterns=list(MINISTRAL_TEXT_CODEC_FILES),
            local_files_only=local_files_only,
        )
        snapshots[sender.semantic_arm] = Path(str(path))
    return snapshots


def prepare_panel(
    results_root: Path,
    source_bundle: Path,
    *,
    profile: BenchmarkProfile,
    item_offset: int,
    item_count: int,
) -> dict[str, object]:
    """Build, or reuse, the sender prompt panel of one item range."""
    target = prepared_panel_path(results_root, item_offset, item_count)
    qids = profile.question_ids[item_offset : item_offset + item_count]
    if target.is_file():
        panel = load_ministral_prompt_panel(target)
        if panel.benchmark_profile != profile or panel.qids != qids:
            raise RuntimeError("existing Ministral prepared panel differs from this run")
    else:
        snapshots = tokenizer_snapshots()
        codecs = {
            arm: load_registered_text_codec(snapshot, arm) for arm, snapshot in snapshots.items()
        }
        panel = build_prepared_panel(
            source_bundle, selected_qids=qids, codecs=codecs, profile=profile
        )
        write_ministral_prompt_panel(target, panel)
    return {
        "path": str(target),
        "prepared_sha256": panel.fingerprint,
        "qids": list(panel.qids),
        "prompt_rows": panel.prompt_rows,
    }


def main(argv: list[str] | None = None) -> int:
    """Prepare the Ministral sender panel from a source bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--benchmark-profile", default=FANOUTQA_NATURAL_DEV50.profile_id)
    parser.add_argument("--item-offset", type=int, default=0)
    parser.add_argument("--item-count", type=int, required=True)
    args = parser.parse_args(argv)
    payload = prepare_panel(
        args.results_root,
        args.source_bundle,
        profile=profile_by_id(args.benchmark_profile),
        item_offset=args.item_offset,
        item_count=args.item_count,
    )
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
