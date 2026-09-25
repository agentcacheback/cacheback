"""Unpack the source bundles the benchmark configs read.

The archive under `--dest` is checked against its pinned digest and unpacked;
when it is absent it is downloaded from ``RCC_DATA_URL`` or ``--base-url``
first.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path

#: Bundle name to its pinned archive digest and the archive's file name.
BUNDLES: dict[str, tuple[str, str]] = {}


def _register() -> None:
    from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
    from rcc.run.selector_dev.contract import BUNDLE_SHA256

    profile = FANOUTQA_NATURAL_DEV50
    BUNDLES["fanoutqa-natural-dev50"] = (
        profile.source_archive_sha256,
        f"fanoutqa-m3-source-{profile.source_logical_fingerprint[:16]}.tar.zst",
    )
    BUNDLES["selector-dev-v8"] = (BUNDLE_SHA256, "selector-dev-v8.tar.zst")


def sha256_file(path: Path) -> str:
    """Return the hex digest of one file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, dest: Path, *, base_url: str | None = None) -> Path:
    """Download, verify, and unpack one bundle, returning its directory."""
    _register()
    try:
        expected, archive_name = BUNDLES[name]
    except KeyError as exc:
        raise ValueError(f"unknown bundle {name!r}; choose from {sorted(BUNDLES)}") from exc
    base = base_url or os.environ.get("RCC_DATA_URL")
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / archive_name
    target = dest / name
    if target.is_dir() and any(target.iterdir()):
        return target
    if not archive.is_file() or sha256_file(archive) != expected:
        if base is None:
            raise RuntimeError(
                f"{archive} is missing or differs from the registered digest; place the "
                f"released {archive_name} there, or name where to download it with "
                "RCC_DATA_URL or --base-url"
            )
        urllib.request.urlretrieve(f"{base.rstrip('/')}/{archive_name}", archive)
    observed = sha256_file(archive)
    if observed != expected:
        raise RuntimeError(f"{archive_name}: digest {observed} differs from registered {expected}")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(target)], check=True)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch one named bundle into a local data directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_parser = sub.add_parser("fetch")
    fetch_parser.add_argument("name")
    fetch_parser.add_argument("--dest", type=Path, default=Path("data"))
    fetch_parser.add_argument("--base-url")
    args = parser.parse_args(argv)
    path = fetch(args.name, args.dest, base_url=args.base_url)
    sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
