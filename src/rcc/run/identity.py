"""Byte-stable identity primitives shared by the pipeline adapters.

An encoding profile fixes one serialization, and a fingerprint is the SHA-256 of
the bytes it renders. The rest of the tree reaches them through `rcc.run.io`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal, cast


@dataclass(frozen=True)
class EncodingProfile:
    """One serialization and digest contract."""

    name: str
    serializer: Literal["json", "sorted_kv"]
    sort_keys: bool = True
    separators: tuple[str, str] | None = None
    allow_nan: bool = True
    default_digest_chars: int | None = None


class _JsonCompactLegacyProfile(EncodingProfile):
    """The compact-JSON digest the fleet identities use."""


class _JsonCompactStrictProfile(EncodingProfile):
    """The strict compact-JSON digest, which refuses NaN."""


class _JsonDefaultLegacyProfile(EncodingProfile):
    """The default-JSON fingerprint, truncated to 16 characters."""


class _SortedKvLegacyProfile(EncodingProfile):
    """The sorted k=v text fingerprint the registration identities use."""


class _RuntimeLegacyProfile(EncodingProfile):
    """The runtime fingerprint, truncated to 12 characters."""


json_compact_legacy = _JsonCompactLegacyProfile(
    name="json_compact_legacy",
    serializer="json",
    separators=(",", ":"),
)
json_compact_strict = _JsonCompactStrictProfile(
    name="json_compact_strict",
    serializer="json",
    separators=(",", ":"),
    allow_nan=False,
)
json_default_legacy = _JsonDefaultLegacyProfile(
    name="json_default_legacy",
    serializer="json",
    default_digest_chars=16,
)
sorted_kv_legacy = _SortedKvLegacyProfile(
    name="sorted_kv_legacy",
    serializer="sorted_kv",
)
runtime_legacy = _RuntimeLegacyProfile(
    name="runtime_legacy",
    serializer="json",
    separators=(",", ":"),
    default_digest_chars=12,
)

PROFILES: dict[str, EncodingProfile] = {
    profile.name: profile
    for profile in (
        json_compact_legacy,
        json_compact_strict,
        json_default_legacy,
        sorted_kv_legacy,
        runtime_legacy,
    )
}
ProfileRef = EncodingProfile | str
_MISSING = object()


def _resolve_profile(profile: ProfileRef) -> EncodingProfile:
    if isinstance(profile, EncodingProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown identity encoding profile {profile!r}") from exc


def canonical_bytes(value: object, profile: ProfileRef) -> bytes:
    """Return the bytes one identity profile renders a value as."""
    selected = _resolve_profile(profile)
    if selected.serializer == "sorted_kv":
        if not isinstance(value, Mapping):
            raise TypeError("sorted_kv_legacy requires a mapping")
        pairs = cast(Mapping[str, Any], value)
        text = "|".join(f"{key}={item}" for key, item in sorted(pairs.items()))
        return text.encode()

    kwargs: dict[str, Any] = {"sort_keys": selected.sort_keys}
    if selected.separators is not None:
        kwargs["separators"] = selected.separators
    if not selected.allow_nan:
        kwargs["allow_nan"] = False
    return json.dumps(value, **kwargs).encode()


def _digest_chars(profile: EncodingProfile, digest_chars: int | None) -> int | None:
    selected = profile.default_digest_chars if digest_chars is None else digest_chars
    if selected is not None and not 0 < selected <= 64:
        raise ValueError("digest_chars must be between 1 and 64")
    return selected


def fingerprint(
    value: object,
    profile: ProfileRef,
    digest_chars: int | None = None,
) -> str:
    """Hash canonical bytes with SHA-256, truncating the hex digest if asked."""
    selected = _resolve_profile(profile)
    digest = hashlib.sha256(canonical_bytes(value, selected)).hexdigest()
    chars = _digest_chars(selected, digest_chars)
    return digest if chars is None else digest[:chars]


def registration_fingerprint(
    body: object,
    profile: ProfileRef,
    digest_chars: int | None = None,
) -> str:
    """Hash a registration body exactly as supplied, without adding fields."""
    return fingerprint(body, profile, digest_chars=digest_chars)


def seal(
    body: Mapping[str, Any],
    field: str,
    profile: ProfileRef,
    digest_chars: int | None = None,
) -> dict[str, Any]:
    """Return a copy of the body with its own fingerprint stored in ``field``."""
    if field in body:
        raise ValueError(f"unsealed body already contains seal field {field!r}")
    sealed = dict(body)
    sealed[field] = registration_fingerprint(body, profile, digest_chars)
    return sealed


def _echo_value(values: Mapping[str, Any], field: str) -> str:
    if field not in values:
        return "<missing>"
    return repr(values[field])


def _logical_prefix(parts: Iterable[object], width: int) -> str:
    if not 1 <= width <= 64:
        raise ValueError("logical digest width must be between 1 and 64")
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:width]


def logical_seed(parts: Iterable[object], width: int) -> int:
    """Return an integer digest prefix over pipe-delimited seed parts."""
    return int(_logical_prefix(parts, width), 16)


def logical_id(parts: Iterable[object], width: int) -> str:
    """Return a hexadecimal digest prefix over pipe-delimited id parts."""
    return _logical_prefix(parts, width)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def runtime_signature(model_revisions: dict[str, str]) -> dict[str, Any]:
    """Return the hardware, software, and model identity of one timing run."""
    import torch

    gpu: dict[str, Any] | None = None
    driver = "unavailable"
    if torch.cuda.is_available():
        # torch's CUDA stubs type device properties as Unknown, which strict
        # type checking rejects; the alias reads the same attributes off the
        # same object.
        cuda: Any = torch.cuda
        properties = cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_vram_bytes": int(properties.total_memory),
        }
        try:
            driver = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            driver = "unavailable"
    return {
        "gpu": gpu,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvidia_driver": driver,
        "vllm": _package_version("vllm"),
        "transformers": _package_version("transformers"),
        "model_revisions": dict(sorted(model_revisions.items())),
    }
