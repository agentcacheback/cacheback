"""Family-native decode registrations for production model adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeProtocol:
    """One model family's sampling and thinking identity."""

    family: str
    temperature: float
    top_p: float
    top_k: int | None
    enable_thinking: bool
    presence_penalty: float = 0.0

    def __post_init__(self) -> None:
        """Refuse an invalid sampling contract."""
        if not self.family:
            raise ValueError("decode family must be nonempty")
        if self.temperature <= 0:
            raise ValueError(f"{self.family}: temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"{self.family}: top_p must lie in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"{self.family}: top_k must be positive or off")

    @property
    def profile_id(self) -> str:
        """Return the stable family-native profile name."""
        return f"{self.family}-family-native-v1"

    @property
    def identity_hash(self) -> str:
        """Return one fingerprint over the complete decode contract."""
        payload = json.dumps(
            {"profile_id": self.profile_id, **self.to_dict()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def backend_sampling(self) -> dict[str, float | int]:
        """Return backend sampling fields, omitting top-k when it is off."""
        sampling: dict[str, float | int] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            sampling["top_k"] = self.top_k
        return sampling

    def to_dict(self) -> dict[str, object]:
        """Return the signed JSON-compatible decode identity."""
        return {
            "family": self.family,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "enable_thinking": self.enable_thinking,
        }


DECODE_PROTOCOLS: dict[str, DecodeProtocol] = {
    "qwen3": DecodeProtocol(
        family="qwen3",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        enable_thinking=True,
    ),
    "gemma4": DecodeProtocol(
        family="gemma4",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        enable_thinking=True,
    ),
    "ministral3": DecodeProtocol(
        family="ministral3",
        temperature=0.7,
        top_p=0.95,
        top_k=None,
        enable_thinking=True,
    ),
}


def decode_protocol(family: str) -> DecodeProtocol:
    """Return one family registration and refuse an unknown family."""
    try:
        return DECODE_PROTOCOLS[family]
    except KeyError:
        raise ValueError(
            f"no registered decode protocol for model family {family!r}; "
            f"registered families are {sorted(DECODE_PROTOCOLS)}"
        ) from None


__all__ = ("DECODE_PROTOCOLS", "DecodeProtocol", "decode_protocol")
