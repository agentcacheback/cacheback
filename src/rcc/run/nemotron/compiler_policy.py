"""The compiler-policy phase rows the Nemotron lane banks.

Both calls are inert on the other lanes, so every seat can make them.
"""

from collections.abc import Callable
from typing import Any


def policy_fields(backend: Any, lane: str) -> dict[str, Any]:
    """Return the live compiler policy, or no fields at all off the Nemotron lane."""
    return {"compiler_policy": backend.runtime_policy_record()} if lane == "nemotron" else {}


def close_backend(
    backend: Any,
    bank: Callable[[dict[str, Any]], Any],
    *,
    lane: str,
    policy: str,
    role: str,
) -> None:
    """Bank the policy after the workload, then release the engine either way."""
    try:
        if lane == "nemotron":
            bank(
                {
                    "kind": "phase",
                    "phase": f"nemotron_{role}_compiler_complete",
                    "policy": policy,
                    **policy_fields(backend, lane),
                }
            )
    finally:
        backend.close()
