"""Read-only runtime checks with actionable setup errors."""

from __future__ import annotations

import os
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from rcc.hf import Agent


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _minimum(installed: str | None, minimum: tuple[int, int, int]) -> bool:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", installed or "")
    return match is not None and tuple(map(int, match.groups())) >= minimum


def _vllm_issues(versions: dict[str, str | None], cuda: bool, allow_unstable: bool) -> list[str]:
    installed = versions["vllm"]
    issues: list[str] = []
    if installed != "0.11.1" and not (installed == "0.26.0" and allow_unstable):
        issues.append("Install vllm==0.11.1; 0.26.0 requires allow_unstable=True.")
    if platform.system() != "Linux":
        issues.append("Run the vLLM adapter in Linux with CUDA, or use backend='hf' on CPU.")
    if not cuda:
        issues.append("CUDA is unavailable; check the GPU driver and CUDA-enabled PyTorch install.")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        issues.append("Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before importing or creating vLLM.")
    pins = {"0.11.1": ("2.9.0", "4.57.1"), "0.26.0": ("2.11.0", "5.13.0")}
    expected = pins.get(installed or "")
    if expected is not None:
        for package, pin in zip(("torch", "transformers"), expected, strict=True):
            if (versions[package] or "").split("+")[0] != pin:
                issues.append(f"The validated vLLM {installed} stack uses {package}=={pin}.")
    return issues


def _agent_issues(agent: Agent) -> list[str]:
    from rcc import vllm
    from rcc.transport import validate_model

    try:
        validate_model(agent.model)
        if agent.engine is not None:
            vllm.validate_binding_config(agent.engine)
            vllm.require_idle(agent.engine)
    except (ValueError, RuntimeError) as error:
        return [str(error)]
    return []


def check(
    agent: Agent | None = None, *, backend: str | None = None, allow_unstable: bool | None = None
) -> dict[str, Any]:
    """Report environment or bound-agent issues without downloading weights or running a model."""
    if agent is not None:
        if backend not in (None, agent.backend) or allow_unstable not in (
            None,
            agent.allow_unstable,
        ):
            raise ValueError("backend and allow_unstable come from the bound agent; omit them")
        backend, allow_unstable = agent.backend, agent.allow_unstable
    backend = "hf" if backend is None else backend
    if backend not in ("hf", "vllm"):
        raise ValueError("backend must be 'hf' or 'vllm'")
    versions = {name: _version(name) for name in ("rclc", "torch", "transformers", "vllm")}
    issues: list[str] = []
    if not _minimum(versions["torch"], (2, 4, 0)):
        issues.append("Install torch>=2.4.")
    if not _minimum(versions["transformers"], (4, 57, 1)) or _minimum(
        versions["transformers"], (6, 0, 0)
    ):
        issues.append("Install transformers>=4.57.1,<6; tested pins are 4.57.1 and 5.13.0.")
    cuda = torch.cuda.is_available()
    if backend == "vllm":
        issues.extend(_vllm_issues(versions, cuda, bool(allow_unstable)))
    if agent is not None:
        issues.extend(_agent_issues(agent))
    return {
        "ok": not issues,
        "scope": "environment" if agent is None else "bound_agent",
        "backend": backend,
        "versions": versions,
        "cuda_available": cuda,
        "issues": issues,
    }
