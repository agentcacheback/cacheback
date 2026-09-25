"""Declared Inductor policy and live attestation for native Nemotron engines."""

from __future__ import annotations

import importlib
import json
import os
from typing import Any

POLICY_ID = "nemotron-inductor-deterministic-v1"
ENVIRONMENT = {
    "TORCHINDUCTOR_DETERMINISTIC": "1",
    "VLLM_DISABLE_COMPILE_CACHE": "0",
    "VLLM_USE_AOT_COMPILE": "1",
    "VLLM_FORCE_AOT_LOAD": "0",
}
PRECISION_POLICY = {
    "float32_matmul_precision": "highest",
    "matmul_allow_tf32": False,
    "bf16_reduced_precision_reduction": True,
    "fp16_reduced_precision_reduction": True,
    "cudnn_allow_tf32": True,
    "deterministic_algorithms": False,
    "inductor_deterministic": True,
}


def precision() -> dict[str, Any]:
    """Observe precision independently of the eager deterministic-algorithms flag."""
    import torch
    import torch._inductor.config as inductor_config

    matmul = torch.backends.cuda.matmul
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": matmul.allow_tf32,
        "bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
        "fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "inductor_deterministic": bool(inductor_config.deterministic),
    }


def precision_delta(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Name every changed precision field without modifying the process state."""
    return {
        key: {"expected": expected.get(key), "actual": observed.get(key)}
        for key in sorted(expected.keys() | observed.keys())
        if expected.get(key) != observed.get(key)
    }


def _environment() -> None:
    if any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise RuntimeError(
            "Nemotron compiler environment differs; export "
            + " ".join(f"{key}={value}" for key, value in ENVIRONMENT.items())
            + " before starting Python"
        )


def _resolved_flags() -> dict[str, str]:
    envs = importlib.import_module("vllm.envs")
    return {
        key: str(int(bool(getattr(envs, key)))) for key in ENVIRONMENT if key.startswith("VLLM_")
    }


def _restore_inductor() -> None:
    import torch._inductor.config as inductor_config

    inductor_config.deterministic = True


def _resident_model(llm: Any) -> Any:
    from rcc.models.nemotron.bridge import resident_model

    return resident_model(llm)


def before_engine() -> dict[str, Any]:
    """Refuse a late or conflicting process policy before importing vLLM."""
    _environment()
    state = precision()
    if state != PRECISION_POLICY:
        raise RuntimeError(
            "Nemotron initial precision differs from the registered policy: "
            + json.dumps(precision_delta(PRECISION_POLICY, state), sort_keys=True)
        )
    return state


def _model_state(llm: Any, *, capture: bool) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    eager = bool(config.model_config.enforce_eager)
    if eager != capture:
        raise RuntimeError("Nemotron engine eager mode differs from the declared role")
    compiler = dict(config.compilation_config.inductor_compile_config)
    states: dict[str, dict[str, Any]] = {}
    for name, module in _resident_model(llm).named_modules():
        if not hasattr(module, "aot_compiled_fn"):
            continue
        settings = dict(module.compilation_config.inductor_compile_config)
        states[name] = {
            "loaded": bool(getattr(module, "was_aot_compile_fn_loaded_from_disk", False)),
            "compiled": bool(getattr(module, "compiled", False)),
            "has_aot_callable": module.aot_compiled_fn is not None,
            "inductor_compile_config": settings,
        }
    if compiler.get("deterministic") is not True or any(
        state["inductor_compile_config"].get("deterministic") is not True
        for state in states.values()
    ):
        raise RuntimeError("Nemotron model compiler is missing the deterministic policy")
    if not capture and (
        not states or not any(state["has_aot_callable"] for state in states.values())
    ):
        raise RuntimeError("Nemotron compiled engine has no observed AOT callable")
    return {"enforce_eager": eager, "inductor_compile_config": compiler, "model_modules": states}


def after_engine(llm: Any, before: dict[str, Any], *, capture: bool) -> dict[str, Any]:
    """Restore the independent Inductor policy after Torch's initialization cleanup."""
    state = precision()
    if any(state[key] != value for key, value in before.items() if key != "inductor_deterministic"):
        raise RuntimeError(
            "Nemotron precision changed during engine initialization: "
            + json.dumps(precision_delta(before, state), sort_keys=True)
        )
    # Torch's Dynamo cleanup couples Inductor to the restored eager flag.
    _restore_inductor()
    record = {
        "policy": POLICY_ID,
        "mode": "eager_capture" if capture else "aot_compile_or_load",
        "environment": dict(ENVIRONMENT),
        "before_init": before,
        "after_init_before_restore": state,
        "after_restore": precision(),
    }
    attest(llm, record, capture=capture)
    return record


def attest(llm: Any, record: dict[str, Any], *, capture: bool) -> dict[str, Any]:
    """Reject drift and return the actual policy values without asserting output equality."""
    _environment()
    state = precision()
    if (
        state != PRECISION_POLICY
        or state != record["after_restore"]
        or state != record["before_init"]
    ):
        raise RuntimeError(
            "Nemotron compiler precision drift after initialization: "
            + json.dumps(
                {
                    "observed": state,
                    "policy_delta": precision_delta(PRECISION_POLICY, state),
                    "before_init_delta": precision_delta(record["before_init"], state),
                    "after_restore_delta": precision_delta(record["after_restore"], state),
                },
                sort_keys=True,
            )
        )
    resolved = _resolved_flags()
    if any(value != ENVIRONMENT[key] for key, value in resolved.items()):
        raise RuntimeError("Nemotron resolved vLLM compiler environment differs")
    return {
        "initialization": record,
        "precision": state,
        "resolved_environment": resolved,
        **_model_state(llm, capture=capture),
    }
