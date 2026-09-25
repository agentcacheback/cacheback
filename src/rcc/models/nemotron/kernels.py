"""Pinned prebuilt Mamba kernels for the native Nemotron producer."""

import importlib
from functools import partial
from typing import Any

import torch

# Revisions of the Hub's kernel-type repositories, which the loader reads;
# the model-type mirrors of the same names carry a different history.
KERNEL_PINS = (
    ("kernels-community/causal-conv1d", "3b054b4ad592fd456610273463a457362634a33c"),
    ("kernels-community/mamba-ssm", "4e97b86d553be8b550e38d477ed52d4a686aa119"),
)
KERNEL_VARIANT = "torch211-cxx11-cu130-x86_64-linux"
DOWNLOAD_PINS = (("huggingface_hub", "1.10.1"), ("tqdm", "4.67.1"))


def _chunk_scan(scan: Any, x: torch.Tensor, dt: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
    """Pack timesteps before the pinned scan's signed 32-bit pointer arithmetic."""
    # HF slices dt from a 22,656-wide projection. Long prefills overflow the
    # kernel's sequence-stride product; packing preserves all 128 head values.
    return scan(x, dt.contiguous(), *args, **kwargs)


def enable_native_kernels(model: Any = None) -> None:
    """Load pinned kernels; optionally bind a TP1 model to caller-stream execution."""
    if model is not None:
        devices = {parameter.device for parameter in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise RuntimeError("Nemotron native kernels require one CUDA device")
        if any(hasattr(module, "_hf_hook") for module in model.modules()):
            raise RuntimeError("Nemotron native kernels do not support Accelerate dispatch hooks")
    loader: Any = importlib.import_module("kernels")
    conv = loader.get_kernel(KERNEL_PINS[0][0], revision=KERNEL_PINS[0][1])
    mamba = loader.get_kernel(KERNEL_PINS[1][0], revision=KERNEL_PINS[1][1])
    native: Any = importlib.import_module("transformers.models.nemotron_h.modeling_nemotron_h")
    native.causal_conv1d_fn = conv.causal_conv1d_fn
    native.causal_conv1d_update = conv.causal_conv1d_update
    native.selective_state_update = mamba.selective_state_update
    native.mamba_chunk_scan_combined = partial(_chunk_scan, mamba.mamba_chunk_scan_combined)
    native.mamba_split_conv1d_scan_combined = mamba.mamba_split_conv1d_scan_combined
    functions = (
        native.causal_conv1d_fn,
        native.causal_conv1d_update,
        native.selective_state_update,
        native.mamba_chunk_scan_combined,
        native.mamba_split_conv1d_scan_combined,
    )
    if not all(callable(fn) for fn in functions):
        raise RuntimeError("Pinned Nemotron fast kernel exports are incomplete")
    native.is_fast_path_available = True
    if model is None:
        return  # Bootstrap validates exports before allocating a model.
    for module in model.modules():
        if isinstance(module, native.NemotronHMamba2Mixer):
            # Pinned conv and Triton kernels honor the caller stream. The HF
            # wrapper switches only Mamba to stream zero without ordering its
            # input/output against the surrounding norm and residual operations.
            module.forward = module.cuda_kernels_forward
