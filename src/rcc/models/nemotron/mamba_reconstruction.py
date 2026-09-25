"""Signed SSD reconstruction with the pinned kernels' intermediate precision."""

from __future__ import annotations

from typing import Any

import torch


def scan(
    x: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    d: torch.Tensor,
    state: torch.Tensor,
    storage: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct one SSD chunk, including coefficient/state/weighted-B casts."""
    cumulative = (dt * a).cumsum(0)
    cb = torch.einsum("thn,shn->tsh", c, b)
    decay = (cumulative[:, None] - cumulative[None, :]).clamp(max=0).exp()
    coefficient = cb * decay * dt[None]
    coefficient *= torch.ones(x.shape[0], x.shape[0]).tril()[:, :, None]
    local = torch.einsum("tsh,shp->thp", coefficient.to(storage).float(), x)
    incoming = torch.einsum("thn,hpn->thp", c, state.to(storage).float())
    incoming *= cumulative.exp()[:, :, None]
    output = (local + incoming + x * d[None, :, None]).to(storage).float()
    scale = (cumulative[-1:] - cumulative).clamp(max=0).exp() * dt
    weighted_b = (b * scale[:, :, None]).to(storage).float()
    final = torch.einsum("thp,thn->hpn", x, weighted_b)
    final += state * cumulative[-1].exp()[:, None, None]
    return output, final


def reconstruct(
    chunks: list[dict[str, Any]], a: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow recorded scan/update boundaries with FP32 recurrent state."""
    first = chunks[0]
    heads, width = first["x"].shape[1:]
    state = torch.zeros(heads, width, first["all_b"].shape[-1])
    outputs: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for chunk in chunks:
        storage = chunk["x"].dtype
        x, b, c, dt = [chunk[k].float() for k in ("x", "all_b", "c", "dt")]
        b, c = [v.repeat_interleave(heads // v.shape[1], dim=1) for v in (b, c)]
        references.append(chunk["reference"].reshape_as(x))
        if chunk["route"] == "native_update":
            for t in range(x.shape[0]):
                state *= (dt[t] * a).exp()[:, None, None]
                state += x[t, :, :, None] * (dt[t, :, None] * b[t])[:, None, :]
                y = torch.einsum("hpn,hn->hp", state, c[t]) + x[t] * d[:, None]
                outputs.append(y.to(storage).float()[None])
        else:
            size = chunk["chunk_size"]
            for start in range(0, x.shape[0], size):
                stop = min(x.shape[0], start + size)
                y, state = scan(
                    x[start:stop],
                    b[start:stop],
                    c[start:stop],
                    dt[start:stop],
                    a,
                    d,
                    state,
                    storage,
                )
                outputs.append(y)
    return torch.cat(outputs), torch.cat(references)
