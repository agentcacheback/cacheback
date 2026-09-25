"""Compact Mamba histories and receiver-conditioned contribution reductions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch

from rcc.models.nemotron.mamba_reconstruction import reconstruct


def _norm(value: torch.Tensor) -> torch.Tensor:
    """Bind the pinned last-axis vector norm despite incomplete Torch stubs."""
    dynamic: Any = value
    return cast(torch.Tensor, dynamic.norm(dim=-1))


@dataclass
class History:
    """Observe one layer in source order without retaining the full value history."""

    source_length: int
    trace: bool = False
    count: int = 0
    a: torch.Tensor | None = None
    d: torch.Tensor | None = None
    chunks: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    calls: dict[str, int] = field(default_factory=dict[str, int])

    def append(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        dt: torch.Tensor,
        a: torch.Tensor,
        d: torch.Tensor,
        reference: torch.Tensor,
        route: str,
        *,
        chunk_size: int = 128,
    ) -> None:
        """Bank compact immutable copies of the actual post-convolution kernel inputs."""
        t, heads, _ = x.shape
        if dt.shape != (t, heads) or b.shape != c.shape or b.shape[0] != t:
            raise ValueError("Mamba input geometry differs")
        if heads % b.shape[1] or a.shape != (heads,) or d.shape != (heads,):
            raise ValueError("Mamba grouped-head geometry differs")
        if any(not bool(torch.isfinite(z).all()) for z in (x, b, c, dt, a, d)):
            raise ValueError("Mamba kernel input is nonfinite")
        if not bool((a < 0).all()) or not bool((dt > 0).all()):
            raise ValueError("Mamba transition or timestep is invalid")
        if self.a is None:
            self.a, self.d = a.detach().float().cpu(), d.detach().float().cpu()
        else:
            torch.testing.assert_close(a.detach().float().cpu(), self.a, rtol=2e-6, atol=1e-7)
            torch.testing.assert_close(d.detach().float().cpu(), self.d, rtol=0, atol=0)
        source_count = min(t, max(0, self.source_length - self.count))
        query_start = max(0, min(t, self.source_length - self.count))
        chunk: dict[str, Any] = {
            "b": b[:source_count].detach().cpu().clone(),
            "norm": _norm(x[:source_count].detach().float()).cpu(),
            "dt": dt.detach().float().cpu().clone(),
            "query_c": c[query_start:].detach().cpu().clone(),
        }
        if self.trace:
            chunk.update(
                x=x.detach().cpu().clone(),
                c=c.detach().cpu().clone(),
                all_b=b.detach().cpu().clone(),
                reference=reference.detach().reshape_as(x).float().cpu().clone(),
                route=route,
                chunk_size=chunk_size,
            )
        self.chunks.append(chunk)
        self.count += t
        self.calls[route] = self.calls.get(route, 0) + 1

    def tensors(self, question: tuple[int, int]) -> dict[str, torch.Tensor]:
        """Require complete prefill, rollout and exact appended-question coverage."""
        if self.count != self.source_length + question[1] or self.a is None:
            raise ValueError("Mamba history omitted or duplicated source/query tokens")
        result = {
            k: torch.cat([chunk[k] for chunk in self.chunks])
            for k in ("b", "norm", "dt", "query_c")
        }
        if result["b"].shape[0] != self.source_length:
            raise ValueError("Mamba source history differs from worker length")
        result["a"] = self.a
        return result

    def reconstruction(self) -> dict[str, Any]:
        """Compare a short actual kernel trace against its signed SSM recurrence."""
        if not self.trace or self.a is None or self.d is None:
            raise ValueError("reconstruction requires an explicit short trace")
        actual, reference = reconstruct(self.chunks, self.a, self.d)
        error = (actual - reference).abs()
        outside = int((error > 0.03 + 0.03 * reference.abs()).sum())
        if outside:
            raise RuntimeError(f"Mamba recurrence reconstruction failed: {outside} elements")
        return {
            "pass": True,
            "tokens": actual.shape[0],
            "arithmetic": "pinned SSD BF16 coefficient/state/weighted-B casts; FP32 recurrence",
            "max_abs": float(error.max()),
            "outside_envelope": outside,
        }


def reduce_history(
    history: History, question: tuple[int, int], device: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compute query and write-strength log magnitudes without a full attention map."""
    values = history.tensors(question)
    length = history.source_length
    b, norm, dt, a = (values[k].to(device) for k in ("b", "norm", "dt", "a"))
    query_c = values["query_c"][question[0] : question[1]].to(device).float()
    cumulative = (dt.double() * a.double()).cumsum(0)
    query_cumulative = cumulative[length + question[0] : length + question[1]]
    repeats = dt.shape[1] // b.shape[1]
    pieces: list[torch.Tensor] = []
    for start in range(0, length, 1024):
        end = min(length, start + 1024)
        cb = torch.einsum("qgn,tgn->qtg", query_c, b[start:end].float())
        cb = cb.repeat_interleave(repeats, dim=-1)
        log_values = (dt[start:end].double() * norm[start:end].double()).log()
        coefficients = (
            cb.abs().double().log()
            + log_values[None]
            + query_cumulative[:, None, :]
            - cumulative[None, start:end, :]
        )
        pieces.append(torch.logsumexp(coefficients, dim=(0, 2)).cpu())
    query_log = torch.cat(pieces)
    bnorm = _norm(b.float()).repeat_interleave(repeats, dim=-1)
    write_log = (dt[:length].double() * norm.double() * bnorm.double()).log().logsumexp(-1).cpu()
    if torch.isnan(query_log).any() or not torch.isfinite(query_log).any():
        raise RuntimeError("query contribution reduction is invalid")
    denominator = torch.logsumexp(query_log, 0)
    diagnostics = {
        "tokens": history.count,
        "routes": history.calls,
        "question_rows": question[1] - question[0],
        "worker_log_mass": float(denominator),
        "rolled_fraction_of_worker_mass": float(
            torch.exp(torch.logsumexp(query_log[-40:], 0) - denominator)
        ),
        "last2048_fraction_of_worker_mass": float(
            torch.exp(torch.logsumexp(query_log[-2048:], 0) - denominator)
        ),
        "finite_source_scores": int(torch.isfinite(query_log).sum()),
    }
    return {"query_log": query_log, "write_log": write_log}, diagnostics
