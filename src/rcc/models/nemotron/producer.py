"""vLLM hybrid prefill and latent continuation over resident model weights."""

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from rcc.latent.rollout import Realign, build_realign, latent_rollout
from rcc.models.nemotron.mamba_history import reduce_history
from rcc.models.nemotron.mamba_observe import ObserveMamba
from rcc.models.nemotron.prefill import VllmHybridPrefill
from rcc.models.qwen.capture import capture_qwen_scores
from rcc.models.qwen.engine import HopProduct, WorkerProduct
from rcc.transforms.select.core.kernels import locate_question


class NemotronProducer:
    """Preserve recurrent state while reusing the shared rollout and selector."""

    def __init__(
        self,
        model: Any,
        prefill: VllmHybridPrefill,
        *,
        latent_steps: int = 40,
        observe_mamba: bool = False,
        realign_enabled: bool = True,
    ) -> None:
        """Bind an eval-mode native Nemotron model without copying its weights."""
        self.model = model
        self.device = next(model.parameters()).device
        self.latent_steps = latent_steps
        self.prefill = prefill
        self.observe_mamba = observe_mamba
        self.realign: Realign = build_realign(model, enabled=realign_enabled)

    def produce_worker(
        self,
        prompt_ids: torch.Tensor,
        *,
        judger_ids: torch.Tensor,
        judger_mask: torch.Tensor,
        question_ids: Sequence[int],
    ) -> WorkerProduct:
        """Capture receiver-conditioned attention, optionally with Mamba history."""
        if not self.observe_mamba:
            return self._produce_worker(prompt_ids, judger_ids, judger_mask, question_ids)
        lo, hi, found = locate_question(judger_ids, question_ids)
        if not found or self.latent_steps != 40:
            raise ValueError("Mamba scoring requires a located receiver question and forty steps")
        observer = ObserveMamba(self.prefill.llm, self.model, prompt_ids.shape[1] + 40)
        base_bytes = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            base_bytes = torch.cuda.memory_allocated(self.device)
        with observer:
            product = self._produce_worker(prompt_ids, judger_ids, judger_mask, question_ids)
        started = time.perf_counter()
        signals: dict[int, dict[str, torch.Tensor]] = {}
        for layer, history in observer.histories.items():
            if (
                set(history.calls) != {"vllm_prefill", "native_update", "native_scan"}
                or history.calls["native_update"] != 41
                or history.calls["native_scan"] != 1
            ):
                raise RuntimeError("Mamba scoring missed a prefill, latent or receiver route")
            signals[layer], _ = reduce_history(history, (lo, hi), str(self.device))
        self._sync()
        peak = (
            int(torch.cuda.max_memory_allocated(self.device) - base_bytes)
            if self.device.type == "cuda"
            else 0
        )
        return replace(
            product,
            mamba_signals=signals,
            capture_s=product.capture_s + time.perf_counter() - started,
            capture_peak_bytes=peak,
        )

    def _produce_worker(
        self,
        prompt_ids: torch.Tensor,
        judger_ids: torch.Tensor,
        judger_mask: torch.Tensor,
        question_ids: Sequence[int],
    ) -> WorkerProduct:
        """Prefill, roll forty embeddings and capture the two attention vectors."""
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 2:
            raise ValueError("Nemotron prompt must have shape [1, L], L >= 2")
        ids = prompt_ids.to(self.device)
        self._sync()
        started = time.perf_counter()
        prefix: Any = ids[0, :-1]
        past = self.prefill.extract(prefix.tolist())
        self._sync()
        extract_s = time.perf_counter() - started
        started = time.perf_counter()
        rolled = latent_rollout(
            self.model,
            ids[:, -1:],
            latent_steps=self.latent_steps,
            realign=self.realign,
            past=past,
            record_embeds=True,
        )
        self._sync()
        roll_s = time.perf_counter() - started
        base_bytes = 0
        if self.device.type == "cuda":
            if not self.observe_mamba:
                torch.cuda.reset_peak_memory_stats(self.device)
            base_bytes = torch.cuda.memory_allocated(self.device)
        started = time.perf_counter()
        scores, found = capture_qwen_scores(
            self.model,
            rolled.past,
            judger_ids.to(self.device),
            judger_mask.to(self.device),
            question_ids,
        )
        self._sync()
        capture_s = time.perf_counter() - started
        peak = (
            int(torch.cuda.max_memory_allocated(self.device) - base_bytes)
            if self.device.type == "cuda"
            else 0
        )
        length = int(ids.shape[1]) + self.latent_steps
        if rolled.embeds is None or rolled.embeds.shape[1] != self.latent_steps + 1:
            raise RuntimeError("Nemotron rollout lost embedding rows")
        with torch.no_grad():
            embeds = torch.cat(
                (self.model.get_input_embeddings()(ids[:, :-1]), rolled.embeds), dim=1
            )
        if rolled.past.get_seq_length() != length:
            raise RuntimeError("Nemotron capture mutated the worker continuation")
        return WorkerProduct(embeds, scores, length, found, extract_s, roll_s, capture_s, peak)

    def _produce_carried_hop(
        self,
        prefix_rows: torch.Tensor,
        tail_ids: torch.Tensor,
        *,
        judger_ids: torch.Tensor,
        judger_mask: torch.Tensor,
        question_ids: Sequence[int],
        consume_past: bool = False,
    ) -> HopProduct:
        """Produce one aligned Nemotron hop over optional carried rows."""
        del consume_past
        if (
            tail_ids.ndim != 2
            or tail_ids.shape[0] != 1
            or tail_ids.shape[1] < 2
            or prefix_rows.ndim != 3
            or prefix_rows.shape[0] != 1
        ):
            raise ValueError("Nemotron hop needs tail ids [1, L] and prefix rows [1, P, D]")
        ids = tail_ids.to(self.device)
        prefix = prefix_rows.to(self.device)
        head_ids = ids[:, :-1]
        with torch.no_grad():
            head_rows = torch.cat((prefix, self.model.get_input_embeddings()(head_ids)), dim=1)
        started = time.perf_counter()
        past = self.prefill.extract_embeds(head_rows[0])
        self._sync()
        extract_s = time.perf_counter() - started
        started = time.perf_counter()
        rolled = latent_rollout(
            self.model,
            ids[:, -1:],
            latent_steps=self.latent_steps,
            realign=self.realign,
            past=past,
            record_embeds=True,
        )
        self._sync()
        roll_s = time.perf_counter() - started
        base_bytes = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            base_bytes = torch.cuda.memory_allocated(self.device)
        started = time.perf_counter()
        scores, found = capture_qwen_scores(
            self.model,
            rolled.past,
            judger_ids.to(self.device),
            judger_mask.to(self.device),
            question_ids,
        )
        self._sync()
        capture_s = time.perf_counter() - started
        peak = (
            int(torch.cuda.max_memory_allocated(self.device) - base_bytes)
            if self.device.type == "cuda"
            else 0
        )
        if rolled.embeds is None:
            raise RuntimeError("Nemotron hop rollout lost its embedding rows")
        embeds = torch.cat((head_rows, rolled.embeds), dim=1)
        expected = int(prefix.shape[1]) + int(ids.shape[1]) + self.latent_steps
        if rolled.past.get_seq_length() != expected or embeds.shape[1] != expected:
            raise RuntimeError("Nemotron hop differs from prefix plus prompt plus latent geometry")
        return HopProduct(
            embeds=embeds,
            scores=scores,
            length=expected,
            question_found=found,
            extract_s=extract_s,
            roll_s=roll_s,
            capture_s=capture_s,
            capture_peak_bytes=peak,
            prefix_rows=int(prefix.shape[1]),
            prompt_rows=int(ids.shape[1]),
        )

    def produce_hop(
        self,
        prefix_rows: torch.Tensor | None,
        tail_ids: torch.Tensor,
        *,
        judger_ids: torch.Tensor,
        judger_mask: torch.Tensor,
        question_ids: Sequence[int],
        consume_past: bool = False,
    ) -> HopProduct:
        """Produce one aligned Nemotron hop over optional carried rows."""
        del consume_past
        if self.observe_mamba:
            raise ValueError("Nemotron chain hops do not support Mamba observation")
        if prefix_rows is None:
            product = self._produce_worker(tail_ids, judger_ids, judger_mask, question_ids)
            return HopProduct(
                embeds=product.embeds,
                scores=product.scores,
                length=product.length,
                question_found=product.question_found,
                extract_s=product.extract_s,
                roll_s=product.roll_s,
                capture_s=product.capture_s,
                capture_peak_bytes=product.capture_peak_bytes,
                prefix_rows=0,
                prompt_rows=int(tail_ids.shape[1]),
            )
        return self._produce_carried_hop(
            prefix_rows,
            tail_ids,
            judger_ids=judger_ids,
            judger_mask=judger_mask,
            question_ids=question_ids,
        )

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
