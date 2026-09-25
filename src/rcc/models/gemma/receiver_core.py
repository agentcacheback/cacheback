"""The Gemma receiver mechanism shared by the resident and split routes.

It holds the shared embedding table, the registered chat turn, the engine
identity check, and the assembly that turns one capture artifact into a prompt.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from rcc.models.gemma.capture_types import CaptureArtifact
from rcc.models.gemma.contract import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    EMBEDDING_SHAPE,
    LATENT_POLICIES,
    LATENT_STEPS,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    N_SEEDS,
    QSNAP_ARMS,
    RECEIVER_ENABLE_THINKING,
    RECEIVER_SLIDING_WINDOW,
    REGISTERED_CUT_ARMS,
    STOP_IDS,
    arm_seeds,
    sampling_contract,
)
from rcc.models.gemma.embedding import await_shared_embedding
from rcc.models.gemma.engine import Sampling, free_gpu_memory
from rcc.models.gemma.engine_contract import REGISTERED_SPEC, suppression_bad_words
from rcc.models.gemma.engine_roles import (
    CAPTURE_ENGINE_IDENTITY,
    GemmaEngineIdentity,
    verify_engine_identity,
)
from rcc.models.gemma.layout import (
    arm_payload_layout,
    payload_blocks,
)
from rcc.models.gemma.mechanism import (
    channel_token_ids,
    encode,
    token_embeds,
)
from rcc.models.gemma.parity import (
    qwen_flat_payload_identity,
    selected_indices_sha256,
)
from rcc.models.gemma.results import PreparedArm
from rcc.models.gemma.schedule import TEXT_ARMS, gemma_answer_closer
from rcc.models.gemma.source_geometry import assemble_receiver_request
from rcc.models.gemma.window import (
    require_registered_thinking_turn,
    require_window_fit,
)
from rcc.run.fleet.admission import AdmissionGate
from rcc.run.fleet.latency import WARMUP_BODY, WARMUP_MAX_TOKENS, warmup_seed
from rcc.run.fleet.stream import ItemStreamTracker, StreamDecoder
from rcc.run.fleet.vllm import VllmEngineHandle

SeedResolver = Callable[[str, str], tuple[int, ...]]


class GemmaReceiverCore:
    """One warm Gemma decode engine and the payload assembly it serves."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        shared_weight: Path,
        runtime_fingerprint: str,
        publication_id: str,
        engine: Any,
        model_load_s: float,
        seed_resolver: SeedResolver = arm_seeds,
        flat_interleave_arms: tuple[str, ...] = (),
        engine_contract: GemmaEngineIdentity = CAPTURE_ENGINE_IDENTITY,
        text_report_bundles: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
        shared_weight_wait_s: float = 0.0,
    ) -> None:
        """Bind the shared table, the registered turn, and one live engine.

        ``shared_weight_wait_s`` is how long to wait for the table: zero on the
        resident route, which captures before it serves.
        """
        self.tokenizer = tokenizer
        self.seed_resolver = seed_resolver
        self.text_report_bundles: dict[str, dict[str, dict[str, Any]]] = {
            arm: {qid: dict(bundle) for qid, bundle in rows.items()}
            for arm, rows in (text_report_bundles or {}).items()
        }
        self.flat_interleave_arms = tuple(flat_interleave_arms)
        allowed_flat_arms = set(LATENT_POLICIES) | set(QSNAP_ARMS)
        if len(set(self.flat_interleave_arms)) != len(self.flat_interleave_arms) or any(
            arm not in allowed_flat_arms for arm in self.flat_interleave_arms
        ):
            raise ValueError("Gemma Qwen-flat receiver arms must uniquely name latent base arms")
        self.engine_contract = engine_contract
        self.weight, self.embedding_digests = await_shared_embedding(
            shared_weight,
            checkpoint=CHECKPOINT_ID,
            revision=CHECKPOINT_REVISION,
            runtime_fingerprint=runtime_fingerprint,
            publication_id=publication_id,
            expected_shape=EMBEDDING_SHAPE,
            wait_s=shared_weight_wait_s,
        )
        # The gather runs on the receiver's device and ``token_embeds`` lands
        # every row on the CPU for vLLM. The bytes are the same either way: a
        # bf16 product of two bf16 values is exact in float32 on both devices.
        if torch.cuda.is_available():
            self.weight = self.weight.to(device="cuda")
        self.channel_open, self.channel_close = channel_token_ids(tokenizer)
        delimiter_ids = encode(tokenizer, "\n\n")
        if len(delimiter_ids) != 1:
            raise RuntimeError(
                f"the fragment delimiter must be one token, got {len(delimiter_ids)}; "
                "the tier budget charges exactly one row per boundary"
            )
        self.delimiter = self._embed(delimiter_ids)
        prefix, suffix = require_registered_thinking_turn(tokenizer)
        if not RECEIVER_ENABLE_THINKING or suffix[-1] == self.channel_close:
            raise RuntimeError("Gemma receiver thinking flag did not take effect")
        self.turn_prefix = self._embed(prefix)
        self.turn_suffix = self._embed(suffix)
        self.bad_words = suppression_bad_words(tokenizer, REGISTERED_SPEC)
        self._require_sampling_contract()
        if engine is None:
            raise RuntimeError("Gemma receiver requires the resident capture engine")
        self.engine: Any | None = engine
        self.handle: VllmEngineHandle | None = None
        self.decoder: StreamDecoder | None = None
        self.gate: AdmissionGate | None = None
        self.tracker: ItemStreamTracker | None = None
        self.engine_identity: dict[str, Any] = {}
        self.engine_identity_sha256 = ""
        self.model_load_s = float(model_load_s)
        self.warmup_s = 0.0
        self.kv_pool_tokens = 0

    @staticmethod
    def _require_sampling_contract() -> None:
        sampling_spec = cast(dict[str, Any], sampling_contract())
        required_sampling = {
            "temperature",
            "top_p",
            "top_k",
            "presence_penalty",
            "answer_ceiling",
            "report_ceiling",
            "sample_tags",
        }
        if not required_sampling.issubset(sampling_spec):
            missing = sorted(required_sampling - set(sampling_spec))
            raise ValueError(f"Gemma receiver sampling contract is missing {missing!r}")
        if sampling_spec["sample_tags"] != [f"s{index}" for index in range(N_SEEDS)]:
            raise ValueError("Gemma receiver sampling tags differ from the three logical draws")
        if int(sampling_spec["answer_ceiling"]) < 1:
            raise ValueError("Gemma receiver answer ceiling must be positive")

    def _embed(self, token_ids: Sequence[int]) -> torch.Tensor:
        return token_embeds(
            self.weight,
            token_ids,
            dtype=torch.bfloat16,
        )

    def _arm_seeds(self, qid: str, arm: str) -> tuple[int, ...]:
        """Resolve one arm's draws through the injected seed resolver."""
        resolver = getattr(self, "seed_resolver", arm_seeds)
        seeds = tuple(resolver(qid, arm))
        if len(seeds) != N_SEEDS:
            raise RuntimeError(f"{qid}/{arm}: seed resolver returned {len(seeds)} draws")
        return seeds

    def _answer_ceiling(self) -> int:
        """Return the benchmark ceiling from the registered Gemma contract."""
        raw_ceiling = sampling_contract()["answer_ceiling"]
        if isinstance(raw_ceiling, bool) or not isinstance(raw_ceiling, int):
            raise RuntimeError("Gemma receiver answer ceiling is malformed")
        return raw_ceiling

    def _retained_text(
        self,
        item: dict[str, Any],
        keeps: tuple[tuple[int, ...], ...],
    ) -> str:
        """Decode what actually shipped, frame rows included: the keep is framed."""
        return "\n\n".join(
            self.tokenizer.decode(
                [
                    int(item["prompt_ids"][worker][position])
                    for position in keep
                    if position < int(item["prompt_ids"][worker].numel())
                ]
            )
            for worker, keep in enumerate(keeps)
        )

    def _require_protected_tail(
        self,
        item: dict[str, Any],
        worker: int,
        keep: tuple[int, ...],
    ) -> None:
        length = int(item["prompt_ids"][worker].numel())
        tail = [position for position in keep if position >= length]
        if len(tail) != LATENT_STEPS:
            raise RuntimeError(
                f"{item['qid']}/w{worker}: keep carries {len(tail)} latent thoughts, "
                f"expected the protected {LATENT_STEPS}"
            )

    def _latent_block(
        self,
        item: dict[str, Any],
        capture: CaptureArtifact,
        worker: int,
        keep: tuple[int, ...],
    ) -> torch.Tensor:
        """The full arm's per-worker payload: framed rows in source order, then cargo."""
        self._require_protected_tail(item, worker, keep)
        framed = item["prompt_ids"][worker]
        length = int(framed.numel())
        evidence = [position for position in keep if position < length]
        rolled = capture.rolled_by_worker[worker]
        if not evidence:
            return rolled
        return torch.cat((self._embed([int(framed[position]) for position in evidence]), rolled))

    def warm(self) -> None:
        """Fail-closed verify and warm the engine registered for this role."""
        engine = self.engine
        if engine is None:
            raise RuntimeError("Gemma resident engine was closed before receiver warmup")
        self.engine_identity = self._verify_engine()
        encoded_identity = json.dumps(
            self.engine_identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        self.engine_identity_sha256 = hashlib.sha256(encoded_identity).hexdigest()
        warm_started = time.perf_counter()
        warm_prompt = torch.cat(
            (
                self.turn_prefix,
                self._embed(encode(self.tokenizer, WARMUP_BODY) or [0]),
                self.turn_suffix,
            ),
            dim=0,
        )
        engine.decode_embeds_full(
            [warm_prompt],
            Sampling(
                temperature=0.0,
                max_tokens=WARMUP_MAX_TOKENS,
                stop_token_ids=tuple(sorted(STOP_IDS)),
                bad_words=self.bad_words,
            ),
            # The warmup completion is discarded and never banked, so its draw
            # sits outside every panel seed law; the seed comes from the shared
            # warmup derivation.
            seeds=[warmup_seed("gemma4", "resident_receiver")],
        )
        self.warmup_s = time.perf_counter() - warm_started
        self.handle = VllmEngineHandle(
            engine,
            require_public_enqueue=True,
            detokenize=False,
        )
        # The closer is bound on every lane and is inert here: the trigger
        # cannot fire on a draw that already carries a closer id.
        self.decoder = StreamDecoder(
            self.handle,
            clock=time.monotonic,
            closer=gemma_answer_closer(
                self._embed,
                channel_open=self.channel_open,
                channel_close=self.channel_close,
                answer_ceiling=self._answer_ceiling(),
            ),
        )
        pool_tokens = engine.kv_pool_tokens()
        if pool_tokens is None:
            raise RuntimeError("Gemma vLLM KV pool capacity is unreadable")
        self.kv_pool_tokens = int(pool_tokens)
        self.gate = AdmissionGate(
            pool_tokens,
            answer_ceiling=self._answer_ceiling(),
            max_in_flight=MAX_NUM_SEQS,
        )
        self.tracker = ItemStreamTracker(self.decoder, self.gate)

    def _verify_engine(self) -> dict[str, Any]:
        return verify_engine_identity(self.engine, self.engine_contract)

    def _prepare(self, item: dict[str, Any], capture: CaptureArtifact, arm: str) -> Any:
        if arm in TEXT_ARMS:
            from rcc.models.gemma.text_results import prepare_text_receiver

            try:
                bundle = self.text_report_bundles[arm][str(item["qid"])]
            except KeyError as exc:
                raise RuntimeError(f"{item['qid']}/{arm}: text report bundle is missing") from exc
            return prepare_text_receiver(self.tokenizer, self.weight, item, bundle)
        started = time.perf_counter()
        prompt_ids = cast(tuple[torch.Tensor, ...], item["prompt_ids"])
        lengths = [int(ids.numel()) for ids in prompt_ids]
        try:
            keeps = capture.keeps_by_arm[arm]
            selection_s = capture.selection_s_by_arm[arm]
        except KeyError as exc:
            raise RuntimeError(f"{item['qid']}/{arm}: selection artifact is missing") from exc
        if len(keeps) != len(lengths):
            raise RuntimeError(f"{item['qid']}/{arm}: selection worker roster is incomplete")
        layout_stats = {"delimiter_rows": 0, "tier_spans": 0, "tier_rows": 0}
        if arm == "floor":
            blocks = []
            kinds: list[str] = []
            retained = ""
            block_kind = "none"
        elif arm == "full":
            blocks = [
                self._latent_block(item, capture, worker, keep) for worker, keep in enumerate(keeps)
            ]
            kinds = [f"worker_memory_w{worker}" for worker in range(len(keeps))]
            retained = self._retained_text(item, keeps)
            block_kind = "worker_memory"
        else:
            if arm not in REGISTERED_CUT_ARMS:
                raise RuntimeError(f"{item['qid']}/{arm}: unregistered receiver arm")
            for worker, keep in enumerate(keeps):
                self._require_protected_tail(item, worker, keep)
            layout, block_kind = arm_payload_layout(
                arm,
                capture.layouts_by_arm,
                flat_interleave=arm in getattr(self, "flat_interleave_arms", ()),
            )
            blocks, kinds, layout_stats = payload_blocks(
                block_kind,
                prompt_ids,
                layout,
                keeps,
                capture.rolled_by_worker,
                self._embed,
                self.delimiter,
            )
            retained = self._retained_text(item, keeps)
        prompt = self._embed(cast(Sequence[int], item["request_ids"]))
        assembly = assemble_receiver_request(
            turn_prefix=self.turn_prefix,
            memory_blocks=blocks,
            prompt=prompt,
            turn_suffix=self.turn_suffix,
        )
        if arm in REGISTERED_CUT_ARMS:
            # The whole chat turn follows the payload, so the rows the window
            # owes the question are prefix, prompt, and suffix.
            require_window_fit(
                tag=f"{item['qid']}/{arm}",
                tier_rows=int(layout_stats["tier_rows"]),
                trailing_rows=(assembly.prefix_rows + assembly.prompt_rows + assembly.suffix_rows),
            )
        answer_ceiling = self._answer_ceiling()
        if assembly.total_rows + answer_ceiling > MAX_MODEL_LEN:
            raise RuntimeError(
                f"{item['qid']}/{arm}: {assembly.total_rows} prompt rows plus the "
                f"{answer_ceiling} ceiling exceed {MAX_MODEL_LEN}"
            )
        return PreparedArm(
            prompt=assembly.rows.cpu(),
            keeps=keeps,
            segments=assembly.segments,
            retained_text=retained,
            block_kind=block_kind,
            block_kinds=tuple(kinds),
            window_binds=assembly.total_rows > RECEIVER_SLIDING_WINDOW,
            delimiter_rows=int(layout_stats["delimiter_rows"]),
            tier_spans=int(layout_stats["tier_spans"]),
            tier_rows=int(layout_stats["tier_rows"]),
            handoff_rows=assembly.latent_rows,
            base_prompt_rows=assembly.total_rows - assembly.latent_rows,
            selection_s=selection_s,
            receiver_prepare_s=time.perf_counter() - started,
            selected_indices_sha256=selected_indices_sha256(keeps),
            qwen_flat_payload_identity=(
                qwen_flat_payload_identity(blocks) if block_kind == "qwen_interleave" else None
            ),
        )

    def _sample_params(self, seed: int) -> Any:
        if self.handle is None:
            raise RuntimeError("Gemma vLLM stream handle is not ready")
        sampling_spec = cast(dict[str, Any], sampling_contract())
        return self.handle.sampling(
            Sampling(
                temperature=float(sampling_spec["temperature"]),
                top_p=float(sampling_spec["top_p"]),
                top_k=int(sampling_spec["top_k"]),
                presence_penalty=float(sampling_spec["presence_penalty"]),
                max_tokens=int(sampling_spec["answer_ceiling"]),
                stop_token_ids=tuple(sorted(STOP_IDS)),
                bad_words=self.bad_words,
            ),
            seed,
        )

    def close(self) -> None:
        """Abort outstanding requests and release receiver resources."""
        if self.decoder is not None:
            self.decoder.abort_all()
        if self.engine is not None:
            self.engine.close()
        self.decoder = None
        self.handle = None
        self.engine = None
        self.weight = torch.empty(0)
        free_gpu_memory(settle_seconds=1.0)


__all__ = ("GemmaReceiverCore", "SeedResolver")
