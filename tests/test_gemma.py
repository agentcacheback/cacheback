"""The Gemma runtime: its payload layout, keeps, capture, recurrence, handoff, and bridge."""

from __future__ import annotations

import json
import random
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tests.gemma_split_support import (
    _EMBEDDING_DIGESTS,
    _FRAME_PREFIX,
    _FRAME_SUFFIX,
    _GLOBAL_LAYERS,
    _HIDDEN,
    _QID,
    _REPLAYS,
    _rolled_tensor,
    _write_rolled,
    embedding_table,
)

from rcc.benchmarks.fanoutqa.gemma_data import RollFrame
from rcc.latent.rollout import build_realign, latent_rollout
from rcc.models.gemma import capture_bridge as bridge_module
from rcc.models.gemma import capture_runtime as capture
from rcc.models.gemma import capture_support, handoff, selection
from rcc.models.gemma.capture_bridge import (
    GemmaEngineCaptureBridge,
    gemma4_hf_view,
    rolling_cache_from_artifact,
)
from rcc.models.gemma.capture_connector import (
    extract_hybrid_kv,
    read_layer_pages,
)
from rcc.models.gemma.capture_support import LATENT_ROLL_SCHEMA, latent_roll_filename
from rcc.models.gemma.capture_types import CaptureArtifact
from rcc.models.gemma.contract import (
    ARM_POLICIES,
    ARMS,
    CUT_ARMS,
    LATENT_POLICIES,
    LATENT_REALIGN_ENABLED,
    LATENT_ROLL_PREFIX_TEMPLATE,
    LATENT_ROLL_SUFFIX_TEMPLATE,
    LATENT_STEPS,
    QSNAP_ARM_POLICIES,
    QSNAP_ARMS,
    REPORT_STYLE,
    SELECTION_SCHEMA,
    WORKERS_PER_ITEM,
)
from rcc.models.gemma.embedding import (
    embedding_marker_path,
    embedding_validation_path,
    hardlink_shared_embedding,
    load_validated_shared_embedding,
    publish_shared_embedding_once,
    shared_embedding_path,
)
from rcc.models.gemma.layout import (
    QWEN_BLOCK_KIND,
    ArmLayout,
    arm_payload_layout,
    flat_interleave_layout,
    payload_blocks,
)
from rcc.models.gemma.prompts import worker_prompt
from rcc.models.gemma.selection import (
    SelectionArtifact,
    load_or_bank_selection,
    payload_length,
    worker_keep,
)
from rcc.run.fleet.clocks import HandoffClocks, clock_record_path
from rcc.run.gemma.cell_banks import ShardLog
from rcc.run.io import sha256_file
from rcc.topologies.fanout.layout import interleaved_blocks, source_layout
from rcc.transforms.select.query_support.capture_bank import BANK_SCOPE_GLOBAL_ONLY, LayerBank

# --- The flat-interleave payload layout.

_MEMORY = 4_096


_PAYLOAD = _MEMORY + LATENT_STEPS


_DIM = 4


def _keeps() -> tuple[tuple[int, ...], ...]:
    keep = tuple(range(0, _MEMORY, 2 * 16 * 2))
    spans = tuple(position for start in keep for position in range(start, start + 16))
    tail = tuple(range(_MEMORY, _PAYLOAD))
    return tuple((*spans, *tail) for _ in range(3))


def _fixture():
    memory_ids = tuple(torch.arange(_MEMORY) % 7 for _ in range(3))
    rolled = tuple(
        torch.full((LATENT_STEPS, _DIM), float(worker), dtype=torch.bfloat16) for worker in range(3)
    )
    table = torch.arange(8 * _DIM, dtype=torch.bfloat16).reshape(8, _DIM)

    def embed(ids: Sequence[int]) -> torch.Tensor:
        return table[[value % 8 for value in ids]]

    return memory_ids, rolled, embed


def test_flat_payload_is_per_worker_evidence_then_cargo_without_delimiters() -> None:
    keeps = _keeps()
    memory_ids, rolled, embed = _fixture()
    layout = flat_interleave_layout(keeps, (_MEMORY,) * 3)
    assert layout == source_layout(keeps, (_MEMORY,) * 3)
    assert layout.tier_start == len(layout.span_order)
    assert layout.cargo_first is False
    blocks, kinds, stats = interleaved_blocks(memory_ids, layout, keeps, rolled, embed)
    assert kinds == [f"worker_memory_w{worker}" for worker in range(3)]
    assert stats["delimiter_rows"] == stats["tier_rows"] == stats["tier_spans"] == 0
    for worker, block in enumerate(blocks):
        evidence = [position for position in keeps[worker] if position < _MEMORY]
        assert int(block.shape[0]) == len(evidence) + LATENT_STEPS
        assert torch.equal(
            block[: len(evidence)],
            embed([int(memory_ids[worker][position]) for position in sorted(evidence)]),
        )
        assert torch.equal(block[-LATENT_STEPS:], rolled[worker])
    broken = ArmLayout(span_order=tuple(reversed(layout.span_order)), tier_start=0)
    with pytest.raises(RuntimeError, match="per-worker source-order layout"):
        interleaved_blocks(memory_ids, broken, keeps, rolled, embed)


def test_payload_and_arm_mapping_only_expose_the_live_emitter() -> None:
    keeps = _keeps()
    memory_ids, rolled, embed = _fixture()
    layout = flat_interleave_layout(keeps, (_MEMORY,) * 3)
    blocks = payload_blocks(
        QWEN_BLOCK_KIND, memory_ids, layout, keeps, rolled, embed, torch.empty(1, _DIM)
    )
    assert blocks[1] == [f"worker_memory_w{worker}" for worker in range(3)]
    assert arm_payload_layout("r2", {"r2": layout}, flat_interleave=True) == (
        layout,
        QWEN_BLOCK_KIND,
    )
    with pytest.raises(ValueError, match="no banked payload layout"):
        arm_payload_layout("r2", {})
    with pytest.raises(ValueError, match="require flat-interleave"):
        arm_payload_layout("r2", {"r2": layout})


# --- The HF-phase keeps and layouts.

_MEMORY_TOKENS = 5_208


_FRAME = RollFrame((11, 12, 13), (14, 15))


_PROMPT_TOKENS = len(_FRAME.prefix_ids) + _MEMORY_TOKENS + len(_FRAME.suffix_ids)


_PAYLOAD_TOKENS = _PROMPT_TOKENS + LATENT_STEPS


def _item() -> dict[str, Any]:
    memory = tuple(torch.arange(_MEMORY_TOKENS) for _worker in range(3))
    return {
        "qid": "q1",
        "memory_ids": memory,
        "prompt_ids": tuple(_FRAME.prompt(ids) for ids in memory),
        "frame": _FRAME,
    }


def _scores() -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(
        {"support": torch.arange(_PAYLOAD_TOKENS, dtype=torch.float32)} for _worker in range(3)
    )


def _snap_scores() -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(
        {"snap": torch.arange(_PAYLOAD_TOKENS, dtype=torch.float32) * (worker + 1)}
        for worker in range(3)
    )


def test_qsnap_selection_banks_exact_w16_indices_and_flat_layouts(tmp_path: Any) -> None:
    scores = _snap_scores()
    # The explicit flat mechanism the banking wraps.
    derived = selection._derive_qsnap_selection(_item(), scores)
    assert set(derived.keeps_by_arm) == set(QSNAP_ARMS)
    path = tmp_path / "qsnap" / "q1.json"
    first = selection.load_or_bank_qsnap_selection(_item(), scores, path)
    assert first.reloaded is False
    assert set(first.keeps_by_arm) == set(QSNAP_ARMS)
    for arm in QSNAP_ARMS:
        ratio = int(arm.removeprefix("qsnap_r"))
        assert QSNAP_ARM_POLICIES[arm] == f"gemma4_12b_r{ratio}_w16"
        assert all(
            keep == tuple(selection.qsnap_worker_keep(arm, scores[worker]["snap"], _PAYLOAD_TOKENS))
            for worker, keep in enumerate(first.keeps_by_arm[arm])
        )
        assert all(
            layout == flat_interleave_layout(keeps, (_PROMPT_TOKENS,) * 3)
            for arm, keeps, layout in [(arm, first.keeps_by_arm[arm], first.layouts_by_arm[arm])]
        )
    assert first.keeps_by_arm == derived.keeps_by_arm
    assert first.layouts_by_arm == derived.layouts_by_arm
    replayed = selection.load_or_bank_qsnap_selection(_item(), scores, path)
    assert replayed.reloaded is True
    assert replayed.keeps_by_arm == first.keeps_by_arm
    assert replayed.layouts_by_arm == first.layouts_by_arm


def test_selection_banks_one_flat_layout_for_each_live_arm(tmp_path: Any) -> None:
    first = selection.load_or_bank_selection(
        _item(), _scores(), tmp_path / "selection.json", seat_keys_by_ratio={}, ratio_cuts={}
    )
    assert set(first.keeps_by_arm) == set(ARMS)
    assert set(first.layouts_by_arm) == set(CUT_ARMS)
    assert first.keeps_by_arm["floor"] == ((), (), ())
    assert worker_keep("full", torch.empty(0), _PAYLOAD_TOKENS) == list(range(_PAYLOAD_TOKENS))
    for arm in LATENT_POLICIES:
        assert first.layouts_by_arm[arm] == flat_interleave_layout(
            first.keeps_by_arm[arm], (_PROMPT_TOKENS,) * 3
        )


def test_short_worker_protects_sink_and_rolled_tail_at_deep_ratios() -> None:
    length = 2_769 + LATENT_STEPS
    scores = torch.arange(length, dtype=torch.float32)
    keep = selection.worker_keep("r128", scores, length)
    assert keep == [0, *range(2_769, length)]
    assert selection.worker_keep("r64", scores, length)


# --- The resident-engine capture route.


class _Descriptor:
    def __init__(self, *, is_global: bool, window: int | None) -> None:
        self.is_global = is_global
        self.window = window


class _Row:
    def __init__(self, *, is_global: bool, window: int | None) -> None:
        self.descriptor = _Descriptor(is_global=is_global, window=window)


class _Bank:
    def __init__(self, length: int) -> None:
        self.length = length

    @staticmethod
    def selected_rows(_scope: str | None = None) -> list[_Row]:
        return [
            _Row(is_global=True, window=None),
            _Row(is_global=False, window=1_024),
        ]

    @staticmethod
    def replay_moments(
        *, scope: str | None = None, device: torch.device | str | None = None
    ) -> object:
        del scope, device
        return object()

    def replay_score(self, policy: str, *, scope: str | None = None) -> torch.Tensor:
        del policy, scope
        return torch.arange(self.length, dtype=torch.float32)


class _DurableBank:
    def __init__(self, *, device_moments: object, durable: torch.Tensor) -> None:
        self.device_moments = device_moments
        self.durable = durable

    def replay_moments(
        self, *, scope: str | None = None, device: torch.device | str | None = None
    ) -> object:
        assert scope == BANK_SCOPE_GLOBAL_ONLY
        assert device == torch.device("cpu")
        return self.device_moments

    def replay_score(self, policy: str, *, scope: str | None = None) -> torch.Tensor:
        assert policy == ARM_POLICIES["r2"]
        assert scope == BANK_SCOPE_GLOBAL_ONLY
        return self.durable


class _Model:
    def __init__(self, hidden: int = capture.EMBEDDING_SHAPE[1]) -> None:
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[dict[str, Any]] = []
        self._weight = torch.zeros(2, hidden)

    def parameters(self) -> Any:
        return iter((self.parameter,))

    def get_input_embeddings(self) -> Any:
        return SimpleNamespace(weight=self._weight)

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(past_key_values="rolling-past")


class _Rolling:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class _Bridge:
    def __init__(self, model: _Model, past: _Rolling) -> None:
        self.model = model
        self.tokenizer = None
        self.past = past
        self.extracted: list[int] | None = None

    def extract(self, token_ids: Sequence[int]) -> _Rolling:
        self.extracted = list(token_ids)
        return self.past

    def release_view(self) -> None:
        pass


_FRAME_CAPTURE = capture.RollFrame((11, 12, 13), (14, 15))


# The Qwen selectable domain: the whole framed prompt plus the latent tail.
_PAYLOAD_TOKENS_CAPTURE = _PROMPT_TOKENS + capture.LATENT_STEPS


def _item_capture(length: int = _MEMORY_TOKENS) -> dict[str, Any]:
    memory = tuple(torch.arange(length) for _worker in range(3))
    return {
        "qid": "q1",
        "memory_ids": memory,
        "prompt_ids": tuple(_FRAME_CAPTURE.prompt(ids) for ids in memory),
        "frame": _FRAME_CAPTURE,
        "judger_ids": torch.tensor([[20, 21]], dtype=torch.long),
        "question_ids": torch.tensor([30, 31], dtype=torch.long),
    }


def _fake_roll(rolling_past: object) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def roll(model: Any, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        calls.append({"model": model, "input_ids": input_ids, **kwargs})
        length = int(input_ids.shape[1])
        return SimpleNamespace(
            past=rolling_past,
            embeds=torch.zeros(
                1,
                length + capture.LATENT_STEPS,
                capture.EMBEDDING_SHAPE[1],
                dtype=torch.bfloat16,
            ),
        )

    return roll, calls


def test_durable_qsnap_replay_is_bit_exact_and_pins_the_same_keeps() -> None:
    live = torch.arange(_PAYLOAD_TOKENS_CAPTURE, dtype=torch.float32)

    class Bank:
        @staticmethod
        def replay_score(policy: str, *, scope: str | None = None) -> torch.Tensor:
            assert policy == "snap"
            assert scope == BANK_SCOPE_GLOBAL_ONLY
            return live.clone()

    durable = capture_support.durable_qsnap_score(
        "q1/w0",
        cast(Any, Bank()),
        live,
        payload=_PAYLOAD_TOKENS_CAPTURE,
    )
    assert torch.equal(
        durable,
        capture_support.pinned_rows("q1/w0", live, _PAYLOAD_TOKENS_CAPTURE),
    )

    class DriftedBank(Bank):
        @staticmethod
        def replay_score(policy: str, *, scope: str | None = None) -> torch.Tensor:
            result = Bank.replay_score(policy, scope=scope)
            result[-1] += 1
            return result

    with pytest.raises(RuntimeError, match="differs from live capture"):
        capture_support.durable_qsnap_score(
            "q1/w0",
            cast(Any, DriftedBank()),
            live,
            payload=_PAYLOAD_TOKENS_CAPTURE,
        )


def test_capture_rolls_the_latent_cargo_onto_the_landed_rolling_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    model = _Model()
    bank = _Bank(_PAYLOAD_TOKENS_CAPTURE)
    rolling = _Rolling(_PROMPT_TOKENS - 1)
    bridge = _Bridge(model, rolling)
    rolled_past = object()
    seen: list[tuple[Any, Any]] = []
    roll, roll_calls = _fake_roll(rolled_past)
    monkeypatch.setattr(capture, "latent_rollout", roll)
    monkeypatch.setattr(
        capture,
        "require_rolling_cache",
        lambda past, active: seen.append((past, active)),
    )
    monkeypatch.setattr(
        capture,
        "capture_query_support",
        lambda *_args, **_kwargs: SimpleNamespace(
            found=True,
            moments=object(),
            layer_bank=bank,
        ),
    )
    monkeypatch.setattr(
        capture_support,
        "selector_score",
        lambda _moments, _composition: torch.arange(_PAYLOAD_TOKENS_CAPTURE, dtype=torch.float32),
    )
    started = time.perf_counter()
    result = capture._capture_worker(
        bridge, _item_capture(), 0, cast(Any, "realign"), _FRAME_CAPTURE
    )
    capture_s = time.perf_counter() - started
    assert model.calls == []
    assert len(roll_calls) == 1
    call = roll_calls[0]
    assert call["model"] is model
    framed = torch.cat(
        (
            torch.tensor(_FRAME_CAPTURE.prefix_ids),
            torch.arange(_MEMORY_TOKENS),
            torch.tensor(_FRAME_CAPTURE.suffix_ids),
        )
    )
    assert bridge.extracted == framed[:-1].tolist()
    assert torch.equal(call["input_ids"], framed[-1:].unsqueeze(0))
    assert call["latent_steps"] == capture.LATENT_STEPS == 40
    assert call["realign"] == "realign"
    assert call["past"] is rolling
    assert call["record_embeds"] is True
    assert seen == [(rolled_past, model)]
    clock_names = {"extract_s", "roll_s", "capture_query_s", "parity_s"}
    assert set(result) == {"support", "_layer_bank", "_rolled", *clock_names}
    assert all(result[name] >= 0.0 for name in clock_names)
    assert sum(result[name] for name in clock_names) <= capture_s
    rolled = result["_rolled"]
    assert rolled.shape == (capture.LATENT_STEPS, capture.EMBEDDING_SHAPE[1])
    assert rolled.dtype == torch.bfloat16
    # The Qwen selectable domain: every framed prompt row plus the tail, so
    # the score is as long as the roll and frame rows compete.
    support = result["support"]
    assert support.shape == (_PAYLOAD_TOKENS_CAPTURE,)
    # The pre-cut sink pin has already run: row zero carries the maximum.
    assert float(support[0]) == float(_PAYLOAD_TOKENS_CAPTURE - 1)
    assert torch.equal(support[1:], torch.arange(1, _PAYLOAD_TOKENS_CAPTURE, dtype=torch.float32))


def test_every_framed_score_carries_the_qwen_pre_cut_sink_pin_at_the_cache_row_count() -> None:
    # One contract, two halves: a score the domain cannot explain is refused,
    # and an explainable one is pinned without mutating the replayed vector.
    with pytest.raises(RuntimeError, match="framed score covers"):
        capture_support.pinned_rows(
            "q1/w0", torch.zeros(_PAYLOAD_TOKENS_CAPTURE - 1), _PAYLOAD_TOKENS_CAPTURE
        )
    values = torch.tensor([0.0, 5.0, 2.0])
    pinned = capture_support.pinned_rows("q1/w0", values, 3)
    assert pinned.tolist() == [5.0, 5.0, 2.0]
    assert values.tolist() == [0.0, 5.0, 2.0]


def test_the_pinned_frame_templates_rebuild_the_text_arm_worker_prompt() -> None:
    class _BodyTokenizer:
        @staticmethod
        def apply_chat_template(messages: Any, **kwargs: Any) -> str:
            assert kwargs["enable_thinking"] is True
            return str(messages[0]["content"])

        @staticmethod
        def decode(ids: Any, skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            return " ".join(str(int(value)) for value in ids)

    question = "Who directed each of these films?"
    evidence = [4, 5, 6]
    body = worker_prompt(
        _BodyTokenizer(),
        question,
        evidence,
        report_style=REPORT_STYLE,
    )
    assert body == (
        LATENT_ROLL_PREFIX_TEMPLATE
        + _BodyTokenizer.decode(evidence)
        + LATENT_ROLL_SUFFIX_TEMPLATE.format(question=question)
    )

    class _BrokenTokenizer(_BodyTokenizer):
        @staticmethod
        def apply_chat_template(*_args: Any, **_kwargs: Any) -> str:
            raise TypeError("thinking flag unsupported")

    with pytest.raises(RuntimeError, match="required thinking flag"):
        worker_prompt(_BrokenTokenizer(), question, evidence, report_style=REPORT_STYLE)


def test_durable_support_replay_keeps_the_registered_selections_or_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_moments = object()
    device_moments = object()
    live = torch.arange(_PAYLOAD_TOKENS_CAPTURE, dtype=torch.float32)

    def bind(durable_for_device: torch.Tensor | None = None) -> None:
        monkeypatch.setattr(
            capture_support,
            "selector_score",
            lambda moments, _composition: (
                live
                if moments is live_moments
                else (durable_for_device if durable_for_device is not None else live)
            ),
        )

    def replay(durable: torch.Tensor) -> torch.Tensor:
        return capture_support.durable_support_score(
            "q1/w0",
            cast(LayerBank, _DurableBank(device_moments=device_moments, durable=durable)),
            cast(Any, live_moments),
            device=torch.device("cpu"),
            payload=_PAYLOAD_TOKENS_CAPTURE,
        )

    # A durable score that moves inside a kept span is the registered selection.
    bind()
    durable = live.clone()
    durable[len(_FRAME_CAPTURE.prefix_ids)] = 0.125
    result = replay(durable)
    assert torch.equal(
        result,
        capture_support.pinned_rows("q1/w0", durable, _PAYLOAD_TOKENS_CAPTURE),
    )
    assert result.shape == (_PAYLOAD_TOKENS_CAPTURE,)

    # A durable score that changes the keep is refused.
    with pytest.raises(RuntimeError, match="durable support replay changes the r2 keep"):
        replay(torch.flip(live, dims=(0,)))

    # So is a same-device global replay that is not bit-exact with the live one.
    device_replay = live.clone()
    device_replay[0] = 0.125
    bind(device_replay)
    with pytest.raises(RuntimeError, match="same-device global layer replay"):
        replay(live.clone())


def test_capture_refuses_a_bank_without_sliding_attention_rows() -> None:
    bank = _Bank(8)
    bank.selected_rows = lambda _scope=None: [_Row(is_global=True, window=None)]
    with pytest.raises(RuntimeError, match="no sliding rows"):
        capture.require_bank_geometry("q1/w0", cast(LayerBank, bank))


def test_capture_replay_restores_the_original_live_clock_and_cargo(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    banks = tmp_path / "banks"
    banks.mkdir()
    path = banks / "q1_w0_layer_bank.pt"
    path.write_bytes(b"bank")
    rolled_dir = tmp_path / "rolled"
    rolled_dir.mkdir()
    rolled_path = rolled_dir / capture.latent_roll_filename("q1", 0)
    rolled_path.write_bytes(b"rolled")
    shard_log = SimpleNamespace(
        banked_capture=lambda _qid, _worker: {
            "capture_s": 12.3456,
            "layer_bank_file": path.name,
            "latent_roll_file": rolled_path.name,
        },
    )
    expected = {"support": torch.arange(8)}
    expected_rolled = torch.zeros(2, 2)
    monkeypatch.setattr(capture, "replay_worker", lambda *_args: expected)
    monkeypatch.setattr(capture, "load_rolled", lambda *_args, **_kwargs: expected_rolled)
    monkeypatch.setattr(
        capture,
        "_capture_worker",
        lambda *_args: pytest.fail("a complete replay must not recapture"),
    )
    scores, rolled, capture_s, reloaded, audit = capture._capture_or_replay_worker(
        _Bridge(_Model(), _Rolling(_PROMPT_TOKENS - 1)),
        _item_capture(),
        0,
        banks=banks,
        shard_log=cast(ShardLog, shard_log),
        realign=cast(Any, None),
        frame=_FRAME_CAPTURE,
    )
    assert scores is expected
    assert rolled is expected_rolled
    assert capture_s == 12.3456
    assert reloaded is True
    assert audit is None

    events: list[Any] = []
    audit_log = SimpleNamespace(
        bank_capture=lambda row, **fields: events.append(("banked", row, fields))
    )
    audit_bank = SimpleNamespace(schema="fixture", to_payload=lambda: {"bank": "payload"})
    monkeypatch.setattr(
        capture_support,
        "atomic_torch_save",
        lambda _value, _path: events.append("bank"),
    )
    monkeypatch.setattr(
        capture_support,
        "load_rolled",
        lambda *_args, **_kwargs: events.append("rolled") or expected_rolled,
    )
    monkeypatch.setattr(
        capture_support,
        "replay_worker",
        lambda *_args, **_kwargs: events.append("replay") or expected,
    )
    capture_support.DeferredAudit(
        item=_item_capture(),
        worker=0,
        bank=cast(Any, audit_bank),
        scores=expected,
        rolled=expected_rolled,
        bank_path=tmp_path / "deferred_bank.pt",
        rolled_path=rolled_path,
        frame=_FRAME_CAPTURE,
        shard_log=cast(ShardLog, audit_log),
        capture_s=12.3456,
        include_qsnap=False,
        extract_s=1.0,
        roll_s=2.0,
        capture_query_s=3.0,
        parity_s=4.0,
    ).run()
    assert events[:3] == ["bank", "rolled", "replay"]
    banked = events[3]
    assert banked[0] == "banked"
    assert banked[1]["audit_s"] >= 0.0
    assert {
        name: banked[1][name] for name in ("extract_s", "roll_s", "capture_query_s", "parity_s")
    } == {
        "extract_s": 1.0,
        "roll_s": 2.0,
        "capture_query_s": 3.0,
        "parity_s": 4.0,
    }

    # A banked capture whose latent roll is gone is refused, never recaptured.
    rollless = SimpleNamespace(
        banked_capture=lambda _qid, _worker: {
            "capture_s": 12.3456,
            "layer_bank_file": path.name,
        },
    )
    monkeypatch.setattr(
        capture,
        "_capture_worker",
        lambda *_args: pytest.fail("a timed capture must not recapture"),
    )
    with pytest.raises(RuntimeError, match="no latent roll"):
        capture._capture_or_replay_worker(
            _Bridge(_Model(), _Rolling(_PROMPT_TOKENS - 1)),
            _item_capture(),
            0,
            banks=banks,
            shard_log=cast(ShardLog, rollless),
            realign=cast(Any, None),
            frame=_FRAME_CAPTURE,
        )


# --- The native recurrence and the fleet order.


def test_native_recurrence() -> None:
    """Observe all forty actual inputs, and tell recurrence from a scaled first step."""
    from rcc.models.gemma.recurrence import build_native_realign

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: list[torch.Tensor] = []

        def forward(self, inputs_embeds: torch.Tensor | None = None, **kwargs: Any) -> Any:
            x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]) if inputs_embeds is None else inputs_embeds
            if inputs_embeds is not None:
                self.inputs.append(x.clone())
            return SimpleNamespace(
                last_hidden_state=x + torch.tanh(x.roll(1, -1)), past_key_values=None
            )

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Backbone()
            self.embedding = torch.nn.Embedding(8, 4)
            with torch.no_grad():
                self.embedding.weight.fill_(0.5)
            self.embedding.register_buffer("embed_scale", torch.tensor(2.0))

        def get_input_embeddings(self) -> torch.nn.Embedding:
            return self.embedding

    model = Model()
    base = build_realign(model, enabled=False)
    native = build_native_realign(model)
    assert float(base.target_norm) == 1.0 and float(native.target_norm) == 2.0
    raw = latent_rollout(
        model, torch.tensor([[1]]), latent_steps=40, realign=base, record_embeds=True
    )
    raw_inputs = model.model.inputs.copy()
    model.model.inputs.clear()
    new = latent_rollout(
        model, torch.tensor([[1]]), latent_steps=40, realign=native, record_embeds=True
    )
    assert len(model.model.inputs) == len(raw_inputs) == 40
    assert torch.allclose(torch.stack(raw_inputs).norm(dim=-1), torch.ones(40, 1, 1))
    assert torch.allclose(torch.stack(model.model.inputs).norm(dim=-1), torch.full((40, 1, 1), 2.0))
    assert new.embeds is not None and raw.embeds is not None
    assert torch.allclose(new.embeds[:, 1], raw.embeds[:, 1] * 2)
    assert not torch.allclose(new.embeds[:, 2:], raw.embeds[:, 2:] * 2)
    _assert_native_embedding_input()


def test_native_embedding_input() -> None:
    """Verify actual native Gemma ID/embed equivalence on its supported pin."""
    _assert_native_embedding_input()


def _assert_native_embedding_input() -> None:
    """Native Gemma reads ids and their own embeddings identically."""
    import transformers

    from rcc.models.gemma.recurrence import build_native_realign

    if not hasattr(transformers, "Gemma4UnifiedForCausalLM"):
        return  # A transformers pin without Gemma 4 cannot build the fixture.
    with torch.random.fork_rng():
        torch.manual_seed(7)
        config = transformers.Gemma4UnifiedTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=2,
            head_dim=8,
            global_head_dim=8,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=16,
            attention_k_eq_v=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        config._attn_implementation = "eager"
        model = transformers.Gemma4UnifiedForCausalLM(config).eval()
    ids = torch.tensor([[1, 7, 11, 19]])
    with torch.no_grad():
        embedding = model.get_input_embeddings()
        weight = cast(torch.Tensor, embedding.weight)
        native = model.model(input_ids=ids, use_cache=False).last_hidden_state
        supplied = model.model(inputs_embeds=embedding(ids), use_cache=False).last_hidden_state
        raw = model.model(inputs_embeds=weight[ids], use_cache=False).last_hidden_state
        realign = build_native_realign(model)
    assert torch.equal(native, supplied)
    assert not torch.allclose(native, raw)
    assert torch.allclose(realign.target_norm, embedding(torch.arange(64)).norm(dim=-1).mean())


def test_native_fleet_order() -> None:
    """Resolve the published eleven arms, plus the full control the twelve-arm config adds."""
    from pathlib import Path

    from rcc.benchmarks.fanoutqa import FANOUTQA_NATURAL_DEV50
    from rcc.run.config import load_run_config
    from rcc.run.gemma.split_adapter import GemmaSplitPlacementAdapter
    from rcc.run.plan import ResolutionContext, resolve_plan

    expected = (
        "latent_query_support_r8",
        "text_primary",
        "issue_only",
        "text_medium",
        "text_small",
        *(f"latent_query_support_r{ratio}" for ratio in (2, 4, 16, 32, 64, 128)),
    )
    root = Path(__file__).resolve().parents[1]
    config = load_run_config(root / "configs/gemma-fanoutqa-natural-dev50.toml")
    plan = resolve_plan(config, ResolutionContext(git_commit="a" * 40))
    # The natural config runs the twelve-arm profile: the published eleven in
    # their order, then the full-memory control.
    assert tuple(arm.arm_id for arm in plan.arms) == (*expected, "full")
    assert plan.execution_expected_rows.total == 50 * 12 * 3
    adapter = GemmaSplitPlacementAdapter(FANOUTQA_NATURAL_DEV50)
    assert adapter.arm_names == expected
    assert tuple(adapter.resolve(root, panel="production", source_commit="a" * 40)) == expected


# --- The cross-process handoff manifest.

_MEMORY_TOKENS_HANDOFF = 96


_PROMPT_TOKENS_HANDOFF = _FRAME_PREFIX + _MEMORY_TOKENS_HANDOFF + _FRAME_SUFFIX


_PAYLOAD_TOKENS_HANDOFF = payload_length(_PROMPT_TOKENS_HANDOFF)


_LENGTHS = (_PAYLOAD_TOKENS_HANDOFF,) * WORKERS_PER_ITEM


# A restricted roster keeps the fixture honest and cheap: one baseline and one
# arm that actually runs the ranked cut.
_ARMS = ("floor", "r2")


_PUBLICATION_ID = "gemma-split-handoff-fixture"


# The shared table is checked at fixture width: what it covers is the
# publication, the linking, and the digests, none of which is width-bound.
_TABLE_SHAPE = (16, _HIDDEN)


_CHECKPOINT = "fixture/gemma-split"


_REVISION = "0" * 40


_RUNTIME = "f" * 16


def _publish(
    run_root: Path,
    table: torch.Tensor,
    *,
    publication_id: str = _PUBLICATION_ID,
) -> dict[str, str]:
    return publish_shared_embedding_once(
        run_root,
        table,
        runtime_fingerprint=_RUNTIME,
        publication_id=publication_id,
        checkpoint=_CHECKPOINT,
        revision=_REVISION,
        expected_shape=_TABLE_SHAPE,
    )


@dataclass(frozen=True)
class _Capture:
    """One fake three-worker capture on disk, plus its in-memory selection."""

    root: Path
    manifest: Path
    rolled: tuple[Path, ...]
    selection_path: Path
    weight: Path
    selection: SelectionArtifact
    cargo: tuple[torch.Tensor, ...]


def _write_embedding(root: Path) -> Path:
    weight = root / "shared" / "input_embedding_weight.bin"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"shared table stand-in")
    embedding_marker_path(weight).write_text(
        json.dumps({"schema": "fanoutqa-shared-embedding-v1"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    embedding_validation_path(weight, _PUBLICATION_ID).write_text(
        json.dumps({"publication_id": _PUBLICATION_ID}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return weight


def _bank_selection(root: Path) -> tuple[Path, SelectionArtifact]:
    item: dict[str, Any] = {
        "qid": _QID,
        "prompt_ids": tuple(
            torch.arange(_PROMPT_TOKENS_HANDOFF) for _worker in range(WORKERS_PER_ITEM)
        ),
    }
    scores = tuple(
        {"support": torch.arange(_PAYLOAD_TOKENS_HANDOFF, dtype=torch.float32) * (worker + 1)}
        for worker in range(WORKERS_PER_ITEM)
    )
    path = root / "selections" / f"{_QID}.json"
    artifact = load_or_bank_selection(
        item,
        scores,
        path,
        seat_keys_by_ratio={},
        ratio_cuts={},
        arms=_ARMS,
    )
    return path, artifact


def _capture(root: Path, *, write: bool = True) -> _Capture:
    cargo = tuple(_rolled_tensor(worker) for worker in range(WORKERS_PER_ITEM))
    rolled: list[Path] = []
    for worker, tensor in enumerate(cargo):
        path = root / "rolled" / latent_roll_filename(_QID, worker)
        _write_rolled(path, worker, tensor)
        rolled.append(path)
    selection_path, selection = _bank_selection(root)
    weight = _write_embedding(root)
    manifest = handoff.handoff_manifest_path(root, _QID)
    capture = _Capture(
        root=root,
        manifest=manifest,
        rolled=tuple(rolled),
        selection_path=selection_path,
        weight=weight,
        selection=selection,
        cargo=cargo,
    )
    if write:
        _write_manifest(capture)
    return capture


def _write_manifest(capture: _Capture) -> Path:
    return handoff.write_handoff_manifest(
        capture.manifest,
        qid=_QID,
        rolled_paths=capture.rolled,
        selection_path=capture.selection_path,
        embedding_weight=capture.weight,
        publication_id=_PUBLICATION_ID,
        hidden=_HIDDEN,
        frame_prefix_length=_FRAME_PREFIX,
        frame_suffix_length=_FRAME_SUFFIX,
        lengths=_LENGTHS,
        embedding_digests=_EMBEDDING_DIGESTS,
        global_layers=_GLOBAL_LAYERS,
        audition_replays_by_ratio=_REPLAYS,
    )


def _producer_artifact(capture: _Capture) -> CaptureArtifact:
    """The artifact the resident producer banks for this same capture."""
    return CaptureArtifact(
        qid=_QID,
        keeps_by_arm=capture.selection.keeps_by_arm,
        layouts_by_arm=capture.selection.layouts_by_arm,
        rolled_by_worker=capture.cargo,
        selection_s_by_arm=capture.selection.selection_s_by_arm,
        audition_s_by_ratio=dict.fromkeys(_REPLAYS, 0.5),
        audition_replays_by_ratio=dict(_REPLAYS),
        reports=(),
        report_failed=False,
        report_failure=None,
        capture_s=1.25,
        capture_s_by_worker=(0.5,) * WORKERS_PER_ITEM,
        report_generation_s=0.0,
        embedding_digests=dict(_EMBEDDING_DIGESTS),
        global_layers=_GLOBAL_LAYERS,
        reloaded_captures=WORKERS_PER_ITEM,
        reloaded_selections=1,
    )


def _manifest_body(capture: _Capture) -> dict[str, Any]:
    body: Any = json.loads(capture.manifest.read_text(encoding="utf-8"))
    return dict(body)


def _rewrite(capture: _Capture, body: dict[str, Any]) -> None:
    capture.manifest.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")


def test_handoff_manifest_round_trips_every_receiver_banked_field(tmp_path: Path) -> None:
    capture = _capture(tmp_path)
    source = _producer_artifact(capture)

    # No out-of-band geometry: the receiver holds the manifest and nothing else.
    artifact = handoff.load_capture_artifact(capture.manifest, qid=_QID)

    assert artifact.qid == _QID
    assert len(artifact.rolled_by_worker) == WORKERS_PER_ITEM
    for worker, expected in enumerate(capture.cargo):
        loaded = artifact.rolled_by_worker[worker]
        assert loaded.dtype == torch.bfloat16
        assert loaded.shape == (LATENT_STEPS, _HIDDEN)
        assert torch.equal(loaded, expected)
    # Every capture field results.py banks into a row, except the producer
    # clocks it charges the producer for, must survive the process boundary.
    assert artifact.keeps_by_arm == source.keeps_by_arm
    assert artifact.layouts_by_arm == source.layouts_by_arm
    assert artifact.selection_s_by_arm == source.selection_s_by_arm
    assert artifact.embedding_digests == source.embedding_digests
    assert artifact.global_layers == source.global_layers
    assert artifact.audition_replays_by_ratio == source.audition_replays_by_ratio
    assert artifact.reloaded_captures == source.reloaded_captures
    assert artifact.reloaded_selections == source.reloaded_selections
    # The audition ledger keeps its ratio roster, so a hybrid arm resolves its
    # own audition instead of refusing; only the measured seconds go neutral.
    assert artifact.audition_s_by_ratio == dict.fromkeys(_REPLAYS, 0.0)
    assert artifact.capture_s == 0.0
    assert artifact.capture_s_by_worker == (0.0,) * WORKERS_PER_ITEM
    assert artifact.report_generation_s == 0.0

    # The producer clocks the receiver cannot measure travel beside the
    # manifest, bound to its fingerprint, and neither may be invented.
    fingerprint = str(_manifest_body(capture)["fingerprint"])
    handoff.write_handoff_clocks(
        capture.manifest,
        qid=_QID,
        fingerprint=fingerprint,
        spill_save_s=0.25,
        producer_s=1.5,
    )
    clocks = handoff.read_handoff_clocks(capture.manifest, qid=_QID, fingerprint=fingerprint)
    assert clocks == HandoffClocks(spill_save_s=0.25, producer_s=1.5)
    with pytest.raises(handoff.GemmaHandoffError, match="producer clocks must be nonnegative"):
        handoff.write_handoff_clocks(
            capture.manifest,
            qid=_QID,
            fingerprint=fingerprint,
            spill_save_s=0.25,
            producer_s=-1.0,
        )
    body = json.loads(clock_record_path(capture.manifest).read_text(encoding="utf-8"))
    del body["producer_s"]
    clock_record_path(capture.manifest).write_text(
        json.dumps(body, sort_keys=True) + "\n", encoding="utf-8"
    )
    malformed = "clock record entry producer_s is malformed"
    with pytest.raises(handoff.GemmaHandoffError, match=malformed):
        handoff.read_handoff_clocks(capture.manifest, qid=_QID, fingerprint=fingerprint)


def test_handoff_manifest_names_relocates_and_fingerprints_its_bundle(tmp_path: Path) -> None:
    capture = _capture(tmp_path)
    from rcc.models.gemma.capture_support import load_latent_roll

    old_roll = torch.load(capture.rolled[0], weights_only=True)
    old_roll["schema"] = "not-the-registered-schema"
    old_path = tmp_path / "original-roll.pt"
    torch.save(old_roll, old_path)
    with pytest.raises(RuntimeError, match="artifact identity differs"):
        load_latent_roll(
            old_path,
            qid=_QID,
            worker=0,
            hidden=_HIDDEN,
            frame_prefix_length=old_roll["frame_prefix_length"],
            frame_suffix_length=old_roll["frame_suffix_length"],
        )

    body = _manifest_body(capture)
    files = dict(body["files"])
    assert body["schema"] == handoff.HANDOFF_SCHEMA
    assert body["qid"] == _QID
    assert set(files) == {
        "rolled_w0",
        "rolled_w1",
        "rolled_w2",
        handoff.SELECTION_KEY,
        handoff.EMBEDDING_MARKER_KEY,
        handoff.EMBEDDING_VALIDATION_KEY,
    }
    for name, entry in files.items():
        stored = str(entry["path"])
        assert not Path(stored).is_absolute(), name
        assert ".." not in Path(stored).parts, name
        path = capture.manifest.parent / stored
        assert path.is_file(), name
        assert entry["sha256"] == sha256_file(path)
    assert files[handoff.SELECTION_KEY]["path"] == f"selections/{_QID}.json"
    assert files["rolled_w1"]["path"] == f"rolled/{latent_roll_filename(_QID, 1)}"
    meta = dict(body["meta"])
    assert meta["latent_roll_schema"] == LATENT_ROLL_SCHEMA
    assert meta["selection_schema"] == SELECTION_SCHEMA
    assert meta["selection_arms"] == list(_ARMS)
    assert meta["latent_steps"] == LATENT_STEPS
    assert meta["realign_enabled"] == LATENT_REALIGN_ENABLED
    assert meta["workers"] == WORKERS_PER_ITEM
    assert meta["hidden"] == _HIDDEN
    assert meta["lengths"] == list(_LENGTHS)
    assert meta["embedding_digests"] == _EMBEDDING_DIGESTS
    assert meta["global_layers"] == list(_GLOBAL_LAYERS)
    assert meta["audition_replays_by_ratio"] == {}
    assert meta["embedding"]["publication_id"] == _PUBLICATION_ID
    assert meta["embedding"]["weight_path"] == "shared/input_embedding_weight.bin"

    whole = _capture(tmp_path / "whole", write=False)
    stray = tmp_path / "stray" / latent_roll_filename(_QID, 0)
    _write_rolled(stray, 0, whole.cargo[0])
    strayed = _Capture(
        root=whole.root,
        manifest=whole.manifest,
        rolled=(stray, *whole.rolled[1:]),
        selection_path=whole.selection_path,
        weight=whole.weight,
        selection=whole.selection,
        cargo=whole.cargo,
    )
    # A file outside the bundle cannot be named relative to the manifest, so
    # the writer refuses it rather than recording a path only it can resolve.
    with pytest.raises(
        handoff.GemmaHandoffError,
        match=r"rolled_w0: handoff file .* is outside the manifest root",
    ):
        _write_manifest(strayed)

    embedding_validation_path(whole.weight, _PUBLICATION_ID).unlink()
    with pytest.raises(
        handoff.GemmaHandoffError,
        match="q1/embedding_validation: handoff file is missing",
    ):
        _write_manifest(whole)

    # Every named file is recorded relative to the manifest, so the whole
    # bundle moves to the receiving root and still reads.
    capture = _capture(tmp_path / "produced")
    before = handoff.load_capture_artifact(capture.manifest, qid=_QID)

    moved = tmp_path / "received" / "bundle"
    moved.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(capture.root), str(moved))
    assert not capture.manifest.exists()

    artifact = handoff.load_capture_artifact(handoff.handoff_manifest_path(moved, _QID), qid=_QID)

    assert artifact.keeps_by_arm == before.keeps_by_arm
    assert artifact.layouts_by_arm == before.layouts_by_arm
    assert artifact.embedding_digests == before.embedding_digests
    for worker, expected in enumerate(capture.cargo):
        assert torch.equal(artifact.rolled_by_worker[worker], expected)

    # Writing the same capture twice writes the same bytes, and the
    # fingerprint the reader re-derives is the one the writer recorded.
    capture = _capture(tmp_path / "first")
    first = capture.manifest.read_bytes()

    _write_manifest(capture)
    assert capture.manifest.read_bytes() == first

    body = _manifest_body(capture)
    assert body["fingerprint"] == handoff.manifest_fingerprint(body)
    verified = handoff.read_handoff_manifest(capture.manifest, qid=_QID)
    assert verified["fingerprint"] == body["fingerprint"]
    # The same capture bytes under another root keep every content digest and
    # every recorded path. The banked selection carries its own measured
    # clocks, so only its digest moves.
    other = _capture(tmp_path / "second")
    other_body = _manifest_body(other)
    content = {
        name: entry["sha256"]
        for name, entry in body["files"].items()
        if name != handoff.SELECTION_KEY
    }
    assert {
        name: entry["sha256"]
        for name, entry in other_body["files"].items()
        if name != handoff.SELECTION_KEY
    } == content
    assert content.keys() == handoff.file_roster() - {handoff.SELECTION_KEY}
    assert {name: entry["path"] for name, entry in other_body["files"].items()} == {
        name: entry["path"] for name, entry in body["files"].items()
    }


def test_handoff_loader_refuses_every_corrupted_foreign_or_tampered_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One flipped byte anywhere the manifest names, in the cargo or in the
    # selection, is a digest mismatch and never a silent read.
    for index, name in enumerate(("rolled_w1", "selection")):
        capture = _capture(tmp_path / f"flipped{index}")
        target = capture.manifest.parent / str(_manifest_body(capture)["files"][name]["path"])
        raw = bytearray(target.read_bytes())
        raw[-1] ^= 0x01
        target.write_bytes(bytes(raw))
        with pytest.raises(
            handoff.GemmaHandoffError,
            match=f"{name}: handoff file digest differs",
        ):
            handoff.load_capture_artifact(capture.manifest, qid=_QID, lengths=_LENGTHS)

    intact = _capture(tmp_path / "intact")
    # Out-of-band geometry is a cross-check, so a disagreement refuses.
    with pytest.raises(
        handoff.GemmaHandoffError,
        match=r"handoff payload lengths .* differ from the manifest",
    ):
        handoff.load_capture_artifact(
            intact.manifest,
            qid=_QID,
            lengths=(_PAYLOAD_TOKENS_HANDOFF + 1,) * WORKERS_PER_ITEM,
        )
    # And a manifest is one item's: another qid cannot read it at all.
    with pytest.raises(handoff.GemmaHandoffError, match="handoff manifest identity differs"):
        handoff.load_capture_artifact(intact.manifest, qid="q2", lengths=_LENGTHS)

    capture = _capture(tmp_path / "short_cargo", write=False)
    # A short cargo tensor passes its digest and still fails its geometry.
    _write_rolled(capture.rolled[2], 2, _rolled_tensor(2, rows=LATENT_STEPS - 1))
    _write_manifest(capture)
    with pytest.raises(
        handoff.GemmaHandoffError,
        match="q1/w2: latent roll artifact tensor is malformed",
    ):
        handoff.load_capture_artifact(capture.manifest, qid=_QID, lengths=_LENGTHS)

    # An absolute recorded path is a producer-local path a receiver cannot
    # trust, whatever it happens to resolve to on this filesystem.
    escaped = _capture(tmp_path / "escaped")
    body = _manifest_body(escaped)
    body["files"]["rolled_w0"]["path"] = str(escaped.rolled[0])
    body["fingerprint"] = handoff.manifest_fingerprint(body)
    _rewrite(escaped, body)
    with pytest.raises(
        handoff.GemmaHandoffError,
        match="rolled_w0: handoff file path escapes the manifest root",
    ):
        handoff.load_capture_artifact(escaped.manifest, qid=_QID)

    # A manifest that dropped the carried geometry is not self-sufficient.
    blank = _capture(tmp_path / "blank")
    body = _manifest_body(blank)
    del body["meta"]["lengths"]
    body["fingerprint"] = handoff.manifest_fingerprint(body)
    _rewrite(blank, body)
    with pytest.raises(handoff.GemmaHandoffError, match="handoff geometry lengths is malformed"):
        handoff.load_capture_artifact(blank.manifest, qid=_QID)

    short = _capture(tmp_path / "short")
    body = _manifest_body(short)
    del body["files"]["rolled_w2"]
    body["fingerprint"] = handoff.manifest_fingerprint(body)
    _rewrite(short, body)
    with pytest.raises(handoff.GemmaHandoffError, match="handoff manifest file roster differs"):
        handoff.load_capture_artifact(short.manifest, qid=_QID, lengths=_LENGTHS)

    stale = _capture(tmp_path / "stale")
    body = _manifest_body(stale)
    body["meta"]["hidden"] = _HIDDEN * 2
    _rewrite(stale, body)
    with pytest.raises(handoff.GemmaHandoffError, match="handoff manifest fingerprint differs"):
        handoff.load_capture_artifact(stale.manifest, qid=_QID, lengths=_LENGTHS)

    # A receiver that loaded another table than the one the bundle names is
    # refused rather than banking two embedding digests nobody compared.
    monkeypatch.undo()
    named = _capture(tmp_path / "named")
    with pytest.raises(handoff.GemmaHandoffError, match="this receiver loaded"):
        handoff.load_capture_artifact(
            named.manifest,
            qid=_QID,
            shared_embedding_digests={"marker_sha256": "0" * 64},
        )


def test_shared_embedding_publishes_once_and_links_into_every_arm_root(tmp_path: Path) -> None:
    # The split fleet has arms with no producer at all, so the table is
    # published above the arms and linked down into each one.
    run_root = tmp_path / "run"
    table = embedding_table(_TABLE_SHAPE)
    published = _publish(run_root, table)
    weight = shared_embedding_path(run_root)
    assert weight.is_file()
    assert embedding_marker_path(weight).is_file()
    assert embedding_validation_path(weight, _PUBLICATION_ID).is_file()

    # Publishing again is a no-op that adopts the published bytes, and a
    # second attempt over the same table writes its own clock record.
    digest = sha256_file(weight)
    assert _publish(run_root, table) == published
    assert sha256_file(weight) == digest
    assert _publish(run_root, table, publication_id="attempt-2") == published
    assert embedding_validation_path(weight, "attempt-2").is_file()
    # Rows already bank the published digest, so other bytes never replace it.
    with pytest.raises(RuntimeError, match="already published as other bytes"):
        _publish(run_root, table + 1)

    arm_root = tmp_path / "arms" / "floor" / "handoff"
    linked = hardlink_shared_embedding(run_root, arm_root, publication_id=_PUBLICATION_ID)
    assert linked == shared_embedding_path(arm_root)
    # One copy of the weights on this filesystem, and byte-identical files.
    assert linked.stat().st_ino == weight.stat().st_ino
    for source, target in (
        (weight, linked),
        (embedding_marker_path(weight), embedding_marker_path(linked)),
        (
            embedding_validation_path(weight, _PUBLICATION_ID),
            embedding_validation_path(linked, _PUBLICATION_ID),
        ),
    ):
        assert sha256_file(target) == sha256_file(source)
    # A receiver validates the arm copy exactly as it validates the run one.
    value, digests = load_validated_shared_embedding(
        linked,
        checkpoint=_CHECKPOINT,
        revision=_REVISION,
        runtime_fingerprint=_RUNTIME,
        publication_id=_PUBLICATION_ID,
        expected_shape=_TABLE_SHAPE,
    )
    assert digests == published and torch.equal(value, table)
    assert hardlink_shared_embedding(run_root, arm_root, publication_id=_PUBLICATION_ID) == linked

    # A divergent file already in an arm root refuses; it is never replaced.
    stray = tmp_path / "arms" / "r2" / "handoff"
    shared_embedding_path(stray).parent.mkdir(parents=True)
    shared_embedding_path(stray).write_bytes(b"not the published table")
    with pytest.raises(RuntimeError, match="differs from the run"):
        hardlink_shared_embedding(run_root, stray, publication_id=_PUBLICATION_ID)


# --- The vLLM 0.26 capture bridge.


def tiny_gemma4_text_config(*, sliding_window: int = 4) -> Any:
    """The tiny Gemma4Unified text config the capture-bridge fixtures share."""
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedTextConfig,
    )

    return Gemma4UnifiedTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=2,
        # the real checkpoint mixes GQA factors per layer type, so the fixture
        # rehearses that geometry instead of a uniform simplification
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        head_dim=8,
        global_head_dim=8,
        max_position_embeddings=4096,
        sliding_window=sliding_window,
        layer_types=["sliding_attention"] * 5
        + ["full_attention"]
        + ["sliding_attention"] * 5
        + ["full_attention"],
        use_bidirectional_attention="vision",
        num_kv_shared_layers=0,
    )


def tiny_gemma4(seed: int, *, sliding_window: int = 4) -> Any:
    """A seeded tiny text-only Gemma4Unified, the class the capture checks read."""
    from transformers import Gemma4UnifiedForCausalLM

    config = tiny_gemma4_text_config(sliding_window=sliding_window)
    config._attn_implementation = "eager"
    random.seed(seed)
    torch.manual_seed(seed)
    return Gemma4UnifiedForCausalLM(config).eval()


def _fused_engine_tensors(model: Any) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    config = model.config
    named: dict[str, torch.Tensor] = {}
    for layer, layer_type in enumerate(config.layer_types):
        base = f"model.layers.{layer}"
        query = state[f"{base}.self_attn.q_proj.weight"]
        key = state[f"{base}.self_attn.k_proj.weight"]
        value = key if layer_type == "full_attention" else state[f"{base}.self_attn.v_proj.weight"]
        named[f"{base}.self_attn.qkv_proj.weight"] = torch.cat((query, key, value))
        named[f"{base}.mlp.gate_up_proj.weight"] = torch.cat(
            (state[f"{base}.mlp.gate_proj.weight"], state[f"{base}.mlp.up_proj.weight"])
        )
        for suffix in (
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "layer_scalar",
        ):
            named[f"{base}.{suffix}"] = state[f"{base}.{suffix}"]
    named["model.embed_tokens.weight"] = state["model.embed_tokens.weight"]
    named["model.norm.weight"] = state["model.norm.weight"]
    return named


@pytest.mark.live
def test_zero_copy_gemma_view_is_bit_exact_on_the_tiny_model() -> None:
    reference = tiny_gemma4(17).eval()
    resident = _fused_engine_tensors(reference)
    view = gemma4_hf_view(resident, reference.config)

    resident_storages = {tensor.untyped_storage().data_ptr() for tensor in resident.values()}
    assert all(
        parameter.untyped_storage().data_ptr() in resident_storages
        for parameter in view.parameters()
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = reference(input_ids).logits
        actual = view(input_ids).logits
    assert torch.equal(actual, expected)


def _hybrid_fixture(*, device: str = "cpu") -> tuple[Any, tuple[Any, ...], Any]:
    config = tiny_gemma4_text_config(sliding_window=4)
    sliding_names = [
        f"model.layers.{index}.self_attn.attn"
        for index, layer_type in enumerate(config.layer_types)
        if layer_type == "sliding_attention"
    ]
    full_names = [
        f"model.layers.{index}.self_attn.attn"
        for index, layer_type in enumerate(config.layer_types)
        if layer_type == "full_attention"
    ]
    sliding = SimpleNamespace(
        block_size=2,
        num_kv_heads=2,
        head_size=8,
        sliding_window=4,
        dtype=torch.float32,
    )
    full = SimpleNamespace(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        sliding_window=None,
        dtype=torch.float32,
    )
    groups = (
        SimpleNamespace(layer_names=sliding_names, kv_cache_spec=sliding),
        SimpleNamespace(layer_names=full_names, kv_cache_spec=full),
    )
    pages: dict[str, torch.Tensor] = {}
    for names, heads, block_size, head_dim in (
        (sliding_names, 2, 2, 8),
        (full_names, 1, 4, 8),
    ):
        for name in names:
            tensor = torch.zeros(10, heads, block_size, 2 * head_dim, device=device)
            if device == "meta":
                pages[name] = tensor
                continue
            for block_id in range(1, 10):
                for offset in range(block_size):
                    tensor[block_id, :, offset].fill_((block_id - 1) * block_size + offset)
            pages[name] = tensor
    layers = read_layer_pages(pages, SimpleNamespace(kv_cache_groups=groups))
    spec = SimpleNamespace(
        num_prompt_tokens=7,
        block_ids=((1, 2, 3, 4), (1, 2)),
    )
    return config, layers, spec


@pytest.mark.live
def test_heterogeneous_pages_rebuild_the_rolling_logical_length() -> None:
    config, layers, spec = _hybrid_fixture()
    artifact = extract_hybrid_kv(layers, spec)

    assert artifact.layers[0].start_position == 4
    assert artifact.layers[0].keys[0, 0, :, 0].tolist() == [4.0, 5.0, 6.0]
    assert artifact.layers[5].keys[0, 0, :, 0].tolist() == list(map(float, range(7)))
    cache = rolling_cache_from_artifact(
        artifact,
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert cache.get_seq_length() == 7
    assert cache.layers[0].cumulative_length == 7
    assert cache.layers[0].keys.shape[-2] == 3
    assert cache.layers[5].keys.shape[-2] == 7
    # Same device and dtype: the cache adopts the artifact tensors, no copy.
    assert cache.layers[5].keys is artifact.layers[5].keys
    assert cache.layers[5].values is artifact.layers[5].values
    _, layers, spec = _hybrid_fixture(device="meta")
    assert extract_hybrid_kv(layers, spec).layers[0].keys.device.type == "meta"


@pytest.mark.live
def test_extraction_and_rebuild_fail_closed_on_invalid_cache_data() -> None:
    config, layers, spec = _hybrid_fixture()
    bad_spec = SimpleNamespace(
        num_prompt_tokens=spec.num_prompt_tokens,
        block_ids=((1, 2, 0, 4), (1, 2)),
    )
    with pytest.raises(RuntimeError, match="null or duplicate"):
        extract_hybrid_kv(layers, cast(Any, bad_spec))

    artifact = extract_hybrid_kv(layers, spec)
    bad_keys = artifact.layers[0].keys.clone()
    bad_keys[0, 0, 0, 0] = torch.nan
    bad_layer = replace(artifact.layers[0], keys=bad_keys)
    bad_artifact = replace(artifact, layers=(bad_layer, *artifact.layers[1:]))
    with pytest.raises(RuntimeError, match="non-finite"):
        rolling_cache_from_artifact(
            bad_artifact,
            config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


class _SuffixingEngine:
    def __init__(self, *, fail_step: bool = False) -> None:
        self.fail_step = fail_step
        self.logical_id = ""
        self.internal_id = ""
        self.live = False
        self.aborted: list[tuple[list[str], bool]] = []

    def add_request(self, request_id: str, _prompt: Any, _sampling: Any) -> str:
        self.logical_id = request_id
        self.internal_id = f"{request_id}-deadbeef"
        self.live = True
        return self.internal_id

    def has_unfinished_requests(self) -> bool:
        return self.live

    def step(self) -> list[Any]:
        if self.fail_step:
            raise RuntimeError("step failed")
        self.live = False
        return [SimpleNamespace(request_id=self.logical_id, finished=True)]

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        self.aborted.append((request_ids, internal))
        self.live = False


def _suffixing_bridge(engine: _SuffixingEngine) -> GemmaEngineCaptureBridge:
    parameter = torch.nn.Parameter(torch.zeros(1))
    model = SimpleNamespace(
        config=SimpleNamespace(),
        parameters=lambda: iter((parameter,)),
    )
    return GemmaEngineCaptureBridge(
        SimpleNamespace(llm_engine=engine),
        model,
        tokenizer=None,
        request_factory=lambda _request_id, token_ids: (token_ids, None),
    )


@pytest.mark.live
def test_extraction_tracks_vllms_suffixed_internal_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _SuffixingEngine()
    bridge = _suffixing_bridge(engine)
    requested: list[str] = []
    taken: list[str] = []
    discarded: list[str] = []
    artifact = object()
    rebuilt = object()
    monkeypatch.setattr(bridge_module, "request_gemma_extraction", requested.append)
    monkeypatch.setattr(
        bridge_module,
        "take_gemma_extraction",
        lambda request_id: taken.append(request_id) or artifact,
    )
    monkeypatch.setattr(bridge_module, "discard_gemma_extraction", discarded.append)
    monkeypatch.setattr(
        bridge_module,
        "rolling_cache_from_artifact",
        lambda value, *_args, **_kwargs: rebuilt if value is artifact else None,
    )

    assert bridge.extract([1, 2, 3]) is rebuilt
    assert requested == [engine.internal_id]
    assert taken == [engine.internal_id]
    assert discarded == [engine.internal_id]
    assert engine.internal_id == f"{engine.logical_id}-deadbeef"
    assert engine.aborted == []

    # A drive that raises aborts the same internal id and discards its slot.
    failing = _SuffixingEngine(fail_step=True)
    requested.clear()
    discarded.clear()
    with pytest.raises(RuntimeError, match="step failed"):
        _suffixing_bridge(failing)._drive("logical", [1, 2])
    assert requested == ["logical-deadbeef"]
    assert discarded == ["logical-deadbeef"]
    assert failing.aborted == [(["logical-deadbeef"], True)]
