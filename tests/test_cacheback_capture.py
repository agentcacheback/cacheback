"""The CacheBack capture pass: its kernels, its two seams, its fallbacks, and its variants."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tests.conftest import prefill, random_ids, tiny_config

import rcc.latent.cache as latent_cache
import rcc.transforms.select.core.kernels as kernels
from rcc.latent.cache import cache_kv, cache_length
from rcc.latent.rollout import build_realign, latent_rollout_batched
from rcc.transforms.select.baselines import chunkkv, h2o, kvzip
from rcc.transforms.select.core.types import (
    AttentionLayerObserver,
    BatchedPastLike,
    CaptureSpec,
    DecoderBackbone,
    KVPair,
)
from rcc.transforms.select.query_support import (
    capture_bank,
    capture_batch,
    capture_layers,
    capture_single,
)
from rcc.transforms.select.query_support.capture_bank_format import (
    BANK_SCOPE_ALL_LAYER,
    BANK_SCOPE_GLOBAL_ONLY,
)
from rcc.transforms.select.query_support.fixed_spans import fixed_span_keep
from rcc.transforms.select.query_support.methods import reducers, variants
from rcc.transforms.select.query_support.methods.batch import (
    capture_query_support_batched_from_past,
)
from rcc.transforms.select.query_support.methods.compose import selector_score
from rcc.transforms.select.query_support.methods.single import capture_query_support

# --- The kernels.


def _assert_kv_identity(actual: list[KVPair], expected: list[KVPair]) -> None:
    assert len(actual) == len(expected)
    for (actual_key, actual_value), (expected_key, expected_value) in zip(
        actual, expected, strict=True
    ):
        assert actual_key is expected_key
        assert actual_value is expected_value


def test_cache_kv_covers_layered_to_legacy_and_legacy_tuple_branches() -> None:
    keys0 = torch.zeros(1, 1, 3, 2)
    values0 = torch.ones(1, 1, 3, 2)
    keys1 = torch.full((1, 1, 3, 2), 2.0)
    values1 = torch.full((1, 1, 3, 2), 3.0)
    legacy: list[KVPair] = [(keys0, values0), (keys1, values1)]
    layered = SimpleNamespace(
        layers=[
            SimpleNamespace(keys=keys0, values=values0),
            SimpleNamespace(keys=keys1, values=values1),
        ]
    )
    converted = SimpleNamespace(to_legacy_cache=lambda: legacy)

    assert latent_cache.cache_kv is kernels.cache_kv
    _assert_kv_identity(kernels.cache_kv(None), [])
    _assert_kv_identity(kernels.cache_kv(layered), legacy)
    _assert_kv_identity(kernels.cache_kv(converted), legacy)
    _assert_kv_identity(kernels.cache_kv(tuple(legacy)), legacy)
    assert kernels.cache_length(tuple(legacy)) == 3
    assert kernels.cache_length(None) == 0
    assert latent_cache.cache_length(tuple(legacy)) == kernels.cache_length(tuple(legacy))


def test_get_backbone_fallback_and_decoder_protocol() -> None:
    class TinyBackbone:
        def __call__(self, **kwargs: object) -> object:
            return kwargs

    tiny: DecoderBackbone = TinyBackbone()
    with_decoder = SimpleNamespace(get_decoder=lambda: tiny)
    fallback = SimpleNamespace(model=tiny)

    assert kernels.get_backbone(with_decoder) is tiny
    assert kernels.get_backbone(fallback) is tiny
    result = kernels.get_backbone(fallback)(input_ids=torch.tensor([[1]]))
    assert result == {"input_ids": torch.tensor([[1]])}


def test_kv_repeat_and_causal_bias_hold_their_shapes_and_values() -> None:
    values = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    assert kernels.repeat_kv(values, 1) is values
    assert torch.equal(kernels.repeat_kv(values, 2), values.repeat_interleave(2, dim=1))

    got = kernels.causal_bias(2, 3, torch.float32, torch.device("cpu"))
    minimum = torch.finfo(torch.float32).min
    expected = torch.tensor([[[[0.0, 0.0, 0.0, 0.0, minimum], [0.0, 0.0, 0.0, 0.0, 0.0]]]])
    assert torch.equal(got, expected)


def test_locate_question_tensor_branch_and_pool_passthrough() -> None:
    judger_ids = torch.tensor([[4, 8, 9, 8, 9, 2]])
    question = torch.tensor([8, 9])
    assert kernels.locate_question(judger_ids, question) == (1, 3, True)
    assert kernels.locate_question(judger_ids, torch.tensor([7])) == (5, 6, False)

    scores = torch.tensor([0.1, 0.8, 0.2, 0.4])
    for kernel in (0, 1):
        assert kernels.pool_scores(scores, kernel) is scores
    # max pooling with kernel 3 pulls each token up to its best neighbour
    assert torch.equal(kernels.pool_scores(scores, 3), torch.tensor([0.8, 0.8, 0.8, 0.4]))


class _SetterModel:
    def __init__(self, *, raises: bool = False) -> None:
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.calls: list[str] = []
        self.raises = raises

    def set_attn_implementation(self, implementation: str) -> None:
        self.calls.append(implementation)
        if self.raises:
            raise RuntimeError("setter unavailable")
        self.config._attn_implementation = implementation


def test_attention_implementation_exception_and_config_fallbacks() -> None:
    setter_error = _SetterModel(raises=True)
    kernels.set_attention_implementation(setter_error, "eager")
    assert setter_error.calls == ["eager"]
    assert setter_error.config._attn_implementation == "eager"

    config_only = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    kernels.set_attention_implementation(config_only, "eager")
    assert config_only.config._attn_implementation == "eager"

    setter = _SetterModel()
    kernels.set_attention_implementation(setter, "flash_attention_2")
    assert setter.config._attn_implementation == "flash_attention_2"

    swapped = _SetterModel()
    assert kernels.use_eager_attention(swapped) == "sdpa"
    assert swapped.calls == ["eager"]
    assert swapped.config._attn_implementation == "eager"


def test_lower_right_causal_bias_available_and_unavailable_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = torch.tensor([[1.0]])

    def fake_lower_right(query_length: int, key_length: int) -> torch.Tensor:
        assert (query_length, key_length) == (2, 4)
        return expected

    monkeypatch.setattr(
        kernels,
        "import_module",
        lambda name: SimpleNamespace(causal_lower_right=fake_lower_right),
    )
    actual = kernels.lower_right_causal_bias(2, 4)
    assert actual is not None
    assert torch.equal(actual, expected)

    def missing(_name: str) -> object:
        raise ImportError("optional Torch feature")

    monkeypatch.setattr(kernels, "import_module", missing)
    assert kernels.lower_right_causal_bias(2, 4) is None


# --- The batched capture seam over already-padded rows.


def _padded(model: Any) -> Any:
    """Build the batched padded fixture from seeds 41 and 71."""
    prompts = [random_ids(length, seed=41 + row) for row, length in enumerate((5, 8, 3))]
    return latent_rollout_batched(
        model,
        prompts,
        latent_steps=2,
        realign=build_realign(model, enabled=False),
    )[0]


def _judgers() -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[int]]]:
    """Build the judger ids, masks, and question spans paired with that fixture."""
    ids = [random_ids(length, seed=71 + row) for row, length in enumerate((7, 5, 9))]
    masks = [torch.ones_like(item) for item in ids]
    questions = [ids[0][0, 2:5].tolist(), ids[1][0, 1:3].tolist(), ids[2][0, 4:8].tolist()]
    return ids, masks, questions


def _snapshot(past: object) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Clone cache tensors for restoration assertions."""
    return [(key.clone(), value.clone()) for key, value in cache_kv(past)]


@dataclass
class _MinimalPaddedStub:
    """A structural stand-in that is not a `rollout.BatchedPast`."""

    past: Any
    mask: torch.Tensor
    live: list[int]


def test_batched_past_like_accepts_a_minimal_structural_stub(tiny_model: Any) -> None:
    padded = _padded(tiny_model)
    ids, masks, questions = _judgers()
    stub = _MinimalPaddedStub(padded.past, padded.mask, list(padded.live))
    assert isinstance(stub, BatchedPastLike)

    results = capture_query_support_batched_from_past(
        tiny_model,
        stub,
        ids,
        masks,
        questions,
        pool_kernel=3,
        energy_pool_kernel=3,
        row_energy_pool_kernel=3,
    )

    assert len(results) == len(ids)
    assert all(result.energy is not None and result.row_energy is not None for result in results)


def test_snap_only_capture_makes_zero_reducer_calls(
    tiny_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    padded = _padded(tiny_model)
    ids, masks, questions = _judgers()
    reducer_calls = 0

    def count_reducer(*_args: object, **_kwargs: object) -> None:
        nonlocal reducer_calls
        reducer_calls += 1

    monkeypatch.setattr(reducers, "accumulate_capture_energies", count_reducer)
    results = capture_query_support_batched_from_past(
        tiny_model,
        padded,
        ids,
        masks,
        questions,
    )

    assert reducer_calls == 0
    assert all(result.energy is None and result.row_energy is None for result in results)


def test_batched_capture_restores_cache_and_attention_on_hook_failure(
    tiny_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    padded = _padded(tiny_model)
    ids, masks, questions = _judgers()
    before = _snapshot(padded.past)
    mask_before = padded.mask.clone()
    live_before = list(padded.live)
    implementation = str(tiny_model.config._attn_implementation)

    def fail_inside_hook(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("forced batched hooked-forward failure")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fail_inside_hook)
    with pytest.raises(RuntimeError, match="forced batched hooked-forward failure"):
        capture_query_support_batched_from_past(
            tiny_model,
            padded,
            ids,
            masks,
            questions,
        )

    assert str(tiny_model.config._attn_implementation) == implementation
    assert capture_batch.current_context() is None
    assert cache_length(padded.past) == int(mask_before.shape[1])
    assert torch.equal(padded.mask, mask_before)
    assert padded.live == live_before
    for (before_key, before_value), (after_key, after_value) in zip(
        before,
        cache_kv(padded.past),
        strict=True,
    ):
        assert torch.equal(before_key, after_key)
        assert torch.equal(before_value, after_value)


# --- The single-item capture seam.


def _snapshot_capture(past: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(key.clone(), value.clone()) for key, value in cache_kv(past)]


def _assert_snapshot(past: Any, before: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    assert cache_length(past) == int(before[0][0].shape[2])
    for (before_key, before_value), (after_key, after_value) in zip(
        before, cache_kv(past), strict=True
    ):
        assert torch.equal(before_key, after_key)
        assert torch.equal(before_value, after_value)


def test_capture_registers_attention_and_mask_interfaces_together() -> None:
    """The capture key must be valid for both Transformers dispatch layers."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    capture_single._register_capture()

    assert callable(ALL_ATTENTION_FUNCTIONS[capture_single.CAPTURE_IMPL])
    assert callable(ALL_MASK_ATTENTION_FUNCTIONS[capture_single.CAPTURE_IMPL])


def test_public_capture_returns_typed_outputs_and_leaves_the_cache_intact(
    tiny_model: Any,
) -> None:
    past = prefill(tiny_model)
    judger = random_ids(length=7, seed=411)
    mask = torch.ones_like(judger)
    question = judger[0, 2:5].tolist()
    before = _snapshot_capture(past)

    result = capture_query_support(
        tiny_model,
        past,
        judger,
        mask,
        question,
        pool_kernel=3,
        spec=CaptureSpec(
            energy_pool_kernel=3,
            row_energy_pool_kernel=3,
            support_orders=(2.0,),
        ),
    )

    assert result.found is True
    assert result.snap.shape == (cache_length(past),)
    assert result.snap.dtype == torch.float32
    assert bool(torch.isfinite(result.snap).all())
    assert result.energy is not None and result.row_energy is not None
    assert result.moments is not None
    assert set(result.moments.e) == {2.0}
    _assert_snapshot(past, before)
    assert str(tiny_model.config._attn_implementation) == "eager"


class _ContextObserver(AttentionLayerObserver):
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, int, int]] = []
        self.contexts: list[object] = []

    def observe_layer(
        self,
        per_head_sum: torch.Tensor,
        *,
        query_rows: int,
        kv_groups: int,
    ) -> None:
        context = capture_single.current_context()
        assert context is not None
        self.contexts.append(context)
        self.calls.append((per_head_sum, query_rows, kv_groups))


def test_observer_receives_each_layer_inside_the_live_capture_context(
    tiny_model: Any,
) -> None:
    past = prefill(tiny_model)
    judger = random_ids(length=6, seed=412)
    observer = _ContextObserver()

    result = capture_query_support(
        tiny_model,
        past,
        judger,
        torch.ones_like(judger),
        judger[0, 1:4],
        spec=CaptureSpec(collect_statistics=True),
        observer=observer,
    )

    assert result.statistics is not None
    assert len(observer.calls) == int(tiny_model.config.num_hidden_layers)
    assert all(query_rows == 3 for _values, query_rows, _groups in observer.calls)
    assert all(groups == 2 for _values, _rows, groups in observer.calls)
    assert all(value.ndim == 2 for value, _rows, _groups in observer.calls)
    assert len(observer.contexts) == len(observer.calls)


def test_capture_restores_cache_and_attention_when_hooked_forward_raises(
    tiny_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    past = prefill(tiny_model)
    judger = random_ids(length=6, seed=413)
    before = _snapshot_capture(past)
    implementation = str(tiny_model.config._attn_implementation)
    before_implementations = tuple(
        (id(config), value)
        for config, value in kernels.snapshot_attention_implementations(tiny_model)
    )

    def fail_inside_hook(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("forced hooked-forward failure")

    from transformers.models.qwen3 import modeling_qwen3

    monkeypatch.setattr(modeling_qwen3, "eager_attention_forward", fail_inside_hook)
    with pytest.raises(RuntimeError, match="forced hooked-forward failure"):
        capture_query_support(
            tiny_model,
            past,
            judger,
            torch.ones_like(judger),
            judger[0, 1:4].tolist(),
        )

    assert str(tiny_model.config._attn_implementation) == implementation
    after_implementations = tuple(
        (id(config), value)
        for config, value in kernels.snapshot_attention_implementations(tiny_model)
    )
    assert after_implementations == before_implementations
    _assert_snapshot(past, before)


def test_batched_capture_refuses_layer_aware_attention(
    tiny_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deferred batched path must fail before touching a hybrid model."""
    monkeypatch.setattr(
        tiny_model.config,
        "layer_types",
        ["sliding_attention", "full_attention"],
        raising=False,
    )
    with pytest.raises(RuntimeError, match="deferred for hybrid attention"):
        capture_batch.capture_batched(
            tiny_model,
            cast(BatchedPastLike, object()),
            [],
            [],
            [],
            pool_kernel=1,
        )


def test_capture_rejects_a_refused_attention_swap(
    tiny_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warning-only setter refusal must become a capture error."""
    past = prefill(tiny_model)
    judger = random_ids(length=6, seed=414)
    implementation = str(tiny_model.config._attn_implementation)

    def refuse(_implementation: str) -> None:
        return None

    monkeypatch.setattr(tiny_model, "set_attn_implementation", refuse)
    monkeypatch.setattr(tiny_model.model, "set_attn_implementation", refuse, raising=False)
    with pytest.raises(RuntimeError, match="swap was refused"):
        capture_single.capture_context(
            tiny_model,
            past,
            judger,
            torch.ones_like(judger),
            judger[0, 1:4],
            pool_kernel=1,
            spec=CaptureSpec(),
            observer=None,
            lower_right=False,
        )
    assert str(tiny_model.config._attn_implementation) == implementation


def _tiny_qwen_capture_model(implementation: str) -> Any:
    """Build an independent random Qwen cell for backend output parity."""
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(942)
    return Qwen3ForCausalLM(tiny_config(implementation)).eval()


@pytest.mark.parametrize("implementation", ("eager", "sdpa"))
def test_tiny_qwen_capture_output_parity_per_backend(implementation: str) -> None:
    """All-global Qwen output remains byte-identical for both delegates."""
    model = _tiny_qwen_capture_model(implementation)
    past = prefill(model, length=8)
    judger = random_ids(length=6, seed=415)
    mask = torch.ones_like(judger)
    question = judger[0, 1:4]
    armed_values: list[torch.Tensor] = []
    handle = model.model.layers[0].self_attn.register_forward_hook(
        lambda _module, _inputs, output: armed_values.append(output[0].detach().clone())
    )
    capture_single.capture_context(
        model,
        past,
        judger,
        mask,
        question,
        pool_kernel=1,
        spec=CaptureSpec(),
        observer=None,
        lower_right=False,
    )
    handle.remove()

    vanilla_values: list[torch.Tensor] = []
    handle = model.model.layers[0].self_attn.register_forward_hook(
        lambda _module, _inputs, output: vanilla_values.append(output[0].detach().clone())
    )
    with torch.no_grad():
        model.model(
            input_ids=judger[:, :4],
            past_key_values=copy.deepcopy(past),
            position_ids=torch.arange(8, 12).unsqueeze(0),
            attention_mask=torch.cat([torch.ones(1, 8, dtype=mask.dtype), mask[:, :4]], dim=1),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
    handle.remove()
    assert len(armed_values) == len(vanilla_values) == 1
    assert torch.equal(armed_values[0], vanilla_values[0])


def test_qwen_sdpa_golden_vector_and_keep_are_frozen() -> None:
    """The all-global SDPA score holds to one float32 ulp; its keep identity is exact."""
    model = _tiny_qwen_capture_model("sdpa")
    past = prefill(model)
    judger = random_ids(length=7, seed=411)
    result = capture_query_support(
        model,
        past,
        judger,
        torch.ones_like(judger),
        judger[0, 2:5],
        pool_kernel=3,
        spec=CaptureSpec(),
    )
    expected = torch.tensor(
        [
            0.11942168325185776,
            0.14117491245269775,
            0.1427055448293686,
            0.1427055448293686,
            0.1427055448293686,
            0.13082243502140045,
            0.1382938027381897,
            0.1382938027381897,
            0.1382938027381897,
            0.11418920755386353,
            0.11418920755386353,
            0.11418920755386353,
        ],
        dtype=torch.float32,
    )
    # CPU attention kernels differ across platforms below one float32 ulp
    # (4.5e-8 measured between two torch 2.11 builds); the keep below is the
    # exact pin, and it must not move under that noise.
    assert torch.allclose(result.snap, expected, rtol=0, atol=1e-7)
    assert fixed_span_keep(result.snap, 6, (0, 11), span_size=4) == [0, 4, 5, 6, 7, 11]


def _tiny_gemma3_capture_model(implementation: str, *, all_global: bool = False) -> Any:
    """Build a six-layer random Gemma3 text-decoder fixture on either pin."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    layer_types = (
        ["full_attention"] * 6 if all_global else ["sliding_attention"] * 5 + ["full_attention"]
    )
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=4,
        layer_types=layer_types,
        query_pre_attn_scalar=64,
    )
    config._attn_implementation = implementation
    torch.manual_seed(940)
    return Gemma3ForCausalLM(config).eval()


def _tiny_gemma4_capture_model(implementation: str, *, strip_towers: bool = True) -> Any:
    """Build a six-layer random Gemma4Unified text-decoder fixture."""
    pytest.importorskip("transformers.models.gemma4_unified")
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedAudioConfig,
        Gemma4UnifiedTextConfig,
        Gemma4UnifiedVisionConfig,
    )

    text_config = Gemma4UnifiedTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        max_position_embeddings=128,
        sliding_window=4,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        use_bidirectional_attention="vision",
        num_kv_shared_layers=0,
    )
    config = Gemma4UnifiedConfig(
        text_config=text_config,
        vision_config=Gemma4UnifiedVisionConfig(
            mm_embed_dim=32,
            mm_posemb_size=8,
            output_proj_dims=32,
        ),
        audio_config=Gemma4UnifiedAudioConfig(audio_embed_dim=32),
    )
    config._attn_implementation = implementation
    torch.manual_seed(941)
    model = Gemma4UnifiedForConditionalGeneration(config).eval()
    if strip_towers:
        model.model.embed_vision = None
        model.model.embed_audio = None
    return model


def _hybrid_cache(model: Any, *, track_grad: bool = False) -> Any:
    """Create a cache with five sliding layers and one global layer."""
    from transformers.cache_utils import Cache, DynamicLayer, DynamicSlidingWindowLayer

    layers: list[Any] = [DynamicSlidingWindowLayer(sliding_window=4) for _ in range(5)]
    layers.append(DynamicLayer())
    cache = Cache(layers=layers)
    prefill = torch.arange(10, dtype=torch.long).reshape(1, 10) % 64
    backbone = kernels.get_backbone(model)
    if track_grad:
        backbone(
            input_ids=prefill,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    else:
        with torch.no_grad():
            backbone(
                input_ids=prefill,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
    return cache


def _cache_fingerprint(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor, int | None], ...]:
    """Capture cache bytes and sliding cumulative lengths for mutation checks."""
    return tuple(
        (layer.keys.clone(), layer.values.clone(), getattr(layer, "cumulative_length", None))
        for layer in cache.layers
    )


@pytest.mark.parametrize("implementation", ("eager", "sdpa"))
@pytest.mark.parametrize("family", ("gemma3", "gemma4_unified"))
def test_hybrid_gemma_capture_matches_vanilla_attention_and_leaves_the_cache_intact(
    implementation: str,
    family: str,
) -> None:
    """Exercise mask parity, global-layer scoring, and non-mutating cache capture."""
    model = (
        _tiny_gemma3_capture_model(implementation)
        if family == "gemma3"
        else _tiny_gemma4_capture_model(implementation)
    )
    past = _hybrid_cache(model)
    next_reference = copy.deepcopy(past)
    before = _cache_fingerprint(past)
    judger = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
    judger_mask = torch.ones_like(judger)
    question = judger[0, 1:4]
    observer_values: list[torch.Tensor] = []
    observer_layers: list[int] = []

    class Observer(AttentionLayerObserver):
        def observe_layer(
            self,
            per_head_sum: torch.Tensor,
            *,
            query_rows: int,
            kv_groups: int,
        ) -> None:
            assert query_rows == 3
            assert kv_groups == 2
            active = capture_single.current_context()
            assert active is not None
            observer_layers.append(active.layer_descriptors[-1].layer_idx)
            observer_values.append(per_head_sum.detach().clone())

    armed_values: list[torch.Tensor] = []
    backbone: Any = kernels.get_backbone(model)
    handle = backbone.layers[0].self_attn.register_forward_hook(
        lambda _module, _inputs, output: armed_values.append(output[0].detach().clone())
    )
    result, context = capture_single.capture_context(
        model,
        past,
        judger,
        judger_mask,
        question,
        pool_kernel=3,
        spec=CaptureSpec(collect_statistics=True),
        observer=Observer(),
        lower_right=False,
    )
    handle.remove()

    assert context.callback_hits == 6
    assert context.scoring_policy == "global_layers"
    assert observer_layers == [5]
    assert len(observer_values) == 1
    assert [descriptor.layer_type for descriptor in context.layer_descriptors] == [
        "sliding",
        "sliding",
        "sliding",
        "sliding",
        "sliding",
        "global",
    ]
    assert all(descriptor.logical_memory_length == 10 for descriptor in context.layer_descriptors)
    assert [descriptor.physical_key_length for descriptor in context.layer_descriptors] == [
        7,
        7,
        7,
        7,
        7,
        14,
    ]
    assert [descriptor.absolute_key_offset for descriptor in context.layer_descriptors] == [
        7,
        7,
        7,
        7,
        7,
        0,
    ]
    assert [descriptor.window for descriptor in context.layer_descriptors] == [4, 4, 4, 4, 4, None]
    assert bool(torch.isfinite(observer_values[0]).all())
    assert bool((observer_values[0] > 0).any())
    assert bool(torch.isfinite(result.snap).all())
    assert bool((result.snap > 0).any())
    assert len(armed_values) == 1

    # A consumed cache yields the same statistics bit for bit and comes back with
    # the judger rows appended; the default copy left the caller's cache alone.
    consumed = copy.deepcopy(past)
    taken, _taken_context = capture_single.capture_context(
        model,
        consumed,
        judger,
        judger_mask,
        question,
        pool_kernel=3,
        spec=CaptureSpec(collect_statistics=True),
        observer=None,
        lower_right=False,
        consume_past=True,
    )
    assert torch.equal(taken.snap, result.snap)
    assert kernels.cache_logical_length(consumed) == kernels.cache_logical_length(past) + 4
    memory_length = 10
    base = torch.ones(1, memory_length, dtype=judger_mask.dtype)
    vanilla_values: list[torch.Tensor] = []
    handle = backbone.layers[0].self_attn.register_forward_hook(
        lambda _module, _inputs, output: vanilla_values.append(output[0].detach().clone())
    )
    with torch.no_grad():
        backbone(
            input_ids=judger[:, :4],
            past_key_values=copy.deepcopy(past),
            position_ids=torch.arange(memory_length, memory_length + 4).unsqueeze(0),
            attention_mask=torch.cat([base, judger_mask[:, :4]], dim=1),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
    handle.remove()
    assert len(vanilla_values) == 1
    assert torch.equal(armed_values[0], vanilla_values[0])

    next_ids = torch.tensor([[16]], dtype=torch.long)
    with torch.no_grad():
        before_next = backbone(
            input_ids=next_ids,
            past_key_values=next_reference,
            use_cache=True,
            return_dict=True,
        ).last_hidden_state
        after_next = backbone(
            input_ids=next_ids,
            past_key_values=copy.deepcopy(past),
            use_cache=True,
            return_dict=True,
        ).last_hidden_state
    assert torch.equal(before_next, after_next)
    after = _cache_fingerprint(past)
    assert len(after) == len(before)
    for (after_key, after_value, after_length), (before_key, before_value, before_length) in zip(
        after,
        before,
        strict=True,
    ):
        assert torch.equal(after_key, before_key)
        assert torch.equal(after_value, before_value)
        assert after_length == before_length


# --- Fallbacks and failure paths.


def _gemma3(
    implementation: str,
    *,
    all_global: bool = False,
    global_layers: int = 1,
) -> Any:
    """Build the tiny Gemma3 text decoder used by both Transformers pins."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    if all_global:
        layer_types = ["full_attention"] * 6
    else:
        layer_types = ["sliding_attention"] * (6 - global_layers) + [
            "full_attention"
        ] * global_layers
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        sliding_window=4,
        layer_types=layer_types,
        query_pre_attn_scalar=64,
    )
    config._attn_implementation = implementation
    torch.manual_seed(944)
    return Gemma3ForCausalLM(config).eval()


def _gemma3_conditional_generation(implementation: str, *, strip_tower: bool = True) -> Any:
    """Build a tiny multimodal Gemma3 wrapper on both Transformers pins."""
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig

    text_config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=4,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        query_pre_attn_scalar=64,
    )
    config = Gemma3Config(text_config=text_config)
    config._attn_implementation = implementation
    torch.manual_seed(946)
    model = Gemma3ForConditionalGeneration(config).eval()
    if strip_tower:
        model.model.vision_tower = None
        cast(Any, model.model).multi_modal_projector = None
    return model


def _gemma4(implementation: str, *, strip_towers: bool = True) -> Any:
    """Build the venv-only Gemma4Unified config with optional towers."""
    pytest.importorskip("transformers.models.gemma4_unified")
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedAudioConfig,
        Gemma4UnifiedTextConfig,
        Gemma4UnifiedVisionConfig,
    )

    text_config = Gemma4UnifiedTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        max_position_embeddings=128,
        sliding_window=4,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        use_bidirectional_attention="vision",
        num_kv_shared_layers=0,
    )
    config = Gemma4UnifiedConfig(
        text_config=text_config,
        vision_config=Gemma4UnifiedVisionConfig(
            mm_embed_dim=32,
            mm_posemb_size=8,
            output_proj_dims=32,
        ),
        audio_config=Gemma4UnifiedAudioConfig(audio_embed_dim=32),
    )
    config._attn_implementation = implementation
    torch.manual_seed(945)
    model = Gemma4UnifiedForConditionalGeneration(config).eval()
    if strip_towers:
        model.model.embed_vision = None
        model.model.embed_audio = None
    return model


def _hybrid_cache_from_config(model: Any, *, track_grad: bool = False) -> Any:
    """Prefill a config-shaped hybrid cache, optionally with autograd on."""
    from transformers.cache_utils import Cache, DynamicLayer, DynamicSlidingWindowLayer

    configured = getattr(model.config, "layer_types", None)
    layer_types = tuple(configured) if configured is not None else ("full_attention",) * 6
    layers: list[Any] = [
        DynamicSlidingWindowLayer(sliding_window=4)
        if "sliding" in str(layer_type)
        else DynamicLayer()
        for layer_type in layer_types
    ]
    cache = Cache(layers=layers)
    ids = torch.arange(10, dtype=torch.long).reshape(1, 10) % 64
    backbone = kernels.get_backbone(model)
    if track_grad:
        backbone(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
    else:
        with torch.no_grad():
            backbone(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
    return cache


def _capture_args(
    question_rows: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | list[int]]:
    """Return deterministic query, mask, and question ids for one capture."""
    if question_rows == 3:
        judger = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
    else:
        judger = torch.arange(question_rows, dtype=torch.long).reshape(1, question_rows) % 64
    question: torch.Tensor | list[int] = (
        judger[0, 1:4] if question_rows == 3 else judger[0].tolist()
    )
    return judger, torch.ones_like(judger), question


@pytest.mark.parametrize("implementation", ("eager", "sdpa"))
def test_clone_cache_value_fallback_reconstructs_grad_cache_and_preserves_caller(
    implementation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-leaf fallback preserves cache structure and caller bytes."""
    model = _gemma3(implementation)
    past = _hybrid_cache_from_config(model, track_grad=True)
    assert all(layer.keys.requires_grad and not layer.keys.is_leaf for layer in past.layers)
    before = _cache_fingerprint(past)
    original_deepcopy = copy.deepcopy

    def fail_for_original(value: object, memo: dict[int, object] | None = None) -> object:
        if value is past:
            raise RuntimeError("forced non-leaf cache deepcopy failure")
        return original_deepcopy(value, memo)

    monkeypatch.setattr(capture_single.copy, "deepcopy", fail_for_original)
    clone: Any = capture_single._clone_cache(past)
    assert type(clone) is type(past)
    assert clone.get_seq_length() == past.get_seq_length() == 10
    assert [type(layer) for layer in clone.layers] == [type(layer) for layer in past.layers]
    assert [getattr(layer, "sliding_window", None) for layer in clone.layers] == [4] * 5 + [None]
    assert [getattr(layer, "cumulative_length", None) for layer in clone.layers] == [10] * 5 + [
        None
    ]
    for source, copied in zip(past.layers, clone.layers, strict=True):
        assert copied is not source
        assert torch.equal(copied.keys, source.keys)
        assert torch.equal(copied.values, source.values)

    judger, mask, question = _capture_args()
    _result, context = capture_single.capture_context(
        model,
        past,
        judger,
        mask,
        question,
        pool_kernel=1,
        spec=CaptureSpec(),
        observer=None,
        lower_right=False,
    )
    assert [descriptor.absolute_key_offset for descriptor in context.layer_descriptors] == [
        7,
        7,
        7,
        7,
        7,
        0,
    ]
    after = _cache_fingerprint(past)
    for (after_key, after_value, after_length), (before_key, before_value, before_length) in zip(
        after, before, strict=True
    ):
        assert torch.equal(after_key, before_key)
        assert torch.equal(after_value, before_value)
        assert after_length == before_length


def _gemma3_snap(model: Any) -> tuple[torch.Tensor, str]:
    """Run a capture with a fixed all-global Gemma3 prompt."""
    past = model(input_ids=torch.arange(10).reshape(1, 10) % 64, use_cache=True).past_key_values
    judger, mask, question = _capture_args()
    result, context = capture_single.capture_context(
        model,
        past,
        judger,
        mask,
        question,
        pool_kernel=1,
        spec=CaptureSpec(),
        observer=None,
        lower_right=False,
    )
    return result.snap, context.scoring_policy


@pytest.mark.parametrize("implementation", ("eager", "sdpa"))
def test_all_global_prepared_mask_score_is_bit_equal_to_bias_span(
    implementation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new prepared-mask scorer agrees exactly with the trusted span path."""
    trusted = _gemma3(implementation, all_global=True)
    prepared = _gemma3(implementation, all_global=True)
    trusted_snap, trusted_policy = _gemma3_snap(trusted)
    monkeypatch.setattr(capture_single.kernels, "has_layer_aware_attention", lambda _model: True)
    prepared_snap, prepared_policy = _gemma3_snap(prepared)
    assert trusted_policy == "all_layers"
    assert prepared_policy == "global_layers"
    assert torch.equal(prepared_snap, trusted_snap)


def test_all_sliding_global_policy_refuses_instead_of_returning_zero() -> None:
    """A global-only policy requires a real global layer to score."""
    model = _gemma3("eager")
    model.config.layer_types = ["sliding_attention"] * 6
    past = _hybrid_cache_from_config(model)
    judger, mask, question = _capture_args()
    with pytest.raises(RuntimeError, match="at least one global layer"):
        capture_single.capture_context(
            model,
            past,
            judger,
            mask,
            question,
            pool_kernel=1,
            spec=CaptureSpec(),
            observer=None,
            lower_right=False,
        )


@pytest.mark.parametrize("implementation", ("eager", "sdpa"))
def test_statistics_softmax_runs_only_on_global_layers(
    implementation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global statistics stay live while sliding layers skip discarded work."""
    model = _gemma3(implementation)
    past = _hybrid_cache_from_config(model)
    judger, mask, question = _capture_args()
    calls: list[tuple[int, ...]] = []
    original_softmax = capture_single.torch.softmax

    def record_softmax(
        input_tensor: torch.Tensor,
        dim: int | str | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        calls.append(tuple(int(size) for size in input_tensor.shape))
        return original_softmax(input_tensor, dim=dim, dtype=dtype)

    monkeypatch.setattr(capture_single.torch, "softmax", record_softmax)
    result, context = capture_single.capture_context(
        model,
        past,
        judger,
        mask,
        question,
        pool_kernel=1,
        spec=CaptureSpec(collect_statistics=True),
        observer=None,
        lower_right=False,
    )
    assert len(calls) == 1
    assert context.layer_descriptors[-1].is_global
    assert result.statistics is not None
    assert bool(torch.isfinite(result.statistics.rank_mean).all())


def _tiny_qwen(implementation: str) -> Any:
    """Build the all-global Qwen fixture the replay checks read."""
    from tests.conftest import tiny_config
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(947)
    model = Qwen3ForCausalLM(tiny_config(implementation)).eval()
    model.config.max_position_embeddings = 512
    return model


def _bank_capture(
    model: Any,
    *,
    all_layers: bool,
    collect_layer_bank: bool = True,
    question_rows: int = 3,
) -> tuple[Any, Any]:
    """Capture one score and its exact per-layer bank."""
    past = model(input_ids=torch.arange(10).reshape(1, 10) % 64, use_cache=True).past_key_values
    judger, mask, question = _capture_args(question_rows)
    return capture_single.capture_context(
        model,
        past,
        judger,
        mask,
        question,
        pool_kernel=3,
        spec=CaptureSpec(
            energy_pool_kernel=3,
            row_energy_pool_kernel=3,
            support_orders=(2.0,),
            collect_layer_bank=collect_layer_bank,
            bank_include_sliding=all_layers,
        ),
        observer=None,
        lower_right=False,
    )


@pytest.mark.parametrize("implementation", ("eager", "sdpa"))
@pytest.mark.parametrize("family", ("gemma3", "qwen3"))
def test_labeled_layer_bank_replays_live_scores_and_keep(
    implementation: str,
    family: str,
) -> None:
    """Banked rows reproduce snap and support scores on both fixture shapes."""
    model = _gemma3(implementation) if family == "gemma3" else _tiny_qwen(implementation)
    result, context = _bank_capture(model, all_layers=True)
    bank = result.layer_bank
    assert isinstance(bank, capture_layers.LayerBank)
    expected_layers = 6 if family == "gemma3" else int(model.config.num_hidden_layers)
    assert len(bank.rows) == expected_layers
    assert [row.descriptor.layer_idx for row in bank.rows] == list(range(expected_layers))
    assert [row.descriptor.layer_type for row in bank.rows] == (
        ["sliding"] * 5 + ["global"] if family == "gemma3" else ["global"] * expected_layers
    )
    assert all(row.descriptor.logical_memory_length == 10 for row in bank.rows)
    assert all(row.w16_snap.ndim == 1 for row in bank.rows)
    assert all(row.w16_snap.shape == (1,) for row in bank.rows)
    replay_snap = bank.replay_score("snap", scope="global-only")
    assert torch.equal(replay_snap, result.snap)
    assert result.moments is not None
    live_support = selector_score(result.moments, "support-p2-a1")
    replay_support = bank.replay_score("support-p2-a1", scope="global-only")
    assert torch.equal(replay_support, live_support)
    live_support_default = selector_score(result.moments, "support-p2-a2")
    replay_support_default = bank.replay_score("support-p2-a2", scope="global-only")
    assert torch.equal(replay_support_default, live_support_default)
    live_keep = fixed_span_keep(result.snap, 4, (0, 9), span_size=4)
    assert fixed_span_keep(replay_snap, 4, (0, 9), span_size=4) == live_keep
    assert context.layer_descriptors == [row.descriptor for row in bank.rows]


@pytest.mark.parametrize("implementation", ("eager", "sdpa"))
@pytest.mark.parametrize("family", ("gemma3_hybrid", "qwen3_global"))
def test_long_chunked_layer_bank_replays_live_order_exactly(
    implementation: str,
    family: str,
) -> None:
    """Rows beyond one chunk replay in the exact live accumulation order."""
    model = (
        _gemma3(implementation, global_layers=2)
        if family == "gemma3_hybrid"
        else _tiny_qwen(implementation)
    )
    result, context = _bank_capture(model, all_layers=True, question_rows=257)
    bank = result.layer_bank
    assert isinstance(bank, capture_layers.LayerBank)
    global_rows = [row for row in bank.rows if row.descriptor.is_global]
    assert len(global_rows) >= 2
    assert all(len(row.snap_chunks) == 2 for row in global_rows)
    replay_snap = bank.replay_score("snap", scope="global-only")
    assert torch.equal(replay_snap, result.snap)
    assert result.moments is not None
    for selector in ("support-p2-a1", "support-p2-a2"):
        assert torch.equal(
            bank.replay_score(selector, scope="global-only"),
            selector_score(result.moments, selector),
        )
    assert context.hi - context.lo == 257


@pytest.mark.parametrize("implementation", ("eager", "sdpa"))
def test_banking_sliding_rows_does_not_change_the_scored_global_score_or_keep(
    implementation: str,
) -> None:
    """Collecting local diagnostics cannot enter the global-only fold."""
    model = _gemma3(implementation)
    result_off, _ = _bank_capture(model, all_layers=False)
    model = _gemma3(implementation)
    result_on, _ = _bank_capture(model, all_layers=True)
    model = _gemma3(implementation)
    result_plain, _ = _bank_capture(model, all_layers=False, collect_layer_bank=False)
    assert result_off.layer_bank is not None
    assert result_on.layer_bank is not None
    scored_off = result_off.layer_bank.replay_score("snap", scope="global-only")
    scored_on = result_on.layer_bank.replay_score("snap", scope="global-only")
    global_rows = [row for row in result_on.layer_bank.rows if row.descriptor.is_global]
    assert len(global_rows) == 1
    assert torch.equal(result_plain.snap, result_off.snap)
    assert torch.equal(result_off.snap, result_on.snap)
    assert result_plain.energy is not None and result_off.energy is not None
    assert result_plain.row_energy is not None and result_off.row_energy is not None
    assert torch.equal(result_plain.energy, result_off.energy)
    assert torch.equal(result_plain.row_energy, result_off.row_energy)
    assert result_plain.moments is not None and result_off.moments is not None
    assert torch.equal(result_plain.moments.snap, result_off.moments.snap)
    assert torch.equal(result_plain.moments.e[2.0], result_off.moments.e[2.0])
    assert torch.equal(scored_off, scored_on)
    assert fixed_span_keep(result_off.snap, 4, (0, 9), span_size=4) == fixed_span_keep(
        result_on.snap, 4, (0, 9), span_size=4
    )


def _implementation_snapshot(model: Any) -> tuple[str, ...]:
    """Read the top-level and every configured sub-model implementation."""
    config = model.config
    subconfigs = tuple(
        subconfig
        for name in ("text_config", "vision_config", "audio_config")
        if (subconfig := getattr(config, name, None)) is not None
    )
    return tuple(
        str(getattr(subconfig, "_attn_implementation", None)) for subconfig in (config, *subconfigs)
    )


def _set_mixed_implementations(model: Any) -> None:
    """Set mixed top-level and sub-config values without the recursive setter."""
    config = model.config
    subconfigs = tuple(
        subconfig
        for name in ("text_config", "vision_config", "audio_config")
        if (subconfig := getattr(config, name, None)) is not None
    )
    for subconfig, implementation in zip(
        (config, *subconfigs),
        ("eager", "eager", "sdpa", "sdpa"),
        strict=False,
    ):
        vars(subconfig)["_attn_implementation_internal"] = implementation


def test_subconfig_attention_implementations_restore_after_success_and_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed multimodal config strings survive both capture exit paths."""
    model = _gemma3_conditional_generation("eager")
    _set_mixed_implementations(model)
    expected = _implementation_snapshot(model)
    judger, mask, question = _capture_args()
    capture_single.capture_context(
        model,
        _hybrid_cache_from_config(model),
        judger,
        mask,
        question,
        pool_kernel=1,
        spec=CaptureSpec(),
        observer=None,
        lower_right=False,
    )
    assert _implementation_snapshot(model) == expected

    from transformers.models.gemma3 import modeling_gemma3

    def fail(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("forced Gemma3 hooked-forward failure")

    monkeypatch.setattr(modeling_gemma3, "eager_attention_forward", fail)
    with pytest.raises(RuntimeError, match="forced Gemma3 hooked-forward failure"):
        capture_single.capture_context(
            model,
            _hybrid_cache_from_config(model),
            judger,
            mask,
            question,
            pool_kernel=1,
            spec=CaptureSpec(),
            observer=None,
            lower_right=False,
        )
    assert _implementation_snapshot(model) == expected


def test_text_capture_rejects_loaded_multimodal_towers() -> None:
    """The full multimodal wrapper cannot enter the text-only capture seam."""
    model = _gemma3_conditional_generation("eager", strip_tower=False)
    with pytest.raises(ValueError, match="no loaded vision/audio tower"):
        capture_single._assert_text_decoder_inputs(model, torch.tensor([[11, 12, 13]]))


@pytest.mark.parametrize(
    "token_field",
    (
        "image_token_id",
        "image_token_index",
        "video_token_id",
        "audio_token_id",
        "boi_token_id",
        "boi_token_index",
        "eoi_token_id",
        "eoi_token_index",
        "boa_token_id",
    ),
)
def test_text_capture_rejects_every_multimodal_token_field(token_field: str) -> None:
    """Every Gemma multimodal boundary token is rejected before the forward."""
    model = _gemma4("eager")
    setattr(model.config, token_field, 63)
    with pytest.raises(ValueError, match="text-decoder inputs only"):
        capture_single._assert_text_decoder_inputs(model, torch.tensor([[11, 63, 13]]))
    if token_field != "image_token_id":
        return
    # The public seam runs the same guard, so a placeholder id never reaches a forward.
    seam_model = _tiny_gemma4_capture_model("eager")
    seam_model.config.image_token_id = 63
    judger = torch.tensor([[11, 12, 63, 14, 15]], dtype=torch.long)
    with pytest.raises(ValueError, match="text-decoder inputs only"):
        capture_single.capture_context(
            seam_model,
            _hybrid_cache(seam_model),
            judger,
            torch.ones_like(judger),
            judger[0, 1:4],
            pool_kernel=1,
            spec=CaptureSpec(),
            observer=None,
            lower_right=False,
        )


def test_layer_schedule_length_mismatch_is_loud(
    tiny_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial schedule cannot silently become an all-global capture."""
    past = tiny_model(input_ids=random_ids(8, seed=946), use_cache=True).past_key_values
    monkeypatch.setattr(tiny_model.config, "layer_types", ["full_attention"])
    with pytest.raises(RuntimeError, match="layer_types length does not match"):
        capture_single.capture_context(
            tiny_model,
            past,
            random_ids(5, seed=947),
            torch.ones(1, 5, dtype=torch.long),
            [1, 2, 3],
            pool_kernel=1,
            spec=CaptureSpec(),
            observer=None,
            lower_right=False,
        )


def test_missing_attention_registry_entry_is_loud(
    tiny_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend with no mask registry entry cannot fall through silently."""
    past = tiny_model(input_ids=random_ids(8, seed=948), use_cache=True).past_key_values
    monkeypatch.setattr(tiny_model.config, "_attn_implementation", "flash_attention_2_bogus")
    with pytest.raises(RuntimeError, match="cannot delegate mask implementation"):
        capture_single.capture_context(
            tiny_model,
            past,
            random_ids(5, seed=949),
            torch.ones(1, 5, dtype=torch.long),
            [1, 2, 3],
            pool_kernel=1,
            spec=CaptureSpec(),
            observer=None,
            lower_right=False,
        )


def test_callback_hit_count_mismatch_is_loud(
    tiny_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated decoder cannot publish a partial capture result."""
    past = tiny_model(input_ids=random_ids(8, seed=950), use_cache=True).past_key_values
    monkeypatch.setattr(
        tiny_model.model,
        "layers",
        torch.nn.ModuleList([tiny_model.model.layers[0]]),
    )
    with pytest.raises(RuntimeError, match="callback hit count mismatch"):
        capture_single.capture_context(
            tiny_model,
            past,
            random_ids(5, seed=951),
            torch.ones(1, 5, dtype=torch.long),
            [1, 2, 3],
            pool_kernel=1,
            spec=CaptureSpec(),
            observer=None,
            lower_right=False,
        )


def test_external_scorers_refuse_hybrid_models_before_touching_inputs() -> None:
    """Legacy scorers fail loudly until they own hybrid mask preparation."""
    config = SimpleNamespace(model_type="qwen3", layer_types=["sliding_attention"])
    model = SimpleNamespace(config=config, get_decoder=lambda: SimpleNamespace(config=config))
    ids = torch.zeros(1, 0, dtype=torch.long)
    fake_past = cast(Any, object())
    with pytest.raises(RuntimeError, match="refuses hybrid attention"):
        h2o.memory_h2o_scores(model, fake_past, ids, None)
    with pytest.raises(RuntimeError, match="refuses hybrid attention"):
        chunkkv.memory_chunkkv_scores(model, fake_past, ids, None)
    with pytest.raises(RuntimeError, match="refuses hybrid attention"):
        kvzip.memory_kvzip_scores(model, fake_past, ids, cast(Any, None))
    # KVzip reports the hybrid shape before its Qwen family gate, so a non-Qwen
    # hybrid gets the same refusal rather than a family complaint.
    other = SimpleNamespace(
        model_type="gemma3", layer_types=["sliding_attention", "full_attention"]
    )
    other_model = SimpleNamespace(config=other, get_decoder=lambda: SimpleNamespace(config=other))
    with pytest.raises(RuntimeError, match="refuses hybrid attention"):
        kvzip.memory_kvzip_scores(other_model, fake_past, ids, cast(Any, None))


# --- The two registered sliding-row variants.

MEMORY_TOKENS = 96


SLIDING_WINDOW = 48


SUPPORT = "support-p2-a2"


SLIDETAIL_ARM = "gemma4_12b_r4_w16_support_slidetail"


NORM_ARM = "gemma4_12b_r4_w16_support_slidenorm"


def _hybrid(*, all_global: bool = False) -> Any:
    """A tiny hybrid Gemma3 decoder whose sliding window spans several W16 blocks."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    layer_types = (
        ["full_attention"] * 6 if all_global else ["sliding_attention"] * 4 + ["full_attention"] * 2
    )
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        sliding_window=SLIDING_WINDOW,
        layer_types=layer_types,
        query_pre_attn_scalar=64,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(2611)
    return Gemma3ForCausalLM(config).eval()


def _capture(model: Any, *, include_sliding: bool = True) -> Any:
    """Capture one bank through the public single-item capture seam."""
    ids = torch.arange(MEMORY_TOKENS, dtype=torch.long).reshape(1, MEMORY_TOKENS) % 64
    past = model(input_ids=ids, use_cache=True).past_key_values
    judger, mask, question = _capture_args(3)
    result, _context = capture_single.capture_context(
        model,
        past,
        judger,
        mask,
        question,
        pool_kernel=3,
        spec=CaptureSpec(
            energy_pool_kernel=3,
            row_energy_pool_kernel=3,
            support_orders=(2.0,),
            collect_layer_bank=True,
            bank_include_sliding=include_sliding,
        ),
        observer=None,
        lower_right=False,
    )
    assert result.layer_bank is not None
    return result


def test_hybrid_capture_banks_sliding_rows_that_are_zero_outside_their_window() -> None:
    bank = _capture(_hybrid()).layer_bank
    assert bank is not None
    sliding = [row for row in bank.rows if not row.descriptor.is_global]
    assert len(sliding) == 4, "the hybrid fixture banked no sliding rows"
    for row in sliding:
        start, end = capture_bank.covered_span(row)
        assert end - start > 0 and (start > 0 or end < MEMORY_TOKENS), (
            "the fixture window must cover a strict subset of the memory"
        )
        outside = torch.cat([row.snap[:start], row.snap[end:]])
        assert not bool(outside.any()), "a sliding row carries votes outside its own window"


def test_sliding_tail_variant_differs_from_the_global_only_support_cut() -> None:
    result = _capture(_hybrid())
    bank = result.layer_bank
    assert bank is not None
    live = selector_score(result.moments, SUPPORT)
    tail = bank.replay_score(SLIDETAIL_ARM)
    assert tail.shape == live.shape
    assert bool(torch.isfinite(tail).all()) and bool((tail >= 0).all())
    assert not torch.equal(tail, live), (
        "the sliding tail equals the global cut: the sliding rows did not enter the vote"
    )
    # The tail is exactly the all-layer replay of the shipped composition, so
    # the variant adds rows rather than changing the composition.
    assert torch.equal(
        tail, selector_score(bank.replay_moments(scope=BANK_SCOPE_ALL_LAYER), SUPPORT)
    )


@pytest.mark.parametrize("arm", (SLIDETAIL_ARM, NORM_ARM))
def test_variant_replays_bit_exactly_after_a_weights_only_round_trip(
    tmp_path: Path, arm: str
) -> None:
    bank = _capture(_hybrid()).layer_bank
    assert bank is not None
    live = bank.replay_score(arm)
    path = tmp_path / "variant-bank.pt"
    bank.save(path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(raw, dict)
    reloaded = capture_bank.LayerBank.load(path)
    assert torch.equal(reloaded.replay_score(arm), live), (
        f"{arm} does not recut bit-exactly from its own durable bank"
    )


@pytest.mark.parametrize("arm", (SLIDETAIL_ARM, NORM_ARM))
def test_variant_refuses_a_bank_with_no_sliding_rows(arm: str) -> None:
    """Global-only and all-global banks both refuse: neither has a sliding tail."""
    global_only = _capture(_hybrid(), include_sliding=False).layer_bank
    assert global_only is not None
    with pytest.raises(RuntimeError, match="cannot replay sliding scope"):
        global_only.replay_score(arm)
    all_global = _capture(_hybrid(all_global=True)).layer_bank
    assert all_global is not None
    with pytest.raises(RuntimeError, match="needs sliding rows"):
        all_global.replay_score(arm)
    # The fold is total; only the policy gates. It still runs on those rows.
    assert (
        variants.variant_fold_score(
            list(all_global.rows), "slidenorm", SUPPORT, pool_kernel=1, support_orders=(2.0,)
        ).shape[0]
        > 0
    )


@pytest.mark.parametrize("arm", (SLIDETAIL_ARM, NORM_ARM))
def test_variant_refuses_a_narrowed_replay_scope(arm: str) -> None:
    bank = _capture(_hybrid()).layer_bank
    assert bank is not None
    with pytest.raises(RuntimeError, match="cannot replay at scope"):
        bank.replay_score(arm, scope=BANK_SCOPE_GLOBAL_ONLY)


@pytest.mark.parametrize(
    "name",
    (
        "gemma4_12b_r4_w16_support_bogus",
        "gemma4_12b_r4_w16_snap_slidetail",
        "gemma4_12b_r3_w16_support_slidetail",
        "gemma4_12b_r4_w8_support_slidenorm",
        "gemma4_12b_r256_w16_support_slidenorm",
        "gemma4_12b_r4_support_slidenorm",
        "gemma4_12b_r4_w16_support_conc",
        "qwen3_8b_r4_w16_support_slidetail",
        "qwen3_8b_r4_w16_support_slidenorm",
        "slidetail",
        "support_slidenorm",
    ),
)
def test_bank_refuses_unregistered_variant_names(name: str) -> None:
    bank = _capture(_hybrid()).layer_bank
    assert bank is not None
    with pytest.raises(ValueError, match="not a registered bank selector"):
        bank.replay_score(name)


def _inject_outside_window(bank: Any, position: int = 0) -> Any:
    """Put mass at a position a sliding row cannot have voted on.

    The injection covers snap and every moment tensor, since a variant folds
    all of them; the row is the first sliding one whose window excludes it.
    """
    row = next(
        candidate
        for candidate in bank.rows
        if not candidate.descriptor.is_global
        and position < candidate.descriptor.absolute_key_offset
    )
    for name in ("snap", "energy", "row_energy", "support_einf", "support_rinf"):
        tensor = getattr(row, name)
        if tensor is not None:
            tensor[position] = 1.0
    for mapping in (row.support_e, row.support_eprime, row.support_r):
        for tensor in mapping.values():
            tensor[position] = 1.0
    for chunk in row.snap_chunks:
        chunk[position] = 1.0
    return row


def test_a_sliding_row_with_outside_window_mass_is_refused_by_every_path(
    tmp_path: Path,
) -> None:
    """Forged votes never reach a score, from memory or from a durable file.

    A durable round trip reproduces the malformed score exactly, so only the
    window check can refuse it: at every variant, at the cut, and at load.
    """
    result = _capture(_hybrid())
    bank = result.layer_bank
    assert bank is not None
    # Positive control first: the clean bank scores, and its durable file loads.
    clean_tail = bank.replay_score(SLIDETAIL_ARM)
    assert clean_tail.shape[0] == MEMORY_TOKENS
    bank.require_window_support()
    for clean_row in bank.rows:
        capture_bank.require_window_support(clean_row)
    clean_path = tmp_path / "clean.pt"
    bank.save(clean_path)
    assert torch.equal(
        capture_bank.LayerBank.load(clean_path).replay_score(SLIDETAIL_ARM), clean_tail
    )

    row = _inject_outside_window(bank)
    covered = (row.descriptor.absolute_key_offset, row.descriptor.physical_memory_length)
    assert covered[0] > 0, "the probe needs a row whose covered interval excludes position 0"
    # 1. Neither variant may score it, from memory.
    for arm in (SLIDETAIL_ARM, NORM_ARM):
        with pytest.raises(RuntimeError, match=r"outside its own attention window|cannot vote"):
            bank.replay_score(arm)
    # 2. Nor may the shipped global cut, which shares the same row resolution.
    with pytest.raises(RuntimeError, match="cannot vote"):
        bank.replay_score("gemma4_12b_r4_w16_support")
    # 3. Nor may the row-level accessor the variants read.
    with pytest.raises(RuntimeError, match="cannot vote"):
        bank.selected_rows(BANK_SCOPE_ALL_LAYER)
    # 4. The durable path refuses at load, before any caller can hold the bank.
    malformed = tmp_path / "malformed.pt"
    torch.save(bank.to_payload(), malformed)
    raw = torch.load(malformed, map_location="cpu", weights_only=True)
    assert isinstance(raw, dict), "the payload still round-trips as plain tensors"
    with pytest.raises(RuntimeError, match="cannot vote"):
        capture_bank.LayerBank.load(malformed)


@pytest.mark.parametrize(
    "field",
    ("snap", "snap_chunk", "energy", "row_energy", "support_e", "support_einf"),
)
def test_outside_window_mass_is_caught_in_every_banked_tensor(field: str) -> None:
    """Forging any one tensor is enough to refuse: no vector is unchecked."""
    bank = _capture(_hybrid()).layer_bank
    assert bank is not None
    row = next(
        candidate
        for candidate in bank.rows
        if not candidate.descriptor.is_global and candidate.descriptor.absolute_key_offset > 0
    )
    if field == "snap_chunk":
        row.snap_chunks[0][0] = 1.0
    elif field == "support_e":
        next(iter(row.support_e.values()))[0] = 1.0
    else:
        tensor = getattr(row, field)
        assert tensor is not None
        tensor[0] = 1.0
    with pytest.raises(RuntimeError, match="cannot vote"):
        bank.replay_score(SLIDETAIL_ARM)


def test_unknown_variant_token_is_refused_at_the_scorer() -> None:
    assert set(variants.VARIANT_FOLD_RULE) == set(variants.VARIANT_TOKENS)
    bank = _capture(_hybrid()).layer_bank
    assert bank is not None
    with pytest.raises(ValueError, match="unknown selector variant"):
        variants.variant_score(bank, "sideways", SUPPORT)
