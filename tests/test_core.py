"""The engine-free core: the cache, the meters, RoPE relocation, composition, and the handoff."""

import dataclasses

import pytest
import torch
from tests.conftest import make_cache, random_ids

from rcc import (
    Axis,
    Kind,
    KVCache,
    Meter,
    MeterViolation,
    Pipeline,
    RopeParams,
    Transform,
    factored,
    measure,
)
from rcc.harness.handoff import handoff
from rcc.harness.score import (
    exact_match,
    extract_answer,
    normalize_answer,
    qa_prompt,
    score_prediction,
    token_f1,
)
from rcc.rope import apply_rope_k, rope_cos_sin, unapply_rope_k


def test_axis_properties():
    cache = make_cache(layers=3, kv_heads=2, length=16, head_dim=8)
    assert cache.layers == 3
    assert cache.kv_heads == 2
    assert cache.length == 16
    assert cache.head_dim == 8
    assert cache.bits == 16  # float16
    for axis in Axis:
        assert cache.axis_size(axis) > 0


def test_mismatched_key_value_shapes_rejected():
    good = make_cache()
    with pytest.raises(ValueError, match="disagree"):
        KVCache(keys=good.keys, values=good.values[:, :, :-1], positions=good.positions)


def test_mismatched_dtypes_rejected():
    good = make_cache()
    with pytest.raises(ValueError, match="disagree"):
        KVCache(keys=good.keys, values=good.values.float(), positions=good.positions)


def test_wrong_positions_length_rejected():
    good = make_cache(length=8)
    with pytest.raises(ValueError, match="positions"):
        KVCache(keys=good.keys, values=good.values, positions=torch.arange(7))


def test_non_4d_keys_rejected():
    with pytest.raises(ValueError, match="layers, kv_heads, length, head_dim"):
        KVCache(
            keys=torch.zeros(2, 8, 4),
            values=torch.zeros(2, 8, 4),
            positions=torch.arange(8),
        )


def test_meters_analytic_math():
    cache = make_cache(layers=2, kv_heads=2, length=8, head_dim=4, dtype=torch.float16)
    meters = cache.meters()
    payload = 2 * (2 * 2 * 8 * 4) * 2  # K and V, numel, 2 bytes per fp16 element
    positions = 8 * 8  # int64
    assert meters.resident_bytes == payload
    assert meters.wire_bytes == payload + positions
    assert measure(cache) == meters


def test_bad_rope_inv_freq_rejected():
    good = make_cache(head_dim=4)
    with pytest.raises(ValueError, match="inv_freq"):
        KVCache(
            keys=good.keys,
            values=good.values,
            positions=good.positions,
            rope=RopeParams(inv_freq=torch.ones(5)),  # head_dim/2 is 2, not 5
        )


def test_from_prefill_shapes_positions_rope_and_meters(tiny_model):
    ids = random_ids(length=10, seed=4)
    cache = KVCache.from_prefill(tiny_model, ids)
    assert cache.keys.shape == (2, 2, 10, 16)
    assert cache.values.shape == (2, 2, 10, 16)
    assert torch.equal(cache.positions, torch.arange(10))
    assert cache.rope is not None
    assert cache.rope.inv_freq.shape == (8,)
    meters = cache.meters()
    assert meters.resident_bytes > 0
    assert meters.wire_bytes > meters.resident_bytes


def test_from_prefill_accepts_unbatched_ids(tiny_model):
    cache = KVCache.from_prefill(tiny_model, random_ids(length=6, seed=5)[0])
    assert cache.length == 6


def test_from_prefill_rejects_batched_ids(tiny_model):
    with pytest.raises(ValueError, match="single-sequence"):
        KVCache.from_prefill(tiny_model, torch.zeros(2, 6, dtype=torch.long))


def test_to_hf_cache_round_trips_tensors(tiny_model):
    ids = random_ids(length=8, seed=6)
    cache = KVCache.from_prefill(tiny_model, ids)
    past = cache.to_hf_cache()
    layers = past.layers
    assert len(layers) == cache.layers
    for i, layer in enumerate(layers):
        assert torch.equal(layer.keys[0], cache.keys[i])
        assert torch.equal(layer.values[0], cache.values[i])


@torch.no_grad()
def _continue(model, past, first_logits, past_len, n_new):
    tokens = []
    nxt = first_logits.argmax(-1, keepdim=True)
    total = past_len
    for _ in range(n_new):
        tokens.append(int(nxt[0, 0]))
        out = model(
            input_ids=nxt,
            past_key_values=past,
            position_ids=torch.tensor([[total]]),
            attention_mask=torch.ones(1, total + 1, dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        total += 1
    return tokens


def test_round_trip_decode_is_exact(tiny_model):
    # The exactness guarantee: a cache built by from_prefill and rebuilt by
    # to_hf_cache decodes token-for-token identically to a monolithic forward.
    length, n_new = 14, 10
    ids = random_ids(length=length, seed=7)

    with torch.no_grad():
        direct = tiny_model(
            input_ids=ids,
            position_ids=torch.arange(length).unsqueeze(0),
            attention_mask=torch.ones(1, length, dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
    direct_tokens = _continue(
        tiny_model, direct.past_key_values, direct.logits[:, -1], length, n_new
    )

    prefix, last = ids[:, : length - 1], ids[:, length - 1 :]
    cache = KVCache.from_prefill(tiny_model, prefix)
    past = cache.to_hf_cache()
    with torch.no_grad():
        primed = tiny_model(
            input_ids=last,
            past_key_values=past,
            position_ids=torch.tensor([[length - 1]]),
            attention_mask=torch.ones(1, length, dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
    cache_tokens = _continue(
        tiny_model, primed.past_key_values, primed.logits[:, -1], length, n_new
    )

    assert cache_tokens == direct_tokens


def test_factored_two_axes_multiply():
    before = make_cache(length=16, dtype=torch.float16)
    after = KVCache(
        keys=before.keys[:, :, :4].to(torch.uint8),
        values=before.values[:, :, :4].to(torch.uint8),
        positions=before.positions[:4],
    )
    ratio = factored(before, after)
    assert ratio.per_axis == {Axis.LENGTH: 4.0, Axis.BITS: 2.0}
    assert ratio.product == 8.0
    rendered = str(ratio)
    assert rendered.startswith("8x = ")
    assert "4x(length)" in rendered
    assert "2x(bits)" in rendered
    identity = factored(before, before)
    assert identity.per_axis == {}
    assert identity.product == 1.0
    assert str(identity) == "1.0x (identity)"


def _rope_params(head_dim: int = 16, base: float = 10000.0) -> RopeParams:
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    return RopeParams(inv_freq=inv_freq, attention_scaling=1.0)


def test_relocation_moves_positions():
    rope = _rope_params()
    gen = torch.Generator().manual_seed(2)
    k = torch.randn(1, 2, 4, 16, generator=gen)
    src = torch.tensor([[3, 5, 8, 11]])
    tgt = torch.arange(4).unsqueeze(0)
    sc, ss = rope_cos_sin(rope, src)
    tc, ts = rope_cos_sin(rope, tgt)
    moved = apply_rope_k(unapply_rope_k(k, sc, ss), tc, ts)
    # relocating to identical positions is a no-op; to different positions is not.
    same = apply_rope_k(unapply_rope_k(k, sc, ss), sc, ss)
    assert torch.allclose(same, k, atol=1e-5)
    assert not torch.allclose(moved, k, atol=1e-4)


@pytest.mark.parametrize("scaling", [1.0, 1.3])
def test_unapply_inverts_apply_under_attention_scaling(scaling: float):
    # cos/sin carry attention_scaling s, so apply multiplies by s * Rot(theta)
    # and a sine-negation-only inverse would return s^2 * k. The true inverse
    # divides the s^2 back out.
    rope = RopeParams(inv_freq=_rope_params().inv_freq, attention_scaling=scaling)
    gen = torch.Generator().manual_seed(3)
    k = torch.randn(1, 2, 7, 16, generator=gen)
    positions = torch.arange(7).unsqueeze(0)
    cos, sin = rope_cos_sin(rope, positions)
    round_trip = unapply_rope_k(apply_rope_k(k, cos, sin), cos, sin)
    assert torch.allclose(round_trip, k, atol=1e-5, rtol=1e-5)

    # Relocation composes unapply at src with apply at tgt: the result must be
    # a pure rotation of k (norm preserved), never rescaled by s or s^2.
    src = torch.tensor([[3, 5, 8, 11]])
    tgt = torch.arange(4).unsqueeze(0)
    sc, ss = rope_cos_sin(rope, src)
    tc, ts = rope_cos_sin(rope, tgt)
    k4 = torch.randn(1, 2, 4, 16, generator=gen)
    moved = apply_rope_k(unapply_rope_k(k4, sc, ss), tc, ts)
    pair = k4.view(1, 2, 4, 2, 8)
    moved_pair = moved.view(1, 2, 4, 2, 8)
    assert torch.allclose(moved_pair.norm(dim=-2), pair.norm(dim=-2), atol=1e-5, rtol=1e-5)


def test_cos_sin_shape_and_scaling():
    rope = RopeParams(inv_freq=_rope_params().inv_freq, attention_scaling=2.0)
    positions = torch.arange(5).unsqueeze(0)
    cos, sin = rope_cos_sin(rope, positions)
    assert cos.shape == (1, 5, 16)
    assert sin.shape == (1, 5, 16)
    # attention_scaling multiplies both; at position 0 the angle is 0 so cos == scaling.
    assert torch.allclose(cos[0, 0], torch.full((16,), 2.0))


def test_rope_cos_sin_rejects_1d_positions():
    rope = _rope_params()
    with pytest.raises(ValueError, match="position_ids"):
        rope_cos_sin(rope, torch.arange(5))


def test_rope_cos_sin_matches_model(tiny_model):
    rotary = tiny_model.model.rotary_emb
    rope = RopeParams(
        inv_freq=rotary.inv_freq.clone(),
        attention_scaling=float(rotary.attention_scaling),
    )
    positions = torch.arange(9).unsqueeze(0)
    cos_ours, sin_ours = rope_cos_sin(rope, positions)
    dummy = torch.zeros(1, 1, 1, dtype=next(tiny_model.parameters()).dtype)
    cos_model, sin_model = rotary(dummy, positions)
    assert torch.allclose(cos_ours.to(cos_model.dtype), cos_model, atol=1e-6)
    assert torch.allclose(sin_ours.to(sin_model.dtype), sin_model, atol=1e-6)


class HalveLength(Transform):
    """Keep the first half of the length axis, honestly declared."""

    kind = Kind.ARTIFACT
    meters = frozenset({Meter.WIRE_BYTES, Meter.RESIDENT_BYTES})
    axis = Axis.LENGTH

    def apply(self, cache: KVCache) -> KVCache:
        keep = cache.length // 2
        return KVCache(
            keys=cache.keys[:, :, :keep],
            values=cache.values[:, :, :keep],
            positions=cache.positions[:keep],
        )


class LyingNoop(Transform):
    """Declares it moves nothing, then halves the length axis."""

    kind = Kind.ARTIFACT
    meters = frozenset()
    axis = Axis.LENGTH

    def apply(self, cache: KVCache) -> KVCache:
        return HalveLength().apply(cache)


class HonestNoop(Transform):
    """Declares nothing, moves nothing."""

    kind = Kind.WRAPPER
    meters = frozenset()
    axis = None

    def apply(self, cache: KVCache) -> KVCache:
        return cache


def test_empty_pipeline_round_trips():
    cache = make_cache()
    out = Pipeline().apply(cache)
    assert out is cache
    assert torch.equal(out.keys, cache.keys)
    assert torch.equal(out.values, cache.values)
    assert torch.equal(out.positions, cache.positions)


def test_or_composition_builds_pipeline():
    pipe = HalveLength() | HonestNoop()
    assert isinstance(pipe, Pipeline)
    assert len(pipe) == 2
    longer = pipe | HonestNoop()
    assert len(longer) == 3
    assert len(pipe) == 2  # composition is not mutation


def test_apply_runs_in_order():
    cache = make_cache(length=16)
    assert Pipeline((HalveLength(),)).apply(cache).length == 8
    out = Pipeline((HalveLength(), HalveLength())).apply(cache)
    assert out.length == 4


def test_same_axis_composition_warns():
    with pytest.warns(UserWarning, match="same axis"):
        Pipeline((HalveLength(), HalveLength()))


def test_cross_axis_composition_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Pipeline((HalveLength(), HonestNoop()))


def test_undeclared_meter_movement_raises():
    cache = make_cache()
    with pytest.raises(MeterViolation, match="LyingNoop"):
        Pipeline((LyingNoop(),)).apply(cache)


def test_input_cache_never_mutated():
    cache = make_cache(length=16)
    frozen = dataclasses.replace(cache)  # shallow copy of the container
    before_k = cache.keys.clone()
    Pipeline((HalveLength(),)).apply(cache)
    assert torch.equal(cache.keys, before_k)
    assert cache.length == frozen.length


def test_handoff_is_deterministic(tiny_model):
    ids = random_ids(length=10, seed=1)
    query = random_ids(length=3, seed=2)
    cache = KVCache.from_prefill(tiny_model, ids)
    first = handoff(cache, tiny_model, query, max_new_tokens=6)
    second = handoff(cache, tiny_model, query, max_new_tokens=6)
    assert torch.equal(first, second)


def test_handoff_stops_at_eos_and_drops_it(tiny_model):
    # Default decode is fixed-length; stop_at_eos halts at the first end-of-turn
    # id and drops it, so the QA path never carries hallucinated trailing tokens.
    ids = random_ids(length=8, seed=5)
    query = random_ids(length=3, seed=6)
    cache = KVCache.from_prefill(tiny_model, ids)
    full = handoff(cache, tiny_model, query, max_new_tokens=6)
    assert full.shape == (1, 6)
    generation_config = tiny_model.generation_config
    saved = generation_config.eos_token_id
    try:
        generation_config.eos_token_id = int(full[0, 1])
        stopped = handoff(cache, tiny_model, query, max_new_tokens=6, stop_at_eos=True)
    finally:
        generation_config.eos_token_id = saved
    assert stopped.shape == (1, 1)
    assert int(stopped[0, 0]) == int(full[0, 0])


def test_handoff_rejects_batched_query(tiny_model):
    ids = random_ids(length=6, seed=3)
    cache = KVCache.from_prefill(tiny_model, ids)
    query = torch.zeros(2, 3, dtype=torch.long)
    try:
        handoff(cache, tiny_model, query)
    except ValueError as exc:
        assert "single-sequence" in str(exc)
    else:
        raise AssertionError("expected ValueError for batched query")


def test_full_cache_handoff_equals_direct_decode(tiny_model):
    # The same exactness guarantee through the public API: a from_prefill cache
    # handed off and reprefilled with the query decodes token-for-token
    # identically to one monolithic forward over [context, query].
    context_len, query_len, n_new = 9, 3, 5
    context = random_ids(length=context_len, seed=11)
    query = random_ids(length=query_len, seed=12)
    full_ids = torch.cat([context, query], dim=1)

    with torch.no_grad():
        direct = tiny_model(
            input_ids=full_ids,
            position_ids=torch.arange(context_len + query_len).unsqueeze(0),
            attention_mask=torch.ones(1, context_len + query_len, dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
    direct_tokens: list[int] = []
    nxt = direct.logits[:, -1, :].argmax(-1, keepdim=True)
    past = direct.past_key_values
    total = context_len + query_len
    for _ in range(n_new):
        direct_tokens.append(int(nxt[0, 0]))
        with torch.no_grad():
            out = tiny_model(
                input_ids=nxt,
                past_key_values=past,
                position_ids=torch.tensor([[total]]),
                attention_mask=torch.ones(1, total + 1, dtype=torch.long),
                use_cache=True,
                return_dict=True,
            )
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        total += 1

    cache = KVCache.from_prefill(tiny_model, context)
    generated = handoff(cache, tiny_model, query, max_new_tokens=n_new)

    assert generated.shape == (1, n_new)
    assert generated[0].tolist() == direct_tokens


def test_extract_answer_marker_wins_only_inside_the_first_line():
    assert extract_answer("Answer: Paris\nsome trailing note Answer: Lyon") == "Paris"


def test_extract_answer_takes_first_line_over_trailing_chatter():
    # qa_prompt primes the reply with "Answer: ", so the answer leads and a
    # hallucinated next turn after it must not displace the answer, even when
    # that turn restates an "Answer:" marker of its own.
    assert extract_answer("SIGMA365\nHuman: what is the next question") == "SIGMA365"
    assert extract_answer("Paris\n\nQuestion: capital of Spain?\nAnswer: Madrid") == "Paris"


def test_extract_answer_strips_wrappers():
    assert extract_answer("Answer: **Joe Biden**.") == "Joe Biden"


def test_extract_answer_none_on_empty():
    assert extract_answer("") is None
    assert extract_answer(None) is None


def test_normalize_answer_squad_convention():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert normalize_answer(None) == ""


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("the big red dog", "big red cat", 0.5714285714285715),
        ("", "", 1.0),
        ("paris", "", 0.0),
        ("", "paris", 0.0),
    ],
)
def test_token_f1_partial_overlap(prediction: str, gold: str, expected: float):
    assert token_f1(prediction, gold) == expected


def test_exact_match_on_normalized_strings():
    assert exact_match("joe biden", "joe biden") == 1.0
    assert exact_match("joe biden", "joe biden jr") == 0.0


def test_score_prediction_takes_best_over_gold_and_aliases():
    f1, em = score_prediction(extract_answer("Answer: JFK"), "John F Kennedy", aliases=["JFK"])
    assert em == 1.0
    assert f1 == 1.0


def test_score_prediction_degenerate_empty_gold():
    assert score_prediction("", "") == (1.0, 1.0)
    assert score_prediction("something", "") == (0.0, 0.0)


def test_qa_prompt_raw_fallback_without_chat_template():
    class NoTemplateTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            raise ValueError("no chat template")

    prompt = qa_prompt(NoTemplateTokenizer(), "Who wrote Hamlet?")
    assert prompt == "Question: Who wrote Hamlet?\nAnswer:"


def test_qa_prompt_uses_chat_template_and_enable_thinking_false():
    calls: list[dict[str, object]] = []

    class TemplateTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            tokenize: bool,
            add_generation_prompt: bool,
            **kwargs: object,
        ) -> str:
            calls.append(kwargs)
            return "TEMPLATED: " + messages[0]["content"]

    prompt = qa_prompt(TemplateTokenizer(), "Who wrote Hamlet?")
    assert prompt.startswith("TEMPLATED: Question: Who wrote Hamlet?")
    assert prompt.endswith("Answer: ")
    assert calls[0] == {"enable_thinking": False}
