"""CacheBack score composition: reducers, the support correction, energy votes, and folds."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from tests.conftest import prefill, random_ids

import rcc.transforms.select.query_support.capture_single as capture_single
import rcc.transforms.select.query_support.methods.compose as compose
import rcc.transforms.select.query_support.methods.reducers as reducers
from rcc.latent.cache import cache_kv, cache_length, split_past_rows
from rcc.latent.rollout import build_realign, latent_rollout_batched
from rcc.transforms.select.core import kernels
from rcc.transforms.select.core.types import CaptureSpec
from rcc.transforms.select.query_support import capture_bank
from rcc.transforms.select.query_support.capture_bank_format import (
    BANK_SCOPE_ALL_LAYER,
)
from rcc.transforms.select.query_support.capture_layers import w16_mean
from rcc.transforms.select.query_support.fixed_spans import fixed_span_keep, locbench_rn_keep
from rcc.transforms.select.query_support.methods import variants
from rcc.transforms.select.query_support.methods.batch import (
    capture_query_support_batched_from_past,
)
from rcc.transforms.select.query_support.methods.compose import (
    SUPPORT_DEFAULT_ALPHA,
    SUPPORT_MOMENT_SCALE,
    support_corrected_scores,
)
from rcc.transforms.select.query_support.methods.reducers import (
    normalized_row_energy_scores,
    normalized_stream_energy_scores,
    row_energy_scores,
    stream_energy_scores,
    stream_pool_sum_scores,
)
from rcc.transforms.select.query_support.methods.scorers import (
    memory_energy_votes,
    memory_row_energy_votes,
    memory_votes_with_energies,
    memory_votes_with_support_moments,
)

# --- Composition and reducers.


def _bundle() -> compose.SupportMomentBundle:
    return compose.SupportMomentBundle(
        snap=torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32),
        e={
            2.0: torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.float32),
            8.0: torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32),
        },
        eprime={
            2.0: torch.tensor([0.0625, 0.125, 0.25, 0.5], dtype=torch.float32),
            8.0: torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.float32),
        },
        r={
            2.0: torch.tensor([0.25, 0.125, 0.25, 0.5], dtype=torch.float32),
            8.0: torch.tensor([0.5, 0.25, 0.5, 1.0], dtype=torch.float32),
        },
        einf=torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32),
        rinf=torch.tensor([1.0, 0.5, 1.0, 2.0], dtype=torch.float32),
    )


def test_bundle_round_trip_and_shipped_support_composition_are_bit_exact() -> None:
    bundle = _bundle()
    vectors = bundle.as_vectors()
    restored = compose.SupportMomentBundle.from_vectors(vectors)

    assert restored.orders == (2.0, 8.0)
    for name, vector in vectors.items():
        assert torch.equal(restored.as_vectors()[name], vector)

    shipped = compose.support_corrected_scores(bundle.snap, bundle.e[2.0], bundle.r[2.0])
    assert torch.equal(compose.compose_score(bundle, order=2.0, alpha=1.0), shipped)
    assert torch.equal(compose.selector_score(bundle, "support-p2-a1"), shipped)
    default = compose.support_corrected_scores(bundle.snap, bundle.e[2.0], bundle.r[2.0], alpha=2.0)
    assert torch.allclose(compose.selector_score(bundle, "support-p2-a2"), default)
    assert torch.equal(compose.selector_score(bundle, "snap"), bundle.snap)
    assert torch.equal(
        compose.correction(bundle, 2.0),
        torch.tensor([2.0, 0.5, 0.5, 0.5], dtype=torch.float32),
    )
    assert torch.equal(
        compose.correction(bundle, "inf"),
        torch.tensor([2.0, 0.5, 0.5, 0.5], dtype=torch.float32),
    )
    pure, alignment = compose.concentration_and_alignment(bundle, 2.0)
    assert torch.equal(pure, torch.tensor([4.0, 1.0, 1.0, 1.0], dtype=torch.float32))
    assert torch.equal(alignment, torch.full((4,), 0.5, dtype=torch.float32))
    assert torch.equal(
        compose.selector_score(bundle, "support-pinf-a2"),
        torch.tensor([1.0, 0.125, 0.1875, 0.25], dtype=torch.float32),
    )


def test_composition_math_and_selector_parsing_are_pinned() -> None:
    assert compose.SUPPORT_MOMENT_SCALE == 256.0
    values = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32)
    assert torch.equal(
        compose.moment_power(values, 2.0),
        torch.tensor([4096.0, 16384.0, 65536.0], dtype=torch.float32),
    )
    assert compose.parse_selector("snap") == (2.0, 0.0)
    assert compose.parse_selector("support") is None
    assert compose.parse_selector("support-p2-a1") == (2.0, 1.0)
    assert compose.parse_selector("support-p8-a4") == (8.0, 4.0)
    assert compose.parse_selector("support-pinf-a2") == ("inf", 2.0)


def test_reducers_are_bit_exact_on_fixed_seeds() -> None:
    expected = {
        410: (
            (
                0.3943900465965271,
                0.8814104795455933,
                0.8814104795455933,
                1.162787914276123,
                0.770758330821991,
            ),
            (
                0.6732648015022278,
                0.9622892737388611,
                0.9622892737388611,
                1.216868281364441,
                0.9433767795562744,
            ),
        ),
        411: (
            (
                0.7817579507827759,
                0.7817579507827759,
                0.6321190595626831,
                0.8358668088912964,
                0.7808015942573547,
            ),
            (
                1.0396630764007568,
                1.0396630764007568,
                0.6749083995819092,
                0.9258888363838196,
                0.8829700350761414,
            ),
        ),
    }
    for seed, (stream_expected, row_expected) in expected.items():
        torch.manual_seed(seed)
        weights = torch.rand(4, 2, 7, dtype=torch.float32)
        assert torch.equal(
            reducers.stream_energy_scores(weights, mem_len=5, kv_groups=2, pool_kernel=3),
            torch.tensor(stream_expected, dtype=torch.float32),
        )
        assert torch.equal(
            reducers.row_energy_scores(weights, mem_len=5, kv_groups=2, pool_kernel=3),
            torch.tensor(row_expected, dtype=torch.float32),
        )


# --- The variant folds on synthetic rows.

SUPPORT = "support-p2-a2"


SYNTH_LOGICAL = 96


SYNTH_ORDERS = (2.0,)


def _synth_row(
    layer_idx: int,
    *,
    layer_type: str,
    values: torch.Tensor,
    window: int | None = None,
    offset: int = 0,
    physical: int | None = None,
) -> Any:
    """One banked row with a chosen covered span and a chosen per-layer vote.

    The support moments are set equal, so the correction is one at every
    position and the composed score is the pooled snap, which isolates the fold.
    """
    from rcc.transforms.select.core.types import LayerDescriptor

    covered = SYNTH_LOGICAL if physical is None else physical
    descriptor = LayerDescriptor(
        layer_idx=layer_idx,
        layer_type=layer_type,
        logical_memory_length=SYNTH_LOGICAL,
        physical_key_length=covered + 1,
        absolute_key_offset=offset,
        window=window,
        query_length=1,
    )
    ones = torch.ones(SYNTH_LOGICAL, dtype=torch.float32)
    return capture_bank.LayerBankRow(
        descriptor=descriptor,
        snap=values.clone(),
        energy=ones.clone(),
        row_energy=ones.clone(),
        support_e={2.0: ones.clone()},
        support_eprime={2.0: ones.clone()},
        support_r={2.0: ones.clone()},
        support_einf=ones.clone(),
        support_rinf=ones.clone(),
        prepool_e2=None,
        prepool_r2=None,
        heads=1,
        query_rows=1,
        kv_groups=1,
        snap_chunks=(values.clone(),),
    )


#: The two widths every fold test runs at: 1 isolates the fold arithmetic, and
#: 7 is the production pool width. The pool is a nonlinear max, so results at
#: width 1 do not carry over to width 7.
FOLD_WIDTHS = (1, 7)


def _fold(rows: list[Any], variant: str, pool_kernel: int = 1) -> torch.Tensor:
    return variants.variant_fold_score(
        rows, variant, SUPPORT, pool_kernel=pool_kernel, support_orders=SYNTH_ORDERS
    )


@pytest.mark.parametrize("pool_kernel", FOLD_WIDTHS)
def test_on_an_all_global_bank_the_mean_fold_ranks_exactly_as_the_global_only_cut(
    pool_kernel: int,
) -> None:
    """Constant coverage makes the mean fold a uniform rescale of the sum fold.

    Every layer covers every position, so the mean fold is a uniform positive
    rescale: it reorders nothing and the keep set matches the global cut's.
    """
    torch.manual_seed(11)
    rows = [
        _synth_row(index, layer_type="global", values=torch.rand(SYNTH_LOGICAL) + 0.1)
        for index in range(4)
    ]
    summed = _fold(rows, "slidetail", pool_kernel)
    meaned = _fold(rows, "slidenorm", pool_kernel)
    # Constant coverage commutes with the max pool, so this holds at every width.
    assert torch.allclose(meaned * 4.0, summed, atol=1e-6), "coverage is constant at 4 layers"
    assert not torch.equal(meaned, summed), "the fold must actually divide"
    for ratio in (2, 4, 8, 16, 32, 64, 128):
        assert locbench_rn_keep(
            meaned, float(ratio), protected=(0,), span_size=16
        ) == locbench_rn_keep(summed, float(ratio), protected=(0,), span_size=16), (
            f"r{ratio} keep differs under a uniform rescale"
        )


@pytest.mark.parametrize("pool_kernel", FOLD_WIDTHS)
def test_the_mean_fold_undoes_the_coverage_gradient_the_sum_fold_creates(
    pool_kernel: int,
) -> None:
    """The normalization changes the ranking, not only the scale.

    A tail position collects six votes of 1.5 and a deep position two of 3.0,
    so the sum ranks the tail first and the mean ranks the deep position first.
    """
    deep, tail = 10, 80
    globals_ = []
    for index in range(2):
        values = torch.zeros(SYNTH_LOGICAL)
        values[deep] = 3.0
        values[tail] = 1.5
        globals_.append(_synth_row(index, layer_type="global", values=values))
    slidings = []
    for index in range(4):
        values = torch.zeros(SYNTH_LOGICAL)
        values[tail] = 1.5  # a sliding layer cannot see the deep position at all
        slidings.append(
            _synth_row(
                2 + index,
                layer_type="sliding",
                values=values,
                window=32,
                offset=SYNTH_LOGICAL - 32,
                physical=32,
            )
        )
    rows = [*globals_, *slidings]
    summed = _fold(rows, "slidetail", pool_kernel)
    meaned = _fold(rows, "slidenorm", pool_kernel)
    assert summed[tail].item() == pytest.approx(9.0)
    assert summed[deep].item() == pytest.approx(6.0)
    assert summed[tail] > summed[deep], "the sum fold prefers the tail on coverage alone"
    assert meaned[tail].item() == pytest.approx(1.5)
    assert meaned[deep].item() == pytest.approx(3.0)
    assert meaned[deep] > meaned[tail], "the mean fold must undo the coverage gradient"
    # The folds disagree about which position scores highest, the
    # width-independent statement. The comparison is by value, not index: at
    # width 7 `argmax` names a plateau's first position, not the source.
    assert summed[tail] == summed.max()
    assert meaned[deep] == meaned.max()
    # And they are not a uniform rescale of one another, unlike the all-global
    # case above, because the coverage gradient is not constant here.
    ratio = summed[tail] / meaned[tail]
    assert not torch.allclose(meaned * ratio, summed, atol=1e-6)


def test_coverage_counts_divide_by_the_covering_layers_not_the_layer_count() -> None:
    """A position inside exactly one sliding window divides by 9, not by 48."""
    globals_ = [
        _synth_row(index, layer_type="global", values=torch.zeros(SYNTH_LOGICAL))
        for index in range(8)
    ]
    # Forty staggered sliding layers: layer i covers the last (i + 1) positions,
    # so position 96-40 is inside exactly one of them.
    slidings = [
        _synth_row(
            8 + index,
            layer_type="sliding",
            values=torch.zeros(SYNTH_LOGICAL),
            window=index + 1,
            offset=SYNTH_LOGICAL - (index + 1),
            physical=index + 1,
        )
        for index in range(40)
    ]
    counts = variants.coverage_counts([*globals_, *slidings])
    assert counts.shape == (SYNTH_LOGICAL,)
    assert counts[SYNTH_LOGICAL - 1].item() == 48.0, "the last position is covered by all 48"
    assert counts[SYNTH_LOGICAL - 40].item() == 9.0, "one sliding layer plus the eight globals"
    assert counts[0].item() == 8.0, "a deep position is covered by the globals alone"
    # General case: position p is covered by the 8 globals plus every sliding
    # layer whose window reaches it, which is 40 - (96 - p) + 1 of them.
    for depth in (5, 20, 33):
        position = SYNTH_LOGICAL - depth
        assert counts[position].item() == 8.0 + (40 - depth + 1)
    # A gap in coverage is a broken capture, not a divide by zero.
    with pytest.raises(RuntimeError, match="do not cover every memory position"):
        variants.coverage_counts(slidings)


def test_the_mean_fold_divides_by_coverage_before_it_pools() -> None:
    """The mean fold divides by coverage before it pools, not after.

    Two globals and one sliding row covering [2, 4) with snap mass 9 at
    position 2 give 3.0 at position 1 under a width-3 pool, not 4.5.
    """
    logical = 4
    from rcc.transforms.select.core.types import LayerDescriptor

    def row(index: int, *, sliding: bool, values: list[float]) -> Any:
        ones = torch.ones(logical, dtype=torch.float32)
        covered = 2 if sliding else logical
        return capture_bank.LayerBankRow(
            descriptor=LayerDescriptor(
                layer_idx=index,
                layer_type="sliding" if sliding else "global",
                logical_memory_length=logical,
                physical_key_length=covered + 1,
                absolute_key_offset=2 if sliding else 0,
                window=2 if sliding else None,
                query_length=1,
            ),
            snap=torch.tensor(values),
            energy=ones.clone(),
            row_energy=ones.clone(),
            support_e={2.0: ones.clone()},
            support_eprime={2.0: ones.clone()},
            support_r={2.0: ones.clone()},
            support_einf=ones.clone(),
            support_rinf=ones.clone(),
            prepool_e2=None,
            prepool_r2=None,
            heads=1,
            query_rows=1,
            kv_groups=1,
            snap_chunks=(torch.tensor(values),),
        )

    rows = [
        row(0, sliding=False, values=[0.0, 0.0, 0.0, 0.0]),
        row(1, sliding=False, values=[0.0, 0.0, 0.0, 0.0]),
        row(2, sliding=True, values=[0.0, 0.0, 9.0, 0.0]),
    ]
    assert variants.coverage_counts(rows).tolist() == [2.0, 2.0, 3.0, 3.0]
    meaned = variants.variant_fold_score(
        rows, "slidenorm", SUPPORT, pool_kernel=3, support_orders=(2.0,)
    )
    assert meaned[1].item() == pytest.approx(3.0), "the fold must divide before the pool"
    assert meaned[1].item() != pytest.approx(4.5), "pool-then-divide would give 4.5 here"
    assert meaned[2].item() == pytest.approx(3.0)
    # The sum fold is untouched by any of this.
    summed = variants.variant_fold_score(
        rows, "slidetail", SUPPORT, pool_kernel=3, support_orders=(2.0,)
    )
    assert summed[1].item() == pytest.approx(9.0)


def test_hybrid_gqa_factors_share_one_bank_and_one_file() -> None:
    """One bank holds rows of two different kv_groups factors.

    Gemma 4 12B banks kv_groups 16 on globals and 2 on sliding rows, each
    folded with its own factor, so only the geometry above them is shared.
    """
    from dataclasses import replace

    from rcc.transforms.select.query_support import capture_bank_format

    values = torch.ones(SYNTH_LOGICAL, dtype=torch.float32)
    tail_values = torch.zeros(SYNTH_LOGICAL, dtype=torch.float32)
    tail_values[SYNTH_LOGICAL - 32 :] = 1.0
    global_row = replace(_synth_row(0, layer_type="global", values=values), heads=16, kv_groups=16)
    sliding_row = replace(
        _synth_row(
            1,
            layer_type="sliding",
            values=tail_values,
            window=32,
            offset=SYNTH_LOGICAL - 32,
            physical=32,
        ),
        heads=16,
        kv_groups=2,
        energy=tail_values.clone(),
        row_energy=tail_values.clone(),
        support_e={2.0: tail_values.clone()},
        support_eprime={2.0: tail_values.clone()},
        support_r={2.0: tail_values.clone()},
        support_einf=tail_values.clone(),
        support_rinf=tail_values.clone(),
    )
    bank = capture_bank.LayerBank(
        scope=BANK_SCOPE_ALL_LAYER,
        support_orders=SYNTH_ORDERS,
        rows=[global_row, sliding_row],
    )
    rows = bank.selected_rows(BANK_SCOPE_ALL_LAYER)
    assert {row.kv_groups for row in rows} == {2, 16}
    payload = capture_bank_format.payload_from_bank(bank)
    assert "kv_groups" not in payload["geometry"]
    reloaded = capture_bank_format.rows_from_payload(capture_bank_format.validate_payload(payload))
    assert {row.kv_groups for row in reloaded} == {2, 16}


def test_heads_and_length_disagreements_still_refuse_and_name_themselves() -> None:
    """Relaxing kv_groups must not relax the load-bearing half of the check."""
    from dataclasses import replace

    values = torch.ones(SYNTH_LOGICAL, dtype=torch.float32)
    base = _synth_row(0, layer_type="global", values=values)
    bad_heads = replace(_synth_row(1, layer_type="global", values=values), heads=2)
    bank = capture_bank.LayerBank(
        scope=BANK_SCOPE_ALL_LAYER,
        support_orders=SYNTH_ORDERS,
        rows=[base, bad_heads],
    )
    with pytest.raises(RuntimeError, match="heads_and_query_rows"):
        bank.selected_rows(BANK_SCOPE_ALL_LAYER)
    bad_rows = replace(_synth_row(1, layer_type="global", values=values), query_rows=2)
    disagreeing = capture_bank.LayerBank(
        scope=BANK_SCOPE_ALL_LAYER, support_orders=SYNTH_ORDERS, rows=[base, bad_rows]
    )
    with pytest.raises(RuntimeError, match="heads_and_query_rows"):
        disagreeing.selected_rows(BANK_SCOPE_ALL_LAYER)


def test_mixed_stream_counts_fold_with_equal_layer_weight() -> None:
    """A mixed-kv_groups bank folds with equal weight per layer.

    An unweighted fold would weight sliding layers 8:1 over globals; `stream_weights` equalizes.
    The rows' moments are deliberately not proportional, which is what an unweighted fold fails on.
    """
    from dataclasses import replace

    tail = torch.zeros(SYNTH_LOGICAL, dtype=torch.float32)
    tail[SYNTH_LOGICAL - 32 :] = 1.0
    ones = torch.ones(SYNTH_LOGICAL, dtype=torch.float32)

    def moments(row: Any, scale_e: float, scale_r: float, mask: torch.Tensor) -> Any:
        return replace(
            row,
            energy=mask.clone(),
            row_energy=mask.clone(),
            support_e={2.0: mask * scale_e},
            support_eprime={2.0: mask * scale_e},
            support_r={2.0: mask * scale_r},
            support_einf=mask.clone(),
            support_rinf=mask.clone(),
        )

    def sliding(values: torch.Tensor) -> Any:
        return _synth_row(
            1,
            layer_type="sliding",
            values=values,
            window=32,
            offset=SYNTH_LOGICAL - 32,
            physical=32,
        )

    global_row = moments(
        replace(_synth_row(0, layer_type="global", values=ones), heads=16, kv_groups=16),
        2.0,
        3.0,
        ones,
    )
    # 8 streams summing to 16 e and 8 r: per-stream means are 2 e and 1 r.
    mixed_sliding = moments(replace(sliding(tail), heads=16, kv_groups=2), 16.0, 8.0, tail)
    reference_sliding = moments(replace(sliding(tail), heads=16, kv_groups=16), 2.0, 1.0, tail)
    for variant in ("slidetail", "slidenorm"):
        mixed = _fold([global_row, mixed_sliding], variant, pool_kernel=1)
        reference = _fold([global_row, reference_sliding], variant, pool_kernel=1)
        assert torch.allclose(mixed, reference, rtol=1e-6, atol=1e-7), variant


def test_the_file_format_refuses_kv_groups_that_do_not_divide_heads() -> None:
    """3 and 17 are positive but impossible under 16 heads.

    The reducer requires kv_groups to divide the head count, so the durable
    format requires it too.
    """
    from dataclasses import replace

    from rcc.transforms.select.query_support import capture_bank_format

    ones = torch.ones(SYNTH_LOGICAL, dtype=torch.float32)
    row = replace(_synth_row(0, layer_type="global", values=ones), heads=16, kv_groups=16)
    bank = capture_bank.LayerBank(
        scope=BANK_SCOPE_ALL_LAYER, support_orders=SYNTH_ORDERS, rows=[row]
    )
    for bad in (3, 17):
        payload = capture_bank_format.payload_from_bank(bank)
        payload["rows"][0]["kv_groups"] = bad
        with pytest.raises(RuntimeError, match="do not divide"):
            capture_bank_format.validate_payload(payload)


# --- The energy votes and the fast capture.


def _assert_capture_parity(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Elementwise-relative, scale-relative, and keep-set parity, fast against eager.

    Squared attention lives far below the absolute 5e-4, and a peak-relative bound is blind to
    rank reversals among small columns, so relative error and a width-1 keep set are binding.
    """
    assert torch.allclose(actual, expected, atol=5e-4)
    scale = float(expected.abs().max())
    assert scale > 0
    assert float((actual - expected).abs().max()) <= 1e-5 * scale
    # every column within 1e-4 of its own magnitude, tiny floor for true zeros
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-30)
    mem_len = int(expected.shape[0])
    for span_size in (1, 32):
        for ratio in (2, 4, 8):
            budget = math.ceil(mem_len / ratio)
            fast_keep = fixed_span_keep(actual, budget=budget, protected=(0,), span_size=span_size)
            eager_keep = fixed_span_keep(
                expected, budget=budget, protected=(0,), span_size=span_size
            )
            assert fast_keep == eager_keep, f"keep drift at ratio {ratio} width {span_size}"


def test_specialist_stream_beats_larger_global_consensus() -> None:
    weights = torch.tensor(
        [
            [[0.5, 0.3]],
            [[0.5, 0.3]],
            [[0.0, 0.3]],
            [[0.0, 0.3]],
        ]
    )
    scores = stream_energy_scores(weights, mem_len=2, kv_groups=2, pool_kernel=1)
    linear = stream_pool_sum_scores(weights, mem_len=2, kv_groups=2, pool_kernel=1)
    global_mass = weights.sum(dim=(0, 1))

    assert global_mass[1] > global_mass[0]
    assert linear[1] > linear[0]
    assert scores[0] > scores[1]
    assert torch.allclose(linear, torch.tensor([0.5, 0.6]))
    assert torch.allclose(scores, torch.tensor([0.25, 0.18]))


def test_w16_heat_bins_keep_layer_and_stream_axes_with_short_tail() -> None:
    values = torch.arange(2 * 18, dtype=torch.float32).reshape(2, 18)
    binned = w16_mean(values)
    assert binned.shape == (2, 2)
    torch.testing.assert_close(binned[:, 0], values[:, :16].mean(dim=-1))
    torch.testing.assert_close(binned[:, 1], values[:, 16:].mean(dim=-1))


def test_pooling_happens_inside_each_stream_before_energy_collapse() -> None:
    weights = torch.tensor(
        [
            [[0.0, 1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0, 0.0]],
        ]
    )
    scores = stream_energy_scores(weights, mem_len=5, kv_groups=1, pool_kernel=3)
    assert torch.equal(scores, torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0]))


def test_gqa_head_and_question_row_duplication_leave_scores_invariant() -> None:
    weights = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
            [[0.5, 0.1, 0.0], [0.1, 0.3, 0.2]],
        ]
    )
    expected = stream_energy_scores(weights, mem_len=3, kv_groups=1, pool_kernel=1)
    duplicated_heads = weights.repeat_interleave(2, dim=0)
    duplicated_rows = weights.repeat_interleave(2, dim=1)

    assert torch.allclose(
        stream_energy_scores(duplicated_heads, mem_len=3, kv_groups=2, pool_kernel=1),
        expected,
    )
    assert torch.allclose(
        stream_energy_scores(duplicated_rows, mem_len=3, kv_groups=1, pool_kernel=1),
        expected,
    )
    row_expected = row_energy_scores(weights, mem_len=3, kv_groups=1, pool_kernel=1)
    assert torch.allclose(
        row_energy_scores(duplicated_heads, mem_len=3, kv_groups=2, pool_kernel=1),
        row_expected,
    )
    assert torch.allclose(
        row_energy_scores(duplicated_rows, mem_len=3, kv_groups=1, pool_kernel=1),
        row_expected,
    )


def test_row_energy_preserves_a_sparse_question_token_signal() -> None:
    weights = torch.tensor([[[0.8, 0.2], [0.0, 0.2]]])
    averaged = stream_energy_scores(weights, mem_len=2, kv_groups=1, pool_kernel=1)
    rowwise = row_energy_scores(weights, mem_len=2, kv_groups=1, pool_kernel=1)

    assert torch.allclose(averaged, torch.tensor([0.16, 0.04]))
    assert torch.allclose(rowwise, torch.tensor([0.32, 0.04]))


def test_row_energy_pools_each_question_row_before_squaring() -> None:
    weights = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ]
    )
    rowwise = row_energy_scores(weights, mem_len=5, kv_groups=1, pool_kernel=3)
    averaged = stream_energy_scores(weights, mem_len=5, kv_groups=1, pool_kernel=3)

    assert torch.equal(rowwise, torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5]))
    assert torch.equal(averaged, torch.full((5,), 0.25))


def test_normalized_energy_gives_each_kv_stream_unit_total_vote() -> None:
    weights = torch.tensor(
        [
            [[0.9, 0.0, 0.0]],
            [[0.0, 0.2, 0.2]],
        ]
    )
    unnormalized = stream_energy_scores(
        weights,
        mem_len=3,
        kv_groups=1,
        pool_kernel=1,
    )
    normalized = normalized_stream_energy_scores(
        weights,
        mem_len=3,
        kv_groups=1,
        pool_kernel=1,
    )

    assert torch.allclose(unnormalized, torch.tensor([0.81, 0.04, 0.04]))
    assert torch.allclose(normalized, torch.tensor([1.0, 0.5, 0.5]))
    assert torch.isclose(normalized.sum(), torch.tensor(2.0))


def test_normalized_row_energy_preserves_row_weighting_inside_each_stream() -> None:
    weights = torch.tensor(
        [
            [[0.8, 0.0], [0.0, 0.2]],
            [[0.0, 0.3], [0.0, 0.3]],
        ]
    )
    normalized = normalized_row_energy_scores(
        weights,
        mem_len=2,
        kv_groups=1,
        pool_kernel=1,
    )

    assert torch.allclose(normalized, torch.tensor([16.0 / 17.0, 18.0 / 17.0]))
    assert torch.isclose(normalized.sum(), torch.tensor(2.0))


@pytest.mark.parametrize(
    "scorer",
    [normalized_stream_energy_scores, normalized_row_energy_scores],
)
def test_normalized_energy_zero_stream_is_finite(scorer: Any) -> None:
    weights = torch.zeros(2, 2, 3)
    scores = scorer(weights, mem_len=3, kv_groups=1, pool_kernel=1)

    assert torch.equal(scores, torch.zeros(3))
    assert torch.isfinite(scores).all()


def test_reducer_uses_float32_before_squaring_small_values() -> None:
    weights = torch.full((2, 1, 4), 1e-4, dtype=torch.float16)
    scores = stream_energy_scores(weights, mem_len=4, kv_groups=1, pool_kernel=1)
    assert scores.dtype == torch.float32
    assert torch.all(scores > 0)


@pytest.mark.parametrize(
    ("weights", "mem_len", "kv_groups", "pool_kernel", "message"),
    [
        (torch.zeros(2, 4), 4, 1, 1, "three-dimensional"),
        (torch.zeros(3, 1, 4), 4, 2, 1, "divisible"),
        (torch.zeros(2, 0, 4), 4, 1, 1, "question row"),
        (torch.zeros(2, 1, 4), 5, 1, 1, "mem_len"),
        (torch.zeros(2, 1, 4), 4, 1, 2, "odd"),
    ],
)
def test_reducer_rejects_invalid_geometry(
    weights: torch.Tensor,
    mem_len: int,
    kv_groups: int,
    pool_kernel: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        stream_energy_scores(
            weights,
            mem_len=mem_len,
            kv_groups=kv_groups,
            pool_kernel=pool_kernel,
        )


def _eager_capture_reference(
    model: Any,
    past: Any,
    judger_ids: torch.Tensor,
    judger_mask: torch.Tensor,
    question_ids: Sequence[int],
    *,
    reducer: Any,
    pool_kernel: int,
    n_sink: int,
) -> tuple[torch.Tensor, bool]:
    """Compose a pure per-layer reducer over a full eager attention replay."""
    mem_len = cache_length(past)
    lo, hi, found = kernels.locate_question(judger_ids, question_ids)
    q_len = int(judger_ids.shape[1])
    base = torch.ones(1, mem_len, dtype=judger_mask.dtype, device=judger_ids.device)
    groups = int(model.config.num_attention_heads) // int(model.config.num_key_value_heads)
    get_decoder = getattr(model, "get_decoder", None)
    backbone: Any = get_decoder() if callable(get_decoder) else model.model
    previous = kernels.use_eager_attention(model)
    try:
        with torch.no_grad():
            out = backbone(
                input_ids=judger_ids,
                past_key_values=past,
                position_ids=torch.arange(mem_len, mem_len + q_len).unsqueeze(0),
                attention_mask=torch.cat([base, judger_mask], dim=1),
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        scores = torch.zeros(mem_len)
        for layer_attention in out.attentions:
            scores += reducer(
                layer_attention[0, :, lo:hi, :],
                mem_len=mem_len,
                kv_groups=groups,
                pool_kernel=pool_kernel,
            ).cpu()
    finally:
        kernels.set_attention_implementation(model, previous)
        if cache_length(past) != mem_len:
            past.crop(mem_len)
    if n_sink > 0 and mem_len > 0:
        scores[: min(n_sink, mem_len)] = scores.max()
    return scores, found


def test_fast_capture_matches_eager_stream_reference(tiny_model: Any) -> None:
    past = prefill(tiny_model)
    judger = random_ids(length=7, seed=411)
    mask = torch.ones_like(judger)
    question = judger[0, 2:5].tolist()

    expected, expected_found = _eager_capture_reference(
        tiny_model,
        past,
        judger,
        mask,
        question,
        reducer=stream_energy_scores,
        pool_kernel=3,
        n_sink=1,
    )
    actual, actual_found = memory_energy_votes(
        tiny_model,
        past,
        judger,
        mask,
        question,
        pool_kernel=3,
        n_sink=1,
    )

    assert actual_found is expected_found is True
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    assert actual.device == expected.device
    assert torch.isfinite(actual).all()
    assert torch.all(actual >= 0)
    _assert_capture_parity(actual, expected)


@pytest.mark.parametrize("kernel_path", ["flash_causal_bias", "dense_bias"])
def test_fast_row_energy_capture_matches_eager_row_reference(
    tiny_model: Any,
    monkeypatch: pytest.MonkeyPatch,
    kernel_path: str,
) -> None:
    if kernel_path == "dense_bias":
        # without the CausalBias the capture routes SDPA through the dense
        # additive mask; the skinny vote softmax is identical on both paths
        monkeypatch.setattr(kernels, "lower_right_causal_bias", lambda *_a, **_k: None)
    past = prefill(tiny_model)
    judger = random_ids(length=7, seed=416)
    mask = torch.ones_like(judger)
    question = judger[0, 2:5].tolist()

    expected, expected_found = _eager_capture_reference(
        tiny_model,
        past,
        judger,
        mask,
        question,
        reducer=row_energy_scores,
        pool_kernel=3,
        n_sink=1,
    )
    actual, actual_found = memory_row_energy_votes(
        tiny_model,
        past,
        judger,
        mask,
        question,
        pool_kernel=3,
        n_sink=1,
    )

    assert actual_found is expected_found is True
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    assert actual.device == expected.device
    assert torch.isfinite(actual).all()
    assert torch.all(actual >= 0)
    _assert_capture_parity(actual, expected)


def test_row_chunked_fold_matches_whole_accumulation() -> None:
    """The chunked fold plus finish equals one whole-row accumulate call."""
    from rcc.transforms.select.query_support.methods.reducers import (
        EnergyAccumulator,
        EnergyChunkState,
        accumulate_capture_energies,
        finish_capture_energies,
        fold_capture_energies,
    )

    torch.manual_seed(7)
    weights = torch.rand(4, 9, 11)

    def make_ctx(energy: int | None, row: int | None) -> EnergyAccumulator:
        return EnergyAccumulator(
            mem_len=9,
            energy_pool_kernel=energy,
            row_energy_pool_kernel=row,
        )

    for kernel in (1, 3):
        whole = make_ctx(kernel, kernel)
        accumulate_capture_energies(whole, weights, kv_groups=2)
        for chunk in (2, 4, 9):
            folded = make_ctx(kernel, kernel)
            state = EnergyChunkState()
            for start in range(0, int(weights.shape[1]), chunk):
                fold_capture_energies(
                    folded,
                    weights[:, start : start + chunk, :],
                    kv_groups=2,
                    state=state,
                )
            finish_capture_energies(folded, state)
            assert folded.energy_sum is not None and whole.energy_sum is not None
            assert folded.row_energy_sum is not None and whole.row_energy_sum is not None
            torch.testing.assert_close(folded.energy_sum, whole.energy_sum)
            torch.testing.assert_close(folded.row_energy_sum, whole.row_energy_sum)


def test_capture_scores_invariant_to_vote_row_chunk(
    tiny_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused capture returns the same three vectors at any row chunk."""

    past = prefill(tiny_model)
    judger = random_ids(length=7, seed=419)
    question = judger[0, 1:5].tolist()
    mask = torch.ones_like(judger)
    baseline = memory_votes_with_energies(
        tiny_model, past, judger, mask, question, pool_kernel=3, n_sink=1
    )
    monkeypatch.setattr(capture_single, "VOTE_ROW_CHUNK", 2)
    chunked = memory_votes_with_energies(
        tiny_model, past, judger, mask, question, pool_kernel=3, n_sink=1
    )
    assert chunked[3] is baseline[3] is True
    for small, whole in zip(chunked[:3], baseline[:3], strict=True):
        torch.testing.assert_close(small, whole)


def test_generalized_moments_ride_one_capture_and_pin_p2_parity(tiny_model: Any) -> None:
    past = prefill(tiny_model)
    judger = random_ids(length=7, seed=420)
    question = judger[0, 1:5].tolist()
    mask = torch.ones_like(judger)

    bundle, found = memory_votes_with_support_moments(
        tiny_model,
        past,
        judger,
        mask,
        question,
        orders=(1.5, 2.0, 3.0, 8.0),
        pool_kernel=3,
        n_sink=0,
    )
    snap, energy, row_energy, energy_found = memory_votes_with_energies(
        tiny_model,
        past,
        judger,
        mask,
        question,
        pool_kernel=3,
        n_sink=0,
    )

    assert found is energy_found is True
    torch.testing.assert_close(bundle.snap, snap)
    torch.testing.assert_close(bundle.e[2.0], energy * SUPPORT_MOMENT_SCALE**2)
    torch.testing.assert_close(bundle.r[2.0], row_energy * SUPPORT_MOMENT_SCALE**2)
    assert set(bundle.e) == {1.5, 2.0, 3.0, 8.0}
    assert torch.isfinite(bundle.rinf).all()
    assert torch.all(bundle.rinf >= bundle.einf)


def test_combined_capture_accumulation_matches_independent_calls_bitwise() -> None:
    from rcc.transforms.select.query_support.methods.reducers import (
        EnergyAccumulator,
        accumulate_capture_energies,
    )

    torch.manual_seed(3)
    weights = torch.rand(4, 3, 11)
    frozen = weights.clone()

    def make_ctx(energy: int | None, row: int | None) -> EnergyAccumulator:
        return EnergyAccumulator(
            mem_len=9,
            energy_pool_kernel=energy,
            row_energy_pool_kernel=row,
        )

    for kernel in (1, 3):
        both = make_ctx(kernel, kernel)
        accumulate_capture_energies(both, weights, kv_groups=2)
        only_energy = make_ctx(kernel, None)
        accumulate_capture_energies(only_energy, weights, kv_groups=2)
        only_row = make_ctx(None, kernel)
        accumulate_capture_energies(only_row, weights, kv_groups=2)

        assert both.energy_sum is not None and only_energy.energy_sum is not None
        assert both.row_energy_sum is not None and only_row.row_energy_sum is not None
        assert torch.equal(both.energy_sum, only_energy.energy_sum)
        assert torch.equal(both.row_energy_sum, only_row.row_energy_sum)
        # the shared-grouping energy branch is the exact pure reducer
        assert torch.equal(
            both.energy_sum,
            stream_energy_scores(weights, mem_len=9, kv_groups=2, pool_kernel=kernel),
        )
        # the fused in-place row branch matches the pure reducer to float32
        torch.testing.assert_close(
            both.row_energy_sum,
            row_energy_scores(weights, mem_len=9, kv_groups=2, pool_kernel=kernel),
        )
        # in-place squaring never touches the caller's attention
        assert torch.equal(weights, frozen)


def test_collect_votes_opt_out_changes_no_returned_vector(tiny_model: Any) -> None:
    past = prefill(tiny_model)
    judger = random_ids(length=7, seed=417)
    mask = torch.ones_like(judger)
    question = judger[0, 2:5].tolist()
    lower = kernels.lower_right_causal_bias(1, 1) is not None

    def capture(collect_votes: bool) -> Any:
        result, context = capture_single.capture_context(
            tiny_model,
            past,
            judger,
            mask,
            question,
            pool_kernel=1,
            spec=CaptureSpec(energy_pool_kernel=7, row_energy_pool_kernel=7),
            observer=None,
            lower_right=lower,
            collect_votes=collect_votes,
        )
        assert result.found is True
        return context

    ctx_on = capture(True)
    ctx_off = capture(False)
    # the flag skips only the unused snap accumulation; both selector
    # accumulators are bit-identical with it on or off
    assert ctx_on.votes is not None
    assert ctx_off.votes is None
    assert torch.equal(ctx_on.energy_sum, ctx_off.energy_sum)
    assert torch.equal(ctx_on.row_energy_sum, ctx_off.row_energy_sum)


def test_fast_capture_restores_state_after_reducer_failure(
    tiny_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    past = prefill(tiny_model)
    judger = random_ids(length=6, seed=415)
    question = judger[0, 1:4].tolist()
    before = [(key.clone(), value.clone()) for key, value in cache_kv(past)]
    length = cache_length(past)
    implementation = tiny_model.config._attn_implementation

    def fail_reducer(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise RuntimeError("forced energy reducer failure")

    monkeypatch.setattr(reducers, "fold_capture_energies", fail_reducer)
    with pytest.raises(RuntimeError, match="forced energy reducer failure"):
        memory_energy_votes(tiny_model, past, judger, torch.ones_like(judger), question)

    assert tiny_model.config._attn_implementation == implementation
    assert capture_single.current_context() is None
    assert cache_length(past) == length
    for (before_key, before_value), (after_key, after_value) in zip(
        before, cache_kv(past), strict=True
    ):
        assert torch.equal(before_key, after_key)
        assert torch.equal(before_value, after_value)


def test_fast_capture_reports_question_fallback(tiny_model: Any) -> None:
    past = prefill(tiny_model)
    judger = random_ids(length=6, seed=413)
    absent = random_ids(length=3, seed=414)[0].tolist()
    scores, found = memory_energy_votes(
        tiny_model,
        past,
        judger,
        torch.ones_like(judger),
        absent,
    )
    assert found is False
    assert scores.shape == (cache_length(past),)
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("span_size", [1, 2, 5, 10, 15, 32, 64])
def test_energy_scores_compose_with_locbench_exact_budget_schedule(span_size: int) -> None:
    scores = torch.linspace(0.0, 1.0, 79).square()
    protected = (0, 76, 77, 78)
    keep = fixed_span_keep(scores, budget=40, protected=protected, span_size=span_size)
    assert keep == sorted(set(keep))
    assert len(keep) == 40
    assert set(protected).issubset(keep)
    # A corrected score protects the same rows, whose row energy is zero.
    row_energy = torch.ones(79)
    row_energy[list(protected)] = 0.0
    corrected = support_corrected_scores(scores, torch.ones(79), row_energy)
    corrected_keep = fixed_span_keep(corrected, budget=40, protected=protected, span_size=span_size)
    assert len(corrected_keep) == 40
    assert set(protected).issubset(corrected_keep)


# --- The support correction.


def test_consensus_rows_leave_q_snap_unchanged() -> None:
    snap = torch.tensor([0.2, 0.4, 0.1])
    energy = torch.tensor([0.03, 0.08, 0.01])

    corrected = support_corrected_scores(snap, energy, energy)

    assert torch.equal(corrected, snap)


def test_sparse_row_support_exactly_corrects_row_dilution() -> None:
    snap = torch.tensor([0.4, 0.2])
    energy = torch.tensor([0.16, 0.04])
    row_energy = torch.tensor([0.32, 0.04])

    corrected = support_corrected_scores(snap, energy, row_energy)

    assert torch.allclose(corrected, torch.tensor([0.8, 0.2]))


def test_zero_energy_uses_neutral_support_and_stays_finite() -> None:
    snap = torch.tensor([0.25, 0.5, 0.75])
    energy = torch.tensor([0.0, 0.2, 0.0])
    row_energy = torch.tensor([0.0, 0.4, 0.0])

    corrected = support_corrected_scores(snap, energy, row_energy)

    assert torch.equal(corrected, torch.tensor([0.25, 1.0, 0.75]))
    assert torch.isfinite(corrected).all()


def test_numerical_floor_preserves_a_row_signal_when_energy_underflows() -> None:
    floor = torch.finfo(torch.float32).tiny
    corrected = support_corrected_scores(
        torch.tensor([0.5 * floor]),
        torch.tensor([0.0]),
        torch.tensor([floor]),
    )

    assert torch.equal(corrected, torch.tensor([0.5 * floor]))


@pytest.mark.parametrize(
    ("snap", "energy", "row_energy", "message"),
    [
        (torch.zeros(2, 2), torch.zeros(4), torch.zeros(4), "one-dimensional"),
        (torch.zeros(3), torch.zeros(4), torch.zeros(4), "same shape"),
        (torch.zeros(4), torch.zeros(3), torch.zeros(4), "same shape"),
    ],
)
def test_support_corrected_scores_reject_invalid_geometry(
    snap: torch.Tensor,
    energy: torch.Tensor,
    row_energy: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        support_corrected_scores(snap, energy, row_energy)


def test_live_presplit_batched_scores_match_serial_rows_and_restore_cache(
    tiny_model: Any,
) -> None:
    realign = build_realign(tiny_model, enabled=False)
    prompts = [random_ids(length, seed=520 + row) for row, length in enumerate((5, 8, 3))]
    padded, _embeds = latent_rollout_batched(
        tiny_model,
        prompts,
        latent_steps=2,
        realign=realign,
    )
    judgers = [random_ids(length, seed=530 + row) for row, length in enumerate((7, 5, 9))]
    masks = [torch.ones_like(item) for item in judgers]
    questions = [
        judgers[0][0, 2:5].tolist(),
        judgers[1][0, 1:3].tolist(),
        judgers[2][0, 4:8].tolist(),
    ]
    splits = split_past_rows(padded.past, padded.mask)
    # Snapshot first: the serial captures below must leave the cache alone too.
    before = [(key.clone(), value.clone()) for key, value in cache_kv(padded.past)]
    mask_before = padded.mask.clone()
    reference = []
    for row, split in enumerate(splits):
        snap, energy, row_energy, found = memory_votes_with_energies(
            tiny_model,
            split,
            judgers[row],
            masks[row],
            questions[row],
            pool_kernel=3,
            n_sink=0,
        )
        corrected = support_corrected_scores(snap, energy, row_energy, alpha=SUPPORT_DEFAULT_ALPHA)
        assert found is True
        assert corrected.shape == snap.shape == (cache_length(split),)
        assert torch.isfinite(corrected).all()
        reference.append((support_corrected_scores(snap, energy, row_energy), found))

    results = capture_query_support_batched_from_past(
        tiny_model,
        padded,
        judgers,
        masks,
        questions,
        pool_kernel=3,
        energy_pool_kernel=3,
        row_energy_pool_kernel=3,
    )

    assert torch.equal(padded.mask, mask_before)
    for (before_key, before_value), (after_key, after_value) in zip(
        before,
        cache_kv(padded.past),
        strict=True,
    ):
        assert torch.equal(before_key, after_key)
        assert torch.equal(before_value, after_value)
    for result, (reference_scores, reference_found) in zip(results, reference, strict=True):
        assert result.found == reference_found
        assert result.energy is not None and result.row_energy is not None
        actual_scores = support_corrected_scores(result.snap, result.energy, result.row_energy)
        torch.testing.assert_close(actual_scores, reference_scores, rtol=1e-4, atol=1e-30)
