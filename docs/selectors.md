# Selectors

`transfer` accepts a selector callable and defaults to CacheBack. Budgeting,
payload encoding and delivery stay in transport; the selector chooses positions.
The public type alias is `rcc.selectors.Selector`.

## Built-in selectors

| Selector | Scores | Options |
| --- | --- | --- |
| `cacheback` (default) | Request attention with the support correction | `span_size=16` |
| `qsnap` | Mean request attention per position (CacheBack without the correction) | `span_size=16` |
| `chunkkv` | ChunkKV: the sender's last rows attend to earlier ones; request-agnostic | `chunk_size=16`, `window_size=32`, `kernel_size=5` |

```python
from functools import partial

from rcc import transfer
from rcc.selectors import chunkkv, qsnap

await transfer(senders, receivers, requests, selector=qsnap)
await transfer(senders, receivers, requests, selector=partial(chunkkv, window_size=64))
```

All three run one attention capture over a private cache copy, leaving the
sender intact; on Hugging Face models load the sender with `sdpa` or `eager`
attention, because FlashAttention backends are not supported during capture.
They score global layers only, keep the first position and
[latent tail](api.md#relative-and-bounded-budgets), and fill the budget with
whole spans. CacheBack and QSnap need the sender state plus request to fit the
model's context limit. ChunkKV averages scores within `chunk_size` chunks,
always keeps its observation window and needs a sender longer than
`window_size`. QSnap and ChunkKV match eager-attention references in the CPU
suite; they carry no GPU validation or quality claim here. The paper's measured
baselines stay on `paper`.

## Custom selection

```python
import torch
from rcc import SenderState, transfer

def recent(sender: SenderState, request_ids: torch.Tensor, budget: int) -> list[int]:
    """Keep the most recent positions within the resolved budget."""
    length = sender.input_embeds.shape[0]
    return list(range(length - budget, length))

await transfer(senders, receivers, requests, selector=recent)
```

The function receives one validated sender, its request IDs shaped `[1, tokens]`
on the sender device, and a positive position budget no larger than its state.
Return a sequence of Python integers or a one-dimensional int32/int64 tensor.
Transport rejects empty, duplicate, noninteger, out-of-range or over-budget
selections, then orders the positions by their original source index.

Custom selectors can return fewer positions and define their own retention
rules; `recent` above does not reserve the first position. RCC does not run
CacheBack or disable gradients around a custom selector. The callable must leave
the sender's model, cache and input rows unchanged. Exceptions propagate before
any delivery; selection is reused across receivers of the same request.

To compare a method with CacheBack, see [Contributing](../CONTRIBUTING.md).

## Package layout

```text
src/rcc/
|-- __init__.py           # Public API
|-- __main__.py           # `python -m rcc doctor`
|-- transport.py          # Routing, budgets, validation and delivery
|-- message.py            # Payload representations
|-- selection.py          # Opt-in selection records and HTML view
|-- latent.py             # Built-in continuous rollout
|-- reasoning.py          # Callable reasoning and output validation
|-- checkpoint.py         # Safetensors state snapshots
|-- diagnostics.py        # Setup checks behind `rcc.check`
|-- hf.py                 # Hugging Face bindings and Agent
|-- vllm.py               # Existing-agent state adapter
|-- _async.py             # Cancellation-safe worker calls
|-- _cache.py             # Cache copying and decoder access
|-- selectors/
|   |-- __init__.py       # Selector type and built-in selectors
|   |-- cacheback.py      # cacheback, cacheback_scores, support_score
|   |-- qsnap.py          # qsnap, qsnap_scores
|   |-- chunkkv.py        # chunkkv, chunkkv_scores
|   |-- fixed_spans.py    # Span schedule and shared protection rules
|   |-- _capture.py       # Private attention capture over a cache copy
|   `-- _support.py       # Request attention statistics for CacheBack and QSnap
`-- capture/              # vLLM pages, connector and weight views
```

## CacheBack default

CacheBack uses the paper's `support-p2-a2` setting, with order `p = 2` and dose
`alpha = 2`. Attention from the request supplies three quantities:

- `votes`: attention mass averaged over request rows and query heads, summed
  over layers, and max-pooled with a seven-position kernel.
- `energy`: attention averaged across request rows before pooling and squaring,
  with grouped-query heads averaged within each KV head.
- `row_energy`: attention pooled and squared separately for each request row,
  then averaged, retaining evidence important to individual request tokens.

`support_score` multiplies `votes` by `(row_energy / energy) ** 2`. The ratio is
computed in float32; positions with both energies zero receive a neutral
correction of one. The public adapter supports dense Qwen3. Other model families
remain in the replication snapshot. The capture keeps the arithmetic of the
validated implementation, so CacheBack scores are bit-identical to it.

## Span width and budget

CacheBack ranks fixed spans of `span_size=16` positions (W=16) by mean score.
For W=4:

```python
from functools import partial
from rcc.selectors import cacheback

await transfer(senders, receivers, requests, selector=partial(cacheback, span_size=4))
```

`span_size` must be a positive integer. It changes selection granularity, not
the `ratio` or `budget`. W=4 has CPU coverage; no GPU or quality claim is made.
The last span may contribute a partial window to fill the selected count
exactly, and selected rows keep their original order.

The API guide covers [budgets](api.md#relative-and-bounded-budgets),
[reasoning](api.md#reasoning) and [representations](api.md#representations).
The paper's forty-step reasoning configuration is a replication setting, not an
API default. Baseline selectors, KV transforms, capture-bank replay and
benchmark settings are on `paper`; see [the replication guide](replication.md).
