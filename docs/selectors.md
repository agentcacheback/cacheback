# Selectors

`transfer` accepts a selector callable and defaults to CacheBack. Budgeting,
payload encoding and delivery stay in transport; the selector chooses positions.
The public type alias is `rcc.selectors.Selector`.

## Custom selection

```python
import torch
from rcc import SenderState, transfer

def recent(sender: SenderState, request_ids: torch.Tensor, budget: int) -> list[int]:
    """Keep the most recent positions within the resolved budget."""
    length = sender.input_embeds.shape[0]
    return list(range(length - budget, length))

transfer(senders, receivers, requests, selector=recent)
```

The function receives one validated sender, its request IDs shaped `[1, tokens]`
on the sender device, and a positive position budget no larger than its state.
Return a sequence of Python integers or a one-dimensional int32/int64 tensor.
Transport rejects empty, duplicate, noninteger, out-of-range or over-budget
selections, then orders the positions by their original source index.

Custom selectors can return fewer positions and define their own retention rules.
The example above keeps recent rows without reserving the first position. RCC
does not run CacheBack or disable gradients around a custom selector. The callable
must leave the sender's model, cache and input rows unchanged. Exceptions propagate
before any delivery; selection is reused across receivers of the same request.

Want to compare your method with CacheBack? See the
[selector challenge and benchmark contribution guide](../CONTRIBUTING.md).

## Package layout

```text
src/rcc/
|-- transport.py          # Routing, budgets, validation and delivery
|-- message.py            # Payload representations
|-- latent.py             # Optional continuous rollout
|-- selectors/
|   |-- __init__.py        # Selector callable type and public exports
|   |-- cacheback.py       # Default selection function
|   |-- fixed_spans.py     # Span selection
|   |-- core/             # Tensor and cache helpers
|   `-- query_support/    # CacheBack attention capture and scoring
|-- vllm.py               # Existing-agent state adapter
`-- capture/              # vLLM pages, connector and weight views
```

## CacheBack default

CacheBack scores each sender's state against the receiver's request. It runs
one query-support capture over a cache copy and leaves the original intact.
The receiver prefills selected input rows into its own cache. Capture requires
the sender state plus request to fit the model's context limit.

The default is the paper's `support-p2-a2` setting, with order `p = 2` and dose
`alpha = 2`. Attention from the request supplies three quantities:

- `snap`: attention mass averaged over request rows and query heads, summed
  over layers, and max-pooled with a seven-position kernel.
- `energy`: attention averaged across request rows before pooling and squaring,
  with grouped-query heads averaged within each KV head.
- `row_energy`: attention pooled and squared separately for each request row,
  then averaged, retaining evidence important to individual request tokens.

CacheBack multiplies `snap` by `(row_energy / energy) ** 2`. The ratio is computed
in float32; positions with both energies zero receive a neutral correction of
one. The public adapter currently supports dense Qwen3. Other model families
and the paper's comparison methods remain in the replication snapshot.

## Position budget

CacheBack ranks fixed spans of `span_size=16` positions (W=16) by their mean score.
To try W=4, configure the selector with the standard library:

```python
from functools import partial
from rcc.selectors import cacheback

transfer(senders, receivers, requests, selector=partial(cacheback, span_size=4))
```

`span_size` must be a positive integer. It changes selection granularity, not
the `ratio` or `budget`. W=4 has CPU coverage; no GPU or quality claim is made.
The first source position and the sender-declared final `latent_steps` positions are
protected and count toward the budget. The last span may contribute a contiguous
partial window to fill the selected count exactly.
Selected rows retain their original order. A budget smaller than the protected
positions is rejected.

The default relative `r4` count is `I + ceil((T - I) / 4)`, where `T` is the
sender's total positions and `I` is its declared inherited prefix length.
Set `ratio=r` to choose another compression factor. The bounded alternative,
`budget=B`, selects `min(B, T)` positions. Both rerank the whole input, so the
inherited prefix changes the relative count without protecting those positions.

Set `SenderState.latent_steps` to the number already produced by the existing
agent. Additional rollout is opt-in through `transfer(..., latent_steps=G)`.
With `"full"` or `"full_with_request"`, selection sees the extended state. With
`"selected"` or `"selected_with_request"`, the selector receives the budget minus
the `G` rows generated afterward. Both timings count the additional rows in the
message allowance.
See [latent rollout](api.md#latent-rollout) for conditioning and receiver-side use.
The paper's forty-step configuration is a replication setting, not an API default.

See [the API guide](api.md) for payload representations and
[the replication guide](replication.md) for baseline selectors, KV transforms,
capture-bank replay and benchmark settings on `paper`.
