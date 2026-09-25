# State-transfer API

```python
rcc.transfer(sender, receiver, request, ratio=4, representation="embeddings")
```

This is a synchronous, in-process handoff between existing agents. It returns
`None`. It does not own an agent loop, submit model requests, load weights or
generate answers. Your callback can deliver to a queue or your own network
transport. There is no implicit server or wire serialization protocol.

## Lists and requests

Each of `sender`, `receiver` and `request` accepts one item or a sequence.
Every receiver is called once per request with a `Delivery` containing:

- `request`: the original request string or token-ID tensor.
- `messages`: a tuple of selected `Message` objects in sender-list order.

All senders contribute to each delivery. Requests broadcast to all receivers;
there is no positional matching between the two lists. For specialized routes,
make separate calls. A string request uses each sender's tokenizer; a token-ID
request must be a nonempty int64 tensor shaped `[1, tokens]` meaningful to all
senders. Empty lists are rejected.

All messages are prepared before callbacks begin. Preparation failures deliver
nothing. A callback exception propagates immediately; earlier callbacks may
already have received their delivery. There are no automatic retries. Shared
payload tensors must be treated as read-only; clone before mutating them.

## Relative and bounded budgets

Relative (`rX`) is the default. With neither option supplied, RCC uses `r4`.
Both options accept positive integers and apply independently per sender,
per request. Supplying both `ratio` and `budget` is an error.

```python
transfer(senders, receivers, requests)               # Relative r4 (default).
transfer(senders, receivers, requests, ratio=16)     # Relative r16.
transfer(senders, receivers, requests, budget=1024)  # Bounded at 1,024 positions.
```

For total input length `T` and inherited prefix length `I`, the paper's rules
resolve the position budget passed to the selector:

| Mode | Position budget |
| --- | --- |
| Relative `ratio=r` | `I + ceil((T - I) / r)` |
| Bounded `budget=B` | `min(B, T)` |

For a fresh sender, `I = 0`, so relative compression keeps `ceil(T / r)` positions.
In a chain, set `SenderState.inherited_positions` to the number of leading rows
received from prior agents. Newly added prompt and current latent rows make up
`T - I`. Relative memory grows by the compressed new rows at each hop; bounded
memory stops growing when the cap binds. RCC does not infer or track chain history.

With CacheBack, both modes rerank inherited and new positions together. Inherited rows can be
dropped; they are not protected merely because they arrived in an earlier message.
The first position and the current `latent_steps` tail are always kept and count
toward the selected total. A count too small to hold them raises an error.
CacheBack uses support-p2-a2 and spans of 16; the last selected span can be partial
to fill the exact count. Additional latent generation is opt-in.

## Latent rollout

```python
transfer(senders, receivers, requests, latent_steps=40,
         latent_context="selected_with_request")
```

| `latent_context` | Timing | Inputs for latent generation |
| --- | --- | --- |
| `"full"` (default) | Before selection | Existing prompt/state |
| `"full_with_request"` | Before selection | Existing prompt/state plus receiver request |
| `"selected"` | After selection | Selected input rows |
| `"selected_with_request"` | After selection | Selected input rows plus receiver request |

`latent_steps=0` is the default and disables additional generation.
`SenderState.latent_steps` describes an existing latent tail. The transfer argument
adds steps to it; it does not rerun or replace existing thoughts.
The prompt can already contain the task question. Conditioning controls only
whether the receiver request is additionally supplied during rollout, separately
from the request CacheBack uses for selection. RCC uses the request exactly as
provided, without inserting a chat template or reading a receiver's private prompt.

Full-context prompt-only rollout is reused across requests. Request-conditioned
or selected-context rollout runs separately per sender/request, and all receivers
of that request share the result. Selected context contains only the chosen rows;
discarded prompt instructions are not silently restored. Selection itself may
already be request-conditioned with `latent_context="selected"`.

For `G` additional steps, use `T + G` in the budget formulas above. Both timings
therefore have the same total allowance. With either full-context mode, the
selector sees the extended state; CacheBack protects the existing and new latent
tail. Custom selectors retain their own protection rules. With either
selected-context mode, RCC reserves `G` positions, passes the remaining budget
to the selector, then appends `G`
latent rows. Insufficient room for selected context is an error. For example,
80 source positions and 4 new steps at `r4` allow 21 transmitted positions;
selected-context rollout keeps 17 source positions and appends 4 latent rows.

Rollout uses the paper's Qwen rule: feed the final hidden row back as a continuous
input, normalized to the mean input-embedding norm, without the optional
output-to-input realignment map. It uses a private cache and leaves the original
state unchanged. Conditioning request tokens are not included in the returned
state or payload. When conditioning is enabled, RCC replays the generated latent
rows over the original prefix to build a cache consistent with the retained rows.
Rollout and any selection capture must fit the model context limit.

For vLLM senders, these forwards use the HF view of the resident weights while
the engine is idle. These public rollout options have CPU coverage under both
Transformers pins; live GPU validation and benchmark comparisons are outstanding.

### Receiver-side steps

Use the same helper inside the receiver callback, after prefilling its handoff
and any receiver-owned prompt into a `SenderState`:

```python
from rcc import rollout

receiver_state = rollout(receiver_state, steps=40)
```

To additionally condition on the receiver request, pass `request=delivery.request`.
The helper returns a new state with the extended cache and exact input rows;
the receiver continues from that cache in its own loop. Its additional local
rows consume model context, not transmission budget. `steps=0` returns the
original state without a model forward. Dense Qwen3 is currently supported.

## Selectors

CacheBack is the default callable, exported as `rcc.selectors.cacheback`:

```python
from rcc.selectors import cacheback

transfer(senders, receivers, requests, selector=cacheback)
transfer(senders, receivers, requests, selector=my_selector)
```

CacheBack uses W=16 by default. Pass `selector=partial(cacheback, span_size=4)`
(with `partial` from `functools`) to try W=4 without changing the handoff budget.
See [span width](selectors.md#position-budget) for the configuration and limits.

The contract is `selector(sender, request_ids, budget) -> positions`.
The selector runs once per sender per request; its result is reused for all
receivers. It receives validated state, tokenized request IDs on the sender's
device, and the resolved relative or bounded budget.

Return a nonempty sequence of Python integers or a one-dimensional int32/int64
tensor. Positions must be unique, inside the sender's input rows, and no more
numerous than the budget. Transport sorts them into source order before encoding.
Invalid output or a selector exception aborts preparation before any delivery.

Custom selectors may use less than their budget and choose their own protected
positions. CacheBack alone applies the paper's first-position and current-latent
protection, full-budget span selection, and attention capture. Custom callbacks
run in the caller's gradient mode and must leave sender state unchanged.
Changing the selector does not expand the runtime adapter's supported models.
See [the selector guide](selectors.md) for a complete custom function and layout.

## vLLM default

Use dense Qwen3 with **vLLM 0.11.1 / Torch 2.9.0 / Transformers 4.57.1**.
Install `requirements/vllm.txt` in a separate Linux CUDA environment, then the
package. This is the paper's established Qwen engine path. The new public
adapter still needs a live GPU smoke test before release.

Enable capture when creating your existing engine:

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    tensor_parallel_size=1,
    enable_prefix_caching=False,
    enforce_eager=True,
    kv_transfer_config={
        "kv_connector": "RCCCaptureConnector",
        "kv_connector_module_path": "rcc.capture.connector",
        "kv_role": "kv_both",
    },
)
```

The adapter supports one in-process, unquantized dense Qwen3 engine with a
single KV cache group and full prompt state. It does not support tensor
parallel, sliding-window, hybrid or quantized caches. Capture copies selected
request pages; it does not write into the engine's cache pool.

In the agent's existing submission loop, register the request before stepping
the engine. On 0.11.1 the capture ID is the ID supplied to `add_request`:

```python
from rcc.vllm import request_capture, sender_from_vllm

request_capture(request_id)
llm.llm_engine.add_request(request_id, prompt, sampling_params)
# Your existing loop drives llm.llm_engine.step() until prompt prefill completes.
sender = sender_from_vllm(llm, request_id, prompt_token_ids)
```

`prompt_token_ids` is the exact one-dimensional int64 prompt tensor, including
any chat-template tokens. The adapter captures the completed **prompt**; later
generated tokens are not included. It shares the live engine's weights through
an HF-shaped view, so keep that engine alive and idle during rollout and selection. Call
`discard_capture(request_id)` on canceled requests or unused captures to free
request-scoped tensors. A successful `sender_from_vllm` releases the registry
entry after constructing the state. Failed validation leaves it available.
If this token-ID prompt begins with inherited positions, pass their count as
`inherited_positions=...` to `sender_from_vllm` for relative chain budgeting.

For an agent that already exposes the paper's HF-shaped model, cache and input
rows, construct `SenderState` directly as below. This also accommodates its
existing latent steps without rerunning them.

### Unstable vLLM 0.26.0

Install `requirements/vllm-unstable.txt` in a **different environment**.
This public adapter is experimental and **lacks live GPU testing**. Both the
engine connector and state adapter require explicit opt-in:

```python
# Add to kv_transfer_config when constructing the engine:
kv_transfer_config["kv_connector_extra_config"] = {"allow_unstable": True}
```

For 0.26.0, also pass `attention_backend="FLASH_ATTN"` to `LLM`; its attention
configuration replaces the older environment variable. Keep eager execution,
prefix caching disabled and tensor parallel size one.

Unlike 0.11.1, 0.26.0 returns a randomized internal request ID. Register that
returned ID after submission and before the first engine step:

```python
internal_id = llm.llm_engine.add_request(request_id, prompt, sampling_params)
request_capture(internal_id)
# Your agent drives its existing loop.
sender = sender_from_vllm(llm, internal_id, prompt_token_ids, allow_unstable=True)
```

Use the internal ID for cleanup too. The `paper` branch separately preserves
its original 0.26.0 Gemma, Ministral and Nemotron runtimes and measured settings.

## Hugging Face state

Bind the state your existing agent already computed:

```python
from rcc import SenderState, transfer

sender = SenderState(
    model=sender_model,
    past_key_values=sender_cache,
    input_embeds=sender_input_rows,
    tokenizer=sender_tokenizer,
)
transfer(sender, receiver_inbox.append, "Find the owner and launch date.")
```

`input_embeds` is `[positions, hidden]`: the exact rows used to build the
unpadded single-sequence cache, on the model's device and dtype. These are input
embeddings, not final hidden states. Put the model in eval mode. CacheBack
runs a request against a cache copy and leaves the original cache intact.
Keep the sender idle during transfer: the view is not a snapshot and attention
capture changes process-global attention dispatch temporarily. Concurrent CacheBack
calls and RCC rollout forwards share a lock; separate processes are required for isolation.

For a sender with continuous latent reasoning, include those exact input rows
and set `latent_steps=G` so CacheBack protects its final G rows. Optional `token_ids` is an
int64 vector aligned with all input rows, using `-1` for continuous positions.
For a chain, also set `inherited_positions` to the carried prefix length; it must
not overlap the current latent tail. Inherited latent rows belong to that prefix,
so they can be discarded during reranking. This state binding also supports
chains whose inherited input rows cannot be reconstructed from token IDs alone.
See [quickstart.py](../examples/quickstart.py) for native prefill plus delivery.

## Receiver continuation

The callback receives state. Your agent decides when and how to resume:

```python
delivery = receiver_inbox.pop(0)
inputs = delivery.for_hf(receiver_model.get_input_embeddings().weight)
with torch.inference_mode():
    received = receiver_model.model(**inputs, use_cache=True)
```

This creates fresh receiver KV state. It does not splice in foreign KV.
`delivery.request` is metadata; it is **not automatically appended** to the
handoff rows. Your agent adds its instructions, request and continuation using
its own prompt format. Multi-sender message boundaries remain available in
`delivery.messages`; the convenience methods concatenate them in sender order.

For a vLLM receiver configured with `enable_prompt_embeds=True`:

```python
prompt = delivery.for_vllm(matching_embedding_weight)
# Your receiver's own loop submits this prompt when ready.
```

This returns `{"prompt_embeds": cpu_rows}`. See vLLM's
[prompt embedding documentation](https://docs.vllm.ai/en/v0.26.0/features/prompt_embeds/).
Both paths require compatible representations. Matching hidden width alone
does not establish compatibility: use the same checkpoint and tokenizer unless
you have independently validated a cross-model mapping. Opaque text-only APIs
cannot expose the required sender state. A framework callback alone cannot
make an incompatible model understand these rows.

## Representations

| Option | Payload | Status |
| --- | --- | --- |
| `embeddings` (default) | All selected input rows | Paper handoff representation |
| `token_ids+continuous` | Source token IDs, continuous vectors, original order | Experimental; untested in end-to-end GPU runs |

Mixed messages use `-1` in `Message.token_ids` to mark each continuous row.
`Message.materialize(weight)` reconstructs the original order using a matching
receiver embedding table. The sender checks that selected discrete IDs
reproduce its input rows before sending. The receiver checks shape and ID
bounds; the application must ensure checkpoint/tokenizer compatibility.

`Message.positions` counts reconstructed positions. `Message.nbytes` counts
tensor bytes, excluding Python objects, metadata and transport framing.
