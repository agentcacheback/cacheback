# State-transfer API

```python
await rclc.transfer(sender, receiver, request, ratio=4, representation="embeddings")
```

`transfer` is an async, in-process handoff between existing agents. It returns
`None`; await it before consuming the delivery. It does not own an agent loop,
load weights or generate answers. A bound sender prepares uncached state on
demand, including a vLLM capture request. A receiver callback can deliver to a
queue or your own network transport; there is no server or wire protocol.

Await it inside an async function or notebook; a script can wrap its entry point
in `asyncio.run(main())`, as the [quickstart](../examples/quickstart.py) does.
`rclc.transfer_sync(...)` takes the same arguments, runs the same pipeline
including async callbacks, and requires a thread without a running event loop.

## Concurrency

- Model work runs in worker threads. State snapshots and selection serialize;
  reasoning callbacks can await other work without holding that lock. Attention
  forwards, including selector capture and RCLC reasoning, share a process-wide
  lock. HF and in-process vLLM use the same path.
- Keep participating agents, histories and engines idle, unchanged and
  exclusively owned until each bind, transfer, append, update, save or load
  finishes. A sender is read in place, not snapshotted, and attention capture
  temporarily changes process-global attention dispatch. A vLLM engine must
  stay alive while agents or senders use its shared-weight view.
- Native generation and `update()` block and stay under your agent loop's
  control. Keep native model work in one serialized loop, not overlapping
  transfer or append on the same model. Async transfer does not make
  `LLM.generate` an AsyncLLM endpoint.
- Use separate processes for isolation.

Receivers can be bound agents, ordinary callbacks or async callbacks. Callbacks
run in request order, then receiver-list order; async callbacks are awaited.
Keep ordinary callbacks short, or move blocking work into your own worker.

Cancellation during a worker operation waits for it to restore model state, then
raises `CancelledError` without delivering anything; awaited reasoning callbacks
receive it directly. A callback exception propagates immediately. Cancellation
or failure during a callback leaves earlier deliveries intact, with no rollback
or retry. Participating state can be reused once cancellation completes.

## Bind an existing agent

```python
import rclc

agent_x = rclc.bind(model, tokenizer, messages=history_x, backend="hf")
agent_y = rclc.bind(model, tokenizer, messages=history_y, backend="hf")
await rclc.transfer(agent_x, agent_y, question)
inputs = agent_y.pop()
output = model.generate(**inputs, return_dict_in_generate=True)
agent_y.update(inputs=inputs, generation=output)
await rclc.transfer(agent_y, agent_x, follow_up)  # Send the continuation back.
```

`bind` is the constructor alias for `rclc.Agent`; an agent's position in
`transfer` sets its role. Lists can mix agents and `SenderState` objects. Use
`backend="hf"` for a loaded HF model or `backend="vllm"` for an in-process
engine configured as in [vLLM default](#vllm-default).

The binding accepts `messages`, or explicit `prompt`/`past_key_values`, or
`generation`. Messages supply receiver history even when explicit sender state is
given. With messages alone, sender prefill is lazy and later sends reuse the
cached history: HF prefills only an appended suffix, vLLM re-captures when
history changes, and editing an earlier token causes a fresh prefill. Explicit
state stays current until `append` or `update` changes it. `max_new_tokens=128`
is the default output reservation and `context_limit` defaults to the model's;
both feed [receiver budget fitting](#receive-and-fit-the-handoff).

### Continue, append and resume

Pass `update` the exact dictionary from the agent's latest `pop()` or awaited
`inputs()`, and generate from it without adding `input_ids`. HF returns new token
IDs for these continuous inputs; RCLC keeps the exact input vectors, appends those
tokens and completes any uncached suffix on a private cache copy. Received
positions count as inherited state for relative budgets. Known token IDs from
the receiver prompt, history and new tokens are kept; only received continuous
rows are `-1`, so later mixed payloads send those positions as IDs.

A second `pop()` raises until `update()` records the continuation, so an agent
cannot send stale state. Failed updates and pops preserve the previous state,
pending inputs and queue. For answers that will not be sent onward,
`pop(discard_pending=True)` replaces the pending continuation. RCLC never appends
answers to your chat history, and repeated requests stay separate queued
handoffs with their own prompt snapshots.

For ordinary HF token-input generation use `agent.update(generation=output)`;
for a current prompt/cache use `agent.update(exact_ids, past_key_values=cache)`.
`agent.update()` with no arguments discards explicit state and returns to the
message history. `agent.sender_state()` exposes the low-level state. The limits
of the [lower-level helpers](#lower-level-hugging-face-helpers) apply.

```python
await agent_y.append(rendered_tool_result)
await rclc.transfer(agent_y, agent_x, follow_up)       # Or continue locally:
inputs = await agent_y.inputs(max_new_tokens=64)
```

`append` takes only new rendered text or exact `[1, tokens]` IDs, including any
role delimiters your loop needs; it applies no chat template, generates nothing
and leaves `messages` alone. Received vectors and inherited counts survive. HF
extends a private cache copy; vLLM re-prefills the combined continuous input.
Record pending generation first. Invalid input, context overflow or cancellation
keeps the previous state. After a received continuation, appending to `messages`
alone does not update continuous state: use `append`, or `update()` to reset.

`inputs()` prepares the retained state for native generation (`llm.generate` for
vLLM), checks output space and records pending inputs like `pop()` without
consuming queued handoffs. Record the result before preparing another
continuation, saving or sending state.

### Save and restore

```python
await agent_y.save("agent.safetensors")
restored = rclc.bind(model, tokenizer, max_new_tokens=64)
await restored.load("agent.safetensors")
```

Snapshots hold input vectors, aligned token IDs and inherited/latent counts, not
weights, KV caches, chat history, queued handoffs or sampling settings. Bind the
same checkpoint and dtype, with your receiver history and settings, before
loading; both backends restore by re-prefill. Save and load check a hash of the
weights, forward configuration and tokenizer vocabulary. The weight hash is
cached per model object, so bind a new model object instead of editing weights
in place.

Record pending generation and consume queued handoffs before `save` or
`load`. Save replaces the file atomically; cancellation waits for the write,
which may already have completed. Load validates rows and metadata before
prefill; a failed or cancelled load keeps the previous state. Treat snapshot
files as private model input.

### Inspect, check and view selection

```python
agent_y.inspect()       # Queue, per-sender positions, payload bytes and remaining space.
rclc.check(agent_y)      # Signature: check(agent=None, *, backend=None, allow_unstable=None)
await rclc.transfer(agent_x, agent_y, question, record_selection=True)
report = agent_y.inspect(selection=True)
html = agent_y.selection_html()   # display(HTML(html)) in a notebook, or write to a file.
```

- `inspect()` runs no forward pass, consumes nothing and returns a
  JSON-serializable report. `remaining_positions` excludes the prompt and
  reserved output; negative means it will not fit. `cached_positions` describes
  prepared state, not changed chat history.
- `rclc.check` reports versions, GPU availability and bound-engine configuration.
  With an agent, `backend` and `allow_unstable` come from it; a conflicting value
  raises `ValueError`. `python -m rclc doctor` prints the same checks as JSON with
  fixes and exits nonzero on issues; add `--backend vllm` and, for 0.26.0,
  `--allow-unstable`. Neither downloads weights or establishes GPU correctness.
- `agent.engine` is the vLLM engine or `None`; `agent.model` is the HF model or
  shared-weight view.

Selection recording is off by default. It adds CPU bookkeeping but no inference,
works with both representations and custom selectors, and keeps decoded spans,
including omitted text, in local metadata outside payload bytes. Inspect before
`pop()` consumes the delivery.

- The HTML highlights retained spans in green; `selection_html(request_index=1)`
  shows the second queued request.
- Each entry in `report["deliveries"][0]["selection"]` has one sender's
  `indices`, `source_positions`, fitted `budget`, `spans` and
  `added_latent_positions`, zero-based in the state the selector saw, with
  exclusive span `stop`. Full-context reasoning is already in that state;
  post-selection reasoning has no source index.
- Low-level receivers use `delivery.selection_html()` and
  `message.selection.inspect()`.
- Text is decoded from token IDs and can differ in formatting or split Unicode
  characters; indices are authoritative. Latent, other continuous and unknown-ID
  positions are labelled separately, and provenance covers this handoff only.

## Lower-level Hugging Face helpers

```python
from rclc import HFReceiver, sender_from_hf, transfer

sender = sender_from_hf(model, tokenizer, sender_messages)
receiver = HFReceiver(model, tokenizer, messages=receiver_history,
                      max_new_tokens=128, context_limit=4096)
await transfer(sender, receiver, "Who owns Cedar?")
tokens = model.generate(**receiver.pop(), do_sample=False)
```

`bind` is built on these. They take a loaded dense Qwen3 model in eval mode,
unpadded single sequences and no beam search. `receiver_history` holds system
instructions and prior turns, not the new request; the receiver appends the
request and renders the assistant prefix with thinking disabled. RCLC does not
mutate your history or run tools.

Use the same checkpoint, weights and tokenizer at both ends. `transfer` rejects
known architecture, checkpoint name/revision and vocabulary mismatches for
`HFReceiver` before selection. Metadata cannot detect modified weights, so local
copies should keep the checkpoint name and revision. Custom callbacks own their
checks. Other families, quantized and sharded models are not validated.

`sender_from_hf` accepts chat messages (templated with no new assistant prefix),
rendered text (no special tokens added) or int64 IDs shaped `[1, tokens]`.
Without a chat template, pass text or IDs; an agent bound without a tokenizer
needs token IDs and rejects text with `ValueError`. Other forms:

```python
sender = sender_from_hf(model, tokenizer, generation=output)  # After model.generate
sender = sender_from_hf(model, tokenizer, exact_ids, past_key_values=cache)
```

`generation` needs `return_dict_in_generate=True` output from a token prompt with
greedy decoding or sampling; `output.sequences` must include the whole prompt.
Beam outputs, ambiguous combinations and `inputs_embeds` generations are
rejected (use `SenderState` for those rows). Without a cache the helper prefills
once; a complete cache is reused; a prefix cache is cloned and extended with the
missing tokens, such as the final generated token or appended tool results. The
caller's cache is unchanged and a longer cache is rejected. Length cannot prove
content identity, so pass exact IDs: re-rendering can change the prefix.
`SenderState` is the route for continuous latent rows and inherited messages.
The [existing-agent example](../examples/existing_agents.py) reuses a
conversation cache and continues with native generation.

### Receive and fit the handoff

`context_limit` cannot exceed the model's. `HFReceiver` reserves one output
position unless `max_new_tokens` is set. `pop()` passes that allowance to
`generate`, so do not pass it again.

Before selection, RCLC renders each receiver's history plus request. Handoff
space is the context limit minus those positions and the output allowance. The
relative or bounded budget stays each sender's maximum; if the sum fits, budgets
are unchanged. Otherwise RCLC reserves the built-in selectors'
[protected positions](#relative-and-bounded-budgets) and shares the rest
equally, redistributing unused shares from short senders. Custom selectors get
a one-position floor and enforce their own protection. Post-selection reasoning reserves its declared rows for every
selector. Remainders follow sender-list order, and inherited positions count in
the relative budget.

Every receiver of a request, including generic callbacks, gets the same
messages, so the tightest receiver sets the limit; different requests can get
different budgets. All requests are planned before selection; impossible budgets
deliver nothing. The sender still needs its own space for the request and any
reasoning.

Each receiver snapshots its rendered history IDs with the request, so later
history changes only affect future transfers. `pop()` puts received rows before
that prompt and starts a fresh prefill without splicing a receiver cache.
`len(receiver)` counts queued requests; an empty inbox raises `IndexError`. A
successful pop consumes one delivery; keep the inputs to retry generation.
`pop(prompt=..., max_new_tokens=64)` overrides the queued prompt or allowance,
checked against the context limit without reselecting. Failed preparation
leaves the delivery queued. A tensor request with bound chat history is
rejected because its framing is unknown.

## Lists and requests

`sender`, `receiver` and `request` each accept one item or a sequence. Every
receiver is called once per request with a `Delivery` of `request` (the original
string or token-ID tensor) and `messages` (selected `Message` objects in
sender-list order). Requests broadcast to all receivers with no positional
matching; make separate calls for specialized routes. A string request uses each
sender's tokenizer; a token-ID request must be a nonempty int64 `[1, tokens]`
tensor meaningful to all senders. Empty lists are rejected.

All messages are prepared before callbacks begin, so preparation failures
deliver nothing. [Callback failures](#concurrency) do not roll back. Treat
shared payload tensors as read-only.

## Relative and bounded budgets

Relative (`rX`) is the default; with neither option, RCLC uses `r4`. Both take
positive integers and apply per sender, per request. Supplying both is an error.

```python
await transfer(senders, receivers, requests)               # Relative r4 (default).
await transfer(senders, receivers, requests, ratio=16)     # Relative r16.
await transfer(senders, receivers, requests, budget=1024)  # Bounded at 1,024 positions.
```

For total input length `T` and inherited prefix length `I`, the paper's rules
resolve the position budget passed to the selector:

| Mode | Position budget |
| --- | --- |
| Relative `ratio=r` | `I + ceil((T - I) / r)` |
| Bounded `budget=B` | `min(B, T)` |

A fresh sender has `I = 0`. In a chain, `SenderState.inherited_positions` counts
leading rows received from prior agents; new prompt and current latent rows make
up `T - I`. Relative memory grows by the compressed new rows at each hop; bounded
memory stops growing when the cap binds. Bound agents record inherited positions
in `update(inputs=..., generation=...)`; `SenderState` callers supply the count.

The built-in selectors rerank inherited and new positions together in both
modes, so inherited rows can be dropped. They always keep the first position and
the current `latent_steps` tail, which count toward the total. A budget too small
for each selector's required positions plus reserved reasoning rows raises
`ValueError` for any receiver type.

## Reasoning

Reasoning is off by default. Pass a callable, with settings bound by
`functools.partial`, to append continuous rows to each sender before sending:

```python
from functools import partial

await rclc.transfer(senders, receivers, requests,
                   reasoning=partial(rclc.latent_mass, steps=40), budget=128)
await rclc.transfer(senders, receivers, requests,
                   reasoning=partial(rclc.latent_mass, steps=40),
                   reasoning_context="selected_with_request", reasoning_budget=40)
```

`rclc.latent_mass` is the paper's Qwen rule: feed the final hidden row back as a
continuous input, normalized to the mean input-embedding norm. Call it directly
with `await rclc.latent_mass(state, steps=50)`, or `latent_mass_sync` to block. It
runs sequential steps in a worker on a private cache and leaves the state
unchanged; cancellation waits for the rollout. Request tokens used for
conditioning are not kept; RCLC replays the new rows over the original prefix.

| `reasoning_context` | Timing | Inputs for reasoning |
| --- | --- | --- |
| `"full"` (default) | Before selection | Existing prompt/state |
| `"full_with_request"` | Before selection | Existing prompt/state plus receiver request |
| `"selected"` | After selection | Selected input rows |
| `"selected_with_request"` | After selection | Selected input rows plus receiver request |

`"full"` runs once per sender; the others run once per sender and request, shared
by all its receivers. The request is used as given, without a chat template.
Selected context holds only the chosen rows, not discarded instructions.

- Before selection, the budget uses the extended state. Optional
  `reasoning_budget=N` caps new rows, otherwise the sender's remaining context
  does. Built-in selectors keep the new latent tail.
- After selection, `reasoning_budget=N` is required. RCLC uses `T + N` in the
  budget formula, gives the selector the rest and appends up to `N` rows: with
  `steps=50`, `reasoning_budget=50` and `budget=128`, at most 78 source rows are
  selected. Unused space is not refilled.

`SenderState.latent_steps` describes an existing latent tail; reasoning adds to
it without rerunning it. `reasoning_context` and `reasoning_budget` require
`reasoning`. See [GPU validation](development.md#gpu-validation) for tested
paths; other custom algorithms and benchmark comparisons are unvalidated.

### Custom methods

```python
async def my_method(state: rclc.SenderState, *, max_positions: int,
                    request: torch.Tensor | None = None) -> rclc.SenderState:
    ...
```

Async methods run on the caller's loop and can await tools, services or
`rclc.latent_mass`; synchronous methods run in a worker with the same validation.
A method owns its stopping rule and may append 0 to `max_positions` rows. Return
a consistent `SenderState`: add the appended count to `latent_steps`, mark new
rows `-1` when token IDs exist, and keep the model, tokenizer, inherited count,
input prefix and cache prefix. RCLC checks shapes, finite values and prefixes; the
method must compute correct KV for its rows. It receives private copies of rows,
IDs, cache and request; the model and tokenizer are shared and read-only. This is
trusted code, not a sandbox. Async methods should propagate cancellation and
clean up; blocking methods cannot be interrupted. Any exception or invalid result
prevents all deliveries for the call.

On a receiver, prefill its handoff and prompt into a `SenderState` and call
`await rclc.latent_mass(receiver_state, steps=40)`, optionally with
`request=delivery.request`. These rows use model context, not transmission
budget. `steps=0` returns the original state.

## Selectors

CacheBack is the default `selector`; `rclc.selectors.qsnap` and
`rclc.selectors.chunkkv` are built-in alternatives. Any callable
`selector(sender, request_ids, budget) -> positions` works; it runs once per
sender and request and is reused for every receiver. See
[the selector guide](selectors.md) for the contract, W=4 spans and an example.

## vLLM default

Use dense Qwen3 with **vLLM 0.11.1 / Torch 2.9.0 / Transformers 4.57.1**.
Install `requirements/vllm.txt` in a separate Linux CUDA environment, then the
package. This is the paper's Qwen engine path; Qwen3-8B A100 runs through this
adapter are in [GPU validation](development.md#gpu-validation).

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

from vllm import LLM

import rclc

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    tensor_parallel_size=1,
    enable_prefix_caching=False,
    enforce_eager=True,
    enable_prompt_embeds=True,
    gpu_memory_utilization=0.45,
    kv_transfer_config={
        "kv_connector": "RCCCaptureConnector",
        "kv_connector_module_path": "rclc.capture.connector",
        "kv_role": "kv_both",
    },
)
agent_x = rclc.bind(llm, backend="vllm", messages=history_x)
```

Capture must be enabled when the engine is created. The adapter supports one
in-process, unquantized dense Qwen3 engine with a single KV cache group and full
prompt state. Tensor parallel, sliding-window, hybrid, quantized and FP8 KV
caches are unsupported; keep `kv_cache_dtype="auto"`. Unsupported configuration
raises `ValueError` at the connector or binding check. Capture copies request
pages and never writes into the engine's cache pool.

Continue as in [Bind an existing agent](#bind-an-existing-agent) with
`llm.generate(**inputs)`. The engine supplies the tokenizer. `pop` returns
`prompts={"prompt_embeds": ...}`, `sampling_params` with the reserved output cap
and `use_tqdm=False`. You may edit `inputs["sampling_params"]` within that cap
and with `n=1`; keep the prompt rows when recording a continuation.

vLLM prefills token IDs or continuous rows and the connector gathers the prompt
KV, as in the paper branch's Qwen engine. Rollout and selection use an HF-shaped
view of the resident weights, loading no second checkpoint; leave GPU memory for
captured pages and these side passes. The receiver re-prefills selected rows.
With messages the first send captures lazily; with `prompt=...` binding captures
at once. Capture uses a discarded one-token greedy request. `update` re-prefills
the retained prefix plus output tokens, because `RequestOutput` does not expose
its KV, and needs one spare context position. The engine's context limit applies.

If your loop already captured the prompt, skip the second prefill with
`rclc.bind(llm, backend="vllm", prompt=exact_ids, request_id=capture_id)` or
`agent.update(exact_ids, request_id=capture_id)`, where `exact_ids` is the
captured `[1, tokens]` prompt without generated tokens. For token generations,
pass the complete prompt and output IDs to `agent.update(...)`. The lower-level
interface registers the request in your submission loop before stepping; on
0.11.1 the capture ID is the `add_request` ID:

```python
from rclc.vllm import request_capture, sender_from_vllm

request_capture(request_id)
llm.llm_engine.add_request(request_id, prompt, sampling_params)
# Your existing loop drives llm.llm_engine.step() until prompt prefill completes.
sender = sender_from_vllm(llm, request_id, prompt_token_ids)
```

`prompt_token_ids` is the exact one-dimensional int64 prompt, including
chat-template tokens; only the **prompt** is captured. Call
`discard_capture(request_id)` for cancelled or unused captures. A successful
`sender_from_vllm` releases the entry; failed validation keeps it. Pass
`inherited_positions=...` if the prompt begins with inherited rows. An agent with
the paper's HF-shaped model, cache and rows can build
[`SenderState` directly](#low-level-sender-state), keeping its latent steps.

### Opt-in vLLM 0.26.0

Install `requirements/vllm-unstable.txt` in a **different environment**. It
passed the Qwen3-8B GPU journey, but generation was slower than 0.11.1 in that
workload. Opt in on the connector and binding (`allow_unstable=True`), and pass
`attention_backend="FLASH_ATTN"` to `LLM` instead of the environment variable.
Keep eager execution, prefix caching disabled and tensor parallel size one.
0.26.0 returns a randomized internal request ID: register it after submission
and before the first step, and use it for cleanup.

```python
kv_transfer_config["kv_connector_extra_config"] = {"allow_unstable": True}
internal_id = llm.llm_engine.add_request(request_id, prompt, sampling_params)
request_capture(internal_id)
sender = sender_from_vllm(llm, internal_id, prompt_token_ids, allow_unstable=True)
```

The `paper` branch separately preserves its 0.26.0 Gemma, Ministral and
Nemotron runtimes and measured settings.

## Low-level sender state

```python
from rclc import SenderState, transfer

sender = SenderState(
    model=sender_model,
    past_key_values=sender_cache,
    input_embeds=sender_input_rows,
    tokenizer=sender_tokenizer,
)
await transfer(sender, receiver_inbox.append, "Find the owner and launch date.")
```

`input_embeds` is `[positions, hidden]`: the exact input embeddings (not final
hidden states) that built the unpadded single-sequence cache, on the model's
device and dtype, with the model in eval mode. Built-in selectors score a
private cache copy; see [Concurrency](#concurrency) for sharing rules.

- `latent_steps=G` marks the final G rows as a continuous latent tail, which
  built-in selectors keep.
- Optional `token_ids` is an int64 vector aligned with all rows, `-1` for
  continuous positions.
- In a chain, `inherited_positions` is the carried prefix length. It must not
  overlap the latent tail; inherited latent rows can be dropped when reranking.
  This carries inherited rows that token IDs cannot reconstruct.

## Receiver continuation

```python
delivery = receiver_inbox.pop(0)
inputs = delivery.for_hf(receiver_model.get_input_embeddings().weight)
with torch.inference_mode():
    received = receiver_model.model(**inputs, use_cache=True)
prompt = delivery.for_vllm(matching_embedding_weight)  # {"prompt_embeds": cpu_rows}
```

This builds fresh receiver KV and never splices foreign KV. `delivery.request`
is metadata, **not appended** to the rows; your agent adds its instructions,
request and continuation. `delivery.messages` keeps sender boundaries; the
convenience methods concatenate in sender order. `for_vllm` needs
`enable_prompt_embeds=True`; see vLLM's
[prompt embedding documentation](https://docs.vllm.ai/en/v0.26.0/features/prompt_embeds/).
Matching hidden width does not establish compatibility: use the same checkpoint
and tokenizer unless you have validated a cross-model mapping. Opaque text-only
APIs cannot expose the required sender state.

## Representations

| Option | Payload | Status |
| --- | --- | --- |
| `embeddings` (default) | All selected input rows | Paper handoff representation |
| `token_ids+continuous` | Source token IDs, continuous vectors, original order | Experimental; GPU validation limited to dense Qwen3 |

Mixed messages mark continuous rows with `-1` in `Message.token_ids`, and
`Message.materialize(weight)` rebuilds the original order from a matching
embedding table. The sender checks that selected IDs reproduce its input rows;
the receiver checks shape and ID bounds; the application ensures
checkpoint/tokenizer compatibility. `Message.positions` counts reconstructed
positions; `Message.nbytes` counts tensor bytes only, excluding Python objects,
metadata and transport framing.
