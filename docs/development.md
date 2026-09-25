# Development harness

Install once in a checkout of `main`:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
scripts/install-hooks.sh
```

Run `.venv/bin/python scripts/check.py` before committing. The same command runs
in Git commit, push and merge hooks and in CI. Stage tracked changes before
committing so the checked source matches the index. The complete CPU suite is
small enough to run at every Git gate; there is no affected-test scheduler.

| Rule | Enforcement |
| --- | --- |
| Deterministic formatting, imports, docstrings, lint and no library prints | Ruff |
| Strict library types | Pyright over `src/rcc` |
| Warn over 400 nonblank lines; block over 600 | `scripts/check_file_size.py` |
| Fewer than three consecutive comment lines | `scripts/check.py` |
| No punctuation dashes or invisible Unicode characters in source/docs | `scripts/check.py` |
| No common credential patterns or committed environment files | `scripts/check_secrets.py` |
| Existing complexity counts may only shrink | C901 counts in `scripts/check.py` |
| New behavior has a meaningful offline check | CPU pytest suite and review |
| Both agent configurations and all Git hooks remain wired | `scripts/check.py` |
| No ordinary Git-hook bypasses from agent shell calls | `scripts/block_no_verify.py` |

The edit hooks share the text gates; Git and CI additionally run all lint,
types, complexity and CPU tests. Shell edits are checked at the Git/CI boundary.
The hook guard is an assistance mechanism, not a security sandbox.

`.claude/settings.json` runs the edit and shell guards in Claude Code.
`.codex/hooks.json` wires the same guards in Codex. Project-local Codex hooks
require trust review before they run; installing the Git hooks does not grant
that trust. Review the exact definitions through Codex's `/hooks` interface.
See the [official hook documentation](https://learn.chatgpt.com/docs/hooks).

The research harness's claim ledger, AWS fleet controls, notebook audit tiers
and affected-test routing do not belong to this small transport package. The
`paper` branch retains the frozen experiments. Do not present this reduced
main-branch harness as the entire research operations harness.

## Test strategy

Prefer complete user journeys over one test per helper. The main branch has
16 CPU product cases, one harness case and one separately selected GPU case.
Keep this set small: extend an existing scenario when adding a related behavior,
and add a case only for a distinct failure mode or workflow.

The CPU cases cover native Qwen prefill, real request-conditioned selection,
independent eager-attention scores, protected budgets, receiver continuation,
multiple senders/receivers/requests, dense and mixed representations, cache
preservation, four-hop relative and bounded budgets, custom selectors and invalid
outputs, optional latent rollout before/after selection with both conditioning
modes, receiver-side rollout, failure recovery, numerical anchors and both vLLM page layouts.
The demo case records a three-hop relay with a native tiny model at W=16 and W=4.
It checks source offsets, causal row lineage, cumulative budgets, measured timing
fields and trace serialization. A second demo case executes the Colab notebook
from a fresh source archive using a tiny local checkpoint. It runs both CLI
recordings and verifies saved logs, model metadata and the portable replay ZIP;
Colab authentication, installation and GPU execution remain owner-run checks.
The vLLM layout case simulates the engine boundary; it cannot validate a running
vLLM engine. Broader research primitives and their tests remain on `paper`.

The owner-run GPU case keeps one engine session for chunked sender prefill,
capture and delivery. It compares four baseline receiver continuations with
native token inputs, then runs 64 continuations across all four latent contexts,
relative/bounded budgets and dense/mixed payloads. The second receiver adds local
latent steps. It checks matching inputs and decoded tokens across encodings,
position budgets, finite rows and unchanged sender caches. The CPU adapter case
exercises this same scenario with a tiny HF decoder standing in for vLLM;
that simulation does not validate a live engine.

In the default vLLM environment with the dev extra installed:

```bash
python -m pytest -m gpu tests/test_runtime.py
```

For the separate unstable 0.26.0 environment, explicitly opt in:

```bash
RCC_TEST_UNSTABLE=1 python -m pytest -m gpu tests/test_runtime.py
```

This is also runnable as `python examples/vllm_transfer.py` (add
`--allow-unstable` for 0.26.0). A passing default GPU run is required before
release. Neither GPU pin has been executed for this public adapter yet.
The Git gates continue to run the complete CPU suite; this test consolidation
does not relax formatting, types, comments, complexity or hook rules.

## Runtime compatibility

CI is configured to run the complete CPU suite with Transformers 4.57.1 and 5.13.0. To reproduce
either environment locally, install the chosen pin alongside the dev extra:

```bash
.venv/bin/python -m pip install -e ".[dev]" "transformers==4.57.1"
.venv/bin/python scripts/check.py
```

### Clean Linux installs

The Ubuntu CI matrix covers Python 3.10 and 3.12 with both Transformers pins.
Each job creates an isolated venv, installs the package as a wheel rather than
an editable checkout, checks dependency consistency and runs the harness.
Linux execution is pending; the completed local checks used macOS on Apple
Silicon, Python 3.11 and CPU PyTorch.

On a Windows machine, use Ubuntu under WSL2 for these Linux checks. This does
not establish native Windows compatibility. vLLM 0.11.1 also documents WSL as
its Windows route in the [upstream requirements](https://github.com/vllm-project/vllm/blob/v0.11.1/docs/getting_started/installation/gpu.md#requirements).
No GPU is needed for the CPU install checks. In a fresh checkout of `main`:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install ".[dev]" "transformers==4.57.1"
.venv/bin/python -m pip check
scripts/install-hooks.sh
.venv/bin/python scripts/check.py
```

Repeat in fresh environments for Python 3.12 and Transformers 5.13.0 to cover
the four CI combinations. Keep GPU execution separate and explicitly selected.

Engine or cache changes need both pins. GPU execution, vLLM engine capture and
benchmark claims require their own validation; a tiny CPU model is not evidence
of measured end-to-end behavior. The `token_ids+continuous` representation is
experimental and untested in end-to-end GPU runs even when its CPU reconstruction
check passes.

Keep comments short, reuse existing primitives, and review source changes before
committing. Ask before adding a dependency, raising a limit, weakening an
assertion or changing a measured claim. Never bypass the harness to make a
commit pass.
