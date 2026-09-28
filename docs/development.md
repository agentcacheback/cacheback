# Development harness

Install once in a checkout of `main`:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
scripts/install-hooks.sh
```

The installer restores the `pre-push` and `pre-merge-commit` symlinks before
setting executable permissions. Python ZIP extraction turns Git archive symlinks
into plain target-name files; run the installer in extracted Colab source trees
as well as normal checkouts. The archive regression test covers this path.

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

## Tests

Prefer complete user journeys over one test per helper. Extend an existing
scenario when adding related behavior, and add a case only for a distinct
failure mode or workflow. The CPU suite runs a tiny native Qwen3 model and covers
the HF bindings, selection and its records, budgets, both payloads, the demo and
Colab notebooks in offline test mode, async fan-in with cancellation, native
continuation, state append, snapshot restore, setup diagnostics, the small
selector evaluation, the development harness and a simulated vLLM engine boundary.
Reasoning runs in all four contexts, synchronously against a full-forward
numerical reference and through the async API, plus a BF16 cache-prefix
regression; both Transformers pins pass it. The vLLM simulation cannot validate a
running engine. Research primitives and their tests remain on `paper`.

The GPU journey in `scripts/validate_vllm.py` runs chunked sender prefill,
capture, every reasoning context with relative and bounded budgets, dense and
mixed payloads, native-token parity and a bound-agent return handoff after
appending input. Run it in the default vLLM environment with the dev extra
installed:

```bash
python -m pytest -m gpu tests/test_runtime.py
```

For the separate opt-in 0.26.0 environment:

```bash
RCC_TEST_UNSTABLE=1 python -m pytest -m gpu tests/test_runtime.py
```

It also runs as `python scripts/validate_vllm.py` (add `--allow-unstable` for
0.26.0). The archived runs below used Qwen3-8B:

```bash
python scripts/validate_vllm.py --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
```

Set `VLLM_PINS` in the [Colab notebook](../examples/colab.ipynb) to run
three repetitions per pin on the same Qwen3-8B revision, with timings, memory
usage and raw outputs.

## GPU validation

The short vLLM example is in `examples/vllm_transfer.py`; the full journey is in
`scripts/validate_vllm.py`. Re-run it after any engine, capture or selector change.

On 2026-09-28 UTC, the release source passed three fresh-process runs per vLLM
pin on one A100-SXM4-40GB, using Qwen3-8B BF16 at revision
`b968826d9c46dd6066d109eabc6255188de91218`. Each run exercised 45 operations
(5 captures, 19 transfers, 21 generation calls) and 68 four-token continuations,
including every reasoning context, dense/mixed payload parity and the bound-agent
return handoff. All 68 continuations per pin were token-identical across
repetitions and to the 2026-09-27 run of the pre-cleanup source.

HF returned both expected answers with no reasoning, with
`partial(latent_mass, steps=50)`, and with a custom async callback awaiting two
latent steps, each identical to the 2026-09-27 answers. The 50-step case used a
bounded 128-position budget, because relative r4 cannot fit the reserved latent
rows of these short prompts; the other cases used r4. These are smoke checks,
not accuracy scores. The same run passed the four Linux CPU combinations below.

| Median total per journey | vLLM 0.11.1 | vLLM 0.26.0 |
| --- | ---: | ---: |
| Sender captures (5 calls) | 0.238 s | 0.287 s |
| Transfers, including reasoning | 18.402 s | 18.455 s |
| Native generation calls | 4.653 s | 5.677 s |
| Peak allocated GPU memory | 17.91 GiB | 17.80 GiB |

Keep 0.11.1 as default: 0.26.0 passed correctness but generation took 22% longer
in this workload. These are synchronized operation timings, excluding model
loading and initial kernel compilation. They cover short continuations with
eager execution, not general serving throughput. The 2026-09-27 run made two
captures per journey; per-call capture time is unchanged.

The [summary](validation/release-a100/summary.json) records every repetition,
the Linux matrix and the setup correction. The [full archive](validation/release-a100/results.zip)
contains the exact source, executed scripts, raw outputs, logs and environment
pins. Its source ZIP SHA-256 is
`7408283f043f525141a57f5cc88254390d13576ab1ab2ea40c0419bf450ab675`.
The Colab runtime was deleted after downloading and verifying the archive.
The [previous async run](validation/async-a100/summary.json) is kept for comparison.

[Earlier reasoning validation](validation/reasoning-a100/summary.json) predates
the async API. It retains the [cached-prefix regression](validation/reasoning-a100/before-fix.log)
and its [partial result](validation/reasoning-a100/before-fix.json); the CPU
reasoning journey covers that fix. The [original vLLM runs](validation/vllm-a100/)
used the removed `latent_steps` interface.

## Runtime compatibility

CI runs the complete CPU suite with Transformers 4.57.1 and 5.13.0, then
builds both distributions and runs `twine check`. To reproduce either
environment locally, install the chosen pin alongside the dev extra:

```bash
.venv/bin/python -m pip install -e ".[dev]" "transformers==4.57.1"
.venv/bin/python scripts/check.py
```

### Clean Linux installs

The Ubuntu CI matrix covers Python 3.10 and 3.12 with both Transformers pins.
Each job creates an isolated venv, installs the package as a wheel rather than
an editable checkout, checks dependency consistency and runs the harness.
All four combinations also passed for the release source on Colab Linux on
2026-09-28: 27 CPU tests per combination plus the full harness and dependency
checks ([results](validation/release-a100/summary.json)). The validation environments
include pip because the notebook tests record `pip freeze`. The initial public
snapshot passed [Ubuntu CI](https://github.com/maxr0ssi/rclc/actions/runs/36208175275).
Local checks also passed on macOS Apple Silicon with Python 3.11 and CPU PyTorch.

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
of measured end-to-end behavior. The `token_ids+continuous` representation
remains experimental. Its live GPU validation covers dense Qwen3 only; other
model families remain untested end to end.

## Package release

The distribution is `rclc`; import it as `rcc`. Version 0.1.0 remains unpublished.
Do not publish until the default vLLM GPU gate above passes and the full CPU
matrix is green for the release commit. Keep the paper branch unchanged.

Build and inspect both distributions in a separate environment:

```bash
python -m venv /tmp/rclc-build
/tmp/rclc-build/bin/python -m pip install build twine
/tmp/rclc-build/bin/python -m build
/tmp/rclc-build/bin/python -m twine check --strict dist/*
```

Install the wheel into a fresh environment, run `pip check`, then run the
quickstart and cached-conversation example from outside the source directory.
Inspect the source archive too: its explicit allowlist includes only the library
and package metadata, so local research data cannot enter a release build.
Publish the inspected files using a project-scoped PyPI token through Twine's
interactive password prompt; never put credentials in a command or commit:

```bash
/tmp/rclc-build/bin/python -m twine upload dist/rclc-0.1.0*
```

PyPI owns the final name-availability decision. Do not use `rcc`: that name
belongs to an unrelated package. A future release needs a new version in both
`pyproject.toml` and `src/rcc/__init__.py`; PyPI versions cannot be overwritten.
