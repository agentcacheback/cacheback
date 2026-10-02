# CacheBack booking relay

The replay labels the method CacheBack; `rclc` remains its recording key.

Three agents pass evidence to a fourth. Maya's booking reference J7 leads to
Harbour Hub, whose passport requirement overrides the original desk's ID advice.

## Watch

The [booking section](../index.html#booking-demo), below the main coding video, presents the recorded final answers
and completion times before the replay. Selected positions use teal highlights.

Open <https://agentcacheback.github.io/demo/> and press **Run**. To serve it
locally, run this from the repository root and open <http://127.0.0.1:8765/demo/>:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

The replay shows the first question from a **Qwen3-8B on an A100** recording at
r4 and **W=4**, with expanded documents. There are no width or handoff controls.
The [raw recording](trace-w4.json) retains both questions, full-context answers,
intermediate messages, exact token origins and timings.

- Columns run left to right through the three contexts,
  `booking-confirmation.txt`, `friday-desk-changes.txt` and
  `collection-id-rules.txt`. Each shows the source document with RCLC highlights
  and the generated text handoff. The last column shows both unedited answers
  and their total times.
- Text boxes follow the newest generated line. Documents and messages scroll
  within their compact panes. On smaller screens, agent cards form two columns,
  then one on phones, with their agent labels repeated above the content.
  An embedded replay resizes to its content, so its outer frame never needs
  horizontal or vertical scrolling. Scroll the containing page normally.
- **Replay** restarts. **Clear** stops playback and resets the highlights,
  messages and clocks.
- The reviewed 8B answer shows `✓ Harbour Hub · Passport` when it finishes. The
  annotation matches that exact question and answer; other recordings are not
  graded or marked correct.

Playback defaults to **1x**; **2x** and **4x** speed it up. Each method
progresses independently from its measured handoff durations and total
completion time, so the faster one finishes while the other keeps streaming, and
the clocks show recorded time. Playback starts from the first animation-frame
timestamp, keeping embedded-frame clock origins from producing negative time. Completion times are measured; streaming is
illustrative. Text streams at an estimated constant character rate within each
stage, because the recording has no per-token timestamps or separate
prefill/decode timings. Reduced-motion preferences disable the highlight
animation.

## Record

Install RCLC as described in the [repository README](../README.md#install), then
run:

```bash
python demo/record.py                                      # Qwen3-0.6B on CPU
python demo/record.py --model Qwen/Qwen3-8B --device cuda  # 8B on a CUDA GPU
```

The recorder defaults to r4 and W=4 and writes `demo/trace-w4.json`, the file
the viewer loads. `--span-size`, `--output`, `--ratio`, `--dtype` and
`--revision` override the width, file, compression, precision and model
revision. Final answers allow 1,024 generated tokens and stop earlier at EOS.
Each question first records an uncompressed full-context answer using the same
model and receiver instructions.

For Colab, upload [colab.ipynb](colab.ipynb) and select a 24 GB-class or larger
GPU with a high-RAM host. Prepare its source bundle from a committed checkout:

```bash
mkdir -p output/colab
git archive --format=zip --output=output/colab/rclc-demo-colab-source.zip HEAD
```

Upload that ZIP when prompted, mount Drive and run the cells in order. Each run
saves its source, configuration, environment, complete logs and traces to a new
`MyDrive/rclc-demo/<run-id>/` folder. The notebook retains its width comparison
for analysis; the viewer shows W=4 only. The last cell downloads a portable
replay; serve its extracted directory with the command above. Replay needs no
model.

## Reading the result

The four roles share one model. Agent 1 reads the booking, agent 2 adds the
operations notice, agent 3 adds the ID rules, and agent 4 answers from the final
handoff alone. Later agents cannot reread discarded evidence.

Text generates new notes; each text column is the actual message generated at
that step, not a summary written for the viewer. RCLC sends selected input
embeddings through `rclc.transfer_sync`, with no extra latent steps. Highlights
follow exact tokenizer offsets, including word fragments. As another document
arrives, selection can drop evidence retained at the previous handoff.

Both methods get the same cumulative r4 allowance:
`sum(ceil(source_positions / 4))` over the documents seen so far. Text can stop
early. The inherited RCLC prefix is reranked, not protected. These are position
budgets, not equal byte budgets; prompt framing is outside both caps.

The current expanded sources have 770, 773 and 771 positions, giving handoff
caps of **193, 387 and 580**.

W=4 means four-token selection spans. It gives finer selection boundaries; it
is a demo setting, not a general claim that smaller spans are better.
The correct answers are **Harbour Hub, with a passport** and **No, a driving
licence alone is not accepted there**. Raw outputs preserve mistakes. Two
fictional questions do not establish an accuracy benchmark or retrieval failure.

Total times include all three handoffs and final receiver generation. GPU work
is synchronized at timing boundaries. Model loading, tokenization and warmup
are excluded. Methods run sequentially with alternating order between questions.
These are individual observations, not a repeated performance benchmark. This
recorder uses native Hugging Face state and does not validate vLLM.

## Files

- [record.py](record.py): documents, receiver baseline, relay and timing.
- [colab.ipynb](colab.ipynb): GPU execution, Drive persistence and export.
- [trace-w4.json](trace-w4.json): full W=4 recording, including both questions.
- [trace.json](trace.json): W=16 recording of the same run, kept for the
  notebook's width comparison.
- [index.html](index.html): static presentation with no build or dependencies.

Tests: `tests/test_demo.py` covers the recorder and notebook export;
[tests/demo.html](../tests/demo.html) is the browser journey (serve the
repository root and open it).
