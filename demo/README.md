# CacheBack demo

Three agents pass evidence to a fourth. Maya's booking reference J7 leads to
Harbour Hub, whose passport requirement overrides the original desk's ID advice.

## Watch

Open <https://maxr0ssi.github.io/rclc/demo/> and press **Run**. To serve it
locally instead, from the repository root:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765/demo/> and press **Run**.

The topology runs left to right. The three separate contexts are labelled
`booking-confirmation.txt`, `friday-desk-changes.txt` and `collection-id-rules.txt`.
Each column shows a source document with the
RCLC highlights and the corresponding generated text handoff. After three
handoffs, the last column shows the two unedited answers and their total times.
Text boxes automatically follow the newest generated line. All documents and
messages are scrollable. On a small screen, scroll the whole
diagram horizontally. **Replay** starts it again. **Clear** stops playback and
resets the highlights, messages and clocks.

The reviewed 8B answer shows `✓ Harbour Hub · Passport` when it finishes.
That annotation matches the exact reviewed question and answer; other recordings
are not automatically graded or marked correct.

**1x speed** is the default. Each method progresses independently using its
measured handoff durations and total completion time. The faster method finishes
while the other keeps streaming. Select **2x** or **4x** to speed up playback;
the methods still progress independently and the clocks show recorded time.
Text streams at an estimated constant character rate within each stage. The
recording has no per-token timestamps or separate prefill/decode timings.
Completion times are measured; streaming is illustrative. Reduced-motion
preferences disable the highlight animation.

The presentation shows the first question at **W=4**. There are no width or
handoff controls. The [raw recording](trace-w4.json) retains both questions,
full-context answers, intermediate messages, exact token origins and timings.
This replay uses **Qwen3-8B on an A100**, recorded at r4 with expanded
documents. Its raw trace is unchanged from run `20260925T013950Z-f770ef6b`.
The original archive and logs are preserved separately.

## Record

Install RCC as described in the repository README, then run:

```bash
python demo/record.py --span-size 4 --output demo/trace-w4.json
```

For 8B on a CUDA GPU:

```bash
python demo/record.py --model Qwen/Qwen3-8B --device cuda \
    --span-size 4 --output demo/trace-w4.json
```

The recorder defaults to r4. `--ratio`, `--dtype` and `--revision` can override
compression, precision and model revision. Final answers allow 1,024 generated
tokens and stop earlier at EOS. Each question first records an uncompressed
full-context answer using the same model and receiver instructions.

For Colab, upload [colab.ipynb](colab.ipynb) and select a 24 GB-class or larger
GPU with a high-RAM host. Prepare its source bundle from a committed checkout:

```bash
mkdir -p output/colab
git archive --format=zip --output=output/colab/rcc-demo-colab-source.zip HEAD
```

Upload that ZIP when prompted, mount Drive and run the cells in order. Each run
saves its source, configuration, environment, complete logs and traces to a new
`MyDrive/rcc-demo/<run-id>/` folder. The notebook retains its width comparison for
analysis; the viewer shows W=4 only. The last cell downloads a portable replay.
Serve its extracted directory using the command above. No model is needed to
replay it. The anonymous submission ZIP is unrelated and remains untouched.

## Reading the result

The four roles share one model. Agent 1 reads the booking, agent 2 adds the
operations notice, agent 3 adds the ID rules, and agent 4 answers from the final
handoff alone. Later agents cannot reread discarded evidence.

Text generates new notes. RCLC sends selected input embeddings through
`rcc.transfer`, with no extra latent steps. Highlights follow exact tokenizer
offsets, including word fragments. As another document arrives, selection can
drop evidence retained at the previous handoff. Each text column is the actual
message generated at that step; it is not a summary written for the viewer.

Both methods get the same cumulative r4 allowance:
`sum(ceil(source_positions / 4))` over the documents seen so far. Text can stop
early. The inherited RCLC prefix is reranked, not protected. These are position
budgets, not equal byte budgets; prompt framing is outside both caps.

The current expanded sources have 770, 773 and 771 positions, giving handoff
caps of **193, 387 and 580**. They add 489-505 positions per document without
changing the original evidence or questions. The earlier CPU sources have
280, 284 and 266 positions, with caps of 70, 141 and 208. Lengthening the sources
also increases the absolute r4 budget.

W=4 means four-token selection spans. It gives finer selection boundaries;
it is a demo setting, not a general claim that smaller spans are better.
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
- [index.html](index.html): static presentation with no build or dependencies.

The existing end-to-end tests cover the native recorder, exact causal row
lineage, budgets and notebook export. With the server running, open
<http://127.0.0.1:8765/tests/demo.html> for one browser journey covering all three
speeds, streaming, independent completion, replay and clear. It reloads the demo
after the controlled-clock checks and verifies playback with native browser
timing, leaving a working replay. No model output is
rewritten for presentation; Markdown bold markers are rendered as emphasis.
