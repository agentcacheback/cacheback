# Project website

The project page lives at <https://maxr0ssi.github.io/rclc/>. GitHub Pages
publishes the repository root from `main`, using the existing Pages settings.
Merging a homepage change into `main` updates the site automatically.
The root `.nojekyll` file serves these static files without Jekyll processing.
The canonical URL and social preview point to this Pages address and the coding
video poster. Share the homepage, or append `#demo` to link to the main demo.

## Layout

- `index.html` at the repository root: the homepage and published abstract.
  The top-left header reads RCLC. CacheBack is the main title, with
  receiver-conditioned communication between LLM agents as its subtitle.
  Authors appear directly beneath it, followed by DAP Lab and Columbia
  University. The linked lab logo loads from the official
  [DAP Lab website](https://daplab.cs.columbia.edu/).
- `website/assets/`: CSS, browser JavaScript, selected results, and paper curves.
- `demo/`: the existing recorded booking relay, still served at `/rclc/demo/`.
- `demo/coding/`: the main 36-second coding video, poster, and recorded results.
- `website/check.mjs`: dependency-free Node checks, also run by CI.

Links to assets and the embedded demo are relative to the homepage. The same
files work under the GitHub Pages `/rclc/` prefix and a local HTTP server.
There is no build step, API, model server, or hosting account to configure.
Google Fonts is optional; system fonts are the fallback. The replay reads its
recorded JSON from this repository and does not run inference.

## Preview and verify

From the repository root, using Node 22 or newer:

```sh
node --check website/assets/app.js
node website/check.mjs
python3 -m http.server 4174 --bind 127.0.0.1
```

Open <http://127.0.0.1:4174/> and check the hero, narrow-screen layout,
copy buttons, main video playback, fullscreen, and embedded booking replay. The offline
check validates local assets and anchors, the demo route and recording,
the displayed benchmark bars and annotations, and homepage answers and times
against the raw demo recording.
The homepage and its assets stay below 100 KB, with JavaScript below 6 KB
gzip. Demo media and recordings are separate from that budget. The native video
player uses `preload="none"`, so the 5.8 MB MP4 loads when requested. There is
no video service, autoplay, or additional dependency. A small play-button overlay
starts the native player; native controls remain available without JavaScript.

## Content and demos

The page runs from introduction and the four-model comparison to a visible
TL;DR, method, detailed results, demos, quickstart, and citation. The overview
explains the problem, contribution, and selected Qwen result; the unchanged
published abstract expands directly beneath it. LongBench curves expand under
the results.
Use two names in the explanatory copy: receiver-conditioning is the idea;
CacheBack is its simplest, robust, training-free implementation. Keep RCLC in
the small header and repository references. Introduce latent state in the
method section; label the charts and recorded demo CacheBack. The published
abstract, paper title, citation, and recorded answers remain unchanged.
Teal identifies CacheBack, including selected positions in the recorded replay.
Desktop layouts above 800px render at 85% scale, about 5% larger than the
previous 81% layout, with wider outer margins. Medium widths use 94.5%; phones
at 520px and below retain their original text sizing.

The method section diagrams receiver-conditioning: a request travels from receiver
to sender, and selected state returns to the receiver. CacheBack is introduced
beneath it as the simplest, robust, training-free implementation. This is a
static explanation of the protocol, not a measured attention visualization.

`assets/results.json` contains the selected FanOutQA points. The hero shows
all four models with shared zero-based accuracy and seconds scales. Gains
and speedups use the unrounded values. Four SVG curves show the paper's
fuller results. Keep captions, experimental setup, and measured values
together when editing. The abstract is reproduced under CC BY 4.0 with
emphasis added; its attribution is on the page.

The main demo is the supplied 3840×2160, 60 fps, silent video in
`demo/coding/rclc-cacheback-4k.mp4`. Preserve its original bytes. Its poster is
a frame from the video; native controls provide playback and fullscreen.
The README links to the homepage's `#demo` anchor with the same poster.

The video reconstructs a Django patch task from the recorded Qwen3-8B run
`bughunt-16333-run-20261001T051520Z`, sample `s1`. Seven workers provide input
to a coordinator. The supplied report identifies the missing `save_m2m()` call.
CacheBack finishes in 25.66205141 s and text in 113.20635836 s; the 4.41× ratio
applies to this case only. Both produce the same patch and pass one new test
and 87 regression tests. Startup and subsequent grading are excluded from
the clocks. The two arms ran separately, and video playback speed varies.
`demo/coding/evidence.json` retains the recorded results used in the caption;
the homepage offers a text description of the silent video. The offline check
validates the displayed times and ratio against these results.

The secondary booking relay shows both recorded final answers and measured total times
first, with model-quoted evidence and the embedded replay in native disclosures.
The full-width teal "Replay the booking handoffs" control opens the replay.
The frame follows the replay's content height. Agent cards wrap on narrow
screens; only the individual document and message panes retain internal scrolling.
Embedded sizing uses the content's layout height, top position, and bottom margin.
Overflow stays available as a fallback; content is never hidden to remove a
scrollbar. The browser regression checks full document height and footer bounds
in the actual homepage, at desktop and phone widths and through disclosure toggles.
The answer text, evidence, and times in `index.html` must match the first case
in `demo/trace-w4.json`; the offline check enforces this. See
[the demo guide](../demo/README.md) to record or update it.

The two `upcoming-demo-*` slots remain inside the hidden `upcoming-demos`
container. Replace their content and remove `hidden` when the demos are ready.
Use a real recording or iframe,
with the task, model, hardware, and timing scope stated in its caption.
Keep measured benchmark results distinct from individual demonstrations.

Hugging Face resources are omitted until an upload is ready. Paper, code,
and citation links are already live.
