# Changelog

All notable changes to MMH3Tools are documented here.
This project follows [Semantic Versioning](https://semver.org/).

**Input ordering is append-only** for anything published. ComfyUI serialises widget
values positionally, so new inputs must be added at the END of a node's input list.
Never insert or reorder existing inputs, or saved workflows silently rebind to the
wrong widgets. A node that has not shipped may still be reordered freely — say so in
the entry, and migrate any local workflow in the same commit.

## [Unreleased] — 0.89.0

### Added

- **`MMH3EmbeddingSelect` — "MMH3 Embedding Select", `MMH3Tools/conditioning`.** A
  dropdown picker for H3 text embeddings plus a `chunks` range, chained through
  `previous`, emitting the spec string `MMH3ReferenceMultiPrompt.embeddings` parses.

  0.88.0 shipped that input as a typed multiline field, which is not how a model file
  gets chosen anywhere in ComfyUI. A typo there is especially bad: core drops an
  unresolvable `embedding:` with a log line rather than an error, so the render just
  quietly lacks the embedding. The reference node's schema is unchanged — the picker's
  output is a STRING, so it wires straight into the existing input.

  The dropdown lists the **whole folder**. A first cut filtered it to 5120-wide files,
  which was wrong twice over: no other loader filters, and it meant reading every
  header at schema-build time, so a file whose header would not parse was included or
  excluded on the strength of an exception — `conditioningPOS.safetensors` on this
  machine returns no readable shape and was being listed by that accident rather than
  by intent. The width is checked at execute instead, naming both numbers; an
  unreadable header reports its slot count as `?` and is allowed through.

## [Unreleased] — 0.88.0

### Added

- **`MMH3ReferenceMultiPrompt`: `embeddings` (appended LAST).** Prepends H3 text
  embeddings to every chunk's prompt, so `embedding:<name>` does not have to be typed
  into nine pipe-separated prompts by hand and retyped whenever an LLM rewrites them.

  One filename per line from `models/embeddings/`. A bare name goes on every chunk;
  `name: N` or `name: A-B` (1-based) schedules it, which costs nothing to support
  because each chunk already has its own prompt. Lines stack.

  Measured on this core: a plain prompt splices 0 vectors, `minimaxh3_bullet_time`
  splices **94** — exactly that file's row count — and stacking `storm_magic` gives
  **231 = 94 + 137**. So chaining is additive in both cost and effect, and the node
  reads each file's header to print the per-chunk slot total. They are attended at
  EVERY sampling step of every chunk, so that total is the number that matters.

  Two refusals rather than silent no-ops: a name with no matching file stops the run
  (core would drop it with only a log line), and a core that does not splice
  `embedding:` in H3 prompts stops it too. The latter is **probed**, not inferred from
  a version — before #15808 the H3 tokenizer never looked for the marker and it went
  through as ordinary words, which is invisible in the output.

  These are DiffSynth-Studio's *Diffusion Templates*: textual inversion for H3, offered
  as a lightweight alternative to LoRA on a model whose size makes LoRA training hard.

  Note the open question recorded when they were first examined — the files are
  encoder-OUTPUT-space vectors (row norms ~648 against the input embedding table's
  ~1.46) being spliced at the INPUT layer. They resolve, they are the right shape and
  they cost what they should; whether they do what DiffSynth intended is unverified.

## [Unreleased] — 0.87.1

### Changed — documentation

- **The MusicVideo entry now names its prompt-writing model.** It said only that the
  `Llama Connectivity` ids are local llama-swap names to swap for your own, and never
  which one the graph ships with — someone had to ask in `minimax_h3_resources`.

  Now records that it is **one** model across three staged calls (definitions, beats,
  shots), shipping as `qwen3.6-fable-27b-uncensored-vision`; that splitting the load
  across three calls rather than one is deliberate; and that a community report has
  **Qwen3-VL-4B-Instruct-Q8_0** working through the same three stages — worth stating,
  since a 27B is out of reach for many and the staged shape is what makes a small model
  viable. Also points at Scene Prompt Builder as the workflow that uses two *different*
  models on purpose (`gemma-4-12b-vision` for definitions, fable-27b for the rest).

## [Unreleased] — 0.87.0

### Added

- **`MMH3ImageList` — "MMH3 Image List", `MMH3Tools/reference`.** An Autogrow of image
  sockets (1–50) emitting a LIST, so many references can be collected without any being
  conformed to one frame.

  KJNodes' `ImageTensorList` is the only other node in reach that returns a list, and it
  takes exactly two inputs — N references meant N-1 chained nodes. Every `inputcount`
  node in KJNodes ends in `torch.cat`, which is the conforming being avoided.

  Socket order is `<Picture i>` order. Empty sockets are skipped so gaps are fine, and a
  socket holding a multi-frame batch expands to one reference per frame. The report
  names every reference's dimensions, says how many distinct shapes survived and what
  batching would have cropped them to, and warns past 9 that references are attended at
  every sampling step.

### Fixed

- **`_build_refs` expands multi-frame entries inside a list.** A list entry holding
  several frames encoded as ONE reference while `MMH3ImageList` counted it as several,
  so the node's reported count and the references actually built disagreed. Now N frames
  is N references however they arrived — verified the two agree at 5 and 5 for a
  4-frame batch plus a still.

## [Unreleased] — 0.86.2

### Added

- **`workflows/MMH3_Looping_RefVideo_Chunked.json`.** ManualPrompt — the simplest graph
  — wired for a windowed video reference, so `window_ref_video` has a worked example.

  A `VHS_LoadVideo` feeds both `ref_videos.ref_video_0` and
  `ref_video_audios.ref_video_audio_0`, and `chunk_frames` / `overlap_frames` are
  link-driven from the **same MMH3 Chunk Schedule outputs the sampler reads** rather
  than typed. That is the part that matters: typed separately they drift, and the
  reference windows silently stop being the spans the chunks render.

  The `ref_images` still is left wired and unwindowed — `<Picture 1>` for identity,
  `<Video 1>` for the windowed motion — with a Note saying so and giving the cost
  arithmetic. Verified the node's widget order matches the schema exactly after
  extending `widgets_values` from 7 to 10.

## [Unreleased] — 0.86.1

### Changed

- **The ControlNet workflow now detects its own control passes.** An `AIO_Preprocessor`
  sits between the video loader and the apply node, defaulting to
  `DepthAnythingV2Preprocessor` at 768, so raw footage goes in. Its dropdown covers all
  five conditions the union checkpoint accepts — canny, depth, HED, M-LSD and
  OpenPose/DWPose — and `none` passes an already-rendered pass straight through.

- **Documentation corrected.** The previous entries said no detector ships and the
  workflow expects a finished pass, on the strength of a single directory-name check
  that missed `comfyui_controlnet_aux` entirely. Five preprocessor packs were installed
  the whole time. The dependency table now names the pack and the node rather than
  telling people to go find one.

## [Unreleased] — 0.86.0

### Added

- **`MMH3ReferenceMultiPrompt`: `window_ref_video`, `chunk_frames`, `overlap_frames`
  (appended LAST, after `unload_text_encoder`).** Cuts the reference video and its
  soundtrack to each chunk's own span, so chunk *i* is conditioned on the footage it
  renders rather than the whole reference every time. Requested by **xwsswww** in
  `minimax_h3_resources`.

  Windowing only the DiT latent would have been cheaper and wrong: the reference also
  goes through Qwen as `<Video k>: ` plus a vision block per 2 frames with timestamps,
  so the text would describe a reference the model was not handed. This re-encodes per
  chunk instead, which keeps the conditioning on distribution — the point of doing it
  at all.

  **The cost is smaller than it sounds, and negative at sampling time.** N encodes
  happen inside ONE text-encoder load, so it is N forward passes rather than N model
  swaps. The vision work is partitioned, not duplicated — each chunk encodes 1/N of
  the frames. And reference tokens are attended at EVERY step, so a per-chunk window
  cuts what each chunk carries: measured **latent_t 177 -> 57, ref_audio_t 1000 -> 320,
  about 32%**, for a 600-frame reference in four windows.

  Spans come from `_plan` + `_window_frame_spans` with `"standard_static"` — the same
  call the sampler makes, copied rather than re-derived, because if they drifted apart
  every chunk would condition on the wrong footage silently. Raises if `chunk_frames`
  is 0, and warns when the window count does not match the prompt count.

  The soundtrack is cut on the **same clock** (seconds), not by latent arithmetic:
  24 fps against 40 Hz is not additive, so deriving one from the other accumulates
  drift. Measured 0.00 frames of drift across all four windows. Timestamps restart at 0
  within each window.

  Off by default, and off is byte-identical to before. Verified that all nine saved
  workflows still map their 7 stored widget values onto the first 7 widgets — the three
  new inputs went in **before** `unload_text_encoder` on the first attempt, which would
  have rebound that trailing `True` onto `window_ref_video`.

## [Unreleased] — 0.85.1

### Fixed

- **`MMH3_LoopingSampler_MusicVideo.json` carries its references as a LIST.** It was
  the only workflow in the repo batching them — core's `BatchImagesNode`, whose helper
  says outright `# resize all images to be the same size as the first image` and runs
  `common_upscale(..., "center")`.

  All three loaders are `resize: False`, so the images arrive at native size, and two of
  the three are `_crop.png`. So references 2 and 3 were being resized and centre-cropped
  to reference 1's frame on every render this workflow has ever done. Replaced with two
  chained `ImageTensorList` nodes, order preserved, so `<Picture 1..3>` still map to the
  same loaders.

  The other ten workflows were checked and need no change: seven wire a single
  `LoadImage` / `LoadAndResizeImage`, and four leave `ref_images` unwired.

## [Unreleased] — 0.85.0

### Added

- **`MMH3ReferenceMultiPrompt.ref_images` accepts a LIST as well as a batch**, so
  references of different shapes keep their own geometry.

  A batch is one tensor and a tensor cannot be ragged, so every batching node conforms
  its inputs first — core's `ImageBatch` and KJNodes' `ImageBatchMulti` both run
  `common_upscale(..., "center")` against image 1's height and width. References that
  were not already the same shape arrived here **resized and centre-cropped**, and the
  per-image sizing in this node then correctly sized the damage. Nothing downstream can
  detect it, because by then every image genuinely does share one frame.

  KJNodes' `ImageTensorList` returns a Python list typed IMAGE and chains to any depth;
  `_build_refs` now iterates either form. Each list entry gets its own aspect-preserving
  `tw x th`, which is what core's per-socket node does. Verified: portrait 576x1024,
  wide 960x540 and square 800x800 through a list come out at three distinct aspect
  ratios; the same three through a batch come out identical.

  `None` entries in a list are skipped rather than raising. The conditioning
  fingerprint hashes a list entry by entry — `repr()` of a tensor list is truncated, so
  two different reference sets could otherwise share a cache key — and reference ORDER
  changes the fingerprint, because it changes which image is `<Picture 1>`.

### Changed

- **The node reports what each reference actually became**, e.g.
  `<Picture 1..3>: 576x1024->576x1024  960x540->960x544  800x800->800x800`, and says so
  explicitly when more than one reference arrives as a batch — the one case where the
  damage happened upstream and cannot be reported any other way.

## [Unreleased] — 0.84.2

### Changed — documentation

- **Requirements rewritten for what the pack now actually needs.** The old text said
  "no patches, no carried diffs", which stopped being true the moment the ControlNet
  work landed:

  - **PR #15860 is a carried diff, and a DRAFT** — the only one the pack asks for, and
    the only one that can move underneath you. Called out with the `git apply` line, and
    scoped: everything except `MMH3CondSetApplyControl` and the ControlNet workflow
    still runs on stock.
  - **The union checkpoint** (~2.1 GB int8_convrot, or 3.9 GB bf16) and where it goes.
  - **A preprocessor pack is required and does not ship here.** The ControlNet takes a
    DETECTED control video — canny/depth/HED/MLSD/pose — not raw footage. Nothing
    stated this, and it is the thing most likely to waste someone's afternoon.
  - **`av>=17.0.0`**, because current core raised its own floor and the failure is
    `cannot import name 'ColorPrimaries' from 'av.video.reformatter'` at startup, which
    reads like a broken install rather than a version bump.
  - The control video must cover the WHOLE clip, and the checkpoint is
    guidance-distilled — guidance 1.0 through a `BasicGuider`, not CFG.

## [Unreleased] — 0.84.1

### Added

- **`workflows/MMH3_Looping_I2V_ControlNet.json`.** The ManualPrompt graph with a Fun
  ControlNet on the generate pass — copied rather than authored, so the sampler ladder,
  schedule, upscale passes and save chain are untouched. `ControlNetLoader` + a control
  video feed `MMH3CondSetApplyControl`, which sits between the multiprompt node's
  `cond_set` and sampler 269.

  The two refine samplers are deliberately NOT given the control: they run at low
  denoise off zeroed conditioning, where a control video would be fighting a picture
  that already exists.

  Vetted like the others: 0 link errors, no dead sockets, widget order checked against
  the node's schema.

## [Unreleased] — 0.84.0

### Added

- **`MMH3CondSetApplyControl` — "MMH3 Cond Set Apply ControlNet",
  `MMH3Tools/conditioning`.** Applies a MiniMax H3 Fun ControlNet across a cond set,
  and makes it chunk-aware.

  Two problems, one node. Core's apply node is `CONDITIONING -> CONDITIONING` while
  this pack's sampler takes a cond set, so they never meet — that part is plumbing,
  and core's shaping logic is called per cond rather than duplicated.

  The other is a silent-wrong. `MiniMaxH3ControlNet.get_control` selects hint frames
  with `torch.arange(pixel_t)` from ZERO in three places (control video, inpaint mask,
  source video) and invalidates its cache on `cond_hint.shape[2:]` alone. Both are
  correct for one whole-clip pass. Under chunking every chunk shares a shape, so chunk
  0's encode is reused for all of them and every chunk is driven by the control video's
  OPENING frames — plausible output, no error.

  Rather than reimplementing `get_control` or overriding `_fit_frames` — which would
  catch the control video and source but miss the mask, whose `arange` is inline — the
  wrapper hands core a WINDOW: the three inputs are sliced to the chunk's span before
  delegating, so arange-from-zero is right because zero is now the chunk's first frame.
  The cache is also invalidated whenever the offset moves.

- **`MMH3LoopingSampler` publishes `transformer_options['mmh3_control_frame0']`** per
  chunk, on both the main and phase-2 guiders. Computed with `frame_at_latent(v0)`, not
  `latents_to_frames` — window bounds are arbitrary latent indices and
  `latents_to_frames` is only meaningful on the 5j+2 grid, where it answers -12 for
  index 1. `model_options` and its `transformer_options` are rebound per chunk rather
  than mutated, since `copy.copy(guider)` shares them — the same shallow-copy trap this
  file already documents for `original_conds`.

  Verified: chunks at latent 0 / 14 / 28 see control frames starting at 0 / 47 / 94,
  each re-encoded, and an absent or zero offset passes through unwindowed.

  > Built against [PR #15860](https://github.com/Comfy-Org/ComfyUI/pull/15860), a
  > **draft**. The node checks for each internal it windows and refuses with a message
  > if core renames one.

## [Unreleased] — 0.83.1

### Fixed

- **The ref-label wrap works again on post-#15808 cores.** #15808 (merged 2026-08-22)
  rewrote `MiniMaxH3Tokenizer.tokenize_with_weights` to route text through
  `self.qwen3vl_32b.tokenize_with_weights(..., disable_weights=True)` and **deleted the
  private `_text_ids()` helper** the wrap called in three places. Its self-test caught
  this and declined to install — which is the design working: labels would otherwise
  have been dropped silently. Now routed through a `_text_entries()` shim that prefers
  core's current path and falls back to `_text_ids` on older cores, so the wrap tracks
  future moves instead of hardcoding one shape.

### Changed

- **`MMH3OfficialTokens` is superseded by core.** #15808 adds all seven tokens at
  tokenizer init. The node already short-circuits when `<d>` is 151669, so on a current
  ComfyUI it passes the CLIP through with *"already patched, nothing to do"*; it stays
  for older cores. Verified on `v0.33.0-49`: all seven ids correct, and
  `<d>[English] hello.</d>` tokenizes to **7 ids** against the 15-with-debris the
  unpatched tokenizer produced.

  Also of note for anyone reading the README's requirements: #15808 is what makes
  `embedding:` resolve in H3 prompts at all, since the same rewrite is what routes text
  through the tokenizer that parses it.

## [Unreleased] — 0.83.0

### Added

- **`WhisperAlignmentToText` — "Whisper to Text (LLM Ready)", `MMH3Tools/audio`.**
  Adopted into the pack. It previously lived in a loose, unpublished
  `ComfyUI-WhisperAlignmentToText` folder, which the README listed as a MusicVideo
  dependency with no link — the only row in that table without one. Somebody asked
  where to find it in `minimax_h3_resources` on 2026-08-24 and got no answer, because
  there was nowhere to point them: the MusicVideo workflow could not be run as
  written by anyone but its author.

  **The node id is unchanged**, so `MMH3_LoopingSampler_MusicVideo.json` keeps working
  and its stored widgets (`[5, '[M:SS]', 'per_line', False]`) still line up — verified
  against the new schema's input order. Anyone still carrying the loose folder should
  delete it, since two packs would otherwise register the same node id.

  Ported behaviour is **identical**, not merely equivalent: checked against the
  original implementation across 60 random alignments spanning every timestamp format,
  output format, interval (including 0) and the timing-data toggle — 0 mismatches on
  all three outputs.

  Its sibling `WhisperAlignmentToSegments` was deliberately **not** adopted. It cuts on
  25 fps and a 4n+1 frame grid, which is LTX's, not H3's 24 fps / 17j+5; MMH3 Window
  Plan and Split Audio to Windows already segment on the right grid. It was also
  unused by every workflow in the repo.

## [Unreleased] — 0.82.0

### Added

- **`MMH3RefAttentionProbe` / `MMH3RefAttentionMap`, `MMH3Tools/utils`.** Which
  reference is each part of the clip attending to, as a `[reference × time]` heatmap.

  Possible because H3 has **no cross-attention** — `grep -c cross_attn
  comfy/ldm/minimax/model.py` returns 0, references sit in the same sequence, so
  attention onto a reference's key rows is a real quantity. The spans are read, not
  guessed: `model.py` lays the sequence out as `("ref_audio", rt * 2)` segments, one
  per reference in order, exposed by `PackedLayout`; the probe captures them by
  patching `PackedLayout.__init__` and keying on `id(position_ids)`, the same
  mechanism Sol-Attn uses for its conditioning sink. The attention override chains any
  existing one, so it runs alongside Sol-Attn.

  **The denominator is exact**, streamed over key chunks inside a memory budget. The
  first cut pooled non-reference keys into centroids the way sol_attn pools its
  routed-out tail, and that is wrong for this measurement: logsumexp over a key block
  is near its MAX while a centroid is its MEAN, so a row attending one sharp key
  elsewhere in the clip had its tail understated and both references reported ~0.50
  when the truth was ~0.005. Caught by a synthetic case where the target rows point at
  each other and both references must read ~0. Only the QUERY side is pooled — one
  centroid per 64 rows, sol_attn's own routing granularity at ~5e-4 cosine.

  **The report judges per moment, never on the time average.** Under a working binding
  the two references average out equal, so an average-based test called clean
  alternation "the signature of no binding" — the exact opposite of the truth. It now
  reports the per-moment margin, the number of lead changes and each reference's share
  of the clip, separating: no reference ever leads / one reference leads the whole clip
  and never hands over / the lead alternates.

  Motivated by `minimax_h3_chatter`, 2026-08-24, where **foxydits** swapped which file
  went into `<Audio 1>` and `<Audio 2>` and the voices stayed swapped — *"the model
  has made the decision that no matter what I prompted … THAT SPECIFIC WRONG VOICE is
  character X's"* — and **ᴊɪɢᴇɴ** independently reported ~90% wrong even when flipping
  input order and subject definitions.

  **Attention mass is where the model LOOKED, not what it took**, and binding to the
  wrong reference looks identical to binding to the right one. Verified on synthetic
  attention only; it has not yet been run against a real render.

## [Unreleased] — 0.81.0

### Fixed

- **All audio is normalised to STEREO before it reaches H3's audio VAE.** Traced by
  **fredbliss** in `minimax_h3_chatter`, 2026-08-22: *"you just do NOT want to pass
  mono audio encoded into h3 … it does not like that nor expect it. convert to
  stereo."* and *"sglang also fails on mono audio btw — which makes sense. it needs
  stereo."*

  Core's `_encode_ref_audio` does `audio_vae.encode(waveform[:1].movedim(1, -1))` with
  no channel check, so a `[B,1,L]` track encodes without complaining and the model
  gets something it was not trained on — silent on this side, refused outright by
  sglang. Plenty of ComfyUI sources are mono: a voice recording, a `LoadAudio` of a
  mono file, anything a separator emitted as one channel.

  Applied at all three encode sites: `use_input_audio`, a reference video's
  soundtrack, and each `ref_audios` entry. Mono is duplicated (identical content on
  both sides, which is what mono means). More than two channels are summed into both
  sides with a warning that the stereo image is gone — an even/odd split looks more
  like a real downmix and is worse, since channel 2 in WAV order is CENTRE and
  dialogue would land on one side only.

### Changed

- **`use_input_audio` trims the waveform to the clip BEFORE encoding it.** It used to
  encode the whole track and drop the latents past the end: identical result, and a
  five-minute track for a sixty-second render meant five minutes of VAE encode to
  keep one. A margin of 8 latents is left on, so the latent-side cut still decides
  the final length and still lands on the grid.

  Reference audio is deliberately NOT trimmed — a reference is chosen, not derived,
  so its length is the user's call — but anything over 30 s now reports its cost,
  since reference tokens are attended at every step of every chunk.

## [Unreleased] — 0.80.1

### Changed

- **All nine workflows carrying `SolAttnMiniMax` moved to the v3 node**, the file
  linked at the bottom of [comfy-kitchen PR #117](https://github.com/Comfy-Org/comfy-kitchen/pull/117).

  v3 keeps `node_id="SolAttnMiniMax"`, so it cannot coexist with v2 — but the INPUT
  LIST changed, which is why Kijai's instruction was "old node is dead, need the new
  file (v3) and remake the node in workflows". `tau` moved under a new `selection`
  dynamic combo (`adaptive tau` | `top-k (SLA)`), `keep_percent` arrived with it, and
  `routed_cap_percent` was removed. Widget values serialize positionally, so carrying
  v2's list across would have put `tau` into `selection` and `start_percent` into
  `tau`, silently.

  Values were remapped **by name**, not position, and `routed_cap_percent` was 0 in
  every workflow so nothing was lost. All nine now read
  `["adaptive tau", 1.3, 0.2, 0.9, 12288, "exact_kv_and_rows", false, "2d_frame",
  true, false, true, "33-35, 39-42"]` (Upscale keeps its own `verbose`/`dense_blocks`),
  and the `inputs` array was rebuilt — `selection`, `selection.tau`, and the
  `selection.tau_profile` socket.

  The DynamicCombo serialization was taken from a real saved graph rather than
  guessed: core's `SaveAudioAdvanced` stores `['audio/...', 'mp3', 'V0']` — linear, in
  schema order, only the SELECTED option's nested widgets, namespaced
  `format.quality`. Verified afterwards against the installed v3 schema: widget order,
  socket set, value count and model links all check out on all nine, 0 problems.

## [Unreleased] — 0.80.0

### Added

- **`MMH3ChunkScheduleFrames` — "MMH3 Chunk Schedule (Frames)", `MMH3Tools/calculators`.**
  MMH3 Chunk Schedule asked in FRAMES rather than seconds. Requested by a user who
  already holds frame counts and does not want a duration rounded on the way in.

  The solver already worked entirely in GROUPS — `seconds_to_groups` was the only time
  conversion anywhere on the input side, and the tiling search, `av_align` and the
  reachable-overlap ladder are all unit-free. So this needed exactly one new
  converter, `frames_to_groups`, and nothing else.

  **It still snaps.** Values land on the nearest `17j+5` and are still solved
  together, so 1445 and 1446 both resolve to 1450. Skipping the time conversion is
  not the same as accepting arbitrary frames, and the node says so in its
  description. Defaults (1433 / 481 / 73) mirror the seconds node's 60.0 / 20.0 / 3.0
  exactly, so the two agree out of the box.

### Changed

- **The two schedule nodes share one implementation.** `execute` was split into a
  module-level `solve_and_report(c_req, a_req, b_req, prefer, chunks, av_align,
  asked_line)`; each node converts its own request into groups and supplies its own
  "asked for" line, and everything after that is common. Copying a 100-line report
  would have let the two drift, and the drift would be silent — the numbers would
  still look plausible.

  **`MMH3ChunkSchedule`'s behaviour is unchanged**, verified against a 288-case
  golden snapshot taken before the refactor (4 lengths x 2 windows x 2 overlaps x 3
  `prefer` x 2 chunk counts x 3 `av_align`), comparing all five numeric outputs and
  the full report text: 0 mismatches.

## [Unreleased] — 0.79.0

### Added

- **`MMH3StreamingSave`: `save_metadata` (append-only, added LAST), default on.**
  Embeds the workflow and prompt in the mp4, so dragging the file back into ComfyUI
  restores the graph — the node previously declared no `hidden` inputs and so never
  saw either.

  Two things decide the implementation, and both are easy to get silently wrong:

  - **It cannot be a command-line argument.** A real workflow is 45–95 KB
    (`MMH3_LoopingSampler_Masking.json` is 45,028 chars) and Windows caps a command
    line near 32,767, so the tags go in an **ffmetadata file** read as a second input.
  - **mp4 drops unknown tags** without `-movflags use_metadata_tags`. Core sets the
    same flag for isobmff. Without it `workflow` and `prompt` vanish with no error.

  Metadata is attached to the FIRST pass, so both endings inherit it: the silent
  path's `os.replace` keeps the file as-is, and the audio mux carries the tags through
  `-c:v copy`. Verified round-tripping the real workflow through both paths — parsed
  back identical, emoji in node titles intact.

  **`faststart` is deliberately not used.** It relocates the moov atom by rewriting
  the whole file at the end, which would undo this node's constant-cost promise on
  exactly the long renders it exists for. Core can afford it because its saver holds
  the video in memory anyway.

  ComfyUI's `--disable-metadata` wins over the widget. Turn the widget off for files
  you are sending out: the workflow carries every prompt and path in the graph.

## [Unreleased] — 0.78.1

### Fixed

- **`MMH3StreamingSave` no longer needs torchaudio (or TorchCodec) to write audio.**
  Reported by a user as `TorchCodec is required for save_with_torchcodec. Please
  install torchcodec to use this function.`

  Since **torchaudio 2.9**, `torchaudio.save` routes through TorchCodec
  unconditionally. TorchCodec is not a ComfyUI requirement, and where it *is* present
  its Windows build frequently cannot load its FFmpeg DLLs — a hard failure at the
  very end of a long render, on machines that have a working ffmpeg binary sitting
  right there. The node already spawns that binary to write the video, so the audio
  now goes through the same door: the waveform is written as raw interleaved `f32le`
  and handed to ffmpeg with `-f f32le -ar <sr> -ac <ch>`.

  No encoder library is involved, so there is nothing for a user to install and
  nothing to break the next time torchaudio moves its backend. Mono is passed through
  as `-ac 1` rather than forced to stereo. Verified end to end against ffprobe for
  both channel counts: correct codec, sample rate, channel count and duration.

  `nodes_align.py` had already been moved off `torchaudio.save` for this exact reason
  and carried the diagnosis in a comment; this was the call site that was missed.

## [Unreleased] — 0.78.0

### Added

- **Two example workflows.** `MMH3_LoopingSampler_Masking.json` — masked v2v, and the
  reference for masking the two halves separately: a SAM3 matte drives `denoise_mask`
  while the supplied track rides through on the Split AV pin, with nothing wired to
  `audio_denoise_mask` on purpose. `MMH3_Looping_I2V_ManualPrompt.json` — image-to-video
  with the prompt typed rather than generated, and the fullest example of the
  three-stage ladder (generate, then two pixel-upscale refine passes, the last holding
  audio under a zero `SolidMask`).

  Both were saved from a session running the pre-0.78.0 schema, so ManualPrompt
  carried `prior_av_latent` / `speed_schedule` sockets that no longer exist; stripped
  with link target-slot renumbering before landing.
- **`MMH3LoopingSampler`: `denoise_mask`, `denoise_mask_mode`, `audio_denoise_mask`
  (append-only, added after `vae`).** A MASK over the whole clip: white regenerates,
  black keeps the input latent.

  It reduces once onto the master grid and merges keep-wins into whatever mask the
  latent already carried, BEFORE the chunk loop — so `_sliced_mask` and `_carry_mask`
  are untouched and the carry composes with it for free.

  The reduction follows the VAE, not a resize, which is where this differs from
  handing a pixel mask to `SetLatentNoiseMask` (core trilinear-interpolates that
  across time, blurring the mask between frames):

  - **spatial**: adaptive pooling, then snapped to the **2×2 latent patch** the DiT
    reads the mask through. Bilinear would average and produce a fractional value on
    every edge, and each such cell denoises at its own timestep (`rows_t = 1-m*sigma`).
    32 pixels is therefore the finest expressible feature.
  - **temporal**: grouped on the real `FRAME_PER_TOKEN` cycle `(1,4,4,4,4)` via
    `frame_at_latent`. The first latent of each 17-frame group covers ONE frame and
    the other four cover four each, so a uniform 17/5 split misplaces an edge.
    Verified: a mask white on pixel frame 0 alone frees latent 0 and no other.
  - **audio**: NOT touched by `denoise_mask` at all. `audio_denoise_mask` is the only
    input that masks the audio half; only its time axis is read, mapped through
    `_audio_index_at` so a frozen span lines up with the picture, and written on the
    audio latent's own axes, temporal on **dim 3**. Unconnected, audio is masked only
    by what the latent already carried.

    An earlier cut of this feature *derived* the audio profile from the video mask
    when no audio mask was wired, so the two modalities could not disagree about a
    frozen span. That reasoning only holds for a mask with temporal intent. A spatial
    mask — a subject matte — is white somewhere in every frame, so the reduction
    returned "free" at every timestep and regenerated the entire track. Measured on a
    SAM3 matte: 75% of the video grid held, 100% of the audio freed. Removed before
    release; the halves are independent.

  Refuses on a core without #15375, where a mask is accepted and silently ignored;
  warns when the input latent is all zeros, since kept regions would pin black.

  Geometry cross-checked against drozbay's MaskVidExperiments, which handles H3
  explicitly and independently arrives at the 2×2 token snap and the causal frame
  cycle.

- **`MMH3SplitAV`: `preserve_masks` (append-only, added LAST).** Default ON. Hands
  each output half its own `noise_mask`, unbinding the pair when the input carries one
  and passing a plain tensor straight through for a video-only latent.

  `MMH3PackAV` has always had a branch for carrying masks back into a pair, but Split
  returned bare `{"samples": ...}` dicts, so a split/repack round trip had nothing
  left to re-pair. The visible consequence: `use_input_audio` installs a mask pinning
  the supplied track, and any graph that split the latent to operate on the video half
  handed the sampler an unmasked track, which was then regenerated. Off restores the
  old lossy behaviour and reports that it dropped a mask.

  **Default chosen as ON deliberately.** The only graphs whose output changes are ones
  whose input latent actually carried a mask — exactly the graphs that were already
  losing it.

- **`MMH3OfficialTokens` — "MMH3 Official H3 Tokens", `MMH3Tools/conditioning`.**
  Adds H3's seven added special tokens (`<d>` 151669 … `<|caption_end|>` 151675) to a
  CLIP's tokenizer.

  ComfyUI's chain is `MiniMaxH3Tokenizer` -> `Qwen3VLSDTokenizer` -> `qwen25_tokenizer/`,
  and that directory is byte-identical to stock Qwen3-VL: 151,643 vocab entries and 26
  added tokens ending at 151668. H3 ships seven more and its model card states the H3
  tokenizer config is required. Without them `<d>` tokenizes as ordinary subwords which
  merge with adjacent whitespace, language tags and punctuation — `' <' 'd' '>['` and
  `'.</' 'd' '>'` — a different token sequence and different hidden states. Traced and
  reported by **fredbliss** in `minimax_h3_chatter`, 2026-08-21.

  Implementation notes: it patches a **deep copy** of the tokenizer, because
  `CLIP.clone()` assigns `n.tokenizer = self.tokenizer` — an in-place patch would stick
  to the loaded model and survive bypassing the node. It verifies each token landed on
  its documented id rather than trusting that appending to a 151669-long vocab lands
  right, and raises without modifying anything if not. It refuses a non-H3 CLIP, since
  the same seven ids are live vocabulary in other models. `enabled` off is a
  pass-through so it can stay wired for A/B.

  **Whether it improves output is unmeasured.** The embedding rows exist
  (`[151936, 5120]`) so the ids are in range, but whether MiniMax trained them is
  unknown, and the evidence so far is three A/B pairs from one person on one setup,
  with a measurement that contradicted the listening. This node exists to make the
  comparison runnable, not because the answer is in.

- **`MMH3MusicScenePlanPrompt`: `music_source` and `treatments` (both append-only,
  added LAST, defaults preserve existing behaviour byte for byte).**

  **`music_source`** — `supplied` (default) / `generated`. The node was written for
  the case where the track exists and is handed to the sampler, and it says so in the
  rules: *"THE SONG IS THE AUDIO"*, *"you are describing something that already
  exists and will be supplied as audio"*. Pointed at a graph where H3 writes the
  audio in the same pass, it faithfully produced prompts describing a track that
  would be supplied, and **quoted no lyrics at all** — the sung words reached the
  writer and were spent on deciding what the picture did. The result was coherent
  instrumental music under a mouth that sang nothing.

  `generated` swaps two rule blocks and adds a third. `overall_soundscape` stops
  claiming a track was provided and becomes ambience-and-action-sound only.
  `non_diegetic_music` becomes the SPEC the model performs — genre, hedged tempo,
  instruments named with a playing style, the invariants restated every chunk, one
  movement clause. And the shots stage gains a block requiring the window's lyrics
  VERBATIM as `<d>[English] ...</d>`, attributed with SINGS rather than says, with
  the singing described physically and cuts timed to the voice. Without that block
  nothing asks for words and none are sung.

  **`treatments`** — `music video` (default) / `restrained`. The default pushes split
  frames hard (*"SPLIT FRAMES ARE A TOOL HERE, so reach for them on purpose"*,
  *"Known to render well: split frames"*), which is right for a lyric-video look and
  wrong when the subject is the performance: a divided frame halves the singer
  exactly when the mouth is the point. `restrained` forbids frame division, inset,
  banded overlay and multiplied performers, and swaps the effect menu for optical
  treatments only.

  The two are independent. Generated audio with music-video treatments is a valid
  combination; so is a supplied track shot restrained.


### Added
- **Two example workflows: `MMH3_Looping_Upscale` and `MMH3_Outpaint`.** Upscale is a
  refine pass over an existing render — chunked pixel upscale to an `MMH3UpscaleLadder`
  target, then `MMH3SplitAV` so the audio half can be re-packed under a zero `SolidMask`
  and held while the looping sampler re-samples only the video. Outpaint reframes a
  landscape clip to 9:16 via `MMH3OutpaintLatent` + `MMH3ReframePads`, sampled in one
  `MMH3ContextWindows` pass rather than through the looping sampler. Neither uses the
  prompt-building nodes, so RES4LYF is their only extra dependency.

- **`docs/looping-sampler.md` — new `Recipes` section (§8).** Six entries, one per
  shipped workflow, read off their saved widget values rather than recommended from
  tuning — the section says so, since nothing here is a measured optimum. Sections
  renumbered: Symptom -> lever 8->9, Observed 9->10, Not yet measured 10->11, with
  the three internal cross-references updated. Also corrects a pre-existing slip in
  the header, which pointed at "Section 9" for what is still unknown when that had
  become Observed.

  Writing it surfaced an undocumented tension, now logged under Not yet measured:
  the music-video workflow ships `overlap_strength_audio` at 1.0, which is the value
  Observed calls "tinny on chunk 2". That measurement was taken on T2VA with
  GENERATED audio; the music-video graph pins a real track, where full pinning is
  arguably the intent. The comparison has never been run.

- **`docs/music-video.md` — "MiniMax H3 Music Video — Field Guide".** The music-video
  chain had no doc beyond its README node entries, while the looping sampler,
  context windows and Regenerate-2K each had one. Same skeleton as
  `docs/looping-sampler.md`: mental model, the grid, alignment, music analysis, the
  three stages, typography, recipes, symptom → lever, `Observed`, `Not yet measured`.

  The **`Observed — 2026-08-15`** block moved out of README into §9 of the guide,
  where it sits next to the chain it describes rather than inside the node reference;
  README keeps a pointer. Nothing was reworded in the move.

  §10 is deliberately longer than §9: the chain has one full run behind it, and the
  guide says so at the top rather than implying more.

- **README: the example-workflow dependency list is now derived, not asserted.** It
  named only LlamaOmni and RES4LYF; the real set is eight packs, read off the
  `cnr_id` / `aux_id` each workflow records per node. KJNodes and VideoHelperSuite
  were needed by seven workflows and mentioned nowhere.

  It also names **`SolAttnMiniMax`** — Kijai's single-file Sol-Attn node (arXiv
  2607.24027), reaching H3's attention through comfy-kitchen's CUDA kernels, so it
  wants `comfy_kitchen` built with `sol_attn`. Seven of the eight workflows carry it.
  It is a speed override rather than pipeline logic, so anyone without it can delete
  the node and wire `ModelAttentionBackend` straight to the LoRA loader.
  `ModelAttentionBackend` beside it is comfy-core.

- **README: removed a stale `Requires ComfyUI v0.30.0+` line** four lines above the
  Requirements section, which says `v0.33.0-20-gff6c8a8a` or newer. The loose number
  was the one that would land someone on a core where `carry="mask"` fails silently.

- **Upscale-pass sampler settings corrected in three workflows.** Every pass whose
  latent comes from `MMH3ChunkedPixelUpscale` now runs `sampling_start_step 0`,
  `sampling_end_step 1000`, `phase2_start_step 0`, empty `keyframe_indices`.

  Two passes in `MMH3_Looping_I2V_PromptBuilding` held `sampling_end_step 0`, which
  is a zero-length window: the guard against that is nested under `if start > 0`, so
  at `start = 0` it never fires and the sampler returns its input latent unsampled
  with no error. Those passes were doing nothing. Sampler behaviour is unchanged —
  this is a settings fix only.

- **README: a "What has actually been run" block, above Requirements.** `carry="mask"`
  is the tested path and every observation in the docs was measured on it;
  `carry="keyframe"` has **never generated a clip**. The guide construction is
  unit-tested against a fake sampler — anchored at frame 0, multi-step clip plus
  audio, no mask — but that is structure, not output.

  It stays in the node because it is written and gated, not because it is
  recommended. Saying so at the top costs nothing and stops the `carry` dropdown
  reading as two equally exercised options.

  `docs/looping-sampler.md` now agrees in three places: the `keyframe` route
  description, the Not-yet-measured entry (rewritten from "whether keyframe beats
  mask" to "never run" — a different claim), and the symptom table, which had been
  offering `carry="keyframe"` as a remedy for visible seams without noting that
  trying it is an experiment.

- **Dead `prior_av_latent` / `speed_schedule` sockets stripped from the saved
  workflows.** Hiding an input from the node schema does not remove it from a graph
  someone already saved: the workflow file carries its own copy of each node's input
  list, and the frontend rebuilds from that. So both sockets kept appearing on the
  sampler after a restart even though the backend no longer served either.

  Removing an input is not a plain delete — the links table addresses targets by
  index (`link[4]`), so every link pointing past the removed slot has to be
  renumbered or the graph silently rewires. Each file was validated before and after
  (every input's link id must belong to a link whose target slot equals that input's
  index) and written only if it passed.

### Removed
- **`MMH3_I2V_2K` example workflow.** Long out of date against the current nodes.
  Its one finding that was not documented elsewhere — that windowing is *faster* at
  2K — is already carried in full by `docs/context-windows.md` ("Measured: windowing
  is FASTER at high resolution"), math included, so nothing is lost. Recoverable from
  history if it is ever wanted back.

### Changed
- **`MMH3LoopingSampler`: the `prior_av_latent` input is withheld from the schema.**
  The prior-continuation path never behaved correctly in practice, so the socket no
  longer appears on the node. **The code is not removed** — `execute` still accepts
  `prior_av_latent=None` and the prepend / `5j+2` re-grid / phase-offset / audio-drift
  implementation is intact; the schema entry is commented out in place and restoring it
  is uncommenting one block. It was the LAST input, so re-adding it appends and cannot
  disturb widget order.

  It is a socket rather than a widget, so no saved graph's `widgets_values` shift. A
  workflow that wired it loses that one link and nothing else; none of the workflows in
  this repo wired it.

- **`MMH3ScenePlanPrompt` — new `mode` toggle (append-only): `cinematic` (default) /
  `talking_head`.** `talking_head` swaps the escalation prompts for an ABSOLUTELY-LOCKED
  continuous take: the `shots` stage holds one fixed-tripod frame, forbids cuts / camera
  moves / new action, and writes a continuing spoken monologue instead of advancing a
  scene. In that mode it does NOT require a `beat_sheet`, `brief` becomes the monologue's
  topic (`_BRIEF_TH`, "what they talk about" rather than "never speak aloud"), and the
  continuity block is reworded to "same unbroken shot, carry the monologue forward"
  (`_CONTINUITY_TH`). Default `cinematic` → existing workflows untouched.

  The talking-head case has no cut to hide the chunk boundary, so it doubles as the
  cleanest test of whether the looping sampler joins chunks seamlessly at all.

- **`MMH3ScenePlanPrompt` — new `prev_detailed` input (append-only, `shots` stage).**
  Accepts the previous chunk's `detailed_description` and appends a CONTINUITY block
  telling the writer to open [Shot 1] on that chunk's FINAL frame -- same positions,
  poses, injuries, props, camera -- and advance, instead of opening a fresh scene.
  Wire it from `MMH3PromptAccumulate.prior_context` (mode `last`) through the loop's
  carried value; empty on beat 0. Optional and LAST in the input list, so existing
  saved workflows load unchanged; past beat 0 with nothing wired, the node warns in
  its report.

  Fixes chunk-to-chunk staging drift: the `shots` writer previously saw only the beat
  SHEET (summaries), never the previous chunk's realised output, so it continued the
  STORY but re-invented the STAGING -- camera and pose reset at every cut. Verified on
  a T2V render: later chunks now open "the camera snaps from [the previous chunk's
  final shot]", and injuries/positions carry monotonically across chunks.

- **`MMH3LoadSkill` — "MMH3 Load Skill", `MMH3Tools/prompt`.** One skill file in, text
  out, for `extra_rules` on either scene-plan node. Inputs `skill` (combo, populated
  from `styles/`), `previous` (force_input, for chaining), `enabled`; outputs
  `extra_rules`, `report`. New node, so no ordering constraint yet.

  Chainable rather than multi-select: selecting several in one node means deciding up
  front which kinds exist and how many of each you may have, and a chain decides
  nothing. The type lives in the filename, and `fingerprint_inputs` watches the folder
  mtimes so editing a skill file re-runs the node instead of serving a cached prompt.

- **`styles/` — 33 skill files.** Four `look-`, six `typography-`, twenty-three
  `experiment-`, plus `_README.md` (underscore-prefixed files are notes, not skills,
  and do not appear in the dropdown). The `experiment-` set is deliberately
  unvalidated — written for what we want to find out H3 can do — and the node flags
  them as untested in its report so a result is judged on its own.

  Not a port of MiniMax's nine published H3 skills: those are agent procedures for
  their own hub, wrapped around the visual guidance in numbered steps, confirmation
  gates and shot counts written for 15-second clips, none of which survives contact
  with a grid-locked window and a pinned master audio.

- **`MMH3ChunkSchedule` — "MMH3 Chunk Schedule", `MMH3Tools/calculators`.** Inputs
  `total_seconds`, `window_seconds`, `overlap_seconds`, `prefer`, `chunks`; outputs
  `total_frames`, `window_frames`, `overlap_frames`, `chunk_count`, `seconds_per_chunk`,
  `report`. New node, so no ordering constraint yet.

  **The frame calculator converts; nothing solved.** Asking it three times, once per
  value, cannot see the constraint that matters, because that constraint is a
  relationship between the three. Observed 2026-08-17: 60s / 20s window / 3s overlap
  passes every per-value check and produces four chunks whose last strides 7.08s
  against the others' 17.00s, re-rendering 12.2 seconds that a previous chunk already
  made under a different prompt. Widget precision is not the bug and fixing it would
  not have helped.

  With `t = 5c+2`, `L = 5a+2`, `O = 5b+2`, stride is `5(a-b)` — a multiple of 5 for
  any a and b, so phase safety is free and the five-window pulse cannot occur. The
  real condition is `(c - a) % (a - b) == 0`, which reduces to small integers, so the
  solver is a plain sweep rather than anything clever.

  `chunks` pins the count and makes the window a result. That is the number most
  worth holding an opinion about: prompts written, joins made. An unreachable count
  is released with a note rather than raised, per the pack's report-never-halt rule.

  **`av_align` — the 40 Hz audio grid.** H3 video runs are `17j+5` frames but audio
  latents tick at 40 Hz, and only every THIRD run is whole on both: 39, 90, 141, 192,
  step 51 frames. Off the grid the preserved audio pins to an instant up to a third
  of a tick (8.3 ms) from the preserved video, at the carry edge and again at every
  chunk start, since the STRIDE has to be exact too. `ignore` (default) is unchanged
  and flags when no rung on the ladder aligns; `prefer` ranks aligned schedules
  first; `require` returns only aligned ones.

  At 60s exactly there is NO aligned schedule at any window or overlap -- its total
  is 84 groups and alignment needs `2 mod 3` -- so `require` moves the total by one
  group before it gives up the chunk count, that being the cheaper concession. An
  earlier ordering released the count first and then honoured it anyway, so the
  report claimed a release that had not happened.

  Matters when chunks GENERATE their audio; a supplied master track pinned at
  mask 0 has no per-chunk audio seam to misalign.

  ⚠️ The ladder marks a rung reachable only if its WHOLE schedule qualifies. Marking
  by the overlap alone offered rungs the solver would never take: under `require` the
  stride must land on the grid too, which leaves every THIRD rung — 45 latents apart,
  not 15. Observed 2026-08-17: a 4s overlap request returned 1.62s while the ladder
  appeared to offer 3.75s. The rows now print their stride and say so.

  The report lists the **reachable overlaps** for the current chunk count. With the
  total and the count fixed, stride `(c-b)/n` must be whole, so valid overlaps sit `n`
  groups apart — the chunk count IS the overlap's step size, and more chunks means a
  COARSER overlap. Observed 2026-08-17: nudging the overlap jumped 17 to 32 latents
  with nothing between, which is the rule rather than a snapping bug.

  `tests/test_chunk_schedule.py` asserts the property that matters: every schedule it
  emits is fed back through **MMH3WindowPlan** and must come out with a single
  stride. The node agreeing with its own arithmetic would prove nothing.

- **`MMH3MotionOverload` — "MMH3 Motion Overload", `MMH3Tools/utils`.** Measures which
  latent time tokens of a rendered clip carry more motion than one token can
  represent. Inputs `latent`, `quantile` (0.75), `phase_normalize` (on); outputs
  `profile_json`, `hot_over_cold`, `report`. New node, so no ordering constraint yet.

  **Why it exists.** `FRAME_PER_TOKEN` is `(1,4,4,4,4)` — four of every five tokens
  span four pixel frames. When motion is fast enough that those four frames need four
  distinct poses, one token cannot represent them and the decode smears. The defect is
  structural, so it does not respond to steps or resolution: the poses were never
  generated, and re-denoising cannot recover what was never there. The community calls
  the artefact "roping" and the second-pass fix "de-roping".

  **This is the measurement half only.** It reads a finished latent and reports; it
  retimes nothing and fixes nothing. Built first, deliberately, because whether this
  footage has the problem at all is unmeasured here — the material is largely
  performance to camera, not the backflips and sword arcs the artefact was reported on.

  Method: third difference of the latent along the token axis, `|d3|`, averaged over
  channels and space, then phase-normalised. Prior art is the jerk oracle in
  matlowai/ComfyUI-MAINodes (MIT); reimplemented against `common.py`'s
  `frame_at_latent` rather than copied, and `tests/test_motion_overload.py` asserts
  our grid agrees with theirs token by token.

  **`phase_normalise()` is reusable and applies beyond this node.** A phase-0 token
  spans one pixel frame and phases 1–4 span four, so a raw per-token statistic is
  measuring the grid as much as the signal. Any future per-token score on H3 needs it.

- **`workflows/` — five looping example workflows added, old `MMH3_Looping_T2V`
  removed.** `MMH3_Looping_Cinematic` (successor to `MMH3_Looping_T2V`, wiring the new
  `prev_detailed` continuity feed), `MMH3_Looping_Monologue` (`talking_head` mode),
  `MMH3_Looping_I2V_PromptBuilding`, `MMH3_LoopingSampler_MusicVideo`, and
  `MMH3_LoopingSampler_Regenerate2K`. README's Example workflows section documents each.

### Changed
- **#15375 merged upstream 2026-08-18, so the pack now needs NO core patches.**
  Minimum ComfyUI is `v0.33.0-20-gff6c8a8a`. The Requirements section said "two
  upstream PRs that are still open" and led with a warning about applying diffs; it
  now says stock ComfyUI and a version number.

  **#15316 is out of the docs entirely.** It was always listed as needed by "nothing
  -- optional" (it reserves text-encoder VRAM and removes a minute-long hang when
  conditioning carries image references). Documenting an optional third-party diff in
  a Requirements table reads as a requirement. Anyone who wants it can fetch it.

  `docs/core-changes.md`'s PR table is now history rather than instructions, and its
  fetch-and-apply recipe is scoped to reviving an old build.

  ⚠️ **The merged version is not byte-identical to the PR head.** It pools to the token
  grid BEFORE quantizing and uses `torch.ceil` where the PR used `torch.round`, so
  every partial mask value now rounds UP toward generate. 0.0 and 1.0 are unaffected;
  anything between them is. Re-baseline before comparing a partial-`overlap_strength`
  run against one made earlier.

### Changed — prompts
- **The wardrobe is now REQUIRED in `subject_definitions`.** The rule listed clothing
  as one example of visible permanent detail -- "build, hair, clothing, markings" --
  and the writer skipped it, producing a subject defined by hair, eyes and skin
  markings with nothing worn. Same failure shape as the `[Shot 1]` marker: mentioned,
  never demanded.

  It has to live there because `subject_definitions` is the only section repeated
  byte-identically in every chunk. An outfit established in chunk 0's
  `detailed_description` is invisible to chunks 1..N by construction -- that section is
  the one thing that varies -- so the subject changes clothes partway through the film.

- **`talking_head` shots may no longer introduce persistent appearance.** Wardrobe,
  hair, build and markings come from the definition and are not re-described or
  changed. Action may still touch them -- a sleeve pushed back, hair moved off the
  face -- since that is the shot doing its job rather than redefining the subject.

  Cinematic mode is deliberately untouched: there a costume change can be a real beat,
  and the same rule would fight the story it exists to tell.

- **`talking_head` shots now REQUIRE the `[Shot 1]` marker.** The rule described the
  marker -- "[Shot 1] carries NO timestamp… do not write a [Shot 2]" -- without ever
  asking for one, so the writer took the actionable half and emitted no shot marker at
  all. `MMH3PromptLint` had been reporting "detailed_description has no [Shot 1]" on
  every chunk and it was correct.

  `[Shot N]` is how H3 is told where shots begin and end. With no marker anywhere,
  nothing in the text declares the chunk to be one continuous shot.

- **A continuation chunk may no longer OPEN.** Chunks after the first now carry: the
  take is already running, no fade in, no cut from black, no light coming up, no
  camera settling, no reveal. Chunk 0 is untouched -- it IS the opening -- and
  cinematic mode is untouched.

  Observed 2026-08-17 on `stream_00159`/`00160`: at the first generated frame of chunk
  1, luma fell **97% in a single frame** (0.7627 -> 0.0240) and recovered over ~3 s
  with the subject's neon circuitry the brightest thing in frame -- H3 opening the
  chunk like a new shot. The carried head measured +1.5% against baseline, so the pin
  was holding; the model simply treated the clip start as a place to make an entrance.

  ⚠️ **Not proven to be the whole cause.** Chunk 2's boundary in the same renders is
  clean (-0.0002), with the same missing marker and the same carry, so something
  chunk-1-specific is also involved. These rules close a real gap in the instructions;
  they are not established as the cure.

### Fixed — documentation

- **`docs/regenerate-2k.md`: `ref_downscale` is closed as a cost lever.** Measured
  2026-08-21 — `2x` came back much worse. The cost arithmetic was right (references
  are attended at every step, so 2x cuts their cost ~4x) but the saving is not
  spendable: this route is in-context regeneration, not super-resolution, so the
  reference IS the detail the 2K pass reads back out. Since the options are only
  `none`/`2x`/`4x`, the gentlest setting is the one that failed and there is nothing
  milder left to try. Moved out of the open-questions list with the answer attached.
- **`docs/regenerate-2k.md` — recorded the model-card vs AMA question on Regenerate-2K,
  with a reconciling reading flagged as conjecture.** The model card says the 2K pass
  *"uses the H3 base model to regenerate its own low-resolution result"* (no separate
  weights); the r/StableDiffusion AMA (Kiro) describes *"a dedicated … regeneration
  checkpoint … not simply the current H3 checkpoint running a second time"* they are
  *"tuning … so it can run locally."* New §1 subsection *Model card vs the AMA* records
  both as sourced fact, notes that *"latent-space DiT checkpoint"* does not by itself
  distinguish it from base and that the AMA wording is a translated paraphrase (Reddit
  uncrawlable), then gives — **explicitly as conjecture** — the reading that the two
  describe different artifacts in time: card = the base-at-2K method they run today (what
  this pack reproduces), AMA = a *lighter* dedicated/distilled, sparse-native checkpoint
  they are building for local use. Stated flat only: the sources, and that a single
  base-at-2K pass produces a correct result here (validated to 8s). The harness is
  checkpoint-agnostic (gets *faster* if the lighter model lands); the three-modality-row
  tensor argument bounds H3-Base only; §6's divergence is unaffected. Intro carries a
  one-line pointer.

- **The feather's recorded MECHANISM was wrong; the decision was not.** 0.72.2 and
  0.73.0 explained the noisy seam as `rows_t = 1 - m*sigma` and the content blend
  `x*m + orig*(1-m)` corresponding "only approximately". Re-read against core
  2026-08-17, they correspond closely, and since #15375 merged
  `scale_latent_inpaint` pre-compensates so every pixel lands at its token's pooled
  strength BY CONSTRUCTION. Whatever caused that seam, it was not this.

  Corrected in `docs/looping-sampler.md`, `MMH3LoopingSampler`'s docstring and
  `MMH3OutpaintLatent`'s. `feather_latents` stays removed -- setting it to 0 removed a
  visible seam, and that observation is untouched. The CHANGELOG entries that first
  recorded it are left as written; they are history.

- **Dropped the `>=0.995 -> 1.0` / `<=0.05 -> 0.0` snapping claim from the README.**
  True of the PR, removed upstream on 2026-08-15, and it was being used to argue that
  hard masks are safe.

- **`MMH3WindowPlan`'s outputs now carry their units in their names.**
  `context_length` -> `context_length (latents)`, `context_overlap` ->
  `context_overlap (latents)`, and the frame-domain outputs gain `(frames)` to match.
  **Display names only — slot ORDER is unchanged, and links serialise by slot index,
  so saved workflows are unaffected.**

  Observed 2026-08-17: `context_length` (117 latents) wired into the looping sampler's
  `chunk_frames`, which takes frames. Nothing errored — 117 is a legal frame count —
  it re-snapped to 32 latents and the sampler ran **11 chunks of 4.46s with a 0.21s
  carry** while Chunk Schedule and Window Plan both reported 3 chunks of 16.50s. The
  latent pair and the frame pair sit five sockets apart because the frame outputs were
  appended later, which is what makes them easy to cross.

### Fixed

- **`MMH3PromptPart` warns when the text never split.** With one piece and `clamp`
  on, EVERY index returned the whole body, so each chunk was spliced with the entire
  beat sheet instead of its own beat — silently, and the damage surfaced downstream as
  an apparent model failure rather than here as a split failure. Now logs a warning and
  puts `NO SPLIT` in the node's notes. Behaviour is otherwise unchanged: this is a
  diagnostic, not a raise.
- **A prompt nested inside its own `detailed_description`.** Observed 2026-08-17: chunk
  0 was clean, every chunk after it contained a complete six-section prompt spliced
  into its own body, with a second `overall_soundscape` / `non_diegetic_music` pair
  trailing after. It rendered without erroring anywhere.

  Cause: `_CONTINUITY_TH` opens "Below is the PREVIOUS chunk's detailed_description",
  but the thing the pack tells you to wire into `prev_detailed` -- `MMH3PromptAccumulate`'s
  `prior_context` -- emits a whole prompt, since it returns the last piece split on
  `|`. Shown a complete prompt under a label calling it one section, the writer
  matched the format it was given and returned a complete prompt. Chunk 0 escaped
  because it has no prior context to imitate. The pack was contradicting itself: the
  docstring said to wire exactly the thing the label misdescribes.

  **`prev_detailed` now extracts `detailed_description`** from whatever it is handed
  (`_prev_body`), passes a bare body through untouched, and reports when it reduced a
  full prompt. This also stops the payload growing: re-sending every earlier prompt in
  full is the bloat `prior_context` exists to avoid.

- **`MMH3ReplaceSection` now refuses a replacement that is itself a prompt.** Two or
  more of the format's own section headers in the `replacement` means a whole prompt
  in the wrong socket, and splicing it nests a prompt inside one of its own sections.
  Two headers rather than one, so prose that happens to mention a field name still
  passes. The message names the likely cause -- a writer shown a complete prompt as
  its example -- rather than only the symptom.

  This is a different failure from the missing-section case above it: nesting needs
  every header PRESENT, so it went down the accepted path both before and after the
  insert change.

- **`MMH3ReplaceSection` refused a definitions skeleton — "no section to replace".**
  The three-stage design depends on the definitions LLM emitting *empty* `summary:`
  and `detailed_description:` headers, and models resist writing a header with nothing
  under it. So the stage that is supposed to open the format produced four sections
  out of six and the splice raised.

  It now **inserts** a missing header at its canonical position instead of refusing.
  That is not guesswork: the Ref2VA order is fixed, so an absent section has exactly
  one place it can go, and the report names every insertion rather than doing it
  silently.

  Guarded by a **half** threshold — below three of six the input is refused. A report,
  a stray paragraph or the refiner's output wired in by mistake can contain one
  accidental `summary:`, and "recovering" that into a five-section skeleton would
  manufacture a prompt out of nothing. Verified at the boundary: bare prose, a report
  with one header, and two of six all refuse; three and four of six insert and report.

  `tests/test_lint.py` had asserted on the word "missing" in the old refusal text and
  was updated to the new message.

- **A flat profile reported infinite separation instead of none.** `contrast()`
  divided by a zero baseline when every token scored the same, so the strongest
  possible "nothing here" came back as `inf` — exactly backwards for a node whose
  purpose is to say when there is nothing to find. A flat profile now returns
  `(1.0, 1.0)` and the report names it. Genuine unbounded separation still returns
  `inf` and renders as the word "unbounded"; `profile_json` emits `null` for it, since
  `json.dumps` would otherwise write a bare `Infinity` that strict parsers reject.
  Caught by the test's static-footage case, which is why that case exists.

  MAINodes documents the same limit as "the oracle can rank but cannot abstain". The
  contrast ratios are this pack's answer to it, and they are the numbers to read —
  the quantile marks a fixed share of tokens hot no matter what it is given, so the
  spans alone are never evidence.

## [0.75.1] - 2026-08-15

Prompt text only — no schema change, no rewiring. From the first full music-video run.

### Fixed
- **Typography rendered flat, like a one-word subtitle.** Two separate rule failures:
  `text bursts` said "keep it SHORT" with no requirement that the fragment carry
  meaning, so it picked the shortest word in the line — a function word, observed as
  **"much"**. And the treatment rule asked only where the text "sits", which is
  satisfied by centring it, which IS the subtitle look.

  Bursts must now stand alone, function words are banned by name, and there is a
  concrete test (printed on a poster, statement or unfinished fragment?). Scale is
  the first decision, with "mid-sized and centred is the single option that looks
  like captioning" stated outright. Applies to `exact lyrics` too — a whole line
  centre-frame is just a longer subtitle.

- **Applied the validated kinetic-typography fix from 2026-08-12** rather than
  re-deriving it: `CHOOSE` the phrase by paraphrase (three words, ALL CAPS), then
  `RENDER` it as a literal quoted string, with the validity rule that *a prompt
  describing text moving without quoting the text is invalid*. The earlier failure —
  "words of devotion cascade in neon", which names no string and makes H3 draw
  invented glyphs — is named in the prompt so it is recognisable.

- **No typography instructions in `subject_definitions`.** The one rule that survived
  the review of MiniMax's own MV/subtitle skill: a `<Subject>` line says what an asset
  IS, and a text directive there leaks into all N chunks because the section is reused
  byte-identically.

### Added
- **Typographic identity, chosen once in `beats`** and inherited by every chunk, built
  from the video's world rather than a font menu.
- **Frame treatments from the video's world**, on the same principle — an effect
  belongs if the song earns it. Split frames, RGB channel split and slow motion named
  as known-good starting points.
- **Split frames as deliberate technique.** Naming two places, times or framings in one
  shot is what triggers them; they were arriving as a side effect and are wanted.
- **A tempo decision per chunk** — slow motion, realtime, or rapid micro-shots — tied
  to the chunk's measured energy so a peak and a near-silent window do not move at the
  same speed.

## [0.75.0] - 2026-08-14

Music video prompt building: a separate chain from the cinematic planner, driven by
the song rather than by an invented arc. Four nodes, 50 in the pack.

### Added
- **`MMH3ForcedAlign`** (*MMH3 Forced Align (Lyrics)*, `MMH3Tools/audio`) — places
  KNOWN lyrics on the timeline with stable-ts. Forced alignment, not transcription:
  the words are given and only timing is solved, so it cannot mishear. Raises rather
  than returning a word sequence that differs from its input.

  Emits the `whisper_alignment` type ComfyUI-Whisper does, so the existing
  `Whisper → Text` / `Whisper → Segments` nodes consume it unchanged, plus JSON so a
  song is aligned once and reloaded.

  The report classifies anomalies from the AUDIO rather than from timings: a gap over
  silence is a correct skip, a gap over audio is a skipped passage, words on silence
  are misplaced. It also prints the section map, which is the one line checkable
  against your own ears.

- **`MMH3MusicAnalysis`** (*MMH3 Music Analysis*, `MMH3Tools/audio`) — librosa BPM,
  key/mode, 4/4 bar grid and a 10 Hz RMS curve from the full mix. Ported from
  music-director's `music.py` **minus** its cut-salience blend and agglomerative
  segmentation: both exist to choose scene boundaries, and the looping sampler's
  windows are uniform and already fixed.

- **`MMH3LyricsToWindows`** (*MMH3 Lyrics to Windows*, `MMH3Tools/audio`) — slices an
  alignment by render window. Inputs mirror `MMH3SplitAudioToWindows` exactly so both
  read the same plan. Emits the window's verbatim lines with **chunk-relative**
  timestamps, ±1 window of context on the same clock, `has_lyrics`, the section (with
  a straddling boundary named), word onsets, and — with `music_json` — the window's
  energy and bar lines.

- **`MMH3MusicScenePlanPrompt`** (*MMH3 Music Scene Plan Prompt*, `MMH3Tools/prompt`)
  — three stages like `MMH3ScenePlanPrompt`, rules inverted where a song demands it:
  the arc is the song's ("do NOT invent an escalation"), a repeated chorus should feel
  like the same chorus, the words and shot timings are supplied rather than invented.
  Typography is rationed once across the whole song in `beats`. A window with
  `has_lyrics: false` takes an instrumental branch that forbids singing and suppresses
  typography even when the beat sheet assigned it. `reference_images` tells the
  definitions stage that attached images ARE the subject and beat the brief on
  appearance.

- `tests/test_forced_align.py`, `tests/test_lyric_windows.py`,
  `tests/test_music_scene.py` — 21 test files, all green.

### Fixed
- **`MMH3ForcedAlign` was not releasing whisper.** `del model` drops a name, not the
  weights: large-v3 is ~6 GB of fp32 and stayed resident across runs. Now mirrors
  music-director's `release_model()` — **`.cpu()` first**, then `gc.collect()` (torch
  modules hold reference cycles, so refcounting alone does not free them in a
  long-lived process), then `empty_cache()`. The report prints MB freed, so a failed
  unload is visible rather than discovered later as missing VRAM.

### Docs
- Scrubbed personal media references from all three shipped workflows: an audio
  filename and two image filenames, plus an **absolute output path**
  (`C:\ComfyUI\output\...`) inside a `VHS_VideoCombine` preview blob. Those are
  written by *previewing*, not by configuring, so any workflow saved after a run
  carries them.

### Notes on what does NOT work, recorded so they are not retried
- **Silero `vad`** made alignment far worse on a produced vocal: 131 of 190 words came
  back zero-length, because it is trained on speech and does not fire on singing.
- **`nonspeech_skip`, `max_word_dur`, `snap_to_onset` and the separator** all decide
  where a word may land. None of them can fix a lyric that under-describes the
  performance — if Suno sings a line three times and the lyrics hold it once, the
  aligner strands two. **Writing the line three times is the fix.**

## [0.74.2] - 2026-08-14

### Added
- **`workflows/MMH3_Scene_Prompt_Builder.json`** — the prompt half on its own, ending
  at the pipe-separated string `MMH3ReferenceMultiPrompt` consumes. No sampler, no VAE,
  no weights: it runs against an LLM server alone, so a film's prompts can be iterated
  without paying for a generation to discover they were wrong. Verified end to end
  2026-08-14.

  Every `MMH3WindowPlan` input is derived from a duration rather than typed, and the
  window calculator's `actual_seconds` drives `seconds_per_chunk` on all three stages
  *and* `MMH3PromptLint.seconds` — so the writer and the checker cannot disagree about
  how long a chunk is. They did disagree in the first run: both sat at a stale 5.2
  against a real 10.7s window, and every shot timestamp landed in the chunk's first
  half.

  `unload_after` is on for the one-shot definitions call and off for beats and shots,
  which share a model that should stay resident across the loop rather than reload per
  iteration.

### Fixed
- **`MMH3ScenePlanPrompt`'s `definitions` stage returned six empty headers.** It opened
  with "Output these six section headers, in exactly this order, and nothing else:"
  followed by the list — which reads as *emit exactly this list*, and that is what the
  model did. `subject_definitions`, `retention_analysis`, `overall_soundscape` and
  `non_diegetic_music` all came back blank, so every chunk spliced its beat into an
  empty skeleton and seven prompts were built on nothing.

  The instruction now names the four sections that MUST carry content, says a blank
  reply is a failed reply, scopes "bare header" to the two per-chunk sections only, and
  requires each header exactly once (the failed run repeated several).

  Observed 2026-08-14 on a 7-chunk run. The `beats` and `shots` stages were unaffected
  and worked as designed — the arc escalated across all seven with no early resolution
  and no repeated line.

## [0.74.1] - 2026-08-13

### Docs
- **The `noise_mask` correction only landed on the deep note.** The Requirements table
  and the silent-failure warning at the top — the ones actually read first — still told
  the one-cause story. Both now carry the three-part account, with the intermediate-value
  case named.
- Added a **Tests** section. 18 test files ship in the repo with no explanation of what
  they are or how to run them; they need ComfyUI's interpreter and nothing else.

### Changed
- `MMH3ScenePlanPrompt`'s `shots` stage now states that the user message repeats its
  beat verbatim and is the beat to EXPAND, not a new instruction. Without it the user
  turn is bare prose with no framing, which invites the model to answer it.

## [0.74.0] - 2026-08-13

### Added
- **`MMH3ScenePlanPrompt`** (*MMH3 Scene Plan Prompt*, `MMH3Tools/prompt`) — builds N
  chunk prompts **section by section** rather than chunk by chunk, in three stages:
  `definitions` (once), `beats` (once), `shots` (N).

  The problem was the loop's shape, not its wording. Writing chunk *i* in isolation
  asks for a complete arc in every chunk, so every chunk resolves — observed as five
  variants of one scene, each with its own climax. Transposed, the shared sections are
  written once and reused verbatim (no drift), escalation is decided where all N beats
  are visible with an explicit "nothing resolves before beat N", and dialogue planned
  across the set cannot repeat a line in three chunks. Costs `1 + 1 + N` calls against
  `2N` — 10 instead of 16 for eight chunks.

  The `definitions` stage emits the whole film-wide skeleton — definitions, retention,
  soundscape, score — plus bare `summary:` / `detailed_description:` headers, because
  `MMH3ReplaceSection` refuses to splice into a prompt with sections missing.
  Soundscape and score are film-wide on the same argument as the definitions: drift
  there is audible.

  The `shots` stage raises without a `beat_sheet` rather than quietly writing a
  self-contained chunk. Its banality rule is scoped to **speech only** — banal lines
  over an escalating scene, never a banal scene, which is what the un-scoped version
  produced.

- **`MMH3PromptPart`** (*MMH3 Prompt Part*, `MMH3Tools/prompt`) — the i-th piece of a
  separated string, plus the count. The join between a beat sheet written all at once
  and a loop rendering one beat per pass. Defaults to the same `|` the accumulator and
  multi-prompt use; strips code fences; past the end repeats the last piece (matching
  how the looping sampler reuses the last cond) or raises.

- `tests/test_scene_plan.py`, including the whole loop end to end: split → replace ×2
  → accumulate, asserting the definitions come out byte-identical in all four windows
  and the multiprompt split still sees exactly four.

- **`unload_text_encoder` on `MMH3Regenerate2KReference`** (default on), matching
  `MMH3ReferenceMultiPrompt`. 2K sampling is the tightest VRAM case in the pack and
  recondition mode is the last thing that needs the encoder, so it hands the VRAM back
  rather than denying the diffusion model room and dropping sampling into system RAM.
  Inert in append mode, which never loads an encoder.

  **Appended last**, after `base_label`, so saved workflows are unaffected.

### Changed
- The eviction logic is now one shared `common.evict_text_encoder()` rather than a copy
  in each node. No behaviour change for `MMH3ReferenceMultiPrompt`.

### Docs
- **Corrected the `noise_mask` notes.** They credited #15375 with per-row *timesteps*
  only. Read against the patch, it does three things: passes the mask to the model as a
  cond, runs preserved rows at the cond timestep, and adds a `scale_latent_inpaint`
  override to `MiniMaxH3` that stock does not have — confirmed against the stock class.

  That third one is what artifacts, and only at **intermediate** mask values, since
  #15375 thresholds ≥0.995 to 1.0 and ≤0.05 to 0.0. It is the mechanism behind the
  0.72.x seam noise that tracked `feather_latents`, the pack's only source of
  intermediate values at `overlap_strength=1.0`.

  Also fixed a claim that had become false with the PR applied: "a noise mask pins at
  the sampler, not the model" is true of **stock**, and #15375 is precisely what
  changes it.

## [0.73.0] - 2026-08-13

### Removed
- **`feather_latents` is gone — from `MMH3LoopingSampler` AND `MMH3SeedOverlap`.**
  It made the seam noisier than no feather at all (0.72.2), and an input whose only
  correct value is its default is a trap rather than a setting.

  ⚠️ **BREAKING for saved workflows using the looping sampler.** It sat at position
  **12 of 21**, so nine later widgets shift: `sampling_start_step`,
  `sampling_end_step`, `phase2_start_step`, `phase2_sampler`, `phase2_guider`,
  `keyframes`, `keyframe_indices`, `vae`, `prior_av_latent`. Re-check any graph built
  before 0.73.0. This deliberately breaks the pack's append-only rule: the node is
  dev-only with a known single user, who accepted the cost rather than keep a dead
  input. `MMH3SeedOverlap`'s was the LAST input, so that one shifts nothing.

  Why it could not simply default to 0 and carry a warning: a ramp writes intermediate
  mask values, and since #15375 was rebased each ramped cell gets its own timestep
  (`rows_t = 1 - m*sigma`) while the sampler blends its content as `x*m + orig*(1-m)`.
  The two correspond only approximately, so the ramped band is rows whose label does
  not match what they hold — and that band is the seam the feather existed to smooth.

  The carry boundary is now a clean step: preserved carry, then full denoise, no
  intermediate values anywhere for the per-row timestep to disagree with. If the step
  shows, the levers are `overlap_frames` and `overlap_strength_video`.

  `test_concat_av.py`'s two feather assertions are replaced by ones that pin the new
  guarantee — the video mask contains only 0.0 and 1.0, and a pinned target region is
  never un-pinned. `MMH3OutpaintLatent` keeps its spatial feather, which serves a
  different purpose, with the caveat recorded in its docstring.

## [0.72.2] - 2026-08-13

### Fixed
- **`feather_latents` makes the seam NOISY on a core with the rebased #15375 — leave
  it at 0.** Observed 2026-08-13: a visible seam appeared after the core swap and
  setting the feather back to 0 removed it.

  0.72.0 documented the continuous mask as an improvement and rewrote this tooltip to
  say the ramp is now "a real ramp rather than a moved preserve/generate boundary".
  For partial `overlap_strength` that is right. **For the feather it is a
  regression**, and this corrects it.

  The ramp writes intermediate mask values. Those used to be binarised at 0.5, so the
  ramp merely moved the preserve/generate boundary and every row was cleanly one
  state or the other — crude, but self-consistent. Now each ramped cell gets its own
  timestep, `rows_t = 1 - m*sigma`, while the sampler blends its CONTENT as
  `x*m + orig*(1-m)`. The two correspond only approximately, so the ramped band is
  rows whose label does not match what they hold — and that band is the seam.

  Corrected in `MMH3LoopingSampler`'s `feather_latents` tooltip, in
  `MMH3OutpaintLatent`'s docstring and report (which still described the old 0.5
  threshold as the reason its treatment steps hard), and in
  `docs/looping-sampler.md`. The old behaviour is retained as the "on a core
  predating the rebase" case, where a feather is harmless.

  Ruled out along the way, all identical at `overlap_strength = 1.0`: the row values
  from `mask_row_values`, the per-row timesteps, the 1/256 quantize-and-snap, and
  `scale_latent_inpaint`. The regression is specific to intermediate mask values,
  which only a feather produces at full strength.

## [0.72.1] - 2026-08-13

### Fixed
- **README caught up with reality.** 0.72.0 corrected the code and
  `docs/core-changes.md` but left the README half-updated:
  - **`MMH3 Lyrics Sectionize` was missing entirely** — added in 0.71.0's tail, after
    the README pass. Coverage is checked by matching `display_name` against the README
    (class ids give ~40 false misses, since the README uses display names throughout):
    **44 nodes, 0 missing.**
  - Four places still described **#15439 as pending**, including a Known Limitations
    entry claiming *"two bugs sit in the way and only one of them is fixed upstream"* —
    both are fixed now, the second by the merged version anchoring the guide on the
    target origin rather than `text_len`.
  - The Requirements lead-in said **"two upstream PRs that have not merged yet"** above
    a table that now includes a merged one.
  - An edit in 0.72.0 joined mid-sentence, leaving *"…when it is needed. #15439 anchors
    a guide at `text_len`, but…"* running together.

  Old behaviour is kept throughout as the "on a core predating the merge" case rather
  than deleted, since the pack still supports those.

## [0.72.0] - 2026-08-13

### Changed
- **#15439 merged upstream, and #15375 was rebased onto it.** Both facts change what
  this pack has to carry.

  - **The hand-merge is gone.** #15375 and #15316 now apply **clean** against a core
    with the merged #15439. The two `seg_t`/`seg_tag` `cond_audio` entries that had to
    be added by hand no longer exist as a manual step.
  - **`patch_guide_origin.py` is obsolete on current core** and correctly stands down.
    The merged #15439 anchors a guide on the target origin by itself — measured on the
    live class, guide `11.000` against target `11.000` with one image reference, where
    the draft gave `-1`. The wrap would over-correct by exactly the reference advance;
    its self-test compares against the target origin, rolls back, and reports
    `is_applied() == False`. **That is the success case.** Kept, inert, for anyone on
    an older ComfyUI.
  - `test_guide_origin.py` now asserts the **invariant** (guide anchors on the target
    origin) rather than the **mechanism** (the wrap is installed), and reports which
    side supplies it. Asserting the mechanism failed on a fixed core, which is
    backwards.

### Fixed
- **`#15375` detection was silently broken by a rename, disabling every masking node.**
  The rebase renamed `mask_row_targets` → `mask_row_values`, and the pack detected the
  PR with `hasattr(mm, "mask_row_targets")`. `MMH3SeedOverlap` therefore refused to run
  with "needs per-row masking, which is not applied" on a core that *did* have it.
  Both the node and `test_concat_av.py` now accept either name.

- **The mask is no longer binary, and the docs said it was.** The rename was the point:

  ```python
  old:  target = m.reshape(-1) >= 0.5   # bool, all-or-nothing
  new:  values = m.reshape(-1)          # float in [0, 1]
  ```

  So partial `overlap_strength` now grades the **timestep conditioning** as well as the
  latent, and a feathered spatial mask no longer hardens at the 0.5 contour.
  `MMH3SeedOverlap`, `MMH3LoopingSampler`'s `feather_latents` tooltip,
  `MMH3OutpaintLatent` and `docs/core-changes.md` all claimed otherwise; corrected, with
  the old behaviour kept as the "on an older core" case. New
  `per_row_mask_is_continuous()` reports which the installed core has.

  `docs/core-changes.md` ended that section with *"Re-check if #15375 changes before
  merge."* This is that re-check.

## [0.71.0] - 2026-08-13

### Added
- **`MMH3MusicCaptionSystemPrompt`** ("MMH3 Music Caption System Prompt",
  `MMH3Tools/prompt`) — the local stand-in for MiniMax's hosted
  `music-caption-rewriter`, for **MiniMax Music 3**.

  Same problem as H3, same shape of answer. Music 3 wants a **three-section Structured
  Caption**, not a comma-separated tag list, and MiniMax ships a hosted rewriter to
  produce one. Running locally there is none, so this emits the rules as a system
  prompt for your own LLM node — exactly what `MMH3TaskSystemPrompt` does for
  Context-IR.

  The format, per the Music 3 model card: **Global Metadata** (genre, subgenre, BPM,
  key, scale, emotional progression, listening scenario, production profile),
  **Vocal Details** (gender, timbre, performance style, harmony, backing vocals,
  effects), **Arrangement** (primary/secondary instruments, section-level instrument
  evolution, groove, bass, percussion, textures, spatial effects). Lyrics go in the
  separate field with `[Intro]`/`[Verse]`/`[Pre-Chorus]`/`[Chorus]`/`[Post-Chorus]`/
  `[Bridge]`/`[Instrumental]`/`[Solo]`/`[Outro]` on their own lines, and parenthesised
  backing vocals.

  Rules that go beyond restating the card, because they are the ones that decide
  whether a caption is structured or merely long: BPM as a **number** and key as a
  **named key** ("mid-tempo" is not a BPM); emotional **progression** rather than an
  emotional label; section-level instrument evolution named against sections that
  actually exist in the lyrics; audible content only.

  Three `lyrics_mode`s emitting mutually exclusive instructions — `write lyrics`
  (caption + lyrics), `lyrics supplied` (words FIXED, caption derived from them, the
  only permitted edit being to add section tags), `instrumental` (caption only, and
  the structure has to be carried by the Arrangement since there are no lyric tags to
  imply it). The caption/lyrics agreement check rides with the first two and is absent
  from the third.

  `suggest_structure` offers a section skeleton sized to `seconds` — conventional song
  shapes, not model constraints, so the LLM is told it may deviate with reason.

  ⚠ **Duration constants are read from the INSTALLED model**
  (`comfy.ldm.minimax_music.ar`), not hardcoded: `MAX_AUDIO_FRAMES / AUDIO_FRAMES_PER_SECOND`
  = 9000/25 = **360.0s**, where the model card says "~5 minutes". The node clamps and
  warns rather than raising, and frames the duration as a **ceiling** throughout —
  Music 3 may end a song earlier, which is why `MiniMaxMusic3TextEncode` returns a
  `seconds` output rather than echoing the request.

  ⚠ **Do not prompt Music 3 from MiniMax's older music guide.** The one in their
  skills repo targets the previous generation's HOSTED API — comma-separated
  descriptors, `--instrumental`, a bitrate setting, 24-hour URLs, "~25-30 seconds per
  generation". Its lyrics tags carry over; its caption advice does not. Tested
  explicitly: `tests/test_music_caption.py` §9 asserts none of that wording is emitted.

  Requires ComfyUI **v0.33.0+** for Music 3 itself; the node degrades to documented
  fallback constants on older cores rather than failing to import.

- **`delivery: spoken word`** on the same node, for dramatic monologue over music.
  Music 3 is a MUSIC model: asking once for spoken delivery is not enough, because the
  take **starts spoken and drifts into song** partway through (observed 2026-08-13).
  So this is a block of countermeasures rather than a request, and each one targets a
  different route back to singing:

  - **negative vocal rules, not just positive** — no melody, no pitched singing, no
    vibrato, no sustained notes, no harmony, no backing vocals, with a speaking
    register and cadence named instead
  - **per-section reinforcement** — the Arrangement restates spoken delivery at every
    section including the last, because a single statement at the top decays, which is
    precisely the observed failure
  - **an instrumental melodic lead** — with no melodic instrument the model has nowhere
    to put a melody except the voice, and it will
  - **no chorus** — `[Chorus]`, `[Pre-Chorus]` and `[Post-Chorus]` name a sung hook, so
    the suggested skeleton drops them entirely and leans on `[Instrumental]`,
    `[Interlude]`, `[Solo]`, `[Intro]` and `[Outro]` to carry structure instead
  - **prose, not verse** — short end-rhymed lines of even length read as a lyric and
    get sung; varied wording on a returning idea, since a word-for-word refrain
    acquires a tune

  Interactions handled rather than ignored: `spoken word` + `instrumental` is
  meaningless (no voice) and is flagged and dropped, and supplied lyrics containing a
  chorus tag are warned about under spoken delivery.

  **These are reasoned from the format and the failure mode, not measured.** The
  chorus-tag and melodic-lead points are the two most likely to matter and the two
  most worth disproving.

- **`lyrics supplied` now BYPASSES the LLM entirely.** The first cut put the words
  into the system prompt and asked the model to reproduce them verbatim; in practice
  it "completely rewrites lyrics, and not well" (observed 2026-08-13). That is not a
  model failing to follow instructions — it is **asking a language model not to be a
  language model**, and no amount of firmer wording fixes it.

  The fix is routing, not persuasion. In that mode the node now:
  - tells the LLM to emit the **caption ONLY**, and explicitly not to copy, quote or
    echo the lyrics or include a `lyrics:` field at all
  - passes the words to the LLM as **context**, so the caption can be derived from
    them — emotional progression, vocal details, and an Arrangement naming the
    sections the lyrics actually contain
  - gains a third output, **`lyrics`**, carrying `supplied_lyrics` **verbatim** for
    wiring straight to `MiniMaxMusic3TextEncode`. The text never enters the LLM's
    output, so it cannot come back altered.

  The report says so explicitly, because the wiring is the whole mechanism: *"the LLM
  writes the CAPTION ONLY. Wire this node's `lyrics` output straight to MiniMax Music3
  Text Encode."* The passthrough is empty in the other two modes, where the lyrics
  legitimately come from the LLM via the split node.

- **`MMH3MusicCaptionSplit`** ("MMH3 Music Caption Split", `MMH3Tools/prompt`) — the
  join that makes the music path a graph rather than a copy-paste. The LLM answers with
  BOTH fields in one string; `MiniMaxMusic3TextEncode` wants them on two sockets.

  Deliberately not a `str.split()`, because real replies arrive wrapped in code fences,
  prefaced with "Sure! Here you go:", with the labels bolded or bulleted, or in
  uppercase — all handled, all tested. Labels found out of order are read in the order
  they appear and flagged. A reply with **no labels at all** becomes the caption with a
  warning, since an LLM that ignored the output format usually still wrote the thing
  that was asked for; failing there would throw away a usable answer.

  Markdown is deliberately **left alone** — `clean_caption()` in
  `comfy/ldm/minimax_music/prompt.py` already strips it downstream, and stripping twice
  risks eating an asterisk that was part of the text.

  Two failure modes get named rather than passed on silently: an **empty caption**
  (Music 3 reads style ONLY from the caption, so an empty one means the model invents
  the whole arrangement) and **lyrics that are section tags with no words** (a
  wordless track, easy to misread as a model failure). `strict` promotes the empty
  caption and the no-labels case to raises; off, it reports and carries on, which is
  what you want while tuning a local model.

  Graph: idea -> LLM (system prompt from the node above) -> **Split** -> `caption` /
  `lyrics` -> `MiniMaxMusic3TextEncode`.



## [0.70.0] - 2026-08-13

### Changed
- **`MMH3TaskSystemPrompt` now tells the writer to lead a shot with its dialogue, not
  trail it after the action.** Added to both `## Shared syntax` and
  `## Supplied dialogue`, so it applies whether the model writes the lines or is
  handed them:

  ```
  good:  The woman (S1) says: <d>[English] I almost didn't come.</d> as she crosses
         the room and sets her bag on the table.
  bad:   The woman crosses the room and sets her bag on the table. She (S1) says:
         <d>[English] I almost didn't come.</d>
  ```

  **Observed 2026-08-13**, ck's, from real generations: a line appended after a run of
  action prose glitches the audio around it. Recorded in
  `docs/context-ir-system-prompt.md` §4 as an observation with its provenance — it is
  **not** a documented MiniMax rule and was not isolated in a controlled A/B, since
  the two orderings differed in other ways too. The plausible mechanism is that a
  trailing line gets placed late on the timeline and the audio is compressed or cut
  around it, making this a special case of the timing problems `<cutoff>` already
  exists for.

  Encoded as a directive rather than a caveat because the cost of following it is zero
  either way: both orderings read identically to a human, so there is nothing to trade
  off against. Worth a controlled pair if it ever needs to be known more precisely.

## [0.69.0] - 2026-08-13

### Added
- **`MMH3Regenerate2KReference` gains RECONDITION MODE, so the base video can be
  labelled and the question can actually be tested.** Optional inputs appended last:
  `clip`, `vae`, `prompt`, `prepend`, `audio_vae`, `ref_image_size`, `ref_images`,
  `ref_videos`, `ref_video_audios`, `ref_audios`. Unwired, nothing changes.

  0.61.9 argued that leaving the 768p unlabelled is correct, because `base_video` is
  the API role with no label and the original prompt cannot name a video that did not
  exist when it was written. That argument may well hold — but it answered *"is this
  equivalent to the hosted API"* when the question asked was *"does the model
  understand it"*, and settled by reasoning what only a run can settle.

  This is **not an append**. The whole conditioning is rebuilt per window: the exact
  stage-1 prompt verbatim, the same media reinserted so their `<Picture i>` /
  `<Video k>` tags come back identical, and the base slice registered as one more
  reference the text encoder sees. Any wired `stage1_cond_set` / `conditioning` is
  ignored and logged as such, because it is replaced rather than extended.

  - **`prompt`** takes the exact 768p prompt, `|`-separated for one per window, the
    same convention as `MMH3ReferenceMultiPrompt`. The cond_set's stored `prompts` are
    a display field and are deliberately not trusted for re-encoding.
  - **`base_label`** (default **`<base_video>`**) is the text tag written in front of
    the base's vision block. The hosted endpoint sends the 768p with
    `role=base_video`, a role distinct from `reference_video`, and whether H3 was
    trained on a matching TEXT tag is an open question — one the model can simply be
    asked rather than argued about. Empty falls back to core's `<Video k>`.
  - **Both halves of the nested source are registered.** `stage1_latent` is a
    NestedTensor AV latent and stays required — the API lists an audio track as
    **mandatory** for regeneration, and §5 pins stage 1's audio into the 2K target, so
    a plain video-only latent is refused rather than quietly regenerating a new
    soundtrack. The video half becomes the base reference; the **audio half is
    registered as its own `<Audio k+1>`**, numbered after any reinserted reference
    audios and emitted before the base video, per the tokenizer's own convention.
  - **`prepend`** goes in front of the prompt and substitutes **two** tags:
    `{base}` → the base video's tag, `{audio}` → `<Audio k+1>`. Neither is hardcoded.
    The default now carries both halves of the test:

    ```
    Regenerate {base} at higher resolution: the same content, timing and framing,
    rendered with greater detail.
    {audio}: fully_copy - the complete source audio serves as the target video's
    complete final audio track.
    ```

    `fully_copy` is the documented `audio reuse` marker. A line mentioning `{audio}`
    is **dropped** when a window carries no audio, so the prompt never names a tag the
    tokenizer did not emit.

### Added — `mmh3tools/patch_ref_labels.py`
- **A reference item may set its own text label.** Core emits a reference's tag as
  ORDINARY TEXT immediately before its vision block:

  ```python
  add_text("<Video %d>: " % counters["video"])
  add_vision(frames[i:i + 2], video_block=True)
  ```

  `_text_ids()` tokenizes that like any other string — there is **no special token
  and no vocabulary entry** for `<Video 1>`. The tag is a convention written in plain
  text and the format is merely hardcoded, so `<base_video>` can go there. Verified
  against the live tokenizer: the leading ids change from `[27, 10724, 220, 16,
  26818, 220]` to `[27, 3152, 19815, 26818, 220]`.

  The wrap adds one optional key: an item carrying `"label"` gets that text instead
  of the counter-generated one. It rewrites by running stock on each item ALONE and
  swapping the leading label tokens, rather than forking the emitter — so any change
  core makes to vision or timestamp emission is tracked for free. Labelled items
  still advance `counters` exactly as before, so **no other item's number shifts**.

  Returns `{encoder_key: [entries]}` like stock, reading the key from stock's own
  output rather than hardcoding `qwen3vl_32b`. INERT unless an item carries `label`,
  so a graph that sets none produces byte-identical tokens. Self-tested at import
  against the live class — unlabelled paths must match stock exactly, a label must
  change something, and labelling one item must not disturb another — and it declines
  to install rather than corrupt conditioning. The node raises if a `base_label` is
  requested while the wrap is not installed, instead of silently ignoring it.

  This is the second runtime wrap on `main`, after `patch_guide_origin`. Same
  justification: no upstream PR carries it, and it is inert unless used.
  - Media is built **once** via `_build_refs` and the base is appended **after** it,
    taking the next free `<Video k>`. Order is load-bearing: the tokenizer assigns
    labels by counting items in the order given, so reinserting the same media in the
    same order is exactly what keeps the reused prompt's tags pointing at the same
    things. Blocks are appended in the same order as the items.
  - The **full-res** slice is decoded for the encoder; `ref_downscale` is a DiT-side
    cost lever and has nothing to do with what Qwen is shown.

  So the arms are: unwired (base unlabelled), reconditioned with an empty `prepend`
  (labelled), and reconditioned with a sentence (labelled and named). The strongest
  alternative framing is video editing's mandated sentence, the only one any task type
  requires: *"The target video is an edited version of `{base}`."*

  Costs a decode and a text encode per window — a Qwen3-VL/DiT swap per chunk.

  Refuses rather than half-working: `clip` without `vae` or `prompt`, `vae`/`prompt`
  without `clip`, reference audio without `audio_vae` (checked *before* `_build_refs`,
  which would otherwise die inside `audio_vae.encode`), and no conditioning of any
  kind. The report warns when nothing was reinserted, since the prompt's tags then
  point at media the encoder never saw. Tested in `tests/test_regen2k_encode.py`.

## [0.68.0] - 2026-08-13

### Fixed
- **`MMH3SizeCappedCopy` no longer ENLARGES a file that is already under
  `target_mb`.** It treated the ceiling as a target: `size_capped_bitrate` solves
  purely from `target_mb` and duration and never looked at the source (`src_mb` was
  read, but only for the log line), and `-b:v` in two-pass x264 is a target
  **average**, not a limit. So a 20 MiB clip with a 95 MiB ceiling was re-encoded up
  to ~95 MiB.

  The extra bits cannot add information — the source's is already fixed. They go
  into finer quantization of an already-decoded picture, i.e. bandwidth spent
  faithfully reproducing the FIRST encode's blocking, ringing and mosquito noise,
  plus residuals that would otherwise be zeroed. The output was strictly larger,
  never better, and slightly worse for being a second lossy encode — after a slow
  two-pass encode that achieved nothing.

  The tell was in the same file: `scale_filter` builds `scale=-2:min(ih,H)` and its
  docstring says "a master already shorter than the cap is never upscaled into it."
  The height cap was a true ceiling; the size cap was not. Now both mean the same
  thing.

  New `capped_copy_plan()` decides, and there are two ways to be inside the ceiling:
  - **Nothing to do** — under size, and within `max_height` (or no cap set). The
    encode is skipped entirely and **the source path is returned unchanged, with no
    copy written**. ⚠ Downstream steps that expect a distinct `_capped` file will
    receive the master itself; do not point a destructive step at this output.
  - **Under size but too TALL** — the downscale still has to run, so the budget is
    clamped to `min(target_mb, src_mb)` and the result cannot inflate either.

  An unknown source height with a cap set is treated as possibly-too-tall: running
  the encode costs time, but skipping it would silently ignore `max_height`.

  `_duration` becomes `_probe`, returning `(duration, height)` and reading both from
  one ffprobe call as **JSON** — `-of default=nw=1:nk=1` emits bare values whose
  order across a stream field and a format field is not guaranteed, and reading a
  height as a duration is exactly the mismatch `_ffprobe_for` exists to prevent.
  Falls back to ffmpeg's banner for both. `_duration` remains as a thin wrapper.
  The UI preview is factored into `_preview_for()` since the no-work path returns a
  source that may sit outside ComfyUI's output tree.

  No schema change. Tested in `test_size_cap.py`: both inside-the-ceiling cases, the
  clamp, exact-boundary behaviour, unknown height with and without a cap, and that
  clamping genuinely lowers the solved bitrate.

## [0.67.0] - 2026-08-13

### Changed
- **`MMH3LoopingSampler` ignores `keyframe_indices` when no `keyframes` are
  attached, instead of raising.** A ladder reuses one graph across passes and
  usually only the first pass carries anchors, so a live index string with the
  image input unplugged is the ordinary state of a refine pass — not a mistake
  worth stopping a run for. Clearing the field between passes was busywork.

  The indices are **not parsed at all** in that case, so an out-of-range or even
  malformed index is inert too. That is deliberate rather than sloppy: parsing
  exists to catch a keyframe that would land somewhere wrong, and with nothing
  being placed there is no such thing. The previous behaviour's justification —
  "silently dropping a keyframe the user asked for is worse than stopping" —
  applies to indices that WILL be used, and those still validate exactly as before.

  Reported rather than truly silent: `keyframe_indices ignored: no keyframes
  attached` in the node's report and the log, but only when the field is
  non-empty. An empty string stays quiet.

  Everything else still raises: an out-of-range index **when images are attached**,
  a count mismatch between images and indices, `keyframes` without a `vae`, and any
  keyframe when PR #15439 is missing. Tested in `test_looping_sampler.py`,
  "keyframe_indices without keyframes"; §10b lost its first case and says why.

## [0.66.0] - 2026-08-13

### Added
- **`MMH3CondSetStripText`** ("MMH3 Cond Set Strip Text", `MMH3Tools/conditioning`)
  — drop the prompt from every entry of a cond_set while the reference media rides
  through untouched. cond_set in, cond_set out, so it drops between any producer
  and the looping sampler.

  **For refine passes whose windows are smaller than the chunk the prompt was
  written for.** Core picks a window's prompt region from the window's MIDPOINT, so
  a window covering a fraction of the timeline is handed text describing all of it
  and asked to render the whole script into its slice — the same failure
  `MMH3CondSetSpread` exists to fix, except at a stage where the fix is not a
  better prompt but no prompt. At low denoise nothing is invented; the content is
  already in the latent and the only thing worth conditioning on is identity.

  This works because text and media live in different halves of a conditioning
  entry: the prompt is the **tensor** `t[0]`, while `minimax_refs` and
  `minimax_keyframes` are keys in the **dict** `t[1]`. Rewriting the first and
  copying the second is what core's own `ConditioningZeroOut` does — this applies
  it per entry, and adds the option to be more surgical than zeroing.

  Two modes:
  - **`zero`** (default) — blank the values, keep the span's length. Always valid,
    needs nothing from the encoder.
  - **`vision only`** — keep only the vision-block positions and drop the prose,
    shrinking `text_len`, which also pulls the target closer since references lay
    out from a cursor starting at `text_len`. Keyed on `minimax_token_tags`
    (1 = text, 0 = vision block including its flanking markers), and the tags are
    sliced **in lockstep** with the embedding: a tag vector that no longer lines up
    with its tokens is worse than leaving the text alone, because the DiT reads
    modality from it. A tags/embedding length disagreement is not guessed at — it
    falls back to zeroing and says so.

  ⚠ **`vision only` leaves an EMPTY text span (`text_len` 0)** on conditioning whose
  references were appended after encoding — `MMH3ImageToRef`, `MMH3LatentToRef` and
  `MMH3Regenerate2KReference` never register with the tokenizer, so there is no
  text-side copy of the image to keep. `PackedLayout` accepts it (verified: the
  sequence simply shifts down, target origin lands at 1.0), but no encoder can
  produce that state, so it is untested against real weights. Reported in the node's
  output and the log rather than prevented — `zero` is the fallback if it misbehaves.

  `prompts` comes out blanked; they would otherwise print in the sampler's report
  describing text the conditioning no longer carries. Tested in
  `test_multiprompt.py` §14: both modes, that the refs survive as the same object,
  lockstep tag slicing against a vision block in the MIDDLE of the sequence, the
  mismatch fallback, multi-entry sets, and that the source cond_set is not mutated.

## [0.65.0] - 2026-08-13

### Changed
- **`MMH3LoopingSampler` fits `keyframes` stills to the target grid instead of
  encoding them at whatever size they arrive at.** No schema change — behaviour
  only, and only where it previously crashed.

  Keyframe rows share the TARGET spatial grid. `PackedLayout` reads only the
  latent's TIME dim (`vt = video_latent.shape[2]`) and sizes the segment from the
  target's `_frame_grid`, so a still at any other resolution reserves the target's
  row count while the tensor patchifies to its own: a 1024x1024 still against a
  1344x768 target reserves 1008 rows and produces 1024. Nothing caught it — not the
  node, which only validated image count, VAE presence and PR #15439, and not the
  layout, which never looks at h/w. It surfaced as a broadcast error deep in the
  model naming nothing.

  This was the third of three keyframe paths and the only unprotected one:
  `MMH3ImageKeyframe` already resizes internally, `MMH3LatentKeyframe` has the
  opt-in `target_width`/`target_height` guard. Fitting rather than refusing is the
  point — a 2-3 stage ladder runs the same still against different target
  resolutions, and making the graph carry a resize per stage is busywork the node
  has the numbers for: it takes them from the master latent it just built.

  Aspect policy **mirrors `MMH3ImageKeyframe`'s `auto`**, which is the stock node's:
  the frame-0 opener STRETCHES because it establishes the clip's geometry, every
  later anchor CENTRE CROPS because it follows geometry already set. Identical
  results whenever the aspect already matches, which is the normal case — and a
  no-op with no resize call at all when the size matches exactly.

  Every resize is reported, to the console and to the node's `report` output:
  `keyframe frame 0 -> chunk 0 local frame 0, resized 6000x3375 -> 1344x768 (stretch)`.

  `carry="keyframe"` is unaffected: it slices the previous chunk's own tail, so its
  dims match by construction. Sizing is factored into `_fit_keyframe()` so it is
  testable without a VAE — `test_looping_sampler.py`, "keyframe fitting": identity
  on a match, both aspect policies, that they diverge on a square source and agree
  when the aspect matches, and that an alpha channel is dropped.

## [0.64.3] - 2026-08-12

### Fixed
- **`MMH3PackAV` normalizes a carried noise mask onto the latent shapes** — the
  actual fix for the chunk-1 crash `cannot reshape tensor of 0 elements into
  shape [-1, 1, 32, 0]`.

  Root cause, established with 0.64.2's mask report: core accepts a noise mask of
  ANY size (`prepare_mask` interpolates it onto the latent at sampling time), so a
  32×32 zero image is a perfectly legal audio pin and works in every whole-clip
  run. The looping sampler, however, SLICES masks by time — which silently assumes
  the time axis is real. It always was in phase 1, because
  `MMH3ReferenceMultiPrompt` manufactures its pin masks time-shaped; PackAV was
  the first path to carry a user-attached mask to the slicer verbatim. Chunk 0
  clamps and limps; a chunk starting past the mask's extent slices zero elements
  and dies three layers down in core with an error naming nothing.

  PackAV now applies core's own `reshape_mask` per half at pack time, so chunked
  and whole-clip runs see identical mask semantics for any core-legal mask.
  Identity (value-equal) for masks already shaped to their latent — and the
  stage-1 path is unaffected by construction, since it never goes through PackAV.
  A stale longer mask is resampled onto the trimmed audio exactly as core would
  have done. The carry log line reports old -> new shapes when normalization
  changed anything.

  The sampler itself is deliberately unchanged (0.64.1's lesson). Tested in
  `test_trim.py` §20: the literal 32×32 crash case, value-equal identity, and the
  stale-length resample.

## [0.64.2] - 2026-08-12

### Added
- **`MMH3LatentInfo` reports the noise mask's halves**, not just "present": type,
  each half's shape, and its time extent against the half it masks. Chunked runs
  slice the mask by time (video dim 2, audio dim 3), so an audio mask whose time
  dim is SHORTER than the audio is flagged as the chunk crash it will become -- a
  chunk starting past its extent slices zero elements and dies in core's
  `reshape_mask` with an error naming nothing. A LONGER mask is flagged as stale
  (sliceable, but built for a different timeline). Diagnostic output only; no
  behavior changes anywhere.

  (0.64.1 was released and reverted the same day -- a fix built on a wrong premise
  about that chunk-1 crash. This report exists to pin the crash's real source
  instead of guessing at it.)

## [0.64.0] - 2026-08-12

### Added
- **`MMH3CondToSet`** ("MMH3 Cond To Set", `MMH3Tools/conditioning`) — wrap an
  already-encoded CONDITIONING as a cond_set, no text encoder involved. The looping
  sampler requires a cond_set and ignores the guider's conditioning, and every other
  producer goes through the CLIP — so a refine pass conditioned by a zero-out had no
  route to the sampler short of loading a 20 GB encoder to tokenize an empty string.
  Unblocks the chunked light-denoise refine of long latents (flat VRAM vs clip
  length, master tensor never on GPU) with the zero-out graph as-is.

  `prompts` is empty strings (the display half of the contract, unfillable without
  text — same as `MMH3Regenerate2KReference`); `fingerprint` is None, nothing reads
  it; `count` replicates the same conditioning, and 1 covers any chunk count since
  the sampler reuses the last entry. Round-trips through `MMH3CondSelect` — tested,
  along with the sampler's empty-set gate, in `test_multiprompt.py` §13.

## [0.63.0] - 2026-08-12

### Added
- **`MMH3ContextWindows` gains `accumulator_device` (`gpu` default / `cpu`),
  appended last** — hosts the per-step fuse accumulators in system RAM. Every write
  to them is already a window-sized slice, so the loop pays one small PCIe transfer
  per window per cond and the fused result returns to the GPU once per step, after
  the loop — when the activation peak is over, so the transfer never coexists with
  it. Frees one full-length fp32 latent of VRAM per evaluated cond for the duration
  of the window loop. Values are identical to the gpu path — same ops, same numbers,
  different device — verified against the gpu path in `tests/test_windows.py` §21
  for both pyramid and relative fuse.

### Changed
- **A cond skipped by cfg 1.0 no longer allocates an accumulator at all.**
  `sampling_function` passes `conds = [cond, None]` at cfg 1.0 and never evaluates
  the None, but the handler (upstream's too — reportable) still allocated a
  full-length zeros accumulator for it and held it through the entire window loop.
  It now allocates nothing and materializes the zeros at fuse time, after the
  loop's activations are freed — the caller receives the same tensor, allocated a
  loop later. Automatic, both accumulator devices, saves one full-length fp32
  latent during the loop.

  Context for both: windows bound the model's *compute*, not the sampler's
  *storage*. The full latent, noise, input and accumulators all sit on the GPU at
  full clip length, which is why a 47-latent window that ran clean at 40s stalled
  at 120s/2K — ~4 GB of full-length copies squeezed the dynamic weight cache into
  thrash. These two changes zero the accumulators' share (the pack's half of the
  term); `x`/`noise`/`latent_image` are core's and remain. Ledger and measurements
  in `docs/context-windows.md`, "Windows bound compute, not storage".

- `_alloc_accumulators` takes the conds list instead of a count (None entries
  allocate nothing); an int still works for older callers.

## [0.62.1] - 2026-08-12

### Removed
- **`MMH3ContextWindowVRAM` (added in 0.62.0) is withdrawn — core already does it.**
  Treat 0.62.0 as never released.

  `comfy/context_windows.py` clamps `noise_shape` to the context window before the
  estimate runs, in `_prepare_sampling_wrapper`, and `create_prepare_sampling_wrapper`
  installs it — which `MMH3ContextWindows` has always called. Verified against the live
  wrapper: `[1,24,847,128,96]` reaches the estimator as `T=47`, reserving 1.4 GB rather
  than 24.9 GB. The node re-clamped an already-clamped shape and changed nothing.

  The reasoning that produced it is recorded in `docs/core-changes.md` under "Already in
  core — do not rebuild", because the arithmetic is correct and only the premise is
  false, which makes it easy to re-derive.

  Two gaps the clamp genuinely does not cover, noted there as where to look instead:
  packed/flat latents are skipped on purpose (the `is_packed` branch, which does not
  affect H3), and only the *estimate* is windowed — the full latent, its sampler copies
  and the fuse accumulators all stay resident and do scale with total length.

## [0.61.11] - 2026-08-12

### Changed
- **§7's prompt-modification hypothesis is now investigated and largely ruled out**,
  reversing 0.61.10. Kept in the doc rather than deleted, because it is the question
  people will ask and the answer is not obvious.

  `"task_type": "regeneration"` in the Query Task response looked like evidence that
  the regeneration job announces itself to the model in text. **List Tasks enumerates
  the field**, and it does not: `filter.task_type` takes `"generation"`,
  `"h3_context_ir"`, `"regeneration"`. That namespace is job classification — sibling
  to `generation`, which is plainly not a prompt marker — a different tier from
  `[audio reuse]`. The regeneration page frames it identically, as the handle by which
  tasks are managed through the shared Query / List / Cancel endpoints.

  The same enum settles the mechanism. **Context-IR is itself an API task type**
  (`modality: "text"`), so prompt expansion is a call whose output you submit onward —
  and there is **no** Context-IR variant for regeneration. Regeneration consumes the
  already-expanded prompt directly without routing back through expansion, removing the
  most plausible route by which a marker would get added.

  Surviving, and recorded as such: the endpoint could still append something of its
  own, which is unobservable from outside. But a single unchunked pass already produces
  a correct result with no marker, so the base video is evidently legible from layout
  alone.

- §1's "latent-only is the right semantics" caveat softened to match — the check was
  worth making, and it came back negative.

## [0.61.10] - 2026-08-12

### Added
- §7 records the prompt-modification hypothesis. **Superseded by 0.61.11**, which
  investigated it and found the cited evidence does not support it.

### Changed
- The "latent-only is the right semantics" conclusion in §1 now carries its own
  caveat. It assumes the prompt reaches the model unchanged. If the hosted pipeline
  rewrites the prompt to name the base video, then `base_video` does have a label
  after all — one this pack never writes.

## [0.61.9] - 2026-08-12

### Added
- **What an API `role` actually does — and why this pack's latent-only reference is
  the right semantics rather than a compromise.** A role decides the slot, and the
  slot decides the label the prompt uses: Comfy's Context-IR node says reference
  images are *"referred to in the prompt as 'Image 1'..'Image 9'"*. Locally that is
  the `minimax_ref_items` path, where `comfy/text_encoders/minimax.py` keeps a counter
  per kind and injects `<Picture %d>` into the text stream.

  So a reference has two halves: a `ref_items` entry the TEXT ENCODER labels, and a
  `minimax_refs` block the DiT attends. **`base_video` is the role with no label** —
  the prompt handed to regeneration is the original one, which refers to the original
  references and never mentions the 768p, because when it was written the 768p did not
  exist.

  `MMH3Regenerate2KReference` adds the block and no `ref_items` entry, so no label is
  created. That was chosen to avoid a VAE roundtrip; it is independently correct. The
  missing tokenizer registration that `nodes_refs.py` calls a KNOWN LIMITATION is a
  limitation for an ordinary reference and not one for a base video.

- **Where the local reproduction stops being equivalent**, stated as three hard
  boundaries rather than left implicit:

  * **The base competes for a reference slot.** The hosted endpoint budgets it
    separately — excluded from the 15s reference-video cap, not counted toward the
    3-video limit. This pack expresses the base AS a reference, so it has no separate
    budget. A 768p made with 3 reference videos would need a 4th video reference,
    past what Ref2VA documents. The hosted endpoint can regenerate that source and
    this pack cannot. A T2VA source has no reference videos, so it never arises.
  * **No way to tag the role.** The open layout has kinds, not roles.
  * **Local compute.** Full sampling at 2K with references attended every step.

## [0.61.8] - 2026-08-12

### Added
- **`docs/regenerate-2k.md` §1 gains a third source: Comfy-Org/ComfyUI#15471**, which
  adds `MinimaxHailuo03ContextIRNode` and `MinimaxHailuo03RegenerateNode` — official
  API nodes for the two hosted modules. An independent implementation of the same
  endpoint, and it corroborates every constraint this pack derived:

  > FPS strictly 23.9–24.1 · dimensions divisible by 32, max **1,032,192** pixels ·
  > "107 to 362 frames in steps of 17 (4 to 15 seconds at 24 FPS)"

  `1,032,192 = 768 x 1344`, the same `MAX_PIXELS` read out of `adapt_canvas`, and
  107–362 in steps of 17 is the `17j+5` grid. Two implementations reaching identical
  numbers is the strongest available confirmation that §3 is right.

  Its `prompt` input is **required** and documented as *"The exact prompt used to
  generate the source video."* That is now three independent sources — model card,
  API reference, node signature — for reusing stage 1's conditioning rather than
  re-encoding. It is the design decision this pack can be most confident about.

### Changed
- **The `base_video` caveat widened rather than narrowed.** The node builds its
  content list with `base_video`, `first_frame`, `last_frame` and `reference_image`,
  while reference videos and audios carry no role at all — so the hosted layout
  distinguishes at least four positions, not the two the doc previously assumed. The
  open layout has kinds (`image`, `audio`, `video`, `video_audio`) and no way to say
  "this video is the base one". §1 now says the gap is a different labelling scheme
  in code we do not have, not a single missing flag.

  Also recorded: the node implements only the `content` route. There is no
  `source_task_id` input, so even Comfy's official integration hands over the exact
  original inputs.

## [0.61.7] - 2026-08-12

### Fixed
Sanity check across every doc, run mechanically against the live schemas rather than
by reading. All 39 nodes are documented, every relative link and in-page anchor
resolves, every quoted category exists, and the grid constants in prose match
`common.py`. Four real errors, all in prose that named things:

- **README recommended "MiniMax H3 Sigma Shift"**, which does not exist under that
  name and is not this pack's node. It is stock ComfyUI's, node id
  `MiniMaxH3SigmaShift`, and its DISPLAY name is `ModelSamplingMiniMaxH3` — so a
  reader searching the node menu for "Sigma Shift" finds nothing. Both names and its
  provenance are now stated.
- **README called the dimension calculator "MiniMax H3 Dim Calculator".** It is
  `MMH3 Dimension Calculator`.
- **`docs/core-changes.md` referenced `MMH3LatentToKeyframes`**, which was renamed to
  `MMH3LatentKeyframe`.
- **`docs/looping-sampler.md` documented a `scene_frames` input on
  `MMH3KeyframePlanner` that does not exist**, and claimed `carry` "changes chunk
  lengths and the trim". The planner takes `total_frames` / `chunk_frames` /
  `overlap_frames` / `include_start` / `include_end` and has no `carry` input, and
  the trim was removed in 0.47.0 when chunks became slices of one master. Replaced
  with what is true: the planner runs the same `_plan` as the sampler, and the carry
  route changes what a chunk is conditioned on, never how long it is.

The `k+2` grid-safe trim mention that survives in §3 is explicitly historical ("an
earlier version of this node…") and is correct as written.

## [0.61.6] - 2026-08-12

### Fixed
- **`docs/looping-sampler.md` §8 blamed the noise seed for "every chunk looks like
  chunk 0". It cannot be that.** `_chunk_noise` adds the chunk index to the seed
  inside the sampler, so it advances whatever you wire — there is nothing to
  misconfigure. The row sent you looking at the one thing that is handled
  automatically. It now points at the conditioning: the report's per-chunk `prompt N`,
  then the cond_set, since one cond or N near-identical ones look the same from here.
- **The same table told you to set `overlap_strength_audio` to 1.0** for lipsync drift
  — the value measured on 2026-08-10 as producing tinny second-chunk audio, and
  recorded as such two sections further down in the same file. Now says so.
- **§9 still called 1.0 the default.** It moved to 0.9 in 0.53.1. Saved workflows keep
  their own value, so an older graph may still sit on 1.0 — which the entry now says,
  because that is the case where it still bites.
- **§10 claimed nothing between 0.0 and 1.0 had been swept**, duplicating §9 and
  contradicting it. Narrowed to what is actually unknown: below 0.8.

## [0.61.5] - 2026-08-12

### Observed
- **A single unchunked 2K pass works, at 362 frames — the official ceiling.** That is
  the exact configuration MiniMax documents, so the local reproduction of
  H3-Regenerate-2K is validated at the full length the hosted endpoint accepts.

  It also localises the open problem. The same clip, same length, same references,
  same conditioning and same model diverges when split into two chunks (§6) and is
  correct in one pass. So the variable is **chunking itself**, not the reference
  slicing, the schedule, the conditioning or the checkpoint — the single-pass result
  eliminates all of those at once.

  `docs/regenerate-2k.md` restated accordingly: the status line now separates "the
  documented method works" from "the extension past it does not yet", and §7 drops the
  single-pass entry it had listed as the top priority. `overlap_strength_video = 0`
  becomes the remaining hypothesis, and it only matters for clips genuinely over 362
  frames — anything shorter should be run unchunked.

## [0.61.4] - 2026-08-12

### Added
- **`docs/regenerate-2k.md` §1 now leads with what the endpoint refuses to do**, which
  turns out to decide how the whole document reads:

  > This endpoint only regenerates videos that meet the MiniMax-H3 768P output
  > specifications to produce 2K output. **It does not perform general-purpose
  > processing of arbitrary videos.**

  That is not a note about input formats, and the API's structure shows it. The
  `source_task_id` route takes the id of a previous succeeded generation — whitelist
  gated, owned by the calling account, queryable for 7 days. If a spec-compliant FILE
  were sufficient, a task id would be pointless; you would upload the video. It exists
  because the endpoint needs something the file does not carry. The `content` route
  names that something: the exact original inputs, including the FINAL prompt.

  So the model is **re-running the original generation at 2K with the 768p as an
  additional in-context anchor**, not upscaling a clip. The base video is one input
  among the original set, not the subject — which is what the model card means by
  "in-context regeneration is also an example of task generalization".

  Three consequences now stated: the generation context must be possessed rather than
  approximated (this pack satisfies it trivially, since `stage1_cond_set` IS the final
  conditioning); the method cannot be applied to footage not generated with H3, no
  matter how it is resized, because the conditioning does not exist; and the
  362-frame ceiling is H3's own single-pass budget showing through rather than a
  property of regeneration.

## [0.61.3] - 2026-08-12

### Fixed
- **`docs/regenerate-2k.md` claimed "the open weights expose no `base_video` kind".
  That conflated ComfyUI's port with the checkpoint.** `PackedLayout` accepting four
  kinds is a fact about `comfy/ldm/minimax/model.py`, not about the weights, and a
  port can omit what a checkpoint supports.

  Replaced with what the tensors actually establish, which is both narrower and
  stronger. `adaln_proj.linear.weight` is `[96768, 8]` and `96768 = 6 x 5376 x 3`:
  three modality rows, structurally, since a fourth would need a different shape. And
  no other tensor is indexed by reference role — the non-block inventory is
  `adaln_t_table`, `audio_patch_proj`, `condition_proj`, `final_layer.*`,
  `rope.inv_freq`, `token_refiner.*`. So the checkpoint holds no learned parameter a
  new role could select.

  What is NOT derivable, and is now marked as such: which kinds occupy which row.
  `seg_tag` is ComfyUI's dictionary. The weights say three rows exist; they do not say
  what belongs in each.

  Net effect on the claim: the weights rule nothing in, they only rule out the
  checkpoint as the hiding place. A role distinction lives in layout — segment
  position and position ids — which is code MiniMax did not publish, not weights we
  could inspect.

## [0.61.2] - 2026-08-12

### Changed
- **`docs/regenerate-2k.md` §1 now sources every design decision** from MiniMax rather
  than inferring it, quoting the model card in `MiniMax-AI/MiniMax-H3` and the
  `/v2/video_regeneration` API reference.

  The model card settles the approach: *"instead of using a conventional dedicated
  super-resolution module, we use the H3 base model to regenerate its own
  low-resolution result through an in-context manner"* — which is why the 2K pass is
  built on `minimax_refs` and not on anything resembling an upscaler.

  The API settles the inputs. It requires *"the exact same inputs used for original
  768P generation"* plus *"exactly one video item with `type=video_url` and
  `role=base_video`"*, and states the text must be *"the final prompt actually sent to
  the model when generating the 768P source video, not the original prompt."* That is
  precisely what feeding stage 1's own `cond_set` does.

  Its `base_video` specification also corroborates the dimension rules independently:
  audio track mandatory, 24 fps, dimensions divisible by 32, area <= 768x1344.

### Added
- **The 362-frame ceiling, which changes what the open divergence means.** The API
  accepts *"107–362 frames (~4–15 seconds, in 17-frame increments)"* — so the
  documented method has a hard upper bound of 362 frames and no chunking at all.

  The clip that diverged was **exactly 362 frames**. MiniMax would have regenerated it
  in one pass. So the failure appears precisely where the pipeline stops reproducing
  the documented method and starts extending it, and the first thing to try is a
  single unchunked pass — now the top entry in §7.

  Also recorded as unverified rather than glossed: `role=base_video` is a distinct role
  from `reference_video`, and this pack appends the 768p as an ordinary video
  reference. Whether the hosted module treats that role differently cannot be
  determined from outside.

## [0.61.1] - 2026-08-12

### Added
- **`docs/regenerate-2k.md`** — the 2K path had node entries in the README and no
  document explaining how the pieces fit. Covers what MiniMax's Regenerate-2K module
  actually is (quoting the model card and their own script, which exports ONE expanded
  prompt "for H3-Base and regeneration"), why stage-1 dimensions are not a choice, why
  the reference is sliced per window and what that costs, and why the audio is pinned
  rather than regenerated.

  It records the **open divergence past one chunk** with what the logs ruled out —
  schedule misalignment and a weak reference pathway, both eliminated by the run's own
  output — and the untested carry hypothesis, including why `overlap_frames = 0` is
  not the way to test it.

  Refine-vs-regenerate is left in the README rather than duplicated, so the two cannot
  drift apart.

## [0.61.0] - 2026-08-12

### Added
- **`MMH3AdaLNRefPatch`** (`MMH3Tools/model`) — take AdaLN modulation from another H3
  checkpoint, per block. Reads only the `adaln_proj` tensors from the source (~100MB
  of a 20GB file), never the rest.

  **Measured on the local int8 checkpoints.** fl2va and ref2va are the same model
  except for AdaLN: attention, MLP, `condition_proj`, both patch projections and both
  output heads all sit at cosine **0.999+**, while every `adaln_proj` lands between
  **-0.42 and -0.91** — anti-correlated, not merely different. `adaln_t_table` is
  shared (0.99985), so it is a real weight difference and not a change of input basis.
  AdaLN is where reference conditioning is routed into the residual stream, which
  makes that one component the whole difference between a checkpoint that can
  condition on a reference and one that cannot.

  `blocks` takes ranges and lists — `25-49`, `0-2,40-49`, `-1` — so the four published
  hybrid checkpoints become widget values rather than downloads, and non-contiguous
  sets become available. `final_layer` covers `final_layer.adaln_proj`, cosine
  **-0.830**, which the published hybrids leave at fl2va.

  **No strength slider, deliberately.** The two AdaLNs are anti-correlated at
  near-equal norms (272.8 vs 272.1 on block 25), so a linear blend cancels rather than
  mixes: at 0.5 the modulation collapses to **32%** of either endpoint and the model
  runs with most of its conditioning routing switched off. That reads as a broken
  merge, not a dial set halfway. Each block takes one side or the other — which is
  what the published hybrids do, and now we know it is the only sound operation
  rather than caution.

  **No per-row or per-term control**, for a different reason: the difference is
  uniform. All three modality rows (video/cond/ref, text, audio) and all six terms
  (shift/scale/gate across msa and mlp) sit in the same band, so there is no
  sub-structure to isolate. Both controls were designed, then dropped on the
  measurement.

## [0.60.0] - 2026-08-12

### Added
- **`MMH3ChunkedPixelUpscale`** — stage-1 latent → 2K latent, through pixels, one
  chunk at a time. For the **refine** leg of a 768p → 2K pass, where the upscaled
  frames go back into a sampler rather than to a file.

  **Why it is latent→latent and not an IMAGE slicer.** The obvious shape — slice an
  IMAGE batch, upscale each slice, return one IMAGE — saves nothing. The wall is the
  *returned* tensor: at 2x from 2K the output is 108 MB/frame against the input's 27,
  and a node handing back an IMAGE must materialise all of it however it was filled.
  Chunking only pays when the result leaves the graph or feeds a consumer that is
  also chunked. So the chunking runs the whole way across — decode a slice, upscale,
  re-encode, keep the latent, drop the pixels. Latents are ~1/100 the size of their
  frames, so the accumulated output is small at any length and the pixel footprint is
  one chunk.

  **Why it goes through pixels.** A 24-channel latent at /16 is not a spatially
  smooth signal; interpolating between latent positions produces codes the decoder
  never saw, and it renders them as blocking. `downscale_video_latent` is bilinear
  but only ever touches *reference* slices, which are never denoised — approximate
  context is fine, approximate content is not.

  **The grids line up exactly**, which is what makes it cheap: decode emits 17 frames
  per latent group of 5, and `encode_temporal` consumes non-overlapping 17-frame
  clips. Every decode batch hands the encoder a whole number of clips, and the round
  trip preserves length — `5j+2 → 17j+5 frames → pad to 17(j+1) → 5(j+1) latents →
  token_drop 3 → 5j+2`. Only the final chunk ever needs padding.

  Both inherited traps are handled: decode chunks carry left context and lookahead
  and drop the trailing 5 except at the true end (as `MMH3StreamingSave`), and
  `token_drop` plus the tail pad are applied **once at the end** rather than per
  chunk (as `MMH3StreamingEncode`, where per-chunk dropping silently loses 3 latents
  each pass).

  `method` offers `rtx_vsr` — NVIDIA Video Super Resolution, measured stateless
  across frames, so chunking cannot change its result — plus torch bicubic/bilinear/
  nearest-exact. **`nvvfx` is imported lazily**, so the pack does not depend on the
  RTX node pack being installed; a missing binding is reported as a wiring problem.
  Upscaling itself runs in sub-batches of one clip, since a whole chunk resized in
  one call would hold 3.4 GB at 2688x1536 and defeat the chunking above it.

  `groups_per_chunk` is capped from the **target** resolution against the 32-bit
  index limit — that ceiling bites harder here than in `MMH3StreamingEncode` because
  the frames are already upscaled. Audio rides through untouched; this is a
  resolution pass and audio has no resolution.

  **AV NestedTensors are handled on both sides.** The input goes through
  `unpack_av` (so a plain video-only latent works too, with no audio to carry), and
  the output is repacked as a NestedTensor with audio reconciled to the video half's
  dtype and device. A stale `noise_mask` is dropped rather than carried: one cut for
  the 768p grid is the wrong shape at 2K, and a sampler applying it to the wrong
  rows is worse than not having it.

  Covered by `tests/test_chunked_upscale.py` (grid arithmetic, guards, upscale
  paths) and `tests/test_chunked_upscale_av.py`, which drives the node end to end
  on a stub VAE with the real shapes and grid, so the decode slicing, context/tail
  trim, per-clip encode and single `token_drop` all execute.

### Note
- `MMH3Regenerate2KDims` does **not** guarantee an integer scale between stages — it
  guarantees an exact *aspect*. At 16:9 the default `target_long_edge` of 2048 gives
  **1.5x** (latent 84x48 → 126x72). For an integer factor at 16:9 use **2688** (2x);
  stage 1 is 6 of that aspect's 224x128 units, so integer scales land on multiples
  of 6. Matters only if you were counting on integral latent dims.

## [0.59.0] - 2026-08-12

### Added
- **`MMH3SizeCappedCopy`** — two-pass transcode of a finished video to a hard file
  size ceiling, for upload limits. Chains off `MMH3StreamingSave`'s `file_path`
  output; works on any video file, not just H3 output.

  CRF cannot have a size ceiling by construction — it targets quality and the file
  lands where it lands — so a delivery copy under a fixed limit has to be a second
  encode rather than a setting on the first. This keeps the master at whatever
  quality it was written at and puts the compromise in a separate file.

  Solves the video bitrate from the measured duration, then two-passes libx264 at
  it. Notes on the parts that are easy to get wrong:

  - **The budget is MiB, not MB.** Upload limits are quoted in binary megabytes; at
    a 100 "MB" ceiling the two differ by 5 MB, which is larger than the safety
    margin. Solving in decimal would leave that unspent on every encode.
  - **`SIZE_SAFETY` (0.97) scales the video bitrate only.** Audio encodes at exactly
    the rate asked for, so the margin has to come out of video alone or the total
    overshoots. Two-pass lands within ~1–2%, the container adds a fraction more, and
    the only failure that matters here is landing over.
  - **`max_height` exists because bitrate alone is not enough.** 20 minutes under
    95 MiB is ~520 kbps; at 2K that is mush, at 720p it is watchable. `min(ih,N)` so
    a source already shorter is never upscaled into the cap, `-2` for the width so
    the aspect holds and both dimensions stay even for yuv420p. The comma inside
    `min()` is backslash-escaped — an unescaped one ends the filter.
  - **The scale filter is applied in both passes.** Pass 1's stats describe the
    picture pass 2 encodes; different filters between them make the stats wrong.
  - Pass 1 writes `-f null -` rather than an mp4 to the null device, which the mp4
    muxer rejects as non-seekable. Only the stats log matters there.
  - Duration comes from the ffprobe **beside** the resolved ffmpeg, not from PATH —
    a duration read off a different build than the one encoding is how you get a
    silently wrong bitrate. Falls back to parsing ffmpeg's own banner, since
    imageio-ffmpeg ships no ffprobe at all.

  Impossible budgets raise rather than encoding mush, and a solved bitrate under
  150 kbps warns with the knobs that would fix it. Lives in `nodes_save.py` for the
  ffmpeg discovery it shares with the streaming node; it never touches a VAE.
  `MMH3StreamingSave` is unchanged — no input added, moved, or renamed — so saved
  workflows are unaffected.

  Covered by `tests/test_size_cap.py`.

## [0.58.1] - 2026-08-11

### Fixed
- **`MMH3CondSelect` raised `IndexError: list index out of range` on a cond_set from
  `MMH3Regenerate2KReference`.** The 2K node returned `"prompts": []` because it
  builds conds from encoded conditioning and may never see the text, and CondSelect
  indexed `cond_set["prompts"][i]` unconditionally. Wiring the 2K cond_set into a
  guider through CondSelect -- the documented way to do it -- crashed every time.

  Fixed on both sides. CondSelect now guards: the conditioning is what callers
  actually wire, and losing a display label is not worth an exception. And the 2K
  node fills `prompts` properly -- carrying stage 1's text through, index-matched
  per window, so the 2K cond_set is a complete one rather than half-populated.
  With a single `conditioning` there is no text to carry, so entries read
  `(2K pass, window N)` rather than being blank.

  A genuinely out-of-range `index` still raises, which is the check that was
  supposed to be doing this job.

## [0.58.0] - 2026-08-11

### Fixed
- **`MMH3FrameCalculator`'s `seconds` widget stepped by 0.01, so every arrow-click
  landed between two achievable durations.** The node snapped silently, the widget
  kept showing the typed value, and the frame count moved a whole 17-frame group
  with nothing on screen to say so. Downstream that is a window appearing: reported
  2026-08-11 as "3 windows when I thought it'd be 2".

  The grid IS the widget now. Achievable durations are `(17j+5)/24`, which is
  `5/24` with a spacing of exactly `17/24` — so `min` is 5/24 and `step` is 17/24,
  and the arrows walk 0.208, 0.917, 1.625, 2.333, 3.042 ... with zero drift.

  The default moves 5.0 -> 5.167 for the same reason: 5.0 was never achievable, it
  just looked like it was. Saved workflows keep their own widget value.
- **It had no `label` output**, while the README claimed calculators emit "concise
  typed outputs plus a short label". Added, and it names the snap:
  `5.167s = 124 frames, 37 latents  (snapped up from 5.000s, +0.167s)`, plus a note
  when the move exceeds half a step. A typed value can still be anything, so the
  snap has to be visible rather than merely correct.

### Changed
- **`include_audio` removed from `MMH3Regenerate2KReference`.** Measured at 0.56% of
  the reference cost — 320 audio latents against 57,456 patch positions for the
  video half of a 192-frame window at 1344x768 — so turning it off saved nothing and
  removed the alignment lipsync needs. There was no correct setting for "off", and
  a source with no audio half already produces a video-only block without a toggle.
- **Stage 1's audio is written into the 2K target and pinned** (`noise_mask` 1 for
  video, 0 for audio), the same mechanism as `use_input_audio` minus the encode
  since it is already latents. Left empty, the 2K pass would have generated an
  entirely NEW soundtrack: paying for it, and drifting from the one the picture was
  cut to. A resolution pass has no business touching audio. Raises if the target's
  audio length does not match the source's, because that would place it at the
  wrong moments rather than merely sounding wrong.

## [0.57.0] - 2026-08-11

### Added
- **`unload_text_encoder` on `MMH3ReferenceMultiPrompt`, default ON.** Evicts the
  text encoder from VRAM once every prompt is encoded.

  H3's text encoder is large, and this node is the last thing in a run that needs
  it. Left resident it occupies room the diffusion model then cannot get, and
  sampling falls back to system RAM -- which does not error, it just stops making
  progress. Reported 2026-08-11: a workflow hanging while diffusing from RAM, worked
  around by dropping a KJNodes VRAM_Debug into the graph.

  It calls `unload_model_and_clones(clip.patcher)`, NOT `unload_all_models()`. That
  is the difference from the VRAM_Debug workaround: this evicts the text encoder and
  its clones alone and leaves the VAEs resident. Core uses the same call in
  `sampler_helpers.py`.

  **Default ON is a behaviour change** for saved workflows, which have no widget
  value and so take the default. Chosen deliberately: the cost of being wrong in
  this direction is reloading the encoder, and the cost of being wrong in the other
  is a hang.

  It frees the memory; it cannot guarantee the diffusion model then takes it. If
  something re-touches the CLIP before the sampler runs, it reloads.

## [0.56.0] - 2026-08-11

### Added
- **`MMH3Regenerate2KReference`** (`MMH3Tools/conditioning`) — the second pass of a
  768p -> 2K run, with the reference **sliced per window**.

  A cond_set is already per chunk: the sampler takes `conds[i]` for chunk `i` and
  passes `minimax_refs` through untouched. So a reference attached to cond `i`
  reaches chunk `i` and nothing else, and the slicing is entirely a build-time
  concern — no sampler change, no ref building inside the loop.

  It matters because reference tokens are re-attended at EVERY sampling step. Giving
  every chunk the whole 768p clip multiplies that by the chunk count; on a 12-window
  clip, slicing measured about **9.9x less reference attention per chunk**.

  Takes stage 1's own `cond_set`, pairing `conds[i]` with ref slice `i`, so
  per-window prompts survive into 2K. A single `conditioning` is the one-prompt
  alternative; wiring both raises rather than silently picking one, and a short
  cond_set repeats its last prompt with a warning, as the looping sampler does.

  Latent-only, following `MMH3LatentToRef`: the reference is appended for the DiT and
  never shown to the text encoder. Correct here because the prompt IS the original
  context from the 768p pass, so the encoder has nothing to learn from seeing the
  video again — and it means no VAE roundtrip and no CLIP load in the 2K pass.

  Verified: every ref slice is bit-equal to its window of the source, computed from
  the same `_plan` the sampler runs; the incoming conditioning is not mutated; the
  output latent's length matches the source exactly.
- **`MMH3Regenerate2KDims`** — matched stage-1 and stage-2 dimensions for a two-pass
  768p -> 2K run, the shape H3-Regenerate-2K uses ("feeds the 768p result together
  with the original context back into H3 to regenerate at 2K").

  **Stage 1 is not a choice.** It reproduces core's `adapt_canvas` — 768 short edge,
  area capped at 768*1344, axes rounded to 32 — because that is what H3-Base emits
  whatever you ask for. A stage-1 number that merely looks reasonable makes stage 2
  an upscale of something never rendered. Asserted equal to `core.adapt_canvas` for
  all five ratios in both orientations.

  **Stage 2 is an integer multiple of stage 1's on-grid unit**, not the requested
  long edge rounded to 32. Rounding each axis independently drifts the aspect: 16:9
  at a 2048 long edge lands on 2048x1184 = 1.7297, and that squeeze is in every
  frame. Reducing w1:h1 to its smallest 32px-aligned pair gives a unit that can only
  scale exactly — 16:9 becomes 2016x1152 at 1.50x, and the label says why it is 2016
  rather than the 2048 asked for. Aspect now matches to 1e-9 across all ten cases.

  Warns on a downscale, on jumps past 4x, and whenever the requested long edge could
  not be honoured exactly.

### Fixed
- `test_streaming_save.py` broke against **ComfyUI v0.32.0**: the H3 VAE work
  (#15446 / #15486) made `decode_temporal` preallocate through a new
  `decode_output_shape`, and call `_finalize_pixels`, neither of which the test's
  stub had. `decode_output_shape` and `_decode_temporal_chunks` are now bound from
  core so the stub keeps tracking real geometry; `_finalize_pixels` is deliberately
  identity, because core's version clamps to [0,1] and this stub encodes each
  frame's latent index as its pixel value — clamping would collapse every index
  above 1 and the frame-to-latent tracing would pass on garbage.

## [0.55.0] - 2026-08-10

### Added
- **`chunk_frames = 0` means one chunk over everything being generated** — and
  therefore one prompt, which is the point.

  The size is region + carry + grid padding, and being one grid step short silently
  costs a second chunk and a second prompt. Measured before this: a 5s addition
  needed `chunk_frames` 141, an 8s one 226 — always `addition + 34`, but only because
  the padding happened to land the same way each time. That is not arithmetic to do
  by hand, and wiring the frame calculator to both `length` and `chunk_frames` (the
  obvious thing to do) never produces it.

  Verified across every prior x addition pair from 3s/2s to 47s/20s: exactly one
  chunk, grid valid, prior kept verbatim, region fully covered. With no prior it
  means one chunk over the whole clip.

  Non-zero values are unchanged — set it when you want the chunk sized for VRAM and
  are willing to supply a prompt per chunk.

## [0.54.0] - 2026-08-10

### Fixed
- **The tail of the new region was never sampled, and the prior's length changed the
  schedule.** Two faults in `prior_av_latent`, both from the latent grid.

  A standalone clip is `5j+2` latents, so prior + new is `5k+4` — not a valid clip.
  `latents_to_frames()` floored it and the leftover latents fell outside every
  window. The **new** region is now padded up so the combined total is `5j+2`: at
  most 4 latents of extra generation, never fewer frames than asked for. Padding the
  prior instead would mean inventing or discarding real footage.

  Separately, the schedule was planned over the combined clip, so the prior's length
  shifted every window boundary — `chunk_frames` 124 gave one generated chunk after a
  10s prior and two after a 20s one. It now covers only the generated span (one carry
  plus the new region) and is offset onto the combined clip, which is what "start
  from the carry chunk forward" actually means. Measured identical across 3s, 5s,
  10s, 20s and 47s priors.

  The padding also settles the phase: `offset = prior_t - overlap = (5a+2) - (5m+2)`
  is a multiple of 5, so windows still start on phase 0, which H3's
  `FRAME_PER_TOKEN (1,4,4,4,4)` indexing requires.

  Audio is resized from the combined video length rather than padded by the same
  count, since it runs at 40Hz against 24fps.

  The tests now assert **coverage** — that every latent past the prior was written —
  which is what nothing checked before, and is why both faults shipped.
- **The prior's tail was being altered.** The first generated chunk overlaps the
  prior — that is what the carry is — so its slice covers the prior's last `carried`
  latents and the write-back landed on them. The noise mask protects that region only
  at strength 1.0, and `overlap_strength_audio` now defaults to **0.9**, so the
  source's last fraction of a second came back regenerated.

  The prior is now restored verbatim after the loop. The carry has already served its
  purpose as context by then, so nothing is lost: the generated content was still
  conditioned on the true prior. Output is the input plus the addition, byte for byte,
  at any strength.

  The 0.53.0 test missed this because it asserted the prior only up to `pt - 7` —
  excluding exactly the region that was changing. It now checks every prior latent,
  video and audio, and repeats the check at strengths 0.5 / 0.9.
- **A prior whose audio does not match its own video is refused.** `VAEEncodeAudio`
  counts audio from the track's duration, independent of the video latents, and
  encoders routinely pad past the last frame. The two axes are concatenated
  separately, so a mismatch would shift everything after the prior with no error. The
  message gives the drift in seconds and points at `MMH3 Trim AV`.

## [0.53.1] - 2026-08-10

### Changed
- **`MMH3LoopingSampler`'s `overlap_strength_audio` default is 0.9**, was 1.0. 1.0
  fully pins the carried audio and produces tinny second-chunk audio; 0.8–0.95 were
  both measured good. The default was sitting on the one value known to fail.

  Existing workflows are unaffected — ComfyUI serialises widget values, so a saved
  graph keeps whatever it already had. Only newly added nodes pick up 0.9.

  `MMH3SeedOverlap`'s matching input is left at 1.0: same mechanism, but the
  measurement was taken on the looping sampler and has not been repeated there.

## [0.53.0] - 2026-08-10

### Added
- **`prior_av_latent` on `MMH3LoopingSampler`** — continue an existing render. The
  prior is copied to the output verbatim and never sampled; `latent` describes only
  the new region, and the output is prior + new.

  The schedule is planned over the **combined** clip and every window finishing
  inside the prior is skipped. Whichever window first reaches past it becomes chunk
  0, already overlapping it, so `prev_end = prior_t` feeds the ordinary carry rule
  and the prior's tail is carried like any earlier chunk's. That is why the prior's
  length does not have to line up with anything: the skip is computed, not assumed.
  Tested at 57, 124, 192, 311, 481 and 900 prior frames.

  Prompts map to the GENERATED chunks — cond 0 is the first chunk actually sampled,
  not the first window of the combined clip.

  Appended as the last input, so no existing link or widget moves. Refuses a prior
  whose channels or frame size differ from the target, and refuses mixing an AV
  prior with a video-only target.

### Observed
- **`overlap_strength_audio` 0.8–0.95 both sound good**; 1.0 gives tinny second-chunk
  audio (0.52.0). Recorded in `docs/looping-sampler.md` §9. The default is still 1.0
  and is now known to be the wrong end of the range.

## [0.52.0] - 2026-08-10

### Fixed
- **`overlap_strength_audio`'s tooltip asserted a value, and the value was wrong.**
  It said *"Lipsync wants this at or near 1.0"* — a guess, never measured, sitting in
  the UI where it reads as fact. A T2VA run on 2026-08-10 produced **tinny
  second-chunk audio at exactly 1.0**, which is also the default. The tooltip was
  steering toward the failure.

  Both copies now state the mechanism only: `mask = 1 - strength`, 1.0 pins the
  carried audio to the previous chunk, 0.0 regenerates it, set independently of the
  video strength. Same in `MMH3SeedOverlap`.
- **`MMH3ReferenceMultiPrompt`'s `length` tooltip was stale**, still describing the
  pre-0.47 model — *"this is one CHUNK, and the master is however many chunks
  accumulate to"*. Both the looping sampler and context windows have taken the
  latent as the WHOLE clip since 0.47.0. It now says so, and drops a hardware-
  dependent VRAM claim the sampler already warns about at runtime.

### Changed
- **Conjecture removed from tooltips across the pack.** A tooltip says what a control
  does — scale, extremes, interactions — and does not recommend a value or guess an
  outcome. Audited all of them; 16 carried advisory language, and the ones that were
  guesses rather than citations are rewritten: `fuse_method` ("usually what you
  want"), `feather_latents` ("Untested"), `on_problem` ("worth it when…"),
  `MMH3ReframePads` mode ("usually right for an orientation flip"), and
  `prior_context_mode` ("Usually the right choice", added earlier the same day).

  Attributed advice stays — `ref_image_size` citing MiniMax's recommendation for
  faces is a citation, not a guess.
- `docs/looping-sampler.md` gains an **Observed** section, dated and tied to a run,
  which is where findings like the audio one belong. "Not yet measured" is now §10
  and no longer claims nothing has been generated, because things have.

## [0.51.0] - 2026-08-10

### Added
- **`MMH3PromptAccumulate` gets `prior_context_mode`** — `all` (default, unchanged) /
  `last` / `last_definitions`. Appended, so nothing rebinds.

  `prior_context` re-sent **every** earlier prompt in full, every iteration. On a
  127s clip in 20.75s windows that is 31.5 KB by window 7 — about 7,900 tokens of
  "here is everything you already wrote" against a few hundred tokens for the new
  audio. Roughly 20:1 in favour of copying, and the late windows duly re-describe
  the early ones: in one observed run three of seven windows shared a cut list and
  the same lyric appeared in four.

  The sharper problem was *what* it re-sent. The header says
  *"only detailed_description should differ"* — and then supplied every earlier
  `detailed_description` as an example to match. The instruction and the payload
  pointed in opposite directions.

  `last_definitions` sends only the previous window's `subject_definitions` and
  `retention_analysis`: the sections that must stay byte-identical, and nothing
  else. `summary` is deliberately left out too, so it can describe its own window.
  Measured on a six-window prior: 2,037 chars → 384.

  A prompt whose sections will not parse falls back to sending it whole, with a
  logged warning. Consistency is this output's entire job, so sending nothing is
  the one wrong answer.

### Changed
- The report now says how much `prior_context` is carrying — mode, how many earlier
  prompts, chars and approximate tokens — and warns past three windows on `all`.
  None of this was visible before, which is why the growth went unnoticed.

## [0.50.0] - 2026-08-10

### Changed
- **All 34 nodes are filed into submenus**, mirroring how LTXAVTools organises its
  pack. Everything used to sit flat under `MMH3Tools`, which is a 34-item menu with
  no grouping — fine at ten nodes, not at thirty-four.

  | category | n | |
  |---|---|---|
  | `MMH3Tools` | 2 | the two plain calculators, exactly where LTXAVTools puts its own |
  | `MMH3Tools/sampling` | 2 | LoopingSampler, ContextWindows |
  | `MMH3Tools/calculators` | 3 | WindowPlan, KeyframePlanner, UpscaleLadder |
  | `MMH3Tools/prompt` | 6 | AssetPlan, TaskSystemPrompt, WindowContext, PromptAccumulate, ReplaceSection, PromptLint |
  | `MMH3Tools/conditioning` | 3 | ReferenceMultiPrompt, CondSelect, CondSetSpread |
  | `MMH3Tools/reference` | 5 | the ImageToRef / LatentToRef / Keyframe family |
  | `MMH3Tools/latent` | 6 | PackAV, SplitAV, JoinAV, ConcatAV, TrimAV, SeedOverlap |
  | `MMH3Tools/audio` | 1 | SplitAudioToWindows |
  | `MMH3Tools/utils` | 6 | LatentInfo, FindDivergence, ReframePads, OutpaintLatent, the two Streaming nodes |

  `sampling`, `calculators`, `latent`, `audio` and `utils` carry the same names
  LTXAVTools uses, so the two packs read the same way. Where the packs genuinely
  differ the names differ: LTXAVTools has `IC-LoRA`, `Training` and `dataset`, which
  MMH3 has no equivalent of; MMH3 has `reference`, `conditioning` and `prompt`,
  which is where most of its surface actually is.

  **Saved workflows are unaffected.** ComfyUI resolves a node by its `node_id`, not
  its category — a category is only the Add Node menu path and the search index. No
  `node_id`, input, output or widget changed.
- The `NODES` registration list is grouped and commented to match, so the file reads
  in the same order as the menu.

## [0.49.0] - 2026-08-10

### Fixed
- **The clamped final chunk overwrote the one before it, under the wrong prompt.**
  Core pulls the LAST window back so it ends on the clip end, which makes it
  physically overlap its predecessor by far more than the nominal overlap — 62
  latents against a nominal 7 on a 127s clip in 20s chunks. `carried` was computed
  from the nominal value, so the other 55 were regenerated under the *last* chunk's
  conditioning and written over content the previous chunk had already drawn.

  Measured before the fix: **187 frames (7.8s)** clobbered at 127s/481, **289 frames
  (12.0s)** at 180s/481, 68 frames at 210s/192. It reads as the last section's prompt
  bleeding backwards over the tail of the second-to-last — and because it depends on
  how badly the clip length divides by the stride, it looks intermittent between runs.

  `carried` now comes from where the previous window actually ended, so a clamped
  tail conditions on more context and generates only genuinely new content. Middle
  windows are unaffected: there actual == nominal, which is why every existing
  fixture passed straight through this. The report names a clamped tail when it
  happens, and the per-chunk line now prints the real carry instead of the nominal.

### Added
- **`MMH3WindowContext`** — one line of text telling the writing model which span of
  the song a window covers, for the per-window prompt loop.

  Without it the loop hands the model the same text every iteration and only the
  audio changes, so on a repetitive track — where the windows sound alike — nothing
  distinguishes window 5 from window 2. Add `MMH3PromptAccumulate`'s `prior_context`,
  which says to keep the earlier sections' definitions byte-identical, and there is
  one strong instruction pulling toward sameness and none pushing back. Observed
  result: the late windows re-describe the same shots and the same ending, and one
  lyric lands in four consecutive prompts.

  The span comes from `_plan` — the same function `MMH3WindowPlan`,
  `MMH3SplitAudioToWindows` and `MMH3LoopingSampler` use — so the timecode names the
  audio the window actually renders. `MMH3SplitAudioToWindows` already emitted
  `first_frame`/`last_frame` and they would have worked, but they come from a
  different path and would drift the first time the schedule changed.

  It also states that `[Shot N]` timestamps stay window-local. That is the obvious
  failure of handing a model a song timecode: it starts writing `[Shot 2] At
  01:41.300`, which H3 reads against the window's own clock. Toggle with
  `state_local_times` if the system prompt already says it.

  Concatenate onto the **END** of the writing model's prompt — it has to outweigh
  `prior_context`, which sits at the end of the system prompt.

## [0.48.0] - 2026-08-10

### Added
- **`MMH3LoopingSampler` can window its sigma schedule**, and switch solver
  mid-schedule. Both ported from LTXAVTools' looping sampler with its semantics
  unchanged, since the two nodes should not disagree about what a step number means.

  * `sampling_start_step` / `sampling_end_step` — absolute indices into the incoming
    schedule, sliced exactly as core `SplitSigmas` does (first output
    `sigmas[:step+1]`, second `sigmas[step:]`, sharing the boundary sigma). `end`
    leaves a partially denoised latent; `start` re-noises to that sigma and finishes
    from there. Absolute means a two-pass run is `end N` then `start N`, with no
    arithmetic. An empty window raises instead of silently rendering nothing.
  * `phase2_start_step` + optional `phase2_sampler` / `phase2_guider` — a heavy
    solver for the first steps, something cheap for the rest. Rebased onto whatever
    window `sampling_start_step` leaves, so it stays absolute alongside the other
    two. A cut point outside the window is simply not a cut.

  All three apply **within every chunk**, not across chunks.

  `phase2_guider` goes through the same `_chunk_guider` rebind as the main one. Only
  its guidance settings are wanted; its own positive would hand the tail of every
  chunk whatever prompt happened to be wired to it — the same class of bug as the
  shallow-copy leak fixed earlier, arriving by a different door.

  These are not guide controls, and the docstring says so. A keyframe guide is
  registered on the conditioning and re-injected every step, so it is structural for
  the whole chunk; releasing it mid-schedule would mean changing the packed layout
  between steps, which is not expressible.

### Changed
- **The node's inputs are reordered**, and `carry` moves above the two overlap
  strengths it governs. Both their tooltips said "`mask` carry only" while sitting
  *above* the widget that selects the mode, which read backwards. Grouping is now
  sampling core / content / chunking / carry tuning / schedule / keyframes.

  This rebinds saved workflows, which is normally forbidden here. It is allowed
  because the node is **not published**, and the only local workflow using it already
  predated 0.47.0 — it still carried `chunks` and `overlap_latents` values, so it was
  rebinding wrong before this change. It has been migrated; the two values that no
  longer translate are reset to the node's defaults.
- **The sampler's denoised output is what gets written back**, not its first return.
  They are identical when the schedule reaches sigma 0, so no existing render
  changes — they diverge only when `sampling_end_step` stops early, which is the
  whole point of the new widget. LTXAVTools takes the denoised one for the same
  reason.
- The report names an active schedule window and an active phase 2, and says
  **PARTIALLY denoised** when the schedule stops short. A partially denoised master
  otherwise looks like a bad seed rather than a setting.

### Fixed
- `docs/looping-sampler.md` still documented `chunks` and `overlap_latents`, which
  0.47.0 replaced with `chunk_frames` / `overlap_frames`, and still described the
  latent as "one chunk's template".

## [0.47.0] - 2026-08-10

### Changed
- **BREAKING, and a redesign.** `MMH3LoopingSampler` takes the WHOLE clip's latent
  and derives the chunk count from it. `chunks` is gone; `chunk_frames` and
  `overlap_frames` replace `overlap_latents`. Nothing published depended on the old
  shape.

  It was inverted before: one chunk's latent plus a chunk count, which made the
  total emergent. LTXAVTools' sampler takes the whole clip and fills it, and that is
  the right way round -- you hold a song of known length and do not know how many
  chunks that is.

  Almost every rough edge of the last few versions came from the inversion:

  * the `length` widget meaning chunk where it read as duration, and a whole-clip
    value exhausting VRAM on chunk 0
  * `use_input_audio` giving every chunk the SAME slice, since one template was
    cloned
  * no way to ask how long the result would be, and a 7.8s overshoot when guessing
  * a `k+2` grid-safe trim costing ~7 frames per seam, because separately allocated
    chunks had to be concatenated
  * `_chunk_origins` and a hand-built master timeline, to answer questions the clip
    would have answered

  All of it goes. Chunks are slices of one master, sampled and written back in
  place: no join, no trim, no loss, and **the output is exactly the input length**.
  Each chunk slices its own span of audio, so a pinned track reaches every chunk.

  The schedule now comes from `_plan` -- the SAME function `MMH3WindowPlan` and
  `MMH3SplitAudioToWindows` use. Chunks and windows are one thing by construction,
  so the prompt written against window 3's audio is what chunk 3 renders, and
  `window_count` really is the chunk count rather than merely resembling it.

- `MMH3KeyframePlanner` follows: `total_frames` / `chunk_frames` / `overlap_frames`
  in, `chunk_count` out, same `_plan`.

### Fixed
- The planner put each chunk's destination at its span END. The final chunk is
  clamped to finish on the clip, so it overlaps its predecessor by much more than
  the nominal overlap -- and a frame at the predecessor's span end then sits inside
  the last chunk's NEW content, which claims it. The second-to-last chunk was left
  with no destination and the last got two. Destinations are now the last frame each
  chunk actually renders.

## [0.46.2] - 2026-08-10

### Fixed
- The whole-clip-as-a-chunk warning only reached the report string, which is returned
  after every chunk finishes -- and the failure it warns about is chunk 0 exhausting
  VRAM, so nobody ever saw it. It logs before sampling starts now.

## [0.46.1] - 2026-08-10

### Fixed
- `MMH3SeedOverlap` **discarded the target's own noise mask**. It built both masks
  from `torch.ones` and never read the incoming one, so a track pinned by
  `use_input_audio` was regenerated everywhere except the carry overlap -- silently,
  since the output still looked like a valid AV latent.

  Only reachable since 0.45.0 put a mask on the multi-prompt node's latent, and only
  under `carry="mask"`; `carry="keyframe"` never calls SeedOverlap so it was unaffected.

  The prepended carry has no incoming mask of its own -- those are new rows -- so it
  is still simply set. Everything after it now starts from what the target asked for,
  defaulting to full denoise, which leaves the no-mask case bit-identical.

  The feather composes with `minimum` rather than assignment, so easing back toward
  full denoise cannot un-pin a region the target deliberately preserved.

## [0.46.0] - 2026-08-10

### Changed
- The post-ref guide-origin correction is a **runtime wrap** now
  (`mmh3tools/patch_guide_origin.py`), not a core edit. Core is reverted to stock
  #15439 + #15375; both PRs stay applied, only this pack's own change comes out.

  A core edit for something no PR carries is a diff to re-apply after every `git
  pull`, and one more thing to remember when reading a bug report from someone who
  does not have it. As a wrap it travels with the pack and comes out the moment
  upstream lands its own -- `apply()` detects a core that already anchors guides on
  the target origin and declines.

  It wraps rather than rewriting source: `PackedLayout.__init__` is a plain method
  with no closure cells, so it wraps at a callable boundary, and the fix is a pure
  position shift applied after stock has built the layout. No dependency on core's
  source text, and no core source embedded in this pack.

  Inert unless BOTH guides and references are present -- with either alone the cursor
  never leaves `text_len` and stock is correct. Self-tested at import against the live
  class; rolls back rather than misplacing a guide.

  On `main` rather than `keyframe-anchors`, deliberately: the looping sampler needs it
  and there is no upstream PR to wait for.

## [0.45.1] - 2026-08-10

### Changed
- `MMH3SeedOverlap`'s log said *"trim N frames after decode"*, which reads as an
  instruction and is wrong inside `MMH3LoopingSampler` -- that joins in LATENT space
  and cuts `k+2` latents itself, a few frames more than the carry. It now states the
  fact: the first N frames reproduce the source and come off at the join.

- `MMH3LoopingSampler` warns when `latent` is longer than ~20s with more than one
  chunk. That is almost always the whole clip wired in where ONE CHUNK's latent
  belongs -- it runs, and renders `chunks` copies of the master. The `latent` tooltip
  now says to size it from `MMH3WindowPlan.window_frames`, not `total_frames`.

## [0.45.0] - 2026-08-10

### Added
- `MMH3ReferenceMultiPrompt` takes an optional `audio` and a `use_input_audio`
  toggle, both appended. On, the track is encoded here and written into the latent's
  audio half in place of silence, with a noise mask that pins it: video free, audio
  preserved. That is the case where you HAVE the soundtrack and only want the video
  generated against it. A `ref_audio` is a voice to imitate; this IS the audio.

  An encode will not land on the required `round(frames / 24 * 40)` exactly, so it is
  trimmed or padded. Padding is silence at the END -- looping a short track would put
  a seam somewhere no prompt describes.

### Fixed
- Wiring any `ref_audio` or `ref_video_audio` raised `AttributeError`. `_build_refs`
  called `MiniMaxH3ReferenceToVideo._encode_ref_audio`, but core has that as a
  MODULE-level function, not a classmethod. Every audio reference path in the node
  was dead.

## [0.44.1] - 2026-08-10

### Fixed
- `MMH3WindowPlan.overlap_frames` emitted **-12** when the overlap was 0. An overlap
  of 0 is legal, and 0 is off the 5j+2 grid, so `latents_to_frames` floored it to the
  group below. Fed into `MMH3SplitAudioToWindows` that re-snaps to something
  arbitrary. Uses `frame_at_latent`, the general form, which gives 0.

## [0.44.0] - 2026-08-10

### Added
- `MMH3KeyframePlanner` - end-anchored keyframe indices for a chained run, ported
  from LTXAVTools' `LTXKeyframePlanner`. Frame 0 opens, every later index is a
  chunk's own LAST frame, the final one is `-1`. Each chunk then generates toward
  its destination image and the next continues from the arrived state through the
  ordinary carry; start-anchoring would put each image in the NEXT chunk and invite
  a snap at every seam.

  It builds the schedule from the same numbers `MMH3LoopingSampler` uses, so the two
  cannot disagree. `scene_frames` overrides when scenes do not coincide with chunks.

### Fixed
- **The keyframe ownership rule used the wrong span.** It measured a chunk's head by
  what the CARRY covers, but what the join removes is the TRIM -- and under
  `carry="mask"` those differ, since the trim is `k+2`: 22 frames against a 17-frame
  carry.

  So a chunk's own last frame was assigned to the NEXT chunk, into the region that
  gets trimmed away. A keyframe placed there would have been rendered and then
  thrown out. Writing the planner is what surfaced it: its indices are exactly chunk
  ends, which is precisely the case that broke.

## [0.43.0] - 2026-08-10

### Added
- `MMH3LoopingSampler` takes `keyframes` (an IMAGE batch), `keyframe_indices` and
  `vae`, all appended. This is LTXAVTools' `optional_cond_image_indices` behaviour,
  ported: one index per image, comma separated, negatives counting from the end.

  **The indices are GLOBAL across the master, not per chunk** -- the same choice
  LTXAV's `_calculate_keyframe_per_tile_indices` makes. You place a shot where it
  belongs in the finished clip and the node works out which chunk owns it and what
  the local frame is. Only the arithmetic differs: H3's frames-per-latent is
  `1,4,4,4,4`, not a uniform scale.

  The master is chunk 0 whole, then every later chunk minus `trim`, so chunk i's
  local latent 0 sits at master latent `cum_i - trim`. That is a multiple of 5 in
  BOTH carry modes -- `(5a+2)-(5m+2) = 5(a-m)` -- so every chunk stays on phase 0 and
  `frame_at_latent` is valid on the origins.

  Ownership rule: consecutive chunks overlap, and a frame inside a chunk's carried
  HEAD is trimmed at the join, so anchoring there paints a frame nobody sees. Each
  index goes to the chunk that actually RENDERS it. Global frame 351 across four
  192-frame chunks lands in chunk 1 at local 181, not chunk 2 at local 11.

  Negatives are resolved here, not passed through: `PackedLayout` takes them
  literally, so `cond_t` would fall below `text_len` into the text token positions.
  Out of range raises rather than dropping a keyframe silently, as do a count
  mismatch against the image batch and a missing vae.

  Images are encoded ONCE, not per chunk. Guides are independent of `carry` -- they
  work with the masked carry too -- and a chunk's carry guide and user keyframes go
  on in a single `conditioning_set_values`, since setting it twice would replace
  rather than merge.

## [0.42.1] - 2026-08-10

### Fixed
- `MMH3LoopingSampler` crashed on a **Basic Guider**. `Guider_Basic.set_conds` takes
  ONE argument and its `original_conds` has no `"negative"` key at all, so reading the
  negative raised `KeyError` and calling `set_conds` with two raised `TypeError`. Both
  guider shapes work now; a Basic Guider simply has no negative to carry.

  Also documents what the guider is FOR, since it was not obvious: it supplies the
  model, the cfg, and the negative. Its POSITIVE is replaced every chunk from the
  cond_set, so whatever is wired there is ignored -- wire `MMH3 Cond Select` at index
  0 so the graph is valid and says what it means.

## [0.42.0] - 2026-08-10

### Added
- `MMH3WindowPlan` gains an `overlap_frames` output, appended.

  `context_length` and `context_overlap` are LATENTS -- they exist for
  `MMH3ContextWindows`. `MMH3SplitAudioToWindows` takes FRAMES. Wiring
  `context_overlap` into it re-snaps a latent count as a frame count and the
  splitter's schedule silently stops matching the plan: at 260/124/22 the second
  window starts at frame 102 with the real overlap and 119 with the latent one.

  Every LLM prompt after the first is then written against audio that window never
  renders, which reads as the model ignoring the prompt. The node already computed
  the frame value for its own report and just did not emit it.

## [0.41.1] - 2026-08-10

### Fixed
- `MMH3PromptAccumulate.prior_context` could not actually be used: the node sits AFTER
  the writing model, so feeding its output back into that model is a cycle within one
  iteration, and ComfyUI refuses it.

  `prompt` is now optional, so a SECOND copy at the top of the loop body -- fed only
  the loop's carried value -- reads the context before the model runs.

  `prior_context` is also derived from what came IN rather than from the result. With
  a prompt supplied that is the same set either way, but for the context copy, taking
  it from the result would drop the most recent window -- the one the model most needs
  to stay consistent with.

## [0.41.0] - 2026-08-10

### Added
- `MMH3PromptAccumulate` - append one prompt to a running pipe-separated string, for a
  for-loop that writes one prompt per window. Exists because a loop cannot hand a
  growing list between iterations, only values carried back through its END node, so
  per-window prompts have to accumulate as text and be split apart later - which is
  exactly what `MMH3ReferenceMultiPrompt` now takes.

  **The first pass is the case that goes wrong.** A loop's carried slot is unwired on
  iteration 0, so `accumulated` arrives as None. A naive accumulator emits a leading
  separator, or `str(None)` -> the literal text "None", and the downstream split reads
  either as a real prompt. None, "" and whitespace are all treated as "nothing yet".

  Also: strips code fences a writing model wrapped its answer in, refuses a separator
  with no pipe in it (which would silently produce ONE enormous prompt instead of N),
  reports when a prompt is identical to an earlier one - the loop may not be advancing
  - and emits a `count` that matches what `MMH3ReferenceMultiPrompt` will actually
  parse.

  `prior_context` formats the EARLIER prompts for feeding back to the writing model.
  Do not use an LLM node's `history` input for this when the node attaches audio or
  images: a chat history keeps the base64 of every prior turn, so a handful of windows
  becomes megabytes re-sent every iteration, and the model ends up looking at all the
  previous windows' audio while writing this one.

## [0.40.1] - 2026-08-10

### Fixed
- `carry="keyframe"` now refuses when a chunk carries a reference AND a guide on a
  core that anchors guides to `text_len` rather than the target origin.

  #15439 anchors on `text_len`, but the target begins at `cursor`, which the refs
  advance. Measured on the real `PackedLayout`, guide versus target origin: **-1**
  with one image reference, **-320** with a chunk's worth of voice audio, **-321**
  with both. Nothing errors -- the guide anchors into the reference region instead
  of the clip, and `cond_audio` goes with it, so the carried tail's AUDIO lands
  early too.

  It bites precisely the configuration #15439 exists to enable, since the same PR
  fixes the `cond_video_latents` clobber so guides and refs can coexist.

  This pack carries a local core correction for it (`docs/core-changes.md`) that no
  PR has yet, so the node MEASURES rather than assumes -- `_guide_origin_correct()`
  builds a probe layout and compares the two origins. Guides without a reference are
  correct on stock #15439, so those still run.

## [0.40.0] - 2026-08-09

### Added
- `MMH3LoopingSampler` gains `carry` (appended last). `mask` keeps the existing
  behaviour; `keyframe` passes the previous tail as a GUIDE anchored at frame 0.
  Requires **#15439**, and refuses up front rather than dying on chunk 1 after chunk
  0 has already been paid for.

  The guide carries a MULTI-STEP clip plus its audio at the same `cond_t` -- not a
  still, and no VAE round trip, since the tail is already latent. A 5m+2 tail off a
  5j+2 clip starts at step 5(j-m), always phase 0, so the slice is exactly what a
  fresh encode of those frames would produce.

  **It is also cheaper at the join.** `mask` grows the chunk by a multiple of 5, so
  the trim must be carry+2 to keep the master on the 5j+2 grid, and that +2 takes ~7
  frames of real content per seam. A keyframe carry is 5m+2 already, so trimming
  exactly it is grid-safe and nothing extra is lost. Over 4 chunks: 222 latents via
  `mask` against 207 via `keyframe`, both on grid, audio matching in each.

  Anchored at 0 rather than a negative index, because `PackedLayout` takes negatives
  literally -- `cond_t` would fall below `text_len`, into the text positions.

- Stale guide bookkeeping is stripped off incoming conditioning every chunk, in both
  modes. This node registers all of its own guides; anything arriving pre-registered
  came from an upstream guide node or a cond cached from a previous run, and would
  anchor the chunk to somebody else's frames. From LTXAVTools -- same leak, same cause.

## [0.39.0] - 2026-08-09

### Added
- `MMH3LoopingSampler` (PROTOTYPE) - N chained chunks in one node execution. Each chunk
  seeds its head from the previous tail via `MMH3SeedOverlap`, masks it so the model
  conditions on it without redrawing it, and generates forward. The graph is the same
  size for 4 chunks or 40, which is the whole point: driving N chunks from the graph
  costs a copy of every downstream node per chunk.

  The trade is that nothing inside the loop can be a graph node, so every prompt must
  exist BEFORE sampling starts - wire a `cond_set` from `MMH3ReferenceMultiPrompt`,
  which also means one text-encoder load for the whole sequence.

  Two things taken from LTXAVTools because they are not obvious. The per-chunk guider
  is `copy.copy` plus a REBOUND `original_conds` dict: the shallow copy shares that
  dict, `set_conds` assigns into it, and chunk 0 would overwrite the base conditioning
  for every later chunk. And per-chunk noise needs its seed bumped or every chunk gets
  identical noise, which reads as the model refusing to advance.

  Untested against real weights.

### Fixed
- `MMH3ConcatAV` sized its audio drop from `latents_to_frames(n_b - k)`, which is
  invalid when `k = 5m+2` because B's REMAINDER is off grid and the conversion floors
  to the group below. It dropped ~20 audio latents too many per seam and **compounded**:
  four chained chunks measured **1.48 seconds** short, against video that was correct.

  That is the family a chained loop is forced into, since only an on-grid TOTAL can be
  decoded. So when the total is on grid the drop is now taken from what the MASTER
  needs - `(A_audio + B_audio) - frames_to_audio_t(latents_to_frames(total))` - which is
  exact in every case the old formula got wrong. The `k = 5m` route is unchanged: B's
  remainder is on grid there, the total is not, and B's own difference is exact.

  The trim log also used `latents_to_frames` on the same off-grid value; it now uses
  `frame_at_latent`, which is the general form.

## [0.38.0] - 2026-08-08

### Changed
- **BREAKING.** `MMH3ReferenceMultiPrompt` takes ONE pipe-separated `prompts` string
  and ONE batched `ref_images` input, replacing the autogrow `prompt_1..32` and
  `ref_image_1..9` sockets. Up to 41 sockets become 2. These nodes are still
  developmental, so the append-only rule was set aside deliberately here; it applies
  again once workflows depend on them.

  The point is graph SIZE. A loop that accumulates one prompt per window wires
  straight into a single string input, so the graph is the same whatever N is.
  `ref_videos` / `ref_video_audios` / `ref_audios` keep their autogrow slots: a
  batched IMAGE is ambiguous for video (N stills or one N-frame clip), so only
  images could collapse.

  Empty pieces are dropped, so a trailing `|` or blank lines between prompts cost
  nothing. A literal `|` inside a prompt over-splits SILENTLY -- the `count` output
  is how many prompts were actually found.

### Fixed
- A batched image in a reference slot contributed only its FIRST frame. `_build_refs`
  sliced `img[:1]` per socket, silently discarding the rest. Every batch element is
  now its own `<Picture i>`, numbered in batch order.

## [0.37.0] - 2026-08-08

### Added
- `MMH3SplitAudioToWindows` gains an `index` input and `audio` / `first_frame` /
  `last_frame` outputs, all appended, so the numbered sockets keep their positions
  and saved workflows are untouched.

  The numbered sockets fan every window across the graph at once, which costs a copy
  of every downstream node per window -- two LLM nodes, an encode and a sampler,
  times N. That is what makes a graph heavy enough to break ComfyUI. The `audio`
  output emits ONE window, chosen by `index`; drive it from a for loop and the graph
  is the same size for 4 windows or 40.

  `index` also reaches past `MAX_WINDOW_AUDIO`. Every window is cut now, not just the
  eight with a socket, so the loop form has no window ceiling. The overflow note says
  so rather than claiming the tail is dropped.

  Out of range raises rather than wrapping, matching `MMH3CondSelect` -- a loop
  running one iteration too many should stop, not quietly re-render window 0.

  See `docs/context-windows.md` for the graph shape. Nothing carries between
  iterations: the reference image and system prompt are fixed and wire in from
  outside, only audio changes, and `forLoopEnd` carries no values.

## [0.36.0] - 2026-08-08

### Removed
- `MMH3InterpolateLatent`, and with it `gap_fill` / `gap_denoise`. **Saved workflows
  containing this node will fail to load.** The idea does not survive the 5j+2 grid.

  It placed source latent `i` at new index `factor * i`. But a latent's pixel-frame
  span is `FRAME_PER_TOKEN[k % 5]`, which is 1 or 4 - so multiplying the index
  reshuffles which sources land on 1-frame slots and which land on 4-frame slots.
  Measured against where uniform time-scaling would put them, over 12 source latents:

      factor 2:  drift 0,3,3,0,0 ...   max 3 frames
      factor 3:  drift 0,6,3,3,0 ...   max 6 frames
      factor 4:  drift 0,9,6,3,0 ...   max 9 frames

  Never zero at any factor, and a SAWTOOTH with period 5 latents = 17 frames = 0.7s
  at 24 fps. That is the judder, and at 4x its amplitude is 0.375s. The source content
  was being retimed, not interpolated.

  0.35.0's neighbour blend was aimed at the wrong layer: blending controls what fills
  the GAPS, but the error is in where the REAL frames sit. It could not have worked.

  A non-uniform `vpos` - placing each source at the new index whose frame OFFSET is
  nearest the ideal, the way `apos` already does for audio - would kill the periodic
  component. Residual drift would remain, irregular rather than rhythmic. Not pursued;
  recorded here in case it is worth revisiting.

- The duplicate `MMH3InterpolateLatent` import in `__init__.py` went with it.

## [0.35.0] - 2026-08-07

### Added
- `MMH3InterpolateLatent` gains `gap_fill` and `gap_denoise` (appended last), for the
  JUDDER observed on the first real run.

  Judder is a generated in-between frame landing plausibly but not ON THE LINE between
  its neighbours. Zeros give the model no bias at all, so plausible-but-wrong is a free
  choice. `gap_fill = interpolate` seeds each gap with a distance-weighted blend of its
  two real neighbours, and `gap_denoise` below 1.0 keeps the sampler mixing that seed
  back in at every step - the model corrects a guess instead of inventing.

  The trade is ghosting: a blend of two frames is a crossfade, so fast motion starts
  smeared and the model has to resolve it. Too low a denoise and it stays smeared.

  **0.5 is the threshold that matters**, since `mask_row_targets` binarises there for the
  per-row timestep. Above it a gap is treated as GENERATE and the seed only biases the
  trajectory; below it the gap is treated as content to KEEP and the blend largely
  survives, ghosting included. 0.6-0.8 is the useful band, and the report says which side
  you are on.

  Positions past the last source HOLD the final real latent rather than fading to black.
  They are tail padding and the trim removes them, but a black tail would still drag the
  last real frames toward it while sampling.

  `zeros` keeps the original behaviour and ignores `gap_denoise`, because mixing
  emptiness back in is not a bias, it is a fade.

## [0.34.1] - 2026-08-07

### Fixed
- **Windowing a MASKED latent crashed.** Any node that attaches a `noise_mask` -
  `MMH3SeedOverlap`, `MMH3OutpaintLatent`, `MMH3InterpolateLatent` - died in
  `_mod_scale_shift` when `MMH3ContextWindows` was also in the graph:

  ```
  RuntimeError: The size of tensor a (640) must match the size of tensor b (866)
  ```

  Core resizes `model_conds` entries only when they are raw tensors, plus hand-written
  cases for `audio_embed` and `vace_context`. A denoise mask is a `CONDRegular`, so it
  fell through UNWINDOWED and the model got a full-length mask against a windowed
  latent - a mod-row weight vector sized for the whole clip, applied to one window.

  `LTXAV` handles this by overriding `resize_cond_for_context_window` on the model class.
  `MiniMaxH3` has no such override, and adding one would mean patching core. This uses
  the handler's own `RESIZE_COND_ITEM` callback instead: no core change, and it stays on
  `main`.

  Each modality is cut on ITS OWN axis - video `[B,1,T,h,w]` on dim 2, audio
  `[B,1,2,T40]` on dim 3. One shared dim would slice audio on its stereo axis, the same
  trap `MMH3WindowingState` exists to avoid.

## [0.34.0] - 2026-08-07

### Added
- `MMH3InterpolateLatent` - spread a latent's frames apart at stride `factor`, mask the
  gaps, and let the model fill them. **Frame rate, not slow motion**: the model has no
  concept of fps, so this makes a clip `factor` times longer and you get the rate
  increase by saving at `factor` x 24 fps.

  ```
  57 latents (192 frames, 8.00s) -> 117 latents (396 frames)
    save at 48.0 fps and trim to 384 frames -> 8.000s, unchanged
    57 of 117 latents are real (51% generated); audio 320 -> 660
  ```

  **The mask survives here, unlike the spatial version of the same idea.**
  `mask_row_targets` max-pools over 2x2 SPATIAL patches and leaves `latent_t` alone, so a
  per-latent temporal alternation reaches the model exactly as written. A per-cell
  spatial checkerboard is flattened to "everything generates" and accomplishes nothing.

  **The factor must be coprime with 5.** A latent at index k is read as spanning
  `FRAME_PER_TOKEN[k%5]` frames - the 1,4,4,4,4 cycle - so moving source i to `factor*i`
  preserves what it MEANS only when `gcd(factor, 5) == 1`. Only 2, 3 and 4 are offered.

  **Audio is interleaved, not stubbed.** H3 conditions video on audio, so silence in the
  gaps would tell the model to close mouths. Real audio at the source positions keeps
  that conditioning honest; whatever it invents between is discarded when the original
  track is muxed over the trimmed result.

  The padded length rarely equals exactly `factor` x the source frames, because it has to
  land on the 17j+5 grid - hence `trim_to_frames`, which is what makes the duration come
  out unchanged.

## [0.33.2] - 2026-08-07

### Changed
- Documented an observed result: **outpainting converges in about HALF the steps of a
  normal generation**, with the scene filling in at ONE step and the rest going to
  detail.

  That follows from the architecture rather than being luck. H3 has no cross-attention -
  everything sits in one packed sequence - so the margin rows attend DIRECTLY to the
  source rows at every layer, not to an encoder's summary of them. Spatial infill has its
  answer visible in the same frame, unlike a temporal continuation where motion has to be
  invented, so composition, palette and lighting settle almost immediately.

  It halves what an aspect change costs, which changes the `MMH3ReframePads` maths:
  `extend` at ~9.8x the attention per step is ~4.9x the generation. The report now says
  both numbers.

## [0.33.0] - 2026-08-07

### Added
- `MMH3ReframePads` - source size plus a target aspect in, four SIGNED edge moves out,
  ready for `MMH3OutpaintLatent`.

  **The mode is the real decision**, and it is a trade rather than a preference. For
  1344x768 to 9:16:

  | mode | result | pixels | attention/step |
  |---|---|---|---|
  | `extend` | 1344x2400 | 3.12x | **~9.8x** |
  | `crop` | 448x768 | 0.33x | 0.1x, and 67% of the frame gone |
  | `balanced` | **768x1344** | **1.00x** | 1.0x |

  `balanced` crops the long axis part of the way and extends the short axis the rest,
  landing at the SOURCE pixel count. For an orientation flip that is almost always the
  answer: pure extension is unaffordable because attention is quadratic, and pure
  cropping throws away most of the frame. It is the default.

### Changed
- **`MMH3OutpaintLatent`'s four sides are SIGNED**: positive moves an edge outward (pad,
  generated), negative inward (crop, discarded). An edge can only ever go one way, so a
  separate crop input per side would be four widgets obliged to stay zero whenever their
  partner is not. Six inputs instead of ten, and `MMH3ReframePads` emits four values
  rather than eight.

  Snapping truncates toward ZERO. `int() // 32` floors, which would send `-33` to `-64`
  and silently crop twice what was asked for; the magnitude is snapped and the sign
  reapplied.

- Reframe rounds to the NEAREST canvas step rather than up. Rounding up meant a source
  already at the target ratio still grew by 32px - 1344x768 is 1.75 and 16:9 is 1.778,
  close enough that the right answer is to do nothing. It now says so, and suppresses the
  "landed on x rather than y" quibble when nothing moved.

## [0.32.0] - 2026-08-07

### Added
- `MMH3OutpaintLatent` - zero-pad an AV latent spatially and attach a feathered denoise
  mask, so a full-denoise pass generates the margin while keeping the original.

  **Zeros, not encoded padding.** Encoding padded pixels bakes STRUCTURED content: black
  or grey encodes to a non-zero latent the model reads as "something is here" and tries
  to preserve, which is where the black-edge artefact comes from. A zero margin is the
  same empty substrate a from-scratch generation starts from.

  **The feather ramps INWARD**, into the source. Feathering outward into the margin would
  blend toward empty and muddy the seam; ramping inward partially regenerates the
  original's outer band, which is what hides the join. Per-axis ramps combine with
  `max()` so a corner takes the stronger of its two rather than their sum.

  **What the feather does here, precisely.** The mask reaches the model twice: the
  sampler's `x*mask + orig*(1-mask)` blends the LATENT continuously, and
  `mask_row_targets` binarises at 0.5 per 2x2 patch to pick a per-row AdaLN timestep. So
  the content is a gradient while the treatment is a step. The report counts how many
  ramped cells cross the threshold rather than pretending the feather is free:

  ```
  1344x768 -> 1344x1280 px (latent 84x48 -> 84x80)
    margin is 40.0% of the frame | feather 64 px = 4 latent cells
    504 ramped cells, 336 of them above the 0.5 threshold ...
  ```

  If the contour shows, pad wider than needed and crop back so the step lands outside the
  final frame.

  Padding is in pixels snapped to 32, which is what keeps the latent dims EVEN - the
  DiT's 2x2 patch grid needs that, and an odd dimension fails deep in the model rather
  than at the node. Audio is untouched and its mask is all-preserve. Needs **#15375**.

## [0.31.0] - 2026-08-07

### Changed
- **`MMH3SeedOverlap` returns to `main`.** The branch is for MONKEYPATCHES — wraps this
  pack maintains indefinitely — not for "anything needing a core change". SeedOverlap
  needs **#15375**, an upstream PR that will merge; applying one of those is ordinary,
  and the node already refuses to run without it rather than appearing to work.

  `keyframe-anchors` narrows to what is genuinely ours: `patch_layout.py`,
  `patch_conds.py`, and `MMH3LatentToKeyframes`, which depends on both.

- `docs/core-patches.md` becomes **`docs/core-changes.md`**, organised around that line:
  PRs you apply versus wraps we maintain, with the revert-and-update procedure and the
  #15371 cautionary tale.

### Fixed
- **`MMH3SeedOverlap`'s docstring overstated partial strength.** It said the AdaLN lerp
  "is what makes a partial strength mean anything". The lerp is real and continuous, but
  the weight reaching it is binarised on the way in:

  ```python
  target  = m.reshape(-1) >= 0.5          # mask_row_targets
  video_w = targets.to(torch.float32)     # 0.0 or 1.0, never between
  ```

  So `overlap_strength` 0.3 and 0.4 both land as "preserved" for TIMESTEP purposes.
  Partial strength still blends the LATENT continuously, through the sampler's own
  `x*mask + orig*(1-mask)` — that part was always true. Both are 0.5-thresholded per
  2×2 patch, since `mask_row_targets` max-pools before comparing.

  That threshold is a choice in an open PR, not a property of H3. Worth re-checking if
  #15375 changes before merge.

## [0.30.0] - 2026-08-07

### Added
- `MMH3TrimAV` - drop video latents from the head and/or tail of an AV latent, cutting
  audio and masks to match. Closes a real hole: `MMH3ConcatAV` could only trim B's head
  WHILE joining, so a single latent could not be cut at all, and `MMH3FindDivergence`
  emitted a `trim_frames` count with nowhere to send it except `MMH3JoinAV` in pixel
  space, after a decode.

  **The grid rule INVERTS relative to `MMH3ConcatAV`.** Trimming one latent:

  | trim | result |
  |---|---|
  | `5m` | `5(j-m)+2` - **on** grid, and removes an exact overlap |
  | `5m+2` | `5(j-m)` - **off** grid |

  In ConcatAV it is the other way round, because there the constraint is on the JOINED
  total rather than on the piece being cut. Same arithmetic, different subject. The
  report names which you got.

- `MMH3SplitAV` - pull an AV latent into plain video and audio latents. The exact
  inverse of `MMH3PackAV` (round-trip is bit-identical), so carrying stage 1's audio
  through an upscale ladder becomes something the graph expresses rather than a
  discipline of never wiring the sampler's audio anywhere.

### Note on the audio math
Both convert BOUNDARIES independently and subtract, via `_audio_index_at`. Two traps
make the naive version wrong: `audio_t = round(frames / 24 * 40)` is not additive, and
`latents_to_frames()` is only meaningful ON the 5j+2 grid - it floors to the group below
for anything else. A head trim of 5 latents is off-grid, and a formula built on
`latents_to_frames` gets its audio count wrong by 17 latents. Tested both ways.

## [0.29.1] - 2026-08-07

### Changed
- The `sung lyrics` rules now time cuts to the voice: place them on a breath, phrase end
  or audible rest, and never inside a sung word. A cut mid-vowel leaves the mouth open on
  both sides of the join and reads as broken however good the lipsync is.

  From MiniMax's own MV skill, which states it as a hard rule. Costs nothing at runtime -
  the model already hears each window's audio via `MMH3SplitAudioToWindows`, and the shot
  timestamps in `detailed_description` ARE the cuts, so this is a prompt rule rather than
  a beat-detection pipeline.

## [0.29.0] - 2026-08-07

### Changed
- **`MMH3TaskSystemPrompt` no longer hands the model a copyable marker menu.** The old
  `retention_analysis` block was:

  ```
  retention_analysis - one line per label, with a marker:
      visible: fully_preserved | partially_preserved | attribute_transfer | weak_reference
      audio:   fully_copy | partially_copy | reference | weak_reference
  ```

  Indented, colon-terminated, sitting directly under the section name - it reads as a
  line to WRITE, and models duly copied it verbatim. The section then looked populated
  while saying nothing about any asset.

  It now shows two worked example lines instead, names the allowed values in prose, and
  forbids writing a list of them. A test asserts no pipe-separated marker list survives
  anywhere in the emitted prompt, so the template cannot come back.

  0.28.1 taught the lint to CATCH the echo. This removes the cause; the lint check stays
  as the backstop.

## [0.28.2] - 2026-08-07

### Fixed
- **The `[Shot 1]` check knew only one of the two formats**, and flagged correct Ref2VA
  prompts. They differ:

  | | style |
  |---|---|
  | A (`T2VA` etc) | `[Shot 1] <style>, <shot 1>` - INSIDE shot 1 |
  | B (`Ref2VA`) | "One or two style sentences **BEFORE** `[Shot 1]`" |

  Format B now allows the lead-in and checks what it should be instead: no timestamp in
  it, and not so long that shot content has leaked in. Format A still requires the body
  to open with `[Shot 1]`. Both formats are checked for having a `[Shot 1]` at all.

### Added
- An undefined label in the **summary** is caught. The format says "reuse existing labels
  only; introduce none here", but only `detailed_description` was ever checked - a
  prompt citing `<Picture 1>` and `<Picture 2>` in its summary with neither defined
  linted clean.
- A retention marker written into `subject_definitions` is caught. Anchored to a marker
  POSITION (after a comma or colon, at end of line) rather than the bare word, because
  `reference` is also ordinary prose: "the voice-timbre reference for <Subject 1>" is
  correct and matched a naive word check.

## [0.28.1] - 2026-08-07

### Fixed
- `MMH3PromptLint` validates that `retention_analysis` says what survives. A prompt
  whose section was the marker MENU echoed back -

  ```
  retention_analysis:
      visible: fully_preserved | attribute_transfer | weak_reference
  ```

  - linted **clean**. The section looked populated, every other check passed, and it
  stated nothing about any asset, which is the section's only job. Three checks now:
  the menu repeated instead of chosen from, a label line carrying no recognised marker,
  and a `<Subject N>` defined with no retention line at all.

  A `<Picture N>` folded into a `<Subject N>` definition is deliberately NOT required to
  have its own line, because the format says it gets none.

## [0.28.0] - 2026-08-07

### Added
- `MMH3ReplaceSection` - splice a rewritten section back into a prompt, so a refinement
  pass cannot drop the rest.

  Asking one instruct model to reproduce five sections VERBATIM and rewrite the sixth is
  the fragile half of the job. A Mistral refinement pass reliably did the rewrite and
  returned the body alone, at which point the lint reported all six sections missing -
  which reads as total failure when the expansion itself was fine. Give the refiner one
  job (return the new body, no labels) and let this node hold the structure. Dropping a
  section becomes impossible rather than unlikely.

  Code fences, a repeated label and markdown decoration are stripped on the way in,
  since the text encoder receives those characters literally.

### Fixed
- `MMH3PromptLint` NAMES a decorated label instead of calling it missing. Instruct
  models format their output as a document - `**subject_definitions:**`, `### summary` -
  and every one of those counted as absent. Six baffling absences now read as six
  instances of one fixable thing, with the offending text quoted. It remains an error:
  H3 was trained on plain labels and the encoder sees the asterisks.

## [0.27.0] - 2026-08-07

### Added
- `MMH3StreamingSave` - decode the video latent in chunks straight into ffmpeg. The
  full pixel tensor never exists, so RAM is constant at any length:

  | | decoded whole | one 17-frame clip |
  |---|---|---|
  | 1344x768, 362f | 4.48 GB | 0.211 GB |
  | 2048x1152, 362f | 10.25 GB | 0.481 GB |
  | 2048x1152, 750f | **21.23 GB** | 0.481 GB |

  **This is NOT the LTX pattern.** `LTXAVStreamingSave` decodes with LEFT context and
  trims, because the LTX VAE is causal. H3's decoder is neither causal nor independent:

  ```python
  t_end_idx = t_start_idx + tokens_chunk_size + token_overlap    # 5 + 2 LOOKAHEAD
  clip_dec_chunk = self.blend(dec_overlap, clip_dec_chunk, self.frame_overlap)
  ```

  It reads two latents ahead and carries `dec_overlap` forward to blend 5 frames into
  the next chunk. A slice decoded alone is wrong at BOTH ends. (Note this is the
  opposite of `MMH3StreamingEncode`, where clips encode independently and chunking is
  bit-identical for free - encode and decode are not symmetric here.)

  So: work in groups of 5 latents (= 17 frames each, the `5j+2` <-> `17j+5` grid seen
  from the VAE's side), decode `[5*g0-5 : 5*g1+2]`, discard the context group's frames,
  and **drop the trailing 5 except on the final batch** - a partial decode writes its
  last chunk's carried part raw, which a full decode never does.

  Verified exact against core's real `decode_temporal` and its real `blend`, with a
  provenance-encoding stub decoder: identical to a full decode at T = 12, 22, 37, 57,
  107 and 1/2/4/100 groups per chunk. The same harness shows the no-context version is
  NOT exact, so the test is sensitive to the thing it claims to check.

  ffmpeg handling matches `LTXAVStreamingSave`: encoder probing with per-encoder
  quality mapping (crf / cq / bitrate / qscale), a binary search order that prefers one
  which actually HAS a working encoder over the first on PATH, stderr to a file rather
  than a pipe that nothing drains, and audio muxed at the end.

## [0.26.0] - 2026-08-07

### Added
- `MMH3SplitAudioToWindows` - cut a track into one clip per context window, so an LLM
  that can hear writes each prompt against the audio that window actually renders.

  A uniform sequential split cannot express the schedule. Windows overlap AND the last
  one is clamped to the clip end, so at 362 frames with a 124/22 window the real spans
  are `0-123, 102-225, 204-327, 238-361` - a uniform stride of 102 would put the fourth
  at `306-429`, past the end of the clip and over audio the model never sees there. The
  prompt written from it would describe music that is not in that window.

  Takes ONE window length rather than a per-segment frame count, because the schedule
  is uniform by construction and the clamping is derived, not chosen. Mono is widened
  to stereo, short tracks are padded rather than yielding ragged segments, and more
  windows than outputs is reported rather than silently dropping the tail.

### Changed
- `_plan()` and `_window_frame_spans()` are now shared by `MMH3WindowPlan` and
  `MMH3SplitAudioToWindows`. If the splitter's spans drifted from the planner's, every
  prompt would be written against audio its window never renders - and it would look
  like the model ignoring the prompt, not like a timing bug.

## [0.25.0] - 2026-08-07

### Changed
- **`main` now runs on stock ComfyUI. Everything that needs a modified core is on the
  `keyframe-anchors` branch.** Moved:

  - `docs/core-patches.md` and `core-patches.diff` - instructions for editing core do
    not belong on a branch that claims not to need them.
  - `MMH3SeedOverlap`. Its own docstring said it "REQUIRES the per-row masking patch;
    without it the node runs but does nothing useful" - stock has no per-row TIMESTEP
    handling, so preserved rows run at the generation timestep and the mask accomplishes
    nothing. Fixing that means editing the DiT's forward. drozbay's per-row masking is
    open upstream as **#15375**; when it merges the node returns unchanged.

- `MMH3ImageKeyframe` **refuses** an interior `frame_index` instead of warning and
  failing deeper in. Stock `PackedLayout` raises on it; warning only moved the error.
- Both keyframe nodes document the truth about references: stock `extra_conds` assigns
  `cond_video_latents` from keyframes and then assigns it AGAIN from references, so any
  reference silently erases every keyframe. The README previously said they compose.

### Note
This is a removal, and it is meant to be reversible. The branch has everything, and
#15375 landing brings `MMH3SeedOverlap` straight back.

## [0.24.1] - 2026-08-07

### Added
- `MMH3WindowPlan` emits `window_frames` (appended last). Under windowing the layout is
  rebuilt **per window** from that window's `latent_t`, so anything taking a
  `target_frame_count` - the keyframe nodes - needs the window's length, not the clip's.

### Fixed
- `MMH3CondSetSpread` now reports where keyframes sit, because they do not behave the
  way the rest of the conditioning does under windowing.

  `patch_latent_shapes` swaps `latent_shapes` for the window's, and `extra_conds` builds
  the layout from `latent_shapes[0][2]`, so `latent_t` is the window's. A first-frame
  anchor is `cond_t = kf_base`, the target origin - which is *that window's* frame 0:

  ```
  FIRST-frame keyframe      whole clip  latent_t=107  keyframe at 320.0
                            one window  latent_t= 37  keyframe at 320.0
  ```

  So an i2v start image is re-imposed at the start of **every** window. A last-frame
  anchor is worse: `minimax_frame_count` is *not* patched per window, so the index check
  still matches the clip (361) while the position comes from the window - landing at
  525.0, the window's end, instead of 921.7.

  Entry 0 is the exception rather than an offender: region 0 *is* the first window, so a
  start frame there lands where it belongs. The report confirms that case and flags only
  keyframes on later entries.

## [0.24.0] - 2026-08-07

### Added
- `MMH3WindowPlan` - work the whole windowing schedule out up front, in frames, and
  emit the latent values the chain needs: `context_length`, `context_overlap`, the
  snapped `total_frames`/`total_latents`, and **`window_count`**.

  Three things were previously only knowable by running a generation: whether your
  window and overlap survive snapping, how many windows you actually get, and which
  frames each one covers. The middle one is the number of prompts to write for
  `split_conds_to_windows` - guess low and windows share a prompt, guess high and the
  last prompts are never reached. Set `prompt_count` and the report says which prompt
  each window would use, and names any that are unreachable.

  The count comes from calling core's own scheduler rather than reimplementing the
  stride arithmetic, so it cannot drift from what sampling really does. A test asserts
  the emitted values pass through `MMH3ContextWindows` unchanged and that the predicted
  count matches the real schedule - otherwise the plan would be a lie.

- `frame_at_latent()` in `common.py` - first pixel frame of ANY latent step.
  `latents_to_frames()` inverts the 5j+2 grid and is only meaningful on it; window
  bounds are arbitrary indices, and asking it about index 1 returns **-12**. That bug
  was live in the planner's first output, reporting a window starting at frame -13.

## [0.23.0] - 2026-08-07

### Added
- `MMH3ContextWindows` gains `split_conds_to_windows` (default off, appended last), and
  `MMH3CondSetSpread` produces the conditioning shape it needs.

  Without it every window is handed the same conditioning, so the model is asked to
  render the whole script into each one - which is what a windowed pass looked like it
  was doing. With it, core picks a prompt per window from the window's own midpoint:

  ```
  center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)
  region       = int(center_ratio * len(cond_in))
  ```

  so prompt 0 covers the start of the timeline and the last covers the end. Verified
  against a real schedule: windows `0-21, 15-36, 30-51, 35-56` map to regions
  `0, 1, 2, 2` - monotonic, every prompt reached, first and last correct.

  `MMH3CondSetSpread` flattens a `cond_set` into ONE conditioning holding every prompt
  in order. Core only splits when a conditioning carries more than one ENTRY, and the
  cond_set holds N separate conditionings, so they have to be concatenated rather than
  nested. `MMH3CondSelect` still takes one prompt for one chunk; this takes all of them
  for one windowed pass. References are shared either way - the cond_set encoded them
  once - so identity does not shift as the region changes, only the prompt does.

  With a single prompt this is a no-op, and the node's report says so rather than
  leaving you to wonder why nothing changed.

## [0.22.1] - 2026-08-07

### Changed
- Keyframe anchoring moved to the **`keyframe-anchors`** branch: `MMH3LatentToKeyframes`,
  `mmh3tools/patch_layout.py`, `tests/test_keyframe_carry.py`, and the
  `step_frame_offsets()`/`FRAME_PER_TOKEN` helpers that exist only to serve them.

  It stays off `main` because it monkeypatches core, and the restriction it works
  around may be lifted upstream — in which case most of the patch wants deleting, not
  shipping. Everything on `main` runs against stock ComfyUI plus the documented diff,
  with no runtime patching.

  Nothing else changes. The corrected reference geometry stays documented here, since
  it is true regardless of which branch you are on.

## [0.22.0] - 2026-08-07

### Added
- `MMH3TaskSystemPrompt` emits its `mode`, and `MMH3PromptLint` takes it as
  `mode_override` (appended last, optional). Wire them and the two can never disagree.

  Setting the mode in two places is a silent failure, because the mode selects which
  section set the linter expects. Linting a three-field prompt as `Ref2VA` reports
  four missing sections that are not missing; linting a six-section prompt as a base
  mode reports three. Both read like the LLM ignored its instructions, and the prompt
  is fine.

### Changed
- Lint reports now lead with `mode X (wired|widget)`. A finding list that does not say
  which format it checked against is unreadable precisely when the mode is the mistake.

## [0.21.1] - 2026-08-07

### Fixed
- `MMH3PromptLint` reported an off-screen voiceover failure that could not be traced
  to anything in the prompt. Two bugs in one pattern:

  ```python
  says in an off-screen voiceover.*?</d>(.{0,120})
  ```

  The `.*?` is unbounded, and under `re.S` it leaps across the whole document to
  whatever `</d>` appears next, then judges the 120 characters after **that**. The
  phrase appears verbatim in the format rules as an instruction, so any text carrying
  them plus an unrelated dialogue block anywhere later reported a failure with the
  lips-closed statement sitting untouched beside the phrase. The dialogue must now
  follow the phrase within 40 characters.

  The trailing window was also **consumed**, so `finditer` skipped past anything after
  it: a prompt whose SECOND voiceover was the broken one linted clean. It is a
  lookahead now.

- The voiceover finding quotes the text it matched, like the neighbouring `<d>` rules
  already did. A finding you cannot locate is a finding you cannot act on.

## [0.21.0] - 2026-08-07

### Added
- Positioned keyframe anchors, developed on the **`keyframe-anchors`** branch rather
  than here. `MMH3LatentToKeyframes` pins the previous chunk's tail as a run of
  anchors on the clip's own timeline, and `mmh3tools/patch_layout.py` unlocks interior
  indices by wrapping `PackedLayout.__init__` at runtime.

  Kept off `main` because it depends on a monkeypatch of core, and the restriction it
  works around may be lifted upstream — in which case most of the patch should be
  deleted rather than shipped. Schema and layout geometry are verified against the
  live class; it has not been run against real weights.

### Fixed
- The keyframe/reference geometry is documented correctly now. References are
  **positioned** — the layout advances a cursor per ref block and the target begins
  after them, contiguously — so a carried tail already sits immediately before the
  clip. The cost is *distance*: a 39-frame carry moves the target origin from
  `text_len` 320 to 385. Audio adds nothing to that, since `FRAME_RESCALE` (5/3) and
  `40/24` are the same number and the layout's `max()` is a no-op.

### Changed
- CHANGELOG and `pyproject.toml` restored; they had drifted three versions behind git
  (0.18 through 0.20 never got entries and `pyproject.toml` still said 0.17.0).

## [0.20.0] - 2026-08-07

### Added
- `MMH3ContextWindows` gains a `freenoise` switch (default off, appended last).
  0.15.x hardcoded it off and stubbed `_apply_freenoise` out entirely. FreeNoise
  copies each window's noise forward into the next window's region, permuted, so
  overlapping windows start from related noise rather than independent noise -
  which is what full-denoise windowing was missing. Shuffles VIDEO only, on its own
  temporal dim; the stock multimodal path would have permuted audio's stereo axis.

## [0.19.2] - 2026-08-07

### Changed
- `MMH3TaskSystemPrompt` format rules tightened - the skeleton now shows three shots
  stacked in ONE field and states that three field labels appear in the entire output,
  once each. A local model was emitting the whole field set per shot.

## [0.19.1] - 2026-08-07

### Fixed
- `MMH3PromptLint` missed repeated sections, and two bugs hid it. Section boundaries
  now tolerate leading whitespace (`\n\s*%s\s*:`) - without it an indented prompt, which
  LLMs produce constantly, never matched the stop and every section ran to end of
  document. And callers pass the FULL section list so a repeat of the same label
  terminates it; the last field otherwise swallowed every repeated block after it,
  which is how a phantom mood word turned up in `non_diegetic_music`.
- Sections are now COUNTED, not tested for presence. `re.search` finds the first and
  stops, so a prompt with every field repeated per shot linted clean.

## [0.19.0] - 2026-08-07

### Added
- `MMH3ReferenceFromLatent` gains `ref_images` (Autogrow, max 9) and `ref_image_size`.
  Stills are emitted BEFORE the carry in `ref_items` so `<Picture N>` numbering matches
  the stock node.

## [0.18.0] - 2026-08-06

### Added
- `MMH3ImageToRef` - append a still image to `minimax_refs`, closing the last hole in
  the conditioning matrix: latents could become refs or keyframes and images could
  become keyframes, but nothing put an image into refs by appending. Stock
  `MiniMaxH3ReferenceToVideo` accepts `ref_images` but builds conditioning from
  clip+prompt rather than appending, so it cannot add a still to conditioning that
  already exists - which is what stacking a reference face alongside carried latent
  refs requires.

  Reference blocks carry their own `latent_h`/`latent_w`, unlike keyframes, so this is
  free to resize. Sizing mirrors the stock node exactly (match/max, scale-down only).
  The label reports tokens per sampling step, since reference rows are attended every
  step and, under context windows, every step of every window: 999 at match versus
  5440 at max on a 3000x4000 source.

## [0.17.0] - 2026-08-06

### Added
- `MMH3StreamingEncode` - chunked VAE encode, so long clips at high resolution can
  be encoded at all. **Output is bit-identical to `VAEEncode`** (max|diff| exactly
  0.00e+00, verified at 39 and 124 frames across chunk sizes 17, 85 and 1700).

  `F.pad(..., mode="reflect")` in H3's `CausalConv3d` requires the tensor to fit
  32-bit indexing - under `2**31` elements. A pixel batch is `[1, 3, T, H, W]`, so
  that is a JOINT ceiling on length and resolution:

  | resolution | max frames | duration |
  |---|---|---|
  | 1024x768 | 906 | 37.7s |
  | 1536x1152 | 396 | 16.5s |
  | 2048x1536 | **226** | 9.4s |

  Past it, `VAEEncode` dies with *"input tensor must fit into 32-bit index math"*.
  That is **not** an OOM, so `raise_non_oom()` re-raises it and ComfyUI's automatic
  retry-with-tiled-encoding never fires - a hard stop rather than a slow fallback.
  The ceiling shrinks as an upscale ladder climbs, so a length that sails through
  stage 1 can fail at stage 3.

  **Chunking is exact here** because `encode_temporal` slices into non-overlapping
  17-frame clips and encodes each with no carried state. Clip boundaries are free -
  unlike LTX, whose encoder has a causal receptive field across boundaries and needs
  left context re-encoded and trimmed per chunk.

  **The trap**: the tail padding and `token_drop` are applied once PER CALL, so
  looping `vae.encode()` over chunks silently loses 3 latents per chunk - 39 frames
  give 12 latents whole but `2+2+2 = 6` as three calls. Not an error; just a shorter
  latent that decodes to a shorter, wrong video. The node therefore drives
  `_adaptive_encode` directly and applies the pad and the drop exactly once, then
  reproduces `encode()`'s moments-to-latent step. The single `token_drop` is what
  turns `5j` clips into the `5j+2` grid.

  `frames_per_chunk` snaps to a multiple of 17 and **does not change the result** -
  it is purely a memory/passes dial. Going around `VAE.encode()` means the node loads
  the model itself, budgeting for one chunk rather than the whole clip.

  Scope, stated plainly: this raises the LENGTH ceiling; it does not by itself give
  constant RAM, because the incoming `IMAGE` batch already exists in full before the
  node runs. Constant RAM needs reading frames from disk per chunk, as LTXAVTools'
  streaming encode does.

## [0.16.0] - 2026-08-06

### Added
- `workflows/minimax h3 I2V 2K.json` - the three-stage I2V-to-2K workflow, and the
  first example shipped with the pack. Generate small, then two low-denoise windowed
  upscale passes, with audio decided in stage 1 and carried forward.

  Uses `MMH3UpscaleLadder`, `MMH3ContextWindows` (both upscale samplers),
  `MMH3FrameCalculator`, `MMH3TaskSystemPrompt` and `MMH3PackAV`, plus
  ComfyUI-LlamaOmni for prompt writing, KJNodes, RES4LYF, VideoHelperSuite and
  rgthree.

  Cleaned before shipping: a replacement `CLIPLoader` had come in with
  `type = "stable_diffusion"` instead of `minimax` (a silent functional break, since
  a fresh CLIPLoader defaults to that and does not announce it); two nodes still
  pointed at `ckinpdx/MMH3Tools`, a deleted repo, at a commit unreachable from the
  current history; the LlamaOmni nodes had no `aux_id` so Manager could not install
  them; and three `videopreview` blocks referenced local output files.

## [0.15.4] - 2026-08-06

### Changed
- **Corrected: windowing is FASTER at high resolution, not slower.** Measured -
  stage 3 at 2K ran about a minute faster windowed than whole. 0.15.3's framing
  ("you pay in passes", a time-for-memory trade) reasoned linearly and was wrong.

  For 57 latents at window 17, overlap 7 (stride 10, 5 windows):

  ```
  attention  proportional to N^2   5 x 17^2/57^2  =  0.44x   56% LESS work
  linear     proportional to N     5 x 17/57      =  1.49x   49% MORE
  ```

  Both ratios are resolution-independent; only the mix varies. At ~131k tokens
  attention dominates so heavily that the 0.44x decides it and the overlap tax is
  noise.

  Practical consequence, now in the tooltip: **smaller windows are not faster.**
  Window 12 has the same 0.44x attention ratio - more windows exactly cancels the
  smaller square - while linear cost rises to 2.1x. Shrink `context_length` for
  memory only.

## [0.15.3] - 2026-08-06

### Changed
- Tooltips now say which knob is the VRAM lever. `context_length` sets peak
  activation cost (it scales with the window, not the clip); `context_overlap`
  changes how many windows run, so it trades time and seam quality, not memory.

  Measured at 192 frames (57 latents), tokens per forward:

  | | full clip | window 17 | window 12 | window 7 |
  |---|---|---|---|---|
  | `1536x864` | 73,872 | 22,032 | 15,552 | 9,072 |
  | `2048x1152` | 131,328 | 39,168 | 27,648 | 16,128 |

  An ordinary `1344x768` 8s generation is 57,456 tokens for comparison - so a
  windowed 2K pass has a **smaller sequence per forward than a normal 768p
  generation**, and attention is quadratic on top of that.

  Stated in the tooltip because it is easy to misread: this reduces ACTIVATION
  memory only. The H3 UNet is ~21 GB even pruned, so windowing does not make the
  model loadable on a card that could not already load it. It helps the user who
  can run H3 but cannot hold activations for a long or high-resolution clip.

### Known limitation
- ComfyUI's VRAM estimator does not shrink for windowed H3. `pack_latents` returns
  `[B, 1, N]`, so `_prepare_sampling_wrapper` sees `is_packed` and skips the
  per-window clamp behind an upstream TODO ("latent_shapes cond isn't attached yet
  at this point"). Real peak memory does drop, but ComfyUI budgets as if the clip
  were unwindowed and offloads more of the model than necessary - slower than it
  needs to be, not an OOM. Fixing it upstream would help every packed-latent model.

## [0.15.2] - 2026-08-06

### Fixed
- **Pulsing across the clip** - the same visual signature as joining off-grid
  latents, but a different cause, and one I created.

  H3's latent groups start at `2+5k`. Window stride is `context_length -
  context_overlap`, and 0.15.0 forced length to `5j+2` **and overlap to `5m`**, so:

  ```
  stride = (5j+2) - 5m = 5(j-m) + 2   =  2 (mod 5)   always
  ```

  Every window start advanced 2 in phase against the group grid, cycling
  `0, 2, 4, 1, 3` — **a five-window beat**. Each window presents its first two
  latents to the model as though they were the 5-frame anchor group, so the
  temporal warp differs per phase and repeats. That is the pulse.

  Fixed by snapping overlap to **`5m+2`** (2, 7, 12, 17...) rather than a multiple
  of 5, which makes the stride a multiple of 5 and puts every window at the same
  phase:

  ```
  stride = (5j+2) - (5m+2) = 5(j-m)   =  0 (mod 5)
  ```

  Default overlap is now 7 rather than 5. Whatever warp remains is then identical
  in every window, so there is no periodic change to see. Note this is the exact
  opposite of what the 0.15.0 tooltip told you to do.

  Test 11 asserts stride divisibility and phase uniformity across three window
  sizes, and asserts that the old `overlap=5` config genuinely does cycle — so the
  bug cannot come back unnoticed.

## [0.15.1] - 2026-08-06

### Fixed
- **Crash on the first sampling step**: `The size of tensor a (2) must match the
  size of tensor b (93) at non-singleton dimension 2`, raised from
  `combine_context_window_results`.

  0.15.0 fixed the per-modality dim in `prepare_window()` and slicing, but two more
  places in `IndexListContextHandler` index a modality tensor with the handler's
  `self.dim`, and both hit audio on its **stereo axis**:

  - `combine_context_window_results()` builds the fuse weights with
    `x_in.shape[self.dim]` and `match_weights_to_dim(..., self.dim)`, so a 93-long
    audio weight vector was sized onto dim 2 (extent 2). That is the crash.
  - `execute()` allocates `counts` via `get_shape_for_dim(m, self.dim)` and `biases`
    as `[0.0] * m.shape[self.dim]`, giving audio a counts tensor of extent 2 instead
    of `T40` and a biases list of length 2. This would have failed immediately after.

  Both are now overridden in `MMH3ContextHandler`, using the **window's own** dim for
  fusing and a per-modality dim for allocation. `execute()` is copied from upstream
  rather than wrapped, because the allocation is inline; the two changed lines are
  marked, and if upstream refactors it breaks loudly here instead of quietly
  windowing the wrong axis.

  Tests 9 and 10 cover exactly this: accumulator extents per modality, and the fuse
  step running on both without raising.

## [0.15.0] - 2026-08-06

### Added
- `MMH3ContextWindows` - sliding-window sampling over a long AV latent, **with no
  core patching**. `MMH3ContextHandler` and `MMH3WindowingState` subclass ComfyUI's
  own windowing.

  **Intended for low-denoise upscale passes only.** At low denoise every window
  starts from the same upscaled base, so coherence comes from the input rather than
  from attention spanning the clip; at full denoise each window invents its own
  content and they disagree. Attach it on stages 2 and 3 of an upscale ladder, never
  on the pass that decides structure.

  Two things stopped stock ComfyUI doing this:
  - `map_context_window_to_modalities` has **zero implementations tree-wide** - the
    name appears twice, at the call site and in its own error message - so the
    multimodal path raises `NotImplementedError` for every model. Overriding
    `prepare_window()` means the hook is never called at all.
  - `WindowingState` uses ONE `dim` for every modality. H3's video is dim 2 and
    audio is dim 3, so the stock path would window audio `[B,32,2,T40]` on its
    **stereo axis** - size 2, not `T40`. No crash; just a ratio of `2/T` and
    nonsense indices.

  Neither needs a core edit, because the handler is only an object in
  `model.model_options["context_handler"]`. That is worth more than convenience: it
  survives `git pull`, and when upstream refactors it fails loudly with an
  `AttributeError` instead of silently doing the wrong thing, which is what a stale
  diff does.

  Audio boundaries are converted independently and subtracted rather than converting
  a window length, because `audio_t = round(frames/24*40)` is not additive - the
  same correction `MMH3ConcatAV` needed. The mapping is exact at every on-grid
  boundary.

  Pinned by the node: windows snap DOWN to `5j+2` latents and overlap to a multiple
  of 5, since the model only ever saw `5j+2` clip lengths; `causal_window_fix` off,
  because it prepends an anchor frame that would push every window to `5j+3`;
  `freenoise` off, since it exists to improve window blending and a low-denoise pass
  has very little noise to shuffle; and looped/batched schedules are not offered,
  because they can emit wrapping windows that the audio mapping cannot express as a
  time span.

- `tests/test_windows.py` - 27 assertions, including a direct check that the stock
  single-`dim` path would have hit the stereo axis, and that windows tile the whole
  audio track with no gap.

- `docs/context-windows.md` - the full read of `comfy/context_windows.py`, what a
  core-side fix would touch, and why the node approach is preferable.

## [0.14.0] - 2026-08-06

### Added
- `MMH3UpscaleLadder` - three exact-aspect, on-grid stages for a progressive
  generate-small-then-denoise-up pipeline. Separate node; `MMH3DimensionCalculator`
  is untouched.

  **Why integer multiples of a unit instead of snapping.** A ratio lands exactly on
  the 32px grid only at integer multiples of its minimal on-grid unit: 16:9 needs
  `w/h = 16/9` with both `/32`, which is `w = 512k, h = 288k`. Working in `k` rather
  than pixels means no stage is ever snapped, so the aspect cannot drift between
  stages - which matters here, because a low-denoise pass onto a slightly different
  aspect resamples the whole frame instead of just adding detail. Limiting the ratio
  set is what makes this possible.

  Three constraints, all measured rather than chosen:
  - every stage exact-aspect and on the 32 grid
  - no step above 2x - a low-denoise pass cannot invent more than that
  - stage 1 at or above `min_megapixels` (default **0.4**, measured): below it the
    first pass stops being upscalable and stage 2 sharpens mush instead of repairing
    structure

  Stage 2 is placed at the geometric mean of stages 1 and 3, clamped to the window
  both step limits allow, so the work spreads evenly across the two passes.

  | ratio | stage 1 | stage 2 | stage 3 | steps |
  |---|---|---|---|---|
  | 16:9 | 1024x576 | 1536x864 | 2048x1152 | 1.50x, 1.33x |
  | 4:3 | 768x576 | 1280x960 | 2048x1536 | 1.67x, 1.60x |
  | 3:2 | 864x576 | 1344x896 | 2016x1344 | 1.56x, 1.50x |
  | 1:1 | 640x640 | 1152x1152 | 2048x2048 | 1.80x, 1.78x |
  | 21:9 | 1120x480 | 1568x672 | 2016x864 | 1.40x, 1.29x |

  Degenerate configurations are reported rather than silently producing a duplicate
  stage, and the two causes are distinguished because they need opposite fixes: a
  total upscale too LARGE for three 2x steps says to raise `min_megapixels` or lower
  the target, while one too SMALL says no on-grid stage fits in between and it is
  really a 2-stage ladder.

### Note
- **2K generation is not possible with the open weights.** H3-Base is 768p; 2K comes
  from H3-Regenerate-2K, which feeds the 768p result plus the original context back
  through H3, and which MiniMax has not open-sourced ("we will release it once it is
  ready"). This ladder is for a local progressive-upscale pipeline, not for asking
  the base model to generate at 2K directly.

## [0.13.0] - 2026-08-06

### Added
- `MMH3PromptLint` - checks a finished, LLM-written prompt against the H3 format
  rules before anything is sampled. Passes the prompt through unchanged, so it sits
  inline between the LLM node and the conditioning node.

  `MMH3TaskSystemPrompt` validates the SETTINGS you gave it; this validates the TEXT
  that came back, which is where the interesting failures are - a local model follows
  most of a long rule list and quietly drops the rest.

  The argument is economic. A chunk is minutes of sampling, and most format errors do
  not crash, they render something subtly wrong: a cut timed past the end of the clip
  simply never happens, a quoted line of dialogue asks for a sign instead of speech, a
  voiceover missing its lips-closed clause gets mouthed. Each costs a full generation
  to find by watching and a second to find here.

  Checks: required sections for the mode's format; body opens with `[Shot 1]`;
  `[Shot 1]` carries no timestamp; timestamps strictly increasing and unique; shot
  numbers 1..N in order; no cut at or past the duration; `<d>` tags balanced, each
  carrying a `[Language]` tag and containing no speaker ID or delivery verb; dialogue
  never in double quotes; every off-screen voiceover followed by the lips-closed
  statement; no dialogue in `overall_soundscape`; no mood words in
  `non_diegetic_music`; every `<Picture/Video/Audio/Subject N>` used in the body
  defined in `subject_definitions`; no `(Sx)` in `retention_analysis`; and a
  `[task type]` prefix on the summary.

  `on_problem` selects `warn` (log and pass through) or `error` (stop the queue).

  Derived from the `lint()` in a standalone H3 film script, generalised to both
  prompt formats - which immediately caught a bug of its own: the shot body is the
  FIRST field in the three-field format but the FOURTH in the six-section one, so
  taking `sections[0]` linted `subject_definitions` and silently passed every shot
  and timestamp check.

- `tests/test_lint.py` - 26 assertions over a clean prompt and a deliberately broken
  one carrying every fault at once.

## [0.12.1] - 2026-08-06

### Fixed
- The `speech` and `sung lyrics` blocks assumed transcription would happen
  implicitly. Buried in a system prompt whose stated job is "convert a rough video
  idea into a structured prompt", a local model treats it as a detail and composes
  from the text idea alone. The block now opens by stating that an audio clip is
  attached and must be listened to first, and each kind makes the transcription an
  ordered step to finish BEFORE composing. Sung lyrics adds that the effort belongs
  there rather than in the prose, and that unclear passages should be omitted rather
  than filled with plausible substitutes - invented words get animated onto the mouth.
- The no-dialogue warning claimed the model "cannot hear" the track. It can:
  LlamaOmni sends `input_audio` and omni models transcribe. The real risk is asking
  one call to transcribe AND compose, so the warning now points at a dedicated ASR
  pass instead.

## [0.12.0] - 2026-08-06

### Added
- `MMH3TaskSystemPrompt` gains `masked_audio` (combo: `none` / `background music` /
  `speech` / `sung lyrics`, appended last), for the **undocumented** technique of
  masking a supplied audio latent so the track survives verbatim into the output.
  This is the base-mode equivalent of Ref2VA's `[audio reuse]` + `fully_copy`, and
  MiniMax's guides do not cover it.

  **The point is that it inverts what the audio fields mean.** In the three-field
  format, `overall_soundscape` and `non_diegetic_music` normally REQUEST sound to be
  generated. With a masked track they DESCRIBE sound that already exists, and their
  only job is to tell the model what it is about to hear so the picture matches.
  Written the usual way they ask for audio that cannot be added, and the video ends
  up expecting events the track never delivers.

  Per-kind rules, because the model has to know what is in the track to generate a
  matching picture:
  - **background music** - goes in `non_diegetic_music`; diegetic instead if a
    visible source produces it. **Nobody speaks**: no `<d>`, no `(Sx)`, mouths closed
    or occupied, since a character shown mid-speech with no voice reads as broken.
    Cut in sympathy with the music but invent no hits or drops the track lacks.
  - **speech** - transcribe into `<d>` at the moment each line is heard so the lips
    match; `(Sx)` by vocal-event order; voice description must match the track, not
    an invented one; explicit mouth movement for the whole line.
  - **sung lyrics** - as speech but *sings*, lyrics verbatim in their original
    language, and describe singing physically (sustained vowels, held notes, breath),
    because sung mouth shapes differ from spoken ones.

  When `masked_audio` is `speech` or `sung lyrics`, the supplied `dialogue` is treated
  as a **transcript of a fixed track**, so the word ceiling added in 0.11.0 is
  replaced by "the track's own timing governs - do not add, cut or re-time lines to
  fit a word estimate". A ceiling would otherwise invite trimming a transcript.

  Three new warnings for configurations that fail silently:
  - `masked_audio` on **Ref2VA**, where `[audio reuse]` + `fully_copy` is the trained
    path and using both describes one track two ways.
  - `speech` / `sung lyrics` with **no dialogue supplied** - the model has to guess
    words it cannot hear, and the lips will not match.
  - `background music` **with** dialogue - the track has no voice to carry it, so any
    `<d>` line is mouthed over silence.

## [0.11.0] - 2026-08-06

### Added
- `MMH3TaskSystemPrompt` gains a `dialogue` input (multiline, appended last), for
  spoken lines that must be used **verbatim**. The rule existed only in
  `docs/context-ir-system-prompt.md` (point 7) and had never made it into the node
  that the pipeline actually calls.

  When set, the system prompt gains a `## Supplied dialogue` block: reproduce each
  line exactly once in order, write no line that is not listed, keep every line and
  cut surrounding action if they do not fit, one `<d>` block each with only the
  language tag and the words inside, never double quotes, punctuation standardised
  to `, . ? !`. The lines themselves are embedded under `DIALOGUE:`.

### Fixed
- **The word budget actively invited padding.** The Constraints block emitted
  "roughly N words of dialogue TOTAL" unconditionally. Harmless when the model
  writes its own lines; destructive when the lines are the user's, because a small
  model handed a word target will pad up to it - and the invented lines arrive
  correctly formatted, in valid `<d>` tags, with plausible `(Sx)` IDs, which makes
  them very easy to miss.

  With `dialogue` set the wording becomes a **ceiling**, states the supplied word
  and line count, and says explicitly not to add lines to reach it. Without it the
  original wording is unchanged.

- The Output section's "invent concrete detail consistent with the intent" licensed
  exactly the padding the new block forbids. With dialogue supplied it narrows to
  "concrete action, camera and ambience detail" and adds "Never invent dialogue."

- The node now warns in its `report` when the supplied dialogue cannot fit the
  duration (e.g. *"40 words but only ~7 fit in 3.750s"*), rather than leaving the
  model to silently drop lines.

## [0.10.0] - 2026-08-05

### Added
- `MMH3ReferenceMultiPrompt` + `MMH3CondSelect` - `MiniMaxH3ReferenceToVideo` with
  N prompts instead of one, for a text-driven sequence where every chunk shares
  the same references and differs only in its prompt.

  **The point is the model swap, not the encode.** Stock does the reference
  resize, `vae.encode`, `audio_vae.encode` and the text encode all inside one
  `execute()`, so N chunks means N copies of the reference work - and N swap
  cycles, because Qwen3-VL-32B and a 33B DiT cannot be resident together in 32GB.
  ComfyUI resolves outputs depth-first, so a naive N-chunk graph runs
  `load TE -> cond -> evict -> load DiT -> sample -> evict -> load TE -> ...`.
  Doing every encode in ONE node execution collapses that to a single swap for
  the whole sequence.

  Outputs a custom `MMH3_COND_SET` type rather than a `CONDITIONING` holding N
  entries, because a multi-entry CONDITIONING means "combine all of these" - a
  mis-wire straight into a sampler would silently merge every prompt into one and
  render plausible-looking garbage. A distinct type makes that unrepresentable.
  Outputs cannot be dynamic (`Autogrow` is `ComfyTypeI`, inputs only), hence the
  select node.

  Prompts are N separate string inputs rather than one delimited field, so a
  local LLM can drive each one independently.

  **Per-prompt memoization**, keyed on `(prompt, reference fingerprint)`. ComfyUI
  caches per node execution, so without it a one-word edit to a single prompt
  would re-run every prompt's Qwen pass. The fingerprint hashes the raw inputs
  *and* the encoded blocks: hashing only the encoded blocks would make cache
  validity depend on the VAE mapping different references to different latents,
  and that is not an assumption worth making when the failure mode is the wrong
  reference used silently in every chunk.

  Still paid per prompt: `clip.tokenize` re-presents the references to Qwen and
  the vision tower plus 50 layers run again. That is inherent - references are
  emitted BEFORE the prompt text, and although `comfy/text_encoders/llama.py`
  threads `past_key_values` through every layer, the CLIP API exposes no way to
  hand it a cached prefix. Negligible for still images; the thing to avoid for
  video references.

- `tests/test_multiprompt.py` - 17 assertions with stubbed clip/vae covering
  encode counts, cache hits and misses, fingerprint invalidation, ref-encode
  reuse, select bounds, and the empty-prompt error.

### Note
- `_build_refs()` **duplicates** the reference-building half of
  `comfy_extras/nodes_minimax_h3.py`, because upstream runs it inline in the same
  `execute()` as the text encode and offers no seam to call. Re-sync it if that
  file changes its sizing, its block keys, or - most fragile - the emission
  ORDER, since the tokenizer assigns `<Picture i>` / `<Audio j>` / `<Video k>` by
  counting items in the order given. A reference video's soundtrack must be
  appended BEFORE the video itself or every later label shifts and the prompt's
  tags stop matching their assets.

## [0.9.0] - 2026-08-05

### Added
- `MMH3ConcatAV` gains `carry_masks` (BOOLEAN, default `false`), appended last so
  saved workflows keep their current behaviour byte for byte.

  Off, the node drops input `noise_mask`s exactly as before (now with a log line
  saying so). On, it concatenates them on the same axes as the latents they
  describe - video dim 2, audio dim 3 - filling an absent side with ones
  ("denoise everything there"), matching the convention `MMH3PackAV` already uses.
  If neither input carries a mask, none is invented.

  The old comment claimed "a per-frame mask cannot span the join". That was never
  true: masks live on the same axes as the latents, so joining them is the same
  `cat` with the same dims. The reason it stays **off by default** is semantic,
  not structural - an inherited mask described a generation that has *already
  happened*, so re-sampling the join would pin two finished seams and regenerate
  everything between them. Turn it on when the join is deliberately the INPUT to a
  bridging pass (MiniMax's `video editing` task type).

  When trimming, the carried mask takes the same **computed** cut as the latent
  (`k` and `drop_audio`, never the raw widget value), and a mask whose length ends
  up disagreeing with its latent is warned about rather than left for
  `prepare_mask` to silently resize.

- `tests/test_concat_av.py` - 27 assertions covering mask carry, the trim
  families, and a `MMH3SeedOverlap` -> `MMH3ConcatAV` round-trip. Run it with
  ComfyUI's venv from the ComfyUI root:
  `venv/Scripts/python.exe custom_nodes/ComfyUI-MMH3Tools/tests/test_concat_av.py`

### Changed
- **`MMH3ConcatAV`'s `trim_b_latents` no longer snaps.** It is now honoured as
  given, clamped only so B keeps its minimum 2 latents.

  Previously it went through `snap_latents()`, which snaps to the `5j+2`
  clip-length grid, so wiring `MMH3SeedOverlap`'s `overlap_latents = 5` in trimmed
  **2**, and 12 of the 17 overlap frames stayed duplicated at the join.

  The snap was not simply a bug, which is worth recording: with `A = 5a+2` and
  `B = 5b+2`, the two things you might want are mutually exclusive.

  | trim | effect |
  |---|---|
  | `5m` | removes a SeedOverlap **exactly**; B's remainder stays on grid; the **total** is `5(a+b)+4-k`, off grid |
  | `5m+2` | total lands **on grid**; ~7 frames of overlap stay duplicated |

  `k` cannot be `0` and `2 (mod 5)` at once, so no snap is right for every use -
  the old one silently picked the second family. The node now honours the value,
  and logs which of the two properties the chosen `k` actually gets. If you need
  both, that is what `MMH3JoinAV` is for: it cuts in pixel space, per frame.

  The audio drop is also corrected. It was `frames_to_audio_t(dropped_frames)`,
  but `audio_t = round(frames / 24 * 40)` is **not additive**, so it now takes the
  difference of the two totals - the same construction `MMH3SeedOverlap` uses to
  size the overlap, so the two round-trip exactly.

### Removed
- The `/mmh3-dim-calc/resolutions` aiohttp route in `nodes_util.py`, along with
  its `server` / `aiohttp` imports. Dead since the dimension calculator moved to
  computing its option lists client-side. It also registered at import time, which
  made the package impossible to import outside a running ComfyUI server.

## [0.7.0] - 2026-08-05

### Added
- `MMH3ImageKeyframe` - inject a still image as a keyframe anchor. Takes
  `image` + `vae` and does the resize/encode internally, because keyframe rows
  share the TARGET spatial grid and cannot be downscaled: a still encoded at any
  other resolution fails deep in the model with a broadcast error. That is the
  likeliest cause of "I VAE-encoded a still and it didn't work".

  It appends to `minimax_keyframes` rather than building conditioning, so it
  composes with a ref2va build - `MiniMaxH3ReferenceToVideo` has no keyframe
  inputs at all, so this is the only way to give the reference checkpoint a
  frame anchor.

  `frame_index` accepts 0, -1, or any interior index. Interior anchors are
  documented as valid by MiniMax but stock `PackedLayout` raises
  `only first/last keyframe anchors are supported`; the node logs a warning
  rather than refusing, so it works once that check is patched.

  `resize=auto` copies the stock node's per-position behaviour: stretch for a
  first-frame anchor (geometry anchor), aspect-preserving centre crop otherwise
  (follower).

  It does NOT register the image with the tokenizer, so `<Picture N>` will not
  resolve in prompt text - same limitation as the rest of the latent-domain
  nodes, and for the same reason.

- `MMH3AssetPlan` and `MMH3TaskSystemPrompt` (`nodes_prompt.py`) - build a
  Context-IR system prompt for your own LLM node from a task type (or
  combination) plus a plan of the assets in play. Emits only the rule blocks
  relevant to the selected tasks instead of the whole spec.

- `MMH3SeedOverlap` restored. It **prepends** the overlap (a multiple of 5
  latents, 17 frames each) so the target keeps its full requested duration and
  the overlap is cleanly cut off afterwards.

- `docs/core-patches.md` + `docs/core-patches.diff` - the three ComfyUI core
  files that must be patched for a keyframe and a reference to coexist, and for
  overlap strength to be continuous rather than boolean. Taken against
  `v0.30.0-1-g14b05228`. These are lost on every `git pull` in ComfyUI, so they
  are now checked in and reappliable with `git apply`.

### Changed
- README: the "noise_mask does not work on H3" section is **removed**. The claim
  was false - `samplers.py` packs latents before sampling and explicitly handles
  `denoise_mask.is_nested`. It is replaced by a correction note and a section on
  why joins happen in pixel space (which is true, and was the real reason).
- README: "whether ref2va responds to keyframe rows is unverified" is resolved -
  it does, once core patches 1 and 2 are applied.
- Per-row masking merged from drozbay's `minimax-h3-per-row-masking` branch, so
  overlap strength is continuous rather than on/off.
- `MMH3PackAV` now carries an input `noise_mask` through instead of dropping it.

### Fixed
- `MMH3DimensionCalculator` failing with a bare "Invalid input" and nothing in
  the console. The JS swaps the resolution list per ratio/orientation, but the
  server validates combo values against the options declared in Python, so
  anything outside the default list was rejected. Declared options are now the
  full union (9 ratios, 103 resolutions) with `validate_inputs` returning True.
- JS/Python rounding mismatch in the same node: Python's `round()` is banker's,
  `Math.round()` is half-up, so 3:2 diverged at exactly 16.5. The JS now uses a
  `roundHalfEven` helper; all 114 generated options match Python.

## [0.6.0] - 2026-08-03

### Added
- `MMH3FindDivergence` gains a `compare` input (`structure` / `raw`, default
  `structure`), appended last so saved nodes pick up the default without rewiring.

  `structure` zero-means and unit-contrasts each frame before comparison. Plain MAE
  cannot distinguish "different content" from "same content, half a stop brighter",
  so an exposure or colour shift between the source and the generated chunk puts a
  floor under every comparison and flattens the curve — which reads exactly like
  "no reproduction found". Measured on a genuine 30-frame reproduction rendered 12%
  brighter with a lifted black level: raw MAE reports error 0.110 at 2.5x separation
  (rejected at the default 0.05 threshold), structure reports 0.0100 at 79x.

  This matters here because the source has been through a VAE round-trip the
  generated chunk has not, so level drift between them is expected.

  Threshold stays around 0.05: a good structure-mode match is ~0.01, mismatches ~0.8.

## [0.5.0] - 2026-08-03

### Added
- `docs/context-ir-system-prompt.md` - a system prompt that replaces MiniMax's
  hosted `H3-Context-IR` (`/v2/h3_context_ir`), which is not open-sourced. H3
  expects a STRUCTURED prompt, not prose; Context-IR is what produces it, so
  running locally you must author that structure yourself. Covers both formats
  (three-field base modes and six-section Ref2VA), mode selection, label rules,
  task types, retention markers, camera/speaker/dialogue syntax, the discrete
  achievable durations, and the chained-work defaults. Portable — usable with a
  local LLM, an enhancer node, or anything else. Use a VISION model when
  references are involved, since `subject_definitions` describes the assets.

### Removed
- **`MMH3SeedOverlap` — removed.** ComfyUI cannot apply a denoise mask to H3's
  NestedTensor AV latents, so the node could never have worked:
  - `KSamplerX0Inpaint.__call__` computes `1. - denoise_mask`, and `NestedTensor`
    defines no `__rsub__` (nor `__torch_function__`), so a nested mask raises
    TypeError.
  - A plain-tensor mask fails differently: `apply_operation` applies it to BOTH
    sub-tensors, so one mask would have to broadcast against video
    `[B,24,T,h,w]` and audio `[B,32,2,T40]` at once.
  - `torch.count_nonzero(latent_image)` in `inner_sample` is not nested-safe.

  Masked / inpaint-style workflows are therefore unavailable for H3 in ComfyUI
  v0.30.0. Continuity comes from the REFERENCE path (`update=False`, the trained
  mechanism), and joins are trimmed after decode.

### Added
- `MMH3FindDivergence` - measures how many frames a continuation reproduces from
  its source, so the join can be trimmed at FRAME granularity. Latent trims are
  restricted to the 5j+2 grid, i.e. 17-frame steps, which is far too coarse for a
  boundary the model does not place on a grid.

  Scores each candidate run length K by the contiguous alignment
  `continuation[i] ~ source[-K+i]`, anchored at the source's last frame. Per-frame
  nearest-match was tried first and does NOT work: in visually repetitive footage
  every new frame also matches something, so divergence is never detected. The
  contiguous form gives ~10x error separation at the true K, and reports a
  best/median separation ratio so a flat (untrustworthy) curve is visible.

## [0.4.0] - 2026-08-03

### Added
- `MMH3PackAV` - zips a video latent and an audio latent into one H3 AV latent.
  Encoding real footage produces two SEPARATE plain latents (`VAEEncode` with the
  video VAE, `VAEEncodeAudio` with the audio VAE) and nothing paired them. This is
  a MODALITY join; `MMH3ConcatAV` is a TIME join. Audio length is reconciled to
  `round(frames / 24 * 40)` by padding with silence or trimming, since the two
  streams run on independent clocks and encoders will not agree exactly. Audio is
  optional — omit it to pair with silence.
- `MMH3SeedOverlap` now also outputs **`overlap_latents`**, wiring straight into
  `MMH3ConcatAV.trim_b_latents` so the overlap is not duplicated at the join.
  The grid arithmetic is closed under this: `(5j+2)+(5k+2)-(5m+2) = 5(j+k-m)+2`,
  so chains stay on-grid indefinitely.
- `MMH3LatentToRef` and `MMH3ReferenceFromLatent` now also output
  **`carried_latents`** — the actual count after snapping and clamping, which is
  what grid math and trims need. `carried_frames` was not enough.

### Changed
- `unpack_av()` takes `name` and `allow_video_only`. Errors now name the failing
  input rather than saying "a latent" was wrong, and a plain 5D video latent is
  accepted where audio is genuinely optional: `MMH3SeedOverlap.source` (seeds video,
  logs that it skipped audio) and `MMH3LatentKeyframe` (only ever reads one frame).
  This is the `VAEEncode`-real-footage path — the video VAE knows nothing about
  audio, so it returns a plain, audio-less latent.
- All new outputs are appended, so existing links do not shift.

## [0.3.0] - 2026-08-03

Calculators now follow the LTXAVTools convention: concise typed outputs plus a
short `label` string instead of a verbose info block, and a flat `MMH3Tools`
category on every node.

### Added
- `MMH3FrameCalculator` - **seconds in**. Outputs `frame_count`,
  `latent_frames`, `audio_latent_frames`, `actual_seconds`, mirroring
  `LTXFrameCalculator` plus the audio count H3 needs. `rounding` is
  nearest / up / down.

  Because frames must be 17j+5 at 24fps, achievable durations are discrete.
  Solving `24s = 5 (mod 17)` gives `s = 8 (mod 17)`, so **8.000s (192 frames) is
  the only whole-second duration in the entire 4-15s supported range**. 5s really
  means 5.167s, 12s means 12.250s.
- `MMH3DimensionCalculator` - outputs `width`, `height`, `width_ref`,
  `height_ref`, `label`, mirroring `LTXDimensionCalculator`. Where LTX emitted a
  fixed `width_half`/`height_half` pair for its two-stage pipeline, H3 has no
  second stage, so the secondary pair is the REFERENCE size driven by
  `downscale_factor` and snapped to what the patch grid supports.
- (superseded) `MMH3DimCalc` - snaps width/height to the 32px grid, reports latent dims and
  tokens per latent frame, and snaps a requested reference downscale factor to
  the nearest value the patch grid actually supports. Outputs both generation and
  reference geometry plus the full list of valid factors.
- `common.supported_downscale_factors()` / `common.snap_downscale()`.

### Fixed
- **`ref_downscale` could distort reference aspect ratio.** Latent dims must stay
  EVEN for the 2x2 patch, so a factor is only valid when latent/f is an even
  integer on both axes — the divisors of `gcd(latent_h//2, latent_w//2)`. For the
  1344x768 canvas that is `[1, 2, 3, 6]`; **4 is not valid** (84/4 = 21, odd).
  The old code forced evenness by subtracting 1 from the odd axis, silently
  changing the reference's aspect ratio. `downscale_video_latent()` now snaps the
  factor instead and returns the factor actually used; `MMH3LatentToRef` logs when
  the request was adjusted.

### Changed
- All node categories flattened from `MMH3Tools/{conditioning,latent,util}` to a
  flat `MMH3Tools`, matching the LTXAVTools convention.
- `downscale_video_latent()` now returns `(tensor, latent_h, latent_w, factor_used)`
  — a 4-tuple, was 3. Internal helper; no node inputs or outputs changed.

### Removed
- `MMH3GridSnap` - superseded by `MMH3FrameCalculator`, which takes seconds rather
  than frames. Nothing had been built on it.

## [0.2.0] - 2026-08-03

### Added
- `MMH3ReferenceFromLatent` - full ref2va conditioning builder fed by a latent
  instead of pixels. Unlike `MMH3LatentToRef` it owns the tokenizer call, so the
  carried chunk registers as a real `<Video 1>` and prompts can use the
  `[video continuation]` task type and reference it by label.

  The DiT still receives pristine latents; the only decode is a 2fps subsample
  handed to Qwen3-VL (3-4 frames for a ~1.6s carry), so no generation loss enters
  the sampling path. `register_with_tokenizer` can disable it entirely.
- `common.empty_av_latent()` and `common.frames_to_qwen_items()`.

### Why
H3 expects a structured six-section prompt whose `subject_definitions`,
`summary` and `retention_analysis` sections refer to assets by `<Video N>` /
`<Audio N>` label. Without tokenizer registration those labels dangle, which
silently degrades output. See `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` in the
MiniMaxAI/MiniMax-H3 repo.

Note: `frames_to_qwen_items()` derives timestamps from real frame positions
rather than from the sample index as the stock node does. Behaviour is identical
at 2fps; it just stays correct if the step ever changes.

## [0.1.0] - 2026-08-03

Initial release. Targets MiniMax H3 as shipped in ComfyUI v0.30.0.

Supersedes the throwaway `ComfyUI-MiniMaxH3Loop` prototype, which was removed.
Node IDs changed from `MiniMaxH3*` to `MMH3*`; no released workflows referenced
the old IDs.

### Added

Conditioning
- `MMH3LatentToRef` - builds a `minimax_refs` block directly from an H3 AV latent,
  skipping the pixel/VAE roundtrip the stock reference node performs. Snaps the
  carry to the 5j+2 grid, optionally carries the matching audio tail as
  `kind="video_audio"`, and supports 2x/4x spatial downscaling to cut per-step
  reference token cost.
- `MMH3LatentKeyframe` - injects a `minimax_keyframes` anchor from a single latent
  frame. `PackedLayout` accepts keyframes and refs together, so this stacks with
  `MMH3LatentToRef`.

Latent
- `MMH3SeedOverlap` - seeds the head of a target latent with a previous chunk's
  tail and emits a matching nested `noise_mask`, giving LTXAV-style overlap
  strength control on a model whose reference path is never denoised. Video and
  audio are masked independently on their own temporal axes, with an optional
  linear feather back to full denoise.
- `MMH3ConcatAV` - joins two AV latents using the correct per-sub-tensor temporal
  axes (video dim 2, audio dim 3), with optional head trim on the second latent
  to drop a seeded overlap region.

Util
- `MMH3LatentInfo` - shapes, implied frame count, audio-length mismatch check,
  grid alignment, noise_mask presence.
- `MMH3GridSnap` - snap frames to the 17j+5 grid and derive latent counts, with a
  warning outside the 124-362 trained range.

### Notes
- Carried references are not registered with the tokenizer, so Qwen3-VL does not
  see them and `<Video k>` prompt tags must not be used for a carried chunk. The
  DiT still receives the latents; only the semantic path is skipped.
- `MMH3LatentKeyframe` requires the source latent to match the target generation's
  spatial dimensions, since keyframe rows share the target grid.
- Whether the `ref2va` checkpoint responds to `cond` (keyframe) rows is unverified.
- Latent-space downscaling in `MMH3LatentToRef` uses bilinear interpolation and is
  approximate.
