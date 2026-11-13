# Node reference

Every node in the pack, with the reasoning behind it. The one-line index is in the
[README](../README.md#nodes).

In the Add Node menu these are filed under `MMH3Tools/…`, following the same layout
as LTXAVTools: `sampling`, `calculators`, `prompt`, `conditioning`, `reference`,
`latent`, `audio`, `utils`, with the two plain calculators at the root. The headings
below group by what a node is *for*, which is close but not identical — the menu path
for any node is in its tooltip.

### Conditioning
- **MiniMax H3 Latent to Reference** — carry a chunk's tail forward as a
  `minimax_refs` block, no VAE roundtrip. `ref_downscale` is the cost lever:
  reference tokens are attended at *every* step, so 2× cuts their cost ~4×.
- **MMH3 Regenerate-2K Reference** — the second pass of a 768p → 2K run, with the
  reference **sliced per window**. A cond_set is already per chunk and the sampler
  passes `minimax_refs` straight through, so a reference attached to cond *i* reaches
  chunk *i* and nothing else — the slicing is a build-time concern and the sampler
  needs no changes. That matters because reference tokens ride every sampling step:
  handing the whole clip to every chunk multiplies that by the chunk count, and on a
  12-window clip slicing measured ~9.9× less reference attention per chunk.

  Feed it stage 1's own `cond_set` and each window keeps **its own** prompt while
  gaining its own reference; a single `conditioning` replicates one to all of them.
  Latent-only, like Latent to Reference — the reference never reaches the text
  encoder, so nothing is decoded and the CLIP is never touched in the 2K pass. That
  also happens to be the right semantics: MiniMax's `base_video` role carries no
  prompt label either, because the prompt is the *original* one and never mentions
  the 768p.

  It **conditions** the pass without seeding it — see
  [Refine vs regenerate](#refine-vs-regenerate). Mechanics, dimensions, audio pinning
  and the open divergence past one chunk are in
  [`regenerate-2k.md`](regenerate-2k.md).
- **MiniMax H3 Image to Reference** — append a still to `minimax_refs`. Fills the
  last hole in the matrix: latents could become refs or keyframes and images could
  become keyframes, but nothing put an image into refs *by appending*. Stock
  `MiniMaxH3ReferenceToVideo` takes `ref_images` but BUILDS conditioning from
  clip+prompt, so it can't add a still alongside carried latent refs.

  Unlike keyframes, reference blocks carry their own `latent_h`/`latent_w`, so this
  is free to resize. `match` scales to the generation's pixel area; `max` uses a
  2048px short edge for best identity — on a 3000×4000 source that's 5440 tokens per
  step against 999, paid at every step of every window.

- **MiniMax H3 Latent Keyframe** — first/last frame anchor from a latent frame.
  Shares the *target* spatial grid, so the source must match generation
  dimensions exactly.
- **MiniMax H3 Image Keyframe** — the same anchor from a **still image**.
  Resizes and encodes internally, precisely because keyframe rows cannot be
  downscaled; a still encoded at the wrong size fails deep in the model with an
  unhelpful broadcast error. Both keyframe nodes *append*, filling a gap in
  `MiniMaxH3ReferenceToVideo`, which has no keyframe inputs of its own.

  Keyframes and references coexist: core concatenates them.

  `frame_index` accepts `0` or `-1` only on these two nodes. For anchors inside the
  clip, use the **Looping Sampler**'s `keyframe_indices`, or core's **Add Guide for
  MiniMax H3**.

### Sequences
- **MiniMax H3 Reference (Multi-Prompt)** + **MMH3 Cond Select** — the stock
  reference node with N prompts. For a text-driven sequence with locked identity,
  every chunk shares one reference set and differs only in its prompt.

  The win is the **model swap**, not the encode. Qwen3-VL-32B and a 33B DiT can't
  be resident together in 32GB, and ComfyUI resolves outputs depth-first, so N
  chunks in a naive graph run `load TE → cond → evict → load DiT → sample → evict
  → …` N times. One node execution collapses that to a single swap for the whole
  sequence, and the references are resized and encoded once instead of N times.

  Per-prompt memoization means editing one prompt re-encodes only that prompt.
  Swapping a reference invalidates all of them.

  **`ref_images` takes a batch OR a list, and the difference is not cosmetic.** A batch
  is one tensor and a tensor cannot be ragged, so every batching node — core's
  `ImageBatch`, KJNodes' `ImageBatchMulti` — resizes and **centre-crops** every image to
  the first one's frame before this node runs. References of different shapes are
  already cropped by then and nothing here can undo it. Wire a **list** instead — use
  **MMH3 Image List**, an Autogrow of image sockets (up to 50) that emits one — and each
  reference keeps its native size and gets its own aspect-correct target, which is what
  core's per-socket node does. KJNodes' `ImageTensorList` also emits a list but takes
  exactly two inputs, so N references need N-1 chained nodes. Either way the node logs each reference's incoming and resolved size, and
  says so when a multi-image batch arrives.

  **A reference video is NEVER padded, and its tail is trimmed to the grid.** It has to
  land on 17j+5, and it gets there by counting **down** — so up to **16 frames are
  discarded from the end**, and nothing puts them back. A 100-frame reference becomes
  90. Padding was considered and rejected: black or a held last frame would assert
  content you never supplied, which is worse than the model simply having no reference
  there.

  What that costs is now **reported** rather than silent. The node logs the grid trim
  with both counts, and warns when the reference covers less of the target than the
  target is long — naming the exact number of frames left unconditioned. That tail is
  filled from the prompt alone, which reads as the model ignoring your reference near
  the end of a clip. On a 192-frame render a 100-frame reference covers **47%**, and
  102 frames have nothing to follow.

  Windowed references usually escape this: the spans `_plan` produces are already
  grid-aligned (a 47-latent window is 158 frames, and 158 % 17 == 5). An
  arbitrary-length whole-clip reference is the exposed case. Make it at least as long
  as the target, or land it on the grid.

  **`window_ref_video` cuts the reference video to each chunk's own span.** Off by
  default and byte-identical when off. On, the node encodes one reference set per
  chunk instead of one for the sequence, so chunk *i* is conditioned on the footage it
  is actually rendering. The spans come from the **same `_plan`** the sampler and
  Window Plan run, so reference window *i* is by construction the span chunk *i*
  renders — wire the sampler's own `chunk_frames` and `overlap_frames`, or they stop
  matching and every chunk conditions on somebody else's footage.

  **One prompt is reused for every window.** Give one and it broadcasts; give exactly
  one per window and they pair up. Anything else **refuses**, because the old
  behaviour — warn, and let the shorter list decide — produced the worst available
  output: the cond list came out short, the sampler pairs chunk *i* with
  `conds[min(i, len-1)]`, and so every chunk after the first reused chunk 0's cond and
  with it chunk 0's reference *window*. The reference restarted from frame 0 on every
  chunk, which reads as the clip looping back on itself at each seam with nothing to
  say why. Reported 2026-08-30 and fixed in 0.91.0.

  **`ref_windows` shows what each chunk actually sees** — the first and last frame of
  every window, in order, two frames per chunk, at native size. The log has always
  printed the spans as numbers, which is enough to check arithmetic and not enough to
  notice a chunk looking at the wrong footage. Preview it; a span that restarts at the
  beginning of the reference means the windows are not the spans the sampler renders.

  The soundtrack is windowed with it, cut on the **same clock** (seconds) rather than
  by latent arithmetic — 24 fps against 40 Hz is not additive, so cutting each from
  seconds is exact where deriving one from the other accumulates drift. Measured 0.00
  frames of drift across four windows.

  It costs N text-encodes rather than one, but **inside a single text-encoder load** —
  N forward passes, not N model swaps, which is the thing this node exists to protect.
  Since 0.98.0 the two encoders are also *grouped*: every window's references are
  VAE-encoded first, then every prompt is text-encoded, rather than alternating.
  The vision work is partitioned, not duplicated. And at sampling time it is *cheaper*:
  reference tokens are attended at every step, and a window is a fraction of the whole
  reference — measured at **~32%** for a 600-frame reference in four windows.
  Timestamps restart at 0 within each window, which is what the model sees at
  generation time.

  **`embeddings` prepends H3 text embeddings to each chunk's prompt.** It is a
  **socket, not a widget**: wire **MMH3 Embedding Select** into it, one node per
  embedding, chained through `previous`. A model file gets picked rather than typed,
  because a name that does not resolve is dropped by core with a log line and no error.

  Scheduling lives on each Select node's own `chunks` field — `all`, `3`, `4-6`,
  1-based — so the chain schedules as well as stacks. That works because every chunk
  already has its own prompt and its own text-encode: scheduling is only a question of
  which prompt strings get the token, with no cross-chunk state to carry. Stacked
  embeddings are additive in both cost and effect — measured, a plain prompt splices 0
  vectors, `bullet_time` splices 94 (its exact row count), and `bullet_time` +
  `storm_magic` splices 231 = 94 + 137.

  These are DiffSynth-Studio's *Diffusion Templates* — textual inversion for H3, a
  lightweight alternative to LoRA on a 33B DiT. They are **not free**: 50–142 token
  slots each, attended at **every** sampling step of every chunk, and the node's log
  prints the per-chunk total so the bill is visible before you pay it.

  A name with no matching file stops the run rather than being dropped with a log
  line.

  **All audio is normalised to STEREO before it reaches the VAE.** H3's audio VAE
  expects two channels and core's `_encode_ref_audio` hands the waveform straight to
  it, so a mono track encodes without complaint and is quietly wrong — sglang refuses
  the same input outright. Mono is duplicated, anything above stereo is summed into
  both sides with a warning that the image is gone. Traced by **fredbliss** in
  `minimax_h3_chatter`, 2026-08-22: *"you just do NOT want to pass mono audio encoded
  into h3."*

  `use_input_audio` also **trims the waveform to the clip before encoding** rather
  than encoding the whole track and discarding latents past the end — a five-minute
  track for a sixty-second render was five minutes of VAE encode to keep one. A small
  margin is left on, so the latent-side cut still decides the final length.
  Reference audio is *not* trimmed, since a reference is chosen rather than derived,
  but anything over 30s is reported: references are attended at every step.

- **MMH3 Image List** — collect many reference images into a LIST rather than a batch,
  one Autogrow socket each, up to 50. Feed it to **MMH3 Reference (Multi-Prompt)**'s
  `ref_images`.

  Batching cannot preserve differing shapes — a tensor cannot be ragged, so core's
  `BatchImagesNode` and KJNodes' `ImageBatchMulti` both centre-crop everything to the
  first image, and by the time it reaches the reference node the crop is undetectable.
  Socket order is `<Picture i>` order; empty sockets are skipped, so gaps are fine; a
  socket holding a multi-frame batch expands to one `<Picture>` per frame.

  The report names every reference's dimensions, says how many distinct shapes survived
  and what batching would have cropped them to, and warns past 9 references — they are
  attended at **every** sampling step, so the cost lands on every chunk of every pass.
  (9 is the hosted API's cap, not the model's.)

- **MMH3 Embedding Select** — choose an H3 text embedding from a dropdown and say which
  chunks carry it, then wire the output into **MMH3 Reference (Multi-Prompt)**'s
  `embeddings`. Chain one per embedding through `previous` to stack them; costs and
  effects are both additive.

  It exists because a model file should never be typed. The reference node's field is a
  spec string, and a typo in one resolves to nothing — core drops an unresolvable
  `embedding:` with a log line, not an error.

  The dropdown lists the **whole folder**, like every other loader. H3's embeddings are
  5120 wide and an SD/SDXL textual inversion in the same folder is 768 or 1280; picking
  one is refused at run time with both numbers named, rather than being hidden from the
  list. A file whose header will not parse is listed and allowed through — its slot
  count simply reports as `?`.

- **MMH3 Apply ControlNet** — a MiniMax H3 **Fun ControlNet** that follows the chunk
  being rendered. MODEL + MODEL_PATCH in, MODEL out, matching core's node since the
  model-patch rework.

  Chunk-awareness is the whole point. Core's `_fit_frames` picks hint frames with
  `torch.arange(frame_count)` — from zero, for the control video, the inpaint mask and
  the source video alike — and `prepare_control_latent` caches the encode keyed
  on the target shape. Every chunk shares a shape, so unwrapped, **all of them are
  driven by the control video's opening frames** and chunk 0's encode is reused
  throughout, with no error anywhere. The wrapper slices those three inputs to the
  chunk's own span before delegating, so core's arange-from-zero is right because zero
  is now the chunk's first frame. The offset comes from the sampler as
  `transformer_options['mmh3_control_frame0']`, computed with `frame_at_latent` — not
  `latents_to_frames`, which is only meaningful on the 5j+2 grid and answers -12 for
  index 1.

  Unset or 0 is a clean pass-through, which is exactly right for a whole-clip pass
  through a stock sampler. The control video must cover the **whole clip**, not one
  chunk: windows are cut from it by frame index, and a short one clamps to its last
  frame rather than erroring.

  The node windows core's internals (`cond_hint_original`, `inpaint_video`,
  `inpaint_mask`, `cond_hint`) and **refuses with a message** if any of them is
  renamed, rather than mis-windowing quietly. The checkpoint goes in
  `models/model_patches/` and loads with core's `ModelPatchLoader`; see
  [`workflows.md`](workflows.md).

- **MMH3 Cond To Set** — the inverse of Cond Select: wrap an already-encoded
  CONDITIONING as a one-entry cond_set, no text encoder involved. The looping
  sampler requires a cond_set and ignores the guider's conditioning, and every
  other producer of one goes through the CLIP — so a refine pass conditioned by
  a zero-out (no prompt, no encoder anywhere in the graph) had no way to reach
  the sampler without loading 20 GB to tokenize an empty string. `count`
  replicates the same conditioning N times; 1 already covers any chunk count,
  since the sampler reuses the last entry.

- **Apply H3 RefMods to Cond Set** — inject a MiniMaxH3Mod bundle into every
  chunk's conditioning. RefMods are reference images/videos packed into one
  safetensors, so a likeness or concept reloads without re-encoding. By default
  mods are inserted BEFORE trailing control blocks, which keeps control schedules'
  negative indexes valid; `append_last` puts them after, and then those indexes
  need adjusting. `retention` multiplies every loader row's strength (0 injects
  nothing). An unwired or all-zero bundle is a passthrough. Mod references carry
  no `<Picture N>` tag — prompt them by description. Needs the separate
  [ComfyUI-MiniMaxH3Mod](https://github.com/Saganaki22/ComfyUI-MiniMaxH3Mod) pack
  for the `mods` input; the node loads without it, but nothing can feed it.
  See [docs/refmods.md](refmods.md). From JalenBrunson's fork.

- **MMH3 Cond Set From Viggle** — adapt the cond_set from Viggle-Animate
  Conditioning (H3, Windowed) for the looping sampler. Both packs emit core's
  `minimax_refs` blocks, so the conditioning is already compatible and only the
  socket type name differed. Viggle bakes each window's slice of the driving clip
  into its own cond, so the two schedules must agree: this derives the
  `chunk_frames`/`overlap_frames` that rebuild Viggle's spans, replans them the
  way the sampler will, and reports whether the boundaries match. Wire both INT
  outputs into the sampler. A differing schedule with the SAME chunk count is the
  dangerous case — the sampler clamps its cond index and would render every chunk
  against the wrong frames without complaint — so it is called out per chunk.

- **MMH3 Cond Set Strip Text** — drop the prompt from every entry of a cond_set
  while the reference media rides through untouched. For a refine pass whose
  windows are **smaller than the chunk the prompt was written for**: core picks a
  window's prompt region from the window's midpoint, so a window covering a
  fraction of the timeline gets text describing all of it and is asked to render
  the whole script into its slice. At low denoise nothing is invented anyway — the
  content is already in the latent, and identity is the only thing worth
  conditioning on.

  It works because the two are in different halves of a conditioning entry: the
  prompt is the tensor, the references are keys in the dict. `zero` blanks the
  text values and keeps the span's length; `vision only` keeps just the image
  tokens and drops the prose, shortening `text_len` — but references appended
  after encoding never registered with the tokenizer, so for those it leaves the
  text span empty. The node reports that rather than preventing it.

### Prompting
- **MMH3 Asset Plan** / **MMH3 Task System Prompt** — build a Context-IR system
  prompt for your own LLM node from the task type (or combination) and the
  assets in play, emitting only the relevant rule blocks. See
  [`context-ir-system-prompt.md`](context-ir-system-prompt.md) for the full spec these are derived from.
- **MMH3 Music Caption System Prompt** — the same idea for **MiniMax Music 3**, whose
  `caption` field wants a three-section Structured Caption (Global Metadata / Vocal
  Details / Arrangement) rather than a tag list. MiniMax ships a hosted
  `music-caption-rewriter` to produce one; locally there is none, so this emits the
  rules for your own LLM. Three `lyrics_mode`s — write, supplied (words fixed), or
  instrumental — and an optional section skeleton sized to the duration.

  Duration constants are read from the **installed** model
  (`comfy.ldm.minimax_music.ar`), so the ceiling is the real 360.0s rather than the
  model card's "~5 minutes". Needs ComfyUI v0.33.0+ for Music 3 itself.

  Note that MiniMax's *older* music guide targets the previous generation's hosted
  API — comma-separated descriptors, `--instrumental`, bitrates. Its lyrics tags carry
  over to Music 3; its caption advice does not.
- **MMH3 Lyrics Sectionize** — split fixed lyrics across numbered `[Verse 1]` /
  `[Verse 2]` sections **without changing a word**, with `[Instrumental]` between them.
  Music 3 allocates time **per section**, so one long block is compressed into one slot
  and the delivery rushes — diagnosed in the community's own testing, where the fix was
  breaking long verses up rather than slowing anything down.

  Deterministic on purpose: an LLM asked to re-emit a fixed lyric rewrites it, which is
  the whole reason this is not part of the caption prompt. Boundaries land on paragraph
  breaks then sentence ends, never mid-sentence, and the word sequence is **compared
  before and after** — it raises rather than drifting. Numbering matters because the
  caption's section-level instrument evolution refers to sections by name.

  Wire its one output twice: to the encoder's `lyrics`, and to the caption node's
  `supplied_lyrics` so the caption is written against the same sectioned text.

- **MMH3 Music Caption Split** — the join to `MiniMaxMusic3TextEncode`: one LLM reply
  in, `caption` and `lyrics` out. Tolerates code fences, preamble, bolded or bulleted
  labels, and a missing lyrics field. Names an empty caption and a tags-but-no-words
  lyrics block rather than passing either on silently, since both look like model
  failures downstream.

  Full path: idea -> LLM (system prompt) -> Split -> caption/lyrics -> Text Encode.
- **MMH3 Prompt Lint** — check a written prompt against the format its `mode`
  implies: missing sections, a `retention_analysis` line with no marker, a hidden
  cut, timestamps out of order, `[Shot 1]` carrying one. Reports rather than
  rewrites.
- **MMH3 Replace Section** — splice one refined section back into a complete prompt.
  The two-model route: the technical model writes the whole prompt, a second expands
  `detailed_description`, this puts it back. Both formats' section sets are known,
  so it refuses a section the selected mode does not have.
- **MiniMax H3 Prompt Accumulate** — append one prompt to a running pipe-separated
  string, for a graph loop writing one prompt per window. Exists because a loop
  carries values, not lists. The first pass is the case that goes wrong: the carried
  slot is unwired on iteration 0, and a naive accumulator emits a leading separator
  or the literal text `None`. `prior_context` formats the earlier prompts for
  feeding back to the writing model — put a second copy at the *top* of the loop
  body to read it, since this node sits after the model and its own output cannot
  reach upstream.

  **`prior_context_mode` is the lever on repetitive output.** `all` (the default)
  re-sends every earlier prompt in full — ~7,900 tokens by window 7 of a 20s-window
  clip, against a few hundred for the new audio, which is roughly 20:1 in favour of
  copying. It also re-sends every earlier `detailed_description`, the one section
  the header asks to *differ*. `last_definitions` sends only the previous window's
  `subject_definitions` and `retention_analysis` — what must stay identical, and
  nothing to imitate for what should not. If late windows are re-describing earlier
  ones, start here.

- **MMH3 Scene Plan Prompt** / **MMH3 Prompt Part** — build N chunk prompts **section
  by section** instead of chunk by chunk.

  Writing chunk *i* in isolation asks the model for a complete arc in every chunk. It
  cannot know it is the middle, so every chunk sets up, escalates and resolves — in
  testing, five variants of one scene, each landing its own climax. That is the loop's
  shape, not the wording, so no amount of rule-tightening fixes it.

  Transposing the loop fixes three things at once. `subject_definitions` and
  `retention_analysis` are written **once** and reused verbatim, so the drift that
  produces a stray `<Subject 2>` with no retention line becomes impossible. Escalation
  is decided where all N chunks are visible — the beat sheet — with an explicit floor:
  nothing resolves before beat N. And dialogue planned across the whole set cannot
  repeat a line in three chunks, which per-chunk planning reliably does.

  It also costs **fewer** LLM calls, not more: `1 + 1 + N` against `2N`. Eight chunks
  goes from 16 calls to 10.

  | stage | calls | writes |
  |---|---|---|
  | `definitions` | 1 | every film-wide section — definitions, retention, soundscape, score — plus bare `summary:` / `detailed_description:` headers |
  | `beats` | 1 | all N summaries, pipe-separated: the escalation ladder |
  | `shots` | N | one chunk's `detailed_description`, given the **whole** beat sheet and told which beat it is |

  Soundscape and score are film-wide for the same reason the definitions are — a sound
  world that drifts between chunks is audible drift — so the `definitions` call emits a
  complete six-section skeleton. The bare headers are not optional: **MMH3 Replace
  Section** refuses to splice into a prompt with sections missing.

  Wiring, using nodes you already have:

  ```
  Scene Plan (definitions) -> LLM ------------------------> skeleton
  Scene Plan (beats)       -> LLM ------------------------> beat sheet
    for i in 0..N-1:
      Prompt Part(beat sheet, i) ------------------------->  beat i
      Prompt Accumulate(carried).prior_context (mode last) -> prev chunk's detailed
      Scene Plan (shots, beat_index=i, beat_sheet=.., prev_detailed=..) -> LLM -> shots i
      Replace Section(skeleton, beat i,  "summary")
      Replace Section(     ^  , shots i, "detailed_description")
      Prompt Accumulate -> pipe-separated string -> MMH3 Reference (Multi-Prompt)
  ```

  **MMH3 Prompt Part** is the join between a sheet written all at once and a loop
  rendering one beat per pass: it splits on the same `|` the accumulator and
  multi-prompt use, tolerates the code fences an LLM adds anyway, and past the end
  either repeats the last beat (matching how the looping sampler reuses the last cond)
  or raises, your choice.

  **Chunk-to-chunk continuity** (`prev_detailed`, optional, append-only input). By
  default the `shots` writer sees only the beat *sheet* — the summaries — never the
  previous chunk's realised output, so it continues the story but re-invents the
  staging, and a cut-to-cut render resets the camera and poses every chunk. Wire the
  loop's carried prompts back through **MMH3 Prompt Accumulate**'s `prior_context`
  output in mode **`last`** into `prev_detailed`, and each chunk is told to open
  [Shot 1] on the previous chunk's FINAL frame and advance — so chunk *i+1* picks up
  where chunk *i* ended (same positions, injuries, camera) instead of opening cold.
  Empty on beat 0; unwired past beat 0, the node warns. This is the cinematic opposite
  of the music-video path's `last_definitions`, which withholds `detailed_description`
  on purpose because there the windows must *differ*.

  **`mode`: `cinematic` (default) / `talking_head`.** `talking_head` repurposes the same
  machinery for one ABSOLUTELY-LOCKED continuous take — a person speaking to camera. The
  `shots` stage holds a fixed-tripod frame (no cut, no camera move, no new action),
  writes a continuing spoken monologue instead of escalating, needs no `beat_sheet`, and
  reads `brief` as the topic the subject talks *about*. Definitions and the continuity
  feed carry over unchanged. Because there is no cut at the chunk boundary to hide a
  seam, it is also the strictest test of whether the looping sampler joins chunks
  seamlessly.

  The `shots` stage refuses to run without a `beat_sheet` rather than quietly writing
  a self-contained chunk — that failure is the one this exists to remove. Its banality
  rule is scoped to **speech only**: banal lines over an escalating scene, never a
  banal scene.

### Music video

A separate three-stage chain from the cinematic one, because a song already has an
arc and the cinematic rules fight it. Full pipeline: separate the vocal, align the
lyrics against it, slice the alignment by render window, then write prompts.

- **MMH3 Forced Align (Lyrics)** — place KNOWN lyrics on the timeline. Forced
  alignment, not transcription: the words are given and only their timing is solved,
  so it cannot mishear. That matters because a transcriber guesses badly at singing,
  and everything downstream inherits the mistake — prompts describing words nobody
  sang, typography quoting a mishearing.

  ⚠ **Feed it the lyrics AS PERFORMED, not as prompted.** Alignment assumes the text
  and the audio hold the same words the same number of times. Suno repeats hooks and
  stutters refrains; one copy of a line against three utterances leaves the aligner
  to pick one and strand the rest, which surfaces as large gaps, stretched words and
  whole sections landing early. **No parameter fixes that** — `nonspeech_skip`,
  `max_word_dur`, VAD and `snap_to_onset` all decide *where a word may land*, and
  none of them can conjure two missing repeats. Writing the line three times does.

  It refuses to return a word sequence that differs from its input, since a
  misaligned lyric is fiction every consumer would quote. Emits the same
  `whisper_alignment` type ComfyUI-Whisper does — so `Whisper → Text` and
  `Whisper → Segments` consume it unchanged — plus JSON, so a song is aligned once
  and reloaded instead of paying for a 3 GB model every run.

  Needs isolated vocals; any separator will do. Needs `stable-ts`, which is one pure-
  Python package on top of `openai-whisper`. `large-v3.pt` is shared with any
  non-ComfyUI install through `folder_paths`, so there is no second copy.

  **The report is the interface.** It prints the section map — the one line you can
  check against your own ears — and classifies every anomaly using the audio itself
  rather than guessing: a gap over silence is a correct skip, a gap over audio is a
  skipped passage, and words sitting on silence are misplaced. That distinction is
  what separates a musical pause from a misalignment, and no amount of timing
  arithmetic can make it.

  ⚠ `vad` (Silero) is available and was **far worse** on a produced vocal —
  131 of 190 words came back zero-length, because Silero is trained on speech and
  does not fire on singing. Useful for spoken word; not for song.

- **MMH3 Music Analysis** — librosa: BPM, key and mode, a 4/4 bar grid, and a 10 Hz
  RMS energy curve, from the FULL MIX rather than the stem. Ported from
  music-director's `music.py` minus its cut-salience blend and agglomerative
  segmentation — both exist to *choose* scene boundaries, and the looping sampler's
  windows are uniform and already fixed.

  What survives is what still helps inside a window someone else decided: bar lines
  are cut candidates alongside word onsets, and energy is the only thing that tells
  an instrumental window whether the music there is a soft fall or a drop.

- **MMH3 Lyrics to Windows** — the join between a song's timeline and a sampler's
  schedule. Inputs mirror **MMH3 Split Audio to Windows** exactly so both read the
  same plan and cannot disagree about which frames window *i* covers.

  Three things it exists to get right, each silently wrong if hand-rolled:

  **Window-relative timestamps.** H3 shot times are measured from the start of the
  CHUNK. A window opening at 70.15s holding a word at 72.40s must emit `00:02.250`;
  absolute time produces prompts H3 cannot act on and nothing errors. Context
  windows are rebased onto *this* window's clock too, so a neighbouring line does
  not read as a second, contradictory timeline.

  **`has_lyrics`.** Intros, instrumental breaks and outros are real windows with
  nothing sung in them, and they need a prompt branch that says so rather than one
  left to invent singing.

  **Section context.** Uniform windows and musical sections do not divide, so a
  straddling window reports `chorus -> bridge (bridge begins at 00:07.000)` rather
  than pretending it sits in one.

- **MMH3 Music Scene Plan Prompt** — the same three stages as **MMH3 Scene Plan
  Prompt**, with the rules inverted where a song demands it:

  | | cinematic | music video |
  |---|---|---|
  | the arc | invented, nothing resolves before beat N | **the song's** — "do NOT invent an escalation" |
  | a repeat | would be redundant | **should feel like the same chorus** |
  | the words | invented | supplied, verbatim, per window |
  | shot times | invented | supplied as word onsets |

  `non_diegetic_music` describes **this** track rather than composing an
  alternative, and `overall_soundscape` opens with "the song is the audio" so it
  does not invent room tone competing with it.

  **Typography is rationed in `beats`, once, across the whole song** — "most should
  not." Decided per chunk with lyrics in hand, every chunk reaches for it and the
  result reads as a lyric video. Two modes: `exact lyrics` quotes the sung line
  verbatim; `text bursts` allows fragments and re-spellings, which is invention
  grounded in the real words rather than replacing them.

  A window with `has_lyrics: false` switches to an instrumental branch that forbids
  singing, asks for visual event instead, and **suppresses typography even when the
  beat sheet assigned it** — there is no line to quote.

  **`music_source`** — `supplied` (default) or `generated`. The rules assume the
  track exists and will be handed to the sampler; `generated` is for a graph where H3
  writes the audio in the same pass. It turns `non_diegetic_music` from a description
  into the **spec the model performs**, stops `overall_soundscape` claiming a track
  was provided, and — the part nothing else supplies — makes the shots stage quote
  the window's lyrics as `<d>[English] ...</d>` with the subject **singing** them.
  Without it the words reach the writer, get spent deciding what the picture does,
  and none are sung.

  **`treatments`** — `music video` (default) or `restrained`. The default reaches for
  split frames on purpose, which is a music-video technique rather than an artifact.
  `restrained` forbids frame division, inset, banded overlay and multiplied
  performers, and keeps the effect vocabulary optical. For a piece whose subject is
  the performance, a split frame halves the singer exactly when the mouth is the
  point.

  `reference_images` tells the definitions stage that attached images ARE the
  subject, that several images are one person from different angles, and that the
  image beats the brief on appearance. Definitions only: the description is written
  once and reused, and re-deriving it per call is how a subject drifts. Without the
  flag a vision model handed pictures and no instruction describes an invented
  character anyway.

  The whole chain — grid, alignment, the three stages, typography, symptom table —
  is written up in [`music-video.md`](music-video.md).

- **MMH3 Official H3 Tokens** — adds H3's seven special tokens to a CLIP's
  tokenizer. **Not needed on current ComfyUI**, which adds all seven at tokenizer
  init; the node detects that and passes the CLIP through untouched. It stays so
  older graphs that have it wired still load.

- **MMH3 Load Skill** — loads one file from the pack's `styles/` folder and emits it
  for `extra_rules` on either scene-plan node. **Chain the nodes to stack skills**:
  wire one node's output into the next node's `previous`, so wiring order is stacking
  order. `enabled` off passes `previous` through untouched.

  One file per node on purpose. Selecting several in one node means deciding up front
  which kinds of skill exist and how many of each you may have; a chain decides
  nothing. The type lives in the filename — `look-`, `typography-`, `experiment-` —
  which is enough to find it in the dropdown, and anything you drop in the folder
  appears there with no registration step.

  An `experiment-` file is flagged in the report as untested: those say what we want
  to find out H3 can do, not what has been observed working, so judge the result on
  its own rather than as a known recipe.

  **Why blocks and not the vendor skills.** MiniMax publish nine H3 skills; all nine
  are agent procedures for their own hub, and two say so outright. Around their visual
  guidance sits numbered steps, confirmation gates, prescribed shot counts and time
  segments written for 15-second clips. Pasted whole into a prompt, that fights a
  grid-locked window and a pinned master audio. `styles/` is the visual core lifted
  out with the procedure left behind.

#### Observed — 2026-08-15

What a first full music-video run taught — the typography corrections, thematic type,
split frames as a technique rather than an artifact, and why slow motion is treated as
a choice here — now lives in
[`music-video.md`](music-video.md) §9, alongside the chain it applies to.

### Sampling
- **MiniMax H3 Looping Sampler** — fill a whole clip chunk by chunk in one node
  execution. The graph is the same size for 4 chunks or 40, which is the point.

  **The latent is the finished clip**, and the chunk count is derived from it — you
  hold a song of known length and do not know how many chunks that is. Chunks are
  slices written back in place, so there is no join, no trim, and the output is
  exactly the length you passed in. Each chunk also slices its own span of audio, so
  a track pinned by `use_input_audio` reaches every chunk.

  The schedule comes from the same `_plan` as **Window Plan** and **Split Audio to
  Windows**, so chunk N renders the audio window N's prompt was written against.
  Two carry routes (masked overlap, or a guide), keyframe indices in clip frames,
  and a per-chunk guider swap.

  Its optional **`denoise_mask`** takes a MASK over the whole clip — white
  regenerates, black keeps the input latent's content — so a region or a span can be
  held while the rest re-renders. It masks the **video half only**; the audio half is
  reached solely through `audio_denoise_mask`. The mask is reduced ONCE onto the
  master grid and
  merged keep-wins (elementwise min) with whatever mask the latent carried and with
  the overlap carry, then sliced per chunk, so `_carry_mask` composes with it and
  nothing in the loop changed.

  Geometry follows the VAE rather than a resize. Spatially it is **pooled, not
  interpolated** (bilinear averages, and every fractional cell then denoises at its
  own timestep), then snapped to the **2×2 patch the DiT reads the mask through** —
  so 32 pixels is the finest feature a mask can express and no token straddles an
  edge. Temporally it groups on the real `FRAME_PER_TOKEN` cycle `(1,4,4,4,4)`, where
  the first latent of every 17-frame group covers a **single** frame; a uniform 17/5
  split puts a temporal edge in the wrong place.

  **`audio_denoise_mask`** masks the audio half, and is the only input that does.
  Only its TIME axis is read: each frame reduces to one value, then maps onto the
  audio grid through `_audio_index_at`, the same boundary conversion the chunk loop
  uses, so a span frozen here lines up with the picture. Left unconnected, audio is
  masked only by whatever the input latent already carried.

  The two are **independent by design**, which is a correction (2026-08-22). Audio
  used to be *derived* from the video mask whenever no explicit audio mask was wired,
  on the reasoning that the two modalities could then never disagree about a frozen
  span. That is only true of a mask carrying temporal intent. A **spatial** mask — a
  SAM3 subject matte, say — is white somewhere in every single frame, so the spatial
  reduction returned "free" at every timestep and regenerated the whole track while
  the video mask did exactly what was asked. Measured on a subject matte: 75% of the
  video grid held, 100% of the audio freed. Deriving one modality from the other is
  not a safety net, so it is gone.

  It warns when the input latent is all zeros, since kept regions would pin black.
  This is a v2v tool.

  The sigma schedule can be **windowed per chunk** with `sampling_start_step` /
  `sampling_end_step` — absolute indices, sliced exactly as core `SplitSigmas` does,
  so a two-pass run is `end N` then `start N` with no arithmetic. `phase2_start_step`
  plus an optional `phase2_sampler` / `phase2_guider` switches solver mid-schedule
  for dual-solver setups. All three carry LTXAVTools' semantics unchanged. See
  [`looping-sampler.md`](looping-sampler.md) — including what is still
  unmeasured.

  **`guide_video`** (0.105.0) is core's **Add Guide for MiniMax H3**, windowed. Core
  anchors a guide clip for ONE generation; it cannot follow a clip across chunks, and
  this node strips any guide arriving on the conditioning, because one upstream guide
  would anchor every chunk to the source's opening frames. So the source comes in
  here instead: each chunk takes its own span of it, encodes that span as one clip
  (cut to 17k+5 frames, as core does) and anchors it at the chunk's frame 0. Every
  chunk is guided by exactly the frames it renders, on the output's own clock.

  That is what guide-latent LoRAs need: edit and style-transfer LoRAs trained on
  guide/target pairs, which have no effect through a reference. A **reference** sits
  on its own clock, so the model uses it as material and motion drifts; a **guide**
  sits on the target's, so it lines up frame for frame. The base model is trained only
  for first/last-frame anchors, so a full-clip guide does nothing useful without such
  a LoRA.

  Frames are centre-cropped to the generation's grid. A source shorter than the clip
  guides what it covers and leaves the rest unguided; a longer one has its excess
  ignored; both are reported. Each span is encoded **fresh** rather than sliced from
  one whole-clip encode, so every guide opens on a real 1-frame token, as a
  single-generation guide does. A guide roughly **doubles each chunk's sequence**:
  chunks run slower and may need a smaller `chunk_frames`. Picture only; keep the
  source's sound with `use_input_audio`. It refuses `carry="keyframe"` and
  `keyframes`, which would stack a second guide on frames it already guides — use
  `carry="mask"`. Unit-tested against a stubbed sampler
  (`tests/test_guide_video.py`); not yet run on a real render.

  **`carry_color_match`** (0.106.0) is SoulX-FlashHead's anti-drift step, applied to
  the carry. A long chain "cooks": each chunk inherits the last one's colour error
  through the carried latents and adds its own. With `first_chunk`, chunk 1's
  decoded frames become the colour anchor. After every chunk, the frames the next
  chunk will carry are decoded, moved onto the anchor's Lab mean and spread frame by
  frame, re-encoded and written back as the carry. `first_chunk_latent` does the same
  on the latents themselves (per channel, no `vae`, no extra time).
  `carry_color_strength` blends between as-sampled (0) and fully matched (1). Off by
  default.

  On a 20 s static talking head it held saturation at 55–60 where the uncorrected
  chain climbed 56 → 102, and the face's red-minus-blue at 48–56 where it climbed
  54 → 106. On its own it does **not** hold identity: the face still drifts by
  15–20 s. With `overlap_strength_video` lowered to **0.6** as well, the same chain
  kept her face and detail to 20 s, at the price of a slightly softer image and a
  larger colour step at seams (one seed, one shot so far). The pixel route costs one
  small decode and encode per chunk (~0.9 s at 512x512); the latent route held colour
  as well at no cost and kept more detail, with a larger seam step. Measurements and
  method in
  [`looping-sampler.md`](looping-sampler.md) §3 and §10.

- **MiniMax H3 Keyframe Planner** — end-anchored keyframe indices for a chained run,
  ported from LTXAVTools' planner. Frame 0 opens, each chunk travels to a keyframe at
  the last frame **it renders**, the final one ends on `-1`. Start-anchoring instead
  would put each image in the NEXT chunk and invite a snap at every seam. Emits
  `indices` for the sampler's `keyframe_indices`, `count` for how many images the
  batch needs, and `chunk_count`.

  Same three numbers as the sampler, same `_plan`, so the two cannot disagree about
  where a chunk ends.
- **MiniMax H3 Context Windows** — windowed sampling over one long latent, per
  modality: video on dim 2, audio on dim 3, each with its own window. Snaps length
  and overlap to the grid, since an overlap that is a multiple of 5 rather than
  `5m+2` walks the window phase `0,2,4,1,3` — a five-window beat, which is the
  pulsing. See [`context-windows.md`](context-windows.md).

  Windows are **not** a way to grow a clip: every window is a slice of one
  preallocated latent, and all of them sit at the same noise level at every step.
  Chaining is what grows.

  Windows bound the model's *compute*, not the sampler's *storage* — the full
  latent, its noise, and the fuse accumulators stay resident at full length, so a
  longer clip still costs VRAM at a fixed window size. Two things trim that:
  a cond skipped by cfg 1.0 no longer allocates its accumulator at all (its zeros
  are materialized after the window loop instead — automatic, saves one full-length
  fp32 latent), and `accumulator_device: cpu` hosts the remaining accumulators in
  system RAM, writing window-sized slices across PCIe during the loop and moving
  the fused result back once per step. Values are identical either way.
- **MMH3 Timeline Preview** — the timeline so far, **playing at real time**, extended
  each time a chunk finishes. Wire it between the model and whatever builds the guider;
  it passes the model through unchanged and only registers itself.

  A chunked render is where a progress bar tells you least: it says step 34 of 160 and
  nothing about whether the shots are the right ones in the right order. Each chunk's
  latent is in hand the moment it is written back, and was being thrown away.

  Each finished chunk is decoded and appended, and what you get is a **timeline of
  the piece so far, played at 24 fps** — real time, which is the only setting that
  says anything about pacing — with **play/pause, a scrub bar, and its own sound**
  when `audio_vae` is wired. `frame_stride` is a fidelity dial, not a speed one: at 3
  you keep a third of the frames and hold each three times as long, so the clip
  still runs at real time for a third of the encode.

  **`audio_vae`** (optional, since 0.100.0) decodes each finished chunk's audio too.
  Wire a stock VAE Loader at `minimax_h3_audio_vae_fp32.safetensors` — the full H3
  audio VAE; there is no tiny variant and none is needed, a chunk of audio decodes in
  milliseconds and the model stays resident. The decode is core's own
  `vae_decode_audio`, loudness normalisation included, so the preview sounds the way
  the render will. The sound is cut to the *kept frames*, not to the latent — the
  video path decodes a grid-valid prefix of each chunk and holds the last kept frame a
  whole stride, and the audio covers exactly that span — so picture and sound stay in
  step at every chunk boundary and under the `max_frames` cap. Sent as mono 16-bit
  WAV at the VAE's rate (32 kHz → 64 KB/s, 3.8 MB at the 60-second default cap). A
  decode that fails once leaves the preview silent for the rest of the run rather
  than failing it. `live_steps` stays video-only.

  **One decode per chunk, never sliced.** The temporal grid does not compose
  additively — 7 latents decode to 22 frames and 2 decode to 5 — so tiling a 12-latent
  chunk out of grid-valid slices yields 12 frames where a single decode yields 39.
  Slicing was built, measured at **half speed**, and removed. Chunks arrive
  grid-aligned from `_plan`, so a whole-chunk decode is valid by construction; a very
  long chunk decodes a grid-valid **prefix** instead (`max_decode_latents`, default 22
  → 73 frames), because a whole chunk materialises at render resolution before it is
  scaled down and 47 latents of 1344×768 is ~1.9 GB in fp32.

  `max_frames` caps the timeline, dropping the oldest, so a long render keeps a moving
  window rather than a timeline that grows without limit.

  **`live_steps`** (off) additionally pushes a still of the chunk being sampled on every
  step, from the `x0` core hands the wrapper as positional arg 5 of the sampler
  callback. Off by default because it replaces the playing timeline with a frozen
  frame, and it stops sending once the timeline exists — otherwise every step would
  re-encode the whole timeline.

  **`suppress_sampler_preview`** (on) silences ComfyUI's own latent preview for the
  duration of the sample, so the sampler node and this one do not draw the same thing
  twice. Every *concrete* `decode_latent_to_preview_image` under `LatentPreviewer` is
  patched, not just the base — VHS subclasses it and would otherwise carry on — and all
  of them are restored when the sample ends.

  **`vae`** is optional and gives a true-colour decode instead of the projection. Wire
  a stock **VAE Loader** at **`taeh3.safetensors`** in `models/vae` — a tiny H3 decoder,
  24 latent channels, 16× spatial — and **not** the full VAE, because this runs on every
  step. A decode that fails once falls back to `latent_rgb_factors` for the rest of the
  run rather than turning a diagnostic into a second thing to debug.

  The decode is `latent_rgb_factors`, a 24×3 projection: no model file, no VAE, no
  download, one matmul. It is an **approximation** — colour is indicative, fine detail
  is absent — and it answers the question the step counter cannot. A latent whose
  channel count does not match the format is refused rather than projected, because a
  wrong projection produces a confident meaningless image.

  **The image lands on THIS node, not on the sampler.** Core's preview channel always
  addresses whichever node is executing, so two live previews overwrite each other —
  there is one slot per running node. Taking the node's own `unique_id` and sending a
  custom event to a DOM widget is the only way to keep them separable, which is why
  the pack ships `web/js/mmh3_live_preview.js`. The widget is `serialize: false`: it
  is a view, not state.

  **Nothing rides the websocket.** The event carries metadata only (`seq`, the
  sprite grid, fps/stride, whether there is sound, chunk count, labels); the browser
  fetches the picture from `/mmh3/preview?node_id=<id>&kind=image` and the sound from
  `kind=audio`, both under the same `seq`. Core's publish loop awaits every connected
  socket in turn, so a large frame on the socket would hold every client's progress
  events behind the slowest tab. Over HTTP it is latest-wins: reassigning a `src`
  aborts the load in flight, so a slow tab falls at most one chunk behind and never
  queues. (Since 0.99.1; earlier versions sent base64 in the event.) The encode runs
  on a worker thread with a one-slot, newest-wins queue (0.99.2), so the sampler does
  not wait for Pillow after each chunk.

  **The picture is a sprite sheet, not an animated file** (0.100.0). An animation
  cannot be seeked, paused, or kept in step with a sound. The kept frames go into one
  roughly-square static WebP grid; the widget's canvas draws frame
  `floor(t × fps / stride)` at whatever `t` the `<audio>` element reports — or a
  timer, when there is no audio — so picture and sound cannot drift apart, and a
  static sheet encodes faster than the animation did. Past WebP's 16383-px side limit
  the tiles shrink to fit rather than the encode failing.

  Any error switches the preview off for the run rather than interrupting it.

  The sampler **discovers** it rather than owning it: `begin_preview` returns `None`
  when nothing is wired, and the sampling path is then byte-identical. The
  wrapper-discovery shape and the FRAME_PER_TOKEN span arithmetic are reimplemented
  from [hradec's ComfyUI-HR-Endless-Sampler](https://github.com/hradec/ComfyUI-HR-Endless-Sampler)
  (Apache-2.0); the own-node addressing is the shape KJNodes' `ModelPreviewOverrideKJ`
  uses. No code was copied from either.
- **MMH3 Get Timeline Frames** — the tiles the live preview accumulated, as a real IMAGE
  batch, one per chunk in order. The strip is transient; this keeps it. Takes the
  preview's MODEL output purely to order itself after the sampler, and refuses with a
  message when nothing was recorded rather than handing on an empty batch. Same
  `latent_rgb_factors` approximation at native latent resolution — 1/16 of the render
  per side, not a substitute for decoding.
- **MMH3 Chunk Schedule** — say roughly what you want; get a schedule that actually
  tiles. Solves total, window and overlap **together** and emits frames.

  The frame calculator answers "what does 22.2 seconds round to". Useful, and not the
  question. Three of those answers chosen independently still leave the last window
  clamped, because what has to hold is a relationship *between* the three: 60s with a
  20s window and 3s overlap gives four chunks whose last one strides 7.08s instead of
  17.00s, re-rendering 12.2 seconds a previous chunk already made under a different
  prompt. No single conversion can see that, so no amount of widget precision fixes it.

  Write the group counts as `t = 5c+2`, `L = 5a+2`, `O = 5b+2`. Then stride is
  `5(a-b)`, a multiple of 5 for **any** a and b — so grid phase is safe automatically
  and the five-window pulse cannot happen. What is not automatic is
  `(c - a) % (a - b) == 0`, and that is the whole search.

  **`chunks` is usually the input you actually have an opinion about** — it is how many
  prompts you write and how many joins the piece has. Set it and the window becomes a
  *result* rather than a second guess. An unreachable count is released and reported,
  never raised.

  **`chunk_count` and `seconds_per_chunk` wire straight into either scene-plan
  node's inputs of the same names**, so the writer and the schedule cannot disagree
  about how many chunks there are or how long one is. `seconds_per_chunk` is the
  WINDOW's duration, not the clip's.

  `prefer` decides what may move: `keep total` holds the deliverable length and shifts
  the window and overlap, `nearest` lets the length drift a few groups for a closer
  fit, `fewer chunks` takes the shortest chunk list it can. The report names every
  move and prints the divisibility as proof.

  It also lists the **reachable overlaps** for the chunk count you are on, because
  the count is the overlap's step size: with the total and the count both fixed the
  stride is `(c-b)/n` and must come out whole, so valid overlaps sit exactly `n`
  groups apart. At 3 chunks that is 2, 17, 32, 47 latents and nothing between — which
  reads as the widget refusing to move until you know why. Asking for MORE chunks
  makes the overlap COARSER, not finer.
- **MMH3 Chunk Schedule (Frames)** — the same node asked in frames. Identical solver,
  identical grid rules, identical outputs; it only skips the seconds-to-frames
  conversion, for when you already hold frame counts and do not want a duration
  rounded on the way in. The two share one implementation rather than being copies,
  so they cannot drift.

  It still **snaps**: values land on the nearest `17j+5` and are still solved
  together. Asking in frames does not mean asking for arbitrary frames — 1445 and
  1446 both resolve to 1450. Its defaults mirror the seconds node's 60.0s / 20.0s /
  3.0s exactly (1433 / 481 / 73), so the two agree out of the box.
- **MMH3 Window Plan** — resolve the whole schedule up front, in frames. How many
  windows you get is how many prompts to write; whether your window and overlap
  survive snapping is otherwise only knowable by running a generation.

  **Every output carries its unit in its name**, because crossing them is the one
  mistake this node invites: `context_length (latents)` / `context_overlap
  (latents)` go to Context Windows, while `window_frames (frames)` /
  `overlap_frames (frames)` go to the looping sampler and Split Audio to Windows.
  The frame pair sits five sockets below the latent pair, which is exactly how they
  get swapped. Crossing them does not error — a latent count is a valid frame
  count, just a much smaller one — it re-snaps and the schedules quietly diverge.
  Observed 2026-08-17: `context_length` 117 wired into the sampler's `chunk_frames`
  re-snapped to 32 latents, so the sampler ran **11 chunks of 4.5s** while both
  planners reported 3 chunks of 16.5s.
- **MMH3 Split Audio to Windows** — cut a track into one clip per window, matching
  the real schedule including the overlap and the clamped final window. The numbered
  sockets fan every window across the graph at once; the `audio` output emits ONE,
  chosen by `index`, so a for loop keeps the graph constant-size. `index` also
  reaches past the numbered ceiling.
- **MMH3 Window Context** — one line saying which span of the song a window covers,
  for the per-window prompt loop. Without it the loop hands the writing model the
  same text every iteration and only the audio changes, so on a repetitive track
  nothing distinguishes window 5 from window 2 — and `prior_context`'s "keep these
  byte-identical" then pulls the late windows onto the same shots. Same `_plan` as
  everything else, so the timecode names the audio the window really renders.
  Concatenate onto the **END** of the model's prompt, after `prior_context`.

### Latent
- **MiniMax H3 Seed Overlap** — **prepends** overlap latents to the target and masks
  them, giving frame-level seam continuity. Prepending rather than overwriting means
  the chunk keeps its full requested duration and the overlap is cut off afterwards.
- **MiniMax H3 Outpaint Latent** — grow or crop a latent's canvas, masking the new
  region so the model fills it. Edges are **signed**: positive pads, negative crops,
  and each snaps toward zero so a value between steps never crops more than asked.
  An inward `feather` ramps into the source region. H3 has no cross-attention, so
  margin rows attend directly to real rows at every layer, and scene fill converges
  in very few steps.
- **MiniMax H3 Join AV** — join two clips in **pixel** space, at frame granularity.
  Latent joins land on 17-frame boundaries; this is what Find Divergence's answer
  feeds.
- **MiniMax H3 Reference from Latent** — build a `minimax_refs` block from a latent
  directly.
- **MMH3 Chunked Pixel Upscale** — stage-1 latent → 2K latent, through pixels, a
  chunk at a time. For the **refine** leg of a 2K pass. See
  [Refine vs regenerate](#refine-vs-regenerate).
- **MiniMax H3 Streaming Encode** / **MMH3 Streaming Save** — encode and export
  in bounded RAM. Save decodes group by group and writes as it goes rather than
  holding the whole clip, which is the difference between exporting a long master and
  running out of memory. Slower per frame; for long videos only. `save_metadata` (default on) embeds
  the workflow and prompt so the file drags back into ComfyUI; it goes in an
  ffmetadata file rather than a command-line argument because a workflow is 45–95 KB
  and Windows caps a command line near 32,767 characters. `faststart` is deliberately
  not applied — relocating the moov atom rewrites the whole file and would undo the
  constant-cost decode this node exists for.
- **Whisper to Text (LLM Ready)** — flattens ComfyUI-Whisper's per-word
  `whisper_alignment` into text a prompt writer can read, with timestamp markers so it
  knows where in the song each line falls. Feed it to the scene-plan nodes as lyrics.
  Markers land on interval **boundaries** rather than on whichever word crossed one,
  so the same song always marks at the same times.

  **Adopted into this pack 2026-08-24.** It used to live in a loose
  `ComfyUI-WhisperAlignmentToText` folder that was never published, so the MusicVideo
  workflow listed a dependency nobody could install. The node id is unchanged, so
  existing graphs keep working — if you have that folder, delete it, or two packs will
  claim the same node. Its sibling `WhisperAlignmentToSegments` was deliberately NOT
  adopted: it cuts on 25 fps and a 4n+1 frame grid, which is LTX's, not H3's 24 fps /
  17j+5, and **MMH3 Window Plan** and **Split Audio to Windows** already do that job on
  the right grid.
- **MMH3 Reference Attention Probe** / **MMH3 Reference Attention Map** — which
  reference is each part of the clip actually attending to. H3 has **no
  cross-attention** (`grep -c cross_attn comfy/ldm/minimax/model.py` → 0): references
  are packed into the same sequence, so this is a measurement rather than an inference
  from the output. The spans are not guessed either — `model.py` records the layout as
  `("ref_audio", rt * 2)` segments, one per reference in order, and the probe captures
  that list the way Sol-Attn does.

  Wire the probe anywhere in the model chain, and put the Map **in the latent chain
  after the sampler** — it takes the sampler's `latent` and passes it through unchanged.
  That input is never read: it exists so the Map has a dependency. Without one the node
  has no edges at all, ComfyUI is free to schedule it *before* the sampler against a
  recording that does not exist yet, and its output would cache against widget values
  that never change. Feed the latent on to whatever decodes or saves.

  Then read it: one band per reference, **named in the gutter**, with a **time ruler in
  seconds** underneath taken from the latent's own length on the real frame grid. It
  chains any existing attention override, so it coexists with Sol-Attn.

  **The colour is stretched over the data's own range, and the report says what that
  range was.** Raw attention mass is not a 0–1 quantity — a single reference on a long
  clip sits around 0.02–0.11, which against a 0–1 ramp renders as flat dark blue with a
  5× variation completely invisible. Scaling is across the whole map rather than per
  band, so references stay comparable.

  **The report gives the denominator too.** A mean of 0.039 means nothing alone; what it
  has to be read against is the reference's share of the sequence, because attention
  equal to that share is indifference. So each reference prints its key rows, that share,
  and the ratio — `1008 of 37632 key rows (2.68%) — uniform attention would give 0.0268,
  measured 1.46x that`.

  **When it records nothing it says which of four things went wrong** — the probe
  never saw an attention call at all, every call was filtered out by `layers`, the
  layout could not be read, the run carried no references, or there were references but
  no rows of the kind `query_rows` asks for. It also prints the segment kinds it *did*
  see, because what was there instead is usually the answer. Each has a different fix,
  and they used to share one message.

  **It judges per moment, never on the time average.** Under a working binding — one
  reference leading while speaker A talks, the other while B talks — the two averages
  come out equal, so an average-based test would call the best possible result "no
  binding". The report gives the per-moment margin, how many times the lead changes,
  and what share of the clip each reference leads, which separates the three outcomes:
  no reference ever leads, one reference leads the entire clip and never hands over,
  or the lead alternates.

  Queries are pooled to one centroid per 64 rows (sol_attn's own routing granularity,
  ~5e-4 cosine). The **denominator is exact**, streamed over key chunks inside a
  memory budget — pooling the keys as well was tried first and is wrong: logsumexp
  over a key block is near its MAX while a centroid is its MEAN, so a row attending
  one sharp key elsewhere had its tail understated and both references came back at
  ~0.50 when the truth was ~0.005.

  **Attention mass is where the model LOOKED, not what it took.** A row can attend a
  reference heavily and not adopt its timbre, and binding to the WRONG reference looks
  identical here to binding to the right one.
- **MMH3 Size Capped Copy** — a second copy of a finished file under a hard size
  ceiling, for upload limits. Chains off Streaming Save's `file_path`; takes any
  video, not just H3 output. `target_mb` is a **ceiling, never a target**: a file
  already under it is not re-encoded, and the node returns the source path
  unchanged rather than writing a copy. See [Delivery copies](#delivery-copies).
- **MiniMax H3 Trim AV** — drop latents from the head and/or tail, cutting audio and
  masks to match. Note the grid rule **inverts** relative to Concat AV: trimming one
  latent, `5m` keeps the result on grid and `5m+2` takes it off, because there the
  constraint is on the joined *total* rather than the piece being cut.
- **MiniMax H3 Split AV** — pull an AV latent into plain video and audio latents. The
  exact inverse of Pack AV, so carrying stage 1's audio through an upscale ladder is
  something the graph states rather than a discipline you have to remember.
  **`preserve_masks`** (default on) hands each half its own `noise_mask`. Pack AV has
  always had a branch for re-pairing masks, but until 2026-08-22 Split emitted bare
  `{"samples": ...}` dicts, so there was never anything left to re-pair and a
  split/repack silently discarded the pin `use_input_audio` installs to protect a
  supplied track. Turn it off to discard the mask deliberately.
- **MiniMax H3 Pack AV** — pair a video latent with an audio latent. Encoding real
  footage gives two *separate* plain latents (`VAEEncode` + `VAEEncodeAudio`) and
  nothing joins them. Audio is reconciled to `round(frames / 24 * 40)`. This is a
  **modality** join; Concat AV is a **time** join.
- **MiniMax H3 Find Divergence** — measures how many frames a continuation
  reproduces from its source, so the join can be trimmed at frame granularity.
- **MiniMax H3 Concat AV** — join two AV latents on the correct axes (video dim 2,
  audio dim 3), with optional `trim_b_latents` and `carry_masks`.

  `trim_b_latents` is honoured as given, because **no single snap is correct**.
  With `A = 5a+2` and `B = 5b+2`:

  | trim | effect |
  |---|---|
  | `5m` | removes a Seed Overlap **exactly**; the total is `5(a+b)+4−k`, **off grid** |
  | `5m+2` | total lands **on grid**; ~7 frames of overlap stay duplicated |

  `k` cannot be `0` and `2 (mod 5)` at once. If you need both, that is what
  **Join AV** is for — it cuts per frame in pixel space. The node logs which
  property the value you gave it actually gets.

### Latent joins happen in pixel space

Latent concatenation is unsound here. Two on-grid chunks sum to `5(j+k)+4`
latents, never back on the `5j+2` grid, so the VAE's 17-frame causal chunking
misaligns from the join onward and the second half pulses. **Join AV** and
**Find Divergence** therefore work on decoded frames, where granularity is one
frame rather than 17, and audio crossfades in the **waveform** domain — the
DAC/BigVGAN latents do not blend.

> **On `noise_mask`:** core passes the mask to the model as a cond, runs preserved
> rows at the cond timestep, and injects preserved regions at H3's cond timestep
> (`VISUAL_COND_TIMESTEP`, 0.999), rescaling the audio half for `audio_scale`. So the
> model knows a region is fixed while it predicts the rest. Per-row masking binarises
> at 0.5 for timestep purposes: an intermediate mask value only blends the latent.
> History in [`core-changes.md`](core-changes.md).

For **audio-driven video**, use an audio reference with the `[audio reuse]` task
type and the `fully_copy` marker, not a mask. That is a trained capability.

### Refine vs regenerate

Two ways to get from a 768p stage 1 to 2K, and **the upscale question only exists in
one of them**:

| | refine | regenerate |
|---|---|---|
| node | **Chunked Pixel Upscale** → sampler | **Regenerate-2K Reference** |
| stage 2 starts from | the upscaled stage-1 latent | an **empty** 2K latent |
| stage 1 arrives as | the thing being denoised | `minimax_refs`, never denoised |
| cost | partial denoise | full sampling at 2K |
| drift from stage 1 | low | possible |
| distribution | off — H3 wasn't trained for this | the trained shape |

Regenerate needs no upscale at all: H3 has no cross-attention, so the reference rows
are attended directly at every layer and the 2K target is generated fresh against
them. Refine is cheaper and holds tighter to stage 1, and that is where an upscale
has to happen.

**Do not upscale in latent space for it.** A 24-channel latent at /16 is not a
spatially smooth signal — interpolating between latent positions gives the decoder
codes it never saw, which is the blocking people mean by "chunky latent upscale".
`downscale_video_latent` is bilinear, but it only ever touches *reference* slices,
which are never denoised; approximate context is fine, approximate content is not.

Chunked Pixel Upscale therefore goes through pixels, and chunks the whole way across
so length is not a constraint. If you are decoding stage 1 anyway for a preview, the
expensive half of the round trip is already paid — only the re-encode is new.

**Stage scales are not integers.** `Regenerate-2K Dimensions` guarantees an exact
*aspect*, not an integer factor. At 16:9 a `target_long_edge` of 2048 is **1.5x**;
**2688** is exactly 2x. Stage 1 is 6 of that aspect's 224x128 units, so integer
scales land on multiples of 6.

### Delivery copies

**Streaming Save's `crf` cannot hit a file size.** CRF targets *quality* — it
spends whatever bitrate the picture needs and the file lands where it lands. That
is the right setting for a master and the wrong one for an upload limit, so a copy
under a fixed ceiling is a **second encode**, not a knob on the first. Wire
`file_path` into **Size Capped Copy**; the master is read, never modified.

It measures the duration, solves the video bitrate for the budget, and two-passes
libx264 at it — landing within a percent or two, biased under. The budget is in
**MiB**, because upload limits are quoted in binary megabytes and at a 100 "MB"
ceiling the two differ by 5 MB.

**`max_height` is not optional past a few minutes.** The budget is duration-driven,
and long videos run out of bitrate before they run out of pixels:

| Length | Video budget at 95 MiB | Sensible height |
|---|---|---|
| 2 min | ~6,300 kbps | native |
| 5 min | ~2,450 kbps | 1080 |
| 20 min | ~520 kbps | 720 |
| 1 hr | ~90 kbps | split the file |

(at the default `audio_kbps` of 128, which comes off the top before video is solved)

At 2K, 520 kbps is mush; at 720p it is watchable. A source already shorter than the
cap is never upscaled into it. Under 150 kbps the node warns rather than pretending
the result is usable.

### Model
- **MiniMax H3 Phrase Attention** — experimental. Bias the video model's attention toward
  one phrase of the prompt; the H3 equivalent of `(phrase:1.3)`. Works with a prompt
  per chunk. `value_scale` removes or subtracts the phrase's content. See [`phrase-attention.md`](phrase-attention.md).
- **MiniMax H3 Carry Attention** — experimental. The same bias on the frames a Looping
  Sampler chunk carries from the previous one (`carry="mask"`): negative makes the chunk
  read its carry less. `reference_bias` and `reference_audio_bias` do the same for the references. See [`phrase-attention.md`](phrase-attention.md#carry-attention).
- **MMH3 AdaLN Reference Patch** — take AdaLN modulation from another H3 checkpoint,
  per block. `fl2va` and `ref2va` are the same model *except* for AdaLN: attention,
  MLP, `condition_proj`, the patch projections and the output heads all measure at
  cosine 0.999+, while every `adaln_proj` lands between −0.42 and −0.91. AdaLN is
  where reference conditioning enters the residual stream, so that one component is
  the whole difference between a checkpoint that can condition on a reference and one
  that cannot. Reads only the `adaln_proj` tensors from the source — ~100 MB of a
  20 GB file.

  `blocks` takes ranges and lists (`25-49`, `0-2,40-49`, `-1`), so the published
  hybrid checkpoints are widget values rather than downloads, and non-contiguous sets
  are possible. `final_layer` covers the last modulation before the output heads
  (cosine −0.830), which those hybrids leave alone.

  **There is no strength slider on purpose.** The two AdaLNs are anti-correlated at
  near-equal norms, so a blend cancels instead of mixing — at 0.5 the modulation drops
  to 32% of either endpoint and most of the conditioning routing switches off. Per
  block it is one side or the other. Per-row and per-term controls are absent too: the
  difference is uniform across all three modality rows and all six terms, so there is
  nothing to isolate.

### Util
- **MMH3 Latent Info** — shapes, frame count, audio-length mismatch, grid
  alignment, mask presence.
- **MMH3 Motion Overload** — which latent time tokens of a **rendered** clip carry
  more motion than one token can represent. Four of every five tokens span four
  pixel frames, so when fast motion needs four distinct poses in those frames, one
  token cannot hold them and the decode smears; the poses were never generated, which
  is why the artefact does not answer to steps or resolution. Third difference along
  the token axis, phase-normalised for the `(1,4,4,4,4)` grid, reported as hot spans
  in tokens, frames and seconds.

  **Read the contrast ratios, not the spans.** A quantile threshold marks a fixed
  share of tokens whatever it is handed, so the profile *ranks* and does not *detect*.
  `hot / cold mean` near 1.00 means the cut separated nothing, and a flat profile is
  reported as having no variation rather than as infinite separation. This is a
  measurement, not a fix — it tells you whether the footage has the problem before
  anything gets built on the answer.
- **MMH3 Cond Set Spread** — spread a cond_set's N prompts across a windowed
  generation, so each window gets the one written for it. Regions are cut per window
  midpoint; guess the prompt count low and windows share a prompt, guess high and the
  last prompts are never reached. **MMH3 Window Plan** tells you the number.
- **MMH3 Reframe Pads** — pick a target aspect and get the four **signed** edges for
  Outpaint Latent. `extend` grows to reach it, `crop` cuts, `balanced` does both.
  Snapped to the canvas multiple, so what it emits is what outpaint will honour.
- **MMH3 Upscale Ladder** — an aspect and a target long edge in, a ladder of
  `width_N`/`height_N` out, every rung on the canvas grid. For staged upscales,
  so the stage sizes agree by construction rather than by arithmetic you redo.
- **MMH3 Regenerate-2K Dimensions** — the two stages of a 768p → 2K pass.
  **Stage 1 is not a choice**: it reproduces core's `adapt_canvas`, because that is
  what H3-Base emits whatever you ask for, and sizing it any other way makes stage 2
  an upscale of something never rendered. Stage 2 is an integer multiple of stage 1's
  on-grid unit, so the aspect is exact — rounding each axis to 32 instead puts 16:9 at
  2048x1184 (1.7297), and that squeeze is in every frame. The label says when the
  requested long edge could not be honoured. Every ratio is tabulated in
  [`regenerate-2k.md`](regenerate-2k.md).

Calculators follow the LTXAVTools convention — concise typed outputs plus a short
`label`, flat category.

- **MMH3 Frame Calculator** — seconds in. → `frame_count`, `latent_frames`,
  `audio_latent_frames`, `actual_seconds`. `rounding` is nearest / up / down.
- **MMH3 Dimension Calculator** — → `width`, `height`, `width_ref`, `height_ref`,
  `label`. Where `LTXDimensionCalculator` emitted a fixed `width_half`/`height_half`
  pair for its two-stage pipeline, H3 has no second stage — the secondary pair is
  the **reference** size, set by `downscale_factor` and snapped to a factor the
  patch grid supports.

#### Achievable durations

Frames must be `17j+5` at 24fps, so durations are discrete. Solving
`24s ≡ 5 (mod 17)` gives `s ≡ 8 (mod 17)` — **8.000s is the only whole-second
duration in the 4–15s range**:

| asked | frames | actual | drift |
|---|---|---|---|
| 4s | 90 | 3.750s | −0.250 |
| 5s | 124 | 5.167s | +0.167 |
| 6s | 141 | 5.875s | −0.125 |
| **8s** | **192** | **8.000s** | **0** |
| 10s | 243 | 10.125s | +0.125 |
| 12s | 294 | 12.250s | +0.250 |
| 15s | 362 | 15.083s | +0.083 |

This matters when chaining: per-chunk drift accumulates against wall-clock, so
plan chunk lengths in frames, not seconds — or use 192-frame chunks, which stay
on whole seconds indefinitely.
- **MMH3 Dimension Calculator** — snaps width/height to
  the 32px grid, reports latent dims and **tokens per latent frame**, and snaps a
  requested reference downscale to a factor the patch grid supports.

#### Valid reference downscale factors

Latent dims are `px/16` and must stay **even** for the 2×2 patch, so a downscale
factor `f` is valid only when `latent/f` is an even integer on both axes — the
divisors of `gcd(latent_h//2, latent_w//2)`:

| canvas | latent | tokens/frame | valid factors |
|---|---|---|---|
| 1344×768 | 84×48 | 1008 | 1, 2, 3, **6** |
| 1024×1024 | 64×64 | 1024 | 1, 2, 4, 8, 16, 32 |
| 1280×704 | 80×44 | 880 | 1, 2 |
| 1152×640 | 72×40 | 720 | 1, 2, 4 |

Note **4× is invalid on the native 1344×768 canvas** (84/4 = 21, odd) and snaps
to 3×. The factor set depends entirely on the aspect ratio.
