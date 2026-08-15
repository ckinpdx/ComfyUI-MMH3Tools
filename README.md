# MMH3Tools

MiniMax H3 latent tooling for ComfyUI — latent-domain conditioning and correct AV
splicing for **chained long-form generation**.

Requires ComfyUI **v0.30.0+** (native H3 support).

## Requirements

Beyond stock ComfyUI, parts of this pack depend on **two upstream PRs that are still
open**, plus one that has since merged and is now simply a minimum ComfyUI version.
Read this before filing a bug — most "it did nothing" reports are a missing diff.

| PR | needed by | without it |
|---|---|---|
| **[#15375](https://github.com/Comfy-Org/ComfyUI/pull/15375)** per-row masking | `MMH3SeedOverlap`, latent outpaint, and **`MMH3LoopingSampler` with `carry="mask"`** — the default | `MMH3SeedOverlap` **refuses to run**. The looping sampler does **not** — a hard mask has no effect at all, so the carry preserves nothing and every chunk starts cold, and an *intermediate* mask value artifacts instead. See the warning below. |
| ~~#15439~~ **merged upstream 2026-08-13** | `MMH3LoopingSampler` with `carry="keyframe"`, and any use of `keyframes` | Nothing to apply — but you need a ComfyUI **newer than `v0.33.0`**. On anything older, both **refuse to run**: stock raises on any anchor that is not first/last. |
| **[#15316](https://github.com/Comfy-Org/ComfyUI/pull/15316)** VRAM reservation | nothing — optional | The minute-long hang when conditioning carries image references. |

> ⚠️ **The one silent failure.** `carry="mask"` is the looping sampler's default and it
> is *not* gated: without #15375 the mask is accepted and ignored, preserved rows still
> run at the generation timestep, and you get seams with no error anywhere. Everything
> else in this pack refuses rather than pretending. If chunks are not carrying, check
> this first.
>
> #15375 is **three** changes, not one — the mask reaching the model as a cond, the
> per-row timesteps, and a `scale_latent_inpaint` override on `MiniMaxH3` that stock
> does not have. The third only matters for *intermediate* mask values, where it shows
> up as artifacting rather than as nothing happening. Full account under
> [Latent joins happen in pixel space](#latent-joins-happen-in-pixel-space).

Both remaining PRs now apply **clean** — #15375 was rebased onto the merged #15439,
so the hand-merge that used to be required is gone. A script that fetches the diffs
fresh is in [`docs/core-changes.md`](docs/core-changes.md); re-fetch rather than
reusing a saved copy, since these get rebased (which is exactly what happened here).

### One runtime patch, applied automatically

`mmh3tools/patch_guide_origin.py` wraps `PackedLayout` at import — **no core edit, and
it survives `git pull`.** ⚠ **Obsolete on current core:** the merged #15439 anchors the
guide correctly by itself, so the wrap's self-test finds nothing to fix and stands down
(`is_applied()` returns False, and the log says so). It stays for anyone on an older
ComfyUI. What follows describes what it does when it *is* needed.

The **draft** #15439 anchored a guide at `text_len`, but the target does not
begin there: references advance a cursor first, so every guide lands *before* the clip
it is meant to anchor — measured at −1 for one image ref, −320 for an audio ref, −321
for both. Nothing errors; the guide just lands in the reference region, and a carried
tail's audio goes early with it.

No PR carries this fix, which is why it is a wrap rather than a diff. The looping
sampler asks `is_applied()` and **refuses** when a chunk carries both a reference and a
keyframe on an unpatched build, rather than rendering a misplaced anchor.

## Why this exists

Three facts about H3 shape everything here:

1. **References are latents that are never denoised.** `PackedLayout` packs them
   with `update=False`, so they are re-injected at every sampling step as pure
   context. There is no shared region between chunks to blend.
2. **The stock reference node takes pixels** and calls `vae.encode()`. In a chain
   the previous chunk is already latent, so that roundtrip is generation loss
   compounding once per hop.
3. **Video and audio latents have different temporal axes.**

   | tensor | shape | temporal dim |
   |---|---|---|
   | video | `[B, 24, T, h, w]` | **2** |
   | audio | `[B, 32, 2, T40]` | **3** (dim 2 is stereo) |

   Generic nested-tensor helpers that assume one shared temporal dim will stack
   audio on its stereo axis — producing 4 channels at unchanged duration instead
   of a longer clip. It fails silently.

## Example workflows

[`workflows/MMH3_I2V_2K.json`](workflows/) — three-stage I2V to 2K. Generate
small, then two low-denoise windowed upscale passes.

The audio is decided **in the first stage** and carried forward; the upscale passes
only refine picture. Both upscale samplers run through **MiniMax H3 Context
Windows**, which at 2K is *faster* than not windowing — five windows of 17 latents
do 44% of the attention work of one pass over 57, and attention dominates at that
sequence length.

Also needs [ComfyUI-LlamaOmni](https://github.com/ckinpdx/ComfyUI-LlamaOmni) for the
prompt-writing step (an omni model transcribes the song's lyrics so the character
lip-syncs), KJNodes, RES4LYF, VideoHelperSuite and rgthree. The prompt nodes are
easy to swap for your own — see the Note on the canvas.

[`workflows/MMH3_Scene_Prompt_Builder.json`](workflows/) — the prompt half on its
own: N chunk prompts written **section by section**, ending at a pipe-separated
string ready for **MMH3 Reference (Multi-Prompt)**. No sampler, no VAE, no weights —
it runs against an LLM server alone, so you can iterate on a film's prompts without
paying for a generation to find out they were wrong.

Three stages, `1 + 1 + N` LLM calls: definitions once, the whole beat sheet once,
then one call per chunk for its shots. See **MMH3 Scene Plan Prompt** below for why
that shape rather than a prompt per chunk.

Every `MMH3WindowPlan` input is derived from a duration rather than typed in —
60s total, 10.7s window, 2s overlap — and the window's `actual_seconds` drives
`seconds_per_chunk` on all three stages **and** the lint, so the writer and the
checker cannot disagree about how long a chunk is.

Needs [ComfyUI-LlamaOmni](https://github.com/ckinpdx/ComfyUI-LlamaOmni) and
ComfyUI-Easy-Use (the for-loop). The model names on the `Llama Connectivity` nodes
are local llama-swap ids — swap them for yours. Two of them matter:
`unload_after` is ON for the one-shot definitions call so its model frees VRAM for
the next, and OFF for beats and shots, which share a model that should stay resident
across every iteration of the loop.

## Nodes

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
  [`docs/regenerate-2k.md`](docs/regenerate-2k.md).
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

  **Fixed in core as of the #15439 merge (2026-08-13).** `extra_conds` used to
  assign `cond_video_latents` from keyframes and then assign it *again* from
  references, so the references won and every keyframe was silently dropped. Core
  concatenates now. On a ComfyUI predating the merge, keyframes and references still
  cannot coexist.

  `frame_index` accepts `0` or `-1` only, because stock `PackedLayout` raises
  *"only first/last keyframe anchors are supported"* and the node refuses rather
  than failing deeper in. MiniMax's guide lists interior anchors as valid and they
  do work; the merged **#15439** removes the restriction in core, and the **Looping
  Sampler** exposes it as `keyframe_indices`.

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

- **MMH3 Cond To Set** — the inverse of Cond Select: wrap an already-encoded
  CONDITIONING as a one-entry cond_set, no text encoder involved. The looping
  sampler requires a cond_set and ignores the guider's conditioning, and every
  other producer of one goes through the CLIP — so a refine pass conditioned by
  a zero-out (no prompt, no encoder anywhere in the graph) had no way to reach
  the sampler without loading 20 GB to tokenize an empty string. `count`
  replicates the same conditioning N times; 1 already covers any chunk count,
  since the sampler reuses the last entry.

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

  This path needs nothing beyond stock ComfyUI. A few nodes on `main` ask for an
  upstream PR and say so; only MONKEYPATCHES live on the **`keyframe-anchors`**
  branch. See [`docs/core-changes.md`](docs/core-changes.md).

### Prompting
- **MMH3 Asset Plan** / **MMH3 Task System Prompt** — build a Context-IR system
  prompt for your own LLM node from the task type (or combination) and the
  assets in play, emitting only the relevant rule blocks. See
  `docs/context-ir-system-prompt.md` for the full spec these are derived from.
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
      Scene Plan (shots, beat_index=i, beat_sheet=...) -> LLM -> shots i
      Replace Section(skeleton, beat i,  "summary")
      Replace Section(     ^  , shots i, "detailed_description")
      Prompt Accumulate -> pipe-separated string -> MMH3 Reference (Multi-Prompt)
  ```

  **MMH3 Prompt Part** is the join between a sheet written all at once and a loop
  rendering one beat per pass: it splits on the same `|` the accumulator and
  multi-prompt use, tolerates the code fences an LLM adds anyway, and past the end
  either repeats the last beat (matching how the looping sampler reuses the last cond)
  or raises, your choice.

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

  `reference_images` tells the definitions stage that attached images ARE the
  subject, that several images are one person from different angles, and that the
  image beats the brief on appearance. Definitions only: the description is written
  once and reused, and re-deriving it per call is how a subject drifts. Without the
  flag a vision model handed pictures and no instruction describes an invented
  character anyway.

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

  The sigma schedule can be **windowed per chunk** with `sampling_start_step` /
  `sampling_end_step` — absolute indices, sliced exactly as core `SplitSigmas` does,
  so a two-pass run is `end N` then `start N` with no arithmetic. `phase2_start_step`
  plus an optional `phase2_sampler` / `phase2_guider` switches solver mid-schedule
  for dual-solver setups. All three carry LTXAVTools' semantics unchanged. See
  [`docs/looping-sampler.md`](docs/looping-sampler.md) — including what is still
  unmeasured.
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
  pulsing. See [`docs/context-windows.md`](docs/context-windows.md).

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
- **MMH3 Window Plan** — resolve the whole schedule up front, in frames. How many
  windows you get is how many prompts to write; whether your window and overlap
  survive snapping is otherwise only knowable by running a generation.

  `context_length` / `context_overlap` are **latents**, for Context Windows.
  `window_frames` / `overlap_frames` are **frames**, for Split Audio to Windows.
  Crossing them re-snaps a latent count as a frame count and the two schedules
  quietly diverge.
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
  Needs **#15375**; refuses without it.
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
  running out of memory. Slower per frame; for long videos only.
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

> **On `noise_mask`:** masks do reach the model — `samplers.py` packs latents
> before sampling and explicitly handles `denoise_mask.is_nested`. Stock is missing
> **three** things, and it is worth separating them, because only the first is
> usually quoted:
>
> 1. **Per-row timesteps.** Preserved rows still run at the generation timestep, so
>    the model gets clean content labelled as noisy and the mask accomplishes nothing.
> 2. **The mask never reaches the model as a cond.** #15375 unpacks it and passes
>    `denoise_mask` / `audio_denoise_mask` through, which is what makes (1) possible.
> 3. **No `scale_latent_inpaint` override on `MiniMaxH3`.** Stock falls back to
>    `BaseModel`'s noise blend; #15375 injects preserved regions at H3's cond timestep
>    (`VISUAL_COND_TIMESTEP`, 0.999) and rescales the audio half for `audio_scale`.
>    Verified against the class directly — stock `MiniMaxH3` has no such method.
>
> (3) is the one that shows up as artifacting, and it is confined to **intermediate**
> mask values: #15375 thresholds ≥0.995 to 1.0 and ≤0.05 to 0.0, so a hard 0/1 mask
> takes the same path either way. That is why the seam noise in 0.72.x tracked
> `feather_latents` and vanished when the feather was removed in 0.73.0 — a feather
> was the only thing in the pack producing intermediate values at
> `overlap_strength=1.0`.
>
> **drozbay's per-row masking fixes all three — upstream PR
> [#15375](https://github.com/Comfy-Org/ComfyUI/pull/15375).** `MMH3SeedOverlap`
> and the outpaint node need it, and refuse to run without it rather than
> appearing to work. Applying an upstream PR is not monkeypatching, which is why
> they live here rather than on `keyframe-anchors` — see
> [`docs/core-changes.md`](docs/core-changes.md).

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
  [`docs/regenerate-2k.md`](docs/regenerate-2k.md).

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

## Carrying content between chunks

On stock ComfyUI there is one channel, and it does not do what its name suggests:

| channel | mechanism | carries | position |
|---|---|---|---|
| `MMH3LatentToRef` | `minimax_refs`, never denoised | identity, voice, motion style | before the clip, contiguously |

Two more need a patched core. `MMH3SeedOverlap` (target latent + `noise_mask`)
needs per-row timestep handling to mean anything -- **#15375**. Positioned anchors on
the clip's own timeline need interior indices and the accumulate fix -- **#15439**,
which the **Looping Sampler** uses as `carry="keyframe"`. Both are upstream PRs
applied to core, not monkeypatches; see [`docs/core-changes.md`](docs/core-changes.md).

**References are positioned.** The layout lays them out from a cursor starting at
`text_len`, a `video`/`video_audio` block advances that cursor by its own temporal
span, and the target uses the cursor's final value as its origin — so a carried tail
sits contiguously immediately before the clip, not floating outside time. What it
costs is *distance*: a 39-frame carry moves target frame 0 from 320 to 385 at
`text_len` 320. Audio is free, though — `FRAME_RESCALE` is 5/3 and `40/24` is 5/3, so
a matched audio tail spans exactly what the video spans and the layout's `max()` is a
no-op.

**On stock, a noise mask pins at the sampler, not the model.** Each step the model
predicts the whole clip and the mask overwrites the pinned region afterwards, so it is
corrected rather than conditioned — it never knows the region is fixed when predicting
the rest. **#15375 changes this**: the mask is passed through as a cond and preserved
rows run at the cond timestep, so the model does know. The distance argument above is
unaffected either way — that is about layout, not masking.

A third channel, **positioned keyframe anchors**, pins a run of consecutive tail
frames on the clip's own timeline at **no distance cost** -- measured, target origin
`text_len + 0` against `text_len + 65` for the same carry as a `video_audio` ref. It
needs **#15439**, and the Looping Sampler's `carry="keyframe"` is it. The
`keyframe-anchors` branch reached the same place with monkeypatches and is superseded
now that core carries the PR. Not yet run against real weights.

## Grid reference

| | relation |
|---|---|
| frames | `17j + 5` |
| video latents | `5j + 2` |
| audio latents | `round(frames / 24 * 40)` |
| trained range | 124–362 frames (~5.2–15.1s) |
| node ceiling | 3600 frames (150s) |

Keep core's **`ModelSamplingMiniMaxH3`** (node id `MiniMaxH3SigmaShift` — it is
searchable under the display name, not the id, and it is a stock ComfyUI node rather
than one of these) at video `12.0` / audio `3.0`, and constant across chunks: the DiT
derives the audio schedule from the video one, so varying it per chunk desynchronises
them.

## Known limitations

- Carried references are **not** registered with the tokenizer, so Qwen3-VL never
  sees them. Don't use `<Video k>` tags for a carried chunk. The DiT still gets the
  latents, so pixel/motion/identity continuity works; only the semantic path is
  skipped. For continuation that's arguably correct — you rarely want the encoder
  re-describing the previous chunk.
- `ref2va` **does** respond to keyframe (`cond`) rows. Two bugs used to sit in the
  way; **both are fixed in core** as of the #15439 merge (2026-08-13). It stops
  `model_base.py` overwriting `cond_video_latents` — it concatenates keyframes-then-refs,
  so refs no longer erase keyframes — and the merged version also anchors the guide on
  the **target origin** rather than on `text_len`, which the draft did not.

  On a core predating the merge the position bug is live: a guide lands `ref_advance`
  units before the clip whenever refs are present — measured at **−1** for one image
  reference and **−320** for a chunk's worth of voice audio. Nothing errors; it just
  anchors into the reference region. `patch_guide_origin.py` corrects that, and
  **stands down by self-test** on a core that no longer needs it. Drift table in
  [`docs/core-changes.md`](docs/core-changes.md).
- Latent-space downscaling is bilinear and approximate.
- Audio seams: the audio VAE is DAC encoder + BigVGAN decoder. Crossfade in the
  **waveform** domain after decode, never in latent space.

## Tests

```bash
cd C:/ComfyUI/custom_nodes/ComfyUI-MMH3Tools
for t in tests/test_*.py; do C:/ComfyUI/venv/Scripts/python.exe "$t" || echo "FAIL $t"; done
```

Plain scripts, no pytest — each prints PASS/FAIL per assertion and exits non-zero on
any failure. They import from `mmh3tools`, so they need ComfyUI's interpreter and
ComfyUI on the path; they never touch weights, a GPU, or the network.

They are here because most of what this pack asserts is **arithmetic that is wrong
silently** — a frame count off the `17j+5` grid, an audio window that drifts a latent
per chunk, a section spliced into the wrong format. None of that raises; it renders,
and looks slightly bad. The tests are the record of which of those have been pinned
down, and several encode a bug that actually shipped. Read them as the honest version
of the claims above.
