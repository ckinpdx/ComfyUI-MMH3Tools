# MiniMax H3 Looping Sampler — Field Guide

`MMH3LoopingSampler` renders N chained chunks inside one node execution, carrying
each chunk's tail into the next.

**Status: nothing here has been generated against real weights.** The arithmetic is
tested — grid alignment, index placement, guider handling, the join — and every
number below that describes *structure* is measured on the real `PackedLayout` or
the real latent shapes. Numbers that would describe *quality* (which overlap looks
best, which strength holds lipsync) are not here, because guessing them would be
worse than leaving the gap. Section 11 lists exactly what is still unknown.

---

## 1. Mental model

One node, N chunks, constant graph size.

Driving N chunks from the graph costs a copy of every downstream node per chunk —
sampler, decode, save — and a graph that size is what starts breaking ComfyUI. A
Python loop inside one node costs the same whether it runs 4 chunks or 40.

The price is that **nothing inside the loop can be a graph node**. No LLM call, no
VAE decode you wire yourself, no preview between chunks. Everything the loop needs
must exist before sampling starts. That is what `cond_set` is for: all N prompts,
encoded up front, in one text-encoder load.

So a full pipeline is two phases:

1. **Prompt phase**, in the graph — a for loop (Easy-Use) running your LLM per
   window, accumulating with `MMH3PromptAccumulate`. This has to be a graph loop
   because LLM nodes are graph nodes.

   For a clip meant to be one continuous scene rather than N independent ones, drive
   that loop from `MMH3ScenePlanPrompt`: two calls *before* the loop fix the shared
   sections and the escalation across all N windows, and the loop only writes each
   window's shots. Otherwise every window writes its own complete arc, because in
   isolation that is the only thing it can do.
2. **Render phase**, in this node — `MMH3ReferenceMultiPrompt` splits the
   accumulated string into N conds, and the loop renders them.

**The latent is the whole clip.** You hold a song of known length; you do not know
how many chunks that is, so the total is given and the chunk count is derived. The
schedule comes from `_plan`, the same function `MMH3WindowPlan` and
`MMH3SplitAudioToWindows` use — so chunk N renders the audio window N's prompt was
written against, and `window_count` really *is* the chunk count.

Each chunk also slices its own span of audio out of the master, so a track pinned by
`use_input_audio` reaches every chunk correctly with no extra wiring.

---

## 2. What the sockets actually do

### `guider` — **its positive is ignored**

The guider supplies the **model**, the **cfg**, and the **negative**. Its positive
is replaced every chunk from `cond_set`, so whatever you wire there never reaches
the model.

Wire `MMH3CondSelect(cond_set, index=0)` into it. Nothing else works better; that
choice just makes the graph honest about what it is.

A **Basic Guider** works too — it simply has no negative to carry. Both shapes are
handled; a Basic Guider's `set_conds` takes one argument and has no `"negative"`
key at all, which is a real difference and not a detail.

If you use a CFG guider, note that a negative built without `minimax_refs` gets a
different packed-sequence length from a positive that has them, so the two will not
batch together in `calc_cond_batch`. It still runs, as separate forward passes per
step. Worth knowing before reading anything into the speed.

### `cond_set` — N prompts, one encode

From `MMH3ReferenceMultiPrompt`. If it holds fewer prompts than there are chunks,
**the last one repeats** and the report says so. That is a legitimate way to render
6 chunks from 2 prompts; it is also what an off-by-one in the prompt phase looks
like, so read the report.

### `latent` — the WHOLE clip

The finished length, not one chunk. Chunks are slices of it, written back in place,
so the output is exactly this length. The input dict is never mutated, so wiring it
elsewhere is safe.

### `chunk_frames` / `overlap_frames` — the count is derived

You give the size of a chunk and how much it carries; the chunk **count** falls out
of that and the clip length. Wire the same values `MMH3WindowPlan` gets and the two
schedules are identical. The prompt count is independent — see above.

---

## 3. The two carry routes

Chunks are **slices of one master latent**, written back in place. So a chunk's
first `overlap` latents already hold the previous chunk's output — there is nothing
to prepend and nothing to join.

`carry` decides what the model is told about those latents.

### `mask`

They are masked, so the model conditions on them without denoising them. Needs
**#15375** (per-row masking) — without it a noise mask has no effect at all.

### `keyframe`

They are passed as a **guide anchored at frame 0** — re-injected every step, never
denoised, carrying a multi-step clip *plus its audio* at the same `cond_t`. Needs
**#15439**, and the guide-origin wrap if references ride along.

**Never run.** The construction is unit-tested; no clip has been generated through
it. See §11.

### No join, no trim, no loss

An earlier version of this node allocated each chunk separately and concatenated
them, which forced a `k+2` grid-safe trim and cost ~7 frames of real content per
seam. Filling one master removes all of it: **the output is exactly the length of
the latent you passed in.**

What remains is a genuine choice about mechanism, not cost. A guide is exact — the
same rows re-injected at every step. A masked region is blended by the sampler, and
since #15375 was rebased (2026-08-13) per-row masking carries a **float** per row
rather than a bool, so a partial mask is genuinely partial. Which holds continuity
better is **untested** (§10).

### No feather — removed in 0.73.0

There was a `feather_latents` input: a linear ramp on the video mask over N latents
after the carried region, easing back to full generation rather than stepping at the
seam. **It made the seam noisier, and it is gone.**

Observed 2026-08-13, and the observation stands: setting the feather back to 0
removed a visible seam. **The mechanism first recorded for it does not.** That
account said the per-row timestep `rows_t = 1 − m·σ` and the content blend
`x·m + orig·(1−m)` corresponded only approximately, leaving a band of rows whose
label did not match what they held.

Re-read against core on 2026-08-17, they correspond closely — and since #15375
merged, `scale_latent_inpaint` pre-compensates so every pixel lands at its token's
pooled strength *by construction*. Whatever made that seam, it was not this. The
input is still gone, on the evidence rather than the theory.

It was removed rather than defaulted to 0 and documented, because an input whose only
correct value is its default is a trap. The carried region is pinned hard and the
transition is a step; if the step shows, the lever is `overlap_frames` and
`overlap_strength_video`, not a ramp.

**This shifted every later widget**, so saved workflows built before 0.73.0 will
rebind `sampling_start_step` onward and need re-checking.

---

## 3b. Windowing the schedule

Ported from LTXAVTools' looping sampler, with its semantics unchanged. All three
step numbers are **absolute indices into the incoming sigma schedule**, and all of
them apply **within every chunk** — not across chunks.

### `sampling_start_step` / `sampling_end_step`

The same slicing core `SplitSigmas` does: its first output is `sigmas[:step+1]`, its
second is `sigmas[step:]`, sharing the boundary sigma.

- **`sampling_end_step`** stops after that step and discards the rest, leaving a
  **partially denoised** latent.
- **`sampling_start_step`** skips the steps before it — the incoming latent is
  re-noised to that sigma and finished from there.

Because they are absolute, a two-pass run needs no arithmetic: `end N` on pass 1,
`start N` on pass 2. `0` / `1000` (the defaults) mean "the whole schedule".

An empty window (`start >= end`) raises rather than silently rendering nothing.

**This is not a guide control.** A keyframe guide is registered on the conditioning
and re-injected every step, so it is structural for the whole chunk — releasing it
mid-schedule would mean changing the packed layout between steps, which is not
expressible. To drop a guide you need a separate pass whose conditioning never had
it.

### `phase2_start_step` + `phase2_sampler` / `phase2_guider`

Dual-solver schedules: a heavy solver for the first steps, something cheap for the
rest. At `phase2_start_step` the sampler/guider pair switches, resample-style
continuation. `0` disables it.

`phase2_guider` is optional and falls back to the main guider. When connected, only
its **guidance settings** are used — like the main guider, its positive is replaced
every chunk from the `cond_set`, or the tail of every chunk would render whatever
prompt happened to be wired to it.

`phase2_start_step` is rebased onto whatever window `sampling_start_step` leaves, so
it stays absolute alongside the other two. A cut point outside the window is simply
not a cut.

---

## 4. Keyframes

`keyframes` (an IMAGE batch) + `keyframe_indices` + `vae`. One index per image,
comma separated, negatives counting from the end. Ported from LTXAVTools'
`optional_cond_image_indices`.

### The indices are GLOBAL across the master

Not per chunk. You place a shot where it belongs in the finished clip and the node
resolves which chunk owns it and what the local frame is — the same choice LTXAV's
`_calculate_keyframe_per_tile_indices` makes. Only the arithmetic differs, since
H3's frames-per-latent is `1,4,4,4,4` rather than a uniform scale.

This works because chunk *i*'s local latent 0 sits at master latent `cum_i − trim`,
which is a multiple of 5 in **both** carry modes — `(5a+2)−(5m+2) = 5(a−m)` — so
every chunk stays on phase 0 and `frame_at_latent` is valid on the origins. For a
57-latent template with a 7-latent carry: origins `[0, 50, 100, 150]`.

### Which chunk owns a frame

Consecutive chunks overlap, so a global frame can fall inside two of them. A frame
inside a chunk's carried **head** is trimmed at the join, so anchoring it there
paints a frame nobody sees. Each index goes to the chunk that actually **renders**
it.

Worked example — four 192-frame chunks, 7-latent carry (22 frames):

> Global frame **351** is covered by chunk 1 (spans 170–361) and chunk 2 (spans
> 340–531). In chunk 2 it would be local frame 11, inside the 22-frame head. So it
> goes to **chunk 1, local frame 181**.

The report prints every placement: `keyframe frame 351 -> chunk 1 local frame 181`.
Read it. It is the only way to see that an index landed where you meant.

### Stills are fitted to the target grid, not taken as they arrive

Keyframe rows share the **target's** spatial grid, and `PackedLayout` never checks
this: it reads only the latent's time dim (`vt = video_latent.shape[2]`) and sizes
the segment from the target's `_frame_grid`. So a still at any other resolution
reserves the target's row count while the tensor patchifies to its own — a
1024x1024 still against a 1344x768 target reserves 1008 rows and produces 1024 —
and the disagreement surfaces as a broadcast error deep in the model, naming
nothing.

The node therefore resizes each still to the generation's resolution before
encoding, taking the numbers from the master latent it already built. That is on
purpose rather than raising: a 2–3 stage ladder runs the same still against
different target resolutions, and a resize per stage in the graph is busywork.

Aspect follows `MMH3ImageKeyframe`'s `auto`, which is the stock node's rule:

| index | fit | why |
|---|---|---|
| frame **0** | stretch | the opener establishes the clip's geometry |
| any other | centre crop | it follows geometry already set |

When the aspect already matches — the normal case — both give the same result, and
when the **size** matches exactly nothing is resampled at all. Every resize is
named in the log and in the `report` output:

```
keyframe frame 0 -> chunk 0 local frame 0, resized 6000x3375 -> 1344x768 (stretch)
```

`carry="keyframe"` needs none of this: it slices the previous chunk's own tail, so
its dimensions match by construction.

### Indices with no images attached are inert

`keyframe_indices` set while the `keyframes` input is unplugged is **ignored**, not
an error. A ladder reuses one graph across passes and usually only the first pass
carries anchors, so a live index string with no images is the ordinary state of a
refine pass rather than a mistake. The indices are not even parsed — with nothing
to place, an out-of-range index is not worth stopping for either.

The report says so when it happens: `keyframe_indices ignored: no keyframes
attached`. The reverse — images attached with an empty index string — has always
been ignored the same way.

### What raises rather than being silently absorbed

- an index past the end of the master, or before its start — **when images are
  attached**
- a count mismatch between images and indices — they are **zipped**, so a short
  list would silently drop keyframes
- `keyframes` without a `vae`
- any keyframe at all when **#15439** is not applied

Two things are no longer in this list: a **size** mismatch is fitted and reported,
and `keyframe_indices` without `keyframes` is ignored.

Negatives are resolved here rather than passed through: `PackedLayout` takes a
negative literally, so `cond_t` would fall **below `text_len`**, into the text token
positions.

Images are encoded **once**, not per chunk. Guides are independent of `carry` —
they work with the masked route too — and a chunk's carry guide plus its user
keyframes go on in a single `conditioning_set_values`, since a second call would
replace rather than merge.

### Planning them: `MMH3KeyframePlanner`

Rather than working the indices out by hand, the planner emits an **end-anchored**
set from the same schedule the sampler uses:

```
4 chunks, 57 latents, carry 7, keyframe  ->  0, 191, 361, 531, -1   (5 images)
```

Frame 0 opens the clip; every later index is a chunk's **own last frame**; the final
one is `-1`. So each chunk generates *toward* its destination image, and the next
continues from the arrived state through the ordinary carry. Start-anchoring instead
would put each image in the NEXT chunk and invite a snap at every seam — LTXAVTools'
reasoning, and it holds here.

Under the ownership rule that lands exactly one keyframe per chunk, with chunk 0
taking two (its opening and its end). `count` is how many images the batch needs.

The planner takes the same `total_frames` / `chunk_frames` / `overlap_frames` the
sampler does and runs the same `_plan`, so the two cannot disagree about where a chunk
ends. It has no `carry` input and does not need one: since 0.47.0 chunks are slices of
one master written back in place, so the carry route changes what a chunk is
*conditioned on*, never how long it is or where it ends.

---

## 5. What the node protects you from

Each of these was a real failure, not a hypothetical.

**Stale guide bookkeeping.** `minimax_keyframes` / `minimax_frame_count` are
stripped off incoming conditioning every chunk, in both modes. This node registers
all its own guides; anything arriving pre-registered came from an upstream guide
node or a cond cached from a previous run, and would anchor the chunk to somebody
else's frames. Straight from LTXAVTools, where the same leak had the same cause.

**The shallow-copy guider bug.** `copy.copy` shares `original_conds`, and
`set_conds` assigns into it — so chunk 0 would overwrite the BASE conditioning and
every later chunk would read chunk 0's conds back as "base". The dict is rebound per
chunk. In LTXAVTools the symptom was every chunk getting chunk 0's speaker.

**Identical noise.** Reusing one noise object gives every chunk the same noise,
which reads as the model refusing to advance. The seed is bumped per chunk; chunk 0
keeps the seed you wired.

**Template mutation.** The latent is cloned per chunk, including its masks.
`NestedTensor` has no `.clone()`, only `.unbind()`, so an AV pair has to be taken
apart and rebuilt — a plain `.clone()` would throw.

**Guides landing before the clip.** See below.

---

## 6. Core changes this depends on

| PR | needed for | if missing |
|---|---|---|
| **#15375** | `carry="mask"` | `MMH3SeedOverlap` refuses |
| **#15439** | `carry="keyframe"`, any `keyframes` | the node refuses up front |
| *local correction* | guides **alongside a reference** | the node refuses that chunk |

The third is ours, not upstream. #15439 anchors `cond_t` on `text_len`, but the
target begins at `cursor`, which the refs advance. Measured on the real
`PackedLayout`, guide versus target origin:

| refs attached | drift |
|---|---|
| none | 0 |
| one image ref | **−1** |
| audio / voice ref | **−320** |
| image + audio | **−321** |

Nothing errors — the guide just anchors into the reference region instead of the
clip, and `cond_audio` goes with it, so a carried tail's **audio** lands early too.
It bites precisely the configuration #15439 exists to enable, since the same PR
fixes the `cond_video_latents` clobber so guides and refs can coexist.

`_guide_origin_correct()` probes for the correction by building a layout with one
ref plus one guide and comparing the origins. `carry="keyframe"` refuses only when a
chunk carries **both** a reference and a guide — guides alone are correct on stock
#15439. See [`core-changes.md`](core-changes.md).

---

## 7. Reading the report

```
7 chunks of 142 latents (481 frames) over 3048 frames (127.00s), overlap 7 latents (22 frames)
  keyframe frame 2588 -> chunk 5 local frame 293
  chunk 0: prompt 0, frames 0-480, 0 carried
  chunk 1: prompt 1, frames 459-939, 22 carried
  ...
  chunk 6: prompt 6, frames 2567-3047, 22 carried
master: 897 latents (3048 frames, 127.00s) -- the input length, exactly
```

- **`prompt N`** climbing 0,1,2,3 means the cond_set is advancing. A repeated number
  means you have fewer prompts than chunks.
- **`0 carried frames` on chunk 0 only.** Anywhere else means the carry failed.
- **`master:`** — audio should match the video duration. If it does not, something
  upstream of `ConcatAV` is wrong, not the sampler.
- **`master:`** should say *"the input length, exactly"*. It always will now — chunks
  are written back in place — but it is the one line that proves nothing was lost.

---

## 8. Recipes

**What the shipped workflows actually do**, read off their saved widget values rather
than recommended from tuning — the numbers below are a starting point someone already
ran, not a measured optimum. Order is
`chunk_frames · overlap_frames · carry · video/audio overlap strength · phase2_start_step`.

**Long-form cinematic** — `MMH3_Looping_Cinematic`
192 · 68 · `mask` · 1.0 / 0.8 · phase 2 at step 2 · `use_input_audio` off, so H3
writes the soundtrack. The widest overlap of any shipped graph.

**Talking head / monologue** — `MMH3_Looping_Monologue`
192 · 68 · `mask` · 1.0 / 0.95 · phase 2 at step 2. Same backbone, higher audio
carry. There is no cut to hide the seam here, which is the point: it doubles as the
cleanest test of whether chunks join at all.

**Music video over a pinned track** — `MMH3_LoopingSampler_MusicVideo`
192 · 22 · `mask` · 1.0 / **1.0** · phase 2 at step 2 · `use_input_audio` **on**.
See §11 — the 1.0 audio strength is deliberate here and not the case §10 warns about,
but the distinction is untested.

**Image-to-video, prompts built in-graph** — `MMH3_Looping_I2V_PromptBuilding`
derived · 22 · `mask` · 1.0 / 0.9 · phase 2 at step 2, then a second sampler at 136
frames, finishing on a chunked pixel-upscale ladder.

**Refine an existing render** — `MMH3_Looping_Upscale`
136 · 22 · `mask` · 1.0 / 0.9 · phase 2 **0** (single solver). The audio half is
re-packed under a zero `SolidMask` before sampling, so only the video is resampled
and the original track survives the pass untouched. Since 2026-08-22 there are two
cheaper ways to say the same thing: leave `Split AV`'s `preserve_masks` on and the
pin travels through the split by itself, or wire a black mask into the sampler's
`audio_denoise_mask`.

**Regenerate-2K** — `MMH3_LoopingSampler_Regenerate2K`
derived · 22 · `mask` · pass 1 at 1.0 / 1.0, pass 2 at **0 / 0** · phase 2 at step 2.
The second pass carries nothing across seams; why that is right for a 2K re-pass is
not written down anywhere, including here.

---

## 9. Symptom → lever

| Symptom | Look at |
|---|---|
| every chunk looks like chunk 0 | the conditioning, not the noise — the sampler adds the chunk index to the seed itself. Read the report's `prompt N` per chunk, then the cond_set: one cond, or N near-identical ones, look the same from here |
| every chunk uses the same prompt | fewer prompts than chunks — the report says so |
| seam visible / discontinuous motion | raise `overlap_frames` first. `carry="keyframe"` is the other lever, but it has never been run (§11) — trying it is an experiment, not a fix |
| lipsync drifts across a seam | check master audio matches video in the report. **Not** `overlap_strength_audio` 1.0 — that is the measured-bad end (§10) |
| a keyframe lands in the wrong place | read the placement lines; indices are frames of the WHOLE clip |
| chunk count is not what you expected | it is derived — check `MMH3WindowPlan` with the same three numbers |
| every chunk has the same music | fixed: chunks slice the master's audio. If it persists, the latent is not the whole clip |
| keyframe seems ignored | it may have landed in a trimmed head — the report says which chunk took it |
| node refuses on chunk 1 with a reference | the post-ref origin correction; see §6 |
| audio shorter than video in the master | `ConcatAV` audio drop — fixed in 0.39.0, check your version |
| whole output looks noisy / unfinished | `sampling_end_step` below the schedule length — the report says "PARTIALLY denoised" |
| phase 2 never seems to engage | `phase2_start_step` is 0, or sits outside the `sampling_start/end` window |
| the tail of every chunk drifts off-prompt | not `phase2_guider` — its positive is replaced per chunk. Look at the cond_set |

---

## 10. Observed

- **`overlap_strength_audio`: 0.8–0.95 both sound good; 1.0 is tinny on chunk 2.**
  Measured 2026-08-10 on T2VA runs. 1.0 fully pins the carried audio, so if chunk 2
  sounds thin this is the first thing to change. The default moved 1.0 → **0.9** in
  0.53.1 for that reason; saved workflows keep whatever value they already had, so an
  older graph may still be sitting on 1.0. The tooltip previously asserted that
  lipsync wanted 1.0 — that was a guess, and these runs contradict it.
- **The T2VA carry works.** Same run, `carry="mask"`, three-field format.

## 11. Not yet measured

Everything here is honest about being unknown.

- **`overlap_strength_audio` 1.0 against a PINNED track.** §10's "1.0 is tinny on
  chunk 2" was measured on T2VA, where the audio is generated and the carry is the
  only continuity. With `use_input_audio` on, every chunk slices the same master, so
  full pinning is arguably the intent rather than a risk — but nobody has run the
  comparison. The music-video workflow ships at 1.0 on that reasoning alone.

- **`overlap_strength_audio` below 0.8.** 0.8–0.95 are known good and 1.0 known bad
  (§10); the bottom of the range is untried, and where it stops preserving the carry
  at all is unknown.
- **Which `overlap_frames` is enough.** The trade is context versus waste, and the
  waste is exact (§3) while the context is not.
- **`carry="keyframe"` has never generated a clip.** Not "unmeasured against mask" —
  unrun. The guide construction is unit-tested against a fake sampler (anchored at
  frame 0, multi-step clip plus audio, no mask) and the arithmetic of the join is
  cheaper, but no output has been looked at. Everything else in this document was
  measured on `mask`.
- **`overlap_strength_video` / `_audio` below 1.0.** Per-row masking binarises at
  0.5 for TIMESTEP purposes, so partial strength only blends the latent
  continuously — see `core-changes.md`. What that looks like is untested.
- **How many chunks before drift compounds.** Other packs report photocopy-style
  degradation over chained audio; whether the masked carry avoids that is unknown.
- **Whether guides at interior indices behave** as #15439's author intends — it is a
  draft PR he has flagged as not fully tested.
