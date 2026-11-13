# MMH3Tools

Long-form video with sound from **MiniMax H3** in ComfyUI: one node that renders a clip
of any length chunk by chunk, and the tools around it for prompts, references, drift,
upscaling and delivery.

## What's in it

### The Looping Sampler

**MiniMax H3 Looping Sampler** fills a whole clip chunk by chunk in one node execution.
The graph is the same size for 4 chunks or 40.

- **The latent you pass in is the finished clip.** The chunk count comes from its
  length. Chunks are written back in place, so there is no join and no trim, and the
  output is exactly the length you asked for.
- **Each chunk carries the end of the last one**, so the picture and the sound continue
  across the join.
- **A prompt per chunk**, from a `cond_set`.
- **Supplied audio is pinned**: a song or a voice track reaches every chunk on the
  right clock.
- **Video-to-video on the same node**: `denoise_mask` holds a region or a span,
  `guide_video` drives edit and style-transfer LoRAs, and **MMH3 Apply ControlNet**
  follows the chunk being rendered.
- **`carry_color_match`** stops colour from cooking over a long chain.

Field guide, with measurements and a symptom-to-lever table:
[`docs/looping-sampler.md`](docs/looping-sampler.md).

Around it:

- **MMH3 Chunk Schedule** solves total length, window and overlap together, so the
  last chunk is not clamped. Say roughly what you want and wire its outputs in.
- **MMH3 Timeline Preview** plays the piece so far at real time, with sound, and
  extends each time a chunk finishes.

### A prompt per chunk, one text-encoder load

**MiniMax H3 Reference (Multi-Prompt)** is the stock reference node with N prompts,
separated by `|`. It encodes them all in one text-encoder load, where a naive graph
swaps the encoder and the model in and out once per chunk. It also takes reference
images as a list (each keeps its own size), cuts a reference video to each chunk's own
span, pins input audio, and schedules H3 text embeddings per chunk.

### Attention controls (experimental)

H3 has no cross-attention: the prompt, the references and the frames sit in one
sequence. These nodes bias how much of it the model reads.

- **MiniMax H3 Carry Attention** makes each chunk read its carried frames less
  (`carry_bias`) and its reference images more (`reference_bias`). It is aimed at the
  over-bake that builds up across chunks. A 173 s static talking head, 27 chunks, held
  with no visible bake or drift at `carry_bias` -1 and `reference_bias` 2 (one run,
  judged by eye). A carry bias of -3 breaks the joins.
- **MiniMax H3 Phrase Attention** makes the model listen harder to one phrase of the
  prompt, or mute it, with a per-chunk schedule. `value_scale` removes or subtracts
  what the phrase contributes.

Details and limits: [`docs/phrase-attention.md`](docs/phrase-attention.md). Each has
an example workflow.

### Prompt writing

- **MMH3 Scene Plan Prompt** builds N chunk prompts section by section: definitions
  once, the whole beat sheet once, then one call per chunk. Chunks written in isolation
  each land their own climax; this decides the arc where all of them are visible.
- **MMH3 Prompt Lint**, **Replace Section**, **Prompt Accumulate** and **Prompt Part**
  are the plumbing for a prompt-writing loop.
- These nodes emit system prompts for your own LLM node. None of them calls an LLM.

### Music video

Forced lyric alignment, beat and energy analysis, lyrics sliced per render window, and
a scene planner whose rules follow the song rather than inventing an arc.
[`docs/music-video.md`](docs/music-video.md).

### Upscale, reframe, deliver

- **Refine or regenerate to 2K**: [`docs/regenerate-2k.md`](docs/regenerate-2k.md).
- **MiniMax H3 Outpaint Latent** and **MMH3 Reframe Pads** change a clip's aspect.
- **MMH3 Streaming Save** exports a long master in bounded RAM, and **MMH3 Size
  Capped Copy** makes a copy under an upload limit.

### Live API

`mmh3tools.live` is a Python API, not a node. It renders a talking head one chunk at a
time, for callers that receive audio as it is spoken. It is what the face worker of
the [Portravix](https://github.com/ckinpdx/portravix) avatar runs.
[`docs/live.md`](docs/live.md).

## Install

Clone into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
git clone https://github.com/ckinpdx/ComfyUI-MMH3Tools
```

The pack has no required Python dependencies. Two music-video nodes need extras:
**MMH3 Forced Align** needs `stable-ts`, and **MMH3 Music Analysis** needs `librosa`.

## Requirements

- **A current ComfyUI.** Verified on `v0.45.0` (2026-11-12) with `comfy-aimdo` 0.6.0
  and `comfy-kitchen` 0.2.41. Everything the pack once needed from upstream is in core,
  so there is nothing to patch. The history is in
  [`docs/core-changes.md`](docs/core-changes.md).
- **MiniMax H3 weights**, loaded the stock way.
- **Fun ControlNet** (only for MMH3 Apply ControlNet): the union checkpoint from
  [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/controlnet),
  in `models/model_patches/`.

## Example workflows

All in [`workflows/`](workflows/). Start with **I2V ManualPrompt** if you have a shot in
mind, or **Cinematic** to have the prompts written for you. Full notes on each, and the
node packs they need, are in [`docs/workflows.md`](docs/workflows.md).

| workflow | what it shows |
|---|---|
| `MMH3_Looping_I2V_ManualPrompt` | Image to video with typed prompts and no LLM. The kitchen-sink example, with a three-stage upscale ladder. |
| `MMH3_Looping_I2V_PromptBuilding` | Image to video that writes its own prompts first. |
| `MMH3_Looping_Cinematic` | Text to video film pipeline: definitions, beats and shots written in stages. |
| `MMH3_Looping_Monologue` | One locked-off talking head. The strictest test of the joins. |
| `MMH3_LoopingSampler_MusicVideo` | Windows locked to the song, lyrics mapped per window. |
| `MMH3_Looping_CarryAttention` | Carry Attention on a long talking head: less attention on the carry, more on the reference image. |
| `MMH3_Looping_PhraseAttention` | The same graph with Phrase Attention biasing one phrase of the prompt. |
| `MMH3_Looping_RefVideo_Chunked` | A reference video cut to each chunk's own span. |
| `MMH3_Looping_I2V_ControlNet` | Fun ControlNet driving the generate pass. |
| `MMH3_LoopingSampler_Masking` | Masked video-to-video: re-render a matted subject, keep the rest. |
| `MMH3_Looping_GuideVideo_StyleTransfer` | Style transfer over a clip of any length, with a guide-latent LoRA. |
| `MMH3_Looping_Upscale` | Refine pass over an existing render. |
| `MMH3_LoopingSampler_Regenerate2K` | Regenerate at 2K against the first pass as a reference. |
| `MMH3_Outpaint` | Reframe a landscape clip to 9:16. |
| `MMH3_Scene_Prompt_Builder` | The prompt half alone: no sampler, no weights. |

Most need [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF),
[KJNodes](https://github.com/kijai/ComfyUI-KJNodes) and
[VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite). The ones
that write prompts need [ComfyUI-LlamaOmni](https://github.com/ckinpdx/ComfyUI-LlamaOmni)
and [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use).

## Nodes

All 67 are under `MMH3Tools/…` in the Add Node menu. The reasoning behind each one is
in [`docs/nodes.md`](docs/nodes.md).

**Sampling**

| node | what it does |
|---|---|
| MiniMax H3 Looping Sampler | Renders a clip of any length chunk by chunk. |
| MiniMax H3 Context Windows | Windowed sampling over one long latent, per modality. [`docs/context-windows.md`](docs/context-windows.md) |
| MMH3 Apply ControlNet | Fun ControlNet that follows the chunk being rendered. |
| MMH3 Timeline Preview | The timeline so far, playing at real time with sound. |
| MMH3 Get Timeline Frames | The preview's frames as an image batch. |

**Schedule and calculators**

| node | what it does |
|---|---|
| MMH3 Chunk Schedule / (Frames) | Solves total, window and overlap together. |
| MMH3 Window Plan | Resolves the whole schedule up front: how many prompts to write. |
| MiniMax H3 Keyframe Planner | End-anchored keyframe indices for a chained run. |
| MMH3 Frame Calculator | Seconds to a valid frame count. |
| MMH3 Dimension Calculator | Sizes on the grid, and valid reference downscales. |
| MMH3 Upscale Ladder | Stage sizes for a staged upscale. |
| MMH3 Regenerate-2K Dimensions | The two stage sizes of a 768p to 2K pass. |

**Conditioning and references**

| node | what it does |
|---|---|
| MiniMax H3 Reference (Multi-Prompt) | N prompts and shared references in one text-encoder load. |
| MMH3 Cond Select / Cond To Set | One entry out of a cond_set; a conditioning into one. |
| MMH3 Cond Set Spread | Spreads a cond_set's prompts across context windows. |
| MMH3 Cond Set Strip Text | Drops the prompt, keeps the references, for refine passes. |
| MMH3 Cond Set From Viggle | Adapts Viggle-Animate's windowed conditioning. |
| Apply H3 RefMods to Cond Set | Injects a RefMod bundle into every chunk. [`docs/refmods.md`](docs/refmods.md) |
| MMH3 Regenerate-2K Reference | The first pass as a per-window reference for a 2K pass. |
| MMH3 Embedding Select | Picks an H3 text embedding and the chunks that carry it. |
| MMH3 Image List | Many reference images as a list, each at its own size. |
| MiniMax H3 Image to Reference / Latent to Reference / Reference from Latent | Append a still or a latent to the references. |
| MiniMax H3 Image Keyframe / Latent Keyframe | Anchor a frame of the chunk to a still or a latent frame. |
| MMH3 Official H3 Tokens | Not needed on current ComfyUI; kept so old graphs load. |

**Model**

| node | what it does |
|---|---|
| MiniMax H3 Carry Attention | Bias attention on carried frames, references and the clip's audio. Experimental. |
| MiniMax H3 Phrase Attention | Bias attention on one phrase of the prompt. Experimental. |
| MMH3 AdaLN Reference Patch | Takes AdaLN modulation from another H3 checkpoint, per block. |

**Prompt**

| node | what it does |
|---|---|
| MMH3 Scene Plan Prompt | Chunk prompts section by section: definitions, beats, shots. |
| MMH3 Music Scene Plan Prompt | The same stages, with rules that follow a song. |
| MMH3 Asset Plan / Task System Prompt | A system prompt for H3's prompt format, from the task and assets. |
| MMH3 Load Skill | Loads a look, typography or experiment block from `styles/`. |
| MiniMax H3 Prompt Accumulate / MMH3 Prompt Part | Build and split a `\|`-separated prompt string in a loop. |
| MMH3 Replace Section | Splices one rewritten section back into a prompt. |
| MMH3 Prompt Lint | Checks a prompt against the format. Reports, never rewrites. |
| MMH3 Window Context | One line saying which span of the song a window covers. |
| MMH3 Music Caption System Prompt / Caption Split / Lyrics Sectionize | Captions and lyrics for MiniMax Music 3. |

**Audio**

| node | what it does |
|---|---|
| MMH3 Forced Align (Lyrics) | Places known lyrics on the timeline. |
| MMH3 Music Analysis | BPM, key, bar grid and energy curve. |
| MMH3 Lyrics to Windows | Slices the alignment per render window. |
| MMH3 Split Audio to Windows | Cuts a track into one clip per window. |
| Whisper to Text (LLM Ready) | Flattens a Whisper alignment into text with time markers. |

**Latent**

| node | what it does |
|---|---|
| MiniMax H3 Pack AV / Split AV | Pair a video latent with an audio latent, and the inverse. |
| MiniMax H3 Trim AV / Concat AV / Join AV | Cut and join clips; Join AV works per frame, in pixels. |
| MiniMax H3 Seed Overlap | Prepends masked overlap latents to a chunk by hand. |
| MMH3 Chunked Pixel Upscale | Latent to a larger latent through pixels, a chunk at a time. |
| MiniMax H3 Outpaint Latent / MMH3 Reframe Pads | Grow or crop the canvas to a new aspect. |
| MiniMax H3 Streaming Encode / MMH3 Streaming Save | Encode and export in bounded RAM. |
| MMH3 Size Capped Copy | A second copy under a hard size ceiling. |

**Diagnostics**

| node | what it does |
|---|---|
| MMH3 Latent Info | Shapes, frame count, grid alignment, mask presence. |
| MMH3 Reference Attention Probe / Map | Which reference each part of the clip attends to. |
| MMH3 Motion Overload | Where a rendered clip has more motion than its tokens can hold. |
| MiniMax H3 Find Divergence | How many frames a continuation reproduces from its source. |

## H3's grid

| | relation |
|---|---|
| frames | `17j + 5` |
| video latents | `5j + 2` |
| audio latents | `round(frames / 24 * 40)` |
| trained range | 124–362 frames (about 5.2–15.1 s) |

Durations are discrete: 8.000 s (192 frames) is the only whole-second length between
4 and 15 s. The calculators and the Chunk Schedule snap for you.

Keep core's **ModelSamplingMiniMaxH3** at video `12.0` / audio `3.0` and constant across
chunks. The model derives the audio schedule from the video one, so varying it per
chunk puts them out of step.

## Status and limits

- **`carry="mask"` is the tested path.** Every example workflow uses it, and every
  measurement in the docs was made on it.
- **`carry="keyframe"` has never generated a clip.** Its plumbing is unit-tested only.
- **Carry Attention and Phrase Attention are experimental**, tested on a handful of
  clips. `reference_audio_bias` and `value_scale` have not been run on real weights.
- **`guide_video` is unit-tested**, not yet run end to end here.
- Carried references are not seen by the text encoder, so don't use `<Video k>` tags
  for a carried chunk.
- Join audio in the waveform after decoding, never in latent space.

## Companion packs

**[ComfyUI-HybridWindows](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows)** by
Jalen Brunson started as a fork of this pack: a Rolling Diffusion take on long H3
generation, with keyframes, resume-from-chunk and V2V flows. It builds on this pack's
window and conditioning helpers, so install both.

## Tests

```bash
for t in tests/test_*.py; do python "$t" || echo "FAIL $t"; done
```

Run them with ComfyUI's Python, from the pack folder. They are plain scripts: each
prints PASS or FAIL per check and exits non-zero on a failure. They need no weights, no
GPU and no network.

## Licence

MIT (see LICENSE). That covers this pack's code only. MiniMax H3's weights, and
everything generated with them, are under the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
Read it, including the territories it covers and its Acceptable Use Policy.
