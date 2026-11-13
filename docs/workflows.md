# Example workflows

Every graph in [`workflows/`](../workflows/), what it shows, and what it needs beyond
this pack. The short table is in the [README](../README.md#example-workflows).

[`workflows/MMH3_Scene_Prompt_Builder.json`](../workflows/) — the prompt half on its
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

[`workflows/MMH3_Looping_Cinematic.json`](../workflows/) — the looping T2V film
pipeline, and the successor to the old `MMH3_Looping_T2V`. Definitions, beat sheet
and shots are written **section by section** (three **MMH3 Scene Plan Prompt**
stages), then the loop re-details each chunk against the **previous chunk's rendered
output** — the second **MMH3 Prompt Accumulate** carries that continuity forward — so
each clip continues the last instead of restarting. **MMH3 Prompt Lint** guards chunk
length against the window.

[`workflows/MMH3_Looping_Monologue.json`](../workflows/) — the same looping backbone in
talking-head mode: one continuous monologue, absolutely-locked camera, no story arc.
A stress test of whether the looping sampler joins chunks seamlessly when there is no
cut to hide the seam. **MMH3 Chunk Schedule** paces the spoken-word budget per window.

[`workflows/MMH3_Looping_I2V_PromptBuilding.json`](../workflows/) — image-to-video that
builds its own prompts (**MMH3 Load Skill** + an LLM) ahead of a three-window looping
sampler, finishing on a chunked pixel-upscale ladder.

[`workflows/MMH3_Looping_I2V_ManualPrompt.json`](../workflows/) — the same
image-to-video start with the prompt **typed, not generated**: no LLM anywhere in the
graph, just a `PrimitiveStringMultiline` feeding **MMH3 Reference MultiPrompt**. It also
carries **MMH3 Timeline Preview**, sitting after the LoRA so every guider in the graph
inherits it — the generate pass and both refines each push their own timeline. Use it
when you already know the shot and want the prompt-building half out of the way.

It is now the **kitchen-sink example** rather than the minimal one: 90 nodes, wiring up
**MMH3 Embedding Select**, **MMH3 Image List**, **MMH3 Keyframe Planner** and the
**Reference Attention Probe/Map** alongside the ladder, so each has a worked example of
how it is loaded. It expects **KJNodes**, **VideoHelperSuite** and **rgthree**, and the
latent-upscale branch expects the third-party
[Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
— that branch is the only part that will not load without it. There is still **no LLM
anywhere in the graph**, which is the thing that separates it from the PromptBuilding
one.

It is also the fullest example of the **three-stage ladder**. One looping sampler
generates at 192-frame chunks, then two more refine passes each sit behind their own
**MMH3 Chunked Pixel Upscale** (`rtx_vsr`, 2688x1536) at the size **MMH3 Upscale
Ladder** picks. The last pass splits the pair and pins the audio under a **zero**
`SolidMask` before re-packing, so only the picture is resampled and the track that
came out of stage one survives both passes untouched. Finishes on a streaming save
plus a size-capped copy.

[`workflows/MMH3_Looping_RefVideo_Chunked.json`](../workflows/) — ManualPrompt driven by a
**windowed video reference**. `window_ref_video` is on, and `chunk_frames` /
`overlap_frames` come from the **same MMH3 Chunk Schedule the sampler reads**, so
reference window *i* is by construction the span chunk *i* renders. The loader's audio
goes to `ref_video_audios` alongside, cut on the same clock.

Do not type those two numbers here. If they drift from the sampler's, the windows stop
matching the chunks and every chunk conditions on somebody else's footage — no error,
plausible output.

The still on `ref_images` is left wired and stays whole: `<Picture 1>` for identity,
`<Video 1>` for the windowed motion. Unwire it for a pure video reference.

[`workflows/MMH3_Looping_I2V_ControlNet.json`](../workflows/) — the ManualPrompt graph
with a **Fun ControlNet** driving the generate pass. A `ModelPatchLoader` and a control
video feed **MMH3 Apply ControlNet**, which sits in the MODEL chain, so every chunk gets
the control windowed to its own span.

The control sits on the **generate guider's branch only** — the two refine passes
read the model straight off the LoRA and run uncontrolled, which is deliberate: they
work at low denoise off zeroed conditioning, where a control video fights a picture
that already exists.

**Put the checkpoint in `models/model_patches`, not `models/controlnet`.**
`ModelPatchLoader` lists only that folder. On one volume a hardlink serves both paths
without a second copy — the union checkpoint is 2.2 GB.

**MMH3 Chunk Schedule** feeds the reference node's `chunk_frames` and `overlap_frames`
as well as the sampler's. `window_ref_video` ships **off**, so this changes nothing
until you turn it on — it is wired ahead precisely so that turning it on cannot put the
reference windows out of step with the spans the sampler renders.

Needs the union checkpoint,
`minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` (about 2.1 GB; a
3.9 GB `bf16` sits beside it), from
[Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/controlnet).
Raw video goes in: an `AIO_Preprocessor` sits between the loader and the apply node,
defaulting to **DepthAnythingV2** at 768, and its dropdown covers all five conditions
the checkpoint accepts. Set it to `none` if your pass is already rendered. The video
must cover the WHOLE clip either way, since windows are cut from it by frame index. One checkpoint covers all five conditions plus inpainting. It is
**guidance-distilled**, so run it at guidance 1.0 through a BasicGuider rather than CFG.

[`workflows/MMH3_LoopingSampler_MusicVideo.json`](../workflows/) — the music-video
variant: windows are locked to musical beats and lyrics mapped per window, feeding the
looping sampler; full-res and size-capped saves. Its three character references are
carried by a chained **`ImageTensorList`** rather than a batch, so each keeps its own
size — the loaders are `resize: False` and two of the three are crops, so batching was
conforming references 2 and 3 to reference 1's frame.

Its prompts are written by **one** local model across three staged calls — definitions,
then beats, then shots — each with its own `Llama Connectivity` and `Llama Options`.
The graph ships pointing at `qwen3.6-fable-27b-uncensored-vision`, which is a
llama-swap id: swap it for yours. The load is split across three calls deliberately
rather than asking one call to produce every prompt at once, which is also what makes a
smaller model viable — a community report has **Qwen3-VL-4B-Instruct-Q8_0** working
through the same three stages. Contrast **Scene Prompt Builder**, which uses two
different models on purpose.

[`workflows/MMH3_LoopingSampler_Regenerate2K.json`](../workflows/) — the Regenerate-2K
path: a looping 2K pass over an existing render, driven by the dedicated **MMH3
Regenerate 2K Dims** / **Reference** nodes.

### What the example workflows need

Beyond this pack and comfy-core:

| | needed by |
|---|---|
| [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) — `ClownSampler_Beta`, sigma nodes | all but Scene Prompt Builder |
| [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) — `VAELoaderKJ`, `LoadAndResizeImage`, `VRAM_Debug` | all but Scene Prompt Builder |
| [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) — `VHS_LoadVideo`, `VHS_LoadAudioUpload` | all but Scene Prompt Builder |
| [ComfyUI-LlamaOmni](https://github.com/ckinpdx/ComfyUI-LlamaOmni) — the prompt-writing nodes | every workflow that builds its own prompts |
| [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) — `easy forLoopStart` / `forLoopEnd` | the prompt-loop workflows |
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) — Fast Groups Bypasser | MusicVideo, I2V Prompt Building, I2V ControlNet |
| [ComfyUI-MelBandRoFormer](https://github.com/kijai/ComfyUI-MelBandRoFormer) — vocal separation | MusicVideo only |
| [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) — `AIO_Preprocessor` | I2V ControlNet only |

**I2V ControlNet also needs the union checkpoint** in `models/model_patches/`, as
above. The detection itself is handled in the workflow by
`AIO_Preprocessor`, whose dropdown covers all five conditions the checkpoint accepts —
`CannyEdgePreprocessor`, `DepthAnythingV2Preprocessor`, `HEDPreprocessor`,
`M-LSDPreprocessor`, `OpenposePreprocessor`/`DWPreprocessor` — so raw video in is fine.

**Sampling defaults (0.104.1).** The generate pass in the examples runs without a turbo
LoRA: `ModelAttentionBackend` on **comfy kitchen attention**, then `BasicScheduler` beta at
**10 steps**, with the phase-1 `ClownSampler` at eta 0.2 handing over to `linear/euler`.
The turbo LoRA loader is still in each graph, **bypassed**. Re-enable it and set the
scheduler back to 8 steps for the few-step path the examples used before.

Two examples keep the turbo LoRA on: **Regenerate-2K** and **Upscale**. Their 2K and
refine passes run only a few steps and are built around it.

**No example carries Kijai's `SolAttnMiniMax` (Sol-Attn) any more.** It was a speed
override, not pipeline logic, and leaving it in made every graph report a missing node
on installs without it. To use it, insert it between `ModelAttentionBackend` and the LoRA
loader; nothing else in a graph depends on it.

The prompt nodes are easy to swap for your own — see the Note on each canvas.

[`workflows/MMH3_Looping_Upscale.json`](../workflows/) — a refine pass over an **existing
render** rather than a generation. The clip is encoded, run through **MMH3 Chunked
Pixel Upscale** (`rtx_vsr`) at the target **MMH3 Upscale Ladder** picks from an aspect
preset, then **MMH3 Split AV** takes the pair apart so the audio half can be re-packed
under a **zero** `SolidMask` — held, never sampled — while the looping sampler
re-samples only the video at the higher resolution. Finishes on a streaming save plus
a size-capped copy. Distinct from Regenerate-2K, which drives the dedicated
Regenerate-2K nodes instead of the ladder.

[`workflows/MMH3_LoopingSampler_Masking.json`](../workflows/) — masked v2v over an
existing clip. **SAM3 Detect** (core) mattes a subject, KJNodes' **GrowMaskWithBlur**
softens it, and the result drives the looping sampler's `denoise_mask`, so the matted
region re-renders while the rest of the frame is pinned to the encoded source.

It is also the reference for how the two halves are masked **independently**. The
supplied track is protected by `use_input_audio`, whose pin rides through **MMH3 Split
AV** (`preserve_masks`) and back through **MMH3 Pack AV** without the sampler being
told anything about it. Nothing is wired into `audio_denoise_mask`: a subject matte is
white somewhere in every frame and carries no temporal intent, so it must not be
allowed to reach the audio half. Wire a mask there only to freeze or free a **span**.

[`workflows/MMH3_Looping_GuideVideo_StyleTransfer.json`](../workflows/) — v2v style transfer
over a clip of any length, through the sampler's **`guide_video`** (0.105.0). The
source's frames go straight into `guide_video`, so each chunk is guided by its own span
of them, on the output's clock; the source's sound rides through `use_input_audio` on
the Reference node, as in the Masking graph this one is built from. The style image is
`<Picture 1>`, and the prompt uses the LoRA's `style_transfer:` trigger.

It needs a **guide-latent LoRA**. A full-clip guide does nothing useful on the base
model, which is trained only for first/last-frame anchors. The graph ships pointing at
Alisson's `minimax_h3_style_transfer_v1.0_r64.safetensors` on the **ref2va** base, from
[Alissonerdx/Minimax-H3-ComfyUI](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI)
(strength 1.0; about 0.7 if the style starts changing the motion). Windows are 10 s,
because a guide roughly doubles each chunk's sequence. `carry` stays `mask`. Built
from the node's unit tests and Alisson's single-generation settings; it has not yet
been run end to end here.

[`workflows/MMH3_Looping_CarryAttention.json`](../workflows/) — the test graph for
**MiniMax H3 Carry Attention**: a static talking head from one reference image and a
supplied audio track, 32 s in 5 chunks, `carry="mask"`, no anti-burn LoRA. The node sits
in the model chain ahead of the guider and scheduler. As saved, `carry_bias` is -1 and
`reference_bias` is 2. Run it with both at 0 on the same seed for the comparison. See
[`phrase-attention.md`](phrase-attention.md#carry-attention).

[`workflows/MMH3_Looping_PhraseAttention.json`](../workflows/) — the same graph with
**MiniMax H3 Phrase Attention** in place of Carry Attention. The phrase "emphatic head
movements and sharply raised eyebrows" is in the prompt, and the node biases it at 3.
It takes the CLIP for its tokenizer and the same `cond_set` the sampler runs; its
`report` output says where the phrase was found. Fill `schedule` to change the bias per
chunk. Built from the Carry Attention graph and not yet run as saved.

Both graphs need only RES4LYF, KJNodes and VideoHelperSuite beyond this pack. Nodes are
coloured by role: green inputs, red model loaders, yellow generation, purple outputs.

[`workflows/MMH3_Outpaint.json`](../workflows/) — reframing, not generation: a landscape
clip is encoded with **MMH3 Streaming Encode**, extended in the latent by **MMH3
Outpaint Latent**, and **MMH3 Reframe Pads** solves the pad geometry for a 9:16 target.
**MMH3 Context Windows** carries one sampler pass across the whole clip instead of
chunking it through the looping sampler.
