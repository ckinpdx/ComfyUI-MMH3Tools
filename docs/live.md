# `mmh3tools.live` — streaming a talking head chunk by chunk

`mmh3tools/live.py` is a Python API, not a node. It exists for real-time use: a caller
receives audio as it is spoken (a TTS stream, say) and must hand back video for each
piece before the next one is due. A graph cannot do that — it renders a whole clip — so
this module packages what the Looping Sampler does **per chunk** into an object that keeps
its state between calls.

It runs inside an initialised ComfyUI: in the server, or with ComfyUI used as a library
(the Portravix avatar's face worker does the latter). It does no process setup of its own,
and the pack does not import it, so it changes nothing for the nodes.

## Use

```python
from mmh3tools.live import LiveConfig, LiveH3, build_conditioning, load_model, load_vae

cfg = LiveConfig(size=512, window_frames=56, steps=2, seed=1234)
model = load_model("fasth3_omniref_qad_nvfp4_comfy.safetensors")      # + sigma shift 12/3, kitchen attention
vae = load_vae("minimax_h3_video_vae_int8_convrot.safetensors")
audio_vae = load_vae("minimax_h3_audio_vae_fp32.safetensors")
cond, template = build_conditioning(clip, vae, audio_vae, image, prompt, cfg)   # once; torch.save it
live = LiveH3(model, vae, audio_vae, cond, template, cfg)

live.render(silence)                       # first window: warm-up, colour anchor, first carry
frames, timing = live.render(audio)        # audio: live.samples_per_chunk float32 mono at cfg.audio_rate
```

- `image` is `[1, H, W, 3]` float in 0..1 (what LoadImage gives). It is resized with
  `ref_image_size="match"`.
- `prompt` is one H3 prompt in the usual structured format. It should describe someone
  talking to the camera, without the words: the words come from the audio, and changing
  the prompt per chunk would mean re-running the text encoder.
- `clip` is H3's text encoder (`CLIPLoader`, type `minimax`). It is the largest model
  involved and is evicted after the encode, so build the conditioning once and cache it.
- `render()` returns `(frames, timing)`: uint8 RGB `[frames_per_chunk, size, size, 3]` and
  `audio_s` / `sample_s` / `decode_s` / `total_s` / `drift_sd` (how far the carry had drifted
  from the colour anchor before correction, in anchor standard deviations).

## What one window is

| | 39-frame window | 56-frame window |
|---|---|---|
| video latents | 12 | 17 |
| carried (latents / frames) | 2 / 5 | 2 / 5 |
| new frames returned | 34 (1.417 s) | 51 (2.125 s) |
| audio in per chunk at 48 kHz | 68,000 samples | 102,000 samples |
| audio latents in the window | 65 | 93 |

Per `render()`:

1. The new audio joins a history of `audio_context_s` (1 s) plus one window. The history is
   encoded with the audio VAE (core resamples to 32 kHz) and the window's latents are cut
   from the end, so the encoder always has left context. They are **pinned** (mask 0): the
   face follows the speech.
2. The window's video latent is zero except for the carry, the previous window's last 2
   latents (one 1-frame token, one 4-frame token), masked at `1 - carry_strength`.
3. Sampling runs through the Looping Sampler's own helpers (`_chunk_guider`,
   `_chunk_noise` with seed + window index, `_run_sampling`), cfg 1.
4. The last 2 latents become the next carry, colour-matched in latent space to the first
   window (`first_chunk_latent`, see `color_carry.py`) when `color_match` is on.
5. The whole window is decoded and the 5 carried frames dropped (the first window, which
   carried nothing, returns its last `frames_per_chunk` frames).

## The recipe and why

`carry_strength` 0.6 + `color_match` came out of 20 s chain tests (2026-10-23, FastH3 Omni
NVFP4, 512x512, one face reference): at 1.0 the carry passes each chunk's error on intact
and a static shot cooks (saturation and warmth climb, the face drifts); the colour match
alone holds colour but not identity or detail; 0.6 alone holds the face but still cooks;
both together held all three for 20 s, and the latent match costs nothing measurable. See
[`looping-sampler.md`](looping-sampler.md), §10 Observed.

## Speed (RTX 5090, 512x512, 2 steps, euler/simple)

| window | per chunk | of which | vs real time |
|---|---|---|---|
| 39 frames | ~1.36 s | 2 forwards ~0.35 s each, decode ~0.41 s, audio ~0.10 s | 0.96x |
| 56 frames | ~1.70 s | 2 forwards ~0.43 s each, decode ~0.62 s, audio ~0.10 s | 0.80x |

Measured inside a worker process with dynamic VRAM on (the legacy patcher overflowed the
32 GB card), the Comfy compiler off (it gained nothing: the forwards are compute-bound),
and `gc.freeze()` after warm-up (a full garbage collection of ComfyUI's object graph cost
0.24 s every few chunks without it). Resolution barely matters below 512 and costs from
544 up (2 steps, 39 frames: 0.73 s sampling at 512, 0.89 at 576, 1.11 at 640, 1.36 at
704); decode is ~0.55 s per window up to 640 and jumps between 640 and 672.

## Limits

- Batch 1, square output, one reference image.
- The carry is the smallest the grid allows (5 frames); the window length is the only
  latency/headroom trade: a 56-frame window has more headroom, a 39-frame one starts
  speech sooner.
- A faint colour step can show at window seams (the carry is corrected, the frames before
  it were not).
- Not reproducible bit for bit across processes.
