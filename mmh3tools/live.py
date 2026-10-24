"""Live MiniMax H3: render a talking head chunk by chunk, outside a ComfyUI graph.

The public, versioned API for real-time use (it is what the Portravix avatar's face worker runs). It packages what
the Looping Sampler does per chunk -- the carry, the latent colour match, pinned speech -- for a caller that gets
audio one chunk at a time and must hand frames back before the next chunk is due. It runs inside an initialised
ComfyUI (as a library or in the server) and does no process setup of its own; the pack does not import it, so it
changes nothing about the nodes.

    model = load_model("fasth3_omniref_qad_nvfp4_comfy.safetensors")
    vae = load_vae("minimax_h3_video_vae_int8_convrot.safetensors")
    audio_vae = load_vae("minimax_h3_audio_vae_fp32.safetensors")
    cfg = LiveConfig(size=512, window_frames=56, steps=2)
    cond, template = build_conditioning(clip, vae, audio_vae, image, prompt, cfg)   # once; cache it
    live = LiveH3(model, vae, audio_vae, cond, template, cfg)
    frames, timing = live.render(audio)    # live.samples_per_chunk float32 mono samples at cfg.audio_rate

Each render() is one window of `window_frames` frames: the previous window's last 2 latents (5 frames) carried at
`carry_strength`, colour-matched in latent space to the first window, plus new frames; the window's audio encoded
with `audio_context_s` of history to its left and pinned, so the face follows the given speech. It returns the
new frames only, `window_frames - 5` of them. The first render has nothing to carry and is a warm-up: it sets the
colour anchor and the first carry.

Measured 2026-10-23 (FastH3 Omni NVFP4, 512x512, RTX 5090, 2 steps): a 56-frame window renders in ~1.7 s for
2.125 s of new video, a 39-frame window in ~1.36 s for 1.417 s. The recipe (carry 0.6 + latent colour match) is
the one that held colour, identity and detail over a 20 s chain; see docs/looping-sampler.md, Observed.
"""
import dataclasses
import time

import numpy as np
import torch

import comfy.model_management as mm
from comfy.nested_tensor import NestedTensor

from .color_carry import decode_frames, latent_anchor_stats, match_carry_latent
from .common import FPS, LATENTS_PER_GROUP, frames_to_audio_t, pack_av, unpack_av
from .nodes_looping_sampler import _chunk_guider, _chunk_noise, _run_sampling, _strip_guide_keys

CARRY_LATENTS, CARRY_FRAMES = 2, 5   # the smallest carry on H3's grid: one 1-frame token + one 4-frame token


@dataclasses.dataclass
class LiveConfig:
    size: int = 512                   # square output, a multiple of 32
    window_frames: int = 56           # 17j+5 frames per window, 5 of them carried
    carry_strength: float = 0.6       # 1.0 copies the carry exactly; lower lets each window partly redraw it
    color_match: bool = True          # latent colour match of the carry to the first window
    steps: int = 2
    sampler: str = "euler"
    scheduler: str = "simple"
    seed: int = 0
    audio_rate: int = 48000           # input rate; core resamples to the audio VAE's 32 kHz
    audio_context_s: float = 1.0      # left context for the audio encoder

    def __post_init__(self):
        if self.size % 32:
            raise ValueError("LiveConfig: size must be a multiple of 32 (16 px VAE cells, 2x2 patches)")
        if self.window_frames < 22 or (self.window_frames - 5) % 17:
            raise ValueError("LiveConfig: window_frames must be 17j+5 and at least 22 (22, 39, 56, 73...)")
        if (self.window_frames - CARRY_FRAMES) * self.audio_rate % FPS:
            raise ValueError("LiveConfig: audio_rate must give a whole number of samples per frame at 24 fps")

    @property
    def window_latents(self):
        return LATENTS_PER_GROUP * ((self.window_frames - 5) // 17) + 2

    @property
    def new_frames(self):
        return self.window_frames - CARRY_FRAMES

    @property
    def samples_per_chunk(self):
        return self.new_frames * self.audio_rate // FPS

    @property
    def chunk_seconds(self):
        return self.new_frames / FPS


def load_model(unet_name, shift_video=12.0, shift_audio=3.0, attention="comfy kitchen attention"):
    """UNETLoader -> MiniMaxH3SigmaShift -> ModelAttentionBackend, as the tested graphs wire them."""
    import nodes
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    model = nodes.UNETLoader().load_unet(unet_name, "default")[0]
    model = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio)).result[0]
    return ModelAttentionBackend.execute(model, attention).result[0]


def load_vae(name, dtype=torch.float16):
    """A video or audio VAE from models/vae, on the main device, as KJNodes' VAELoaderKJ(main_device, fp16) loads it."""
    import comfy.sd
    import comfy.utils
    import folder_paths
    sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", name), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, device=mm.get_torch_device(), dtype=dtype, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def build_conditioning(clip, vae, audio_vae, image, prompt, config):
    """The face + prompt conditioning and the window's latent template, through MMH3ReferenceMultiPrompt
    (ref_image_size "match"). `image`: [1, H, W, 3] float 0..1. Evicts the text encoder afterwards. Encode once and
    cache the result (torch.save of the pair works); the text encoder is the largest model involved."""
    conds, template = build_conditionings(clip, vae, audio_vae, image, [prompt], config)
    return conds[0], template


def build_conditionings(clip, vae, audio_vae, image, prompts, config):
    """Several prompts for the same face in one text-encoder pass (e.g. a talking and a listening prompt, to pick
    per window with render(cond=...)). Returns (list of conditionings in prompt order, template)."""
    from .nodes_multiprompt import MMH3ReferenceMultiPrompt
    prompts = [p.strip() for p in prompts]
    if any("|" in p for p in prompts) or not all(prompts):
        raise ValueError("build_conditionings: prompts must be non-empty and contain no '|' (the multi-prompt "
                         "separator)")
    cond_set, latent, _count, _windows = MMH3ReferenceMultiPrompt.execute(
        clip, vae, audio_vae, config.size, config.size, config.window_frames, "match", prompts=" | ".join(prompts),
        ref_images=image, unload_text_encoder=True).result
    conds = [_strip_guide_keys(c, "live prompt %d" % i) for i, c in enumerate(cond_set["conds"])]
    template = {k: v for k, v in latent.items() if k not in ("samples", "noise_mask")}
    return conds, template


def _make_sampling(model, cond, config):
    """Guider, sampler, sigmas and noise for every window (cfg 1 through BasicGuider)."""
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise
    return (BasicGuider.execute(model, cond).result[0],
            KSamplerSelect.execute(config.sampler).result[0],
            BasicScheduler.execute(model, config.scheduler, int(config.steps), 1.0).result[0],
            RandomNoise.execute(int(config.seed)).result[0])


def _encode_audio(audio_vae, waveform, rate):
    from comfy_extras.nodes_minimax_h3 import _encode_ref_audio
    return _encode_ref_audio(audio_vae, {"waveform": waveform, "sample_rate": rate})[0]


class LiveH3:
    """One live session: keeps the audio history, the carry and the colour anchor between render() calls."""

    def __init__(self, model, vae, audio_vae, cond, template, config=None):
        self.config = config or LiveConfig()
        self.vae, self.audio_vae, self.cond, self.template = vae, audio_vae, cond, template
        self.guider, self.sampler, self.sigmas, self.noise = _make_sampling(model, cond, self.config)
        self.fps = FPS
        self.frames_per_chunk = self.config.new_frames
        self.samples_per_chunk = self.config.samples_per_chunk
        self.chunk_seconds = self.config.chunk_seconds
        self._lat = self.config.size // 16
        self._history = np.zeros(int(round((self.config.audio_context_s + self.config.window_frames / FPS)
                                           * self.config.audio_rate)), np.float32)
        self._audio_t = frames_to_audio_t(self.config.window_frames)
        self.carry = None
        self.anchor = None
        self.index = 0

    def _audio_latent(self):
        wave = torch.from_numpy(self._history.copy())[None, None].repeat(1, 2, 1)        # [1, 2, L] stereo
        z = _encode_audio(self.audio_vae, wave, self.config.audio_rate)
        return z[..., -self._audio_t:].float().cpu()

    def render(self, audio, cond=None):
        """audio: samples_per_chunk float32 mono at config.audio_rate (padded or cut to fit). cond: conditioning for
        this window only, e.g. a listening prompt while nobody speaks (same face and size; from build_conditionings);
        default the session's. The carry keeps motion continuous across a switch. Returns
        (uint8 RGB frames [frames_per_chunk, size, size, 3], timings in seconds)."""
        c = self.config
        t0 = time.perf_counter()
        audio = np.asarray(audio, np.float32).reshape(-1)
        if len(audio) != self.samples_per_chunk:
            audio = np.pad(audio[:self.samples_per_chunk], (0, max(0, self.samples_per_chunk - len(audio))))
        self._history = np.concatenate([self._history, audio])[-len(self._history):]
        a = self._audio_latent()
        t1 = time.perf_counter()

        v = torch.zeros([1, 24, c.window_latents, self._lat, self._lat])
        vm = torch.ones([1, 1, c.window_latents, self._lat, self._lat])
        if self.carry is not None:
            v[:, :, :CARRY_LATENTS] = self.carry
            vm[:, :, :CARRY_LATENTS] = 1.0 - float(c.carry_strength)
        am = torch.zeros([1, 1, a.shape[2], a.shape[3]])                                # the speech is pinned
        latent = pack_av(self.template, v, a, noise_mask=NestedTensor([vm, am]))
        done = _run_sampling(_chunk_noise(self.noise, self.index),
                             _chunk_guider(self.guider, self.cond if cond is None else cond),
                             self.sampler, self.sigmas, latent, 0, 1000)
        dv, _da = unpack_av(done, "live window")
        dv = dv.float().cpu()
        t2 = time.perf_counter()

        had_carry = self.carry is not None
        tail = dv[:, :, -CARRY_LATENTS:]
        drift = 0.0
        if c.color_match:
            if self.anchor is None:
                self.anchor = latent_anchor_stats(dv)
            tail, drift = match_carry_latent(tail, self.anchor, 1.0)
        self.carry = tail
        px = decode_frames(self.vae, dv)
        px = px[CARRY_FRAMES:] if had_carry else px[-self.frames_per_chunk:]
        frames = px.clamp(0, 1).mul(255).round().byte().cpu().numpy()
        t3 = time.perf_counter()
        self.index += 1
        return frames, {"chunk": self.index, "audio_s": t1 - t0, "sample_s": t2 - t1, "decode_s": t3 - t2,
                        "total_s": t3 - t0, "drift_sd": drift}
