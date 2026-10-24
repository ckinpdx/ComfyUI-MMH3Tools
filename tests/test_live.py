"""mmh3tools.live: the chunk-by-chunk API, without a model.

Sampling, the guider and the audio encode are stubbed. Under test: the geometry LiveConfig derives (and refuses);
what each window hands the sampler (empty first window; then the previous window's last 2 latents carried at
1 - carry_strength, colour-matched or raw); the audio pinned everywhere and encoded from the right span of history;
a fresh seed per window; and what render() returns (new frames only, the carried 5 dropped, uint8).
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import torch

from mmh3tools import live as LIVE
from mmh3tools import color_carry as CC
from mmh3tools.common import frames_to_audio_t, latents_to_frames, unpack_av

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def close(a, b, tol):
    return bool((a - b).abs().max() <= tol)


print("\n1. geometry")
for w, lat, new, spc in ((22, 7, 17, 34000), (39, 12, 34, 68000), (56, 17, 51, 102000)):
    c = LIVE.LiveConfig(window_frames=w)
    check("window %d: latents / new frames / samples per chunk" % w,
          (c.window_latents, c.new_frames, c.samples_per_chunk), (lat, new, spc))
for bad, kw in (("window 40", {"window_frames": 40}), ("window 5", {"window_frames": 5}),
                ("size 500", {"size": 500}), ("44.1 kHz with 51 frames", {"audio_rate": 44100})):
    try:
        LIVE.LiveConfig(**kw)
        check("refused: " + bad, "no error", "ValueError")
    except ValueError:
        check("refused: " + bad, True, True)


class FakeGuider:
    def __init__(self):
        self.original_conds = {"positive": "BASE"}
    def set_conds(self, positive, negative=None):
        self.original_conds["positive"] = positive


class FakeNoise:
    seed = 100


SEEN = []          # (video latent, video mask, audio latent, audio mask, noise seed) per window
OUT = []           # what each fake sampling returned (video)
def fake_run_sampling(noise, guider, sampler, sigmas, latent, start, end):
    v, a = unpack_av(latent, "window")
    vm, am = latent["noise_mask"].unbind()
    SEEN.append((v.clone(), vm.clone(), a.clone(), am.clone(), noise.seed))
    out = torch.randn(v.shape, generator=torch.Generator().manual_seed(len(SEEN))) * (1 + len(SEEN))
    OUT.append(out)
    return {"samples": __import__("comfy.nested_tensor", fromlist=["NestedTensor"]).NestedTensor([out, a])}


AUDIO_CALLS = []
def fake_encode(audio_vae, waveform, rate):
    AUDIO_CALLS.append((tuple(waveform.shape), rate, float(waveform[0, 0, -1])))
    t = int(round(waveform.shape[-1] / rate * 40))
    return torch.arange(t, dtype=torch.float32).view(1, 1, 1, t).expand(1, 32, 2, t).clone()


class FakeVae:
    def __init__(self):
        self.decodes = []
    def decode(self, z):
        t = int(z.shape[2])
        self.decodes.append(t)
        frames = latents_to_frames(t)
        idx = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1, 1) / frames
        return idx.expand(1, frames, 32, 32, 3).clone()


LIVE._make_sampling = lambda model, cond, config: (FakeGuider(), "S", torch.linspace(1, 0, 3), FakeNoise())
LIVE._run_sampling = fake_run_sampling
LIVE._encode_audio = fake_encode


def session(**kw):
    SEEN.clear(); OUT.clear(); AUDIO_CALLS.clear()
    vae = FakeVae()
    cfg = LIVE.LiveConfig(size=512, window_frames=56, **kw)
    return LIVE.LiveH3("MODEL", vae, "AVAE", [[torch.zeros(1, 8), {}]], {}, cfg), vae, cfg


print("\n2. the first window: nothing carried, speech pinned")
live, vae, cfg = session()
chunk = np.full(cfg.samples_per_chunk, 0.25, np.float32)
frames, timing = live.render(chunk)
v, vm, a, am, seed = SEEN[0]
check("video window shape", tuple(v.shape), (1, 24, 17, 32, 32))
check("first window is all free (mask 1)", bool((vm == 1).all()), True)
check("audio pinned (mask 0) over the whole window", bool((am == 0).all()), True)
check("audio latents = the window's", int(a.shape[-1]), frames_to_audio_t(56))
check("returns the new frames only, as uint8", (frames.shape, frames.dtype), ((51, 32, 32, 3), np.dtype("uint8")))
check("the first window returns its LAST 51 frames", int(frames[0, 0, 0, 0]),
      int(round(5 / 56 * 255)))
check("timings reported", sorted(k for k in timing if k.endswith("_s")),
      ["audio_s", "decode_s", "sample_s", "total_s"])

print("\n3. later windows: the carry")
live.render(chunk)
v1, vm1, _a1, _am1, seed1 = SEEN[1]
check("carried rows at 1 - carry_strength", round(float(vm1[:, :, :2].unique()), 4), 0.4)
check("the rest free", bool((vm1[:, :, 2:] == 1).all()), True)
expect, _ = CC.match_carry_latent(OUT[0][:, :, -2:], CC.latent_anchor_stats(OUT[0]), 1.0)
check("carry = last 2 latents, colour-matched to window 0", close(v1[:, :, :2], expect, 1e-5), True)
check("nothing else pre-filled", bool((v1[:, :, 2:] == 0).all()), True)
check("a new seed per window", (seed, seed1), (100, 101))
f2, _ = live.render(chunk)
check("later windows drop the 5 carried frames", int(f2[0, 0, 0, 0]), int(round(5 / 56 * 255)))
check("every window decoded whole", vae.decodes, [17, 17, 17])

print("\n4. colour match off: the raw tail is carried")
live, vae, cfg = session(color_match=False, carry_strength=0.8)
live.render(chunk); live.render(chunk)
check("raw last 2 latents", close(SEEN[1][0][:, :, :2], OUT[0][:, :, -2:], 0), True)
check("mask at 1 - 0.8", round(float(SEEN[1][1][:, :, :2].unique()), 4), 0.2)

print("\n5. audio: history in, right length out")
live, vae, cfg = session(audio_context_s=1.0)
live.render(np.full(cfg.samples_per_chunk, 0.5, np.float32))
shape, rate, last = AUDIO_CALLS[0]
check("encodes context + window of history, stereo, at the input rate",
      (shape, rate), ((1, 2, int(round((1.0 + 56 / 24) * 48000))), 48000))
check("newest sample last", last, 0.5)
live.render(np.zeros(100, np.float32))           # short input is padded, not refused
check("short audio padded to a full chunk", AUDIO_CALLS[1][0][-1], AUDIO_CALLS[0][0][-1])

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
