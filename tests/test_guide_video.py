"""MMH3LoopingSampler `guide_video`: a source clip windowed per chunk as a GUIDE, without a model.

The sampler call is stubbed, as in test_looping_sampler.py. Under test: every chunk gets
exactly ONE guide, anchored at its frame 0, encoded from the source frames IT renders
(not the clip's opening frames) and on the 17k+5 clip grid; a short or long source is
reported, not fatal; the combinations that would stack two guides on one frame are
refused; and the guide reaches the conditioning even though upstream guides are stripped.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor

from mmh3tools import nodes_looping_sampler as LS
from mmh3tools.nodes_windows import _plan
from mmh3tools.common import (frames_to_audio_t, frames_to_latents, frame_at_latent,
                              latents_to_frames, VAE_SPATIAL)

H = W = 4
PX_H, PX_W = H * VAE_SPATIAL, W * VAE_SPATIAL
fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def clip(frames):
    t = frames_to_latents(frames)
    at = frames_to_audio_t(latents_to_frames(t))
    return {"samples": NestedTensor([torch.zeros([1, 24, t, H, W]), torch.zeros([1, 32, 2, at])])}


def source(frames, h=PX_H, w=PX_W):
    """Frame i is filled with the value i, so an encoded span reveals where it came from."""
    return torch.arange(frames, dtype=torch.float32).reshape(frames, 1, 1, 1).expand(
        frames, h, w, 3).clone()


class FakeGuider:
    def __init__(self):
        self.original_conds = {"positive": "BASE_POS", "negative": "BASE_NEG"}
    def set_conds(self, positive, negative=None):
        self.original_conds["positive"] = positive


class FakeNoise:
    seed = 1000


class FakeVae:
    """Records every encode: frame count, first/last frame value, spatial size."""
    def __init__(self):
        self.calls = []
    def encode(self, px):
        n = int(px.shape[0])
        self.calls.append({"n": n, "first": float(px[0, 0, 0, 0]), "last": float(px[-1, 0, 0, 0]),
                           "hw": (int(px.shape[1]), int(px.shape[2]))})
        return torch.zeros([1, 24, frames_to_latents(n) if n >= 5 else 1, H, W])


PER_LATENT = 32000 // 40
class FakeAudioVae:
    """One latent per 800 samples at 32 kHz, holding that span's first sample."""
    def encode(self, wave):                      # [1, L, 2]
        n = wave.shape[1] // PER_LATENT
        return wave[0, :n * PER_LATENT:PER_LATENT, 0].reshape(1, 1, 1, n).expand(1, 32, 2, n).clone()


SEEN = []
class FakeSampler:
    def sample(self, noise, guider, sampler, sigmas, latent):
        SEEN.append(guider)
        return latent, latent

LS.SamplerCustomAdvanced = FakeSampler
SIGMAS = torch.linspace(1.0, 0.0, 11)
TOTAL, CHUNK, OV = 481, 124, 22


def run(total=TOTAL, chunk=CHUNK, ov=OV, guide=None, carry="mask", kf=None, kf_idx="",
        vae="new", prompt_extra=None, guide_audio=None, audio_vae="new"):
    SEEN.clear()
    vae = FakeVae() if vae == "new" else vae
    n = len(_plan(total, chunk if chunk > 0 else total, ov, "standard_static")[4])
    extra = prompt_extra or {}
    cs = [[[torch.zeros([1, 8]), dict({"tag": "p%d" % i}, **extra)]] for i in range(n)]
    out = LS.MMH3LoopingSampler.execute(
        FakeNoise(), FakeGuider(), "S", SIGMAS, {"conds": cs}, clip(total), chunk, ov, carry,
        1.0, 1.0, 0, 1000, 0, None, None, kf, kf_idx, vae, guide_video=guide,
        guide_audio=guide_audio,
        audio_vae=FakeAudioVae() if audio_vae == "new" else audio_vae).result
    return out, vae


def guides(g):
    return g.raw_conds[0][0][1].get("minimax_keyframes") or []


print("\n1. every chunk is guided by the source frames IT renders")
(out, n, rep), vae = run(guide=source(TOTAL))
spans = [(frame_at_latent(w.index_list[0]), frame_at_latent(w.index_list[-1] + 1))
         for w in _plan(TOTAL, CHUNK, OV, "standard_static")[4]]
check("one encode per chunk", len(vae.calls), n)
check("one guide per chunk", [len(guides(g)) for g in SEEN], [1] * n)
check("each anchored at the chunk's frame 0",
      {k["resolved_frame_index"] for g in SEEN for k in guides(g)}, {0})
check("each encode starts on its chunk's first frame",
      [c["first"] for c in vae.calls], [float(a) for a, _b in spans])
check("...and covers the chunk exactly", [c["n"] for c in vae.calls], [b - a for a, b in spans])
check("every guide clip is on the 17k+5 grid", all((c["n"] - 5) % 17 == 0 for c in vae.calls), True)
check("guide latent length = chunk latent length",
      [int(guides(g)[0]["latent"].shape[2]) for g in SEEN],
      [frames_to_latents(b - a) for a, b in spans])
check("the report names each chunk's guide frames", rep.count("guide frames"), n)

print("\n2. a single chunk gets the whole source")
(out, n1, rep1), vae1 = run(chunk=0, guide=source(TOTAL))
check("one chunk", n1, 1)
check("one encode of the whole clip", [c["n"] for c in vae1.calls], [TOTAL])

print("\n3. a SHORT source: later chunks get what is left, or nothing")
short = spans[2][0] + 40                     # runs out 40 frames into chunk 2
(out, n3, rep3), vae3 = run(guide=source(short))
check("chunks 0-1 fully guided", [c["n"] for c in vae3.calls[:2]], [b - a for a, b in spans[:2]])
check("chunk 2 gets a clip cut to the grid", vae3.calls[2]["n"], LS._guide_clip_frames(40))
check("chunk 2's clip starts where chunk 2 starts", vae3.calls[2]["first"], float(spans[2][0]))
check("chunks past the end get no guide", [len(guides(g)) for g in SEEN[3:]], [0] * (n3 - 3))
check("reported in the summary", "runs out at frame %d" % short in rep3, True)
check("and per chunk", "no guide (the guide video ended)" in rep3, True)

print("\n4. a LONG source: the excess is unused, and said so")
(out, n4, rep4), vae4 = run(guide=source(TOTAL + 100))
check("still one guide per chunk", len(vae4.calls), n4)
check("reported", "the last 100 are unused" in rep4, True)

print("\n5. frames at another size are centre-cropped to the grid")
(out, _n, rep5), vae5 = run(guide=source(TOTAL, h=PX_H * 2, w=PX_W * 3))
check("encoded at the generation's size", {c["hw"] for c in vae5.calls}, {(PX_H, PX_W)})
check("the resize is named", "resized" in rep5, True)

print("\n6. an upstream guide is still stripped; this one is not")
stale = {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": torch.ones([1, 24, 1, H, W])}]}
(out, n6, _r6), _v6 = run(guide=source(TOTAL), prompt_extra=stale)
check("each chunk carries only its own guide", [len(guides(g)) for g in SEEN], [1] * n6)
check("...and it is not the stale one",
      any(float(guides(g)[0]["latent"].max()) == 1.0 for g in SEEN), False)

print("\n7. what is refused")
for label, kw, err in (
        ("no vae", dict(guide=source(TOTAL), vae=None), "vae"),
        ("with carry='keyframe'", dict(guide=source(TOTAL), carry="keyframe"), "carry='keyframe'"),
        ("with keyframes", dict(guide=source(TOTAL), kf=torch.zeros([1, 64, 64, 3]), kf_idx="0"),
         "keyframes")):
    try:
        run(**kw)
        check("refused: " + label, "no error", err)
    except (ValueError, RuntimeError) as e:
        check("refused: " + label, err in str(e), True)

print("\n8. the clip-length cut matches core's Add Guide")
check("cuts", [LS._guide_clip_frames(k) for k in (0, 4, 5, 21, 22, 38, 39, 124)],
      [0, 0, 5, 5, 22, 22, 39, 124])

print("\n9. guide_audio: every chunk is guided by the audio span IT renders")
TOTAL_A = int(clip(TOTAL)["samples"].unbind()[1].shape[-1])
def track(n, channels=2):
    """n latents of audio; the samples of latent j hold the value j."""
    wave = torch.arange(n, dtype=torch.float32).repeat_interleave(PER_LATENT)
    return {"waveform": wave.reshape(1, 1, -1).expand(1, channels, -1).clone(), "sample_rate": 32000}
def audio_guides(g):
    return [k for k in guides(g) if "audio_latent" in k]
(out, n9, rep9), vae9 = run(guide_audio=track(TOTAL_A), vae=None)
a_spans = [(LS._audio_index_at(w.index_list[0], frames_to_latents(TOTAL), TOTAL_A),
            LS._audio_index_at(min(w.index_list[-1] + 1, frames_to_latents(TOTAL)),
                               frames_to_latents(TOTAL), TOTAL_A))
           for w in _plan(TOTAL, CHUNK, OV, "standard_static")[4]]
check("one audio guide per chunk, and no vae needed", [len(guides(g)) for g in SEEN], [1] * n9)
check("anchored at the chunk's frame 0",
      {k["resolved_frame_index"] for g in SEEN for k in guides(g)}, {0})
check("no picture on it", any("latent" in k for g in SEEN for k in guides(g)), False)
check("each starts on its chunk's first audio latent",
      [float(guides(g)[0]["audio_latent"][0, 0, 0, 0]) for g in SEEN], [float(a) for a, _b in a_spans])
check("...and covers the chunk's audio exactly",
      [int(guides(g)[0]["audio_latent"].shape[-1]) for g in SEEN], [b - a for a, b in a_spans])
check("the report names each chunk's guide audio", rep9.count(", guide audio"), n9)

print("\n10. guide_audio beside guide_video, from a longer mono track")
(out, n10, rep10), vae10 = run(guide=source(TOTAL), guide_audio=track(TOTAL_A + 200, channels=1))
check("a picture guide and an audio guide per chunk",
      [(sum("latent" in k for k in guides(g)), len(audio_guides(g))) for g in SEEN], [(1, 1)] * n10)
check("still sliced on the clip's clock",
      [float(audio_guides(g)[0]["audio_latent"][0, 0, 0, 0]) for g in SEEN],
      [float(a) for a, _b in a_spans])

print("\n11. a SHORT guide_audio, and what is refused")
short_a = a_spans[2][0] + 7
(out, n11, rep11), _v = run(guide_audio=track(short_a), vae=None)
check("chunk 2 gets what is left", int(guides(SEEN[2])[0]["audio_latent"].shape[-1]), 7)
check("chunks past the end get none", [len(guides(g)) for g in SEEN[3:]], [0] * (n11 - 3))
check("reported", ("it runs out" in rep11, "no guide audio (it ended)" in rep11), (True, True))
for label, kw, err in (
        ("with carry='keyframe'", dict(guide_audio=track(TOTAL_A), carry="keyframe"), "carry='keyframe'"),
        ("no audio vae", dict(guide_audio=track(TOTAL_A), audio_vae=None), "audio vae")):
    try:
        run(**kw)
        check("refused: " + label, "no error", err)
    except (ValueError, RuntimeError) as e:
        check("refused: " + label, err in str(e), True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
