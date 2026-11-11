"""MMH3LoopingSampler carry='keyframe': the input mask per chunk, and `carry_crossfade`.

The sampler call is stubbed, as in test_looping_sampler.py. Under test: with a latent that
carries a whole-clip noise_mask (a use_input_audio pin), every keyframe-carry chunk gets ITS
slice of that mask, not the whole clip's; without one, no mask at all; the carry goes in as a
guide at frame 0; and the overlap is faded from the previous chunk's frames into the new
chunk's (smoothstep), or replaced outright with carry_crossfade off.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor

from mmh3tools import nodes_looping_sampler as LS
from mmh3tools.nodes_windows import _plan, _audio_index_at
from mmh3tools.common import frames_to_audio_t, frames_to_latents, latents_to_frames

H = W = 4
fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def clip(frames, pin_audio=False):
    t = frames_to_latents(frames)
    at = frames_to_audio_t(latents_to_frames(t))
    g = torch.Generator().manual_seed(7)
    lat = {"samples": NestedTensor([torch.randn([1, 24, t, H, W], generator=g),
                                    torch.randn([1, 32, 2, at], generator=g)])}
    if pin_audio:
        lat["noise_mask"] = NestedTensor([torch.ones([1, 1, t, H, W]),
                                          torch.zeros([1, 1, 2, at])])
    return lat


class FakeGuider:
    def __init__(self):
        self.original_conds = {"positive": "BASE_POS", "negative": "BASE_NEG"}
    def set_conds(self, positive, negative=None):
        self.original_conds["positive"] = positive


class FakeNoise:
    seed = 1000


CALLS = []
class FakeSampler:
    """Every chunk comes back as a constant equal to 10 * (its call number + 1)."""
    def sample(self, noise, guider, sampler, sigmas, latent):
        v, a = latent["samples"].unbind()
        CALLS.append({"mask": latent.get("noise_mask"), "v_shape": tuple(v.shape),
                      "a_shape": tuple(a.shape), "guider": guider})
        out = dict(latent)
        out["samples"] = NestedTensor([torch.full_like(v, 10.0 * len(CALLS)), a])
        return out, out

LS.SamplerCustomAdvanced = FakeSampler
SIGMAS = torch.linspace(1.0, 0.0, 11)
TOTAL, CHUNK, OV = 481, 124, 22
PLAN = _plan(TOTAL, CHUNK, OV, "standard_static")
N = len(PLAN[4])


def run(pin_audio=False, crossfade=True):
    CALLS.clear()
    cs = [[[torch.zeros([1, 8]), {"tag": "p%d" % i}]] for i in range(N)]
    return LS.MMH3LoopingSampler.execute(
        FakeNoise(), FakeGuider(), "S", SIGMAS, {"conds": cs}, clip(TOTAL, pin_audio), CHUNK,
        OV, "keyframe", 1.0, 1.0, 0, 1000, 0, None, None, None, "", None,
        carry_crossfade=crossfade).result


def spans():
    out, prev_end = [], None
    for i, w in enumerate(PLAN[4]):
        v0, v1 = w.index_list[0], w.index_list[-1] + 1
        out.append((v0, v1, 0 if i == 0 else min(max(0, prev_end - v0), v1 - v0)))
        prev_end = v1
    return out

SP = spans()


print("\n1. a pinned track: every chunk gets ITS slice of the mask")
(_o, n, rep) = run(pin_audio=True)
ok_shape, ok_audio_pinned, ok_video_free = [], [], []
for c in CALLS:
    mv, ma = c["mask"].unbind()
    ok_shape.append(mv.shape[2] == c["v_shape"][2] and ma.shape[3] == c["a_shape"][3])
    ok_audio_pinned.append(float(ma.max()) == 0.0)
    ok_video_free.append(float(mv.min()) == 1.0)
check("mask length = chunk length, video and audio", ok_shape, [True] * n)
check("audio stays pinned in every chunk", ok_audio_pinned, [True] * n)
check("video free (the carry is a guide, not a mask)", ok_video_free, [True] * n)
check("chunks after the first carry one guide at frame 0",
      [len(c["guider"].raw_conds[0][0][1].get("minimax_keyframes") or []) for c in CALLS],
      [0] + [1] * (n - 1))

print("\n2. no pinned track: no mask at all")
run(pin_audio=False)
check("no chunk carries a noise_mask", [c["mask"] is None for c in CALLS], [True] * n)

print("\n3. carry_crossfade fades the overlap from the old chunk into the new one")
(out_on, _n, rep_on), = [run(crossfade=True)]
v_on = out_on["samples"].unbind()[0]
fade = LS._smoothstep(SP[1][2])
v0, _v1, c = SP[1]
# chunk 0 wrote 10s, chunk 1 wrote 20s; chunk 2 later overwrote only its own span
old, new = 10.0, 20.0
want = old * (1 - fade) + new * fade
got = v_on[0, 0, v0:v0 + c, 0, 0]
check("chunk 1's overlap = smoothstep fade 10 -> 20", float((got - want).abs().max()) < 1e-5, True)
check("rising, ends excluded", (float(got[0]) > old, float(got[-1]) < new), (True, True))
check("frames after the overlap are the new chunk's",
      float(v_on[0, 0, v0 + c, 0, 0]), new)
check("report says it fades", "faded in from the previous chunk" in rep_on, True)

(out_off, _n, rep_off) = run(crossfade=False)
v_off = out_off["samples"].unbind()[0]
check("off: the overlap is the new chunk's outright",
      float((v_off[0, 0, v0:v0 + c, 0, 0] - new).abs().max()), 0.0)
check("off: report says it replaces", "the new overlap replaces the old" in rep_off, True)

print("\n%s" % ("ALL PASS" if not fails else "%d FAILED: %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)
