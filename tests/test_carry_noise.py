"""MMH3LoopingSampler `carry_noise`: the next chunk sees a noisy carry, the output never does.

The sampler call is stubbed, as in test_looping_sampler.py, by one that reproduces core's
inpainting blend for masked cells (out * m + latent_image * (1 - m)), since that blend is
what the restore has to undo. Under test: off changes nothing; on, the chunk is handed
(1 - n) * carry + n * noise on the carried rows only, with the mask unchanged; the noise is
seeded per chunk; the clean carry is back in the output at full and at partial
overlap_strength_video; and a keyframe carry is refused.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor

from mmh3tools import nodes_looping_sampler as LS
from mmh3tools.nodes_windows import _plan
from mmh3tools.common import frames_to_audio_t, frames_to_latents, latents_to_frames

H = W = 4
fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def clip(frames):
    t = frames_to_latents(frames)
    at = frames_to_audio_t(latents_to_frames(t))
    g = torch.Generator().manual_seed(7)
    return {"samples": NestedTensor([torch.randn([1, 24, t, H, W], generator=g),
                                     torch.randn([1, 32, 2, at], generator=g)])}


class FakeGuider:
    def __init__(self):
        self.original_conds = {"positive": "BASE_POS", "negative": "BASE_NEG"}
    def set_conds(self, positive, negative=None):
        self.original_conds["positive"] = positive


class FakeNoise:
    seed = 1000


CALLS = []
class BlendSampler:
    """x0 = something new everywhere; masked cells then get core's blend with latent_image."""
    def sample(self, noise, guider, sampler, sigmas, latent):
        v, a = latent["samples"].unbind()
        m = latent.get("noise_mask")
        mv = m.unbind()[0] if isinstance(m, NestedTensor) else (m if m is not None else torch.ones_like(v[:, :1]))
        g = torch.Generator().manual_seed(500 + len(CALLS))
        x0 = torch.randn(v.shape, generator=g)
        out_v = x0 * mv + v * (1.0 - mv)
        CALLS.append({"v": v.clone(), "mask": mv.clone(), "x0": x0.clone(), "out": out_v.clone()})
        out = dict(latent)
        out["samples"] = NestedTensor([out_v, a])
        return out, out

LS.SamplerCustomAdvanced = BlendSampler
SIGMAS = torch.linspace(1.0, 0.0, 11)
TOTAL, CHUNK, OV = 481, 124, 22
PLAN = _plan(TOTAL, CHUNK, OV, "standard_static")
N = len(PLAN[4])


def run(n=0.0, strength=1.0, carry="mask"):
    CALLS.clear()
    lat = clip(TOTAL)
    cs = [[[torch.zeros([1, 8]), {"tag": "p%d" % i}]] for i in range(N)]
    out = LS.MMH3LoopingSampler.execute(
        FakeNoise(), FakeGuider(), "S", SIGMAS, {"conds": cs}, lat, CHUNK, OV, carry,
        strength, 1.0, 0, 1000, 0, None, None, None, "", None, carry_noise=n).result
    return out, lat


def spans():
    out, prev_end = [], None
    for i, w in enumerate(PLAN[4]):
        v0, v1 = w.index_list[0], w.index_list[-1] + 1
        out.append((v0, v1, 0 if i == 0 else min(max(0, prev_end - v0), v1 - v0)))
        prev_end = v1
    return out

SP = spans()


print("\n1. off: the chunk is handed the carry exactly")
(_m, n, rep), _lat = run(0.0)
def prev_tail(i):
    r = SP[i][0] - SP[i - 1][0]          # where chunk i starts inside chunk i-1
    return CALLS[i - 1]["out"][:, :, r:r + SP[i][2]]

check("carry rows = previous chunk's output",
      all(torch.equal(CALLS[i]["v"][:, :, :SP[i][2]], prev_tail(i)) for i in range(1, n)), True)
check("no carry-noise line in the report", "carry noise" in rep, False)

print("\n2. on (0.05): only the carried rows are noised, by the flow mix, mask unchanged")
OFF = [dict(c) for c in CALLS]
(_m, n, rep), _lat = run(0.05)
ok_mix, ok_rest, ok_mask = [], [], []
for i in range(1, n):
    v0, _v1, c = SP[i]
    clean = OFF[i]["v"][:, :, :c]
    g = torch.Generator(device="cpu").manual_seed(1000 + LS.CARRY_NOISE_SEED_OFFSET + i)
    eps = torch.randn(clean.shape, generator=g, dtype=torch.float32)
    want = 0.95 * clean + 0.05 * eps
    ok_mix.append(float((CALLS[i]["v"][:, :, :c] - want).abs().max()) < 1e-6)
    ok_rest.append(torch.equal(CALLS[i]["v"][:, :, c:], OFF[i]["v"][:, :, c:]))
    ok_mask.append(torch.equal(CALLS[i]["mask"], OFF[i]["mask"]))
check("carried rows = 0.95 * carry + 0.05 * seeded noise", ok_mix, [True] * (n - 1))
check("rows after the carry untouched", ok_rest, [True] * (n - 1))
check("mask unchanged", ok_mask, [True] * (n - 1))
check("chunk 0 (nothing carried) untouched", torch.equal(CALLS[0]["v"], OFF[0]["v"]), True)
check("report names it", "carry noise 0.05" in rep, True)
check("each chunk draws its own noise",
      not torch.equal(CALLS[1]["v"][:, :, :SP[1][2]] - OFF[1]["v"][:, :, :SP[1][2]],
                      CALLS[2]["v"][:, :, :SP[2][2]] - OFF[2]["v"][:, :, :SP[2][2]]), True)

print("\n3. the output never contains the noise -- full and partial strength")
for strength in (1.0, 0.6):
    (m_off, _n, _r), _ = run(0.0, strength)
    calls_off = [dict(c) for c in CALLS]
    (m_on, _n, _r), _ = run(0.05, strength)
    v_off, _a = m_off["samples"].unbind()
    v_on, _a2 = m_on["samples"].unbind()
    # same x0 per call in both runs, so the restore must reproduce the off run exactly
    check("strength %.1f: output identical to the run without noise" % strength,
          float((v_on - v_off).abs().max()) < 1e-5, True)

print("\n4. refusal")
try:
    run(0.05, carry="keyframe")
    refused = False
except ValueError:
    refused = True
check("carry='keyframe' refused", refused, True)

print("\n%s" % ("ALL PASS" if not fails else "%d FAILED: %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)
