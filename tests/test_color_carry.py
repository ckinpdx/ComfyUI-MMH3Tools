"""MMH3LoopingSampler `carry_color_match`: the colour-matched carry, without a model.

Under test: the Lab transfer itself (round trip, strength 0 is the identity, a cast frame is pulled
onto the anchor's statistics); and the sampler wiring, with the sampler and VAE stubbed. Off means
no decode or encode at all. On means one anchor decode (chunk 0) plus one decode/encode per carry,
the corrected latents land exactly where the next chunk reads its carry, and a re-encode that comes
back the wrong shape leaves the carry as sampled and says so.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor

from mmh3tools import nodes_looping_sampler as LS
from mmh3tools import color_carry as CC
from mmh3tools.nodes_windows import _plan
from mmh3tools.common import frames_to_audio_t, frames_to_latents, latents_to_frames, VAE_SPATIAL

H = W = 4
PX_H, PX_W = H * VAE_SPATIAL, W * VAE_SPATIAL
MARK = 7.0
fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def close(a, b, tol):
    return bool((a - b).abs().max() <= tol)


print("\n1. the Lab transfer")
g = torch.Generator().manual_seed(0)
# Mid-range, like a lit face. Uniform 0..1 noise sits on the gamut edges, so the transfer clips
# and leaves ~1.7 dE behind; on this it leaves ~0.1.
frames = 0.25 + 0.5 * torch.rand([3, 32, 32, 3], generator=g)
check("rgb -> lab -> rgb round trip", close(CC.lab_to_rgb(CC.rgb_to_lab(frames)), frames, 1e-4), True)
anchor = CC.lab_stats(frames)
check("strength 0 is the identity", close(CC.match_frames(frames, anchor, 0.0), frames, 1e-6), True)
cast = (frames * torch.tensor([1.15, 0.95, 0.8])).clamp(0, 1)     # a warm cast, as chained runs drift
fixed = CC.match_frames(cast, anchor, 1.0)
d_before = float((CC.rgb_to_lab(cast).reshape(-1, 3).mean(0) - anchor[0]).norm())
d_after = float((CC.rgb_to_lab(fixed).reshape(-1, 3).mean(0) - anchor[0]).norm())
check("a cast frame is pulled onto the anchor (dE before > 5, after < 1)",
      (d_before > 5.0, d_after < 1.0), (True, True))
half = CC.match_frames(cast, anchor, 0.5)
check("strength 0.5 is halfway in pixels", close(half, 0.5 * cast + 0.5 * fixed, 1e-5), True)


class FakeGuider:
    def __init__(self):
        self.original_conds = {"positive": "BASE_POS", "negative": "BASE_NEG"}
    def set_conds(self, positive, negative=None):
        self.original_conds["positive"] = positive


class FakeNoise:
    seed = 1000


class FakeVae:
    """Decodes to flat grey frames on the clip grid; encodes to MARK so a corrected span is visible."""
    def __init__(self, bad_shape=False):
        self.decodes, self.encodes, self.bad_shape = [], [], bad_shape
    def decode(self, z):
        t = int(z.shape[2])
        self.decodes.append(t)
        return torch.full([1, latents_to_frames(t), PX_H, PX_W, 3], 0.4)
    def encode(self, px):
        n = int(px.shape[0])
        self.encodes.append(n)
        return torch.full([1, 24, frames_to_latents(n) + (1 if self.bad_shape else 0), H, W], MARK)


SEEN = []
class FakeSampler:
    def sample(self, noise, guider, sampler, sigmas, latent):
        SEEN.append(LS.unpack_av(latent, "seen")[0].clone())
        return latent, latent

LS.SamplerCustomAdvanced = FakeSampler
SIGMAS = torch.linspace(1.0, 0.0, 11)
TOTAL, CHUNK, OV = 481, 124, 22
WINDOWS = _plan(TOTAL, CHUNK, OV, "standard_static")[4]


def clip(frames):
    t = frames_to_latents(frames)
    at = frames_to_audio_t(latents_to_frames(t))
    return {"samples": NestedTensor([torch.zeros([1, 24, t, H, W]), torch.zeros([1, 32, 2, at])])}


def run(mode="first_chunk", vae="new", strength=1.0):
    SEEN.clear()
    vae = FakeVae() if vae == "new" else vae
    n = len(WINDOWS)
    cs = [[[torch.zeros([1, 8]), {"tag": "p%d" % i}]] for i in range(n)]
    out = LS.MMH3LoopingSampler.execute(
        FakeNoise(), FakeGuider(), "S", SIGMAS, {"conds": cs}, clip(TOTAL), CHUNK, OV, "mask",
        1.0, 1.0, 0, 1000, 0, None, None, None, "", vae,
        carry_color_match=mode, carry_color_strength=strength).result
    return out, vae


n = len(WINDOWS)
carries = []
for i in range(1, n):
    prev_end = WINDOWS[i - 1].index_list[-1] + 1
    carries.append(min(max(0, prev_end - WINDOWS[i].index_list[0]),
                       len(WINDOWS[i].index_list)))
whole = [c for c in carries if c % 5 == 2]

print("\n2. off: nothing is decoded or encoded")
(out, n2, rep), vae = run(mode="off")
check("no decodes", vae.decodes, [])
check("no encodes", vae.encodes, [])
check("no colour line in the report", "colour" in rep, False)

print("\n3. first_chunk: one anchor decode, then one decode/encode per carry")
(out, n3, rep), vae = run()
check("decodes: chunk 0 whole, then each whole-clip carry",
      vae.decodes, [len(WINDOWS[0].index_list)] + whole)
check("encodes: each carry as a clip of its own frames", vae.encodes,
      [latents_to_frames(c) for c in whole])
check("each later chunk opens on the corrected carry",
      [bool((SEEN[i][:, :, :carries[i - 1]] == MARK).all()) for i in range(1, n)
       if carries[i - 1] % 5 == 2], [True] * len(whole))
check("...and nothing past the carry was touched",
      [bool((SEEN[i][:, :, carries[i - 1]:] == 0).all()) for i in range(1, n)], [True] * (n - 1))
check("summary names the setting", "carry colour match: first_chunk, strength 1.00" in rep, True)
check("every whole-clip carry is reported matched", rep.count("tail colour-matched"), len(whole))
check("any other carry is reported skipped", rep.count("not a whole clip"), len(carries) - len(whole))

print("\n4. a re-encode of the wrong shape leaves the carry as sampled")
(out, _n, rep4), _v = run(vae=FakeVae(bad_shape=True))
check("no chunk opens on MARK", any(bool((s == MARK).any()) for s in SEEN), False)
check("said so", rep4.count("re-encode changed shape"), len(whole))

print("\n5. refused without a vae")
try:
    run(vae=None)
    check("refused: no vae", "no error", "vae")
except ValueError as e:
    check("refused: no vae", "needs the H3 video vae" in str(e), True)

print("\n6. first_chunk_latent: no VAE, each carried latent frame lands on its token kind's anchor")
g = torch.Generator().manual_seed(1)
lat = clip(TOTAL)
v, a = lat["samples"].unbind()
v = torch.randn(v.shape, generator=g) * torch.linspace(0.5, 2.0, 24).view(1, 24, 1, 1, 1) \
    + torch.linspace(-1, 1, 24).view(1, 24, 1, 1, 1)
v[:, :, WINDOWS[0].index_list[-1] + 1:] *= 1.7                      # every later chunk "drifts"
lat = {"samples": NestedTensor([v, a])}
SEEN.clear()
cs = [[[torch.zeros([1, 8]), {"tag": "p%d" % i}]] for i in range(n)]
(out6, n6, rep6) = LS.MMH3LoopingSampler.execute(
    FakeNoise(), FakeGuider(), "S", SIGMAS, {"conds": cs}, lat, CHUNK, OV, "mask",
    1.0, 1.0, 0, 1000, 0, None, None, None, "", None,
    carry_color_match="first_chunk_latent", carry_color_strength=1.0).result
anchor = CC.latent_anchor_stats(v[:, :, :len(WINDOWS[0].index_list)])
ok = []
for i in range(1, n):
    c = SEEN[i][0, :, :carries[i - 1]].float()
    for t in range(c.shape[1]):
        am, asd = anchor[0][0 if t % 5 == 0 else 1]
        ok.append(close(c[:, t].mean(dim=(1, 2)), am, 1e-3) and close(c[:, t].std(dim=(1, 2), unbiased=False), asd, 1e-3))
check("runs with vae=None, every carried frame on its anchor", all(ok), True)
check("each carry reported, with its drift", rep6.count("tail latent-matched"), n - 1)
check("summary names the latent mode", "matched per latent channel" in rep6, True)
check("strength 0 is the identity",
      close(CC.match_carry_latent(v[:, :, :7], anchor, 0.0)[0], v[:, :, :7], 1e-6), True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
