"""MMH3LoopingSampler carry='ref', and the reference-shift patch it relies on.

The sampler call is stubbed, as in test_looping_sampler.py: chunk k comes back as the
constant 10 * (k + 1) in BOTH video and audio, so who wrote what is readable off the output.

Under test:
  * the tail span arithmetic (_ref_span): ends at the previous chunk's end, starts on phase 0
  * each chunk after the first gets a video reference of the previous chunk's tail, an audio
    reference that ends at the previous chunk's end, and a one-frame guide at frame 0 holding
    the chunk's own first latent
  * on the LIVE PackedLayout, that audio reference's rows share coordinates with the target's
    audio rows across the overlap, and run back before the target origin
  * a layout without the private key is exactly stock
  * the video overlap is faded or replaced; the overlap's AUDIO stays the previous chunk's
  * ref_carry_audio block / off
  * a clamped last window falls back to the mask carry
  * colour matching corrects the reference and leaves the master as sampled
  * the combinations that are refused
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor
import comfy.ldm.minimax.model as mm

from mmh3tools import nodes_looping_sampler as LS
from mmh3tools import patch_ref_shift
from mmh3tools.nodes_windows import _plan, _audio_index_at
from mmh3tools.common import (frame_at_latent, frames_to_audio_t, frames_to_latents,
                              latents_to_frames)

H = W = 4
KEY = patch_ref_shift.KEY
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
    def sample(self, noise, guider, sampler, sigmas, latent):
        v, a = latent["samples"].unbind()
        CALLS.append({"mask": latent.get("noise_mask"), "v_shape": tuple(v.shape),
                      "a_shape": tuple(a.shape), "cond": guider.raw_conds[0][0][1]})
        out = dict(latent)
        k = 10.0 * len(CALLS)
        out["samples"] = NestedTensor([torch.full_like(v, k), torch.full_like(a, k)])
        return out, out

LS.SamplerCustomAdvanced = FakeSampler
SIGMAS = torch.linspace(1.0, 0.0, 11)


def spans(plan):
    out, prev_end = [], None
    for i, w in enumerate(plan[4]):
        v0, v1 = w.index_list[0], w.index_list[-1] + 1
        out.append((v0, v1, 0 if i == 0 else min(max(0, prev_end - v0), v1 - v0)))
        prev_end = v1
    return out


def run(total, chunk, ov, pin_audio=False, **kw):
    CALLS.clear()
    n = len(_plan(total, chunk, ov, "standard_static")[4])
    cs = [[[torch.zeros([1, 8]), {"tag": "p%d" % i}]] for i in range(n)]
    return LS.MMH3LoopingSampler.execute(
        FakeNoise(), FakeGuider(), "S", SIGMAS, {"conds": cs}, clip(total, pin_audio),
        chunk, ov, "ref", 1.0, 1.0, **kw).result


# A schedule whose every window overlaps its predecessor by exactly the nominal amount,
# and one whose last window is clamped back onto the clip end.
def find(clamped):
    for total in range(260, 1400, 17):
        p = _plan(total, 124, 5, "standard_static")
        sp = spans(p)
        if len(sp) >= 3 and any(c > p[1] for _a, _b, c in sp) == clamped:
            return total, p, sp
    raise SystemExit("no schedule found (clamped=%s)" % clamped)

TOTAL, PLAN, SP = find(False)
OVL = PLAN[1]
TOTAL_T = frames_to_latents(TOTAL)
TOTAL_A = frames_to_audio_t(latents_to_frames(TOTAL_T))
print("schedule: %d frames, %d chunks, overlap %d latents" % (TOTAL, len(SP), OVL))


print("\n0. the patch installed against the live layout class")
check("patch_ref_shift.is_applied()", patch_ref_shift.is_applied(), True)

print("\n1. _ref_span: ends at the previous chunk's end, starts on the chunk's phase")
check("7-latent tail over a 2-latent overlap", LS._ref_span(35, 2, 7), (30, 37))
check("reference shorter than the overlap", LS._ref_span(35, 7, 2), (40, 42))
check("clip does not reach back that far", LS._ref_span(5, 2, 22), (0, 7))
check("every start is on the chunk's phase",
      [(LS._ref_span(v0, c, r)[0] - v0) % 5 for v0, c, r in
       ((35, 2, 7), (35, 7, 2), (5, 2, 22), (70, 7, 12))], [0, 0, 0, 0])

print("\n2. what each chunk is handed (ref_carry_audio shifted, crossfade off)")
out, n, rep = run(TOTAL, 124, 5, carry_crossfade=False)
v_out, a_out = out["samples"].unbind()
check("chunk 0 has no reference and no guide",
      (CALLS[0]["cond"].get("minimax_refs"), CALLS[0]["cond"].get("minimax_keyframes")),
      (None, None))
kinds, tails, ends, guides, lens, masks = [], [], [], [], [], []
for i in range(1, n):
    v0, v1, c = SP[i]
    a0 = _audio_index_at(v0, TOTAL_T, TOTAL_A)
    ra1 = _audio_index_at(v0 + c, TOTAL_T, TOTAL_A)
    refs = CALLS[i]["cond"]["minimax_refs"]
    kf = CALLS[i]["cond"]["minimax_keyframes"]
    kinds.append([r["kind"] for r in refs])
    # the tail is the previous chunk's constant
    tails.append((refs[0]["latent_t"], float(refs[0]["latent"].min()),
                  float(refs[0]["latent"].max())))
    ends.append((refs[1]["ref_audio_t"], refs[1][KEY] == float(ra1 - a0),
                 float(refs[1]["audio_latent"].min())))
    guides.append((len(kf), kf[0]["resolved_frame_index"], kf[0]["latent"].shape[2],
                   float(kf[0]["latent"].max()), "audio_latent" in kf[0]))
    masks.append(CALLS[i]["mask"] is None)
check("a video reference, then an audio reference", kinds, [["video", "audio"]] * (n - 1))
check("video tail: 7 latents of the previous chunk",
      tails, [(7, 10.0 * i, 10.0 * i) for i in range(1, n)])
check("audio tail: 80 latents of the previous chunk, ending at its end",
      ends, [(80, True, 10.0 * i) for i in range(1, n)])
check("one guide at frame 0: the chunk's own first latent, one frame, no audio",
      guides, [(1, 0, 1, 10.0 * i, False) for i in range(1, n)])
check("no mask: the carry is a reference, not a hold", masks, [True] * (n - 1))
check("report names the route", "ref carry: the previous chunk's last 22 frames" in rep, True)

print("\n3. on the live layout the audio reference sits on the chunk's timeline")
i = 1
v0, v1, c = SP[i]
a0 = _audio_index_at(v0, TOTAL_T, TOTAL_A)
a1 = _audio_index_at(v1, TOTAL_T, TOTAL_A)
ca = _audio_index_at(v0 + c, TOTAL_T, TOTAL_A) - a0
refs = CALLS[i]["cond"]["minimax_refs"]
kf = CALLS[i]["cond"]["minimax_keyframes"]
lay = mm.PackedLayout(8, v1 - v0, H, W, a1 - a0, keyframes=kf, refs=refs)
seg = {}
for a, b, k in lay.segments:
    seg.setdefault(k, (a, b))
t = lay.position_ids[:, 0]
tgt = t[seg["audio"][0]:seg["audio"][1]]
ref = t[seg["ref_audio"][0]:seg["ref_audio"][1]]
R = refs[1]["ref_audio_t"]
check("overlap rows share the target's coordinates",
      float((ref[R - ca:R] - tgt[:ca]).abs().max()) < 1e-9, True)
check("and the rest runs back, one latent per step, before the target origin",
      (float((ref[1:R] - ref[:R - 1] - 1.0).abs().max()) < 1e-9,
       float(ref[0]) == float(tgt[0]) + ca - R), (True, True))
check("it never reaches the text positions", float(ref[0]) >= 8.0, True)
check("the guide sits on the target origin",
      float(t[seg["cond"][0]]), float(t[seg["video"][0]]))
stock = mm.PackedLayout(8, v1 - v0, H, W, a1 - a0, keyframes=kf,
                        refs=[refs[0], {k: v for k, v in refs[1].items() if k != KEY}])
check("without the key the same blocks lay out as stock: audio ends at the origin",
      float(stock.position_ids[seg["ref_audio"][0] + R - 1, 0]) + 1.0,
      float(stock.position_ids[seg["audio"][0], 0]))
check("the patch moved only the audio reference's rows",
      int((lay.position_ids[:, 0] != stock.position_ids[:, 0]).sum()), 2 * R)

print("\n4. the output: video overlap replaced or faded, audio overlap kept")
v0, v1, c = SP[1]
check("crossfade off: the overlap is the new chunk's",
      float((v_out[0, 0, v0:v0 + c, 0, 0] - 20.0).abs().max()), 0.0)
check("the overlap's audio is still the previous chunk's",
      (float(a_out[0, 0, 0, a0:a0 + ca].min()), float(a_out[0, 0, 0, a0:a0 + ca].max())),
      (10.0, 10.0))
check("and the new chunk's from there on", float(a_out[0, 0, 0, a0 + ca]), 20.0)
out_f, _n, rep_f = run(TOTAL, 124, 5, carry_crossfade=True)
got = out_f["samples"].unbind()[0][0, 0, v0:v0 + c, 0, 0]
fade = LS._smoothstep(c)
check("crossfade on: smoothstep 10 -> 20 across the overlap",
      float((got - (10.0 * (1 - fade) + 20.0 * fade)).abs().max()) < 1e-5, True)

print("\n5. ref_carry_audio block / off, and a pinned track")
run(TOTAL, 124, 5, ref_carry_audio="block")
refs = CALLS[1]["cond"]["minimax_refs"]
r0, r1 = LS._ref_span(SP[1][0], SP[1][2], 7)
want_at = _audio_index_at(r1, TOTAL_T, TOTAL_A) - _audio_index_at(r0, TOTAL_T, TOTAL_A)
check("block: one video_audio reference, audio under its frames, not shifted",
      ([r["kind"] for r in refs], refs[0]["ref_audio_t"], KEY in refs[0]),
      (["video_audio"], want_at, False))
run(TOTAL, 124, 5, ref_carry_audio="off")
refs = CALLS[1]["cond"]["minimax_refs"]
check("off: one video reference, no audio",
      ([r["kind"] for r in refs], refs[0]["ref_audio_t"]), (["video"], 0))
run(TOTAL, 124, 5, pin_audio=True, ref_carry_audio="off")
ok = []
for cl in CALLS:
    mv, ma = cl["mask"].unbind()
    ok.append((mv.shape[2] == cl["v_shape"][2], ma.shape[3] == cl["a_shape"][3],
               float(ma.max()) == 0.0, float(mv.min()) == 1.0))
check("a pinned track: each chunk gets its slice, audio kept, video free",
      ok, [(True, True, True, True)] * len(CALLS))
run(TOTAL, 124, 5, ref_carry_frames=39, ref_carry_audio_seconds=0.5)
refs = CALLS[1]["cond"]["minimax_refs"]
check("ref_carry_frames 39 -> 12 latents, 0.5s -> 20 audio latents",
      (refs[0]["latent_t"], refs[1]["ref_audio_t"]), (12, 20))

print("\n6. a clamped last window falls back to the mask carry")
T2, P2, S2 = find(True)
out2, n2, rep2 = run(T2, 124, 5)
last = CALLS[-1]
v0, v1, c = S2[-1]
mv, ma = last["mask"].unbind()
check("the clamped chunk is held by a mask over what it carries",
      (c > P2[1], float(mv[0, 0, :c].max()), float(mv[0, 0, c:].min())), (True, 0.0, 1.0))
check("and gets no reference or guide",
      (last["cond"].get("minimax_refs"), last["cond"].get("minimax_keyframes")),
      (None, None))
check("the report says so", "MASK carry here" in rep2, True)
check("earlier chunks are still reference carries",
      [bool(cl["cond"].get("minimax_refs")) for cl in CALLS[1:-1]], [True] * (n2 - 2))

print("\n7. colour matching corrects the reference, not the master")
out_c, _n, rep_c = run(TOTAL, 124, 5, carry_crossfade=False,
                       carry_color_match="first_chunk_latent")
check("the output is what it was without matching",
      float((out_c["samples"].unbind()[0] - v_out).abs().max()), 0.0)
refs = CALLS[2]["cond"]["minimax_refs"]
kf = CALLS[2]["cond"]["minimax_keyframes"]
r0, r1 = LS._ref_span(SP[2][0], SP[2][2], 7)
check("chunk 2's reference is the matched tail, not the sampled 20s",
      float((refs[0]["latent"] - 20.0).abs().max()) > 1e-3, True)
check("its pinned frame is cut from that same tail",
      float((kf[0]["latent"] - refs[0]["latent"][:, :, SP[2][0] - r0:SP[2][0] - r0 + 1])
            .abs().max()), 0.0)
check("report says the reference was matched", "next reference latent-matched" in rep_c, True)

print("\n8. refused combinations")
def raises(fn):
    try:
        fn()
    except (ValueError, RuntimeError) as e:
        return type(e).__name__
    return None
check("overlap 0", raises(lambda: run(TOTAL, 124, 0)), "ValueError")
check("carry_noise", raises(lambda: run(TOTAL, 124, 5, carry_noise=0.1)), "ValueError")
check("guide_video", raises(lambda: run(TOTAL, 124, 5, vae=object(),
                                         guide_video=torch.zeros([TOTAL, 64, 64, 3]))),
      "ValueError")
real = patch_ref_shift._state["ok"]
patch_ref_shift._state["ok"] = False
check("shifted audio without the patch", raises(lambda: run(TOTAL, 124, 5)), "RuntimeError")
check("block audio does not need it",
      raises(lambda: run(TOTAL, 124, 5, ref_carry_audio="block")), None)
patch_ref_shift._state["ok"] = real

print("\n%s" % ("ALL PASS" if not fails else "%d FAILED: %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)
