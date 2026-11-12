"""MMH3LoopingSampler: the schedule, the carry and the write-back, without a model.

The sampler call is stubbed, so what is under test is everything AROUND it: that
the chunk count is derived from the clip rather than supplied, that chunks are the
SAME schedule the windowing nodes compute, that each chunk gets its own prompt and
its own span of audio, that the master comes out the length it went in, and that
the source guider is never mutated.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.nested_tensor import NestedTensor

from mmh3tools import nodes_looping_sampler as LS
from mmh3tools.nodes_windows import MMH3WindowPlan as PLAN, _plan, _window_frame_spans
from mmh3tools.common import latents_to_frames, frames_to_audio_t, frames_to_latents

H = W = 4
fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def clip(frames, mark_audio=False, pin_audio=False):
    """A whole-clip AV latent. Audio can be ramped so a slice reveals its origin."""
    t = frames_to_latents(frames)
    at = frames_to_audio_t(latents_to_frames(t))
    v = torch.zeros([1, 24, t, H, W])
    a = (torch.arange(at, dtype=torch.float32).reshape(1, 1, 1, at).expand(1, 32, 2, at).clone()
         if mark_audio else torch.zeros([1, 32, 2, at]))
    d = {"samples": NestedTensor([v, a])}
    if pin_audio:
        d["noise_mask"] = NestedTensor([torch.ones([1, 1, t, H, W]),
                                        torch.zeros([1, 1, 2, at])])
    return d


def cond(tag, **extra):
    d = {"tag": tag}; d.update(extra)
    return [[torch.zeros([1, 8]), d]]


def tag_of(c):
    return c[0][1]["tag"]


class FakeGuider:
    def __init__(self):
        self.original_conds = {"positive": "BASE_POS", "negative": "BASE_NEG"}
    def set_conds(self, positive, negative=None):
        self.original_conds["positive"] = positive


class FakeNoise:
    def __init__(self, seed=1000):
        self.seed = seed


class FakeVae:
    def __init__(self): self.calls = 0
    def encode(self, img):
        self.calls += 1
        return torch.zeros([1, 24, 1, H, W])


SEEN = {"guiders": [], "seeds": [], "chunks": [], "calls": []}

class FakeSampler:
    """Returns the chunk with its VIDEO stamped, so the write-back is visible.

    Stamps BOTH return slots: the node takes the denoised one, and a fake that
    only fills the first would pass for the wrong reason.
    """
    def sample(self, noise, guider, sampler, sigmas, latent):
        SEEN["calls"].append({"sampler": sampler, "guider": guider,
                              "steps": len(sigmas) - 1,
                              "hi": float(sigmas[0]), "lo": float(sigmas[-1])})
        if sampler != "PH2":                  # phase 2 continues an open chunk
            SEEN["guiders"].append(guider)
            SEEN["seeds"].append(getattr(noise, "seed", None))
            SEEN["chunks"].append(latent)
        v, a = latent["samples"].unbind()
        marked = torch.full_like(v, float(len(SEEN["chunks"])))
        out = {"samples": NestedTensor([marked, a])}
        return out, out

LS.SamplerCustomAdvanced = FakeSampler

TOTAL, CHUNK, OV = 3048, 481, 22          # a 127s track in 20s chunks
SIGMAS = torch.linspace(1.0, 0.0, 13)     # 12 steps


def run(total=TOTAL, chunk=CHUNK, ov=OV, prompts=None, carry="mask",
        latent=None, kf=None, kf_idx="", vae=None, sigmas=SIGMAS,
        start=0, end=1000, p2_start=0, p2_sampler=None, p2_guider=None):
    for k in SEEN:
        SEEN[k].clear()
    g = FakeGuider()
    n_expect = len(_plan(total, chunk, ov, "standard_static")[4])
    cs = [cond("p%d" % i) for i in range(prompts if prompts is not None else n_expect)]
    out = LS.MMH3LoopingSampler.execute(
        FakeNoise(), g, "S", sigmas, {"conds": cs},
        latent if latent is not None else clip(total),
        chunk, ov, carry, 1.0, 1.0,
        start, end, p2_start, p2_sampler, p2_guider, kf, kf_idx, vae).result
    return out, g


print("\n1. the chunk count is DERIVED from the clip, not supplied")
(out, n, rep), g = run()
check("7 chunks for a 127s clip in 20s pieces", n, 7)
check("sampler ran once per chunk", len(SEEN["chunks"]), 7)
# and it is the SAME schedule the windowing nodes compute
check("matches MMH3WindowPlan's window_count",
      n, PLAN.execute(TOTAL, CHUNK, OV, "standard_static", 0).result[2])

print("\n2. the master comes out the length it went in")
v, a = out["samples"].unbind()
src_v, src_a = clip(TOTAL)["samples"].unbind()
check("video latents unchanged", int(v.shape[2]), int(src_v.shape[2]))
check("audio latents unchanged", int(a.shape[3]), int(src_a.shape[3]))
check("...which is the clip's own length", latents_to_frames(int(v.shape[2])),
      latents_to_frames(int(src_v.shape[2])))

print("\n3. every chunk's output is written back into the master")
# the stub stamps chunk k with k+1, and later chunks overwrite earlier ones in
# their overlap, so the last chunk's stamp must be present at the end
check("the final chunk reached the master", float(v[:, :, -1].max()), 7.0)
check("the first chunk reached the master", float(v[:, :, 0].max()), 1.0)
check("nothing was left unwritten", bool((v > 0).all()), True)

print("\n4. each chunk gets its own prompt, in order")
check("prompt order", [tag_of(gg.raw_conds[0]) for gg in SEEN["guiders"]],
      ["p%d" % i for i in range(7)])

print("\n5. each chunk gets its OWN span of audio")
(out2, _n, _r), _g = run(latent=clip(TOTAL, mark_audio=True))
firsts = [float(c["samples"].unbind()[1][0, 0, 0, 0]) for c in SEEN["chunks"]]
check("audio spans advance, not repeat", firsts == sorted(set(firsts)), True)
check("chunk 0 starts at the track's start", firsts[0], 0.0)
check("every chunk differs", len(set(firsts)), 7)

print("\n6. the guider is copied, never mutated")
check("base positive intact", g.original_conds["positive"], "BASE_POS")
check("distinct copies", len({id(gg) for gg in SEEN["guiders"]}), 7)
check("...none sharing the source's dict",
      any(gg.original_conds is g.original_conds for gg in SEEN["guiders"]), False)
check("every chunk saw the same negative",
      {gg.raw_conds[1] for gg in SEEN["guiders"]}, {"BASE_NEG"})

print("\n7. noise advances per chunk")
check("seven distinct seeds", len(set(SEEN["seeds"])), 7)
check("chunk 0 keeps the wired seed", SEEN["seeds"][0], 1000)

print("\n8. carry='mask' pins the overlap, and honours the master's own mask")
run()
m0 = SEEN["chunks"][0].get("noise_mask")
m1 = SEEN["chunks"][1].get("noise_mask")
check("chunk 0 has no carry to pin", m0, None)
vm1, _am1 = m1.unbind()
ov_lat = _plan(TOTAL, CHUNK, OV, "standard_static")[1]
check("chunk 1 preserves its overlap", float(vm1[:, :, :ov_lat].max()), 0.0)
check("...and generates the rest", float(vm1[:, :, ov_lat:].min()), 1.0)

# a pinned audio track has to survive into every chunk
run(latent=clip(TOTAL, pin_audio=True))
for i in (0, 3, 6):
    _vm, am = SEEN["chunks"][i]["noise_mask"].unbind()
    check("chunk %d keeps the pinned audio" % i, float(am.max()), 0.0)

print("\n9. carry='keyframe' passes the overlap as a guide instead")
if not LS._guides_available():
    print("  SKIP  #15439 not applied")
else:
    run(carry="keyframe")
    check("chunk 0 has no mask", SEEN["chunks"][0].get("noise_mask"), None)
    check("...and no mask on later chunks either",
          SEEN["chunks"][1].get("noise_mask"), None)
    kfs = [gg.raw_conds[0][0][1].get("minimax_keyframes") for gg in SEEN["guiders"]]
    check("chunk 0 has no guide", kfs[0], None)
    check("chunk 1 carries one", len(kfs[1]), 1)
    check("anchored at frame 0", kfs[1][0]["resolved_frame_index"], 0)
    check("a multi-step clip, not a still", int(kfs[1][0]["latent"].shape[2]), ov_lat)
    check("with its audio", "audio_latent" in kfs[1][0], True)

print("\n10. keyframe_indices are frames of the WHOLE clip")
vae = FakeVae()
(out3, _n, rep3), _g = run(kf=torch.zeros([2, 64, 64, 3]), kf_idx="0, 1500", vae=vae)
check("encoded once per image", vae.calls, 2)
placed = [l for l in rep3.splitlines() if "keyframe frame" in l]
check("both placed", len(placed), 2)
check("frame 0 -> chunk 0", "frame 0 -> chunk 0" in placed[0], True)
kfs = [gg.raw_conds[0][0][1].get("minimax_keyframes") for gg in SEEN["guiders"]]
owners = [i for i, k in enumerate(kfs) if k]
check("they land in different chunks", len(set(owners)), 2)

print("\n10b. what is refused")
# NOTE: kf_idx with no images is NOT here -- it is ignored rather than refused, so
# one graph can be reused across passes that do and do not anchor. See the
# "keyframe_indices without keyframes" section below.
for kw, err in ((dict(kf=torch.zeros([1, 64, 64, 3]), kf_idx="0"), "vae"),
                (dict(kf=torch.zeros([1, 64, 64, 3]), kf_idx="99999",
                      vae=FakeVae()), "outside the clip")):
    try:
        run(**kw)
        check("refused: %s" % err, False, True)
    except (ValueError, RuntimeError) as e:
        check("refused: %s" % err, err in str(e), True)

print("\n10c. a keyframe/index count mismatch is trimmed and reported, not refused")
# since 0.97.0: images and indices are zipped, the surplus of either is dropped
vae = FakeVae()
(_o, _n, rep_mm), _g = run(kf=torch.zeros([2, 64, 64, 3]), kf_idx="0", vae=vae)
check("only the paired image is encoded", vae.calls, 1)
check("one keyframe placed", len([l for l in rep_mm.splitlines() if "keyframe frame" in l]), 1)
check("the drop is in the report", "kept 1, dropped the last 1 keyframe image(s)" in rep_mm, True)

print("\n11. a Basic Guider has no negative")
class BasicGuider:
    def __init__(self): self.original_conds = {"positive": "BASE_POS"}
    def set_conds(self, positive): self.original_conds["positive"] = positive
for _k in SEEN:
    SEEN[_k].clear()
o = LS.MMH3LoopingSampler.execute(
    FakeNoise(), BasicGuider(), "S", SIGMAS,
    {"conds": [cond("p%d" % i) for i in range(7)]}, clip(TOTAL),
    CHUNK, OV, "mask", 1.0, 1.0, 0, 1000, 0, None, None, None, "", None).result
check("runs against a one-arg guider", o[1], 7)
check("negative reported as None", SEEN["guiders"][0].raw_conds[1], None)

print("\n12. prompt/chunk count mismatches are reported, not silent")
(_o, n2, rep2), _g = run(prompts=2)
check("still renders every chunk", n2, 7)
check("and says the last repeats", "the last repeats" in rep2, True)
(_o, n3, rep3), _g = run(prompts=12)
check("extras are called out", "the extras are unused" in rep3, True)
try:
    LS.MMH3LoopingSampler.execute(FakeNoise(), FakeGuider(), "S", SIGMAS,
                                  {"conds": []}, clip(TOTAL), CHUNK, OV,
                                  "mask", 1.0, 1.0, 0, 0, 1000, 0, None, None,
                                  None, "", None)
    check("empty cond_set raises", False, True)
except ValueError as e:
    check("empty cond_set raises", "no conditioning" in str(e), True)

print("\n13. MMH3KeyframePlanner plans against the same schedule")
idx, cnt, nch, _rep = LS.MMH3KeyframePlanner.execute(TOTAL, CHUNK, OV, True, True).result
check("chunk_count agrees with the sampler", nch, 7)
check("one keyframe per chunk, plus the opening", cnt, 8)
check("opens at 0", idx.startswith("0,"), True)
check("closes on -1", idx.strip().endswith("-1"), True)

# and the plan is USABLE: every index lands in a distinct chunk
vae = FakeVae()
(_o, _n, rep4), _g = run(kf=torch.zeros([cnt, 64, 64, 3]), kf_idx=idx, vae=vae)
check("the planner's count is what the sampler wanted", vae.calls, cnt)
kfs = [gg.raw_conds[0][0][1].get("minimax_keyframes") or [] for gg in SEEN["guiders"]]
check("every chunk got exactly one planned keyframe",
      [len(k) for k in kfs], [2] + [1] * 6)

check("no start drops one", LS.MMH3KeyframePlanner.execute(
    TOTAL, CHUNK, OV, False, True).result[1], 7)
check("no end drops one", LS.MMH3KeyframePlanner.execute(
    TOTAL, CHUNK, OV, True, False).result[1], 7)

print("\n14. the schedule window, per chunk (LTXAVTools' semantics)")
(_o, _n, rep5), _g = run()
check("unwindowed: one sampler call per chunk", len(SEEN["calls"]), 7)
check("and the full 12 steps", SEEN["calls"][0]["steps"], 12)
check("no window line in the report", "schedule window" in rep5, False)

# end_step drops the tail: sigmas[:step+1], core SplitSigmas' first output
(_o, _n, rep6), _g = run(end=4)
check("end=4 runs 4 steps", SEEN["calls"][0]["steps"], 4)
check("and stops above zero", SEEN["calls"][0]["lo"] > 0.0, True)
check("report says partially denoised", "PARTIALLY denoised" in rep6, True)

# start_step skips the head: sigmas[step:], sharing the boundary sigma
(_o, _n, _r), _g = run(start=4)
check("start=4 runs the remaining 8", SEEN["calls"][0]["steps"], 8)
check("and picks up where end=4 stopped",
      abs(SEEN["calls"][0]["hi"] - float(SIGMAS[4])) < 1e-6, True)

# absolute indices: pass 1 `end N` hands to pass 2 `start N` with no arithmetic
(_o, _n, _r), _g = run(start=4, end=9)
check("start=4 end=9 is a 5-step window", SEEN["calls"][0]["steps"], 5)
check("windowing applies to EVERY chunk",
      [c["steps"] for c in SEEN["calls"]], [5] * 7)

try:
    run(start=6, end=3)
    check("an empty window raises", False, True)
except ValueError as e:
    check("an empty window raises", "contains no steps" in str(e), True)

print("\n15. phase 2 takes over mid-schedule")
(_o, _n, rep7), _g = run(p2_start=4, p2_sampler="PH2")
check("two calls per chunk", len(SEEN["calls"]), 14)
check("phase 1 gets steps 0-3", SEEN["calls"][0]["steps"], 4)
check("phase 2 gets the rest", SEEN["calls"][1]["steps"], 8)
check("phase 1 keeps the main sampler", SEEN["calls"][0]["sampler"], "S")
check("phase 2 uses its own", SEEN["calls"][1]["sampler"], "PH2")
check("report names it", "phase 2 from step 4" in rep7, True)
check("still one chunk per window", _n, 7)

# unconnected phase2_guider falls back to the main one
check("falls back to the phase-1 guider",
      SEEN["calls"][1]["guider"] is SEEN["calls"][0]["guider"], True)

# a connected one is rebound to THIS chunk's prompt, not its own
(_o, _n, _r), _g = run(p2_start=4, p2_sampler="PH2", p2_guider=FakeGuider())
p2_guiders = [c["guider"] for c in SEEN["calls"] if c["sampler"] == "PH2"]
check("phase 2 gets its own guider object",
      p2_guiders[0] is not SEEN["calls"][0]["guider"], True)
check("carrying this chunk's prompt, not its own",
      [tag_of(gg.raw_conds[0]) for gg in p2_guiders],
      ["p%d" % i for i in range(7)])

# phase2_start_step is ABSOLUTE -- it rebases onto the window start leaves
(_o, _n, _r), _g = run(start=2, p2_start=6, p2_sampler="PH2")
check("phase 1 covers steps 2-5", SEEN["calls"][0]["steps"], 4)
check("phase 2 covers 6-11", SEEN["calls"][1]["steps"], 6)

# a cut point outside the window is simply not a cut
(_o, _n, _r), _g = run(end=3, p2_start=8, p2_sampler="PH2")
check("phase 2 past the window never runs", len(SEEN["calls"]), 7)
check("and the sampler stays phase 1", SEEN["calls"][0]["sampler"], "S")

print("\n16. the CLAMPED final window does not overwrite its predecessor")
# Core pulls the last window back so it ends on the clip end, so it physically
# overlaps the one before by much more than the nominal overlap. Carrying only the
# nominal amount left the difference to be regenerated under the LAST prompt and
# written over content the previous chunk had already drawn -- up to 12s of it.
from mmh3tools.common import frame_at_latent
for total, chunk in ((3048, 481), (4320, 481), (5040, 192)):
    length, overlap, pf, _pt, w = _plan(total, chunk, OV, "standard_static")
    worst = 0
    for i, x in enumerate(w):
        if i == 0:
            continue
        v0, v1 = x.index_list[0], x.index_list[-1] + 1
        actual = max(0, w[i - 1].index_list[-1] + 1 - v0)
        carried = min(actual, v1 - v0)          # the shipped rule
        worst = max(worst, actual - carried)
    check("total=%d chunk=%d: nothing clobbered" % (total, chunk), worst, 0)

# and the ragged case really is ragged, or the test above proves nothing
_l, _ov, _pf, _pt, w = _plan(4320, 481, OV, "standard_static")
tail_actual = (w[-2].index_list[-1] + 1) - w[-1].index_list[0]
check("the 180s case genuinely clamps", tail_actual > _ov, True)
check("...by 85 latents", tail_actual - _ov, 85)

# end to end: the second-to-last chunk's stamp must survive in the master
(out2, n2, rep2), _g = run(total=4320, chunk=481)
v2, _a2 = out2["samples"].unbind()
stamps = sorted({float(x) for x in v2[0, 0, :, 0, 0].unique()})
check("every chunk is still visible in the master", len(stamps), n2)
check("the report names the clamped tail", "clamped tail" in rep2, True)

# --- indices with no images attached are inert ----------------------------
# A ladder reuses one graph across passes and usually only the first anchors, so a
# live keyframe_indices string with the image input unplugged is the ordinary state
# of a refine pass. It must not stop the run, and must not be parsed either.
print("\n-- keyframe_indices without keyframes --")

(out_ni, n_ni, rep_ni), _g = run(kf_idx="0, 60, -1", kf=None, vae=None)
check("indices with no images do not raise", n_ni > 0, True)
check("...and the report says they were ignored",
      "keyframe_indices ignored" in rep_ni, True)

# an index that would be OUT OF RANGE is not parsed either, so it cannot raise
(out_oor, n_oor, _r), _g = run(kf_idx="0, 999999", kf=None, vae=None)
check("an out-of-range index is inert too when no images are attached", n_oor > 0, True)

# ...nor is a malformed one, for the same reason: nothing is being placed
(out_bad, n_bad, _r), _g = run(kf_idx="not-a-number", kf=None, vae=None)
check("a malformed index is inert too", n_bad > 0, True)

# an empty string stays quiet -- no note for a field nobody filled in
(_o, _n, rep_empty), _g = run(kf_idx="", kf=None, vae=None)
check("an empty index string says nothing", "keyframe_indices ignored" in rep_empty, False)

# and the guard still bites where it should: images attached, but no vae
try:
    run(kf_idx="0", kf=torch.rand([1, 64, 64, 3]), vae=None)
    check("images without a vae still raise", "no raise", "raise")
except ValueError:
    check("images without a vae still raise", "raise", "raise")


# --- keyframe stills are fitted to the target grid ------------------------
# Keyframe rows share the TARGET spatial grid: PackedLayout reads only the latent's
# time dim and sizes the segment from the target, so a still at any other resolution
# reserves the wrong row count and dies deep in the model. _fit_keyframe is what
# stops that, and it has to be a no-op when the size already agrees.
print("\n-- keyframe fitting --")

def img(h, w):
    return torch.rand([1, h, w, 3])

# already the target size: identity, no note, and the SAME object (no resize call)
same = img(768, 1344)
got, note = LS._fit_keyframe(same, 1344, 768, is_opener=True)
check("matching dims give no note", note, None)
check("...and are passed through untouched", got is same, True)

# the opener stretches: nothing cropped away, both axes forced
op, note_op = LS._fit_keyframe(img(2304, 4096), 1344, 768, is_opener=True)
check("opener is resized to the target", list(op.shape), [1, 768, 1344, 3])
check("...and says so", "stretch" in note_op, True)
check("...naming both sizes", "4096x2304 -> 1344x768" in note_op, True)

# a later anchor centre crops instead
fol, note_fol = LS._fit_keyframe(img(2304, 4096), 1344, 768, is_opener=False)
check("follower is resized to the target", list(fol.shape), [1, 768, 1344, 3])
check("...by centre crop", "centre crop" in note_fol, True)

# the two policies genuinely differ on a mismatched aspect (square -> 1.75)
sq = img(1024, 1024)
a, _ = LS._fit_keyframe(sq, 1344, 768, is_opener=True)
b, _ = LS._fit_keyframe(sq, 1344, 768, is_opener=False)
check("stretch and crop disagree on a square source",
      bool(torch.allclose(a, b, atol=1e-6)), False)

# ...and agree when the aspect already matches, differing only in scale
wide = img(1152, 2016)   # same 1.75 aspect as 1344x768
c, _ = LS._fit_keyframe(wide, 1344, 768, is_opener=True)
d, _ = LS._fit_keyframe(wide, 1344, 768, is_opener=False)
check("stretch and crop agree when the aspect matches",
      bool(torch.allclose(c, d, atol=1e-6)), True)

# an RGBA still is reduced to 3 channels rather than encoded with an alpha
rgba, _ = LS._fit_keyframe(torch.rand([1, 512, 512, 4]), 1344, 768, is_opener=False)
check("an alpha channel is dropped", list(rgba.shape), [1, 768, 1344, 3])

# the fitted result is still image-shaped [B,H,W,C], which is what vae.encode takes
check("output stays 4D image-shaped", op.ndim, 4)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
