import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.context_windows import IndexListContextWindow, WindowingState
from mmh3tools.nodes_windows import (
    MMH3ContextWindows, MMH3WindowingState, _audio_index_at, _snap_grid)
from mmh3tools.common import (
    AUDIO_T_DIM, VIDEO_T_DIM, frames_to_audio_t, latents_to_frames)

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


def make_state(total_v):
    frames = latents_to_frames(total_v)
    total_a = frames_to_audio_t(frames)
    video = torch.zeros([1, 24, total_v, 6, 10])
    audio = torch.zeros([1, 32, 2, total_a])
    return MMH3WindowingState(
        latents=[video, audio], guide_latents=[None, None],
        guide_entries=[None, None], keyframe_idxs=[None, None],
        latent_shapes=[video.shape, audio.shape], dim=VIDEO_T_DIM,
        is_multimodal=True, temporal_downscale_ratio=4), total_a


print("\n1. boundary mapping is exact on the 5j+2 grid")
# 57 latents = 192 frames = 320 audio latents
for n in (0, 2, 7, 12, 57):
    got = _audio_index_at(n, 57, 320)
    want = frames_to_audio_t(latents_to_frames(n)) if n >= 2 else 0
    check("boundary n=%d" % n, got, want)

print("\n2. window covers the right audio span, on the right axis")
st, total_a = make_state(57)
w = IndexListContextWindow(list(range(0, 17)), dim=VIDEO_T_DIM, total_frames=57,
                           context_overlap=5)
pw = st.prepare_window(w, None)
aw = pw.get_window_for_modality(1)
check("primary dim stays 2", pw.dim, VIDEO_T_DIM)
check("audio dim is 3, not 2", aw.dim, AUDIO_T_DIM)
check("audio total is T40, not stereo 2", aw.total_frames, total_a)
check("audio span start", aw.index_list[0], 0)
check("audio span end", aw.index_list[-1] + 1, frames_to_audio_t(latents_to_frames(17)))

print("\n3. slicing with that window hits the temporal axis")
video, audio = st.latents
vs = pw.get_tensor(video)
as_ = aw.get_tensor(audio)
check("video sliced on dim 2", list(vs.shape), [1, 24, 17, 6, 10])
check("audio keeps stereo=2", int(as_.shape[2]), 2)
check("audio sliced on dim 3", int(as_.shape[3]), len(aw.index_list))

print("\n4. the stock single-dim path would have hit the stereo axis")
bad = IndexListContextWindow(list(range(0, 17)), dim=VIDEO_T_DIM, total_frames=57)
check("stock dim on audio = stereo size", int(audio.shape[VIDEO_T_DIM]), 2)
check("...which is not the audio length", int(audio.shape[VIDEO_T_DIM]) == total_a, False)

print("\n5. windows tile the whole audio track with no gap")
st, total_a = make_state(57)
covered = set()
starts = list(range(0, 57 - 17 + 1, 12)) or [0]
for s in starts:
    w = IndexListContextWindow(list(range(s, min(s + 17, 57))), dim=VIDEO_T_DIM,
                               total_frames=57, context_overlap=5)
    covered |= set(st.prepare_window(w, None).get_window_for_modality(1).index_list)
last = IndexListContextWindow(list(range(57 - 17, 57)), dim=VIDEO_T_DIM, total_frames=57)
covered |= set(st.prepare_window(last, None).get_window_for_modality(1).index_list)
check("audio fully covered", sorted(covered) == list(range(total_a)), True)

print("\n6. grid snapping")
for given, want in [(17, 17), (16, 12), (7, 7), (3, 2), (22, 22), (25, 22)]:
    check("snap %d -> %d" % (given, want), _snap_grid(given), want)

print("\n7. node wires a handler without touching core")
class FakeModel:
    def __init__(self): self.model_options = {}
    def clone(self):
        m = FakeModel(); m.model_options = dict(self.model_options); return m
import comfy.context_windows as C
_orig = C.create_prepare_sampling_wrapper
C.create_prepare_sampling_wrapper = lambda m: None
import mmh3tools.nodes_windows as NW
NW.create_prepare_sampling_wrapper = lambda m: None
m, label = MMH3ContextWindows.execute(FakeModel(), 16, 7, "pyramid",
                                      "standard_static", 1).result
h = m.model_options["context_handler"]
check("handler installed", isinstance(h, NW.MMH3ContextHandler), True)
check("length snapped to grid", h.context_length, 12)
check("overlap snapped to 5m+2", h.context_overlap, 7)
check("dim is video", h.dim, VIDEO_T_DIM)
check("causal fix off", h.causal_window_fix, False)
check("freenoise off", h.freenoise, False)
print("   label:", label.splitlines()[0])
C.create_prepare_sampling_wrapper = _orig

print("\n8. non-multimodal latents pass through untouched")
st_plain = MMH3WindowingState(
    latents=[torch.zeros([1, 24, 57, 6, 10])], guide_latents=[None],
    guide_entries=[None], keyframe_idxs=[None], latent_shapes=None,
    dim=VIDEO_T_DIM, is_multimodal=False, temporal_downscale_ratio=4)
w = IndexListContextWindow(list(range(0, 17)), dim=VIDEO_T_DIM, total_frames=57)
check("returned unchanged", st_plain.prepare_window(w, None) is w, True)

print("\n9. accumulators are sized on each modality's OWN dim")
from mmh3tools.nodes_windows import MMH3ContextHandler
from comfy.context_windows import get_matching_context_schedule, get_matching_fuse_method
h = MMH3ContextHandler(
    context_schedule=get_matching_context_schedule("standard_static"),
    fuse_method=get_matching_fuse_method("pyramid"),
    context_length=17, context_overlap=5, context_stride=1, closed_loop=False,
    dim=VIDEO_T_DIM, freenoise=False, causal_window_fix=False)
st, total_a = make_state(57)
accum, counts, biases = h._alloc_accumulators(st.latents, 1)
check("video counts extent", counts[0][0].shape[VIDEO_T_DIM], 57)
check("audio counts extent (not stereo 2)", counts[1][0].shape[AUDIO_T_DIM], total_a)
check("audio counts stereo axis is 1", counts[1][0].shape[VIDEO_T_DIM], 1)
check("video biases length", len(biases[0][0]), 57)
check("audio biases length", len(biases[1][0]), total_a)

print("\n10. the fuse step that crashed now runs on both modalities")
w = IndexListContextWindow(list(range(0, 17)), dim=VIDEO_T_DIM, total_frames=57,
                           context_overlap=5)
pw = st.prepare_window(w, None)
ts = torch.tensor([1.0])
for mod_idx in range(2):
    mw = pw.get_window_for_modality(mod_idx)
    sub = [mw.get_tensor(st.latents[mod_idx])]
    try:
        h.combine_context_window_results(
            st.latents[mod_idx], sub, [None], mw, 0, 1, ts,
            accum[mod_idx], counts[mod_idx], biases[mod_idx])
        check("modality %d fuses" % mod_idx, True, True)
    except RuntimeError as e:
        check("modality %d fuses" % mod_idx, str(e), "no error")
check("audio counts got written on dim 3", float(counts[1][0].sum()) > 0, True)
check("video counts got written on dim 2", float(counts[0][0].sum()) > 0, True)

print("\n11. window PHASE is uniform -- the pulsing bug")
# H3's latent groups start at 2+5k, and stride = length - overlap. Only an overlap
# of 5m+2 makes the stride a multiple of 5, keeping every window at one phase.
# With overlap 5m the stride is 5(j-m)+2, so the phase walks 0,2,4,1,3 -- a
# five-window beat, which is what the pulsing was.
C.create_prepare_sampling_wrapper = lambda m: None
NW.create_prepare_sampling_wrapper = lambda m: None
for L, OV in [(17, 7), (22, 7), (12, 7)]:
    mm, _ = MMH3ContextWindows.execute(FakeModel(), L, OV, "pyramid",
                                       "standard_static", 1).result
    hh = mm.model_options["context_handler"]
    ws = hh.get_context_windows(None, torch.zeros([1, 24, 57, 4, 4]), {})
    phases = [(w.index_list[0] - 2) % 5 for w in ws]
    check("len=%d ov=%d stride is a multiple of 5" % (L, OV),
          (hh.context_length - hh.context_overlap) % 5, 0)
    check("len=%d ov=%d phase uniform" % (L, OV), len(set(phases)), 1)

pulsing = MMH3ContextHandler(
    context_schedule=get_matching_context_schedule("standard_static"),
    fuse_method=get_matching_fuse_method("pyramid"),
    context_length=17, context_overlap=5, context_stride=1, closed_loop=False,
    dim=VIDEO_T_DIM, freenoise=False, causal_window_fix=False)
ws = pulsing.get_context_windows(None, torch.zeros([1, 24, 57, 4, 4]), {})
check("overlap=5 really does cycle the phase",
      len(set((w.index_list[0] - 2) % 5 for w in ws)) > 1, True)
C.create_prepare_sampling_wrapper = _orig

print("\n14. freenoise shuffles VIDEO only, on its own dim")
# Stock's multimodal path shuffles every modality on the primary dim; for audio
# [B,32,2,T40] that is the stereo axis, and it permutes left into right.
import comfy.utils
h2 = MMH3ContextHandler(
    context_schedule=get_matching_context_schedule("standard_static"),
    fuse_method=get_matching_fuse_method("pyramid"),
    context_length=17, context_overlap=7, context_stride=1, closed_loop=False,
    dim=VIDEO_T_DIM, freenoise=True, causal_window_fix=False)
st, total_a = make_state(57)
v0, a0 = st.latents[0].clone(), st.latents[1].clone()
torch.manual_seed(1)
v0.normal_(); a0.normal_()
packed, shapes = comfy.utils.pack_latents([v0, a0])
conds = [[{"model_conds": {"latent_shapes": type("C", (), {"cond": shapes})()}}]]
out = h2._apply_freenoise(packed.clone(), conds, 42)
nv, na = comfy.utils.unpack_latents(out, shapes)
check("video noise changed", bool((nv != v0).any()), True)
check("audio noise UNTOUCHED", bool((na == a0).all()), True)
check("shapes preserved", (tuple(nv.shape), tuple(na.shape)),
      (tuple(v0.shape), tuple(a0.shape)))
# the stereo channels must not have been permuted into each other
check("audio L/R not swapped", bool((na[:, :, 0] == a0[:, :, 0]).all()), True)

print("\n15. the node exposes freenoise and installs the wrapper")
C.create_prepare_sampling_wrapper = lambda m: None
NW.create_prepare_sampling_wrapper = lambda m: None
seen = {"wrapped": False}
_orig_ssw = NW.create_sampler_sample_wrapper
NW.create_sampler_sample_wrapper = lambda m: seen.__setitem__("wrapped", True)
m_off, l_off = MMH3ContextWindows.execute(FakeModel(), 17, 7, "pyramid",
                                          "standard_static", 1, False).result
check("default off", m_off.model_options["context_handler"].freenoise, False)
check("no wrapper when off", seen["wrapped"], False)
check("label says off", "freenoise off" in l_off, True)
m_on, l_on = MMH3ContextWindows.execute(FakeModel(), 17, 7, "pyramid",
                                        "standard_static", 1, True).result
check("on when asked", m_on.model_options["context_handler"].freenoise, True)
check("wrapper installed when on", seen["wrapped"], True)
check("label says ON", "freenoise ON" in l_on, True)
NW.create_sampler_sample_wrapper = _orig_ssw
C.create_prepare_sampling_wrapper = _orig

print("\n16. MMH3WindowPlan: work the schedule out before running it")
from mmh3tools.nodes_windows import MMH3WindowPlan as PLAN
from mmh3tools.common import frame_at_latent, latents_to_frames as _l2f

# frame_at_latent is the general form; latents_to_frames only means anything ON grid,
# and window bounds are arbitrary indices -- asking it about index 1 returns -12
check("agree wherever both are valid",
      all(frame_at_latent(n) == _l2f(n) for n in (2, 7, 12, 17, 37, 57, 107)), True)
check("arbitrary index is sane", frame_at_latent(1), 1)
check("latents_to_frames is not", _l2f(1) < 0, True)
check("cumulative spans", [frame_at_latent(k) for k in range(7)], [0, 1, 5, 9, 13, 17, 18])

L, OV, N, TF, TT, rep, WF, OVF = PLAN.execute(362, 124, 22, "standard_static", 4).result
# context_length / context_overlap are LATENTS; window_frames / overlap_frames are
# FRAMES. Only the frame pair may be wired into MMH3SplitAudioToWindows -- feeding it
# context_overlap re-snaps a latent count as a frame count and the splitter's schedule
# stops matching this plan, so each prompt describes audio its window never renders.
check("overlap_frames is the frame form of context_overlap",
      OVF, latents_to_frames(OV))
check("...and is NOT the latent value", OVF == OV, False)
check("window_frames likewise", WF, latents_to_frames(L))

# an overlap of 0 is legal, and 0 is OFF the 5j+2 grid -- latents_to_frames floors it
# to the group below and emits -12, which then re-snaps to something arbitrary if fed
# back into MMH3SplitAudioToWindows
_z = PLAN.execute(362, 124, 0, "standard_static", 0).result
check("overlap 0 -> context_overlap 0 latents", _z[1], 0)
check("...and overlap_frames 0, not -12", _z[7], 0)
check("context_length in latents", L, 37)
check("context_overlap is 5m+2", OV % 5, 2)
check("total snapped to 17j+5", TF, 362)
check("total latents", TT, 107)
check("window count", N, 4)
# under windowing the layout is rebuilt from the WINDOW's latent_t, so a keyframe
# node's target_frame_count needs this, not the clip length
check("window_frames is the window, not the clip", WF, 124)
# APPEND-ONLY: outputs serialise positionally, so a new one goes on the END and the
# existing order never changes. Pinning the whole list rather than just the last entry,
# since appending a second output is exactly what made the old check pass wrongly.
# RENAMES are fine and this list moves with them -- links serialise by slot INDEX, not
# by name, so a saved workflow survives a rename but not a reorder. The units in these
# names are load-bearing: the latent pair and the frame pair are five sockets apart and
# crossing them silently re-snaps (see CHANGELOG 0.76.0).
check("output order is append-only",
      [o.display_name for o in PLAN.define_schema().outputs],
      ["context_length (latents)", "context_overlap (latents)",
       "window_count", "total_frames (frames)", "total_latents (latents)",
       "report", "window_frames (frames)", "overlap_frames (frames)"])

# the emitted values have to survive the node they feed, or the plan is a lie
C.create_prepare_sampling_wrapper = lambda m: None
NW.create_prepare_sampling_wrapper = lambda m: None
mm2, _ = MMH3ContextWindows.execute(FakeModel(), L, OV, "pyramid", "standard_static", 1).result
hh2 = mm2.model_options["context_handler"]
check("context_length passes through unchanged", hh2.context_length, L)
check("context_overlap passes through unchanged", hh2.context_overlap, OV)
check("predicted count matches the real schedule",
      len(hh2.get_context_windows(None, torch.zeros([1, 24, TT, 4, 4]), {})), N)
C.create_prepare_sampling_wrapper = _orig

rows = [x for x in rep.splitlines() if x.startswith("  ") and "latents" in x]
check("first window starts at frame 0", "frames    0-" in rows[0], True)
check("last window ends on the last frame", "-%d " % (TF - 1) in rows[-1], True)

print("\n17. the plan reports what would otherwise go wrong silently")
# index rather than unpack: outputs are append-only, so a positional unpack breaks
# every time one is added, which is noise rather than a signal
rep5 = PLAN.execute(362, 100, 20, "standard_static", 6).result[5]
check("off-grid window reported", "window 100 ->" in rep5, True)
check("off-grid overlap reported", "overlap 20 ->" in rep5, True)
check("unreachable prompt caught", "never used" in rep5, True)
rep1 = PLAN.execute(192, 3600, 22, "standard_static", 0).result[5]
check("a window covering everything is called out", "windowing does nothing" in rep1, True)

print("\n18. MMH3SplitAudioToWindows: segments match what each window renders")
from mmh3tools.nodes_windows import (
    MMH3SplitAudioToWindows as SPLIT, MAX_WINDOW_AUDIO, _plan, _window_frame_spans)

SR = 44100
def ramp(seconds, channels=1):
    # sample value == its own timestamp, so a slice reveals where it came from
    n = int(seconds * SR)
    w = (torch.arange(n, dtype=torch.float32) / SR).reshape(1, 1, -1)
    return {"waveform": w.repeat(1, channels, 1), "sample_rate": SR}

n_seg, rep_a, *segs = SPLIT.execute(ramp(362 / 24.0), 362, 124, 22, "standard_static").result
check("one segment per window", n_seg, 4)
# segs is the 8 numbered sockets, THEN the indexed audio / first_frame / last_frame
check("unused numbered outputs are None", segs[4:MAX_WINDOW_AUDIO], [None] * 4)

# spans must equal the planner's, since a drift means the LLM hears the wrong audio
_, _, tf, _, wins = _plan(362, 124, 22, "standard_static")
for i, (fa, fb) in enumerate(_window_frame_spans(wins, tf)):
    w = segs[i]["waveform"]
    check("audio_%d starts at frame %d" % (i + 1, fa),
          round(float(w[0, 0, 0]) * 24), fa)
    # the span is [fa/24, (fb+1)/24) -- exclusive end -- so the final sample sits just
    # inside frame fb. round() would carry it into fb+1; floor is the honest test.
    check("audio_%d ends inside frame %d" % (i + 1, fb),
          int(float(w[0, 0, -1]) * 24), fb)

# the clamped final window is the case a uniform sequential split gets wrong:
# stride 102 would have put it at frames 306-429, past a 362-frame clip
check("last window is CLAMPED, not uniform-strided",
      round(float(segs[3]["waveform"][0, 0, 0]) * 24), 238)
check("...and a uniform stride would have said", 102 * 3, 306)

check("mono is widened to stereo", int(segs[0]["waveform"].shape[1]), 2)
_, _, *st = SPLIT.execute(ramp(362 / 24.0, 2), 362, 124, 22, "standard_static").result
check("stereo is left alone", int(st[0]["waveform"].shape[1]), 2)

# a short track is padded rather than yielding ragged segments
n_s, rep_s, *short = SPLIT.execute(ramp(8.0), 362, 124, 22, "standard_static").result
check("short track is reported", "short windows are padded" in rep_s, True)
check("segments stay full length",
      len({int(s["waveform"].shape[-1]) for s in short[:n_s]}), 1)

# more windows than NUMBERED sockets must say so, and point at the index path --
# the tail is no longer dropped, it is simply only reachable through `index`
n_many, rep_many, *many = SPLIT.execute(
    ramp(1000 / 24.0), 1000, 90, 22, "standard_static").result
check("overflow is reported", "numbered sockets" in rep_many, True)
check("...and points at the index path", "no such ceiling" in rep_many, True)

print("\n18b. the indexed output, which is what a for-loop drives")
check("more windows than numbered sockets", n_many > MAX_WINDOW_AUDIO, True)

def at(i, seconds=362 / 24.0, total=362, win=124, ov=22):
    r = SPLIT.execute(ramp(seconds), total, win, ov, "standard_static", i).result
    return r[2 + MAX_WINDOW_AUDIO], r[3 + MAX_WINDOW_AUDIO], r[4 + MAX_WINDOW_AUDIO]

spans_ref = _window_frame_spans(_plan(362, 124, 22, "standard_static")[4], 362)
for i in range(4):
    a_i, fa_i, fb_i = at(i)
    check("index %d matches audio_%d" % (i, i + 1),
          torch.equal(a_i["waveform"], segs[i]["waveform"]), True)
    check("...and reports window %d's own span" % i, (fa_i, fb_i), spans_ref[i])

# the whole point: reachable past the numbered ceiling
tail_a, _, _ = at(n_many - 1, 1000 / 24.0, 1000, 90, 22)
head_a, _, _ = at(0, 1000 / 24.0, 1000, 90, 22)
check("the last window is reachable by index", tail_a is not None, True)
check("and is not window 0 over again",
      torch.equal(tail_a["waveform"], head_a["waveform"]), False)

# out of range is an error, not a wrap -- matching MMH3CondSelect
try:
    at(n_many, 1000 / 24.0, 1000, 90, 22)
    check("out of range raises", False, True)
except ValueError as e:
    check("out of range raises", "only %d window" % n_many in str(e), True)
    check("...and names window_count as the fix", "window_count" in str(e), True)

print("\n19. a denoise mask is windowed per modality -- the masked-windowing crash")
# Core resizes model_conds only for raw tensors plus audio_embed / vace_context. A
# denoise mask is a CONDRegular, so it fell through UNWINDOWED and the model received a
# full-length mask against a windowed latent:
#   RuntimeError: The size of tensor a (640) must match the size of tensor b (866)
# LTXAV overrides resize_cond_for_context_window on the model class; MiniMaxH3 has no
# such override, and adding one would mean patching core. A RESIZE_COND_ITEM callback
# does it from the handler instead, with no core change.
import comfy.conds
import comfy.patcher_extension
from comfy.context_windows import IndexListCallbacks

hh = MMH3ContextHandler(
    context_schedule=get_matching_context_schedule("standard_static"),
    fuse_method=get_matching_fuse_method("pyramid"),
    context_length=22, context_overlap=7, context_stride=1, closed_loop=False,
    dim=VIDEO_T_DIM, freenoise=False, causal_window_fix=False)
cbs = comfy.patcher_extension.get_all_callbacks(
    IndexListCallbacks.RESIZE_COND_ITEM, hh.callbacks)
check("callback is registered exactly once", len(cbs), 1)
cb = cbs[0]

T_M = 117
st_m, A_M = make_state(T_M)
x_m = torch.zeros([1, 24, T_M, 6, 10])
pwm = st_m.prepare_window(hh.get_context_windows(None, x_m, {})[1], None)
awm = pwm.get_window_for_modality(1)

rv = cb("denoise_mask", comfy.conds.CONDRegular(torch.ones([1, 1, T_M, 6, 10])),
        pwm, x_m, "cpu", {})
ra = cb("audio_denoise_mask", comfy.conds.CONDRegular(torch.ones([1, 1, 2, A_M])),
        pwm, x_m, "cpu", {})
check("video mask sliced to the window", int(rv.cond.shape[VIDEO_T_DIM]), len(pwm.index_list))
check("audio mask sliced on ITS OWN dim", int(ra.cond.shape[AUDIO_T_DIM]), len(awm.index_list))
check("audio stereo axis survives", int(ra.cond.shape[2]), 2)
check("an unrelated cond key is left alone",
      cb("cross_attn", comfy.conds.CONDRegular(torch.ones([1, 1, T_M, 6, 10])),
         pwm, x_m, "cpu", {}), None)
check("an already-windowed mask is passed through",
      cb("denoise_mask",
         comfy.conds.CONDRegular(torch.ones([1, 1, len(pwm.index_list), 6, 10])),
         pwm, x_m, "cpu", {}), None)

print("\n20. a None cond (cfg 1.0 uncond) allocates NO accumulator")
h20 = MMH3ContextHandler(
    context_schedule=get_matching_context_schedule("standard_static"),
    fuse_method=get_matching_fuse_method("pyramid"),
    context_length=17, context_overlap=7, context_stride=1, closed_loop=False,
    dim=VIDEO_T_DIM, freenoise=False, causal_window_fix=False)
st20, _ = make_state(57)
a20, c20, b20 = h20._alloc_accumulators(st20.latents, [{}, None])
check("real cond gets a video accumulator", a20[0][0] is not None, True)
check("None cond gets none (video)", a20[0][1], None)
check("None cond gets none (audio)", a20[1][1], None)
check("None cond gets no counts either", c20[0][1], None)
check("int arg still works (older callers)",
      h20._alloc_accumulators(st20.latents, 1)[0][0][0].shape,
      st20.latents[0].shape)

print("\n21. execute end-to-end: cpu accumulators match gpu bit-for-bit, None cond "
      "returns zeros")
# The real handler.execute, driven by a stub calc_cond_batch that returns a
# deterministic function of the window slice. Whatever execute allocates, slices,
# fuses and returns is exercised for real -- only the model call is faked.
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T21 = 57
x21 = torch.arange(1 * 24 * T21 * 4 * 6, dtype=torch.float32).reshape(1, 24, T21, 4, 6)
x21 = (x21 / x21.numel()).to(DEV)
sigmas21 = torch.tensor([1.0, 0.5, 0.0], device=DEV)
mo21 = {"transformer_options": {"sample_sigmas": sigmas21}}
ts21 = torch.tensor([1.0], device=DEV)


class _LF21:
    temporal_downscale_ratio = 4


class _Model21:
    latent_format = _LF21()


def stub_calc(model, sub_conds, sub_x, sub_ts, model_options):
    outs = []
    for c in sub_conds:
        if c is None:  # mirrors core: a None cond contributes zeros
            outs.append(torch.zeros_like(sub_x))
        else:
            outs.append(sub_x * 2.0 + 1.0)
    return outs


def run21(acc_dev):
    h = MMH3ContextHandler(
        context_schedule=get_matching_context_schedule("standard_static"),
        fuse_method=get_matching_fuse_method("pyramid"),
        context_length=17, context_overlap=7, context_stride=1, closed_loop=False,
        dim=VIDEO_T_DIM, freenoise=False, causal_window_fix=False,
        accumulator_device=acc_dev)
    return h.execute(stub_calc, _Model21(), [[{}], None], x21.clone(), ts21, dict(mo21))

out_gpu = run21("gpu")
out_cpu = run21("cpu")
check("two conds out", (len(out_gpu), len(out_cpu)), (2, 2))
check("cond 0 is full-length", list(out_gpu[0].shape), list(x21.shape))
check("results land on the sampler's device",
      (str(out_cpu[0].device.type), str(out_cpu[1].device.type)),
      (x21.device.type, x21.device.type))
check("gpu path reproduces the model output where windows are pure",
      torch.allclose(out_gpu[0][:, :, :10], x21[:, :, :10] * 2.0 + 1.0,
                     atol=1e-6), True)
check("cpu accumulators match gpu", torch.allclose(out_gpu[0], out_cpu[0],
                                                   atol=1e-6), True)
check("None cond returns zeros (gpu)", float(out_gpu[1].abs().max()), 0.0)
check("None cond returns zeros (cpu)", float(out_cpu[1].abs().max()), 0.0)
check("None cond zeros are full-shape", list(out_cpu[1].shape), list(x21.shape))

# and the relative fuse path, which writes per-index instead of per-window
def run21r(acc_dev):
    h = MMH3ContextHandler(
        context_schedule=get_matching_context_schedule("standard_static"),
        fuse_method=get_matching_fuse_method("relative"),
        context_length=17, context_overlap=7, context_stride=1, closed_loop=False,
        dim=VIDEO_T_DIM, freenoise=False, causal_window_fix=False,
        accumulator_device=acc_dev)
    return h.execute(stub_calc, _Model21(), [[{}], None], x21.clone(), ts21, dict(mo21))

outr_gpu = run21r("gpu")
outr_cpu = run21r("cpu")
check("relative fuse: cpu matches gpu",
      torch.allclose(outr_gpu[0], outr_cpu[0], atol=1e-6), True)
check("relative fuse: None cond zeros", float(outr_cpu[1].abs().max()), 0.0)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
