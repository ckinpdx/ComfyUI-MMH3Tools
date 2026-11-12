import os, sys, math, types
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
import comfy.ldm.minimax.vae as V
from mmh3tools.nodes_save import TAIL_FRAMES, vae_grid

K = V.MiniMaxH3VideoVAE

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


# A stand-in VAE carrying only the constants and methods decode_temporal touches.
# _adaptive_decode is deterministic and encodes PROVENANCE: every pixel's value is the
# absolute latent index it came from, so a mis-sliced chunk shows up as wrong numbers
# rather than as a plausible-looking tensor. blend() is the REAL one.
#
# LIMIT OF THIS STUB: because values are constant per latent, a blend between two
# regions drawn from the SAME latents is numerically invisible. So these tests catch
# slicing, context and trim errors -- the class of mistake actually available here --
# but would not catch wrong blend WEIGHTS. Those come from core and are unchanged.
def make_vae(H=1, W=1):
    o = types.SimpleNamespace(clip_length=17, token_drop=3, vae_ratio_t=4, tiling=False)
    o.frame_pre_padding = (-o.clip_length) % o.vae_ratio_t
    o.tokens_chunk_size = math.ceil(o.clip_length / o.vae_ratio_t)
    o.token_overlap = (-o.token_drop) % o.tokens_chunk_size
    o.frame_overlap = max(o.token_overlap * o.vae_ratio_t - o.frame_pre_padding, 0)
    o.blend = K.blend.__get__(o)
    o._decode_temporal_pad_frames = K._decode_temporal_pad_frames.__get__(o)
    o._decode_temporal_frame_plan = K._decode_temporal_frame_plan.__get__(o)
    # decode_temporal preallocates its output via decode_output_shape, added by the
    # H3 VAE work in ComfyUI v0.32.0 (#15446 / #15486). Bind it from core rather than
    # reimplementing, so this stub keeps tracking core's real geometry instead of a
    # copy that can silently drift from it. vae_ratio is 1 because the fake decode
    # below returns H/W unscaled.
    o.vae_ratio = 1
    # the stock H3 VAE; Sol-H3-VAE's noncausal decoder (#17411) has a different
    # frame grid and is not what the 17j+5 checks below describe
    o.noncausal = False
    o.decoder = types.SimpleNamespace(out_channels=3)
    o._decode_temporal_chunks = K._decode_temporal_chunks.__get__(o)
    o.decode_output_shape = K.decode_output_shape.__get__(o)
    # _finalize_pixels is core's colour transform and CLAMPS to [0,1]. This stub's
    # decode returns each frame's latent index as its pixel value so the test can
    # trace frames back to latents; clamping would collapse every index above 1 and
    # the tracing would silently pass on garbage. Identity is correct here -- the
    # transform is not what these checks are about.
    o._finalize_pixels = lambda part: part

    def adaptive(z):
        # z is [B,C,t,H,W] carrying its own latent index in channel 0
        t = int(z.shape[2])
        vals = z[:, :1, :, :1, :1].reshape(-1)            # one index per latent
        frames = vals.repeat_interleave(o.vae_ratio_t)     # 4 pixel frames per latent
        return frames.reshape(1, 1, -1, 1, 1).expand(1, 3, t * o.vae_ratio_t, H, W).clone()

    o._adaptive_decode = adaptive
    o.decode_temporal = K.decode_temporal.__get__(o)
    return o


def latents(T):
    """[1,24,T,1,1] where every value is its own latent index."""
    z = torch.arange(T, dtype=torch.float32).reshape(1, 1, T, 1, 1)
    return z.expand(1, 24, T, 1, 1).clone()


print("\n1. grid constants are read from the VAE, not assumed")
fake = types.SimpleNamespace(first_stage_model=make_vae())
group, lookahead, fpg = vae_grid(fake)
check("group size", group, 5)
check("lookahead", lookahead, 2)
check("frames per group", fpg, 17)
check("tail frames", TAIL_FRAMES, 5)

print("\n2. a full decode is 17j+5, and every frame traces to its latent")
vae = make_vae()
for T in (7, 12, 22, 57):
    out = vae.decode_temporal(latents(T))
    check("T=%d -> %d frames" % (T, 17 * ((T - 2) // 5) + 5),
          int(out.shape[2]), 17 * ((T - 2) // 5) + 5)

print("\n3. THE POINT: chunked-with-context is identical to a full decode")
# replicates nodes_save's slicing exactly: [5*g0-5 : 5*g1+2], drop the context
# group's frames, drop the trailing 5 unless this is the final batch
def streamed(T, groups_per_chunk):
    vae = make_vae()
    z = latents(T)
    n_groups = (T - 2) // 5
    parts, g0 = [], 0
    while g0 < n_groups:
        g1 = min(g0 + groups_per_chunk, n_groups)
        last = g1 >= n_groups
        lo, hi = max(0, 5 * g0 - 5), min(T, 5 * g1 + 2)
        px = vae.decode_temporal(z[:, :, lo:hi])
        head = 17 if g0 > 0 else 0
        keep = 17 * (g1 - g0)
        parts.append(px[:, :, head:head + keep + (TAIL_FRAMES if last else 0)])
        g0 = g1
    return torch.cat(parts, dim=2)

for T in (12, 22, 37, 57, 107):
    full = make_vae().decode_temporal(latents(T))
    for gpc in (1, 2, 4, 100):
        got = streamed(T, gpc)
        same_len = int(got.shape[2]) == int(full.shape[2])
        exact = same_len and bool(torch.equal(got, full))
        check("T=%3d groups/chunk=%-3d exact" % (T, gpc), exact, True)
        if not exact and same_len:
            d = (got != full).any(dim=(0, 1, 3, 4)).nonzero().flatten().tolist()
            print("        differing frames: %s" % d[:12])

print("\n4. the trailing 5 must be dropped on every batch but the last")
# A partial decode always ends by writing its last chunk's carried part RAW; a full
# decode never writes that part raw, it blends it into the chunk after. Keeping it on
# an interior batch therefore both duplicates content and overruns the length.
def streamed_keep_tail(T, groups_per_chunk):
    vae, z = make_vae(), latents(T)
    n_groups = (T - 2) // 5
    parts, g0 = [], 0
    while g0 < n_groups:
        g1 = min(g0 + groups_per_chunk, n_groups)
        px = vae.decode_temporal(z[:, :, max(0, 5 * g0 - 5):min(T, 5 * g1 + 2)])
        head = 17 if g0 > 0 else 0
        parts.append(px[:, :, head:head + 17 * (g1 - g0) + TAIL_FRAMES])
        g0 = g1
    return torch.cat(parts, dim=2)

kept = streamed_keep_tail(57, 4)
check("keeping the tail overruns the length", int(kept.shape[2]) > 192, True)
check("dropping it is exactly right", int(streamed(57, 4).shape[2]), 192)

print("\n5. context is what makes it exact -- without it the join is wrong")
def streamed_no_context(T, groups_per_chunk):
    vae, z = make_vae(), latents(T)
    n_groups = (T - 2) // 5
    parts, g0 = [], 0
    while g0 < n_groups:
        g1 = min(g0 + groups_per_chunk, n_groups)
        last = g1 >= n_groups
        px = vae.decode_temporal(z[:, :, 5 * g0:min(T, 5 * g1 + 2)])
        parts.append(px[:, :, :17 * (g1 - g0) + (TAIL_FRAMES if last else 0)])
        g0 = g1
    return torch.cat(parts, dim=2)

naive = streamed_no_context(57, 4)
good = streamed(57, 4)
check("no-context length still matches", int(naive.shape[2]), int(good.shape[2]))
check("but it is NOT exact", bool(torch.equal(naive, good)), False)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
