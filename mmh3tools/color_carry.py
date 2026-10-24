"""Colour-matched carry for MMH3LoopingSampler: SoulX-FlashHead's anti-drift step, ported.

Chained chunks inherit each other's colour error through the carried latents, and amplify it. A 20 s static
talking-head chain (FastH3 Omni NVFP4, 512x512, 4 steps, 28 chunks, 5-frame carry) went from saturation 56 to
97 and face red-minus-blue 52 to 94 within 10 s: the "cook" long H3 runs are known for. SoulX-FlashHead streams
without it by decoding every chunk, matching its colours to the reference in Lab (each frame's L, a and b mean
and spread moved to the reference's, then blended by strength), and only then re-encoding the frames it carries
into the next chunk. This module provides that transfer; the sampler applies it to the carried latents, with
chunk 1 as the colour anchor. A latent-space variant (no decode or encode) sits at the end.
"""
import torch

from .common import LATENTS_PER_GROUP

# sRGB (D65) <-> CIE Lab. Written out here rather than borrowed so highlights are not clipped on the way in.
_RGB_TO_XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float64)
_XYZ_TO_RGB = torch.linalg.inv(_RGB_TO_XYZ)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def rgb_to_lab(rgb):
    """[..., 3] RGB in 0..1 -> [..., 3] Lab (float32)."""
    rgb = rgb.float().clamp(0.0, 1.0)
    lin = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = (lin @ _RGB_TO_XYZ.to(lin.device, torch.float32).T) / _WHITE.to(lin.device, torch.float32)
    f = torch.where(xyz > _EPS, xyz.clamp_min(1e-12) ** (1.0 / 3.0), (_KAPPA * xyz + 16.0) / 116.0)
    return torch.stack([116.0 * f[..., 1] - 16.0,
                        500.0 * (f[..., 0] - f[..., 1]),
                        200.0 * (f[..., 1] - f[..., 2])], dim=-1)


def lab_to_rgb(lab):
    """[..., 3] Lab -> [..., 3] RGB in 0..1 (out-of-gamut values clipped)."""
    fy = (lab[..., 0] + 16.0) / 116.0
    f = torch.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], dim=-1)
    xyz = torch.where(f ** 3 > _EPS, f ** 3, (116.0 * f - 16.0) / _KAPPA) * _WHITE.to(lab.device, torch.float32)
    lin = (xyz @ _XYZ_TO_RGB.to(lab.device, torch.float32).T).clamp_min(0.0)
    rgb = torch.where(lin > 0.0031308, 1.055 * lin.clamp_min(1e-12) ** (1.0 / 2.4) - 0.055, 12.92 * lin)
    return rgb.clamp(0.0, 1.0)


def lab_stats(frames):
    """[N, H, W, 3] frames -> (mean, std) of each Lab channel over every pixel of every frame: the anchor."""
    lab = rgb_to_lab(frames).reshape(-1, 3)
    return lab.mean(dim=0), lab.std(dim=0, unbiased=False)


def match_frames(frames, stats, strength=1.0):
    """Per frame, move each Lab channel's mean and spread to `stats`, then blend with the original by `strength`
    (FlashHead's transfer). [N, H, W, 3] in 0..1 -> same shape, float32."""
    mean, std = stats
    lab = rgb_to_lab(frames)
    m = lab.mean(dim=(1, 2), keepdim=True)
    s = lab.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
    corrected = lab_to_rgb((lab - m) * (std / s) + mean)
    return (1.0 - strength) * frames.float() + strength * corrected


def decode_frames(vae, video_latent):
    """Decode an H3 video latent to [N, H, W, 3] frames (the pack's usual flattening)."""
    px = vae.decode(video_latent.contiguous())
    if isinstance(px, tuple):
        px = px[0]
    if px.ndim == 5:
        px = px.reshape(-1, *px.shape[-3:])
    return px[..., :3]


def anchor_stats(vae, video_latent):
    """One anchor per batch item: the Lab stats of that item's decoded [B, C, T, H, W] latent."""
    return [lab_stats(decode_frames(vae, video_latent[b:b + 1])) for b in range(video_latent.shape[0])]


def match_carry(vae, video_latent, anchors, strength):
    """Decode a carry span (a whole clip, 5k+2 latents), colour-match each batch item to its anchor and re-encode.

    Returns (latent, drift): the corrected latent, or None when the encode came back a different shape, and the
    mean Lab distance of the span from its anchor BEFORE correction (CIE76 delta E of the means; ~2.3 is a
    just-noticeable difference), which is how far that chunk had drifted."""
    zs, drift = [], 0.0
    for b in range(video_latent.shape[0]):
        px = decode_frames(vae, video_latent[b:b + 1])
        drift += float((rgb_to_lab(px).reshape(-1, 3).mean(dim=0) - anchors[b][0]).norm())
        z = vae.encode(match_frames(px, anchors[b], strength))
        zs.append(z[0] if isinstance(z, tuple) else z)
    z = torch.cat(zs, dim=0)
    drift /= video_latent.shape[0]
    return (z if tuple(z.shape) == tuple(video_latent.shape) else None), drift


# ---- latent-space variant: no VAE round trip ------------------------------------------------------------------
# The same transfer done on the latents themselves: each latent frame's per-channel mean and spread (over H, W) are
# moved to the anchor's. H3 packs a 1-frame token at the start of every 5-latent group and 4-frame tokens after it,
# so the two kinds get separate anchor statistics. Both spans passed in start on group phase 0.

def _phases(t):
    return [i for i in range(t) if i % LATENTS_PER_GROUP == 0], [i for i in range(t) if i % LATENTS_PER_GROUP]


def latent_anchor_stats(video_latent):
    """One anchor per batch item: per-channel (mean, std) of the 1-frame tokens and of the 4-frame tokens of a
    [B, C, T, H, W] latent, or None for a kind the span does not contain."""
    out = []
    for b in range(video_latent.shape[0]):
        z = video_latent[b].float()
        kinds = []
        for idx in _phases(z.shape[1]):
            if not idx:
                kinds.append(None)
                continue
            x = z[:, idx].reshape(z.shape[0], -1)
            kinds.append((x.mean(dim=1), x.std(dim=1, unbiased=False)))
        out.append(kinds)
    return out


def match_carry_latent(video_latent, anchors, strength):
    """Move each latent frame of a [B, C, T, H, W] carry onto its token kind's anchor, blended by `strength`.

    Returns (latent, drift): drift is the mean distance of the frames' channel means from the anchor before
    correction, in anchor standard deviations."""
    out = video_latent.clone()
    drift, n = 0.0, 0
    for b in range(video_latent.shape[0]):
        for t in range(video_latent.shape[2]):
            a = anchors[b][0 if t % LATENTS_PER_GROUP == 0 else 1] or anchors[b][1] or anchors[b][0]
            am, asd = a[0].view(-1, 1, 1), a[1].view(-1, 1, 1).clamp_min(1e-6)
            x = video_latent[b, :, t].float()
            m = x.mean(dim=(1, 2), keepdim=True)
            s = x.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(1e-6)
            drift += float(((m - am).abs() / asd).mean())
            n += 1
            y = (x - m) * (asd / s) + am
            out[b, :, t] = ((1.0 - strength) * x + strength * y).to(out.dtype)
    return out, drift / max(n, 1)
