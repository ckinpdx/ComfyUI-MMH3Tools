"""Does a fully-pinned audio carry reach the model unchanged?

`overlap_strength_audio = 1.0` sets the audio mask to 0, so the region comes back
out of the sampler exactly as it went in -- that part is faithful by construction.
What the MODEL sees during sampling is a different value, and that is what the
rest of the chunk is generated from:

    latent_image  holds   audio_scale * x_audio        (process_latent_in)
    scale_latent_inpaint multiplies by (Sv/Sa)/audio_scale
    the model's forward  multiplies by (Sa/Sv)          (undoing the carry)

Three scalings that must cancel to 1. If they do not, every chunk is generated
against a distorted view of the carried voice, that output becomes the next
chunk's carry, and the distortion compounds -- worse the HARDER you pin, since
more of the injected value reaches the model. A fixed ref_audio would not help,
because references travel a different path entirely.

The two sides read their shifts from DIFFERENT places, which is the thing worth
pinning down: scale_latent_inpaint takes model_sampling.shift/audio_shift, while
the forward takes transformer_options["minimax_h3_sigma_shift_video"/"_audio"]
and only falls back to the model's own. Anything that overrides those in
transformer_options breaks the cancellation without touching either half.
"""

import os, sys, types
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch

import comfy.model_base as model_base
import comfy.utils as utils
from comfy.ldm.minimax.model import time_shift_sigma

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%r want=%r" % (got, want))
    if not ok:
        fails.append(label)

def close(label, got, want, tol):
    ok = abs(float(got) - float(want)) <= tol
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%.6f want=%.6f" % (got, want))
    if not ok:
        fails.append(label)


SHIFT, AUDIO_SHIFT = 5.0, 3.0
AUDIO_SCALE = SHIFT / AUDIO_SHIFT          # ModelSamplingAV.audio_scale
V_SHAPE = torch.Size([1, 24, 22, 6, 10])
A_SHAPE = torch.Size([1, 32, 2, 40])


class Stub:
    latent_shapes = [V_SHAPE, A_SHAPE]
    diffusion_model = types.SimpleNamespace(patch_size=(1, 2, 2))
    model_sampling = types.SimpleNamespace(shift=SHIFT, audio_shift=AUDIO_SHIFT,
                                           audio_scale=AUDIO_SCALE)
    def audio_scale(self):
        return AUDIO_SCALE


stub = Stub()
# bind every helper the method reaches; #15375 as MERGED refactored the pooling
# and quantizing out into _token_grid_masks (and switched round -> ceil there),
# which is exactly the kind of change this test exists to notice
for name in ("scale_latent_inpaint", "_pool_masks_to_token_grid", "_token_grid_masks"):
    setattr(stub, name, types.MethodType(getattr(model_base.MiniMaxH3, name), stub))


def round_trip(sigma_v, model_shift_v=SHIFT, model_shift_a=AUDIO_SHIFT):
    """x_audio -> latent_image -> injected -> what the model recovers."""
    torch.manual_seed(0)
    x0_a = torch.randn(A_SHAPE)                     # the true carried audio
    x0_v = torch.randn(V_SHAPE)
    # process_latent_in scales the audio half by audio_scale
    latent_image, _ = utils.pack_latents([x0_v, x0_a * AUDIO_SCALE])
    noise, _ = utils.pack_latents([torch.randn(V_SHAPE), torch.randn(A_SHAPE)])
    x, _ = utils.pack_latents([torch.randn(V_SHAPE), torch.randn(A_SHAPE)])
    # strength 1.0 -> mask 0 on the audio half (fully preserved)
    vm = torch.ones(V_SHAPE)
    am = torch.zeros(A_SHAPE)
    dm, _ = utils.pack_latents([vm, am])

    ret = stub.scale_latent_inpaint(sigma=torch.tensor([sigma_v]), noise=noise,
                                    latent_image=latent_image, x=x, denoise_mask=dm)
    seen = x * dm + ret * (1.0 - dm)                 # what the sampler hands the model
    seen_a = utils.unpack_latents(seen, [V_SHAPE, A_SHAPE])[1]

    # the model's forward undoes the carry with ITS OWN shifts
    sv = torch.tensor([sigma_v]).clamp(min=1e-6)
    sa = time_shift_sigma(sv, model_shift_v, model_shift_a)
    recovered = seen_a * (sa / sv)
    return x0_a, recovered


print("\n1. shifts agree -- the three scalings must cancel to 1")
for sigma in (1.0, 0.9, 0.7, 0.5, 0.3, 0.15, 0.05, 0.01):
    x0, rec = round_trip(sigma)
    err = (rec - x0).abs().max().item()
    close("sigma %.2f: model recovers the carried audio" % sigma, err, 0.0, 2e-5)

print("\n2. and the injected value is NOT the stored one -- it is scaled on purpose")
torch.manual_seed(0)
x0_a = torch.randn(A_SHAPE)
sv = torch.tensor([0.15])
sa = time_shift_sigma(sv, SHIFT, AUDIO_SHIFT)
factor = float((sv / sa) / AUDIO_SCALE)
print("     at sigma 0.15 the audio half is multiplied by %.4f before the model" % factor)
# audio_scale is shift/audio_shift = 5/3, so the factor sits NEAR 1 and the
# interesting property is that it cancels, not that it is large
check("the factor is not exactly 1", abs(factor - 1.0) > 0.01, True)
check("but it cancels against the model's (Sa/Sv) and audio_scale",
      abs(factor * float(sa / sv) * AUDIO_SCALE - 1.0) < 1e-6, True)

print("\n3. THE FRAGILE PART: the two halves read their shifts from different places")
print("     scale_latent_inpaint -> model_sampling.shift / .audio_shift")
print("     forward              -> transformer_options['minimax_h3_sigma_shift_*']")
worst = 0.0
for sigma in (0.5, 0.15, 0.05):
    for ms_a in (2.5, 3.5, 4.0):                     # an override that disagrees
        x0, rec = round_trip(sigma, model_shift_a=ms_a)
        ratio = (rec.abs().mean() / x0.abs().mean()).item()
        worst = max(worst, abs(ratio - 1.0))
        print("     sigma %.2f, forward audio_shift %.1f vs inpaint %.1f -> carried audio x%.3f"
              % (sigma, ms_a, AUDIO_SHIFT, ratio))
check("a shift disagreement really does mis-scale the carry", worst > 0.05, True)

print("\n4. is anything in the live install setting those overrides?")
import subprocess
hits = subprocess.run(
    ["grep", "-rn", "minimax_h3_sigma_shift", "--include=*.py",
     os.path.abspath(os.path.join(_HERE, "..", "..", "..", "comfy")),
     os.path.abspath(os.path.join(_HERE, "..", "..", "..", "comfy_extras")),
     os.path.abspath(os.path.join(_HERE, "..", "..")),
     ], capture_output=True, text=True).stdout.strip().splitlines()
for h in hits:
    print("     " + h.split("ComfyUI")[-1][:120])
check("the override key exists in core", any("sigma_shift" in h for h in hits), True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
