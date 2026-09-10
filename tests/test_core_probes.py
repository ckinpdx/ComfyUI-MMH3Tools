"""Core-capability probes must answer, never raise.

Each probe is source-inspected or behaviour-probed against whatever ComfyUI is
installed, so the real risk is not a wrong answer -- it is an exception escaping on
a core the probe was not written against, which would take a node down with it. All
of them swallow and return False, and this asserts that contract holds here.

masked_velocity_is_scaled additionally has to DISCRIMINATE, so it is checked against
the source directly rather than trusting itself.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import inspect

from mmh3tools.nodes_loop import (
    _per_row_masking_available,
    masked_velocity_is_scaled,
    per_row_mask_is_continuous,
)
from mmh3tools.nodes_looping_sampler import _guide_origin_correct, _guides_available

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


print("\n1. every probe returns a bool and does not raise")
for name, fn in (("_per_row_masking_available", _per_row_masking_available),
                 ("per_row_mask_is_continuous", per_row_mask_is_continuous),
                 ("masked_velocity_is_scaled", masked_velocity_is_scaled),
                 ("_guides_available", _guides_available),
                 ("_guide_origin_correct", _guide_origin_correct)):
    try:
        check(name, isinstance(fn(), bool), True)
    except Exception as e:
        print("  FAIL  %s raised %s: %s" % (name, type(e).__name__, e))
        fails.append(name)


print("\n2. masked_velocity_is_scaled agrees with the installed source")
try:
    import comfy.ldm.minimax.model as mm
    src = inspect.getsource(mm.MiniMaxH3Model.forward)
    check("matches '* denoise_mask' in forward",
          masked_velocity_is_scaled(), "* denoise_mask" in src)
    # The discriminator must not be satisfied by the pass-through kwarg alone.
    check("kwarg pass-through alone would not match",
          "* denoise_mask" in "denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask",
          False)
except Exception as e:
    print("  FAIL  source cross-check raised %s: %s" % (type(e).__name__, e))
    fails.append("source cross-check")


print("\n%s" % ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
