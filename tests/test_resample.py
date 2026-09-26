"""resample_audio: core's resampler, torchaudio only as a fallback.

ComfyUI dropped torchaudio in #16457, so importing it unconditionally would make
MMH3ForcedAlign and MMH3MusicAnalysis raise ImportError on a clean current install.
The helper prefers comfy.audio.resample and keeps torchaudio for older cores.

Two things need proving. That the substitution is NUMERICALLY free -- these run on a
machine that still has torchaudio, so both can be compared directly. And that the
FALLBACK still works, which cannot be observed here at all: this core has
comfy.audio.resample, so the torchaudio branch is dead code unless it is forced.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch

from mmh3tools import common
from mmh3tools.common import resample_audio

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


torch.manual_seed(0)

print("\n1. identical rates are a passthrough, not a round trip")
x = torch.randn(1000)
check("same object returned", resample_audio(x, 16000, 16000) is x, True)
check("string rates coerced", resample_audio(x, "16000", 16000) is x, True)


print("\n2. bit-identical to torchaudio on the conversions this pack performs")
try:
    import torchaudio
    for sr, dst, shape in ((44100, 16000, (44100,)),     # ForcedAlign, 1-D
                           (48000, 16000, (48000,)),     # ForcedAlign, 1-D
                           (44100, 22050, (1, 44100))):  # MusicAnalysis shape
        sig = torch.randn(*shape)
        a = resample_audio(sig, sr, dst)
        b = torchaudio.functional.resample(sig, sr, dst)
        check("%d->%d %s shape" % (sr, dst, "x".join(map(str, shape))),
              tuple(a.shape), tuple(b.shape))
        check("%d->%d %s exact" % (sr, dst, "x".join(map(str, shape))),
              bool(torch.equal(a, b)), True)
except ImportError:
    print("  SKIP  torchaudio absent -- cannot cross-check (this is the future state)")


print("\n3. the torchaudio fallback is reachable when core has no resample")
# The branch is dead on this core, so hide comfy.audio.resample and force it.
import comfy.audio
_real = comfy.audio.resample
try:
    del comfy.audio.resample
    sig = torch.randn(44100)
    out = resample_audio(sig, 44100, 16000)
    check("fallback produced the right length", tuple(out.shape), (16000,))
    check("fallback matches core's result", bool(torch.equal(out, _real(sig, 44100, 16000))), True)
except ImportError:
    print("  SKIP  neither core nor torchaudio provides a resampler")
finally:
    comfy.audio.resample = _real
check("core resample restored", hasattr(comfy.audio, "resample"), True)


print("\n%s" % ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
