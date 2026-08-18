"""MMH3MotionOverload: the profile ranks tokens, and says when it is ranking noise."""

import json, math, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import torch

from comfy.nested_tensor import NestedTensor

from mmh3tools.common import FRAME_PER_TOKEN, frame_at_latent
from mmh3tools.nodes_motion import (MMH3MotionOverload as MO, contrast, hot_runs,
                                    jerk_profile, phase_normalise, span_frames)

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%r want=%r" % (got, want))
    if not ok:
        fails.append(label)

def close(label, got, want, tol=1e-6):
    ok = abs(float(got) - float(want)) <= tol
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%r want=%r" % (got, want))
    if not ok:
        fails.append(label)


def av(video):
    """Wrap a video tensor as an H3 AV latent; audio is never read."""
    t = video.shape[2]
    audio = torch.zeros(1, 32, 2, max(1, t * 8))
    return {"samples": NestedTensor([video, audio])}


def run(video, **kw):
    return MO.execute(av(video), **kw).result


print("\n1. phase normalisation divides out the (1,4,4,4,4) bias")
# a profile that is PURELY the grid: every phase-0 token 1.0, every other 4.0
prof = np.array([1.0 if i % 5 == 0 else 4.0 for i in range(20)])
flat = phase_normalise(prof)
check("a pure grid signal flattens to 1.0", [round(float(x), 9) for x in flat],
      [1.0] * 20)
check("phase_normalise does not mutate its input", list(prof[:2]), [1.0, 4.0])

print("\n2. the grid matches common.py, not a second copy of it")
# MAINodes' _tok_start_frame(t) = c*17 + (0 if i==0 else 4*(i-1)+1) for c,i=divmod(t,5)
for t in range(0, 23):
    c, i = divmod(t, len(FRAME_PER_TOKEN))
    theirs = c * 17 + (0 if i == 0 else 4 * (i - 1) + 1)
    check("token %2d starts at the same frame" % t, frame_at_latent(t), theirs)

print("\n3. a token run maps onto the frames it covers")
check("tokens [0,1) is the singleton frame 0", span_frames(0, 1, 124), (0, 0))
check("tokens [1,2) covers four frames", span_frames(1, 2, 124), (1, 4))
# token 35 = 7*5, so it is a phase-0 singleton at 7*17 = 119; a run running off
# the end of a 124-frame clip stops at the last real frame
check("a run ending past the clip is clamped", span_frames(35, 40, 124), (119, 123))

print("\n4. hot runs are contiguous half-open spans")
check("two bursts", hot_runs([0, 1, 1, 0, 0, 1, 0]), [(1, 3), (5, 6)])
check("a run touching the end closes", hot_runs([0, 1, 1]), [(1, 3)])
check("nothing hot", hot_runs([0, 0, 0]), [])

print("\n5. STATIC footage: nothing moves, so nothing should stand out")
torch.manual_seed(0)
still = torch.rand(1, 24, 22, 6, 10)
still = still[:, :, :1].expand(-1, -1, 22, -1, -1).contiguous()   # every token identical
_json, hot_cold, report = run(still)
d = json.loads(_json)
check("peak does not tower over a zero baseline", d["peak_over_median"], 1.0)
check("the cut separates nothing", hot_cold, 1.0)
check("and the report says the profile is flat", "NO variation" in report, True)
print("     (hot/cold %.4f on identical tokens -- this is the abstain signal)" % hot_cold)

print("\n6. NOISE: no burst, so the ranking should not find much contrast")
torch.manual_seed(1)
noise = torch.randn(1, 24, 37, 6, 10) * 0.1
_json, hot_cold_noise, _r = run(noise)
check("hot/cold stays near 1 on unstructured noise", hot_cold_noise < 1.5, True)
print("     (hot/cold %.4f)" % hot_cold_noise)

print("\n7. A BURST: one span of abrupt change should outrank the rest")
torch.manual_seed(2)
# real footage is never perfectly still, so the calm tokens carry a little drift:
# without it cold.mean() is exactly 0 and the ratio is unbounded, which tests the
# degenerate path rather than the one this node is for
v = torch.randn(1, 24, 37, 6, 10) * 0.02
base = torch.rand(1, 24, 1, 6, 10)
v = v + base
for t in range(16, 21):                       # alternate hard between two states
    v[:, :, t] += (2.0 if t % 2 else -2.0)
_json, hot_cold_burst, report = run(v)
d = json.loads(_json)
check("a burst outranks the calm tokens", hot_cold_burst > 3.0, True)
check("peak stands above the median", d["peak_over_median"] > 3.0, True)
check("the burst is reported as a span", len(d["spans"]) >= 1, True)
covered = any(s["frame_start"] <= frame_at_latent(18) <= s["frame_end"]
              for s in d["spans"])
check("the span covers the burst's frames", covered, True)
print("     (hot/cold %.2f vs %.2f on noise -- the ratio is what separates them)"
      % (hot_cold_burst, hot_cold_noise))

print("\n8. the quantile marks a fixed share whatever it is given")
for q in (0.5, 0.75, 0.9):
    _j, _hc, _r = run(noise, quantile=q)
    got = json.loads(_j)["hot_tokens"]
    want = int(round((1.0 - q) * 37))
    check("q=%.2f marks about %d of 37" % (q, want), abs(got - want) <= 2, True)

print("\n9. report and JSON agree, and the report says what it cannot do")
_json, _hc, report = run(v)
d = json.loads(_json)
check("tokens agree", ("%d tokens" % d["tokens"]) in report, True)
check("the ranking caveat is stated", "does not detect" in report, True)
check("profile length matches token count", len(d["profile"]), d["tokens"])
check("frames come from the grid", d["frames"], frame_at_latent(37))

print("\n10. too few tokens is reported, not crashed on")
_json, _hc, report = run(torch.rand(1, 24, 2, 6, 10))
check("says so plainly", "too few" in report, True)
check("no spans invented", json.loads(_json)["spans"], [])

print("\n11. a video-only latent is accepted and noted")
out = MO.execute({"samples": torch.rand(1, 24, 22, 6, 10)}).result
check("video-only runs", "MMH3 Motion Overload" in out[2], True)
check("and is noted", "video-only" in out[2], True)

print("\n12. phase_normalize=False leaves the grid in, and says so")
_json, _hc, report = run(noise, phase_normalize=False)
check("the profile reports it", json.loads(_json)["phase_normalized"], False)
check("the report warns", "phase normalisation OFF" in report, True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
