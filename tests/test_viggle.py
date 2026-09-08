"""MMH3CondSetFromViggle: the retype is trivial, the span check is the point.

Viggle bakes a per-window slice of the driving clip into each cond, so cond i is
correct only for Viggle's span i. The sampler plans its own spans and reads
conds[min(i, len(conds) - 1)] -- a schedule that differs but has the SAME chunk
count slips past the sampler's only guard and silently conditions every chunk on
the wrong frames. So the round trip that matters is: spans -> derived settings ->
_plan -> the same spans back.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_viggle import _compare, _derive, _frame_spans
from mmh3tools.nodes_windows import _plan, _window_frame_spans

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def mmh3_spans(total_f, chunk_frames, overlap_frames):
    _l, _ov, planned_f, _t, windows = _plan(
        total_f, chunk_frames, overlap_frames, "standard_static")
    return _window_frame_spans(windows, planned_f)


print("\n1. Viggle's 4-tuple spans reduce to inclusive frame pairs")
check("latent bookkeeping dropped",
      _frame_spans([(0, 100, 0, 30), (85, 185, 25, 30)]),
      [(0, 100), (85, 185)])


print("\n2. round trip: a schedule this pack planned is recovered exactly")
# Standing in for Viggle: any real schedule on the shared 17f/5-latent grid.
for total_f, cf, ovf in ((408, 192, 22), (816, 192, 22), (1200, 288, 39)):
    spans = mmh3_spans(total_f, cf, ovf)
    if len(spans) < 2:
        print("  skip  %d frames -> single chunk, no overlap to recover" % total_f)
        continue
    d_cf, d_ovf = _derive(spans)
    back = mmh3_spans(total_f, d_cf, d_ovf)
    aligned, _lines = _compare(spans, back)
    check("total_f=%d cf=%d ov=%d re-plans identically" % (total_f, cf, ovf),
          aligned, True)


print("\n3. a single-chunk set derives zero overlap rather than a negative one")
check("one span", _derive([(0, 191)]), (192, 0))


print("\n4. equal chunk COUNT with different boundaries is still reported")
# The exact case the sampler's own count check cannot see.
a = [(0, 100), (85, 185)]
b = [(0, 100), (70, 170)]
aligned, lines = _compare(a, b)
check("not aligned", aligned, False)
check("names the offending chunk", any("chunk 1" in l for l in lines), True)
check("same count, so the count guard would have passed", len(a) == len(b), True)


print("\n5. identical spans report a match")
aligned, lines = _compare(a, a)
check("aligned", aligned, True)
check("single summary line", len(lines), 1)


print("\n%s" % ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
