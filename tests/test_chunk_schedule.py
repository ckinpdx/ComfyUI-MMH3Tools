"""MMH3ChunkSchedule: what it emits must tile in the REAL planner, not just its own maths."""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.common import FPS, on_grid, frames_to_latents, snap_frames
from mmh3tools.nodes_schedule import (MMH3ChunkSchedule as CS, chunk_count,
                                      is_av_exact, schedule_av_exact,
                                      from_groups, groups, reachable_overlaps,
                                      seconds_to_groups, seconds_to_groups as _s2g, solve)
from mmh3tools.nodes_windows import MMH3WindowPlan as WP

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%r want=%r" % (got, want))
    if not ok:
        fails.append(label)


def real_strides(total_f, window_f, overlap_f):
    """Chunk starts straight out of MMH3WindowPlan's own report."""
    L, O, cnt, tf, tt, rep, wf, of = WP.execute(
        total_f, window_f, overlap_f, "standard_static", 0).result
    starts = [int(ln.split("latents")[1].split("-")[0])
              for ln in rep.splitlines()[1:1 + cnt]]
    return cnt, sorted({starts[i + 1] - starts[i] for i in range(len(starts) - 1)})


print("\n1. group helpers round-trip on the 5j+2 grid")
for j in range(0, 40):
    check("groups(from_groups(%d))" % j, groups(from_groups(j)), j)
    check("from_groups(%d) on grid" % j, on_grid(from_groups(j)), True)

print("\n2. seconds land on the nearest achievable group")
check("60s",   seconds_to_groups(60.0), 84)
# 20s asks for 480 frames; the neighbours are 464 (j=27) and 481 (j=28), and 481
# is the nearer one -- this rounds to nearest, it does not floor
check("20s",   seconds_to_groups(20.0), 28)
check("3s",    seconds_to_groups(3.0), 4)
check("0.208s (the floor)", seconds_to_groups(5 / 24.0), 0)

print("\n3. chunk_count is the divisibility rule, not a guess")
check("tiles: c=85 a=31 b=4",  chunk_count(85, 31, 4), 3)
check("does not tile: c=85 a=27 b=3", chunk_count(85, 27, 3), None)
check("window bigger than the clip", chunk_count(10, 20, 2), None)
check("stride too small", chunk_count(85, 31, 30), None)

print("\n4. THE PROPERTY: anything it solves must be regular in the real planner")
cases = [(60, 20, 3), (60, 15, 2), (45, 12, 2), (90, 20, 4), (180, 35, 4),
         (30, 10, 1), (120, 25, 5), (75, 18, 3)]
for tot, win, ov in cases:
    tf, wf, of, n, secs, rep = CS.execute(float(tot), float(win), float(ov)).result
    cnt, strides = real_strides(tf, wf, of)
    check("%3ds/%2ds/%ds -> %d chunks, one stride" % (tot, win, ov, n),
          (cnt == n, len(strides)), (True, 1))

print("\n5. an explicit chunk count is honoured, and still tiles")
for want in (2, 3, 4, 5, 6):
    tf, wf, of, n, secs, rep = CS.execute(60.0, 20.0, 3.0, "keep total", want).result
    cnt, strides = real_strides(tf, wf, of)
    check("chunks=%d honoured and regular" % want, (n, cnt, len(strides)),
          (want, want, 1))

print("\n6. emitted frames are on the 17j+5 grid")
for tot, win, ov in cases:
    tf, wf, of, n, secs, rep = CS.execute(float(tot), float(win), float(ov)).result
    check("%ds total on grid" % tot, snap_frames(tf), tf)
    check("%ds window on grid" % tot, on_grid(frames_to_latents(wf)), True)

print("\n7. 'keep total' does not move the total; 'nearest' may")
base = seconds_to_groups(60.0)
c, a, b, n = solve(base, 27, 4, "keep total")
check("keep total holds c", c, base)
c2, a2, b2, n2 = solve(base, 27, 4, "nearest")
check("nearest stays within leeway", abs(c2 - base) <= 4, True)

print("\n8. 'fewer chunks' really does return fewer")
_c, _a, _b, n_near = solve(base, 27, 4, "keep total")
_c, _a, _b, n_few = solve(base, 27, 4, "fewer chunks")
check("fewer chunks <= nearest", n_few <= n_near, True)

print("\n9. an impossible chunk count is released, not raised")
# 60s can reach ANY count up to the cap (a=8,b=6 strides 2 groups for 39 chunks),
# so an impossible ask needs a short clip: 10s cannot be cut 20 ways at a
# minimum 2-group stride, since that needs n*stride <= total groups
tf, wf, of, n, secs, rep = CS.execute(10.0, 3.0, 0.5, "keep total", 20).result
check("still returns a usable schedule", n >= 2, True)
check("and says the count was released", "released" in rep, True)
tf, wf, of, n, secs, rep = CS.execute(60.0, 20.0, 3.0, "keep total", 39).result
check("39 chunks of 60s IS reachable, so nothing is released", "released" in rep, False)
check("and it is honoured exactly", n, 39)

print("\n10. seconds_per_chunk is the WINDOW, not the clip")
# wires straight into `seconds_per_chunk` on either scene-plan node, alongside
# chunk_count, so the writer cannot disagree with the schedule about chunk length
for tot, win, ov in cases:
    tf, wf, of, n, secs, rep = CS.execute(float(tot), float(win), float(ov)).result
    check("%3ds: seconds_per_chunk == window_frames/fps" % tot,
          round(secs, 6), round(wf / float(FPS), 6))
    check("%3ds: and is NOT the clip length" % tot,
          abs(secs - tf / float(FPS)) > 1.0, True)

print("\n11. the report states the move and the tiling proof")
tf, wf, of, n, secs, rep = CS.execute(60.0, 20.0, 3.0).result
check("names what moved", "asked for" in rep, True)
check("shows the divisibility", "% " in rep and "= 0" in rep, True)
check("no native-range claim", "native range" in rep, False)
check("lists the reachable overlaps", "reachable overlaps at" in rep, True)
check("and names the step rule", "the COUNT is the step" in rep, True)

print("\n12. a zero overlap is reported as hard cuts, not silently shipped")
tf, wf, of, n, secs, rep = CS.execute(60.0, 30.0, 0.0).result
if of <= 5:
    check("names the hard cut", "hard cut" in rep, True)
else:
    print("  SKIP  solver kept an overlap for this request (of=%d)" % of)

print("\n13. the overlap ladder: the chunk count IS the step")
c = seconds_to_groups(60.0)
for n in (2, 3, 4, 6):
    opts = reachable_overlaps(c, n)
    gaps = {opts[i + 1] - opts[i] for i in range(len(opts) - 1)}
    check("%d chunks -> overlaps spaced %d groups" % (n, n), gaps, {n})
    check("%d chunks -> every option tiles" % n,
          all(chunk_count(c, b + (c - b) // n, b) == n for b in opts), True)
check("3 chunks includes 17 and 32 latents, nothing between",
      [from_groups(b) for b in reachable_overlaps(c, 3)][:4], [2, 17, 32, 47])

print("\n14. the 40 Hz audio grid")
# only every third H3 run is whole on the audio clock: 39, 90, 141, 192, step 51
check("39 frames exact", is_av_exact(39), True)
check("56 frames not",   is_av_exact(56), False)
check("the exact runs step by 51",
      [f for f in range(5, 250, 17) if is_av_exact(f)], [39, 90, 141, 192, 243])

print("\n15. av_align=require delivers BOTH boundaries exact")
for want in (0, 2, 3, 4):
    args = (60.0, 20.0, 3.0, "keep total", want, "require")
    tf, wf, of, n, spc, rep = CS.execute(*args).result
    check("chunks=%d -> overlap exact" % want, is_av_exact(of), True)
    check("chunks=%d -> stride exact" % want, is_av_exact(wf - of), True)
    if want:
        check("chunks=%d honoured" % want, n, want)
    cnt, strides = real_strides(tf, wf, of)
    check("chunks=%d still tiles" % want, len(strides), 1)

print("\n16. require moves the TOTAL before it gives up the chunk count")
tf, wf, of, n, spc, rep = CS.execute(60.0, 20.0, 3.0, "keep total", 3, "require").result
check("count kept", n, 3)
check("and the note says the total moved, not the count",
      ("total was allowed to move" in rep) and ("count was released" not in rep), True)

print("\n17. ignore is unchanged, and says when nothing on the ladder aligns")
a = CS.execute(60.0, 20.0, 3.0, "keep total", 3, "ignore").result
b = CS.execute(60.0, 20.0, 3.0, "keep total").result
check("ignore is the default", a[:5], b[:5])
check("and flags the empty ladder", "none of them is on the 40 Hz audio grid" in a[5], True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
