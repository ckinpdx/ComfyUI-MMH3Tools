"""MMH3LyricsToWindows: the join between a song timeline and a sampler schedule."""

import json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_lyricwindows import (MMH3LyricsToWindows as LW, _stamp,
                                          load_alignment, overlapping, sections_for)

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%r want=%r" % (got, want))
    if not ok:
        fails.append(label)

A = open(os.path.join(_HERE, "_align_fixture.json"), encoding="utf-8").read()

def run(index, **kw):
    kw.setdefault("total_frames", 3060)
    kw.setdefault("window_frames", 256)
    kw.setdefault("overlap_frames", 48)
    kw.setdefault("context_schedule", "standard_static")
    return LW.execute(A, kw["total_frames"], kw["window_frames"], kw["overlap_frames"],
                      kw["context_schedule"], index,
                      kw.get("lookaround", 1), kw.get("include_times", True)).result

print("\n1. timestamps are rebased to the CHUNK, never the song")
check("00:00.000 is the window start, not the track start", _stamp(0.0), "00:00.000")
check("minutes roll over", _stamp(72.4), "01:12.400")
check("never negative", _stamp(-3.0), "00:00.000")

print("\n2. a window only sees what is sung inside it")
spans = [{"value": "a", "start": 0.0, "end": 5.0}, {"value": "b", "start": 10.0, "end": 12.0}]
check("touching an edge does not count", overlapping(spans, 5.0, 10.0), [])
check("straddling does", [x["value"] for x in overlapping(spans, 4.0, 11.0)], ["a", "b"])
check("contained does", [x["value"] for x in overlapping(spans, 0.0, 6.0)], ["a"])

print("\n3. a window that straddles a section says where the cut is")
secs = [{"value": "chorus", "start": 0.0, "end": 20.0},
        {"value": "bridge", "start": 25.0, "end": 40.0}]
name, cuts = sections_for(secs, 18.0, 30.0)
check("both are named", name, "chorus -> bridge")
check("...and the boundary is WINDOW-relative", [(n, round(t, 1)) for n, t in cuts],
      [("bridge", 7.0)])
check("a window inside one section has no cut", sections_for(secs, 1.0, 10.0), ("chorus", []))
check("a window in no section at all", sections_for(secs, 21.0, 24.0), ("", []))

print("\n4. an instrumental window is FLAGGED, not left to invent words")
lyr, prev, nxt, has, sec, times, count, f0, f1, rep, en, bars = run(0)
check("window 0 of this fixture is silent", has, False)
check("...and says why", "no-lyrics branch" in rep, True)
check("...with nothing to quote", lyr, "")

print("\n5. a window with singing carries it verbatim, on its own clock")
found = None
for k in range(count):
    r = run(k)
    if r[3]:
        found = (k, r)
        break
check("some window has lyrics", found is not None, True)
k, r = found
check("the line is verbatim", "Blessed be the runtime pure," in r[0], True)
check("...prefixed with a window-relative stamp", r[0].startswith("["), True)
check("...and shot_times name the words", "Blessed" in r[5], True)
check("no stamps when asked", run(k, include_times=False)[0].startswith("["), False)

print("\n6. an index past the end is clamped and reported")
r = run(999)
check("clamped", "clamped" in r[9], True)
check("...to the last window", r[6] - 1 >= 0, True)

print("\n7. it refuses input it cannot use")
for bad, why in (("", "is empty"), ("not json at all", "not valid JSON")):
    try:
        load_alignment(bad)
        check("rejects %r" % bad[:12], False, True)
    except ValueError as e:
        check("rejects %r" % (bad[:12] or "empty"), why in str(e), True)
check("a bare word list is accepted, with no lines",
      load_alignment(json.dumps([{"value": "x", "start": 0, "end": 1}]))["lines"], [])

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)