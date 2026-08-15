"""Forced alignment: the parts that do not need a 3 GB model.

The node's contract is that it never alters a word -- so the things worth testing
without weights are the ones that could silently alter one anyway: tag stripping
that eats a sung line, section spans that point at the wrong lines, and the word
comparison that is supposed to catch a mismatch.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_align import (_is_structural, _words_of, parse_lyrics,
                                   sections_to_spans)

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%r want=%r" % (got, want))
    if not ok:
        fails.append(label)


LYRIC = """[Verse 1]
I counted every tide
The lamp kept its own time

[Chorus]
Nobody came
Nobody came for me

[Verse 2]
The boat turned back"""

print("\n1. tags become boundaries, never lyrics")
lines, sections, _ = parse_lyrics(LYRIC)
check("tag lines are not sung", lines, [
    "I counted every tide", "The lamp kept its own time",
    "Nobody came", "Nobody came for me", "The boat turned back"])
check("no bracket survives into the alignable text",
      any("[" in l for l in lines), False)
check("blank lines dropped", "" in lines, False)
check("three sections", [s["name"] for s in sections],
      ["Verse 1", "Chorus", "Verse 2"])
check("spans index the LINES, not the raw text",
      [(s["first_line"], s["last_line"]) for s in sections],
      [(0, 1), (2, 3), (4, 4)])

print("\n2. lyrics with no tags at all still align")
lines2, sections2, _ = parse_lyrics("one line\nanother line")
check("lines kept", lines2, ["one line", "another line"])
check("no sections invented", sections2, [])

print("\n3. a trailing tag with nothing sung under it is dropped")
_, s3, n3 = parse_lyrics("[Intro]\nsome words\n[Outro]")
check("only the section that has words", [s["name"] for s in s3], ["Intro"])
check("...and the untimeable one is REPORTED, not lost", n3["empty_sections"],
      ["Outro"])
check("...and it spans them", (s3[0]["first_line"], s3[0]["last_line"]), (0, 0))

print("\n4. a mid-line bracket is stripped, because in Suno output it is a tag")
# The cost of the other choice is worse: keeping it sends a word nobody sang to the
# aligner. A lyric containing literal brackets loses them -- so it is REPORTED as a
# stripped direction rather than disappearing quietly.
lines4, s4, n4 = parse_lyrics("she said [nothing] at all")
check("the bracket does not reach the aligner", lines4, ["she said at all"])
check("a direction is not a section", s4, [])
check("...and the removal is visible in the report", n4["directions"], ["nothing"])

print("\n5. section spans come from the aligned lines")
spans = [{"value": "I counted every tide", "start": 0.0, "end": 2.5},
         {"value": "The lamp kept its own time", "start": 2.6, "end": 5.0},
         {"value": "Nobody came", "start": 5.5, "end": 7.0},
         {"value": "Nobody came for me", "start": 7.1, "end": 9.4},
         {"value": "The boat turned back", "start": 10.0, "end": 12.0}]
out = sections_to_spans(sections, spans)
check("one per section", [s["value"] for s in out], ["Verse 1", "Chorus", "Verse 2"])
check("verse 1 spans both its lines", (out[0]["start"], out[0]["end"]), (0.0, 5.0))
check("chorus starts at its first line", (out[1]["start"], out[1]["end"]), (5.5, 9.4))
check("last section closes on the last line", (out[2]["start"], out[2]["end"]),
      (10.0, 12.0))

print("\n6. fewer aligned lines than tags claim: clamp, never index past the end")
short = sections_to_spans(sections, spans[:3])
check("sections that still have lines survive",
      [s["value"] for s in short], ["Verse 1", "Chorus"])
check("the chorus is clamped to what exists",
      (short[1]["start"], short[1]["end"]), (5.5, 7.0))
check("no crash on zero lines", sections_to_spans(sections, []), [])

print("\n7. the word comparison that guards the contract")
check("case and punctuation do not count",
      _words_of("Nobody came, for me!"), _words_of("nobody CAME for me"))
check("a dropped word is caught",
      _words_of("nobody came for me") == _words_of("nobody came me"), False)
check("a reordering is caught",
      _words_of("nobody came for me") == _words_of("came nobody for me"), False)
check("apostrophes split rather than vanish",
      _words_of("don't"), ["don", "t"])
check("empty is empty", _words_of(""), [])

print("\n8. the whole round trip a real call makes")
clean = "\n".join(lines)
aligned = " ".join(s["value"] for s in spans)
check("what goes to the aligner matches what comes back",
      _words_of(clean), _words_of(aligned))

print("\n9. pasted straight from Suno")
SUNO = """[Verse 1]
I counted every tide
[soft piano builds]
The lamp kept its own time
[Chorus] Nobody came for me
(ooh, ooh)
[Guitar Solo]
[Outro]
Nobody came"""
sl, ss, sn = parse_lyrics(SUNO)
check("an INLINE tag never reaches the aligner", "[Chorus] Nobody came for me" in sl, False)
check("...but its words do", "Nobody came for me" in sl, True)
check("...and it opens the section on THAT line",
      [(x["name"], x["first_line"]) for x in ss if x["name"] == "Chorus"], [("Chorus", 2)])
check("a direction tag is not a section",
      "soft piano builds" in [x["name"] for x in ss], False)
check("...it is reported instead", sn["directions"], ["soft piano builds"])
check("an instrumental section with no words is reported, not silently dropped",
      sn["empty_sections"], ["Guitar Solo"])
check("parentheticals are KEPT by default", "(ooh, ooh)" in sl, True)
check("...and removed on request",
      "(ooh, ooh)" in parse_lyrics(SUNO, strip_parentheticals=True)[0], False)
check("...which is reported too",
      parse_lyrics(SUNO, strip_parentheticals=True)[2]["parentheticals"], ["ooh, ooh"])

print("\n10. an UNCLOSED bracket is still a tag")
# a truncated paste gives "[Chorus" with no closer; left alone it reaches the
# aligner as a lyric and gets timed as a sung word, with no error anywhere
ul, us, un = parse_lyrics("[Chorus\nNobody came")
check("the fragment is not sung", ul, ["Nobody came"])
check("...and still opens its section", [x["name"] for x in us], ["Chorus"])
fl, fs, fn = parse_lyrics("I counted every tide [fade")
check("a trailing unclosed direction is stripped", fl, ["I counted every tide"])
check("...and reported", fn["directions"], ["fade"])
check("a stray close bracket is left alone (punctuation, not a tag)",
      parse_lyrics("I counted] every tide")[0], ["I counted] every tide"])

print("\n11. structural vs direction is decided on whole words")
for tag, want in (("Verse 1", True), ("Chorus", True), ("Guitar Solo", True),
                  ("Pre-Chorus", True), ("Instrumental", True), ("Outro", True),
                  ("soft piano builds", False), ("whispered", False),
                  ("building intensity", False), ("distorted", False)):
    check("%-20s -> %s" % (tag, "section" if want else "direction"),
          _is_structural(tag), want)
print("\n12. CURLY braces are tags too")
# From a real run: Suno wrote {bridge}, the parser only knew [], so it was aligned
# as a sung word AND its whole section was absorbed by the chorus above it.
cl, cs, cn = parse_lyrics("[verse 1]\nI pledge my flesh\n{bridge}\n"
                          "Take your pulse\n[chorus]\nBlessed be")
check("the brace tag is not sung", "{bridge}" in cl, False)
check("...and it IS a section", [x["name"] for x in cs],
      ["verse 1", "bridge", "chorus"])
check("...so the lines partition correctly",
      [(x["first_line"], x["last_line"]) for x in cs], [(0, 0), (1, 1), (2, 2)])
check("an unclosed brace is handled like an unclosed bracket",
      parse_lyrics("{chorus\nBlessed be")[0], ["Blessed be"])

print("\n13. diagnose separates a break from a misalignment")
from mmh3tools.nodes_align import diagnose
SEC = [{"value": "chorus", "start": 91.08, "end": 114.08}]
mid = diagnose([{"value": "runtime", "start": 93.46, "end": 93.46},
                {"value": "pure,", "start": 105.06, "end": 105.86}],
               [], SEC, 127.64)
check("a gap inside a line is called a misalignment",
      any("MID-LINE" in n for n in mid), True)
brk = diagnose([{"value": "stream.", "start": 78.0, "end": 80.0},
                {"value": "Blessed", "start": 91.08, "end": 91.64}],
               [], SEC, 127.64)
# with no audio it may only say "probably" -- certainty is reserved for the
# energy check, which is the only thing that can actually settle it
check("a gap landing on a section start is called a break",
      any("instrumental" in n and "probably a break" in n for n in brk), True)
check("...and is NOT called a misalignment", any("MID-LINE" in n for n in brk), False)

rep = diagnose([{"value": "x", "start": 0.0, "end": 1.0}],
               [{"value": "same line", "start": 0.0, "end": 2.2},
                {"value": "same line", "start": 10.0, "end": 24.8}], [], 30.0)
check("a repeated line timed wildly differently is flagged",
      any("stretched across a gap" in n for n in rep), True)
check("a line that appears once is not compared",
      any("stretched" in n for n in
          diagnose([], [{"value": "only once", "start": 0.0, "end": 9.0}], [], 30.0)),
      False)
check("zero-length words are flagged",
      any("zero-length" in n for n in
          diagnose([{"value": "be", "start": 5.0, "end": 5.0}], [], [], 30.0)), True)
check("a clean run says nothing",
      diagnose([{"value": "a", "start": 0.0, "end": 0.5},
                {"value": "b", "start": 0.5, "end": 1.0}], [], [], 30.0), [])

print("\n14. the audio settles what the timings cannot")
import numpy as np
SR = 16000
def stem(spans, total=40.0):
    x = np.zeros(int(total * SR), dtype="float32")
    for a, b in spans:
        i, j = int(a * SR), int(b * SR)
        x[i:j] = (np.random.rand(j - i) - 0.5).astype("float32") * 0.3
    return x

W = [{"value": "we", "start": 29.7, "end": 30.02},
     {"value": "outlast.", "start": 35.66, "end": 36.38}]
pause = diagnose(W, [], [], 40.0, samples=stem([(0, 30), (35.6, 40)]), sample_rate=SR)
check("a gap over real silence is cleared, not blamed",
      any("SILENT" in n and "not an error" in n for n in pause), True)
check("...and a voiced word beside it is NOT called stranded",
      any("placed on SILENCE" in n for n in pause), False)

sung = diagnose(W, [], [], 40.0, samples=stem([(0, 40)]), sample_rate=SR)
check("a gap over audible sound IS flagged",
      any("HAS AUDIO" in n for n in sung), True)
check("...and names the lever that fixes it",
      any("nonspeech_skip" in n for n in sung), True)
check("...without overclaiming that the sound is definitely singing",
      any("could" in n and "bleed" in n for n in sung), True)

# the real failure: words placed against the tail of the previous section,
# stranded on silence, with a zero-length word where the audio ran out
W2 = [{"value": "Blessed", "start": 1.0, "end": 1.8},
      {"value": "runtime", "start": 2.7, "end": 2.7},
      {"value": "pure,", "start": 15.0, "end": 15.8}]
strand = diagnose(W2, [], [], 40.0, samples=stem([(14.0, 40)]), sample_rate=SR)
check("words on silence are named", any("placed on SILENCE" in n for n in strand), True)

check("a sparse vocal stem does not flag everything",
      diagnose([{"value": "a", "start": 35.0, "end": 35.5},
                {"value": "b", "start": 36.0, "end": 36.5}],
               [], [], 40.0, samples=stem([(35, 37)]), sample_rate=SR), [])
check("with no audio it never claims evidence",
      any("SILENT" in n or "HAS VOCAL" in n
          for n in diagnose(W, [], [], 40.0)), False)


print("\n15. snapping a stuttered word back to its onset")
from mmh3tools.nodes_align import _envelope, snap_onsets

class _W:
    def __init__(self, w, a, b):
        self.word, self.start, self.end = w, a, b

def _stem(spans, total=40.0):
    x = np.zeros(int(total * SR), dtype="float32")
    for a, b in spans:
        i, j = int(a * SR), int(b * SR)
        x[i:j] = (np.random.rand(j - i) - 0.5).astype("float32") * 0.3
    return x

# a glitched refrain: the word stutters 30.1-35.4, then one clean utterance
ws = [_W("we", 29.7, 30.02), _W("outlast.", 35.66, 36.38)]
moved = snap_onsets(ws, _envelope(_stem([(0, 30.0), (30.1, 35.4), (35.66, 36.4)]), SR))
check("the late word is pulled back to where its audio starts",
      [(m[0], round(m[2], 1)) for m in moved], [("outlast.", 30.1)])
check("...and the word object really moved", round(ws[1].start, 1), 30.1)
check("...but never past the word before it", ws[1].start >= ws[0].end, True)

# a genuine pause must NOT be snapped into
ws2 = [_W("we", 29.7, 30.02), _W("outlast.", 35.66, 36.38)]
check("a real pause is left alone",
      snap_onsets(ws2, _envelope(_stem([(0, 30), (35.6, 40)]), SR)), [])
check("...and its timing is untouched", ws2[1].start, 35.66)

# one loud frame is a click, not an onset
ws3 = [_W("we", 29.7, 30.02), _W("outlast.", 35.66, 36.38)]
check("a single transient is not mistaken for an onset",
      snap_onsets(ws3, _envelope(_stem([(0, 30), (32.0, 32.05), (35.66, 36.4)]), SR)), [])

ws4 = [_W("a", 0.0, 1.0), _W("b", 2.0, 3.0)]
check("a gap under the threshold is never snapped",
      snap_onsets(ws4, _envelope(_stem([(0, 40)]), SR)), [])

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
