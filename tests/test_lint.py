import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_lint import lint_prompt, MMH3PromptLint

CLEAN = """subject_definitions:
<Picture 1> is a storyboard panel for [Shot 1], giving its framing and lighting.
<Subject 1> is the android in <Picture 1>, with large blue eyes and a white ceramic body.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] The target video shows <Subject 1> in a showroom.

retention_analysis:
<Subject 1> (appears in every shot): fully_preserved - her face and body are identical.
<Audio 1>: reference - its timbre guides the delivery without copying the signal.

detailed_description:
[Shot 1] A wide shot of the showroom. <Subject 1> (S1) turns and says: <d>[English] Hello.</d>
[Shot 2] At 00:05.000, the camera cuts to a close-up. She says in an off-screen voiceover:
<d>[English] I am here.</d> while her lips remain completely closed.

overall_soundscape:
A faint electronic hum and soft footsteps on a polished floor.

non_diegetic_music:
Synthesised marimba and sustained pad chords at a moderate tempo."""

BROKEN = """subject_definitions:
<Subject 1> is the android.

summary:
The target video shows her in a showroom.

retention_analysis:
<Subject 1> (S1) (appears in every shot): fully_preserved - identical throughout.

detailed_description:
A wide shot. [Shot 1] At 00:00.000, <Subject 2> and <Picture 3> appear.
[Shot 3] At 00:09.000, she says: <d>She says "Hello." (S1)</d>
[Shot 2] At 00:04.000, he says in an off-screen voiceover: <d>[English] Bye.</d> and walks off.

overall_soundscape:
Footsteps, and then <d>[English] a line of dialogue.</d>

non_diegetic_music:
An epic, uplifting orchestral swell."""

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)

def has(problems, snippet):
    return any(snippet in p for p in problems)

print("\n1. a clean Ref2VA prompt lints clean")
probs = lint_prompt(CLEAN, "Ref2VA", 8.0)
check("no problems", probs, [])

print("\n2. the broken prompt is caught, rule by rule")
probs = lint_prompt(BROKEN, "Ref2VA", 8.0)
for p in probs:
    print("     -", p)
for label, snip in [
    ("[Shot 1] timestamped",             "[Shot 1] carries a timestamp"),
    ("timestamps not increasing",        "not increasing"),
    ("shot numbers out of order",        "not 1..N"),
    ("cut past the end",                 "falls outside"),
    ("missing [Language] tag",            "no [Language] tag"),
    ("speaker ID inside <d>",            "speaker ID inside <d>"),
    ("delivery verb inside <d>",         "delivery verb inside <d>"),
    ("dialogue in double quotes",        "double quotes"),
    ("voiceover without lips-closed",    "lips-closed"),
    ("dialogue in soundscape",           "overall_soundscape contains dialogue"),
    ("mood word 'epic'",                 "'epic'"),
    ("mood word 'uplifting'",            "'uplifting'"),
    ("undefined <Subject 2>",            "<Subject 2> is used"),
    ("undefined <Picture 3>",            "<Picture 3> is used"),
    ("(Sx) in retention_analysis",       "retention_analysis"),
    ("summary has no task prefix",       "task type] prefix"),
]:
    check(label, has(probs, snip), True)

print("\n2b. the two formats put the style in DIFFERENT places")
# A: "[Shot 1] <style>, <shot 1>" -- style INSIDE shot 1
# B: "One or two style sentences BEFORE [Shot 1]."
# Requiring A's shape everywhere flagged correct Ref2VA prompts as malformed.
LEAD = """subject_definitions:
<Subject 1> is the doll.

summary:
[reference generation] x

retention_analysis:
<Subject 1>: fully_preserved - identical.

detailed_description:
%s

overall_soundscape:
hum.

non_diegetic_music:
steady synth bass at 128 BPM."""

check("format B ALLOWS a style lead-in",
      lint_prompt(LEAD % "Live-action, cinematic.\n[Shot 1] A doll on a platform.",
                  "Ref2VA", 0.0), [])
check("format B still needs a [Shot 1]",
      has(lint_prompt(LEAD % "Live-action, cinematic. A doll.", "Ref2VA", 0.0),
          "has no [Shot 1]"), True)
check("a timestamp in the lead-in is caught",
      has(lint_prompt(LEAD % "Live-action. At 00:02.000 she turns.\n[Shot 1] A doll.",
                      "Ref2VA", 0.0), "carry a timestamp"), True)
check("an over-long lead-in is caught",
      has(lint_prompt(LEAD % ("word " * 90 + "\n[Shot 1] A doll."), "Ref2VA", 0.0),
          "lead-in before [Shot 1] is"), True)
# format A is the opposite: prose before [Shot 1] is the error there
A_LEAD = ("integrated_multimodal_description: Live-action. [Shot 1] A doll.\n\n"
          "overall_soundscape: hum.\n\nnon_diegetic_music: synth bass at 128 BPM.")
check("format A rejects a lead-in",
      has(lint_prompt(A_LEAD, "I2VA", 0.0), "does not open with [Shot 1]"), True)

print("\n3. base mode expects the three-field format")
probs = lint_prompt(CLEAN, "T2VA", 8.0)
check("flags missing integrated_multimodal_description",
      has(probs, "missing section: integrated_multimodal_description"), True)

print("\n4. empty prompt")
check("reports empty", lint_prompt("", "Ref2VA", 8.0), ["prompt is empty"])

print("\n5. seconds=0 skips the duration check")
probs = lint_prompt(BROKEN, "Ref2VA", 0.0)
check("no duration problem", has(probs, "falls outside"), False)

print("\n6. node passes the prompt through and counts")
out, report, n = MMH3PromptLint.execute(CLEAN, "Ref2VA", 8.0, "warn").result
check("passthrough", out, CLEAN)
# the report leads with the mode it checked against -- a finding list that does not
# say which format it expected is unreadable when the mode itself is the mistake
check("clean report", report, "mode Ref2VA (widget) -- clean")
check("zero problems", n, 0)

print("\n7. on_problem=error stops the queue")
try:
    MMH3PromptLint.execute(BROKEN, "Ref2VA", 8.0, "error")
    check("raises", False, True)
except ValueError:
    check("raises", True, True)
out, report, n = MMH3PromptLint.execute(BROKEN, "Ref2VA", 8.0, "warn").result
check("warn does not raise, still counts", n > 10, True)

print("\n12. fields repeated PER SHOT -- the malformation that used to lint clean")
REPEATED = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

    integrated_multimodal_description: [Shot 1] 3D CG, a doll on a platform.

    overall_soundscape: Synth bass and percussion.

    non_diegetic_music: Synthwave at 128 BPM.

    integrated_multimodal_description: [Shot 2] At 00:05.833, the camera cuts to a factory floor.

    overall_soundscape: Industrial hum and metallic clanks.

    non_diegetic_music: Same track, bass lower.
"""
probs = lint_prompt(REPEATED, "I2VA", 0.0)
for x in probs:
    print("     -", x)
for f in ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"):
    check("%s duplication caught" % f, has(probs, "%s appears 2 times" % f), True)

print("\n13. INDENTED fields still bound correctly")
# without \\s* in the stop pattern, every section ran to the end of the document and
# every downstream check read the wrong text
from mmh3tools.nodes_lint import _section, _SECTIONS_A
body = _section(REPEATED, "integrated_multimodal_description", _SECTIONS_A)
check("body stops at the next field", "Synth bass" in body, False)
check("body is just shot 1", body.startswith("[Shot 1]"), True)
music = _section(REPEATED, "non_diegetic_music", _SECTIONS_A)
check("music does not swallow the rest", "[Shot 2]" in music, False)

print("\n14. the voiceover rule matches its OWN dialogue, and every occurrence")
# Two bugs, both surfaced by a report nobody could account for. The old pattern
#   says in an off-screen voiceover.*?</d>(.{0,120})
# (a) leapt across the whole document under re.S to ANY </d> and judged the text
# after that, and (b) CONSUMED the trailing window, hiding the next voiceover
# from finditer.
VO = "off-screen voiceover"
FILLER = " filler." * 30
for label, text, want in [
    ("correct usage",
     "She says in an %s <d>[English] we are disposable</d>. Her lips remain closed." % VO,
     False),
    ("correct, punctuated",
     "She says in an %s, calmly: <d>[English] hi</d>. Her lips remain closed." % VO,
     False),
    ("genuinely missing the statement",
     "She says in an %s <d>[English] disposable</d>. She walks off into the neon." % VO,
     True),
    # the phrase appears VERBATIM in the format rules as an instruction, so any
    # text carrying them plus an unrelated <d> later used to report a failure
    ("phrase in prose, unrelated <d> far later",
     'use the exact phrase "says in an %s", then state that lips remain closed.%s'
     "<d>[English] hi</d> she walks away" % (VO, FILLER),
     False),
    ("two voiceovers, both fine",
     "A says in an %s <d>[English] one</d>. Her lips remain closed. "
     "B says in an %s <d>[English] two</d>. Her lips remain closed." % (VO, VO),
     False),
    ("two voiceovers, the SECOND broken",
     "A says in an %s <d>[English] one</d>. Her lips remain closed. "
     "B says in an %s <d>[English] two</d>. She exits." % (VO, VO),
     True),
    ("two voiceovers, both broken",
     "A says in an %s <d>[English] one</d>. She exits. "
     "B says in an %s <d>[English] two</d>. She exits." % (VO, VO),
     True),
]:
    check(label, has(lint_prompt(text, "I2VA", 0.0), "lips-closed"), want)

# the finding quotes what it matched, so an unexplainable report is locatable
quoted = [x for x in lint_prompt(
    "She says in an %s <d>[English] disposable</d>. She walks off." % VO, "I2VA", 0.0)
    if "lips-closed" in x]
check("finding quotes its evidence", "disposable" in quoted[0], True)

print("\n14b. a DECORATED label is named as such, not reported as missing")
# Instruct models format the prompt as a document: **subject_definitions:** or
# ### subject_definitions. That IS a defect -- the text encoder receives those
# characters literally and H3 was trained on plain labels -- but six "missing
# section" lines read as "the model forgot everything" when one substitution fixes it.
from mmh3tools.nodes_lint import _SECTIONS_B
DECORATED = "\n\n".join("**%s:**\nx" % f for f in _SECTIONS_B)
probs = lint_prompt(DECORATED, "Ref2VA", 0.0)
check("all six diagnosed as decorated", sum("DECORATED" in x for x in probs), 6)
check("none called missing", any(x.startswith("missing section") for x in probs), False)
check("the offending text is quoted", any("**subject_definitions:" in x for x in probs), True)

check("a real absence still says missing",
      any(x == "missing section: subject_definitions"
          for x in lint_prompt("nothing here at all", "Ref2VA", 0.0)), True)

for label, form in [("heading", "### %s"), ("bullet", "- %s:")]:
    txt = "\n\n".join((form % f) + "\nx" for f in _SECTIONS_B)
    found = lint_prompt(txt, "Ref2VA", 0.0)
    check("%s form is diagnosed, not silently missed" % label,
          any("DECORATED" in x for x in found), True)

print("\n15. mode can be wired, so it cannot disagree with the system prompt")
from mmh3tools.nodes_lint import MMH3PromptLint as _L
from mmh3tools.nodes_prompt import MMH3TaskSystemPrompt as _T

check("TaskSystemPrompt emits mode",
      [o.display_name for o in _T.define_schema().outputs][-1], "mode")
check("lint takes mode_override last",
      [i.id for i in _L.define_schema().inputs][-1], "mode_override")

# CLEAN is the six-section format; linting it as a base mode reports the whole
# OTHER format missing, which reads like the LLM ignored its instructions
_, rep_wrong, n_wrong = _L.execute(CLEAN, "T2VA", 8.0, "warn").result
_, rep_right, n_right = _L.execute(CLEAN, "T2VA", 8.0, "warn", "Ref2VA").result
check("wrong mode invents problems", n_wrong > 0, True)
check("wired mode fixes it", n_right, 0)
check("report names the mode and its source", rep_right.startswith("mode Ref2VA (wired)"), True)
check("report says widget when not wired", rep_wrong.startswith("mode T2VA (widget)"), True)

try:
    _L.execute(CLEAN, "Ref2VA", 8.0, "warn", "mode: Ref2VA | format B").result
    check("a wrong wire is rejected", False, True)
except ValueError as e:
    check("a wrong wire is rejected", "not one of" in str(e), True)


print("\n15b. retention_analysis actually says what survives")
from mmh3tools.nodes_lint import _MARKERS

def _ret(body):
    return """subject_definitions:
<Subject 1> is the doll, defined by <Picture 1>.

summary:
[reference generation] x

retention_analysis:
%s

detailed_description:
[Shot 1] a doll.

overall_soundscape:
hum.

non_diegetic_music:
beat.""" % body

# The marker MENU echoed instead of chosen from. The section looks populated, every
# other check passes, and it says nothing about what survives -- which is its only job.
check("the menu echo is caught",
      has(lint_prompt(_ret("    visible: fully_preserved | attribute_transfer | weak_reference"),
                      "Ref2VA", 0.0), "repeats the marker MENU"), True)
check("a proper line lints clean",
      lint_prompt(_ret("<Subject 1>: attribute_transfer - traits carry, rendering is new."),
                  "Ref2VA", 0.0), [])
check("every marker is accepted",
      all(lint_prompt(_ret("<Subject 1>: %s - x" % m), "Ref2VA", 0.0) == [] for m in _MARKERS),
      True)
check("a line with no marker is caught",
      has(lint_prompt(_ret("<Subject 1>: she looks the same throughout."), "Ref2VA", 0.0),
          "has no marker"), True)
check("a Subject with no retention line is caught",
      has(lint_prompt(_ret("<Picture 1>: weak_reference - loose guide."), "Ref2VA", 0.0),
          "no retention_analysis line"), True)
# a Picture folded into a Subject definition gets no line of its own, so its absence
# must NOT be flagged -- only Subjects are required to carry a marker
check("a folded Picture is not required to have one",
      has(lint_prompt(_ret("<Subject 1>: fully_preserved - identical."), "Ref2VA", 0.0),
          "<Picture 1> is defined"), False)

print("\n15c. the system prompt contains no copyable marker MENU")
# The model was not ignoring the instruction -- it copied the nearest thing in the system
# prompt that LOOKED like output. The old text put
#     visible: fully_preserved | partially_preserved | attribute_transfer | weak_reference
# indented directly under the section name, which reads as a line to write. Catching the
# echo in the lint treats the symptom; removing the template removes the cause.
import re as _re
from mmh3tools.nodes_prompt import MMH3TaskSystemPrompt as _T
_sys = _T.execute("Ref2VA", False, True, False, False, True, False, 24.0, True).result[0]
check("no pipe-separated marker list survives",
      _re.findall(r"(?m)^.*\|\s*(?:%s).*$" % "|".join(_MARKERS), _sys), [])
check("a worked example is shown instead", "<Subject 1>: fully_preserved - " in _sys, True)
check("and copying a list is forbidden explicitly",
      "never write a list of them" in _sys, True)

print("\n16. MMH3ReplaceSection: the refiner returns a body, the node holds the structure")
from mmh3tools.nodes_prompt import MMH3ReplaceSection as RS

def spliced(body, section="detailed_description", mode="Ref2VA"):
    return RS.execute(CLEAN, body, section, mode).result[0]

out = spliced("[Shot 1] A doll on a platform, neon behind her.")
check("all six sections survive", [x for x in _SECTIONS_B if "\n%s:" % x in "\n"+out],
      _SECTIONS_B)
check("canonical order", [l[:-1] for l in out.splitlines() if l.endswith(":")], _SECTIONS_B)
check("the new body is in", "neon behind her" in out, True)
check("the old body is gone", "turns and says" in out, False)
check("other sections verbatim", "Synthesised marimba" in out, True)

# whatever decoration the model adds gets normalised away
for junk, label in [("```\n[Shot 1] x\n```", "code fences"),
                    ("detailed_description:\n[Shot 1] x", "repeated label"),
                    ("**detailed_description:**\n[Shot 1] x", "decorated label")]:
    o = spliced(junk)
    check("%s stripped" % label, o.count("detailed_description:"), 1)
    check("%s -> body intact" % label, "[Shot 1] x" in o, True)

# the failure that started this: a refiner returning ONLY the body can no longer
# lose the other five sections, because it never had them
check("spliced result lints clean", lint_prompt(spliced(
    "[Shot 1] A wide shot of the showroom. <Subject 1> (S1) turns and says: "
    "<d>[English] Hello.</d>"), "Ref2VA", 8.0), [])

try:
    RS.execute("no sections here", "[Shot 1] x", "detailed_description", "Ref2VA")
    check("a non-prompt original is refused", False, True)
except ValueError as e:
    check("a non-prompt original is refused", "too many to be a prompt" in str(e), True)
try:
    RS.execute(CLEAN, "x", "integrated_multimodal_description", "Ref2VA")
    check("a section from the other format is refused", False, True)
except ValueError as e:
    check("a section from the other format is refused", "not part of" in str(e), True)

print("\nReplaceSection refuses a whole prompt as the replacement")
_FULL = ("subject_definitions:\nA\n\nsummary:\nB\n\nretention_analysis:\nC\n\n"
         "detailed_description:\nBODY\n\noverall_soundscape:\nD\n\nnon_diegetic_music:\nE")
try:
    RS.execute(_FULL, _FULL, "detailed_description", "Ref2VA")
    check("nested prompt refused", False, True)
except ValueError as e:
    check("nested prompt refused", "is itself a prompt" in str(e), True)
ok = RS.execute(_FULL, "a plain new body", "detailed_description", "Ref2VA").result[0]
check("a real section body still splices", "a plain new body" in ok, True)
ok = RS.execute(_FULL, "she reads the summary: aloud", "detailed_description", "Ref2VA").result[0]
check("one header in prose is not a prompt", "reads the summary" in ok, True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
