"""Section-by-section prompt building.

The failure this replaces is a shape failure, not a wording one: chunk-by-chunk
generation asks for a complete arc in every chunk, so each one resolves. So the
things worth asserting are the ones that make the shape hold -- the shots stage
knows WHICH beat it is and refuses to resolve unless it is the last, definitions
are written once, and the beat sheet survives the round trip into a loop.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_scene import MMH3PromptPart as PART
from mmh3tools.nodes_scene import MMH3ScenePlanPrompt as PLAN
from mmh3tools.nodes_prompt import MMH3PromptAccumulate as ACC
from mmh3tools.nodes_prompt import MMH3ReplaceSection as REPL

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%r want=%r" % (got, want))
    if not ok:
        fails.append(label)


def plan(stage, brief="a lighthouse keeper", n=4, secs=8.0, **kw):
    return PLAN.execute(stage, brief, n, secs, **kw).result


SHEET = ("[reference generation] The keeper logs the tide. | "
         "[reference generation] The lamp stutters; a second set of prints. | "
         "[reference generation] The relief boat turns back. | "
         "[reference generation] The keeper writes the last entry.")

print("\n1. definitions: written once, and it says so")
sysp, rep = plan("definitions")
check("emits both sections", ("subject_definitions" in sysp
                              and "retention_analysis" in sysp), True)
check("says they are reused verbatim", "BYTE-IDENTICALLY" in sysp, True)
check("demands one retention line per label", "ONE LINE PER LABEL" in sysp, True)
check("carries the brief", "lighthouse keeper" in sysp, True)
check("...framed as a brief, never spoken", "No character may speak it" in sysp, True)
check("no beat sheet needed", "BEAT SHEET" in sysp, False)

print("\n2. beats: the arc lives here, and only here")
sysp, rep = plan("beats", n=6)
check("asks for exactly n summaries", "6 of them" in sysp, True)
check("nothing resolves early", "Nothing resolves before beat 6" in sysp, True)
check("last one lands", "Beat 6 is the only one allowed to land" in sysp, True)
check("escalation must be physical", "Escalate SOMETHING PHYSICAL" in sysp, True)
check("dialogue planned across the set, not per chunk",
      "No line may appear in more than one beat" in sysp, True)
check("...but not written here", "Do not write the" in sysp and "lines here" in sysp, True)
check("one chunk of duration is stated", "8.0 seconds" in sysp, True)
check("warns when there is no arc to plan",
      "no arc to plan" in plan("beats", n=1)[1], True)

print("\n3. shots: the per-beat stage knows where it sits")
mid, _ = plan("shots", n=4, beat_index=1, beat_sheet=SHEET)
check("names its own beat", "beat 2 of 4" in mid, True)
check("middle beat is forbidden to resolve",
      "still have to happen" in mid and "not settled" in mid, True)
check("gets the WHOLE sheet, not just its own beat",
      all(b.strip() in mid for b in SHEET.split("|")), True)
check("says what the user message is", "It is the beat to EXPAND" in mid, True)
check("points back and forward", ("from where beat 1 left off" in mid
                                  and "beat 3 can continue" in mid), True)

last, _ = plan("shots", n=4, beat_index=3, beat_sheet=SHEET)
check("only the last beat may land", "LAST beat" in last, True)
check("...and the middle one is told the opposite", "LAST beat" in mid, False)

first, _ = plan("shots", n=4, beat_index=0, beat_sheet=SHEET)
check("beat 1 does not re-establish nothing", "beat 0 left off" in first, True)

print("\n4. shots: the rules that produced the bad output are stated")
check("banality is scoped to speech only",
      "BANALITY RULE APPLIES TO SPEECH ONLY" in mid, True)
check("...explicitly, against a banal scene",
      "Never write a banal scene" in mid, True)
check("no beat may resolve into conversation",
      "resolves into conversation" in mid, True)
check("dialogue outside <d> except language + words",
      "go OUTSIDE <d>" in mid and "words go inside:" in mid, True)
check("quotes reserved for on-screen text", "NEVER put spoken lines" in mid, True)
check("[Shot 1] has no timestamp", "carries NO timestamp" in mid, True)
check("timestamps bounded by the chunk", "inside 8.0 seconds" in mid, True)

print("\n5. shots refuses to run blind")
try:
    plan("shots", n=4, beat_index=0)
    check("missing beat_sheet raises", False, True)
except ValueError as e:
    check("missing beat_sheet raises", "beat_sheet" in str(e), True)
    check("...and says why it matters", "own complete arc" in str(e), True)

_, rep = plan("shots", n=4, beat_index=9, beat_sheet=SHEET)
check("index past the end is clamped and reported", "clamped to 3" in rep, True)

print("\n6. definitions flow into the later stages")
DEFS = "subject_definitions:\n<Subject 1> the keeper, grey oilskin."
for stage, kw in (("beats", {}),
                  ("shots", {"beat_index": 0, "beat_sheet": SHEET})):
    s, r = plan(stage, definitions=DEFS, **kw)
    check("%s is given the labels" % stage, "<Subject 1> the keeper" in s, True)
    check("...and told to invent none", "invent none" in s, True)
    check("%s warns when they are missing" % stage,
          "may invent labels" in plan(stage, **kw)[1], True)

print("\n7. extra_rules is appended verbatim")
s, _ = plan("beats", extra_rules="Shoot it on 16mm.")
check("appended", s.rstrip().endswith("Shoot it on 16mm."), True)

print("\n8. PromptPart: one beat out of the sheet")
p, n, rep = PART.execute(SHEET, 1).result
check("count", n, 4)
check("piece 1", p, "[reference generation] The lamp stutters; a second set of prints.")
check("strips the separator whitespace", p.startswith("["), True)
check("report names the piece", "piece 1 of 4" in rep, True)

print("\n9. PromptPart survives what an LLM actually returns")
fenced = "```\n" + SHEET + "\n```"
check("code fences", PART.execute(fenced, 0).result[1], 4)
check("preamble-free after fences",
      PART.execute(fenced, 0).result[0].startswith("[reference"), True)
check("empty trailing separator is not a piece",
      PART.execute(SHEET + " |", 0).result[1], 4)
check("leading separator either", PART.execute("| " + SHEET, 0).result[1], 4)

print("\n10. PromptPart: count mismatch is a choice, not a crash")
p, n, rep = PART.execute(SHEET, 7).result
check("clamped repeats the last", p.endswith("the last entry."), True)
check("...and says so", "past the end" in rep, True)
try:
    PART.execute(SHEET, 7, "|", False)
    check("clamp off raises", False, True)
except ValueError as e:
    check("clamp off raises", "fewer beats than the run has chunks" in str(e), True)
try:
    PART.execute("   ", 0)
    check("empty text raises", False, True)
except ValueError as e:
    check("empty text raises", "nothing to split" in str(e), True)

print("\n11. what the definitions stage must produce for the loop to work")
check("asks for all six headers, in order",
      [plan("definitions")[0].find(s) for s in
       ("subject_definitions:", "summary:", "retention_analysis:",
        "detailed_description:", "overall_soundscape:", "non_diegetic_music:")],
      sorted(plan("definitions")[0].find(s) for s in
             ("subject_definitions:", "summary:", "retention_analysis:",
              "detailed_description:", "overall_soundscape:", "non_diegetic_music:")))
check("...leaving the two per-chunk ones bare",
      "ONLY the bare header for those two" in plan("definitions")[0], True)
# the failure that shipped: "output the six headers and nothing else" was read as
# "emit six bare headers", and every section came back empty
check("...but names the four that MUST carry content",
      "FOUR of them YOU WRITE" in plan("definitions")[0], True)
check("...and calls a blank reply a failure",
      "failed reply" in plan("definitions")[0], True)
check("...and forbids a repeated header",
      "EXACTLY ONCE" in plan("definitions")[0], True)
check("...and says why", "cannot be filled in later" in plan("definitions")[0], True)
check("sound world is film-wide too",
      "reused in every chunk" in plan("definitions")[0], True)

print("\n12. the whole loop: split -> replace -> accumulate -> multiprompt")
# exactly the skeleton the definitions stage asks the LLM for
TEMPLATE = ("subject_definitions:\n<Subject 1> the keeper, grey oilskin.\n"
            "summary:\n"
            "retention_analysis:\n<Subject 1>: fully_preserved - unchanged throughout.\n"
            "detailed_description:\n"
            "overall_soundscape:\nSurf under a running generator.\n"
            "non_diegetic_music:\nNone.")
acc = None
for i in range(4):
    beat, count, _ = PART.execute(SHEET, i).result
    p = REPL.execute(TEMPLATE, beat, "summary", "Ref2VA").result[0]
    p = REPL.execute(p, "shots for beat %d" % (i + 1),
                     "detailed_description", "Ref2VA").result[0]
    acc, n, _, _ = ACC.execute(p, acc).result

check("accumulated all four windows", n, 4)
# both layers separate on "|", so a beat carrying one would corrupt the
# multiprompt split downstream -- it cannot, because splitting on "|" is how the
# beat was produced in the first place
parts = [x.strip() for x in acc.split("|") if x.strip()]
check("...and the multiprompt split still sees exactly four", len(parts), 4)
check("every window carries its own beat",
      all(("The relief boat turns back" in acc, "the last entry" in acc)), True)
check("definitions are byte-identical in all four",
      acc.count("<Subject 1> the keeper, grey oilskin."), 4)
check("retention too", acc.count("<Subject 1>: fully_preserved"), 4)

print("\n13. defaults line up with the nodes on either side")
check("PromptPart's separator splits Accumulate's default ' | '",
      PART.execute("a | b | c", 2).result[0], "c")

print("\nprev_detailed: a whole prompt is reduced to its detailed_description")
from mmh3tools.nodes_scene import _prev_body
FULL = ("subject_definitions:\nA\n\nsummary:\nB\n\nretention_analysis:\nC\n\n"
        "detailed_description:\nTHE BODY\n\noverall_soundscape:\nD\n\nnon_diegetic_music:\nE")
body, note = _prev_body(FULL)
check("pulls the one section", body, "THE BODY")
check("and says it did", bool(note), True)
body, note = _prev_body("a bare shot description")
check("a bare body passes through", body, "a bare shot description")
check("with no note", note, None)
check("empty stays empty", _prev_body("")[0], "")

print("\nthe continuity block gets the BODY, never the whole prompt")
sysp = PLAN.execute("shots", "a brief", 3, 8.0, 1, "", "defs", "", FULL, "talking_head").result[0]
check("body reached the prompt", "THE BODY" in sysp, True)
check("the six-section skeleton did NOT", "retention_analysis:" in sysp.split("PREVIOUS CHUNK")[-1], False)

print("\ntalking_head shots: the shot marker is REQUIRED, not just described")
c0 = PLAN.execute("shots","brief",3,16.5,0,"","defs","","","talking_head").result[0]
c1 = PLAN.execute("shots","brief",3,16.5,1,"","defs","","prev body","talking_head").result[0]
check("chunk 0 asked to BEGIN with [Shot 1]", "BEGIN the section with [Shot 1]" in c0, True)
check("chunk 1 too", "BEGIN the section with [Shot 1]" in c1, True)
check("never [Shot 2]", "Never write a [Shot 2]" in c1, True)
# chunk 0 IS the opening; only later chunks are forbidden from opening
check("chunk 0 may still establish", "ALREADY RUNNING" in c0, False)
check("chunk 0 is told to establish", "FIRST chunk" in c0, True)
check("chunk 1 may not open", "ALREADY RUNNING" in c1, True)
for phrase in ("No fade in", "no cut", "no light coming up", "no camera settling"):
    check("chunk 1 forbids %r" % phrase, phrase.lower() in c1.lower(), True)
check("cinematic mode untouched", "ALREADY RUNNING" in
      PLAN.execute("shots","brief",3,16.5,1,"a beat sheet","defs","","prev","cinematic").result[0], False)

print("\nwardrobe belongs in subject_definitions, not in a shot")
defs = PLAN.execute("definitions","brief",3,16.5,0,"","","","","talking_head").result[0]
check("definitions DEMANDS the wardrobe", "STATE WHAT EVERY PERSON IS WEARING" in defs, True)
check("and gives the reason", "byte-identically in every chunk" in defs, True)
th = PLAN.execute("shots","brief",3,16.5,1,"","defs","","prev","talking_head").result[0]
check("talking_head shots lock appearance", "APPEARANCE COMES FROM THE DEFINITION" in th, True)
check("but action may still touch it", "sleeve pushed back" in th, True)
# cinematic is left alone: a costume change there can be a real beat
cin = PLAN.execute("shots","brief",3,16.5,1,"sheet","defs","","prev","cinematic").result[0]
check("cinematic shots untouched", "APPEARANCE COMES FROM THE DEFINITION" in cin, False)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
