"""Music-video prompt building: the rules that differ from the cinematic planner."""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_musicscene import MMH3MusicScenePlanPrompt as P

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%r want=%r" % (got, want))
    if not ok:
        fails.append(label)

LY = "[00:00.000] Blessed be the runtime pure," + chr(10) + "[00:02.900] OmniLord is the only cure."
TIMES = "00:00.000 Blessed, 00:00.440 be, 00:02.900 OmniLord"

def run(stage, **kw):
    kw.setdefault("brief", "a machine cult")
    kw.setdefault("chunk_count", 8)
    kw.setdefault("seconds_per_chunk", 19.3)
    kw.setdefault("typography", "off")
    return P.execute(stage, kw.pop("brief"), kw.pop("chunk_count"),
                     kw.pop("seconds_per_chunk"), kw.pop("typography"), **kw).result

print("\n1. definitions: the song IS the score")
sysp, rep = run("definitions")
check("does not ask for an invented score",
      "do not" in sysp.lower() and "compose an alternative" in sysp, True)
check("says the song carries the sound", "THE SONG IS THE AUDIO" in sysp, True)
check("still emits the six headers", sysp.count("subject_definitions:") >= 1, True)
check("...and names the four that must be written", "FOUR of them YOU WRITE" in sysp, True)

print("\n2. beats: the arc is the song, NOT an escalation")
sysp, rep = run("beats", lyrics=LY)
check("forbids inventing an escalation", "Do NOT invent an escalation" in sysp, True)
check("a repeated chorus should feel repeated",
      "should FEEL like the same chorus" in sysp, True)
check("...which is the OPPOSITE of the cinematic rule",
      "nothing resolves before" in sysp.lower(), False)
check("carries the lyric", "OmniLord is the only cure." in sysp, True)
check("warns when the lyric is missing",
      "cannot follow the song" in run("beats")[1], True)

print("\n3. typography is rationed in BEATS, not per chunk")
off = run("beats", lyrics=LY, typography="off")[0]
check("silent when off", "ON-SCREEN TEXT" in off or "TYPOGRAPHY" in off, False)
ex = run("beats", lyrics=LY, typography="exact lyrics")[0]
check("assigned across the whole song", "MOST SHOULD" in ex, True)
check("...verbatim mode says so", "VERBATIM" in ex, True)
bu = run("beats", lyrics=LY, typography="text bursts")[0]
check("...burst mode allows a fragment", "does not have to be" in bu, True)
# "much" on screen: burst mode picked the shortest word in the line, because
# nothing said the fragment has to carry meaning on its own
sh = run("shots", beat_index=0, lyrics=LY, shot_times=TIMES, beat_sheet="a",
         typography="text bursts")[0]
check("a burst must stand alone", "MEAN SOMETHING ALONE" in sh, True)
check("...function words banned by name", '"much"' in sh, True)
check("...with a concrete test", "poster" in sh, True)
check("beats is told to pick lines worth quoting",
      "worth putting on a screen" in bu, True)

# observed: typography rendered flat, like a one-word subtitle. Saying where the
# text "sits" is satisfied by centring it, which IS the subtitle look.
for mode in ("text bursts", "exact lyrics"):
    t = run("shots", beat_index=0, lyrics=LY, shot_times=TIMES, beat_sheet="a",
            typography=mode)[0]
    check("%-13s treats text as design" % mode, "NOT A SUBTITLE" in t, True)
    check("%-13s makes scale the first choice" % mode, "SCALE FIRST" in t, True)
    check("%-13s asks it to land on an onset" % mode, "word onset or a bar line" in t, True)
    check("%-13s keeps one treatment per chunk" % mode,
          "One treatment per chunk" in t, True)
# the validated fix from 2026-08-12: "intone their meaning" was read as NEVER emit
# a string, so H3 was handed no text and drew invented glyphs
bt = run("shots", beat_index=0, lyrics=LY, shot_times=TIMES, beat_sheet="a",
         typography="text bursts")[0]
check("burst is split into CHOOSE then RENDER",
      "1. **CHOOSE**" in bt and "2. **RENDER**" in bt, True)
check("...with the three-word all-caps spec", "THREE WORDS, ALL CAPS" in bt, True)
check("...and a validity rule against describing text it never quotes",
      "VALIDITY RULE" in bt and "invalid" in bt, True)
check("...naming the failure that produced gibberish", "cascade in neon" in bt, True)

# the one rule that survived MiniMax's own MV/subtitle skill review
dt = run("definitions")[0]
check("typography cannot leak into subject_definitions",
      "NO TYPOGRAPHY INSTRUCTIONS IN HERE" in dt, True)
check("...because it would reuse in every chunk", "leak into every chunk" in dt, True)

# from brand-promo-video-generator, the one portable idea in the three format skills
mc = run("shots", beat_index=0, lyrics=LY, shot_times=TIMES, beat_sheet="a")[0]
check("cuts are motivated by something carrying across",
      "MOTIVATE THE CUT" in mc and "Say what carries" in mc, True)
check("...without displacing the onset placement",
      "word onset or a bar line" in mc, True)
check("...and not on every cut", "read as a showreel" in mc, True)

check("typography off says nothing about design",
      "NOT A SUBTITLE" in run("shots", beat_index=0, lyrics=LY, shot_times=TIMES,
                              beat_sheet="a")[0], False)

print("\n4. shots: cut on the words it was GIVEN")
sysp, rep = run("shots", beat_index=2, lyrics=LY, shot_times=TIMES,
                section="chorus", beat_sheet="a | b | c")
check("names its chunk", "chunk 3 of 8" in sysp, True)
check("names its section", "chorus" in sysp, True)
check("forbids inventing a timestamp", "DO NOT invent a timestamp" in sysp, True)
check("carries the verbatim lines", "Blessed be the runtime pure," in sysp, True)
check("carries the onsets", "00:00.440 be" in sysp, True)
check("warns when onsets are missing",
      "will invent timestamps" in run("shots", beat_index=0, lyrics=LY,
                                      beat_sheet="a")[1], True)

print("\n5. an instrumental window gets its own branch")
sysp, rep = run("shots", beat_index=0, has_lyrics=False, beat_sheet="a")
check("says nothing is sung", "NOTHING IS SUNG IN THIS CHUNK" in sysp, True)
check("forbids singing", "Nobody is singing here" in sysp, True)
check("...and forbids on-screen text there",
      "NO on-screen text" in sysp, True)
check("report marks it", "INSTRUMENTAL" in rep, True)
ty = run("shots", beat_index=0, has_lyrics=False, beat_sheet="a",
         typography="exact lyrics")[0]
check("typography cannot sneak in with no line to quote",
      "ON-SCREEN TEXT: VERBATIM" in ty, False)

print("\n6. shots refuses to run blind")
try:
    run("shots", beat_index=0, beat_sheet="a")
    check("empty lyrics with has_lyrics raises", False, True)
except ValueError as e:
    check("empty lyrics with has_lyrics raises", "invents them" in str(e), True)

print("\n7. context and clamping")
sysp, rep = run("shots", beat_index=99, lyrics=LY, shot_times=TIMES,
                beat_sheet="a", context_lyrics="the line before")
check("index clamped", "clamped to 7" in rep, True)
check("context is marked do-not-write", "do not write these" in sysp, True)
check("warns with no beat sheet",
      "cannot see the song" in run("shots", beat_index=0, lyrics=LY,
                                   shot_times=TIMES)[1], True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)