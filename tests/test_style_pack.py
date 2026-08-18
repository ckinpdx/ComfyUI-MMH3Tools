"""MMH3LoadSkill: one skill file per node, chained to stack."""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from mmh3tools.nodes_stylepack import (MMH3LoadSkill as LS, is_experiment,
                                       list_skills, read_skill,
                                       strip_frontmatter, styles_dir)

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%r want=%r" % (got, want))
    if not ok:
        fails.append(label)

def run(skill, previous="", enabled=True):
    return LS.execute(skill, previous, enabled).result

print("\n1. the folder is the menu")
skills = list_skills()
check("none is first", skills[0], "none")
check("shipped skills found", len(skills) > 5, True)
check("notes are not skills", any(s.startswith("_") for s in skills), False)
check("no extensions in the label", any("." in s for s in skills), False)

print("\n2. every shipped file has a body")
for name in skills[1:]:
    check("%-42s loads" % name, bool(read_skill(name)), True)

print("\n3. the filename carries the type")
check("experiment- is an experiment", is_experiment("experiment-styled-split-screen"), True)
check("look- is not", is_experiment("look-paper-collage"), False)
check("nothing selected is not", is_experiment("none"), False)

print("\n4. chaining: wiring order is stacking order")
a = run("look-paper-collage")[0]
b = run("typography-crt-phosphor", previous=a)[0]
c = run("experiment-styled-split-screen", previous=b)[0]
check("each link grows the text", len(a) < len(b) < len(c), True)
check("all three present",
      all(k in c for k in ("PAPER COLLAGE", "TYPOGRAPHIC IDENTITY", "SPLIT THE FRAME")), True)
check("order follows the chain",
      c.index("PAPER COLLAGE") < c.index("TYPOGRAPHIC IDENTITY") < c.index("SPLIT THE FRAME"),
      True)
check("blocks are separated by a blank line", c.count(chr(10) + chr(10)) >= 2, True)

print("\n5. a disabled node passes the chain through untouched")
check("passthrough", run("look-handdrawn-live", previous=a, enabled=False)[0], a)
check("...and says so", "passing" in run("none", previous=a, enabled=False)[1], True)

print("\n6. an experiment is flagged every time it is used")
t, r = run("experiment-styled-split-screen")
check("block emitted", "SPLIT THE FRAME" in t, True)
check("flagged untested", "EXPERIMENTAL" in r, True)
check("...and why it matters", "not as a known recipe" in r, True)
check("a look is quiet", "EXPERIMENTAL" in run("look-paper-collage")[1], False)

print("\n7. frontmatter never reaches the prompt")
check("stripped when present",
      strip_frontmatter("---" + chr(10) + "kind: look" + chr(10) + "---" + chr(10) + "BODY"),
      "BODY")
check("a bare file is untouched", strip_frontmatter("BODY"), "BODY")
check("an unterminated fence is left alone",
      strip_frontmatter("---" + chr(10) + "no end"), "---" + chr(10) + "no end")
check("no shipped file leaks it", any("kind:" in read_skill(s) for s in skills[1:]), False)

print("\n8. a missing file is reported, and nothing selected is silent")
check("missing is named", "missing or empty" in run("nonexistent-skill")[1], True)
check("none emits nothing", run("none")[0], "")
check("...quietly", "!" in run("none")[1], False)

print("\n9. a file dropped in appears with no ceremony")
tmp = os.path.join(styles_dir(), "zz_house_style.md")
open(tmp, "w", encoding="utf-8").write("MY HOUSE STYLE")
try:
    check("in the menu", "zz_house_style" in list_skills(), True)
    check("loads verbatim", run("zz_house_style")[0], "MY HOUSE STYLE")
finally:
    os.remove(tmp)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)