import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import copy

import torch
from mmh3tools.nodes_looping_sampler import _chunk_guider
from mmh3tools.nodes_phrase_attention import (
    MMH3PhraseAttention, phrase_span, parse_schedule, bias_at, _blocks, _common, _runs)

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


class Clip:
    """One token per whitespace word."""
    vocab = {}
    def tokenize(self, text, **kw):
        return {"qwen3vl_32b": [[(self.vocab.setdefault(w, len(self.vocab)), 1.0)
                                 for w in text.split()]]}

class Model:
    def __init__(self, options=None):
        self.model_options = options or {}
    def clone(self):
        return Model({k: dict(v) for k, v in self.model_options.items()})

LEAD = 5                                    # vision positions ahead of the prompt

def cond_set(*prompts):
    return {"conds": [[[torch.zeros(1, LEAD + len(p.split()), 4), {}]] for p in prompts],
            "prompts": list(prompts)}

def layout(prompt_tokens, video=25):
    # 3 reference rows, the text segment, then video
    text = LEAD + prompt_tokens
    class Layout:
        segments = [(0, 3, "ref_img"), (3, 3 + text, "text"), (3 + text, 3 + text + video, "video")]
        seq_len = 3 + text + video
    return Layout

PROMPT = "a b X Y Z d e"
seen = {}
def backend(q, k, v, heads, mask=None, skip_reshape=False, **kw):
    seen["mask"] = mask
    return "out"

def call(model, lay=None, rows=None, block=0, mask=None, no_layout=False, chunk=None):
    seen.clear()
    lay = lay or layout(7)
    k = torch.zeros(1, 2, rows or lay.seq_len, 4)
    topts = {"block_index": block}
    if not no_layout:
        topts["minimax_h3_layout"] = lay
    if chunk is not None:
        topts["mmh3_chunk_index"] = chunk
    over = model.model_options["transformer_options"]["optimized_attention_override"]
    return over(backend, k, k, k, 2, mask=mask, skip_reshape=True, transformer_options=topts)

def node(cs=None, bias=0.7, blocks="", schedule="", model=None, phrase="X Y Z"):
    return MMH3PhraseAttention.execute(model or Model(), Clip(), cs or cond_set(PROMPT),
                                       phrase, bias, blocks, schedule).result

def biased(mask):
    return [i for i, x in enumerate(mask[0].tolist()) if x != 0]

def raises(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return ""


print("\n1. locating the phrase")
check("span counted from the end", phrase_span(Clip(), PROMPT, "X Y Z"), (5, 2))
check("phrase at the very end", phrase_span(Clip(), PROMPT, "d e"), (2, 0))
check("absent phrase", phrase_span(Clip(), PROMPT, "Q"), None)
check("shared prefix and suffix", _common([1, 2, 3, 4, 5], [1, 2, 9, 9, 9, 4, 5]), (2, 2))
check("prefix and suffix never overlap", _common([1, 1], [1, 1, 1]), (2, 0))
check("block ranges", _blocks("0-2, 10 45-46"), {0, 1, 2, 10, 45, 46})
check("no blocks means all", _blocks(""), set())
check("chunk runs", _runs([0, 1, 2, 5, 7, 8]), "0-2, 5, 7-8")

print("\n2. the bias lands on the phrase's rows of the packed sequence")
m, rep = node()
check("returns the backend's output", call(m), "out")
mask = seen["mask"]
check("one row, broadcast by the backend", list(mask.shape), [1, 40])
check("phrase rows 10..12 biased", biased(mask), [10, 11, 12])
check("with the bias value", round(float(mask[0, 10]), 2), 0.7)
check("report gives the multiplier", "x2.014" in rep, True)
check("report says every chunk", "every chunk: phrase is 3 tokens" in rep, True)

print("\n3. it leaves other attention calls alone")
call(m, rows=12)
check("token refiner (text only)", seen["mask"], None)
call(m, no_layout=True)
check("no layout", seen["mask"], None)
call(m, mask=torch.ones(1, 40))
check("an existing mask is added to", [round(x, 2) for x in seen["mask"][0, 9:14].tolist()], [1.0, 1.7, 1.7, 1.7, 1.0])
call(m, layout(9))
check("a text segment that is not this cond_set's prompt", seen["mask"], None)

print("\n4. block selection")
m, rep = node(blocks="10-20")
call(m, block=5)
check("outside the range", seen["mask"], None)
call(m, block=15)
check("inside the range", seen["mask"] is not None, True)
check("report names the blocks", "blocks 10-20" in rep, True)

print("\n5. a prompt per chunk, picked by the published chunk index")
# 0: phrase mid-prompt; 1: no phrase; 2: SAME length as 0, phrase elsewhere; 3: longer
cs = cond_set(PROMPT, "a b c d e f g", "a X Y Z b d e", "a b c d X Y Z e f")
m, rep = node(cs)
call(m, layout(7), chunk=0)
check("chunk 0's rows", biased(seen["mask"]), [10, 11, 12])
call(m, layout(7), chunk=1)
check("chunk without the phrase gets no bias", seen["mask"], None)
call(m, layout(7), chunk=2)
check("same length, different place: told apart by chunk", biased(seen["mask"]), [9, 10, 11])
call(m, layout(9), chunk=3)
check("chunk 3's rows follow its own prompt", biased(seen["mask"]), [12, 13, 14])
call(m, layout(9), chunk=7)
check("chunks past the end reuse the last prompt", biased(seen["mask"]), [12, 13, 14])
check("report names the prompt left alone", "prompt 1: phrase not in the prompt" in rep, True)

print("\n6. the schedule")
check("parse", parse_schedule("0: 0\n2: 6\n\n# note\n4:3"), [(0, 0.0), (2, 6.0), (4, 3.0)])
steps = parse_schedule("2: 6\n4: 3")
check("holds from its chunk until the next", [bias_at(steps, 1.5, i) for i in range(6)],
      [1.5, 1.5, 6.0, 6.0, 3.0, 3.0])
check("empty schedule is the base bias", bias_at([], 1.5, 9), 1.5)
check("a bad line is an error", "not 'chunk: bias'" in raises(lambda: parse_schedule("two: 6")), True)

m, rep = node(schedule="0: 0\n2: 6\n4: -3")       # one prompt, repeated
call(m, chunk=0)
check("chunk 0 scheduled off", seen["mask"], None)
call(m, chunk=2)
check("chunk 2 gets 6", float(seen["mask"][0, 10]), 6.0)
call(m, chunk=3)
check("chunk 3 still 6", float(seen["mask"][0, 10]), 6.0)
call(m, chunk=5)
check("chunk 5 gets -3", float(seen["mask"][0, 10]), -3.0)
check("report lays the schedule out",
      "chunk 0 to 1: +0.00" in rep and "chunk 2 to 3: +6.00" in rep and "chunk 4 on: -3.00" in rep, True)
m, _ = node(bias=0.0, schedule="3: 4")
call(m, chunk=1)
check("base bias 0 before the first line", seen["mask"], None)
call(m, chunk=3)
check("...and the schedule still applies after it", float(seen["mask"][0, 10]), 4.0)
m, _ = node(bias=2.0)
call(m)
check("no chunk published: treated as chunk 0", float(seen["mask"][0, 10]), 2.0)

print("\n7. the looping sampler publishes the chunk")
class Guider:
    original_conds = {"positive": "base"}
    model_options = {"transformer_options": {"keep": 1}}
    def set_conds(self, positive):
        self.original_conds["positive"] = positive
g = Guider()
g.original_conds = dict(Guider.original_conds)
c = _chunk_guider(g, "cond", frame0=120, chunk=3)
check("chunk index", c.model_options["transformer_options"]["mmh3_chunk_index"], 3)
check("alongside the frame offset", c.model_options["transformer_options"]["mmh3_control_frame0"], 120)
check("no mask carry by default", c.model_options["transformer_options"]["mmh3_carried_latents"], 0)
check("existing options kept", c.model_options["transformer_options"]["keep"], 1)
check("the wired guider is not written to", g.model_options, {"transformer_options": {"keep": 1}})

print("\n8. chaining and the no-op")
base = Model()
check("bias 0 and no schedule returns the same model", node(bias=0.0, model=base)[0] is base, True)
calls = []
def earlier(func, q, k, v, heads, mask=None, **kw):
    calls.append(mask)
    return func(q, k, v, heads, mask=mask, **kw)
base = Model({"transformer_options": {"optimized_attention_override": earlier}})
m, _ = node(model=base)
call(m)
check("the earlier override still runs, with the bias", calls[0] is seen["mask"] and calls[0] is not None, True)
check("the input model is not patched",
      base.model_options["transformer_options"]["optimized_attention_override"] is earlier, True)

print("\n9. misuse fails loudly")
check("phrase in no prompt", "not in any of" in raises(lambda: node(phrase="Q")), True)
check("a cond_set without text",
      "not in any of" in raises(lambda: node({"conds": cond_set(PROMPT)["conds"], "prompts": [""]})), True)

print("\n10. value_scale scales the phrase's value rows")
def vcall(model, reshape, block=0):
    seen.clear()
    lay = layout(7)
    q = torch.zeros(1, 2, lay.seq_len, 4) if reshape else torch.zeros(1, lay.seq_len, 8)
    v = torch.ones_like(q)
    topts = {"block_index": block, "minimax_h3_layout": lay}
    over = model.model_options["transformer_options"]["optimized_attention_override"]
    over(backend, q, q, v, 2, skip_reshape=reshape, transformer_options=topts)
    return v[0, 0, :, 0].tolist() if reshape else v[0, :, 0].tolist()
def vnode(bias, blocks, schedule, scale, model=None):
    return MMH3PhraseAttention.execute(model or Model(), Clip(), cond_set(PROMPT), "X Y Z",
                                       bias, blocks, schedule, scale).result
WANT = [1.0] * 10 + [-0.5] * 3 + [1.0] * 27
m, rep = vnode(0.0, "", "", -0.5)
check("split heads: only the phrase rows", vcall(m, True), WANT)
check("...and no bias with it", seen["mask"], None)
check("merged heads", vcall(m, False), WANT)
check("report names it", "value x-0.5" in rep, True)
m, _ = vnode(2.0, "10-20", "0: 0", 0.0)
check("outside the blocks", vcall(m, True, block=5), [1.0] * 40)
got = vcall(m, True, block=15)
check("applies where the schedule turns the bias off", (got[10:13], seen["mask"]), ([0.0] * 3, None))
base = Model()
check("bias 0 and value_scale 1 returns the same model", vnode(0.0, "", "", 1.0, base)[0] is base, True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
