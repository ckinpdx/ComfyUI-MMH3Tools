import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from mmh3tools.nodes_phrase_attention import MMH3PhraseAttention, phrase_span, _blocks, _common

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

class Layout:
    # 3 reference rows, then 5 leading vision positions + a 7-token prompt, then video
    segments = [(0, 3, "ref_img"), (3, 15, "text"), (15, 40, "video")]
    seq_len = 40

PROMPT = "a b X Y Z d e"
seen = {}
def backend(q, k, v, heads, mask=None, skip_reshape=False, **kw):
    seen["mask"] = mask
    return "out"

def call(model, rows=40, block=0, layout=Layout, mask=None):
    seen.clear()
    k = torch.zeros(1, 2, rows, 4)
    topts = {"block_index": block}
    if layout is not None:
        topts["minimax_h3_layout"] = layout
    over = model.model_options["transformer_options"]["optimized_attention_override"]
    return over(backend, k, k, k, 2, mask=mask, skip_reshape=True, transformer_options=topts)

def node(bias=0.7, blocks="", model=None, phrase="X Y Z"):
    return MMH3PhraseAttention.execute(model or Model(), Clip(), PROMPT, phrase, bias, blocks).result


print("\n1. locating the phrase")
check("span counted from the end", phrase_span(Clip(), PROMPT, "X Y Z"), (5, 2))
check("phrase at the very end", phrase_span(Clip(), PROMPT, "d e"), (2, 0))
check("shared prefix and suffix", _common([1, 2, 3, 4, 5], [1, 2, 9, 9, 9, 4, 5]), (2, 2))
check("prefix and suffix never overlap", _common([1, 1], [1, 1, 1]), (2, 0))
check("block ranges", _blocks("0-2, 10 45-46"), {0, 1, 2, 10, 45, 46})
check("no blocks means all", _blocks(""), set())

print("\n2. the bias lands on the phrase's rows of the packed sequence")
m, rep = node()
check("returns the backend's output", call(m), "out")
mask = seen["mask"]
check("one row, broadcast by the backend", list(mask.shape), [1, 40])
check("phrase rows 10..12 biased", [round(float(x), 2) for x in mask[0, 9:14]], [0.0, 0.7, 0.7, 0.7, 0.0])
check("nothing else touched", int((mask != 0).sum()), 3)
check("report gives the multiplier", "x2.01" in rep, True)

print("\n3. it leaves other attention calls alone")
call(m, rows=12)
check("token refiner (text only)", seen["mask"], None)
call(m, layout=None)
check("no layout", seen["mask"], None)
given = torch.ones(1, 40)
call(m, mask=given)
check("an existing mask is passed through", seen["mask"] is given, True)

print("\n4. block selection")
m, rep = node(blocks="10-20")
call(m, block=5)
check("outside the range", seen["mask"], None)
call(m, block=15)
check("inside the range", seen["mask"] is not None, True)
check("report names the blocks", "blocks 10-20" in rep, True)

print("\n5. chaining and the no-op")
base = Model()
check("bias 0 returns the same model", node(bias=0.0, model=base)[0] is base, True)
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

print("\n6. misuse fails loudly")
def raises(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return ""
check("phrase not in the prompt", "not in the prompt" in raises(lambda: node(phrase="Q")), True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
