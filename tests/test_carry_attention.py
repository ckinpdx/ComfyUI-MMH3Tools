import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.ldm.minimax.model import PackedLayout
from mmh3tools.nodes_carry_attention import MMH3CarryAttention
from mmh3tools.nodes_looping_sampler import _chunk_guider
from mmh3tools.nodes_phrase_attention import MMH3PhraseAttention

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


class Model:
    def __init__(self, options=None):
        self.model_options = options or {}
    def clone(self):
        return Model({k: dict(v) for k, v in self.model_options.items()})

seen = {}
def backend(q, k, v, heads, mask=None, skip_reshape=False, **kw):
    seen["mask"] = mask
    return "out"

# the live layout: 12 text rows, 6 video latents of 4x6 (6 rows each), 10 audio latents
LAYOUT = PackedLayout(12, 6, 4, 6, 10)
VIDEO = next(a for a, b, kind in LAYOUT.segments if kind == "video")

def call(model, carried=None, rows=None, block=0, mask=None, lay=LAYOUT):
    seen.clear()
    k = torch.zeros(1, 2, rows or LAYOUT.seq_len, 4)
    topts = {"block_index": block}
    if lay is not None:
        topts["minimax_h3_layout"] = lay
    if carried is not None:
        topts["mmh3_carried_latents"] = carried
    over = model.model_options["transformer_options"]["optimized_attention_override"]
    return over(backend, k, k, k, 2, mask=mask, skip_reshape=True, transformer_options=topts)

def biased(mask):
    return [i for i, x in enumerate(mask[0].tolist()) if x != 0]


print("\n1. the bias lands on the carried video rows")
check("video starts after text and audio", VIDEO, 12 + 20)
m = MMH3CarryAttention.execute(Model(), -1.5).result[0]
check("returns the backend's output", call(m, carried=2), "out")
check("one row, broadcast by the backend", list(seen["mask"].shape), [1, LAYOUT.seq_len])
check("the first 2 latents of the video", biased(seen["mask"]), list(range(VIDEO, VIDEO + 12)))
check("with the bias value", float(seen["mask"][0, VIDEO]), -1.5)

print("\n2. it leaves other calls alone")
call(m, carried=0)
check("a chunk with no carry", seen["mask"], None)
call(m)
check("nothing published (not the looping sampler)", seen["mask"], None)
call(m, carried=2, rows=12)
check("token refiner (text only)", seen["mask"], None)
call(m, carried=2, lay=None)
check("no layout", seen["mask"], None)
base = Model()
check("bias 0 returns the same model", MMH3CarryAttention.execute(base, 0.0).result[0] is base, True)

print("\n3. block selection")
m = MMH3CarryAttention.execute(Model(), -1.0, "10-20").result[0]
call(m, carried=2, block=5)
check("outside the range", seen["mask"], None)
call(m, carried=2, block=15)
check("inside the range", seen["mask"] is not None, True)

print("\n4. stacked with Phrase Attention, either order")
class Clip:
    vocab = {}
    def tokenize(self, text, **kw):
        return {"x": [[(self.vocab.setdefault(w, len(self.vocab)), 1.0) for w in text.split()]]}
cs = {"conds": [[[torch.zeros(1, 12, 4), {}]]], "prompts": ["a b X c"]}
def phrase(model):
    return MMH3PhraseAttention.execute(model, Clip(), cs, "X", 3.0).result[0]
for label, m in (("phrase outside", phrase(MMH3CarryAttention.execute(Model(), -1.0).result[0])),
                 ("carry outside", MMH3CarryAttention.execute(phrase(Model()), -1.0).result[0])):
    call(m, carried=1)
    got = seen["mask"][0].tolist()
    check(label + ": phrase row", got[10], 3.0)
    check(label + ": carried rows", got[VIDEO:VIDEO + 7], [-1.0] * 6 + [0.0])

print("\n5. the looping sampler publishes the carry")
class Guider:
    original_conds = {"positive": "base"}
    model_options = {}
    def set_conds(self, positive):
        self.original_conds["positive"] = positive
c = _chunk_guider(Guider(), "cond", frame0=0, chunk=1, carried=2)
check("carried latents", c.model_options["transformer_options"]["mmh3_carried_latents"], 2)

print("\n6. reference_bias lands on the reference rows")
REFS = [{"kind": "image", "latent_h": 4, "latent_w": 6}, {"kind": "image", "latent_h": 2, "latent_w": 4}]
RLAY = PackedLayout(12, 6, 4, 6, 10, refs=REFS)
REF_ROWS = [i for a, b, kind in RLAY.segments if kind == "ref_img" for i in range(a, b)]
RVIDEO = next(a for a, b, kind in RLAY.segments if kind == "video")
def rcall(model, **kw):
    return call(model, rows=RLAY.seq_len, lay=RLAY, **kw)
m = MMH3CarryAttention.execute(Model(), 0.0, "", 1.5).result[0]
rcall(m)
check("both reference images, with no carry published", biased(seen["mask"]), REF_ROWS)
check("with the reference value", float(seen["mask"][0, REF_ROWS[0]]), 1.5)
call(m, carried=2)
check("no references in the layout", seen["mask"], None)
m = MMH3CarryAttention.execute(Model(), -2.0, "", 1.0).result[0]
rcall(m, carried=1)
got = seen["mask"][0].tolist()
check("both: reference rows", [got[i] for i in REF_ROWS], [1.0] * len(REF_ROWS))
check("both: carried rows", got[RVIDEO:RVIDEO + 7], [-2.0] * 6 + [0.0])
rcall(m, carried=0)
check("both, first chunk: references only", biased(seen["mask"]), REF_ROWS)
m = MMH3CarryAttention.execute(Model(), 0.0, "10-20", 1.0).result[0]
rcall(m, block=5)
check("reference bias follows block selection", seen["mask"], None)

print("\n7. reference_audio_bias lands on the reference audio rows")
ALAY = PackedLayout(12, 6, 4, 6, 10, refs=[
    {"kind": "image", "latent_h": 4, "latent_w": 6},
    {"kind": "audio", "ref_audio_t": 5},
    {"kind": "video_audio", "ref_audio_t": 3, "latent_t": 2, "latent_h": 2, "latent_w": 4}])
def rows_of(kind):
    return [i for a, b, k in ALAY.segments if k == kind for i in range(a, b)]
m = MMH3CarryAttention.execute(Model(), 0.0, "", 0.0, 0.75).result[0]
call(m, rows=ALAY.seq_len, lay=ALAY)
check("the voice reference and the video reference's sound", biased(seen["mask"]), rows_of("ref_audio"))
check("two audio spans", sum(k == "ref_audio" for _a, _b, k in ALAY.segments), 2)
m = MMH3CarryAttention.execute(Model(), 0.0, "", 1.0, -0.5).result[0]
call(m, rows=ALAY.seq_len, lay=ALAY)
got = seen["mask"][0].tolist()
check("picture and sound take their own values",
      ({got[i] for i in rows_of("ref_img")}, {got[i] for i in rows_of("ref_audio")}), ({1.0}, {-0.5}))
check("the target audio is left alone", {got[i] for i in rows_of("audio")}, {0.0})

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
