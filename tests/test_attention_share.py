import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.ldm.minimax.model import PackedLayout
from comfy.ldm.modules.attention import attention_pytorch
from mmh3tools.nodes_attention_share import MMH3AttentionShareProbe
from mmh3tools.nodes_carry_attention import MMH3CarryAttention

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)

def close(label, got, want, tol=1e-5):
    check(label + " (%.4f)" % want, abs(got - want) < tol, True)


class Model:
    def __init__(self, options=None):
        self.model_options = options or {}
    def clone(self):
        return Model({k: dict(v) for k, v in self.model_options.items()})

HEADS, DIM = 3, 8
# 12 text rows, a 4x6 reference image (6 rows), 10 audio latents, 6 video latents of 4x6
LAYOUT = PackedLayout(12, 6, 4, 6, 10, refs=[{"kind": "image", "latent_h": 4, "latent_w": 6}])
S = LAYOUT.seq_len
VIDEO, END = next((a, b) for a, b, kind in LAYOUT.segments if kind == "video")
torch.manual_seed(0)
Q, K, V = (torch.randn(1, HEADS, S, DIM) for _ in range(3))

seen = {}
def backend(q, k, v, heads, mask=None, skip_reshape=False, **kw):
    seen["calls"] = seen.get("calls", 0) + 1
    return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape)

def probe_of(model):
    return model.model_options["transformer_options"]["mmh3_attention_share"]

def call(model, carried=2, q=Q, k=K, block=0, lay=LAYOUT, **topts):
    topts.update(block_index=block, mmh3_carried_latents=carried)
    if lay is not None:
        topts["minimax_h3_layout"] = lay
    over = model.model_options["transformer_options"]["optimized_attention_override"]
    return over(backend, q, k, V, HEADS, skip_reshape=True, transformer_options=topts)

def truth(carried, bias=None):
    """Every new video row's full softmax, summed per part of the sequence."""
    scores = Q @ K.transpose(-1, -2) * DIM ** -0.5
    if bias is not None:
        scores = scores + bias
    new = VIDEO + carried * 6
    w = scores.softmax(dim=-1)[:, :, new:END]
    out = {kind: float(w[..., a:b].sum(dim=-1).mean()) for a, b, kind in LAYOUT.segments if kind != "video"}
    if carried:
        out["carry"] = float(w[..., VIDEO:new].sum(dim=-1).mean())
    out["video"] = float(w[..., new:END].sum(dim=-1).mean())
    return out

def shares(model):
    p = probe_of(model)
    return {kind: total / p.calls for kind, total in p.share.items()}


print("\n1. the shares are the softmax of the new rows")
m = MMH3AttentionShareProbe.execute(Model()).result[0]
out = call(m)
check("the backend's output is returned unchanged",
      bool(torch.equal(out, attention_pytorch(Q, K, V, HEADS, skip_reshape=True))), True)
want = truth(2)
got = shares(m)
check("text, reference, audio, carry and the new frames", list(got), ["text", "ref_img", "audio", "carry", "video"])
for kind in want:
    close(kind, got[kind], want[kind])
close("they sum to 1", sum(got.values()), 1.0)
check("the rows behind each share", probe_of(m).rows, {"text": 12, "ref_img": 6, "audio": 20, "carry": 12, "video": 24})

print("\n2. a chunk with no carry")
probe_of(m).reset()
call(m, carried=0)
check("no carry share", list(shares(m)), ["text", "ref_img", "audio", "video"])
close("every video row is a query", shares(m)["video"], truth(0)["video"])
probe_of(m).reset()
call(m, carried=6)
check("a chunk that is all carry measures nothing", probe_of(m).calls, 0)

print("\n3. it sees Carry Attention's bias, either side of it")
row = torch.zeros(1, S)
row[:, VIDEO:VIDEO + 12] = -2.0
row[:, 12:18] = 1.5
want = truth(2, row)
check("the bias moves the truth", want["carry"] < truth(2)["carry"] and want["ref_img"] > truth(2)["ref_img"], True)
for label, m in (("probe outside", MMH3AttentionShareProbe.execute(
                      MMH3CarryAttention.execute(Model(), -2.0, 1.5).result[0]).result[0]),
                 ("probe inside", MMH3CarryAttention.execute(
                      MMH3AttentionShareProbe.execute(Model()).result[0], -2.0, 1.5).result[0])):
    out = call(m)
    check(label + ": the biased output",
          bool(torch.equal(out, attention_pytorch(Q, K, V, HEADS, mask=row, skip_reshape=True))), True)
    close(label + ": carry", shares(m)["carry"], want["carry"])
    close(label + ": reference", shares(m)["ref_img"], want["ref_img"])

print("\n4. the calls it leaves alone")
m = MMH3AttentionShareProbe.execute(Model(), "10-20").result[0]
def measured(**kw):
    seen.clear()
    before = probe_of(m).calls
    call(m, **kw)
    return probe_of(m).calls - before, seen["calls"]
check("a block outside the range still runs", measured(block=5), (0, 1))
check("a block inside it", measured(block=15), (1, 1))
check("the negative pass", measured(block=15, cond_or_uncond=[1]), (0, 1))
check("the positive pass", measured(block=15, cond_or_uncond=[0]), (1, 1))
check("no layout", measured(block=15, lay=None), (0, 1))
check("sliced keys (sparse or window attention)", measured(block=15, k=K[:, :, :40]), (0, 1))
check("sliced queries", measured(block=15, q=Q[:, :, :40]), (0, 1))

print("\n5. averaged over calls, and the report line")
m = MMH3AttentionShareProbe.execute(Model()).result[0]
check("before any call", probe_of(m).line(), "attention: not measured (no dense attention call reached the probe)")
call(m)
call(m)
check("two calls", probe_of(m).calls, 2)
close("the mean of two identical calls", shares(m)["carry"], truth(2)["carry"])
t = truth(2)
check("share and its multiple of uniform attention", probe_of(m).line(),
      "attention from the new frames: " + ", ".join(
          "%s %.1f%% (x%.1f)" % (kind, 100 * t[kind], t[kind] * S / n)
          for kind, n in (("text", 12), ("ref_img", 6), ("audio", 20), ("carry", 12), ("video", 24))))

print("\n6. the probe survives a real clone of the model options")
import comfy.utils
opts = comfy.utils.deepcopy_list_dict(m.model_options)
check("the same object, not a copy", opts["transformer_options"]["mmh3_attention_share"] is probe_of(m), True)

print("\n7. a sample of the rows on a long chunk")
BIG = PackedLayout(12, 40, 4, 6, 10)
bq, bk = torch.randn(1, HEADS, BIG.seq_len, DIM), torch.randn(1, HEADS, BIG.seq_len, DIM)
m = MMH3AttentionShareProbe.execute(Model()).result[0]
over = m.model_options["transformer_options"]["optimized_attention_override"]
over(backend, bq, bk, bk, HEADS, skip_reshape=True, transformer_options={
    "block_index": 0, "minimax_h3_layout": BIG, "mmh3_carried_latents": 8})
close("still sums to 1", sum(shares(m).values()), 1.0)
bv = next(a for a, b, kind in BIG.segments if kind == "video")
check("192 new rows, every third one read", len(range(bv + 48, BIG.seq_len, max(1, 192 // 64) | 1)), 64)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
