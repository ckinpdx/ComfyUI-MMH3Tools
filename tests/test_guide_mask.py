import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.ldm.minimax.model import PackedLayout
from mmh3tools.nodes_carry_attention import MMH3CarryAttention
from mmh3tools.nodes_guide_mask import MMH3GuideMask, masked_runs
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

class Clip:
    vocab = {}
    def tokenize(self, text, **kw):
        return {"x": [[(self.vocab.setdefault(w, len(self.vocab)), 1.0) for w in text.split()]]}
cs = {"conds": [[[torch.zeros(1, 12, 4), {}]]], "prompts": ["a b X c"]}
def phrase(model):
    return MMH3PhraseAttention.execute(model, Clip(), cs, "X", 3.0).result[0]

print("1. the runs: which rows lose which guide")
# 12 text rows, 6 video latents of 4x6 (6 rows each), 10 audio latents;
# a still at pixel frame 12 = latent 3 of 6, and a second at frame 0
KLAY = PackedLayout(12, 6, 4, 6, 10, keyframes=[
    {"resolved_frame_index": 12, "latent": torch.zeros([1, 24, 1, 4, 6])},
    {"resolved_frame_index": 0, "latent": torch.zeros([1, 24, 1, 4, 6])}])
PLAIN = PackedLayout(12, 6, 4, 6, 10)
KF = [(a, b) for a, b, kind in KLAY.segments if kind == "cond"]
KVIDEO = next(a for a, b, kind in KLAY.segments if kind == "video")
runs = masked_runs(KLAY, 1)
check("runs cover the sequence in order",
      [runs[0][0]] + [b for _a, b, _k in runs] == [0] + [a for a, _b, _k in runs[1:]] + [KLAY.seq_len], True)
check("latents 0-1 lose the opener, 2-4 the middle still, 5 nothing",
      [(a - KVIDEO, b - KVIDEO, keys) for a, b, keys in runs[1:]],
      [(0, 12, [KF[1]]), (12, 30, [KF[0]]), (30, 36, [])])
check("rows before the video see everything", runs[0], (0, KVIDEO, []))

print("\n2. one plain attention call per run, equal to a hard mask")
calls = []
def plain(q, k, v, heads, mask=None, skip_reshape=False, **kw):
    calls.append((q.shape[-2], k.shape[-2], mask))
    score = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        score = score + mask
    return score.softmax(-1) @ v
def kcall(model, lay=KLAY, **topts):
    calls.clear()
    over = model.model_options["transformer_options"]["optimized_attention_override"]
    q, k, v = (x[..., :lay.seq_len, :] for x in (Q, K, V))
    return over(plain, q, k, v, 2, skip_reshape=True, transformer_options={
        "block_index": 0, "minimax_h3_layout": lay, **topts})
torch.manual_seed(0)
Q, K, V = (torch.randn(1, 2, KLAY.seq_len, 8, dtype=torch.float64) for _ in range(3))
full = torch.zeros(KLAY.seq_len, KLAY.seq_len, dtype=torch.float64)
for a, b, keys in runs:
    for ka, kb in keys:
        full[a:b, ka:kb] = float("-inf")
m = MMH3GuideMask.execute(Model(), 1).result[0]
got = kcall(m)
check("one plain call per run, none with a mask", calls,
      [(b - a, KLAY.seq_len - sum(kb - ka for ka, kb in keys), None) for a, b, keys in runs])
check("matches one masked attention call", torch.allclose(got, plain(Q, K, V, 2, mask=full)), True)

print("\n3. chained with the bias nodes, either order")
soft = torch.zeros(1, KLAY.seq_len, dtype=torch.float64)
soft[:, KVIDEO:KVIDEO + 6] = -1.0
for a, b in KF:
    soft[:, a:b] = 0.5
def carry(model):
    return MMH3CarryAttention.execute(model, -1.0, guide_bias=0.5).result[0]
for label, m in (("carry outside", carry(MMH3GuideMask.execute(Model(), 1).result[0])),
                 ("mask outside", MMH3GuideMask.execute(carry(Model()), 1).result[0])):
    check(label + ": the biases are cut to each run's keys",
          torch.allclose(kcall(m, mmh3_carried_latents=1), plain(Q, K, V, 2, mask=full + soft)), True)
soft = torch.zeros(1, KLAY.seq_len, dtype=torch.float64)
soft[:, 10] = 3.0
for label, m in (("phrase outside", phrase(MMH3GuideMask.execute(Model(), 1).result[0])),
                 ("mask outside", MMH3GuideMask.execute(phrase(Model()), 1).result[0])):
    check(label + ": the phrase bias reaches every run",
          torch.allclose(kcall(m), plain(Q, K, V, 2, mask=full + soft)), True)

print("\n4. the calls it leaves alone")
m = MMH3GuideMask.execute(Model(), 1).result[0]
kcall(m, lay=PLAIN)
check("no guide in the layout: one whole call", calls, [(PLAIN.seq_len, PLAIN.seq_len, None)])
base = Model()
check("range 0 returns the same model", MMH3GuideMask.execute(base, 0).result[0] is base, True)
m = MMH3GuideMask.execute(Model(), 1, "10-20").result[0]
kcall(m)
check("outside the block range: one whole call", len(calls), 1)
kcall(m, block_index=15)
check("inside the block range", len(calls), len(runs))

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
