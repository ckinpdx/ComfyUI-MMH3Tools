import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import torch
from comfy.ldm.minimax.model import FRAME_RESCALE, PackedLayout
from mmh3tools.nodes_refs import MMH3ImageKeyframe, MMH3LatentKeyframe

fails = []
def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else "  got=%s want=%s" % (got, want)))
    if not ok:
        fails.append(label)


class Vae:
    def encode(self, px):
        return torch.zeros(1, 24, 1, px.shape[1] // 16, px.shape[2] // 16)

COND = [[torch.zeros(1, 8, 4), {"minimax_refs": ["ref"]}]]
IMAGE = torch.zeros(1, 100, 200, 3)

def image_kf(frame_index, cond=COND):
    out = MMH3ImageKeyframe.execute(cond, IMAGE, Vae(), 64, 64, 124, frame_index, "auto").result[0]
    return out[0][1]

def raises(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return ""


print("\n1. Image Keyframe takes any frame of the chunk")
for label, given, want in (("first", 0, 0), ("interior", 60, 60), ("last", 123, 123),
                           ("-1 is the last", -1, 123), ("negatives count from the end", -24, 100)):
    check(label, image_kf(given)["minimax_keyframes"][0]["resolved_frame_index"], want)
check("past the end is an error", "outside the chunk's 124 frames" in raises(lambda: image_kf(124)), True)
check("before the start is an error", "outside the chunk's 124 frames" in raises(lambda: image_kf(-125)), True)
d = image_kf(60)
check("encoded at the target grid", list(d["minimax_keyframes"][0]["latent"].shape), [1, 24, 1, 4, 4])
check("frame count recorded", d["minimax_frame_count"], 124)
check("references are kept", d["minimax_refs"], ["ref"])
check("appends to existing keyframes", len(image_kf(-1, [[COND[0][0], d]])["minimax_keyframes"]), 2)
check("the input conditioning is not written to", "minimax_keyframes" in COND[0][1], False)

print("\n2. core places an interior anchor from it")
lay = PackedLayout(8, 37, 4, 4, 8, keyframes=d["minimax_keyframes"])
seg = {k: a for a, _b, k in lay.segments}
check("cond row sits 60 frames into the target",
      round(float(lay.position_ids[seg["cond"], 0] - lay.position_ids[seg["video"], 0]), 6),
      round(FRAME_RESCALE * 60, 6))

print("\n3. Latent Keyframe")
latent = {"samples": torch.arange(7.0).reshape(1, 1, 7, 1, 1).repeat(1, 24, 1, 4, 4)}
kf = MMH3LatentKeyframe.execute(COND, latent, 124, "last", "last").result[0][0][1]["minimax_keyframes"][0]
check("last position", kf["resolved_frame_index"], 123)
check("takes the source's last latent frame", float(kf["latent"][0, 0, 0, 0, 0]), 6.0)
kf = MMH3LatentKeyframe.execute(COND, latent, 124, "first", "first").result[0][0][1]["minimax_keyframes"][0]
check("first position, first source frame", (kf["resolved_frame_index"], float(kf["latent"][0, 0, 0, 0, 0])), (0, 0.0))
check("a size mismatch is named",
      "cannot be downscaled" in raises(lambda: MMH3LatentKeyframe.execute(COND, latent, 124, "first", "last", 128, 64)), True)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
