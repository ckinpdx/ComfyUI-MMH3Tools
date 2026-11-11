"""Place an AUDIO reference on the target's own timeline instead of ahead of it.

WHAT STOCK DOES
---------------
References are laid out from a cursor that starts at `text_len`; each block sits at the
cursor and advances it by its own span, and the target begins where the cursor ends. So
every reference lies BEFORE the clip in time, in the order it was listed. For an identity
image or a voice sample that is right: they are material, not a moment.

WHAT A CARRIED AUDIO TAIL NEEDS
-------------------------------
A carried tail IS a moment: the last seconds of the previous chunk, which the new chunk's
first frames overlap. Left where stock puts it, the model hears "some audio from before"
and produces a sound-alike. Translated so that each carried audio latent sits at the SAME
time coordinate as the target audio row for that instant -- and the rest runs backwards
from the target origin -- the model hears "this is the track up to here" and continues it.

THE CONTRACT
------------
An `audio` reference block carrying the private key `mmh3_end_t` (a float, in audio latent
frames, measured from the TARGET origin) is translated so the block ENDS at
`target_origin + mmh3_end_t`. Row k of an R-latent block therefore lands at

    target_origin + mmh3_end_t - R + k

and with `mmh3_end_t` equal to the number of target audio latents the tail overlaps, the
last `mmh3_end_t` rows share their coordinates with the target's first `mmh3_end_t` rows.

Nothing else moves. The cursor arithmetic is stock's, so the target origin, every other
reference and every guide are where they would have been -- the block still costs its span
in distance from the text, exactly as an unshifted audio reference does.

THREE PROPERTIES, matching the pack's other patches
---------------------------------------------------
  * INERT without the key. A layout whose references carry no `mmh3_end_t` is byte for
    byte what stock built.
  * ABSOLUTE. The block is moved TO a coordinate computed from the finished layout, not
    by an increment, and each __init__ builds its own layout.
  * SELF-TESTED at import against the live class, and rolled back on any mismatch.

A wrap at the callable boundary rather than a source rewrite, for the reasons given in
patch_guide_origin: core's text is neither matched nor embedded.
"""

import logging

KEY = "mmh3_end_t"
_MARK = "_mmh3_ref_shift_patched"
_state = {"done": False, "ok": False, "msg": ""}


def _ref_audio_segments(layout, refs):
    """[(block, start, stop)] for every reference block that owns audio rows.

    Reference segments follow the guides' in the segment table, in block order: an image
    is one `ref_img`; an audio block is one `ref_audio` (none when it is empty); a video
    block is its `ref_audio` (when it has any) and then its `ref_img`. Walking the two
    lists together is the only way to tell WHICH `ref_audio` belongs to which block.
    """
    segs = [(a, b, k) for a, b, k in layout.segments if k in ("ref_img", "ref_audio")]
    out, i = [], 0
    for blk in refs:
        kind = blk.get("kind")
        rt = int(blk.get("ref_audio_t", 0) or 0)
        if kind == "image":
            want = ["ref_img"]
        elif kind == "audio":
            want = ["ref_audio"] if rt > 0 else []
        elif kind in ("video", "video_audio"):
            want = (["ref_audio"] if rt > 0 else []) + ["ref_img"]
        else:
            want = []
        for w in want:
            if i >= len(segs) or segs[i][2] != w:
                raise RuntimeError("reference segments are not in the expected order")
            if w == "ref_audio":
                out.append((blk, segs[i][0], segs[i][1]))
            i += 1
    if i != len(segs):
        raise RuntimeError("reference segments left over after walking the blocks")
    return out


def _shift(layout, refs):
    origin = None
    for a, _b, kind in layout.segments:
        if kind == "audio":
            origin = float(layout.position_ids[a, 0])
    if origin is None:
        return
    for blk, a, b in _ref_audio_segments(layout, refs):
        if blk.get("kind") != "audio" or blk.get(KEY) is None:
            continue
        rt = int(blk["ref_audio_t"])
        want = origin + float(blk[KEY]) - rt
        layout.position_ids[a:b, 0] += want - float(layout.position_ids[a, 0])


def _selftest(cls):
    import torch
    img = {"kind": "image", "latent_h": 4, "latent_w": 4,
           "latent": torch.zeros([1, 24, 1, 4, 4])}
    aud = {"kind": "audio", "ref_audio_t": 6, "audio_latent": torch.zeros([1, 32, 2, 6])}

    def audio_rows(refs):
        lay = cls(8, 7, 4, 4, 12, refs=refs)
        tgt = [(a, b) for a, b, k in lay.segments if k == "audio"][0]
        ref = [(a, b) for a, b, k in lay.segments if k == "ref_audio"][0]
        return (lay.position_ids[ref[0]:ref[1], 0].clone(),
                float(lay.position_ids[tgt[0], 0]), lay.position_ids.clone())

    stock_rows, origin, stock_all = audio_rows([dict(aud), img])
    moved_rows, origin2, _all = audio_rows([dict(aud, **{KEY: 2.0}), img])
    if abs(origin - origin2) > 1e-9:
        return False, "the target origin moved (%.3f -> %.3f)" % (origin, origin2)
    want = origin + 2.0 - 6 + torch.arange(6, dtype=torch.float64).repeat(2)
    if float((moved_rows - want).abs().max()) > 1e-9:
        return False, "a shifted audio reference did not land where asked"
    again_rows, _o, again_all = audio_rows([dict(aud), img])
    if float((again_all - stock_all).abs().max()) > 0.0 or \
            float((again_rows - stock_rows).abs().max()) > 0.0:
        return False, "a layout without the key changed"
    return True, "audio references can be placed on the target timeline"


def apply(verbose=True):
    """Idempotent. Returns (ok, message)."""
    if _state["done"]:
        return _state["ok"], _state["msg"]
    _state["done"] = True

    try:
        import comfy.ldm.minimax.model as mm
    except Exception as e:
        _state["msg"] = "MiniMax H3 not present in this ComfyUI (%s)" % e
        return False, _state["msg"]

    cls = getattr(mm, "PackedLayout", None)
    if cls is None:
        _state["msg"] = "comfy.ldm.minimax.model.PackedLayout not found"
        return False, _state["msg"]
    if getattr(cls, _MARK, False):
        _state["ok"] = True
        _state["msg"] = "already applied"
        return True, _state["msg"]

    original = cls.__init__

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=None, refs=None, *args, **kwargs):
        original(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=keyframes, refs=refs, *args, **kwargs)
        if refs and any(blk.get(KEY) is not None for blk in refs):
            _shift(self, refs)

    cls.__init__ = __init__
    try:
        ok, msg = _selftest(cls)
    except Exception as e:
        ok, msg = False, "self-test raised %s: %s" % (type(e).__name__, e)
    if not ok:
        cls.__init__ = original
        _state["msg"] = msg + " - rolled back, stock behaviour kept"
        logging.warning("[MMH3Tools] reference-shift patch: " + _state["msg"])
        return False, _state["msg"]

    setattr(cls, _MARK, True)
    _state["ok"] = True
    _state["msg"] = msg
    if verbose:
        logging.info("[MMH3Tools] " + _state["msg"])
    return True, _state["msg"]


def is_applied():
    return bool(_state["ok"])
