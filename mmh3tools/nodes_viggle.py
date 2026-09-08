"""Interop with ComfyUI-Viggle-Animate-H3 (Saganaki22).

Viggle's windowed conditioning node and this pack's looping sampler independently
landed on the same design: a cond_set holding one conditioning per chunk, with the
guider's positive overwritten per chunk. Its reference blocks are the same schema as
ours too -- both conform to core's H3 `minimax_refs` contract -- so the conditioning
needs no translation at all. What blocks the wire is only the socket TYPE NAME:
Viggle returns "VIGGLE_COND_SET", the sampler wants "MMH3_COND_SET".

The spans are the real problem. Viggle bakes a per-window slice of the driving clip
into each cond, so cond i is correct ONLY for Viggle's span i. The sampler plans its
own spans from chunk_frames/overlap_frames and reads conds[min(i, len(conds) - 1)],
so if the two schedules differ, every chunk is conditioned on the wrong stretch of
the clip -- and a matching chunk COUNT hides it completely, because the existing
count check in the sampler is the only thing guarding that index.

So this node does not just retype the dict. It derives the chunk_frames /
overlap_frames that reproduce Viggle's schedule, replans them through the sampler's
own _plan, and reports whether the spans come back identical. Wire the two INT
outputs into the sampler and alignment is structural rather than remembered.
"""

import logging

from comfy_api.latest import io

from .nodes_multiprompt import MMH3CondSet
from .nodes_windows import _plan, _window_frame_spans

ViggleCondSet = io.Custom("VIGGLE_COND_SET")


def _frame_spans(spans):
    """Viggle spans are (first_frame, last_frame, latent0, latent_count) -> (a, b).

    Inclusive on both ends, matching _window_frame_spans, so the two lists compare
    directly. Only the first two entries are read: the latent pair is Viggle's own
    bookkeeping and carries no information we need.
    """
    return [(int(s[0]), int(s[1])) for s in spans]


def _derive(spans):
    """Viggle's frame spans -> the (chunk_frames, overlap_frames) that rebuild them.

    chunk_frames is the WINDOW length, not the stride: _plan feeds it straight to
    context_length. overlap is how far the second window reaches back into the
    first. Both are snapped to the latent grid inside _plan, which is exactly why
    the caller replans instead of trusting this arithmetic.
    """
    chunk_frames = spans[0][1] - spans[0][0] + 1
    overlap_frames = spans[0][1] - spans[1][0] + 1 if len(spans) > 1 else 0
    return int(chunk_frames), int(max(0, overlap_frames))


def _compare(viggle, mmh3):
    """Per-index span diff. Returns (aligned, report lines)."""
    if viggle == mmh3:
        return True, ["spans MATCH: %d chunks, identical boundaries" % len(viggle)]
    lines = ["! spans DIFFER -- %d Viggle chunk(s) vs %d planned here"
             % (len(viggle), len(mmh3))]
    for i in range(max(len(viggle), len(mmh3))):
        v = viggle[i] if i < len(viggle) else None
        m = mmh3[i] if i < len(mmh3) else None
        if v != m:
            lines.append("  chunk %d: viggle %s, here %s"
                         % (i, "-" if v is None else "%d-%d" % v,
                            "-" if m is None else "%d-%d" % m))
    lines.append("  Each cond carries the driving-clip window for ITS span, so a "
                 "mismatch conditions chunks on the wrong frames.")
    return False, lines


class MMH3CondSetFromViggle(io.ComfyNode):
    """Retype a Viggle windowed cond_set for the looping sampler, and check its spans."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3CondSetFromViggle",
            display_name="MMH3 Cond Set From Viggle",
            category="MMH3Tools/conditioning",
            description=(
                "Adapt the cond_set from Viggle-Animate Conditioning (H3, Windowed) "
                "for the MMH3 Looping Sampler. The conditioning itself is already "
                "compatible -- both packs emit core's minimax_refs blocks -- so this "
                "only retypes the socket and verifies the window schedules agree. "
                "Wire chunk_frames and overlap_frames into the sampler so they do."
            ),
            inputs=[
                ViggleCondSet.Input(
                    "viggle_cond_set",
                    tooltip="The cond_set output of Viggle-Animate Conditioning "
                            "(H3, Windowed). Its guider_positive output is not "
                            "needed here -- the sampler replaces the positive per "
                            "chunk from this set.",
                ),
            ],
            outputs=[
                MMH3CondSet.Output(display_name="cond_set"),
                io.Int.Output(display_name="chunk_frames (frames)"),
                io.Int.Output(display_name="overlap_frames (frames)"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, viggle_cond_set) -> io.NodeOutput:
        src = viggle_cond_set or {}
        conds = src.get("conds") or []
        if not conds:
            raise ValueError(
                "MMH3CondSetFromViggle: the Viggle cond_set holds no conditioning.")
        prompts = src.get("prompts") or [""] * len(conds)

        spans = _frame_spans(src.get("spans") or [])
        total_f = int(src.get("total_frames") or 0)

        lines = ["%d chunk(s) from Viggle over %d frames" % (len(conds), total_f)]
        chunk_frames, overlap_frames = 0, 0
        if not spans or not total_f:
            # Nothing to verify against; say so rather than emitting settings that
            # were never checked.
            lines.append("! Viggle cond_set carried no spans/total_frames -- cannot "
                         "derive or verify a schedule. Set chunk_frames and "
                         "overlap_frames on the sampler by hand.")
            logging.warning("[MMH3CondSetFromViggle] %s", lines[-1][2:])
        else:
            chunk_frames, overlap_frames = _derive(spans)
            _l, _ov, planned_f, _t, windows = _plan(
                total_f, chunk_frames, overlap_frames, "standard_static")
            planned = _window_frame_spans(windows, planned_f)
            lines.append("derived chunk_frames %d, overlap_frames %d"
                         % (chunk_frames, overlap_frames))
            aligned, diff = _compare(spans, planned)
            lines.extend(diff)
            if not aligned:
                logging.warning("[MMH3CondSetFromViggle] spans differ; see the report")

        report = "\n".join(lines)
        logging.info("[MMH3CondSetFromViggle] %s", lines[0] if len(lines) == 1
                     else "; ".join(lines[:2]))
        return io.NodeOutput({"conds": conds, "prompts": prompts, "fingerprint": None},
                             chunk_frames, overlap_frames, report)
