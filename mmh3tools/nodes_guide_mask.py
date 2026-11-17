"""Hide a keyframe from the frames next to it.

LongCat-Video-Avatar holds a long static scene with one still placed mid-chunk
and a `mask_frame_range`: the video latents within that many of the still cannot
see it at all, so they do not copy it, while the rest of the chunk still reads
it. This is that mask for H3's guide picture rows.

It is a hard mask with no mask passed: the query rows are split into runs and
each run is one plain attention call over the keys it may see, so the backend
never takes its masked path.
"""

import torch

from comfy_api.latest import io

from .nodes_phrase_attention import _blocks


def masked_runs(layout, reach):
    """[(row_a, row_b, [(key_a, key_b)])] over the whole sequence: the query rows
    a..b cannot see those key rows. Video latents within `reach` of a guide
    picture lose that guide; every other row sees everything."""
    va, vb = next((a, b) for a, b, kind in layout.segments if kind == "video")
    latents = layout.signature[1]
    per_latent = (vb - va) // latents
    times = layout.position_ids[va:vb:per_latent, 0].tolist()
    hidden = [[] for _ in range(latents)]
    for a, b, kind in layout.segments:
        if kind != "cond":
            continue
        # the latent holding each end of the guide: the last one starting at or before it
        first, last = (max(sum(start <= t + 1e-6 for start in times) - 1, 0)
                       for t in layout.position_ids[[a, b - 1], 0].tolist())
        for j in range(max(first - reach, 0), min(last + reach, latents - 1) + 1):
            hidden[j].append((a, b))
    runs = [(0, va, [])]
    for j, keys in enumerate(hidden):
        end = va + (j + 1) * per_latent
        if keys == runs[-1][2]:
            runs[-1] = (runs[-1][0], end, keys)
        else:
            runs.append((end - per_latent, end, keys))
    return runs


class MMH3GuideMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3GuideMask",
            display_name="MiniMax H3 Guide Mask",
            category="MMH3Tools/model",
            description=(
                "Experimental. LongCat's mask_frame_range: the frames around a keyframe "
                "cannot see it, so they do not copy it, while the rest of the chunk still "
                "reads it. For one still repeated through a static scene, with the "
                "Keyframe Planner's 'chunk middle'."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("mask_range", default=2, min=0, max=64,
                             tooltip="The video latents within this many of a guide picture "
                                     "cannot see it. 2 hides a still from 5 latents, about "
                                     "20 frames. Meant for stills; a guide_video spanning "
                                     "the chunk is hidden from all of it. 0 is off."),
                io.String.Input("blocks", default="", optional=True,
                                tooltip="DiT blocks to mask in, e.g. '0-19' or '10-30, 45'. "
                                        "Empty is all 50."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, mask_range, blocks="") -> io.NodeOutput:
        if not mask_range:
            return io.NodeOutput(model)
        want = _blocks(blocks)
        m = model.clone()
        opts = m.model_options.setdefault("transformer_options", {})
        previous = opts.get("optimized_attention_override")

        def override(func, q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
            topts = kwargs.get("transformer_options") or {}
            layout = topts.get("minimax_h3_layout")
            rows = k.shape[-2] if skip_reshape else k.shape[1]
            # the token refiner also runs attention, over the text alone
            if (layout is not None and rows == layout.seq_len
                    and (not want or topts.get("block_index") in want)):
                runs = masked_runs(layout, mask_range)
                if any(keys for _a, _b, keys in runs):
                    backend = func

                    # innermost, so the overrides chained below still see the whole sequence
                    def func(q, k, v, heads, mask=None, **kw):
                        if q.shape[-2] != layout.seq_len or k.shape[-2] != layout.seq_len:
                            return backend(q, k, v, heads, mask=mask, **kw)
                        out = []
                        for a, b, keys in runs:
                            kk, vv, mm = k, v, mask
                            if keys:
                                edges = [0] + [x for span in keys for x in span] + [layout.seq_len]
                                keep = list(zip(edges[::2], edges[1::2]))
                                kk = torch.cat([k[..., s:e, :] for s, e in keep], dim=-2)
                                vv = torch.cat([v[..., s:e, :] for s, e in keep], dim=-2)
                                if mask is not None:
                                    mm = torch.cat([mask[..., s:e] for s, e in keep], dim=-1)
                            if mm is not None and mm.shape[-2] != 1:
                                mm = mm[..., a:b, :]
                            out.append(backend(q[..., a:b, :], kk, vv, heads, mask=mm, **kw))
                        return torch.cat(out, dim=-2)
            if previous is not None:
                return previous(func, q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
            return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)

        opts["optimized_attention_override"] = override
        return io.NodeOutput(m)
