"""How much a looping-sampler chunk's new frames read each part of the sequence.

Chain over-bake is put down to the model reading its own output
(docs/looping-sampler.md §10) and Carry Attention acts on that, but nothing
measured it. This does: the share of attention the chunk's new video rows give
the carried rows, the references, the text, the audio and each other.

H3 hands the attention function q and k already RMS-normed and rotated, so the
softmax of a query row can be computed here exactly. A sample of the new rows
is enough for a share, and it is taken on the call that reaches the backend,
with whatever bias Carry Attention or Phrase Attention added to the mask.

The Looping Sampler reads the result after each chunk and writes it into its
report. Under any other sampler nothing reads it.
"""

import torch

from comfy_api.latest import io

from .nodes_phrase_attention import _blocks

SAMPLES = 64


class AttentionShare:
    """One chunk's running totals. An object, not a dict: cloning a model copies
    the dicts in its options, and the sampler has to read the one the patch writes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0
        self.share = {}
        self.rows = {}

    def measure(self, q, k, mask, layout, carried, scale):
        start, end = next((a, b) for a, b, kind in layout.segments if kind == "video")
        new = start + carried * ((end - start) // layout.signature[1])
        if new >= end:
            return
        spans = [(a, b, kind) for a, b, kind in layout.segments if kind != "video"]
        if new > start:
            spans.append((start, new, "carry"))
        spans.append((new, end, "video"))

        # an odd stride, so the sample does not sit on one column of an even-width grid
        rows = slice(new, end, max(1, (end - new) // SAMPLES) | 1)
        scores = torch.matmul(q[:, :, rows], k.transpose(-1, -2)).float() * scale
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[-2] != 1:
                mask = mask[..., rows, :]
            scores = scores + mask
        weights = scores.softmax(dim=-1)

        self.calls += 1
        self.rows = {}
        for a, b, kind in spans:
            self.share[kind] = self.share.get(kind, 0.0) + float(weights[..., a:b].sum(dim=-1).mean())
            self.rows[kind] = self.rows.get(kind, 0) + b - a

    def line(self):
        if not self.calls:
            return "attention: not measured (no dense attention call reached the probe)"
        total = sum(self.rows.values())
        return "attention from the new frames: " + ", ".join(
            "%s %.1f%% (x%.1f)" % (kind, 100 * self.share[kind] / self.calls,
                                   self.share[kind] / self.calls * total / n)
            for kind, n in self.rows.items())


class MMH3AttentionShareProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3AttentionShareProbe",
            display_name="MiniMax H3 Attention Share Probe",
            category="MMH3Tools/model",
            description=(
                "Experimental. Measures how much of their attention a Looping Sampler "
                "chunk's new frames give the carried frames, the references, the text "
                "and the audio, and writes it into the sampler's report, per chunk. "
                "It changes nothing in the picture. Wire it anywhere in the model "
                "chain; it sees the biases of Carry Attention and Phrase Attention "
                "either side of it."
            ),
            inputs=[
                io.Model.Input("model"),
                io.String.Input("blocks", default="", optional=True,
                                tooltip="DiT blocks to measure, e.g. '0-19' or '10-30, 45'. "
                                        "Empty is all 50, averaged."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, blocks="") -> io.NodeOutput:
        want = _blocks(blocks)
        share = AttentionShare()
        m = model.clone()
        opts = m.model_options.setdefault("transformer_options", {})
        previous = opts.get("optimized_attention_override")
        opts["mmh3_attention_share"] = share

        def override(func, q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
            # measured where the backend is called, so the mask holds every node's bias
            def measured(q, k, v, heads, mask=None, skip_reshape=False, **kw):
                topts = kw.get("transformer_options") or {}
                layout = topts.get("minimax_h3_layout")
                # the negative pass is skipped; sparse and window attention pass slices
                if (skip_reshape and layout is not None
                        and q.shape[-2] == k.shape[-2] == layout.seq_len
                        and (not want or topts.get("block_index") in want)
                        and topts.get("cond_or_uncond") != [1]):
                    share.measure(q, k, mask, layout, topts.get("mmh3_carried_latents", 0),
                                  kw.get("scale", q.shape[-1] ** -0.5))
                return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kw)

            if previous is not None:
                return previous(measured, q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
            return measured(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)

        opts["optimized_attention_override"] = override
        return io.NodeOutput(m)
