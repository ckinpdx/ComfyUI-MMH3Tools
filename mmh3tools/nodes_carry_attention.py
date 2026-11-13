"""Make a looping-sampler chunk read its carried frames more or less.

Chain over-bake follows the model reading its own output (docs/looping-sampler.md
§10). The carry settings change WHAT the next chunk is shown; this changes how
hard it reads it, with the carry left clean. A bias is added to the attention
scores of the carried video rows, as MMH3PhraseAttention does for a phrase.

carry="mask" only: there the carry is the first latents of the target video, and
the sampler publishes how many as `mmh3_carried_latents`. Without that key the
carry bias does nothing.

`reference_bias` is the same bias on the reference image and video rows, and
`reference_audio_bias` on the reference audio rows. The layout labels those
itself, so they need no carry and work under any sampler.

`audio_bias` is on the clip's own audio rows, for a track pinned with
`use_input_audio`: positive makes the picture read the sound more. It has its own
`audio_blocks`, since which blocks to bias is the thing to find out.
"""

import torch

from comfy_api.latest import io

from .nodes_phrase_attention import _blocks


class MMH3CarryAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3CarryAttention",
            display_name="MiniMax H3 Carry Attention",
            category="MMH3Tools/model",
            description=(
                "Experimental. Bias H3's attention toward the frames a Looping Sampler "
                "chunk carries from the previous one (carry='mask'). Negative makes the "
                "chunk copy its carry less, positive more. reference_bias and "
                "reference_audio_bias do the same for the references, and audio_bias "
                "for the clip's own audio."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("carry_bias", default=-1.0, min=-10.0, max=10.0, step=0.05,
                               tooltip="Added to the attention scores of the carried video "
                                       "rows. Their share of attention scales by e^carry_bias: "
                                       "-1 is about a third, -2 about a seventh. 0 is off."),
                io.Float.Input("reference_bias", default=0.0, min=-10.0, max=10.0, step=0.05,
                               optional=True,
                               tooltip="Added to the attention scores of the reference image "
                                       "and video rows. Positive makes every chunk read its "
                                       "references more: +1 is about 2.7x the attention. "
                                       "0 is off."),
                io.Float.Input("reference_audio_bias", default=0.0, min=-10.0, max=10.0,
                               step=0.05, optional=True,
                               tooltip="The same for the reference audio rows: a voice "
                                       "reference, and the sound of a video reference. "
                                       "0 is off."),
                io.String.Input("blocks", default="", optional=True,
                                tooltip="DiT blocks for the three biases above, e.g. '0-19' "
                                        "or '10-30, 45'. Empty is all 50."),
                io.Float.Input("audio_bias", default=0.0, min=-10.0, max=10.0, step=0.05,
                               optional=True,
                               tooltip="Added to the attention scores of the clip's own "
                                       "audio rows. For a pinned input track: positive makes "
                                       "the picture follow the sound more. With generated "
                                       "audio it changes the sound too. 0 is off."),
                io.String.Input("audio_blocks", default="", optional=True,
                                tooltip="DiT blocks for audio_bias alone, e.g. '0-16'. "
                                        "Empty is all 50. `blocks` does not apply to it."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, carry_bias, reference_bias=0.0, reference_audio_bias=0.0,
                blocks="", audio_bias=0.0, audio_blocks="") -> io.NodeOutput:
        if carry_bias == 0 and reference_bias == 0 and reference_audio_bias == 0 and audio_bias == 0:
            return io.NodeOutput(model)
        want = _blocks(blocks)
        audio_want = _blocks(audio_blocks)
        m = model.clone()
        opts = m.model_options.setdefault("transformer_options", {})
        previous = opts.get("optimized_attention_override")

        def override(func, q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
            topts = kwargs.get("transformer_options") or {}
            layout = topts.get("minimax_h3_layout")
            rows = k.shape[-2] if skip_reshape else k.shape[1]
            # the token refiner also runs attention, over the text alone
            if layout is not None and rows == layout.seq_len:
                block = topts.get("block_index")
                spans = []
                if not want or block in want:
                    carried = topts.get("mmh3_carried_latents", 0)
                    if carry_bias and carried:
                        start, end = next((a, b) for a, b, kind in layout.segments if kind == "video")
                        per_latent = (end - start) // layout.signature[1]
                        spans.append((start, start + carried * per_latent, carry_bias))
                    for ref_kind, value in (("ref_img", reference_bias),
                                            ("ref_audio", reference_audio_bias)):
                        if value:
                            spans += [(a, b, value) for a, b, kind in layout.segments
                                      if kind == ref_kind]
                if audio_bias and (not audio_want or block in audio_want):
                    spans += [(a, b, audio_bias) for a, b, kind in layout.segments
                              if kind == "audio"]
                if spans:
                    row = torch.zeros([1, rows], dtype=k.dtype, device=k.device)
                    for a, b, value in spans:
                        row[:, a:b] = value
                    mask = row if mask is None else mask + row
            if previous is not None:
                return previous(func, q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
            return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)

        opts["optimized_attention_override"] = override
        return io.NodeOutput(m)
