"""Make H3 attend harder to one phrase of its prompt.

H3 has no cross-attention: prompt tokens sit in the same sequence as the video
and audio rows, and how much a phrase matters is how much attention those rows
pay to its tokens. This adds a constant to the attention scores of the phrase's
positions, in the DiT blocks, for every query row. A bias of b multiplies the
phrase's share of each row's attention by e^b before renormalising -- the H3
equivalent of `(phrase:1.3)`, which the encoder path cannot express.

Changing what the phrase's tokens SAY does not do this. Pushing a phrase's
Qwen output away from a plain baseline's, 31 degrees on its own tokens, made no
visible change (2026-11-10): every block renormalises its input and the scores
come from per-head RMS-normed q and k, so neither a token's size nor a small
turn of it makes the model listen harder. The score is the lever.

The bias rides the attention functions' own `mask` argument as a [1, S] row,
which every backend broadcasts over queries and heads; it costs S values, not
S x S. The phrase is located from the END of the text segment, because the
prompt is the tail of it and reference vision blocks come first.
"""

import logging
import math

import torch

from comfy_api.latest import io


def _text_ids(clip, text):
    return [t[0] for t in next(iter(clip.tokenize(text).values()))[0]]


def _common(a, b):
    """(prefix, suffix) token counts shared by two id lists, never overlapping."""
    limit = min(len(a), len(b))
    p = 0
    while p < limit and a[p] == b[p]:
        p += 1
    s = 0
    while s < limit - p and a[-1 - s] == b[-1 - s]:
        s += 1
    return p, s


def phrase_span(clip, prompt, phrase):
    """(tokens from the phrase's start to the prompt's end, tokens after the phrase)."""
    before, found, after = prompt.partition(phrase)
    if not found:
        raise ValueError("MMH3PhraseAttention: the phrase is not in the prompt. It must "
                         "match exactly, including case and punctuation.")
    ids = _text_ids(clip, prompt)
    prefix, suffix = _common(ids, _text_ids(clip, before + after))
    return len(ids) - prefix, suffix


def _blocks(text):
    want = set()
    for piece in (text or "").replace(",", " ").split():
        a, _, b = piece.partition("-")
        want.update(range(int(a), int(b or a) + 1))
    return want


class MMH3PhraseAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3PhraseAttention",
            display_name="MiniMax H3 Phrase Attention",
            category="MMH3Tools/model",
            description=(
                "Experimental. Bias H3's attention toward one phrase of the prompt. "
                "bias is added to the attention scores of the phrase's tokens in the "
                "DiT blocks: +0.7 doubles their share of attention, negative mutes them."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", tooltip="Only its tokenizer is used, to find the phrase."),
                io.String.Input("prompt", multiline=True, force_input=True,
                                tooltip="The exact text that was encoded. From a cond_set, "
                                        "wire MMH3 Cond Select's `prompt` output."),
                io.String.Input("phrase", multiline=True, default="",
                                tooltip="The words to listen to, exactly as they appear in the "
                                        "prompt. The first occurrence is used."),
                io.Float.Input("bias", default=0.7, min=-10.0, max=10.0, step=0.05,
                               tooltip="Added to the attention scores of the phrase's tokens. "
                                       "Its share of attention scales by e^bias: 0.7 is about "
                                       "x2, 1.6 about x5, 2.3 about x10. 0 returns the model "
                                       "unchanged."),
                io.String.Input("blocks", default="", optional=True,
                                tooltip="DiT blocks to bias, e.g. '0-19' or '10-30, 45'. "
                                        "Empty is all 50."),
            ],
            outputs=[io.Model.Output(), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, clip, prompt, phrase, bias, blocks="") -> io.NodeOutput:
        from_end, after = phrase_span(clip, prompt, phrase)
        n = from_end - after
        if n <= 0:
            raise ValueError("MMH3PhraseAttention: removing the phrase does not change the "
                             "prompt's tokens, so there is nothing to bias.")
        want = _blocks(blocks)
        report = ("phrase is %d token%s, ending %d before the end of the prompt\n"
                  "bias %+.2f: x%.2f its share of attention, in %s"
                  % (n, "" if n == 1 else "s", after, bias, math.exp(bias),
                     "blocks %s" % blocks.strip() if want else "every block"))
        logging.info("[MMH3PhraseAttention] " + report.replace("\n", "; "))
        if bias == 0:
            return io.NodeOutput(model, report)

        m = model.clone()
        opts = m.model_options.setdefault("transformer_options", {})
        previous = opts.get("optimized_attention_override")

        def override(func, q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
            topts = kwargs.get("transformer_options") or {}
            layout = topts.get("minimax_h3_layout")
            rows = k.shape[-2] if skip_reshape else k.shape[1]
            # the token refiner also runs attention, over the text alone
            if (mask is None and layout is not None and rows == layout.seq_len
                    and (not want or topts.get("block_index") in want)):
                end = next(b for _a, b, kind in layout.segments if kind == "text")
                mask = torch.zeros([1, rows], dtype=k.dtype, device=k.device)
                mask[:, end - from_end:end - after] = bias
            if previous is not None:
                return previous(func, q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
            return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)

        opts["optimized_attention_override"] = override
        return io.NodeOutput(m, report)
