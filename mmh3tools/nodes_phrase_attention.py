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

`value_scale` is the other half: it multiplies the phrase's VALUE rows, what they
contribute once attended to. Below 1 removes the phrase's content rather than the
attention paid to it, and below 0 subtracts it.

PER CHUNK. The model sees a packed sequence, not a chunk, so the looping
sampler publishes `mmh3_chunk_index` and the override reads it: each chunk gets
its own prompt's phrase position and its own bias from the schedule. Outside the
looping sampler nothing is published and the run is treated as chunk 0.
"""

import logging
import math

import torch

from comfy_api.latest import io

from .nodes_multiprompt import MMH3CondSet


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
    """(tokens from the phrase's start to the prompt's end, tokens after the phrase).

    None when the prompt does not contain the phrase.
    """
    before, found, after = prompt.partition(phrase)
    if not found:
        return None
    ids = _text_ids(clip, prompt)
    prefix, suffix = _common(ids, _text_ids(clip, before + after))
    if len(ids) - prefix - suffix <= 0:
        return None
    return len(ids) - prefix, suffix


def _blocks(text):
    want = set()
    for piece in (text or "").replace(",", " ").split():
        a, _, b = piece.partition("-")
        want.update(range(int(a), int(b or a) + 1))
    return want


def _runs(chunks):
    """[0, 1, 2, 5] -> '0-2, 5'"""
    out, start = [], None
    for i, c in enumerate(chunks):
        if start is None:
            start = c
        if i + 1 == len(chunks) or chunks[i + 1] != c + 1:
            out.append("%d" % c if c == start else "%d-%d" % (start, c))
            start = None
    return ", ".join(out)


def parse_schedule(text):
    """'chunk: bias' per line -> [(chunk, bias)] in chunk order."""
    steps = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        chunk, sep, value = line.partition(":")
        try:
            if not sep:
                raise ValueError
            steps[int(chunk)] = float(value)
        except ValueError:
            raise ValueError("MMH3PhraseAttention: schedule line %r is not 'chunk: bias', "
                             "e.g. '2: 6'." % line) from None
    return sorted(steps.items())


def bias_at(steps, base, chunk):
    """The bias for a chunk: the last schedule line at or before it, else `base`."""
    for start, value in reversed(steps):
        if chunk >= start:
            return value
    return base


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
                "DiT blocks: positive makes the model listen harder, negative mutes it. "
                "In the looping sampler it can be scheduled per chunk."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", tooltip="Only its tokenizer is used, to find the phrase."),
                MMH3CondSet.Input("cond_set",
                                  tooltip="The cond_set the sampler runs, for the exact text "
                                          "of every chunk's prompt."),
                io.String.Input("phrase", multiline=True, default="",
                                tooltip="The words to listen to, exactly as they appear in the "
                                        "prompt. The first occurrence in each prompt is used; "
                                        "chunks whose prompt lacks it are left alone."),
                io.Float.Input("bias", default=3.0, min=-10.0, max=10.0, step=0.05,
                               tooltip="Added to the attention scores of the phrase's tokens. "
                                       "Its share of attention scales by e^bias. A phrase is a "
                                       "few tokens among thousands of rows, so it takes a lot: "
                                       "3 is about x20, 6 about x400 and still held together "
                                       "on one clip, 10 did not. 0 is off."),
                io.String.Input("blocks", default="", optional=True,
                                tooltip="DiT blocks to bias, e.g. '0-19' or '10-30, 45'. "
                                        "Empty is all 50."),
                io.String.Input("schedule", multiline=True, default="", optional=True,
                                tooltip="Bias per chunk of the looping sampler, one "
                                        "'chunk: bias' per line. Each line holds from its "
                                        "chunk until the next; chunks before the first line "
                                        "use `bias`. The three lines '0: 0', '2: 6', '4: 3' "
                                        "are off for chunks 0-1, 6 for chunks 2-3 and 3 from "
                                        "chunk 4 on. Empty is `bias` everywhere."),
                io.Float.Input("value_scale", default=1.0, min=-4.0, max=4.0, step=0.05,
                               optional=True,
                               tooltip="Multiplies what the phrase's tokens contribute once "
                                       "attended to, in the same blocks and every chunk. 0 "
                                       "removes the phrase's content, negative subtracts it. "
                                       "1 is off."),
            ],
            outputs=[io.Model.Output(), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, clip, cond_set, phrase, bias, blocks="", schedule="",
                value_scale=1.0) -> io.NodeOutput:
        texts = cond_set.get("prompts") or []
        found = {}                      # prompt text -> span, tokenised once each
        spans, lengths = [], []         # per cond_set entry
        for i, cond in enumerate(cond_set["conds"]):
            text = texts[i] if i < len(texts) else ""
            if text not in found:
                found[text] = phrase_span(clip, text, phrase)
            spans.append(found[text])
            lengths.append(cond[0][0].shape[1])
        if not any(spans):
            raise ValueError(
                "MMH3PhraseAttention: the phrase is not in any of the cond_set's %d prompt%s. "
                "It must match exactly, including case and punctuation; a cond_set built "
                "without text (MMH3 Cond To Set) has no prompt to search."
                % (len(spans), "" if len(spans) == 1 else "s"))

        steps = parse_schedule(schedule)
        want = _blocks(blocks)
        lines = []
        if steps:
            parts = [] if steps[0][0] <= 0 else ["chunks before %d: %+.2f" % (steps[0][0], bias)]
            for n, (start, value) in enumerate(steps):
                last = "on" if n + 1 == len(steps) else "to %d" % (steps[n + 1][0] - 1)
                parts.append("chunk %d %s: %+.2f (x%.4g)" % (start, last, value, math.exp(value)))
            lines.append("bias by chunk -- " + ", ".join(parts))
        else:
            lines.append("bias %+.2f: x%.4g the phrase's share of attention" % (bias, math.exp(bias)))
        if value_scale != 1:
            lines[0] += ", value x%.4g" % value_scale
        lines[0] += ", in %s" % ("blocks %s" % blocks.strip() if want else "every block")
        groups = {}
        for i, span in enumerate(spans):
            groups.setdefault(span, []).append(i)
        for span, idx in groups.items():
            label = "every chunk" if len(groups) == 1 else "prompt %s" % _runs(idx)
            if span is None:
                lines.append("%s: phrase not in the prompt, no bias" % label)
            else:
                n = span[0] - span[1]
                lines.append("%s: phrase is %d token%s, ending %d before the end of the prompt"
                             % (label, n, "" if n == 1 else "s", span[1]))
        report = "\n".join(lines)
        logging.info("[MMH3PhraseAttention] " + "; ".join(lines))
        if bias == 0 and value_scale == 1 and not any(value for _start, value in steps):
            return io.NodeOutput(model, report)

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
                chunk = topts.get("mmh3_chunk_index", 0)
                entry = min(chunk, len(spans) - 1)      # the sampler repeats the last prompt
                value = bias_at(steps, bias, chunk)
                start, end = next((a, b) for a, b, kind in layout.segments if kind == "text")
                # a text segment of another length is not this cond_set's prompt
                if spans[entry] is not None and end - start == lengths[entry]:
                    a, b = end - spans[entry][0], end - spans[entry][1]
                    if value:
                        row = torch.zeros([1, rows], dtype=k.dtype, device=k.device)
                        row[:, a:b] = value
                        mask = row if mask is None else mask + row
                    if value_scale != 1:
                        if skip_reshape:
                            v[..., a:b, :] *= value_scale
                        else:
                            v[:, a:b] *= value_scale
            if previous is not None:
                return previous(func, q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
            return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)

        opts["optimized_attention_override"] = override
        return io.NodeOutput(m, report)
