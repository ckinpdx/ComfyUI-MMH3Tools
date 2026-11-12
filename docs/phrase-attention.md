# MiniMax H3 Phrase Attention

Find **MiniMax H3 Phrase Attention** under **MMH3Tools / model**. It makes H3 attend
harder to one phrase of the prompt. Experimental: run once on real weights, on a single
124-frame chunk.

Wire the model through it on the way to the guider and scheduler, `clip` from the text
encoder loader (only its tokenizer is used), and `prompt` from the exact text that was
encoded. From a cond_set that is **MMH3 Cond Select**'s `prompt` output.

- **phrase** is the words to listen to, exactly as they appear in the prompt. The first
  occurrence is used; a phrase that is not in the prompt is an error.
- **bias** is added to the attention scores of the phrase's tokens. The phrase's share
  of each row's attention scales by e^bias before renormalising: 0.7 is about x2, 1.6
  about x5, 2.3 about x10. 0 returns the model unchanged. Negative mutes the phrase;
  -3 looked like the prompt without it.
- **blocks** limits the bias to some of the 50 DiT blocks, e.g. `0-19`. Empty is all.

## Why the score

H3 has no cross-attention. Prompt tokens sit in one sequence with the video and audio
rows, and a phrase matters as much as those rows attend to it. Every block renormalises
its input and the scores come from per-head RMS-normed queries and keys, so making a
token larger or turning it does not make the model listen harder. Pushing a phrase's
Qwen output 31 degrees away from a plain baseline's made no visible change. The score
is the lever.

The bias rides the attention functions' own `mask` argument as one row of the sequence
length, which the backends broadcast over queries and heads. PyTorch attention and
Comfy Kitchen attention both match a reference calculation with it, at no extra memory.
The node wraps any attention override already on the model, so **Model Attention
Backend** stays in effect.

## Limits

- **One prompt per run.** The phrase is located by counting back from the end of the
  prompt, so with a different prompt per chunk the bias lands on the wrong tokens.
- Not checked with sparse attention (SolAttn); bypass it when using this node.
- Which blocks carry the effect is not measured.

`tests/test_phrase_attention.py` covers locating the phrase, where the bias lands in the
packed sequence, the attention calls it must leave alone, block selection and chaining.
