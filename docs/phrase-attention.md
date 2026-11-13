# MiniMax H3 Phrase Attention

Find **MiniMax H3 Phrase Attention** under **MMH3Tools / model**. It makes H3 attend
harder to one phrase of the prompt. Experimental: run once on real weights, on a single
124-frame chunk.

Wire the model through it on the way to the guider and scheduler, `clip` from the text
encoder loader (only its tokenizer is used), and `cond_set` from the same cond_set the
sampler runs. The node reads every chunk's prompt from it.

- **phrase** is the words to listen to, exactly as they appear in the prompt. The first
  occurrence in each prompt is used. A chunk whose prompt lacks the phrase is left alone;
  a phrase that is in no prompt is an error.
- **bias** is added to the attention scores of the phrase's tokens. The phrase's share
  of each row's attention scales by e^bias before renormalising. A phrase is a few
  tokens among thousands of rows, so it takes a lot: 3 is about x20, 6 about x400.
  0 returns the model unchanged. Negative mutes the phrase;
  -3 looked like the prompt without it.
  High values on a head-movement phrase made the head move to the beat more, in a
  clip whose pinned audio was a song.
- **blocks** limits the bias to some of the 50 DiT blocks, e.g. `0-19`. Empty is all.
- **schedule** sets the bias per chunk of the looping sampler, one `chunk: bias` per
  line. Each line holds from its chunk until the next, and chunks before the first line
  use `bias`. Empty is `bias` everywhere.

      0: 0
      2: 6
      4: 3

  is off for chunks 0-1, 6 for chunks 2-3 and 3 from chunk 4 on. This works with one
  prompt repeated for every chunk: the phrase stays in the prompt throughout and only
  how hard the model listens to it changes.

- **value_scale** multiplies what the phrase's tokens contribute once attended to,
  where the bias changes how much they are attended to. 1 is off, 0 removes the
  phrase's content, negative subtracts it. It follows `blocks` and applies in every
  chunk whose prompt has the phrase; the schedule does not change it. Not run on real
  weights.

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
Backend** stays in effect, and its bias adds to one already set by another node.

## Carry Attention

**MiniMax H3 Carry Attention** is the same bias on the frames a Looping Sampler chunk
carries from the previous one. Chain over-bake follows the model reading its own output
([looping-sampler.md](looping-sampler.md) §10); the carry settings change what the next
chunk is shown, and this changes how hard it reads it, with the carry left clean.

Wire the model through it on the way to the guider and scheduler. **carry_bias** is added to
the scores of the carried video rows: negative makes the chunk copy its carry less (-1
is about a third of the attention, -2 about a seventh), positive more. **blocks** is as
above. It stacks with Phrase Attention in either order.

**reference_bias** is the same bias on the reference image and video rows, in the same
blocks: positive makes every chunk read its references more (+1 is about 2.7x the
attention). It is meant as the other half of a negative carry bias, sending the
attention taken off the chunk's own output to a clean image. It needs no carry, so it
also applies to the first chunk and under any sampler.

**reference_audio_bias** is the same for the reference audio rows: a voice reference,
and the sound of a video reference, whose picture follows `reference_bias`.

- The carry bias is `carry="mask"` only. The sampler publishes how many of each chunk's
  first video latents are carried; a keyframe carry, or any other sampler, gets no
  carry bias.
- The carry bias is video only. The carried audio rows are not biased.
- Every row is biased, the carried rows' attention to each other included, not only the
  new frames' attention to them.
- **173.75 s held** (2026-11-12): a static talking head over supplied audio, 27 chunks
  of 192 frames with a 39-frame overlap, `carry_bias` -1 and `reference_bias` 2 in all
  blocks. No visible bake, drift or broken join, judged by eye, on one run. Hybrid
  fl2va/ref2va checkpoint, 640x640, 12 steps, no anti-burn LoRA, `carry_color_match`
  off, the reference image also anchored as the first frame. No control at 0 / 0 was
  rendered at this length. It is the shipped `MMH3_Looping_CarryAttention` workflow.
- Mapping the carry bias on 32 s in 5 chunks, same shot: -2 held the joins with a
  slight bake left; -3 and -4 broke them. `reference_bias` alone, with the carry bias
  at 0, still left some burn: both do part of the work.
- **The clip's own audio is not biased, on purpose.** A bias and a value scale on the
  target audio rows were built and tested on a static talking head over a pinned
  track (2026-11-12), hoping for a more audio-reactive picture. The bias only
  dampened: positive values did nothing until the image degraded above 5, and no
  block range helped. The value scale did nothing good either. Both were removed.
- `reference_audio_bias` has not been run on real weights. More attention to a
  reference image brings its pose, framing and background along with its look.
- All references of a kind get the same bias. A larger reference has more rows and
  so already takes more attention at the same value.

## Limits

- **Per-chunk behaviour needs the looping sampler.** It publishes the chunk index the
  node reads. Under any other sampler the run is treated as chunk 0. Chunk numbers
  count from 0, as in the sampler's report.
- Per-chunk prompts and the schedule are covered by tests only; they have not been run
  on real weights.
- If the cond_set wired here is not the one the sampler runs, nothing is biased.
- Not checked with sparse attention (SolAttn); bypass it when using this node.
- Which blocks carry the effect is not measured.
- The usable range is wide but not mapped: on one talking-head clip a bias of 6 still
  held together and 10 did not. Values between are untested.

`tests/test_phrase_attention.py` covers locating the phrase, where the bias lands in the
packed sequence, the attention calls it must leave alone, block selection, a prompt per
chunk, the schedule, the sampler publishing the chunk, and chaining.
`tests/test_carry_attention.py` covers the carried rows against the live layout class,
the reference rows, the calls left alone, and stacking with Phrase Attention.
