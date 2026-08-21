# MiniMax H3 Music Video — Field Guide

How the music-video chain fits together, and why it is shaped the way it is. The
README documents what each node *is*; this documents what the chain *does to the
generation*, which decisions are already made for you, and which are yours.

**Status: one full run, 2026-08-15.** Section 8 is what that run taught. Section 9
lists what is still unknown, and it is longer. Everything describing *structure* —
the grid, the timestamp rebasing, which stage supplies what — is read off the code
and holds regardless. Everything describing *quality* comes from a single song, and
one song is not a measurement.

---

## 1. Mental model

A music video is a cinematic chain with the arc removed. The song already has the
arc, the words and their timings; the pipeline's whole job is to stop the model
inventing things that were handed to it.

Every failure in the cinematic pipeline was the section loop being asked to invent
what should have been supplied:

| invented | symptom | supplied by |
|---|---|---|
| the arc | every chunk resolved | `beats` stage |
| the characters | orphan `<Subject 2>` | `definitions` stage |
| the words | bad lyrics, typography claims with no text | aligned lyrics |
| shot times | everything crammed in the first 5.2s | word onsets |

So the chain is a supply line:

```
song ──┬─ vocal stem ── MMH3ForcedAlign ──── alignment_json ──┐
       │                                                      │
       └─ full mix ──── MMH3MusicAnalysis ── bars, energy ────┤
                                                              │
                     MMH3WindowPlan ── the grid ──────────────┤
                                                              ▼
                                             MMH3LyricsToWindows
                                                              │
                        per window: lyrics · prev/next · section · shot_times · has_lyrics
                                                              │
   MMH3MusicScenePlanPrompt ×3   definitions → beats → shots ─┘
                                                              │
                                              MMH3LoopingSampler
```

Three structural facts drive almost every decision below.

1. **The song is the clock, and it is measured, not asserted.** Word onsets come
   from forced alignment against the real vocal. Nothing downstream guesses a time.
2. **The window grid is uniform and fixed before any prompt is written.** Sections
   are context, not structure — see §2.
3. **The audio is pinned, not generated.** `non_diegetic_music` describes *this*
   track rather than composing an alternative, and `overall_soundscape` opens with
   "the song is the audio" so the model does not invent room tone competing with it.

---

## 2. The grid: windows, not sections

**Beat = render chunk.** Lyrics are applied by *window length*, not by verse or
chorus. The existing loop, `beat_index` and `MMH3PromptPart` carry over from the
cinematic pipeline unchanged.

This is the decision most worth understanding, because the alternative is the
obvious one. A section→chunk mapping layer — "the chorus is beats 4–6" — was
considered and rejected: the `17j+5` frame grid is non-negotiable, so a mapping
layer has to reconcile two grids that do not divide, and that reconciliation is
where drift hides. Uniform windows cannot drift because there is nothing to
reconcile.

Sections still reach the writer, as **context**. A window straddling a boundary
reports `chorus -> bridge (bridge begins at 00:07.000)` rather than pretending it
sits in one.

### Window-relative timestamps

H3 shot times are measured from the start of the **chunk**, not the song. A window
opening at 70.15s holding a word at 72.40s must emit `00:02.250`. Absolute time
produces prompts H3 cannot act on **and nothing errors** — this is the failure that
silently wastes a render. `MMH3LyricsToWindows` rebases everything, including the
neighbouring-window context, so a prev/next line does not read as a second,
contradictory timeline.

### `has_lyrics`

Intros, instrumental breaks and outros are real windows with nothing sung in them.
Without the flag they get a prompt left to invent singing. With it, the `shots`
stage switches to an instrumental branch that forbids singing, asks for visual
event instead, and **suppresses typography even when the beat sheet assigned it** —
there is no line to quote.

### Why the inputs mirror `MMH3SplitAudioToWindows`

They read the same `_plan` as `MMH3WindowPlan` and the sampler. Three nodes, one
schedule, so they cannot disagree about which frames window *i* covers. If you wire
them from different numbers you have built two grids and the symptom will look like
a model problem.

---

## 3. Alignment

`MMH3ForcedAlign` needs **isolated vocals** — any separator will do; the shipped
workflow uses MelBandRoFormer. It refuses to return a word sequence that differs
from its input, because a misaligned lyric is fiction every consumer downstream
would quote verbatim.

**The report is the interface.** It prints the section map — the one line you can
check against your own ears — and classifies anomalies using the audio itself: a gap
over silence is a correct skip, a gap over audio is a skipped passage, and words
sitting on silence are misplaced. That distinction is what separates a musical pause
from a misalignment, and no amount of timing arithmetic can make it.

⚠ **Feed it the lyrics as PERFORMED, not as prompted.** Alignment assumes text and
audio hold the same words the same number of times. Suno repeats hooks and stutters
refrains; one copy of a line against three utterances leaves the aligner to pick one
and strand the rest, surfacing as large gaps, stretched words, and whole sections
landing early. **No parameter fixes this.** `nonspeech_skip`, `max_word_dur`, `vad`
and `snap_to_onset` all decide *where a word may land*; none can conjure two missing
repeats. Writing the line three times does.

⚠ **`vad` is for spoken word, not song.** Silero is trained on speech and does not
fire on singing — on a produced vocal, 131 of 190 words came back zero-length.

---

## 4. Music analysis — what survives, and why

`MMH3MusicAnalysis` reads the **full mix**, not the stem: BPM, key and mode, a 4/4
bar grid, and a 10 Hz RMS energy curve.

It is a port of music-director's `music.py` with the cut-salience blend and
agglomerative segmentation removed. Both of those exist to *choose* scene
boundaries, and the windows here are uniform and already fixed — there is nothing
left for them to decide.

What survives is what still helps inside a window someone else decided:

- **Bar lines** are cut candidates alongside word onsets.
- **Energy** is the only thing that tells an instrumental window whether the music
  there is a soft fall or a drop. Without it, `has_lyrics: false` windows are
  identical to each other as far as the writer can see.

---

## 5. The three stages

Same three as `MMH3ScenePlanPrompt` — `definitions`, `beats`, `shots` — with the
rules inverted where a song demands it:

| | cinematic | music video |
|---|---|---|
| the arc | invented, nothing resolves before beat N | **the song's** — "do NOT invent an escalation" |
| a repeat | would be redundant | **should feel like the same chorus** |
| the words | invented | supplied, verbatim, per window |
| shot times | invented | supplied as word onsets |

`definitions` runs **once** and is inherited. `beats` writes all N summaries
together — which is what lets it ration typography across the whole song. `shots`
expands one chunk, and is the only stage inside the loop.

### `reference_images`

Turn it on when images are wired to the `LlamaGenerate` running the **definitions**
stage. It tells the model the images ARE the subject, that several images are one
person from different angles, and that the image beats the brief on appearance.
Without it a vision model handed pictures and no instruction describes an invented
character anyway.

**Definitions only.** The description is written once and reused; re-deriving it per
call is exactly how a subject drifts across a song.

---

## 6. Typography

**Rationed in `beats`, once, across the whole song** — the rule is "most should
not." Decided per chunk with lyrics in hand, every chunk reaches for it and the
result reads as a lyric video.

Two modes. `exact lyrics` quotes the sung line verbatim. `text bursts` allows
fragments and re-spellings — invention *grounded in* the real words rather than
replacing them. Burst mode is where invention is the point; the bad words in early
runs came from omni **transcription**, which was a bad source, not from invention.

`off` never puts words on screen. An instrumental window suppresses typography
regardless of what `beats` assigned.

See §8 for what a real run did to this, which was the largest single correction in
the chain.

---

## 7. Recipes

**Straight music video from a finished track**
Vocal stem → `MMH3ForcedAlign` · full mix → `MMH3MusicAnalysis` · `use_input_audio`
ON so the track is pinned · typography `text bursts` · `reference_images` on the
definitions call with a character batch · window ~20s.

**Instrumental / no vocal**
Skip alignment entirely; `has_lyrics` false everywhere. Energy from
`MMH3MusicAnalysis` is then the *only* signal distinguishing one window from the
next — without it every window reads the same to the writer.

**Lyric video look, deliberately**
`exact lyrics`, and drop the "most should not" ration — the thing the ration exists
to prevent is precisely this look, so getting it means turning the guard off rather
than fighting it.

**Song with heavy repeats (Suno hooks, stuttered refrains)**
Write the repeated line as many times as it is actually sung *before* aligning (§3).
This is a lyrics-input fix, not a parameter fix.

---

## 8. Symptom → lever

| Symptom | Look at |
|---|---|
| typography renders flat, like a single-word subtitle | two causes, both in the rules — see §9. Scale is the first decision; mid-sized and centred **is** the subtitle look |
| the on-screen word is a function word ("much") | `text bursts` asking for SHORT — shortest word in any lyric is a function word. Fixed in the rules; if it recurs, the burst has to mean something standing alone |
| every chunk invents its own font | type identity is chosen ONCE in `beats` and inherited. Chosen per chunk, the video has no design |
| everything crammed into the first few seconds | shot times not reaching the writer — check `MMH3LyricsToWindows` `shot_times` is wired |
| prompts reference times H3 cannot act on, no error | absolute instead of window-relative timestamps (§2). Nothing raises; check a prompt by eye |
| an instrumental window tries to sing | `has_lyrics` not wired from `MMH3LyricsToWindows` |
| all instrumental windows look alike | energy not wired — it is the only thing separating a fall from a drop (§4) |
| the model composes music over the track | `non_diegetic_music` describing an alternative rather than *this* track; `use_input_audio` off |
| whole sections land early, words stretched, large gaps | the lyric text does not match what is sung — repeats (§3). No parameter fixes it |
| 131 of 190 words zero-length | `vad` on a sung vocal (§3) |
| subject drifts across the song | definitions being re-derived per call instead of written once (§5) |
| the video escalates instead of following the song | cinematic rules — wrong node, or `beats` not supplied the song's own arc |
| a chunk comes back in slow motion | not expected on this setup (§9) — `"live-action video"` at the front of the style sentences is the documented remedy |
| windows and chunks disagree about frames | two grids — the align/window/sampler nodes must read the same three numbers (§2) |

---

## 9. Observed — 2026-08-15

From the first full music-video run. Recorded because the same wording would be
re-derived otherwise, and because two of these contradict what a search turns up.

- **Typography rendered flat, "like a single word subtitle".** Two causes, both in
  the rules rather than the model. *Which* word: `text bursts` said "keep it SHORT",
  and the shortest word in any lyric is a function word — the output was the word
  **"much"**. It now requires a burst to mean something standing alone, bans function
  words by name, and offers a test: printed alone on a poster, does it read as a
  statement or as a fragment someone forgot to finish? *How* it looked: the rule
  asked only where the text "sits", which is satisfied by centring it — and centred
  at readable size **is** the subtitle look. Scale is now the first decision, and the
  block says outright that mid-sized and centred is the one option that reads as
  captioning.

- **Thematic type beats a font description.** Letters made of circuit traces for a
  machine song, vapour and sugar-floss for a candy one. The identity is chosen ONCE
  in `beats` and inherited by every chunk — chosen per chunk, each invents its own
  and the video has no design.

- **Split frames and RGB channel split are wanted, not artifacts.** Two places, two
  times or two framings named inside ONE shot is what makes H3 divide the frame. It
  was happening as a side effect of shots describing several things at once; it is
  now stated as a technique to reach for on purpose. The general rule underneath
  both: **a treatment belongs if the song earns it** — the good emergent effects came
  from briefs dense enough for the model to reach into their world.

- **Slow motion is a choice here, not a drift.** Community reports describe H3
  falling into slow motion unbidden, but those cluster around LoRA use (the
  Realism-People LoRA "adds slow mo at times", lightning LoRAs "struggled with slow
  motion"). ck has never seen it unasked on this setup, so the prompt treats it as
  something you ask for. `"live-action video"` at the front of the style sentences is
  kept as a documented remedy if a chunk ever comes back slower than intended.

---

## 10. Not yet measured

Honest about being unknown.

- **Everything above rests on one song.** One run, one genre, one voice. The
  typography corrections are the only findings that have been re-derived from a
  stated cause rather than observed once.
- **Which window length suits a song.** 20s is what has been run. Whether a faster
  song wants shorter windows — more cuts, less room per prompt — is untried, and the
  grid makes it cheap to test.
- **Whether bar lines actually beat word onsets as cut points.** Both are supplied
  to the writer; which one it should prefer, and when, has not been compared.
- **`text bursts` versus `exact lyrics` at equal ration.** The modes have not been
  run against each other on the same song.
- **Instrumental windows.** The branch exists and is exercised, but no run has
  turned on whether energy alone gives the writer enough to differentiate them.
- **Multi-voice songs.** Per-chunk `[SPEAKER n]` works in the sampler, but nothing
  in this chain maps a duet's parts to speakers — the aligner returns one word
  sequence with no speaker attribution.
