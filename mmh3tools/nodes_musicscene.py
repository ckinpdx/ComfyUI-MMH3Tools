"""Section-by-section prompt building for a MUSIC VIDEO.

Same three-stage shape as `MMH3ScenePlanPrompt` and a different set of rules,
because the cinematic version would fight a song.

WHAT CHANGES, AND WHY

  * **The arc is the song's.** The cinematic beats stage exists to invent an
    escalation and forbid early resolution. A song already has its structure, and
    its choruses are MEANT to land the same way twice. Told to escalate, the model
    pushes through a repeat that should feel like a return.
  * **The words are given.** Nothing is invented about what is sung: the aligned
    lyric supplies the text, and the word onsets supply the shot timestamps. The
    two things the cinematic loop got wrong by inventing -- the words and the
    timings -- are supplied here.
  * **Typography is assigned once, across the whole song.** Decided per chunk with
    lyrics in hand, every chunk reaches for it and you get it on all of them.
  * **Some windows have no words at all.** An intro or instrumental break needs its
    own branch, told that nothing is sung, or it will invent singing to fill it.

The shots stage is the only one that runs per window, and it is handed that
window's verbatim lines on the CHUNK's clock -- so a timestamp it writes is one it
was given, not one it guessed.
"""

import logging

from comfy_api.latest import io

STAGES = ["definitions", "beats", "shots"]
TYPOGRAPHY = ["off", "exact lyrics", "text bursts"]

_BRIEF = """=== THE VIDEO'S IDEA ===

This is what the video is ABOUT, so you can build it. It is not narration and no
character speaks it.

%s"""

_DEFINITIONS = """You are writing the FILM-WIDE sections of a MiniMax H3 prompt for a
MUSIC VIDEO, once. They are reused BYTE-IDENTICALLY in every chunk, so they must be
complete and final now.

Your reply is these six headers, in exactly this order:

    subject_definitions:
    summary:
    retention_analysis:
    detailed_description:
    overall_soundscape:
    non_diegetic_music:

FOUR of them YOU WRITE, in full: subject_definitions, retention_analysis,
overall_soundscape and non_diegetic_music. Writing those IS the job; a reply that
returns them blank is a failed reply.

TWO are per-chunk and get filled in later: summary and detailed_description. Emit
ONLY the bare header for those two, nothing after it, in place and in order.

Emit each header EXACTLY ONCE. No preamble, no commentary, no code fences.

## subject_definitions - one line per label

    <Subject N>  reusable visible content: the performer, a costume, a location, a
                 recurring object or motif.
    <Picture N>  standalone only when the image is a concrete frame anchor; if it
                 merely defines a look, cite it INSIDE a <Subject N> line.
    <Video N>    whole-video relationships only: editing, continuation, borrowed
                 camera movement.
    <Audio N>    an audio asset, bound to a speaker where it maps to one.

DEFINE EVERY LABEL YOU WILL USE, AND USE EVERY LABEL YOU DEFINE. Describe what is
VISIBLE and permanent: build, hair, clothing, markings, the space. Not mood, not
backstory.

A music video returns to the same performer and the same places repeatedly, so
these carry more weight here than in a scene: they are what makes chunk 8 look like
chunk 1.

## retention_analysis - ONE LINE PER LABEL, no exceptions

    <Subject 1>: fully_preserved - her build, hair and coat are identical throughout.

Visible markers: fully_preserved, partially_preserved, attribute_transfer,
weak_reference. Audio markers: fully_copy, partially_copy, reference, weak_reference.
Those are values to CHOOSE BETWEEN, never to list.

## overall_soundscape

THE SONG IS THE AUDIO. Do not invent room tone, weather or footsteps competing with
it. Describe the acoustic world only where the picture implies one, and say plainly
that the track carries the sound.

## non_diegetic_music - the actual track, not an invented score

Describe THIS song: instrumentation, tempo, texture, how the vocal sits. You are
describing something that already exists and will be supplied as audio, so do not
compose an alternative."""

_BEATS = """You are writing the SUMMARY of every chunk of a MUSIC VIDEO, all at once,
as a beat sheet. One summary per chunk, %d of them, separated by a single |
character.

Output ONLY the summaries and the separators. No numbering, no labels, no headings.

Each is one paragraph opening with a bracketed task-type prefix, e.g.
[reference generation], reusing only the labels you were given. Introduce no new
labels.

## THE ARC IS THE SONG'S, NOT YOURS

You are given the whole lyric with its sections and timings. That structure is the
plan; your job is to give it pictures, not to impose a second story on top.

- **Do NOT invent an escalation.** A verse into a chorus is a return, not a rise.
- **A repeated chorus should FEEL like the same chorus.** Come back to the same
  place, the same framing, the same motif. Vary the treatment -- closer, wider,
  more damaged, more crowded -- but do not restage it as somewhere new. Repetition
  is what a chorus is for.
- **Let the sections differ from each other**, though. Verses and choruses should
  not look alike; that contrast is the song's own shape and the video should show it.
- A bridge is usually the one place something genuinely changes. Treat it as the
  exception rather than the rule.

## Per chunk

Each chunk is about %.1f seconds -- long enough for several shots, so a summary can
carry more than one image. Say WHERE it is, WHO is in frame, and what the picture is
doing while those words are sung.

Chunks often straddle a section boundary. When one does, say what the picture does
at the turn.
%s
## Length and shape

Write for the words that are actually sung in each chunk. You have them below with
timings, so a summary that describes a moment nothing is sung in is describing
nothing."""

_TYPO_BEATS = """
## TYPOGRAPHY IS ASSIGNED HERE, ONCE, ACROSS THE WHOLE SONG

You can see all %d chunks. Nothing downstream can, so this is the only place it can
be rationed.

- Name the chunks that carry on-screen text and the chunks that do not. MOST SHOULD
  NOT. Text everywhere reads as a lyric video, and the hook stops landing.
- Prefer the hook, the title line, or a line that repeats. %s
- Say it plainly in the summary, e.g. "on-screen text on the hook" -- the chunk's own
  writer will render it and will not add any of its own.
"""

_TYPO_EXACT = ("The text must be the sung line VERBATIM, so pick lines worth "
               "reading whole.")
_TYPO_BURST = ("The text is a short burst drawn from the sung line -- a fragment, "
               "a word, a re-spelling. It does not have to be the whole line and it "
               "does not have to be literal.")

_SHOTS = """You are writing ONE section of ONE chunk of a MUSIC VIDEO:
detailed_description.

Return ONLY that section's text. No section label, no other sections, no preamble,
no code fences, no markdown.

You are writing **chunk %d of %d**, which covers **%s** and runs %.1f seconds.

## Structure

- One or two style sentences FIRST, before [Shot 1]. Look only, no shot content.
- [Shot 1] carries NO timestamp. Every later shot does:
  "[Shot 2] At 00:03.500, the camera cuts to ..." Times strictly increase and stay
  inside %.1f seconds.
- Use ONLY the labels you were given.

## CUT ON THE WORDS

The timings below are measured from the audio. They are the truth about when things
are sung, and they are given in THIS CHUNK's time, starting at 00:00.000.

- Hang your shot changes on them. A cut that lands on a word onset feels like the
  video is listening; one that lands anywhere else feels like a slideshow.
- DO NOT invent a timestamp. Every time you write should be one you were given, or
  sit deliberately between two of them.
- You do not need a shot per line. Two or three strong shots beat six weak ones.

## Write it as a music video

- Give the camera and the light INTENT. Motion, texture, a practical that pulses
  with the track.
- The performer can be present, absent, or multiplied. A music video is not obliged
  to be literal about who is singing.
- Add visual incident that changes no story: weather, crowd, a surface reacting,
  something breaking at the edge of frame.
- Prefer one exact, strange image over four general ones.
- Write only what a camera can record. No interior states, no symbolism, no
  "representing" or "conveying".
%s%s
=== WHAT IS SUNG IN THIS CHUNK (chunk-relative times) ===

%s
=== WORD ONSETS, for cutting ===

%s
"""

_SHOTS_CONTEXT = """
=== THE CHUNK BEFORE / AFTER (context only -- do not write these) ===

%s
"""

_INSTRUMENTAL = """
## NOTHING IS SUNG IN THIS CHUNK

This window is an intro, an instrumental passage or an outro. There are no words.

- Do NOT write singing, lip movement or dialogue. Nobody is singing here.
- This is where VISUAL EVENT carries the chunk instead: something arrives, breaks,
  turns, floods, empties. Give it the thing the words were doing elsewhere.
- It is also the natural place for a change of location or a reveal, since no lyric
  is anchoring the picture.
- NO on-screen text, whatever the beat sheet assigned elsewhere -- there is no line
  to quote.
"""

_TYPO_SHOTS_EXACT = """
## ON-SCREEN TEXT: VERBATIM

The beat sheet assigned on-screen text to this chunk.

- Put the line in DOUBLE QUOTES, exactly as it appears above, spelling and
  punctuation unchanged. Double quotes are what makes H3 render text ON SCREEN.
- Say where it sits, how it behaves and when it appears, using the times above.
- Quote a line that is actually sung in this chunk, and nothing else.
"""

_TYPO_SHOTS_BURST = """
## ON-SCREEN TEXT: BURSTS

The beat sheet assigned on-screen text to this chunk.

- Put it in DOUBLE QUOTES -- that is what makes H3 render text ON SCREEN.
- It does NOT have to be the whole line. A fragment, one word, a re-spelling, a
  fracture of it. Invention is welcome here: you have the real words in front of
  you, so what you make of them is grounded rather than guessed.
- Keep it SHORT. Text on screen is read in a second or not at all.
- Say where it sits, how it behaves and when it appears.
"""


_SHOWN_THE_CHARACTER = """
## YOU ARE BEING SHOWN THE SUBJECT

Reference image(s) are attached to this message. They are not mood or inspiration:
they ARE the subject, and the same images are handed to the video model as
references, so what you write here has to match what it will be rendering.

- Describe WHAT YOU SEE. Build, face, hair, clothing, markings, palette. Do not
  invent an appearance and do not improve on the one in front of you.
- Every detail you write is rendered in every chunk. A feature you add that is not
  in the image gets generated as though it were real, in all of them.
- If several images are attached they are the SAME subject, from different angles or
  moments. Describe the person, not one of the photographs, and do not define a
  second <Subject> for what is only a different shot of the first.
- Where the image and the brief disagree about how someone looks, THE IMAGE WINS.
  The brief says what happens; the image says who it happens to.
"""


class MMH3MusicScenePlanPrompt(io.ComfyNode):
    """System prompt for one stage of music-video prompt building."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3MusicScenePlanPrompt",
            display_name="MMH3 Music Scene Plan Prompt",
            category="MMH3Tools/prompt",
            description=(
                "Build N chunk prompts for a MUSIC VIDEO section by section. Same "
                "three stages as MMH3 Scene Plan Prompt, different rules: the arc is "
                "the song's rather than invented, the sung words and their timings "
                "are supplied rather than guessed, typography is rationed once across "
                "the whole song, and a window with nothing sung in it gets its own "
                "branch."
            ),
            inputs=[
                io.Combo.Input(
                    "stage", options=STAGES, default="definitions",
                    tooltip="'definitions' writes the film-wide sections ONCE. 'beats' "
                            "writes all N summaries together and assigns typography. "
                            "'shots' expands ONE chunk using its own verbatim lyrics "
                            "and word onsets."),
                io.String.Input(
                    "brief", multiline=True, default="",
                    tooltip="What the video is about. Dramatised, never narrated."),
                io.Int.Input(
                    "chunk_count", default=8, min=1, max=64,
                    tooltip="Wire MMH3 Window Plan's window_count."),
                io.Float.Input(
                    "seconds_per_chunk", default=19.3, min=0.2, max=150.0, step=0.1,
                    tooltip="Bounds the shot timestamps and tells the writer how much "
                            "fits in a chunk."),
                io.Combo.Input(
                    "typography", options=TYPOGRAPHY, default="off",
                    tooltip="'off' never puts words on screen. 'exact lyrics' quotes "
                            "the sung line verbatim. 'text bursts' allows fragments "
                            "and re-spellings drawn from it -- invention grounded in "
                            "the real words rather than replacing them."),
                io.Int.Input(
                    "beat_index", default=0, min=0, max=63, optional=True,
                    tooltip="'shots' only: which chunk this call writes, 0-based."),
                io.String.Input(
                    "beat_sheet", multiline=True, default="", optional=True,
                    tooltip="'shots' only: the full pipe-separated beat sheet."),
                io.String.Input(
                    "definitions", multiline=True, default="", optional=True,
                    tooltip="The definitions text, so labels exist and none are "
                            "invented."),
                io.String.Input(
                    "lyrics", multiline=True, default="", optional=True,
                    tooltip="'beats': the whole sectioned lyric. 'shots': MMH3 Lyrics "
                            "to Windows' `lyrics` for THIS chunk, already on the "
                            "chunk's clock."),
                io.String.Input(
                    "context_lyrics", multiline=True, default="", optional=True,
                    tooltip="'shots' only: the neighbouring windows' lines, for "
                            "continuity. Wire prev_lyrics and next_lyrics through a "
                            "concat, or either alone."),
                io.String.Input(
                    "section", default="", optional=True,
                    tooltip="'shots' only: MMH3 Lyrics to Windows' `section`, which "
                            "names a boundary falling inside the chunk."),
                io.String.Input(
                    "shot_times", multiline=True, default="", optional=True,
                    tooltip="'shots' only: MMH3 Lyrics to Windows' `shot_times` -- the "
                            "word onsets the writer should cut on instead of "
                            "inventing timestamps."),
                io.Boolean.Input(
                    "has_lyrics", default=True, optional=True,
                    tooltip="'shots' only: wire MMH3 Lyrics to Windows' `has_lyrics`. "
                            "False switches to the instrumental branch, which forbids "
                            "singing and on-screen text and asks for visual event "
                            "instead."),
                io.String.Input(
                    "extra_rules", multiline=True, default="", optional=True,
                    tooltip="Appended verbatim as a final block."),
                io.Boolean.Input(
                    "reference_images", default=False, optional=True,
                    tooltip="Turn on when reference image(s) are wired to the "
                            "LlamaGenerate running the DEFINITIONS stage. Without it "
                            "the model is handed pictures with no instruction about "
                            "them and describes an invented character anyway. Tells it "
                            "the images ARE the subject, that the same images go to "
                            "the video model, that several images are one person from "
                            "different angles, and that the image beats the brief on "
                            "appearance. Needs a vision-capable model on that call."),
            ],
            outputs=[
                io.String.Output(display_name="system_prompt"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, stage, brief, chunk_count, seconds_per_chunk, typography,
                beat_index=0, beat_sheet="", definitions="", lyrics="",
                context_lyrics="", section="", shot_times="", has_lyrics=True,
                extra_rules="", reference_images=False) -> io.NodeOutput:
        n = max(1, int(chunk_count))
        secs = float(seconds_per_chunk)
        notes, parts = [], []

        # Only definitions is shown the image, deliberately: the description is
        # written once and reused verbatim, and re-deriving it per call is how the
        # subject drifts between chunks.
        if reference_images and stage != "definitions":
            notes.append("reference_images is on for the %s stage, where it does "
                         "nothing -- only definitions is shown the image, so the "
                         "description is written once instead of re-derived" % stage)

        if stage == "definitions":
            parts.append(_DEFINITIONS)
            if reference_images:
                parts.append(_SHOWN_THE_CHARACTER)
            else:
                notes.append("no reference_images, so subject_definitions describes an "
                             "INVENTED character; wire the image to this stage's "
                             "LlamaGenerate and turn this on")

        elif stage == "beats":
            typo = ""
            if typography != "off":
                typo = _TYPO_BEATS % (n, _TYPO_EXACT if typography == "exact lyrics"
                                      else _TYPO_BURST)
            parts.append(_BEATS % (n, secs, typo))
            if not (lyrics or "").strip():
                notes.append("no lyrics given, so the beat sheet cannot follow the "
                             "song; wire the sectioned lyric or the alignment's lines")

        else:
            i = max(0, min(int(beat_index), n - 1))
            if int(beat_index) != i:
                notes.append("beat_index %d outside 0..%d; clamped to %d"
                             % (int(beat_index), n - 1, i))
            sung = (lyrics or "").strip()
            if has_lyrics and not sung:
                raise ValueError(
                    "MMH3MusicScenePlanPrompt: has_lyrics is true but `lyrics` is "
                    "empty. The shots stage writes against the words sung in THIS "
                    "chunk -- without them it invents them, which is the failure this "
                    "node exists to remove. Wire MMH3 Lyrics to Windows, or set "
                    "has_lyrics false for an instrumental window.")

            branch = _INSTRUMENTAL if not has_lyrics else ""
            typo_block = ""
            if has_lyrics and typography == "exact lyrics":
                typo_block = _TYPO_SHOTS_EXACT
            elif has_lyrics and typography == "text bursts":
                typo_block = _TYPO_SHOTS_BURST

            parts.append(_SHOTS % (
                i + 1, n, section.strip() or "an unnamed section", secs, secs,
                branch, typo_block,
                sung or "(nothing is sung in this chunk)",
                (shot_times or "").strip() or "(none)"))
            if (context_lyrics or "").strip():
                parts.append(_SHOTS_CONTEXT % context_lyrics.strip())
            if (beat_sheet or "").strip():
                parts.append("=== THE BEAT SHEET ===\n\n%s" % beat_sheet.strip())
            else:
                notes.append("no beat_sheet, so this chunk cannot see the song's shape "
                             "or what typography it was assigned")
            if has_lyrics and not (shot_times or "").strip():
                notes.append("no shot_times, so the writer has no onsets to cut on and "
                             "will invent timestamps")

        if (brief or "").strip():
            parts.append(_BRIEF % brief.strip())

        if (definitions or "").strip() and stage != "definitions":
            parts.append("=== LABELS ALREADY DEFINED (use these, invent none) ===\n\n%s"
                         % definitions.strip())
        elif stage != "definitions":
            notes.append("no `definitions`, so the writer may invent labels the "
                         "assembled prompt does not define")

        if stage == "beats" and (lyrics or "").strip():
            parts.append("=== THE WHOLE LYRIC, WITH SECTIONS AND TIMES ===\n\n%s"
                         % lyrics.strip())

        if (extra_rules or "").strip():
            parts.append(extra_rules.strip())

        system = "\n\n".join(parts)
        report = ("stage: %s | %d chunk%s of %.1fs | typography: %s%s\n%s"
                  % (stage, n, "" if n == 1 else "s", secs, typography,
                     (" | chunk %d of %d%s"
                      % (min(int(beat_index), n - 1) + 1, n,
                         "" if has_lyrics else " (INSTRUMENTAL)"))
                     if stage == "shots" else "",
                     "\n".join("  ! " + x for x in notes) if notes
                     else "  no warnings"))
        logging.info("[MMH3MusicScenePlanPrompt] %s", report.splitlines()[0])
        return io.NodeOutput(system, report)
