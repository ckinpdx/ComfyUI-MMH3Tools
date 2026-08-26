"""Pick an H3 text embedding from a list, never by typing its filename.

`MMH3ReferenceMultiPrompt.embeddings` takes a spec string, and a typed field is the
wrong way to choose a model file: a typo resolves to nothing, and core drops an
unresolvable `embedding:` with a log line rather than an error. This is the picker.
Chain one per embedding and wire the last one's output into that input.

The dropdown lists the WHOLE folder, like every other loader in ComfyUI. An embeddings
folder usually holds SD/SDXL textual inversions too -- 768 or 1280 wide against H3's
5120, a different model's vocabulary -- and an earlier cut hid those. That was wrong on
two counts: no other picker filters, and doing it meant reading every header at schema
time, so a file whose header would not parse was included or excluded on the strength
of an exception. The width is checked at execute instead, where the error can name both
numbers.
"""

import logging

from comfy_api.latest import io

H3_WIDTH = 5120


def _row_shape(path):
    """(rows, width) from a safetensors header, without reading the tensor."""
    try:
        from safetensors import safe_open
        with safe_open(path, "pt") as f:
            for k in f.keys():
                shape = f.get_slice(k).get_shape()
                if len(shape) == 2:
                    return int(shape[0]), int(shape[1])
    except Exception:
        return None
    return None


def _embedding_names():
    """Every file in models/embeddings, unfiltered.

    Deliberately NOT filtered to H3-width files. No other loader in ComfyUI filters --
    LoraLoader, CheckpointLoaderSimple and CLIPLoader all list the whole folder -- and a
    picker that hides files is surprising in a way a wrong choice is not. Filtering also
    meant reading every header at schema time, which is both a cost and a fragility: a
    file whose header does not parse gets included or excluded on the strength of an
    exception. The width is checked at EXECUTE instead, where a wrong choice can be
    named precisely."""
    try:
        import folder_paths
        return folder_paths.get_filename_list("embeddings")
    except Exception:
        return []


class MMH3EmbeddingSelect(io.ComfyNode):
    """Choose an H3 text embedding and when it applies."""

    @classmethod
    def define_schema(cls):
        options = _embedding_names() or ["<no embeddings found>"]
        return io.Schema(
            node_id="MMH3EmbeddingSelect",
            display_name="MMH3 Embedding Select",
            category="MMH3Tools/conditioning",
            description=(
                "Pick an H3 text embedding from `models/embeddings/` and say which "
                "chunks carry it. Wire the output into MMH3 Reference "
                "(Multi-Prompt)'s `embeddings`; chain several through `previous` to "
                "stack them.\n\n"
                "Every file in the folder is listed, the way the other loaders do it. "
                "H3's embeddings are 5120 wide; an SD or SDXL textual inversion in the "
                "same folder is 768 or 1280 and belongs to a different model's "
                "vocabulary, so choosing one is refused at run time with both widths "
                "named.\n\n"
                "Cost: each embedding occupies its own row count in token slots (50 to "
                "142 for the MiniMax set), attended at EVERY sampling step of every "
                "chunk it is on."
            ),
            inputs=[
                io.Combo.Input(
                    "embedding", options=options,
                    tooltip="Any file in `models/embeddings/`. H3's are 5120 wide; a "
                            "768/1280 SD or SDXL inversion is refused at run time "
                            "rather than hidden here."),
                io.String.Input(
                    "chunks", default="all",
                    tooltip="Which chunks carry it: `all`, a single 1-based index like "
                            "`3`, or a range like `4-6`. Scheduling works because every "
                            "chunk already has its own prompt."),
                io.String.Input(
                    "previous", optional=True, force_input=True,
                    tooltip="Another MMH3 Embedding Select's output, to stack. Their "
                            "costs and their effects are both additive."),
            ],
            outputs=[
                io.String.Output(display_name="embeddings"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, embedding, chunks="all", previous="") -> io.NodeOutput:
        if not embedding or embedding.startswith("<no embeddings"):
            raise ValueError(
                "MMH3EmbeddingSelect: no embedding chosen. `models/embeddings/` has no "
                "H3-width file, or none was picked.")
        rng = (chunks or "all").strip() or "all"
        if rng.lower() != "all":
            body = rng.replace("-", " ").split()
            if not body or not all(p.isdigit() for p in body):
                raise ValueError(
                    "MMH3EmbeddingSelect: `chunks` must be `all`, an index like `3`, or "
                    "a range like `4-6`. Got %r." % chunks)

        # the spec the reference node parses: one line per embedding
        line = embedding if rng.lower() == "all" else "%s: %s" % (embedding, rng)
        spec = ((previous or "").rstrip() + "\n" + line).strip()

        rows = None
        try:
            import folder_paths
            path = folder_paths.get_full_path("embeddings", embedding)
            shape = _row_shape(path) if path else None
        except Exception:
            shape = None
        if shape is not None:
            rows, width = shape
            if width != H3_WIDTH:
                raise ValueError(
                    "MMH3EmbeddingSelect: %s is %d wide and H3's text embeddings are "
                    "%d. That is an SD/SDXL textual inversion -- a different model's "
                    "vocabulary -- and splicing it would hand H3 rows it has no "
                    "meaning for." % (embedding, width, H3_WIDTH))

        lines = ["MMH3 Embedding Select", ""]
        for i, entry in enumerate(spec.splitlines()):
            lines.append("  %d. %s" % (i + 1, entry))
        lines.append("")
        lines.append("  this one: %s slots, on %s"
                     % (rows if rows is not None else "?",
                        "every chunk" if rng.lower() == "all" else "chunk(s) " + rng))
        lines.append("")
        lines.append("  Wire `embeddings` into MMH3 Reference (Multi-Prompt). Slots are "
                     "attended at every sampling step of every chunk they are on.")
        logging.info("[MMH3EmbeddingSelect] %s -> %s (%s slots)", embedding, rng, rows)
        return io.NodeOutput(spec, "\n".join(lines))
