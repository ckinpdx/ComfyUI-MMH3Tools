"""Fun ControlNet across a chunked render.

Core's `MiniMaxH3FunControlPatch` has two faults under chunking, and neither shows
up as an error:

  1. `_fit_frames` picks hint frames with `torch.arange(frame_count)` -- from ZERO,
     for the control video, the inpaint mask and the source video alike. Every chunk
     is therefore driven by the control video's OPENING frames.
  2. `prepare_control_latent` caches on `target_shape`, and every chunk of a render
     shares a shape, so chunk 0's encode is reused for all of them.

Both are right for a whole-clip pass and wrong for a chunked one. The output is
plausible; nothing in the log says the control never advanced.

The fix is a subclass that reads the chunk's first frame from
`transformer_options["mmh3_control_frame0"]`, which MMH3LoopingSampler already
publishes per chunk. The offset shifts the arange and joins the cache key. Unset
means 0, which is exactly right for a single pass through a stock sampler -- this
is inert until something chunks.

REWRITTEN FOR THE MODEL-PATCH API (core #15975, merged 2026-08-31). H3's Fun
ControlNet used to be a ControlNet: a CONDITIONING-in/out node with the control
carried on the cond, wrapping `comfy.controlnet.MiniMaxH3ControlNet`. That class no
longer exists. It is now a MODEL PATCH -- MODEL + MODEL_PATCH in, MODEL out, with a
DIFFUSION_MODEL wrapper -- so this node sits in the model chain beside MMH3 Timeline
Preview rather than between the multiprompt node and the sampler.

The port is smaller than what it replaced: the wrapper is handed
`transformer_options` on every call, so the offset can simply be read, where the old
version had to swap attributes on a cached ControlNet object around a delegated call.
"""

import logging

from comfy_api.latest import io

OFFSET_KEY = "mmh3_control_frame0"
# What the subclass overrides. If core renames one, refuse rather than mis-window.
REQUIRED = ("_fit_frames", "prepare_control_latent", "diffusion_model_wrapper",
            "control_latent", "control_latent_shape")


def _patch_class():
    """Core's control patch class, or None on a ComfyUI without it."""
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch
        return MiniMaxH3FunControlPatch
    except Exception:
        return None


def _apply_node():
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlNetApply
        return MiniMaxH3FunControlNetApply
    except Exception:
        return None


def _supported():
    base = _patch_class()
    if base is None:
        return False, ("this ComfyUI has no MiniMaxH3FunControlPatch. H3 Fun "
                       "ControlNet landed as a model patch in core on 2026-08-31; "
                       "update ComfyUI.")
    missing = [a for a in REQUIRED if not hasattr(base, a) and a not in
               getattr(base, "__init__", lambda: None).__code__.co_names]
    hard = [a for a in ("_fit_frames", "prepare_control_latent",
                        "diffusion_model_wrapper") if not hasattr(base, a)]
    if hard:
        return False, ("core's MiniMaxH3FunControlPatch no longer has %s, so the "
                       "per-chunk window cannot be applied. Without it every chunk "
                       "would be driven by the control video's opening frames."
                       % ", ".join(hard))
    return True, ""


def make_chunk_aware(base):
    """A subclass of core's control patch that honours the chunk offset."""

    class ChunkAware(base):
        _mmh3_offset = 0

        def diffusion_model_wrapper(self, executor, x, timestep, context,
                                    transformer_options={}, **kwargs):
            # Read BEFORE delegating: the base calls prepare_control_latent inside
            # this method, so the offset has to be in place first.
            self._mmh3_offset = int((transformer_options or {}).get(OFFSET_KEY, 0) or 0)
            return super().diffusion_model_wrapper(
                executor, x, timestep, context, transformer_options, **kwargs)

        def _fit_frames(self, frames, frame_count, width, height):
            off = int(getattr(self, "_mmh3_offset", 0) or 0)
            if off > 0 and frames is not None and int(frames.shape[0]) > off:
                frames = frames[off:]
            return super()._fit_frames(frames, frame_count, width, height)

        def prepare_control_latent(self, target_shape):
            # The cache keys on shape alone upstream, and every chunk shares a
            # shape -- so without the offset in the key, chunk 0's encode is handed
            # to every later chunk.
            key = (tuple(target_shape), int(getattr(self, "_mmh3_offset", 0) or 0))
            if self.control_latent is not None and self.control_latent_shape == key:
                return
            self.control_latent = None
            self.control_latent_shape = None
            super().prepare_control_latent(target_shape)
            self.control_latent_shape = key

    return ChunkAware


class MMH3ApplyControl(io.ComfyNode):
    """Apply a Fun ControlNet that follows the chunk being rendered."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ApplyControl",
            display_name="MMH3 Apply ControlNet",
            category="MMH3Tools/sampling",
            description=(
                "Apply a MiniMax H3 Fun ControlNet that follows the chunk being "
                "rendered, instead of the control video's opening frames.\n\n"
                "Core selects hint frames with an arange from ZERO -- for the control "
                "video, the inpaint mask and the source video alike -- and caches the "
                "encode on the target SHAPE, which every chunk of a render shares. "
                "Unwrapped, every chunk is driven by the first frames of the control "
                "video and chunk 0's encode is reused throughout. No error anywhere; "
                "the output just does not follow.\n\n"
                "This reads the chunk's first frame from the offset MMH3 Looping "
                "Sampler publishes, shifts the frame selection by it, and puts it in "
                "the cache key. Wired into a stock sampler the offset is 0 and nothing "
                "changes.\n\n"
                "Takes MODEL and MODEL_PATCH like core's node -- H3 Fun ControlNet is "
                "a model patch since 2026-08-31, not a ControlNet -- so this belongs "
                "in the model chain, not between the prompts and the sampler."
            ),
            inputs=[
                io.Model.Input("model"),
                io.ModelPatch.Input("model_patch"),
                io.Vae.Input("vae"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0,
                               step=0.001, optional=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0,
                               step=0.001, optional=True),
                io.Image.Input("control_video", optional=True),
                io.Mask.Input("mask", optional=True,
                              tooltip="1 marks the regions to regenerate."),
                io.Image.Input("source_video", optional=True,
                               tooltip="Video behind the mask; read only with a mask."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, model, model_patch, vae, strength, start_percent=0.0,
                end_percent=1.0, control_video=None, mask=None,
                source_video=None) -> io.NodeOutput:
        if strength == 0 or (control_video is None and mask is None):
            return io.NodeOutput(model, "strength 0 or no hint wired -- model passed "
                                        "through untouched")
        ok, why = _supported()
        if not ok:
            raise ValueError("MMH3ApplyControl: " + why)

        import comfy.patcher_extension
        base = _patch_class()
        chunk_aware = make_chunk_aware(base)

        model_patched = model.clone()
        model_sampling = model.get_model_object("model_sampling")
        patch = chunk_aware(
            model_patch, vae,
            control_video[..., :3].movedim(-1, 1) if control_video is not None else None,
            mask,
            source_video[..., :3].movedim(-1, 1)
            if mask is not None and source_video is not None else None,
            strength,
            float(model_sampling.percent_to_sigma(start_percent)),
            float(model_sampling.percent_to_sigma(end_percent)),
        )
        patch.register(model_patched)

        lines = ["MMH3 Apply ControlNet", ""]
        lines.append("  control video : %s"
                     % ("%d frames" % int(control_video.shape[0])
                        if control_video is not None else "none"))
        lines.append("  inpaint mask  : %s"
                     % ("%d frames" % int(mask.shape[0]) if mask is not None else "none"))
        lines.append("  strength %.2f over %.3f-%.3f"
                     % (strength, start_percent, end_percent))
        lines.append("")
        lines.append("  chunk-aware: each chunk shifts the hint frames by "
                     "transformer_options['%s'] and keys the encode cache on it."
                     % OFFSET_KEY)
        lines.append("  Through a stock sampler the offset is 0, which is the "
                     "whole-clip behaviour core already has.")
        logging.info("[MMH3ApplyControl] chunk-aware control patch registered")
        return io.NodeOutput(model_patched, "\n".join(lines))
