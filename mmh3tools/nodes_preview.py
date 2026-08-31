"""A filmstrip of the chunks finished so far, pushed while the sampler runs.

A chunked render is the case where a progress bar tells you least: it says step
34 of 160 and nothing about whether chunk 2 is looking at the right footage, or
whether the piece has drifted off its own opening. The information exists -- each
chunk's latent is in hand the moment it is written back -- and it is thrown away.

Two ideas here are taken from hradec's ComfyUI-HR-Endless-Sampler (Apache-2.0),
reimplemented rather than copied:

  1. **The preview is a WRAPPER the sampler discovers, not sampler code.** Wiring
     `MMH3 Live Preview` between the model and the guider registers a config on the
     model patcher; the sampler asks for it and gets None when nobody wired one.
     The sampler therefore carries a nullable object instead of preview logic, and
     the feature costs nothing when it is off.
  2. **Frame timing follows FRAME_PER_TOKEN.** Latents do not cover equal spans --
     the first of every 17-frame group covers ONE frame and the rest cover four --
     so picking "every Nth latent" and calling it even is wrong. Spans are computed
     from the real grid.

The decode is this pack's own choice and simpler than theirs: `latent_rgb_factors`
off `MiniMaxH3Video`, a 24x3 linear projection. It needs no model file, no VAE and
no download, and it costs a matmul. It is an APPROXIMATION -- colour is indicative,
fine detail is not there at all. It is for answering "is this the right shot, in
the right order, moving the right way", which is what the numbers cannot answer.

Transport is core's own `UNENCODED_PREVIEW_IMAGE` binary channel -- the one the
built-in latent previewer uses -- so the image appears on the executing node with
no custom front-end at all.
"""

import logging

import torch

from comfy_api.latest import io

from .common import FPS

# comfy.model_patcher wrapper slot, used as a registry keyed on the patcher.
#
# Core CALLS whatever sits in this slot -- WrapperExecutor.execute does
# `self.wrappers[self.idx](self, *args, **kwargs)` -- so the value cannot be a bare
# config dict. Registering one raised `'dict' object is not callable` from inside
# sampling, after the first chunk had already been queued. It is a pass-through
# callable that carries the config as an attribute instead.
PREVIEW_KEY = "mmh3_live_preview"

# The first latent of each group covers one frame, the rest cover four.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

MAX_STRIP = 16          # past this the strip is unreadable at any sane width
TILE_MIN = 64


class _PreviewRegistration:
    """A pass-through OUTER_SAMPLE wrapper whose only job is to carry config.

    Forwards to the next executor verbatim, so registering one cannot change a
    render -- which is the whole premise of discovering the preview rather than
    building it into the sampler.
    """

    def __init__(self, config):
        self.config = dict(config)

    def __call__(self, executor, *args, **kwargs):
        return executor(*args, **kwargs)


def _wrappers_slot():
    import comfy.patcher_extension
    return comfy.patcher_extension.WrappersMP.OUTER_SAMPLE


def frames_for_latent(index):
    """How many pixel frames the latent at `index` stands for."""
    return FRAME_PER_TOKEN[int(index) % len(FRAME_PER_TOKEN)]


def span_frames(start, stop):
    """Pixel frames covered by latents [start, stop), on the real grid.

    Not `(stop - start) * 4`. A window that begins on a group boundary carries a
    single-frame latent, so the flat multiplication overstates it by three frames
    per group -- which is exactly the arithmetic that puts a preview's timing out
    of step with the clip it is previewing.
    """
    return sum(frames_for_latent(i) for i in range(int(start), int(stop)))


def _to_rgb(latent):
    """[C,T,H,W] video latent -> [T,H,W,3] in 0..1, by the format's own factors."""
    import comfy.latent_formats
    fmt = comfy.latent_formats.MiniMaxH3Video
    factors = getattr(fmt, "latent_rgb_factors", None)
    if not factors:
        return None
    x = latent.detach().to(dtype=torch.float32, device="cpu")
    if x.ndim == 5:
        x = x[0]
    c = x.shape[0]
    w = torch.tensor(factors, dtype=torch.float32)          # [C, 3]
    if w.shape[0] != c:
        # A latent whose channel count does not match the format is not something
        # to guess at -- a wrong projection produces a confident, meaningless image.
        return None
    bias = getattr(fmt, "latent_rgb_factors_bias", None)
    rgb = torch.einsum("cthw,cr->thwr", x, w)
    if bias:
        rgb = rgb + torch.tensor(bias, dtype=torch.float32)
    return rgb.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)


def _tile(rgb_frames, height):
    """Frames -> one row, each scaled to `height`, concatenated on width."""
    import comfy.utils
    out = []
    for f in rgb_frames:
        img = f.permute(2, 0, 1).unsqueeze(0)               # [1,3,H,W]
        h, w = img.shape[2], img.shape[3]
        tw = max(TILE_MIN, int(round(w * (height / float(h)))))
        out.append(comfy.utils.common_upscale(img, tw, height, "bilinear", "disabled"))
    return torch.cat(out, dim=3)


class PreviewSession:
    """Accumulates one frame per finished chunk and pushes the strip."""

    def __init__(self, config, chunk_count):
        self.height = int(config.get("height", 96))
        self.enabled = True
        self.chunk_count = int(chunk_count)
        self.tiles = []
        self.labels = []

    def chunk(self, index, video_latent, v0, v1):
        """Called with a finished chunk's latent. Never raises into the sampler."""
        if not self.enabled:
            return
        try:
            rgb = _to_rgb(video_latent)
            if rgb is None:
                self.enabled = False
                logging.info("[MMH3LivePreview] no usable latent_rgb_factors for this "
                             "latent; preview off for this run")
                return
            # The MIDDLE of the chunk, not its first frame: chunk i's opening
            # latents are the carry from chunk i-1, so a strip of first frames is
            # largely a strip of the previous chunk.
            mid = rgb.shape[0] // 2
            self.tiles.append(rgb[mid])
            self.labels.append("%d:%d-%d" % (index, v0, v1))
            if len(self.tiles) > MAX_STRIP:
                self.tiles = self.tiles[-MAX_STRIP:]
                self.labels = self.labels[-MAX_STRIP:]
            self._send()
        except Exception as error:                       # never break a render
            self.enabled = False
            logging.warning("[MMH3LivePreview] disabled after an error: %s", error)

    def _send(self):
        from PIL import Image
        try:
            from server import PromptServer
            from protocol import BinaryEventTypes
        except ImportError:
            self.enabled = False
            return
        server = getattr(PromptServer, "instance", None)
        if server is None:
            self.enabled = False
            return
        strip = _tile(self.tiles, self.height)[0]            # [3,H,W]
        arr = (strip.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        image = Image.fromarray(arr)
        server.send_sync(
            BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
            ("JPEG", image, None),
            server.client_id,
        )
        logging.debug("[MMH3LivePreview] strip of %d chunk(s): %s",
                      len(self.tiles), " ".join(self.labels))


def begin_preview(model_patcher, chunk_count):
    """A session if something wired MMH3 Live Preview, else None.

    None is the ordinary case and the sampler must stay identical under it -- the
    whole point of discovering the preview rather than owning it.
    """
    try:
        wrappers = model_patcher.get_wrappers(_wrappers_slot(), PREVIEW_KEY)
    except Exception:
        return None
    for entry in wrappers or ():
        if isinstance(entry, _PreviewRegistration):
            return PreviewSession(entry.config, chunk_count)
    return None


class MMH3LivePreview(io.ComfyNode):
    """Show the chunks as they finish."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LivePreview",
            display_name="MMH3 Live Preview",
            category="MMH3Tools/sampling",
            description=(
                "Push a filmstrip of the chunks finished so far while MMH3 Looping "
                "Sampler runs. Wire it between the model and whatever builds the "
                "guider; it passes the model through unchanged and only registers "
                "itself, so the sampler behaves identically when it is absent.\n\n"
                "One tile per finished chunk, taken from the MIDDLE of the chunk -- a "
                "chunk's opening latents are the carry from the one before it, so a "
                "strip of first frames would mostly show the previous chunk.\n\n"
                "The decode is `latent_rgb_factors`, a 24x3 projection of the latent. "
                "No model file, no VAE, one matmul -- and an APPROXIMATION: colour is "
                "indicative, fine detail is absent. It answers whether the shots are "
                "the right ones in the right order, which is what a step counter "
                "cannot.\n\n"
                "The image arrives on core's own preview channel, the one the built-in "
                "latent previewer uses. Any error switches the preview off for the run "
                "rather than interrupting it."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "tile_height", default=96, min=32, max=512, step=8,
                    tooltip="Height of each tile in pixels; width follows the aspect. "
                            "The strip holds the last 16 chunks, so a tall tile on a "
                            "long render makes a very wide image."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, tile_height=96) -> io.NodeOutput:
        import comfy.latent_formats
        if not getattr(comfy.latent_formats.MiniMaxH3Video, "latent_rgb_factors", None):
            raise ValueError(
                "MMH3LivePreview: this ComfyUI's MiniMaxH3Video latent format has no "
                "`latent_rgb_factors`, so there is nothing to project the latent "
                "through. Update ComfyUI, or remove this node.")
        patched = model.clone()
        patched.add_wrapper_with_key(
            _wrappers_slot(), PREVIEW_KEY,
            _PreviewRegistration({"height": int(tile_height)}))
        logging.info("[MMH3LivePreview] registered; tile height %d", int(tile_height))
        return io.NodeOutput(patched)
