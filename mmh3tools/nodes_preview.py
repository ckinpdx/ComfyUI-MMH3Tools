"""A filmstrip of the chunks finished so far, on the preview node's OWN widget.

A chunked render is the case where a progress bar tells you least: it says step
34 of 160 and nothing about whether chunk 2 is looking at the right footage, or
whether the piece has drifted off its own opening. The information exists -- each
chunk's latent is in hand the moment it is written back -- and it was thrown away.

Three ideas here come from other packs, reimplemented rather than copied:

  1. **The preview is a WRAPPER the sampler discovers, not sampler code**
     (hradec's ComfyUI-HR-Endless-Sampler, Apache-2.0). Wiring this node registers
     a callable on the model patcher; the sampler asks for it and gets None when
     nobody wired one, so the feature costs nothing when it is off.
  2. **Frame spans follow FRAME_PER_TOKEN** (same source). Latents do not cover
     equal spans -- the first of every 17-frame group covers ONE frame and the rest
     four -- so "every Nth latent" is not evenly spaced in time.
  3. **The image is addressed to THIS node, not the executing one** (KJNodes'
     ModelPreviewOverrideKJ). Core's `UNENCODED_PREVIEW_IMAGE` channel always lands
     on whichever node is running, so two live previews cannot coexist -- the second
     overwrites the first. Taking the node's own `unique_id` and sending a custom
     event to a DOM widget is what makes the picture, the audio and an attention map
     separable. It costs a small front-end; there is no core route to "put this
     image on that node".

The decode is this pack's own choice: `latent_rgb_factors` off `MiniMaxH3Video`, a
24x3 linear projection. No model file, no VAE, no download, one matmul. It is an
APPROXIMATION -- colour is indicative, fine detail is not there at all. It answers
"is this the right shot, in the right order, moving the right way", which is what
the numbers cannot answer.
"""

import base64
import io as pyio
import logging
import threading

import torch

from comfy_api.latest import io

EVENT = "mmh3_live_preview"

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

# node_id -> the tiles that node last accumulated, so MMH3 Get Preview Frames can
# hand them back as a real IMAGE after the run. A live preview is transient by
# nature; the frames it drew are worth keeping.
_FRAMES = {}
_FRAMES_LOCK = threading.Lock()


def _suppress_default_previews():
    """Silence core's own previewer for the duration of a sample; returns a restore.

    Without this the sampler node keeps drawing its own latent preview alongside
    this node's, which is the thing that made the two look like duplicates. Every
    CONCRETE subclass is patched, not just the base: VHS wraps LatentPreviewer and
    would otherwise carry on emitting.
    """
    import latent_preview
    saved = []
    stack = [latent_preview.LatentPreviewer]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if "decode_latent_to_preview_image" in cls.__dict__:
            saved.append((cls, cls.__dict__["decode_latent_to_preview_image"]))
            cls.decode_latent_to_preview_image = lambda self, *a, **k: None

    def restore():
        for cls, method in saved:
            cls.decode_latent_to_preview_image = method
    return restore


class _PreviewRegistration:
    """The OUTER_SAMPLE wrapper: carries config, and hooks the per-step callback.

    A chunked render is watched STEP by step, not chunk by chunk -- finding out a
    chunk was wrong only once it is finished is finding out too late. Core hands
    this wrapper the sampler's `callback` as positional arg 5
    (`executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback,
    disable_pbar, seed, latent_shapes=...)`), and the callback is
    `(step, x0, x, total_steps)` -- so x0, the denoised prediction, is reachable on
    every step. Wrapping it is the only change; the original is always called.
    """

    def __init__(self, config):
        self.config = dict(config)
        self.session = None

    def __call__(self, executor, *args, **kwargs):
        session = self.session
        if session is None or not session.enabled:
            return executor(*args, **kwargs)

        args = list(args)
        original = args[5] if len(args) > 5 else None

        def callback(step, x0, x, total_steps):
            try:
                session.step(step, total_steps, x0)
            except Exception as error:                   # never break a render
                session.enabled = False
                logging.warning("[MMH3LivePreview] disabled after an error: %s", error)
            if original is not None:
                return original(step, x0, x, total_steps)

        if len(args) > 5:
            args[5] = callback
        restore = None
        try:
            if session.suppress_default:
                restore = _suppress_default_previews()
        except Exception as error:
            logging.info("[MMH3LivePreview] could not suppress the default preview: %s",
                         error)
        try:
            return executor(*args, **kwargs)
        finally:
            if restore is not None:
                restore()


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
        self.quality = int(config.get("quality", 80))
        self.node_id = config.get("node_id")
        self.vae = config.get("vae")
        self.suppress_default = bool(config.get("suppress_default", True))
        self.enabled = self.node_id is not None
        self.chunk_count = int(chunk_count)
        self.tiles = []
        self.labels = []
        # the chunk being sampled RIGHT NOW, and its live frame
        self.current = None
        self.live = None
        self.live_label = ""
        if self.node_id is not None:
            with _FRAMES_LOCK:
                _FRAMES.pop(str(self.node_id), None)

    def set_chunk(self, index, v0, v1):
        """Called before a chunk starts, so the live frame can name itself."""
        self.current = (int(index), int(v0), int(v1))

    def step(self, step, total_steps, x0):
        """Per SAMPLING STEP. x0 is the denoised prediction, which is the whole
        point: a chunk that has gone wrong is visible at step 2, not after it has
        been paid for in full."""
        if not self.enabled or x0 is None:
            return
        video = x0.unbind()[0] if getattr(x0, "is_nested", False) else x0
        rgb = self._render(video)
        if rgb is None:
            return
        self.live = rgb[rgb.shape[0] // 2]
        i, v0, v1 = self.current if self.current else (0, 0, 0)
        self.live_label = "chunk %d  step %d/%d" % (i, int(step) + 1, int(total_steps))
        self._send()

    def _render(self, video):
        """A [T,H,W,3] preview of a video latent, by VAE if one is wired."""
        if self.vae is not None:
            try:
                lat = video if video.ndim == 5 else video.unsqueeze(0)
                out = self.vae.decode(lat)
                if out.ndim == 5:
                    out = out[0]
                return out.detach().to(dtype=torch.float32, device="cpu").clamp(0, 1)
            except Exception as error:
                # One failure is enough; falling back keeps the preview alive rather
                # than turning a diagnostic into a second thing to debug.
                logging.info("[MMH3LivePreview] VAE decode failed (%s); falling back "
                             "to latent_rgb_factors for the rest of the run", error)
                self.vae = None
        return _to_rgb(video)

    def chunk(self, index, video_latent, v0, v1):
        """Called with a finished chunk's latent. Never raises into the sampler."""
        if not self.enabled:
            return
        try:
            rgb = self._render(video_latent)
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
            self.live = None
            self.live_label = ""
            self.labels.append("%d: %d-%d (%df)"
                               % (index, v0, v1, span_frames(v0, v1)))
            if len(self.tiles) > MAX_STRIP:
                self.tiles = self.tiles[-MAX_STRIP:]
                self.labels = self.labels[-MAX_STRIP:]
            with _FRAMES_LOCK:
                _FRAMES[str(self.node_id)] = list(self.tiles)
            self._send()
        except Exception as error:                       # never break a render
            self.enabled = False
            logging.warning("[MMH3LivePreview] disabled after an error: %s", error)

    def _send(self):
        from PIL import Image
        try:
            from server import PromptServer
        except ImportError:
            self.enabled = False
            return
        server = getattr(PromptServer, "instance", None)
        if server is None:
            self.enabled = False
            return
        # Finished chunks, then the one being sampled right now on the end.
        shown = list(self.tiles) + ([self.live] if self.live is not None else [])
        if not shown:
            return
        strip = _tile(shown, self.height)[0]                  # [3,H,W]
        arr = (strip.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        image = Image.fromarray(arr)
        buf = pyio.BytesIO()
        image.save(buf, format="JPEG", quality=self.quality)
        server.send_sync(EVENT, {
            "node_id": self.node_id,
            "image": base64.b64encode(buf.getvalue()).decode("ascii"),
            "mime": "image/jpeg",
            "w": image.width,
            "h": image.height,
            "chunks": len(self.tiles),
            "total": self.chunk_count,
            "labels": list(self.labels),
            "live": self.live_label,
        }, server.client_id)


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
            # The wrapper reads the session off the registration, because the
            # per-step callback lives in the wrapper while the chunk loop lives in
            # the sampler. One object, both ends.
            entry.session = PreviewSession(entry.config, chunk_count)
            return entry.session
    return None


class MMH3LivePreview(io.ComfyNode):
    """Show the chunks as they finish, on this node."""

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
                "The image appears on THIS node, not on the sampler. Core's preview "
                "channel always addresses the executing node, so two live previews "
                "would overwrite each other; addressing this node's own id keeps them "
                "separable.\n\n"
                "One tile per finished chunk, taken from the MIDDLE of the chunk -- a "
                "chunk's opening latents are the carry from the one before it, so a "
                "strip of first frames would mostly show the previous chunk.\n\n"
                "The decode is `latent_rgb_factors`, a 24x3 projection of the latent. "
                "No model file, no VAE, one matmul -- and an APPROXIMATION: colour is "
                "indicative, fine detail is absent. It answers whether the shots are "
                "the right ones in the right order, which is what a step counter "
                "cannot.\n\n"
                "Any error switches the preview off for the run rather than "
                "interrupting it. Use MMH3 Get Preview Frames to keep what it drew."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "tile_height", default=96, min=32, max=512, step=8,
                    tooltip="Height of each tile in pixels; width follows the aspect. "
                            "The strip holds the last 16 chunks, so a tall tile on a "
                            "long render makes a very wide image."),
                io.Int.Input(
                    "jpeg_quality", default=80, min=30, max=100, step=1, optional=True,
                    tooltip="Quality of the preview transport only. Nothing here "
                            "reaches the render."),
                io.Boolean.Input(
                    "suppress_sampler_preview", default=True, optional=True,
                    tooltip="Silence ComfyUI's own latent preview on the sampler node "
                            "while this runs, so the two do not draw the same thing "
                            "twice. Restored when the sample ends."),
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip="Optional true-colour decode. Wire a stock VAE Loader "
                            "at a TINY H3 decoder -- `taeh3.safetensors` in "
                            "`models/vae` -- and NOT the full VAE: this runs on "
                            "every sampling step.\n\n"
                            "Unwired, the preview uses `latent_rgb_factors`, a 24x3 "
                            "projection that costs a matmul and is approximate. A "
                            "decode that fails once falls back to that for the rest of "
                            "the run rather than failing the render."),
            ],
            outputs=[io.Model.Output(display_name="model")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, tile_height=96, jpeg_quality=80,
                suppress_sampler_preview=True, vae=None) -> io.NodeOutput:
        import comfy.latent_formats
        if not getattr(comfy.latent_formats.MiniMaxH3Video, "latent_rgb_factors", None):
            raise ValueError(
                "MMH3LivePreview: this ComfyUI's MiniMaxH3Video latent format has no "
                "`latent_rgb_factors`, so there is nothing to project the latent "
                "through. Update ComfyUI, or remove this node.")
        node_id = cls.hidden.unique_id
        patched = model.clone()
        patched.add_wrapper_with_key(
            _wrappers_slot(), PREVIEW_KEY,
            _PreviewRegistration({
                "height": int(tile_height),
                "quality": int(jpeg_quality),
                "suppress_default": bool(suppress_sampler_preview),
                "vae": vae,
                "node_id": None if node_id is None else str(node_id)}))
        logging.info("[MMH3LivePreview] registered on node %s; tile height %d",
                     node_id, int(tile_height))
        return io.NodeOutput(patched)


class MMH3GetPreviewFrames(io.ComfyNode):
    """Keep what the live preview drew."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3GetPreviewFrames",
            display_name="MMH3 Get Preview Frames",
            category="MMH3Tools/sampling",
            description=(
                "The tiles MMH3 Live Preview accumulated on its last run, as a real "
                "IMAGE batch -- one frame per chunk, in order. The live strip is "
                "transient; this is how it survives the run.\n\n"
                "Takes the preview's MODEL output so it runs after sampling rather "
                "than beside it. The model is passed straight through and is not "
                "read.\n\n"
                "Tiles are the same `latent_rgb_factors` approximation the preview "
                "shows, at their native latent resolution -- 1/16 of the render per "
                "side. Not a substitute for decoding."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The MMH3 Live Preview node's model output. Only used to "
                            "order this node after the sampler."),
            ],
            outputs=[
                io.Image.Output(display_name="frames"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        with _FRAMES_LOCK:
            # A single preview node is the ordinary case; with several, the most
            # recently written wins rather than guessing which one was meant.
            tiles = []
            for key in reversed(list(_FRAMES)):
                if _FRAMES[key]:
                    tiles = list(_FRAMES[key])
                    break
        if not tiles:
            raise ValueError(
                "MMH3GetPreviewFrames: nothing recorded. MMH3 Live Preview has not "
                "run in this process, or the run it made was cached and never "
                "sampled.")
        h = max(int(t.shape[0]) for t in tiles)
        w = max(int(t.shape[1]) for t in tiles)
        import comfy.utils
        out = []
        for t in tiles:
            img = t.permute(2, 0, 1).unsqueeze(0)
            if img.shape[2] != h or img.shape[3] != w:
                img = comfy.utils.common_upscale(img, w, h, "bilinear", "disabled")
            out.append(img.squeeze(0).permute(1, 2, 0))
        return io.NodeOutput(torch.stack(out, dim=0), len(out))
