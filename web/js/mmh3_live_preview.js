import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// The timeline is addressed to the MMH3 Timeline Preview node's own id rather than to
// whichever node is executing. Core's UNENCODED_PREVIEW_IMAGE channel always lands
// on the executing node, so two live previews overwrite each other; this is the only
// route to "put this picture on that node", and it is why this file exists at all.
//
// Payload (see PreviewSession._send / _encode_and_send in mmh3tools/nodes_preview.py):
//   { node_id, seq, mime, w, h,
//     sprite: { cols, rows, tw, th, count }, fps, stride,
//     audio: bool, sr,
//     chunks, total, frames, seconds, labels: [...], live }
//
// The event is metadata only. The picture is a SPRITE SHEET of the kept frames,
// fetched from /mmh3/preview?node_id=<id>&kind=image; the sound, when an audio VAE
// is wired, is a WAV from the same route with kind=audio, under the same `seq`.
// Nothing rides the websocket, where core's publish loop would hold every client's
// progress events behind a large frame.
//
// A sheet rather than an animated file because an animation cannot be seeked,
// paused or kept in step with a sound. The canvas draws frame
// floor(t * fps / stride) at whatever `t` the <audio> element reports -- or a
// timer, when there is no audio -- so picture and sound cannot drift apart.
// Reassigning a src aborts the load in flight: latest-wins, a slow tab falls at
// most one chunk behind and never builds a queue.

const WIDGET_NAME = "mmh3_preview";

// A node inside a subgraph reports a qualified id like "12:4:7"; the leaf is the one
// the graph can resolve. Matches how the payload's node_id is produced.
function findNode(id) {
    if (id === null || id === undefined) return null;
    const parts = String(id).split(":");
    let graph = app.graph;
    for (let i = 0; i < parts.length - 1; i++) {
        const outer = graph?.getNodeById?.(parseInt(parts[i], 10));
        if (!outer?.subgraph) return null;
        graph = outer.subgraph;
    }
    const leaf = parseInt(parts[parts.length - 1], 10);
    return Number.isFinite(leaf) ? graph?.getNodeById?.(leaf) || null : null;
}

function mediaURL(nodeId, kind, seq) {
    // api.apiURL respects a server subpath. `seq` is the cache-buster.
    return api.apiURL(
        `/mmh3/preview?node_id=${encodeURIComponent(nodeId)}&kind=${kind}&seq=${seq}`);
}

const BTN = "flex:0 0 auto;font:11px ui-monospace,Consolas,monospace;color:#ddd;" +
    "background:#2a2a2a;border:1px solid #444;border-radius:3px;padding:0 6px;" +
    "height:18px;line-height:16px;cursor:pointer;";

function ensureWidget(node) {
    if (node._mmh3) return node._mmh3;

    const root = document.createElement("div");
    root.style.cssText =
        "display:flex;flex-direction:column;gap:2px;width:100%;height:100%;" +
        "box-sizing:border-box;padding:2px;overflow:hidden;";

    const canvas = document.createElement("canvas");
    canvas.style.cssText =
        "width:100%;flex:1 1 auto;min-height:0;background:#181818;border-radius:2px;";
    root.appendChild(canvas);

    const bar = document.createElement("div");
    bar.style.cssText = "display:flex;flex:0 0 auto;gap:4px;align-items:center;";
    const play = document.createElement("button");
    play.style.cssText = BTN;
    play.textContent = "▶";
    const scrub = document.createElement("input");
    scrub.type = "range";
    scrub.min = 0; scrub.max = 1000; scrub.value = 0;
    scrub.style.cssText = "flex:1 1 auto;min-width:0;margin:0;";
    const mute = document.createElement("button");
    mute.style.cssText = BTN;
    mute.textContent = "🔊";
    mute.hidden = true;
    bar.appendChild(play); bar.appendChild(scrub); bar.appendChild(mute);
    root.appendChild(bar);

    const caption = document.createElement("div");
    caption.style.cssText =
        "flex:0 0 auto;font:10px ui-monospace,Consolas,monospace;color:#9a9a9a;" +
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    caption.textContent = "waiting for the first chunk";
    root.appendChild(caption);

    // An <audio> element rather than WebAudio: it is the clock, it handles
    // buffering, and swapping its src mid-play keeps currentTime.
    const audio = new Audio();
    audio.loop = true;
    audio.preload = "auto";

    const st = {
        root, canvas, ctx: canvas.getContext("2d"), play, scrub, mute, caption, audio,
        img: null, sprite: null, fps: 24, stride: 3, seq: -1, hasAudio: false,
        playing: false, t0: 0, offset: 0, raf: 0,
    };

    play.onclick = () => (st.playing ? pause(st) : start(st));
    mute.onclick = () => { audio.muted = !audio.muted; mute.textContent = audio.muted ? "🔇" : "🔊"; };
    scrub.oninput = () => seek(st, (scrub.value / 1000) * duration(st));
    // A scrub while stopped should still show the frame it landed on.
    scrub.onchange = () => render(st);

    // serialize:false -- this is a view, not state, and it holds URLs to frames
    // that only exist while the server that drew them is up.
    node.addDOMWidget(WIDGET_NAME, "mmh3_preview", root, { serialize: false });
    node._mmh3 = st;

    if (node.size[0] < 320) node.size[0] = 320;
    if (node.size[1] < 260) node.size[1] = 260;
    return st;
}

function duration(st) {
    return st.sprite ? (st.sprite.count * st.stride) / st.fps : 0;
}

// The clock. The audio element when there is sound, a wall timer otherwise.
function now(st) {
    if (st.hasAudio && st.audio.readyState >= 1) return st.audio.currentTime;
    const d = duration(st);
    if (!d) return 0;
    const t = st.playing ? st.offset + (performance.now() - st.t0) / 1000 : st.offset;
    return t % d;
}

function seek(st, t) {
    if (st.hasAudio) {
        st.audio.currentTime = t;
    } else {
        st.offset = t;
        st.t0 = performance.now();
    }
    render(st);
}

function start(st) {
    if (!st.sprite || st.sprite.count < 2) return;
    st.playing = true;
    st.play.textContent = "⏸";
    if (st.hasAudio) {
        st.audio.play().catch(() => {
            // Autoplay refused: the browser wants a click first. Leave the button
            // showing ▶ so the click it wants is the obvious one.
            st.playing = false;
            st.play.textContent = "▶";
        });
    } else {
        st.t0 = performance.now();
    }
    if (!st.raf) tick(st);
}

function pause(st) {
    if (st.hasAudio) st.audio.pause();
    else st.offset = now(st);
    st.playing = false;
    st.play.textContent = "▶";
}

function tick(st) {
    st.raf = 0;
    render(st);
    if (st.playing) st.raf = requestAnimationFrame(() => tick(st));
}

function render(st) {
    const { canvas, ctx, img, sprite } = st;
    if (!img || !sprite) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const ch = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const t = now(st);
    const idx = sprite.count > 1
        ? Math.min(sprite.count - 1, Math.floor((t * st.fps) / st.stride)) : 0;
    const sx = (idx % sprite.cols) * sprite.tw;
    const sy = Math.floor(idx / sprite.cols) * sprite.th;
    // contain, not cover: a frame cropped to fill hides the edges, which on a
    // reference-anchored render are the part worth looking at.
    const s = Math.min(cw / sprite.tw, ch / sprite.th);
    const dw = sprite.tw * s, dh = sprite.th * s;
    ctx.fillStyle = "#181818";
    ctx.fillRect(0, 0, cw, ch);
    ctx.drawImage(img, sx, sy, sprite.tw, sprite.th, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
    const d = duration(st);
    if (d) st.scrub.value = Math.round((t / d) * 1000);
}

function apply(node, data) {
    const st = ensureWidget(node);
    if (data.seq === undefined || data.seq < st.seq) return;   // stale, already passed
    const firstTimeline = !st.sprite || st.sprite.count < 2;

    // Picture: load off-DOM, swap in on arrival, ignore if something newer landed.
    const img = new Image();
    img.onload = () => {
        if (data.seq < st.seq) return;
        st.seq = data.seq;
        st.img = img;
        st.sprite = data.sprite;
        st.fps = data.fps || 24;
        st.stride = data.stride || 1;
        render(st);
        // The timeline exists now: play it, unasked. Sound may be refused until a
        // click; a timer never is.
        if (firstTimeline && st.sprite.count > 1 && !st.playing) start(st);
    };
    img.src = mediaURL(data.node_id, "image", data.seq);

    // Sound: same seq, same route. Keep the position across the swap so a
    // playing timeline just gets longer.
    const hadAudio = st.hasAudio;
    st.hasAudio = !!data.audio;
    st.mute.hidden = !st.hasAudio;
    if (st.hasAudio) {
        const t = hadAudio ? st.audio.currentTime : now(st);
        const wasPlaying = st.playing;
        // the timer stands in while the new file loads; keep it at the same place
        st.offset = t; st.t0 = performance.now();
        st.audio.src = mediaURL(data.node_id, "audio", data.seq);
        st.audio.load();
        st.audio.addEventListener("loadedmetadata", () => {
            st.audio.currentTime = Math.min(t, Math.max(0, st.audio.duration - 0.05));
            if (wasPlaying) st.audio.play().catch(() => pause(st));
        }, { once: true });
    } else if (hadAudio) {
        // Sound went away mid-run (audio decode failed): carry on with the timer
        // from where the sound was.
        st.offset = st.audio.currentTime;
        st.t0 = performance.now();
        st.audio.pause();
        st.audio.removeAttribute("src");
    }

    // While a chunk is sampling its step counter is the useful number; between
    // chunks, what was banked is. Seconds is the number worth showing: the
    // timeline plays at real time, so it is how much of the piece exists so far.
    const total = data.total ? ` / ${data.total}` : "";
    const done = `${data.chunks}${total} chunks`;
    const secs = data.seconds ? `  ${data.seconds}s` : "";
    const snd = data.audio ? "  ♪" : "";
    st.caption.textContent = data.live
        ? `${data.live}   (${done})`
        : `${done}${secs}${data.frames ? `  ${data.frames}f` : ""}${snd}`;
    node.setDirtyCanvas(true, false);
}

api.addEventListener("mmh3_live_preview", (event) => {
    const data = event.detail;
    if (!data) return;
    const node = findNode(data.node_id);
    if (node) apply(node, data);
});

app.registerExtension({
    name: "mmh3tools.live_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "MMH3LivePreview") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created?.apply(this, arguments);
            ensureWidget(this);
            return r;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            const st = this._mmh3;
            if (st) { st.playing = false; st.audio.pause(); st.audio.removeAttribute("src"); }
            return removed?.apply(this, arguments);
        };
    },
});
