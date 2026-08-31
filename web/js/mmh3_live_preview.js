import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// The filmstrip is addressed to the MMH3 Live Preview node's own id rather than to
// whichever node is executing. Core's UNENCODED_PREVIEW_IMAGE channel always lands
// on the executing node, so two live previews overwrite each other; this is the only
// route to "put this image on that node", and it is why this file exists at all.
//
// Payload (see PreviewSession._send in mmh3tools/nodes_preview.py):
//   { node_id, image: <base64>, mime, w, h, chunks, total, labels: [...] }

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

function ensureWidget(node) {
    if (node._mmh3PreviewRoot) return node._mmh3PreviewRoot;

    const root = document.createElement("div");
    root.style.cssText =
        "display:flex;flex-direction:column;gap:2px;width:100%;height:100%;" +
        "box-sizing:border-box;padding:2px;overflow:hidden;";

    const img = document.createElement("img");
    // contain, not cover: a filmstrip cropped to fill would hide the chunks at the
    // ends, which are the ones worth looking at.
    img.style.cssText =
        "width:100%;flex:1 1 auto;min-height:0;object-fit:contain;" +
        "image-rendering:auto;background:#181818;border-radius:2px;";
    root.appendChild(img);

    const caption = document.createElement("div");
    caption.style.cssText =
        "flex:0 0 auto;font:10px ui-monospace,Consolas,monospace;color:#9a9a9a;" +
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    caption.textContent = "waiting for the first chunk";
    root.appendChild(caption);

    // serialize:false -- this is a view, not state. Saving it would put a base64
    // JPEG into every workflow file.
    node.addDOMWidget(WIDGET_NAME, "mmh3_preview", root, { serialize: false });
    node._mmh3PreviewRoot = root;
    node._mmh3PreviewImg = img;
    node._mmh3PreviewCaption = caption;

    if (node.size[0] < 320) node.size[0] = 320;
    if (node.size[1] < 220) node.size[1] = 220;
    return root;
}

function draw(node, data) {
    ensureWidget(node);
    const img = node._mmh3PreviewImg;
    if (data.image) {
        // A data: URL avoids the object-URL lifetime problem entirely -- there is
        // nothing to revoke, and a dropped frame cannot leak.
        img.src = `data:${data.mime || "image/jpeg"};base64,${data.image}`;
    }
    // While a chunk is sampling its step counter is the useful number; between
    // chunks, what was banked is.
    const total = data.total ? ` / ${data.total}` : "";
    const done = `${data.chunks}${total} done`;
    node._mmh3PreviewCaption.textContent = data.live
        ? `${data.live}   (${done})`
        : (data.labels?.length ? `${done}   last ${data.labels[data.labels.length - 1]}` : done);
    node.setDirtyCanvas(true, false);
}

api.addEventListener("mmh3_live_preview", (event) => {
    const data = event.detail;
    if (!data) return;
    const node = findNode(data.node_id);
    if (node) draw(node, data);
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
    },
});
