"""Apply MiniMaxH3Mod bundles to each chunk's conditioning, keeping controls last."""

from comfy_api.latest import io

from .nodes_multiprompt import MMH3CondSet

H3RefMods = io.Custom("H3_REF_MODS")
_CONTROL_KINDS = ("video", "video_audio", "audio")


def _trailing_controls(refs):
    """Number of trailing non-image blocks (control videos + their audio)."""
    n = 0
    for b in reversed(refs):
        if isinstance(b, dict) and b.get("kind") in _CONTROL_KINDS:
            n += 1
        else:
            break
    return n


class H3RefModCondSetApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModCondSetApply",
            display_name="Apply H3 RefMods to Cond Set",
            category="MMH3Tools/conditioning",
            description=("Apply a MiniMaxH3Mod loader bundle to every chunk's conditioning. "
                         "By default, mods go before trailing control blocks so negative "
                         "control indexes remain valid. Empty/unwired mods are a passthrough. "
                         "Mod references have no <Picture N> tag; prompt by description."),
            inputs=[
                MMH3CondSet.Input("cond_set"),
                io.Float.Input("retention", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Master strength multiplied by each loader row's strength. "
                                       "0 injects nothing; 1 preserves the loader strengths."),
                io.Combo.Input("insert_position", options=["before_controls", "append_last"],
                               default="before_controls",
                               tooltip="Insert before trailing video/audio references to keep control "
                                       "schedules' negative indexes unchanged. append_last adds mods "
                                       "after controls; adjust those indexes if using it."),
                io.Int.Input("controls_override", default=-1, min=-1, max=32,
                             tooltip="-1 detects trailing video/video_audio/audio blocks per entry. "
                                     "An explicit count overrides detection, for example after "
                                     "another Apply node has inserted a video-kind mod."),
                H3RefMods.Input("mods", optional=True,
                                tooltip="Load H3 RefMods / Extract H3 RefMod bundle from "
                                        "ComfyUI-MiniMaxH3Mod. Empty or unwired is a passthrough."),
            ],
            outputs=[MMH3CondSet.Output(display_name="cond_set"),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, cond_set, retention, insert_position, controls_override, mods=None):
        return io.NodeOutput(*cls().apply(
            cond_set, retention, insert_position, controls_override, mods))

    def apply(self, cond_set, retention, insert_position, controls_override, mods=None):
        conds = (cond_set or {}).get("conds") or []
        if not mods or retention <= 0.0:
            why = "no mods wired/selected" if not mods else "retention 0"
            return (cond_set, "[H3RefModCondSetApply] passthrough (%s), %d cond(s) untouched"
                    % (why, len(conds)))

        blocks, lines = [], []
        for row in mods:
            try:
                mod, strength = row
            except (TypeError, ValueError):
                raise ValueError(
                    "H3RefModCondSetApply: 'mods' is not a RefMod loader bundle — wire the "
                    "mods output of Load H3 RefMods / Extract H3 RefMod (ComfyUI-MiniMaxH3Mod).")
            if not hasattr(mod, "ref_block"):
                raise ValueError(
                    "H3RefModCondSetApply: bundle entry %r has no ref_block() — "
                    "ComfyUI-MiniMaxH3Mod version mismatch?" % (getattr(mod, "name", mod),))
            eff = min(1.0, max(0.0, float(strength) * float(retention)))
            block = mod.ref_block(eff)  # None when eff <= 0
            if block is not None:
                blocks.append(block)
                lines.append("  %s @ %.2f (%s, %d tokens)"
                             % (getattr(mod, "name", "?"), eff,
                                getattr(mod, "mode", "?"),
                                getattr(mod, "token_count", 0)))

        if not blocks:
            return (cond_set, "[H3RefModCondSetApply] passthrough (all rows at strength 0), "
                    "%d cond(s) untouched" % len(conds))

        new_conds, insert_info = [], []
        for c in conds:
            out = []
            for t in c:
                d = dict(t[1])
                refs = list(d.get("minimax_refs", []))
                if insert_position == "before_controls":
                    k = _trailing_controls(refs) if controls_override < 0 else \
                        min(int(controls_override), len(refs))
                else:
                    k = 0
                idx = len(refs) - k
                d["minimax_refs"] = refs[:idx] + blocks + refs[idx:]
                insert_info.append((idx, k))
                out.append([t[0], d])
            new_conds.append(out)

        idxs = sorted({i for i, _ in insert_info})
        ctrls = sorted({k for _, k in insert_info})
        report = "\n".join(
            ["[H3RefModCondSetApply] %d mod block(s) into %d cond(s):" % (len(blocks), len(new_conds))]
            + lines
            + ["  inserted at ref index %s, %s trailing control block(s) kept last"
               % ("/".join(map(str, idxs)), "/".join(map(str, ctrls))),
               "  note: mod refs carry no <Picture N> tag — prompt by description"])
        print(report)
        return ({"conds": new_conds,
                 "prompts": (cond_set or {}).get("prompts"),
                 "fingerprint": (cond_set or {}).get("fingerprint")},
                report)
