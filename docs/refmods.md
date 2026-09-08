# Apply H3 RefMods to Cond Set

Find **Apply H3 RefMods to Cond Set** under **MMH3Tools / conditioning**. It takes
an `MMH3_COND_SET` and an optional `H3_REF_MODS` bundle from **Load H3 RefMods** or
**Extract H3 RefMod** in ComfyUI-MiniMaxH3Mod. Connect its output to the sampler or
conditioning selector that previously received the original set.

The adapter applies each loader row's reference block to every conditioning
entry in every chunk. It does not load files, encode images, or require the VLM
pack. ComfyUI-MiniMaxH3Mod supplies the loader/extractor and the mod objects.

- **retention** multiplies each loader row's strength. The effective strength is
  limited to 0–1 and passed to the mod's existing `ref_block()` method.
- **before_controls**, the default insertion position, places mods before trailing
  `video`, `video_audio`, and `audio` reference blocks. Negative control indexes
  therefore continue to address the same controls.
- **append_last** adds mods after all existing references. Control schedules that
  use negative indexes may need adjusting in this mode.
- **controls_override = -1** detects the trailing blocks separately for each
  conditioning entry. An explicit count handles ambiguous layouts, such as a
  video-kind mod inserted by an earlier Apply node.

An empty/unwired bundle or zero retention returns the original set unchanged.
Text tensors, prompts, and existing reference blocks are preserved. Mod references
do not acquire `<Picture N>` tags; describe them in the prompt.

## Moving from vlm_video_prompt

The node keeps its original ID, `H3RefModCondSetApply`, input/output names, widget
order, defaults, and application behavior. Existing workflow links remain valid.
The direct Python `apply(...)` method is also retained.

Remove the old `H3RefModCondSetApply` registration from the VLM pack when installing
this version, so two packs do not register the same ID. Its separate
`H3RefPictureStrength` node remains in the VLM pack. Restart ComfyUI after moving
the registration; workflow nodes do not need to be removed or recreated.

`tests/test_refmods.py` covers workflow schema compatibility, passthrough,
multi-entry insertion, control ordering, strength delegation, and input
immutability. Run it in a ComfyUI Python environment, with `PYTHONPATH` pointing
to ComfyUI if this repository is checked out elsewhere.
