# Core changes MMH3Tools relies on

Two different things, and the difference decides which branch a node lives on.

## Upstream PRs — all merged, nothing to apply

The pack carried two of these for a while. Both are in core now, so this section is
history: it records what they do, because the nodes still depend on the behaviour and
anyone on an older ComfyUI needs to know why a node refuses.

| PR | merged | what it gives the pack |
|---|---|---|
| **[#15375](https://github.com/Comfy-Org/ComfyUI/pull/15375)** drozbay | 2026-08-18 | Per-token masking, three parts: the mask reaches the model as a cond, preserved rows run at the cond timestep, and `MiniMaxH3` gets a `scale_latent_inpaint` override. Without it a hard mask has **no effect at all** — preserved rows still run at the generation timestep, so the model gets clean content labelled as noisy — and an *intermediate* mask value artifacts, since stock falls back to `BaseModel`'s noise blend. |
| **[#15439](https://github.com/Comfy-Org/ComfyUI/pull/15439)** drozbay | 2026-08-13 | `MiniMaxH3AddGuide`: guides at any frame index, audio anchored at the same `cond_t`. |

Minimum ComfyUI is therefore **`v0.33.0-20-gff6c8a8a`**. The fetch-and-apply recipe
below is kept only for reviving an old build; on a current one it applies nothing.

```bash
cd C:/ComfyUI
for pr in 15375; do   # only on a ComfyUI predating the merge
  curl -sL "https://github.com/Comfy-Org/ComfyUI/pull/$pr.diff" -o /tmp/pr$pr.diff
  git apply --check /tmp/pr$pr.diff && git apply /tmp/pr$pr.diff
done
```

**Re-fetch rather than reusing a saved copy.** These get rebased, and a diff cut against
an older base is exactly how #15371 went wrong.

### What #15439 merging changed (2026-08-13)

**The hand-merge is gone.** #15439's `_forward` hunk used to conflict with #15375,
needing two `seg_t`/`seg_tag` `cond_audio` entries added by hand. #15375 was rebased
onto the merged #15439 and applied clean, then merged itself on 2026-08-18 — so there
is nothing left to apply on a current core.

**`patch_guide_origin.py` is obsolete on current core.** The merged #15439 anchors a
guide on the target origin by itself — measured on the live class, guide `11.000`
against target `11.000` with one image reference, where the draft gave `-1`. The wrap
would now over-correct by exactly the reference advance. It does not, because its
self-test compares the shifted result against the target origin and **rolls back**,
leaving stock alone and reporting `is_applied() == False`. That is the success case,
not a failure. It is kept because it is inert and self-disabling, and because anyone
on an older core still needs it.

**The mask is no longer binary.** This document used to warn that `mask_row_targets`
reduced a mask to one bool per patch row, so partial `overlap_strength` graded the
latent but not the timestep — and to re-check if #15375 changed. It changed:

```python
old:  target = m.reshape(-1) >= 0.5   # bool, all-or-nothing
new:  values = m.reshape(-1)          # float in [0, 1]   -- mask_row_values
```

So partial strength now grades the **timestep conditioning** too, and a feathered
spatial mask no longer hardens at the 0.5 contour. The function was **renamed** in the
process: anything detecting #15375 by `hasattr(mm, "mask_row_targets")` silently
stopped detecting it, which disabled every masking node in the pack. `nodes_loop.py`
now accepts either name, and `per_row_mask_is_continuous()` reports which behaviour
the installed core has.

## The post-ref guide origin: a wrap, not a core edit

**No PR carries this. This pack does**, as `mmh3tools/patch_guide_origin.py`, applied
at import. Core is NOT edited for it -- a core edit would be a diff to re-apply after
every `git pull` and to remember when reading a bug report from someone who lacks it.

`cond_t = float(text_len) + FRAME_RESCALE * resolved_frame_index` anchors to
`text_len`. The target begins at `cursor`, which the refs advance. Measured on the
real `PackedLayout`, guide origin versus target origin:

| refs attached | stock #15439 | with the wrap |
|---|---|---|
| none | 0 | 0 |
| one image ref | **-1** | 0 |
| audio / voice ref | **-320** | 0 |
| `video_audio` ref | **-37** | 0 |
| image + audio | **-321** | 0 |

Nothing errors. The guide anchors into the reference region, and `cond_audio` goes
with it -- so a carried tail's **audio** lands early too. It matters *more* under
#15439, not less, because the same PR fixes the `cond_video_latents` clobber
**specifically so guides and refs can coexist**, which makes the broken configuration
reachable.

The wrap lets stock build the layout, then shifts the `cond` and `cond_audio` rows by
the advance:

```python
for a, b, kind in self.segments:
    if kind in ("cond", "cond_audio"):
        self.position_ids[a:b, 0] += advance
```

Uniform addition rather than per-row assignment, so whatever intra-block structure
stock built survives. `_ref_cursor_advance` mirrors the `if refs:` cursor arithmetic;
a test compares it against the target origin the layout actually produced, because a
drift between the two is a silently misplaced guide -- the exact failure this removes.

**On `main`, deliberately.** The pack's rule sends monkeypatches to `keyframe-anchors`,
but `main`'s looping sampler needs this and no upstream PR exists to wait for. It comes
out the moment one lands; `apply()` already detects a core that has its own fix and
declines.

Inert unless BOTH guides and refs are present, self-tested at import against the live
class, and rolls back rather than misplacing a guide. Reported upstream on #15439.

## Monkeypatches — `keyframe-anchors` only

A wrap of core that **this pack maintains indefinitely**, because upstream has no plan to
change the thing it works around. That is what the branch is for.

| patch | what | status |
|---|---|---|
| `mmh3tools/patch_layout.py` | wraps `PackedLayout.__init__` for interior keyframe anchors | **superseded by #15439** |
| `mmh3tools/patch_conds.py` | wraps `MiniMaxH3.extra_conds` so keyframes and references coexist | **superseded by #15439** |

Both are absolute rebuilds, inert unless used, and self-tested at import — they refuse to
install rather than corrupt output. `MMH3LatentKeyframe` depends on them, so it lives
there too.

**#15439 does both of these upstream**, which is the outcome the branch existed to reach:
it deletes the first/last `raise` outright, and it fixes the `cond_video_latents`
overwrite by concatenating keyframes-then-refs — the same order, for the same reason.

The patches detect this and decline: `patch_layout` searches for the
`only first/last keyframe anchors are supported` text, which #15439 removes, so it finds
no anchor and leaves stock alone. Nothing breaks; the branch simply has no work left.
Retire it once #15439 merges rather than while it is still a draft that could be
withdrawn.

## Already in core — do not rebuild

**The context-window VRAM estimate is already correct.** This was rebuilt once, as
`MMH3ContextWindowVRAM` in 0.62.0, and reverted in 0.62.1 when it turned out to
duplicate core exactly. Recorded here so the reasoning that led there does not get
repeated, because it is superficially convincing.

The trap: `_prepare_sampling` estimates VRAM from `noise_shape` and hands it to
`load_models_gpu`, and `BaseModel.memory_required` reduces that shape to
`batch * prod(shape[2:])`. That looks like it must scale with total clip length,
which would mean a long windowed sample reserves many times what it needs and pushes
the model to RAM. The arithmetic is real — at 2K 4:3, 847 latents against a 47-latent
window is 24.9 GB versus 1.4 GB.

It never happens, because core clamps the shape first
(`comfy/context_windows.py`, `_prepare_sampling_wrapper`):

```python
# Scale noise_shape to a single context window so VRAM estimation budgets per-window.
elif handler.dim < len(noise_shape) and noise_shape[handler.dim] > handler.context_length:
    noise_shape[handler.dim] = min(noise_shape[handler.dim], handler.context_length)
```

`create_prepare_sampling_wrapper(model)` installs it, and `MMH3ContextWindows` already
calls that. Verified against the live wrapper: a `[1,24,847,128,96]` shape reaches the
estimator as `T=47`.

Two things the clamp does NOT cover, which is where to look instead when a long
windowed pass is memory-bound:

- **Packed/flat latents are skipped deliberately.** The `is_packed` branch is a
  documented `pass` — `latent_shapes` is not attached yet, so it cannot compute a
  per-window flat latent and over-estimates on purpose. Does not apply to H3, whose
  latents are `[B,24,T,h,w]`.
- **Only the estimate is windowed, not the allocations.** The full latent and every
  sampler copy of it stay resident, as do the fuse accumulators, and those do scale
  with total length. That is real memory, not an estimate, and no wrapper reduces it.

## Deliberately not applied

| PR | why not |
|---|---|
| #15270 pyros-projects | H3 attention patch hooks. Nothing here uses them, and it touches the same file as #15375, so it is pure conflict surface for no current gain. |
| #15353 xiaolibai-sys | 650 lines of pruned-LoRA support, unused here. |
| **#15371** Deno2026 | **Applied, then reverted — it breaks audio encode.** `disable_offload = True` on the audio VAE swaps `CoreModelPatcher` for plain `ModelPatcher`, flipping `assign=self.patcher.is_dynamic()` to False; the weights then load float32 while the encode path still feeds half. It is a competing fix for something **#15377 already solved upstream** using `comfy.ops.cast_to_input`. The lesson: check whether an open PR has been superseded by a merged one before applying it. |

## A note on the mask being binary — RESOLVED

Superseded by the #15375 rebase; see "What #15439 merging changed" above. The mask
now carries per-row floats, so the AdaLN lerp receives a continuous weight. Kept as a
heading because the old behaviour is still what you get on a core predating
2026-08-13, and because it is the reason several tooltips used to say
"binarises at 0.5".

## Reverting and updating

Revert with `git checkout -- <files>`, not the `.pre-*` backups — git HEAD is the
authoritative stock, and the backups differ from it by a BOM.

Always check whether upstream touches the same files before pulling:

```bash
cd C:/ComfyUI
curl -sL "https://api.github.com/repos/Comfy-Org/ComfyUI/compare/$(git rev-parse HEAD)...master" \
  | python -c "import sys,json;[print(f['filename']) for f in json.load(sys.stdin)['files']]"
```

If none of `comfy/model_base.py`, `comfy/ldm/minimax/model.py`, `comfy/samplers.py` or
`comfy/text_encoders/minimax.py` appear, pull straight over the top. If any do, revert
first, pull, then re-fetch and re-apply the PRs.
