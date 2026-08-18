# Spec annotator (v1 — plant schedule, rest blend, fists)

A minimal, dependency-free viewer for authoring the hand-written part of
an `action_spec` visually instead of by reading numbers off
`analyze_landmarks.py` and typing frame ranges into JSON. See
`docs/PIPELINE.md` for the full spec schema this feeds into.

## Use

Open `annotator/index.html` directly in Chrome or Edge (no server, no
build step). Then:

1. **Open video…** — the plate's `.mp4` (the same file you ran
   `estimate_pose_gvhmr.py` on).
2. **Open landmarks.json…** — that video's estimator output.
3. **Open action_spec.json…** (to edit an existing spec) or **New
   spec** (to start one). Opening an existing spec via the file picker
   keeps a live handle to it — **Save spec** writes back to the same
   file in place.
4. Scrub the video (arrow keys step one frame at a time) and watch the
   plot below it: pelvis height, wrist depth (the punch signal) and
   ankle height (the plant/kick signal) — the same numbers
   `analyze_landmarks.py` prints, just synced to the video instead of a
   static table.
5. Drag a range on the plot to mark a `plant` window, pick its support
   (`both`/`left`/`right`/`none`), and it appears in the table on the
   right — editable, deletable.
6. Fill in `rest blend` / `fists` frame numbers directly (they're
   single ranges, not worth dragging for).
7. **Save spec.**

Saving only ever writes `plant`, `rest_blend_start`, `rest_blend_end`
and `fists` into the spec — every other field (`arm_overrides`,
`arm_pose`, `reach`, `smooth`, the duel-only fields) is left exactly as
it was in the opened file, so this tool composes with hand-editing the
rest of the spec rather than replacing it.

Firefox and other browsers without the File System Access API still
work for viewing/authoring, but **Save** downloads a new
`action_spec.json` instead of writing in place — merge it in by hand.

## What this deliberately doesn't do (yet)

- No `arm_overrides` authoring. Doing this from a 2D drag on the video
  would need a real position in the spec's METERS/hip-basis convention,
  which needs either the estimator's camera intrinsics (not currently
  stored in `landmarks.json`) or a defined 2D→3D convention — an open
  design question, not a UI feature. Occluded-limb corrections still go
  in by hand for now.
- No 3D/Blender preview, no `arm_pose`/`reach`/`smooth`/duel-field
  editing, no undo history beyond re-opening the file.

These are natural v2 targets if the plant-schedule workflow above
earns its keep.
