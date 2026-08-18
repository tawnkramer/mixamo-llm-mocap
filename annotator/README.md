# Spec annotator

A minimal, dependency-free viewer for authoring the hand-written part of
an `action_spec` visually instead of by reading numbers off
`analyze_landmarks.py` and typing frame ranges into JSON — and for
correcting the specific things a monocular 2D-ish estimator (this
matters most for a MediaPipe-based estimator; see
`docs/MEDIAPIPE_INVESTIGATION.md`) gets structurally wrong: depth, and
whole-body vertical translation (jumps). See `docs/PIPELINE.md` for the
full spec schema this feeds into.

## Use

Open `annotator/index.html` directly in Chrome or Edge (no server, no
build step). Then:

1. **Open video…** — the plate's `.mp4`.
2. **Open landmarks.json…** — that video's estimator output. Opened via
   the file picker, this keeps a live handle so **Save landmarks.json**
   can write corrections back in place.
3. **Open action_spec.json…** (to edit an existing spec) or **New
   spec**. Same in-place-write behavior as landmarks.json, via **Save
   spec**.

### Plant schedule / rest blend / fists

Scrub the video (arrow keys step one frame) and watch the numeric-
signals plot below it — pelvis height, wrist depth (punch signal),
ankle height (plant/kick signal), the same numbers
`analyze_landmarks.py` prints, synced to the video instead of a static
table. Drag a range on the plot to mark a `plant` window and pick its
support (`both`/`left`/`right`/`none`); fill in rest-blend/fists frame
numbers directly. **Save spec** writes `plant`, `rest_blend_start/end`
and `fists` into the opened spec — every other field is left untouched.

### Side view — correcting depth (arm_overrides)

The camera view can't show depth at all, which is exactly the axis a
monocular estimator is weakest on. The side view renders a synthetic
elevation projection of the current frame — horizontal = depth,
vertical = height — computed straight from `landmarks.json`, geometry
ported exactly from `pipeline/lift_to_mixamo.py`'s hip-basis math so the
numbers it writes are correct in the same convention the FK apply reads
them in (not just plausible-looking).

Click **Correct left/right arm here**, drag the highlighted wrist/elbow
dots to where they actually are (lateral position is left as the
estimator's — only depth and height are adjustable from a single side
view), set the `src` window and ramp, **Add override**. Writes into the
spec's `arm_overrides` — same mechanism the pipeline already used for
occluded-limb correction, just driven visually instead of by hand-typed
meters.

**Rig hip height** (top of the panel, defaults to Y Bot's 0.998 m) has
to match your character — it's the one rig-specific constant the
geometry needs; read it off your `rig_profile.json` for any other rig.

### Pelvis height — correcting jump blindness

MediaPipe's (and to a lesser extent GVHMR's) `pelvis_height` can't see
whole-body vertical translation from world landmarks alone — see
`docs/MEDIAPIPE_INVESTIGATION.md`. Click **Edit pelvis height**, then
click on the numeric-signals plot to drop a keyframe pin at a frame
with a height (liftoff, apex, landing…), drag existing pins to adjust.
With ≥2 pins, everything between the first and last pin linearly
interpolates between them; outside that range the original estimator
value passes through unchanged — no automatic ramping, the human
declares the exact window, same idiom the rest of this schema uses.
**Save landmarks.json** writes the corrected `pelvis_height` values
into a copy of the opened file; nothing else in it changes.

Firefox and other browsers without the File System Access API still
work for viewing/authoring, but both Save buttons download a copy
instead of writing in place — merge it in by hand.

## What this deliberately doesn't do

- No leg equivalent of `arm_overrides` — the spec schema doesn't define
  one (kicks/knees use `leg_pose`, a different, rigid-rotation shape),
  so kick/knee depth correction needs a schema addition first, not just
  a UI one.
- No `arm_pose`/`reach`/`smooth`/duel-field editing, no 3D/Blender
  preview, no undo history beyond re-opening the file.
