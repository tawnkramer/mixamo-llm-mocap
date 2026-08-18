# Swapping GVHMR for MediaPipe Pose — feasibility investigation

Spike branch: `investigate/mediapipe-pose-swap`. Goal: find out what it
would actually take to replace `pipeline/estimate_pose_gvhmr.py` (GVHMR,
non-commercial research license) with a permissively-licensed estimator,
since MediaPipe Pose Landmarker (Apache-2.0, no SMPL/SMPL-X dependency)
is the only candidate that escapes the licensing problem entirely (see
prior discussion — 4D-Humans, WHAM, NLF are all MIT *code* but still
regress SMPL/SMPL-X, which is itself non-commercial and registration-gated).

**Not runtime-tested.** This sandbox has no `mediapipe` package, no GPU,
and no sample plate video (plates/ is gitignored, author's clips only).
Everything below is read from the existing pipeline code + MediaPipe's
public docs/API, not verified end-to-end. Treat the prototype script as
a structural spike, not a working replacement.

## The actual contract

`estimate_pose_gvhmr.py` is the *only* file every other pipeline stage
depends on for input data — nothing downstream cares that GVHMR produced
it, they just read `landmarks.json`. So the swap is contained *if* the
replacement can emit the same schema. Per frame, that schema is:

| Field | What it is | Who reads it |
|---|---|---|
| `world[name]` | 33 MediaPipe-named landmarks, meters, per-frame **mid-hip centered**, y down, z negative toward camera | every stage — the core signal |
| `image[name]` | landmarks projected to normalized [0,1] pixel space | `render_preview.py` overlays |
| `incam[name]` | the same 33 landmarks in **camera space** (shared frame across two separately-estimated performers) | `compare_pair.py`, duo root motion |
| `incam_root` | pelvis position in camera space | `lift_to_mixamo.py` root motion (`root_source: incam`) |
| `root` | hip-mid ground trajectory in the performer's own gravity-aligned frame | `lift_to_mixamo.py` root motion fallback (`root_source: global`) |
| `pelvis_height` | **absolute** height above the ground plane (not hip-relative) | `analyze_landmarks.py` (true joint heights), `apply_mixamo_fk.py` (hip-height solve, support planting), `lift_to_mixamo.py` (airborne integration) |
| `gaze` | unit vector, real face direction (nose vs ear-midpoint) | `apply_mixamo_fk.py` head orient, `compare_reference.py` gaze scoring |
| `visibility`/`presence` | per-landmark confidence | currently unused downstream but part of the schema |

The landmark topology (`MP_NAMES`, 33 points) is *already* MediaPipe's
own BlazePose naming — the GVHMR script literally synthesizes MediaPipe-
shaped output from SMPL joints. That was clearly a deliberate choice:
topology compatibility is not a blocker.

## Field-by-field feasibility

| Field | MediaPipe Pose Landmarker gives it? | Notes |
|---|---|---|
| `world` | **Yes, directly** — `pose_world_landmarks` (metric, hip-centered) | Needs an axis remap (MediaPipe's world landmarks use a different up/forward convention than this repo's y-down/z-toward-camera one) — same kind of remap `ayfz_to_mp()` already does for GVHMR, just a different source convention |
| `image` | **Yes, directly** — `pose_landmarks` (normalized image coords) | 1:1 |
| `visibility`/`presence` | **Yes, directly** — native per-landmark fields | This is literally where the schema's field names came from |
| `gaze` | **Partial.** Pose landmarks include nose/eyes/ears, so the same nose-vs-ear-midpoint vector *can* be computed | But these are coarse body-pose landmarks, not a face mesh — expect noisier head orientation than GVHMR's real mesh vertices. MediaPipe's separate **Face Landmarker** (478 points, also Apache-2.0) run alongside Pose could actually give a *better* head orientation than GVHMR's current heuristic, if fused in — an upside, not just a gap |
| `pelvis_height` | **No — and not just missing, structurally hard.** MediaPipe's `pose_world_landmarks` are hip-centered *independently every frame*, with no frame shared across the clip. GVHMR's `pelvis_height` instead comes from a single ground-fixed frame computed once over the whole clip, read *before* GVHMR applies its own per-frame hip-centering. | A hip-centered-per-frame representation is blind to whole-body rigid vertical translation by construction: a two-footed jump with unchanged joint angles produces **zero signal** in ankle-relative-to-hip height. A "detect ground from stable low ankles" heuristic (prototyped below) can pick up crouches/weight shifts (real articulation) but will miss jump height entirely — the one case `pelvis_height`'s biggest consumer (airborne integration) most needs. Recovering it for real needs the same kind of camera-space/metric solve as `incam` below: 2D image-space vertical displacement of the hip plus a depth/scale estimate, not anything derivable from world landmarks alone. This turned out to be the investigation's most important finding — not an engineering-effort gap so much as a genuinely different problem than GVHMR solves |
| `incam` / `incam_root` | **No.** No camera-space translation solve — MediaPipe Pose is a per-frame, per-person 2.5D landmark regressor with no absolute-scale camera pose | Only matters when `root_motion: true`, which today is only the **two-fighter duel specs** (solo clips retarget in place and never read `incam`). Faking it would need a PnP-style solve from 2D landmarks + assumed/measured bone length + camera intrinsics — a real sub-project, not a remap |
| multi-person (`--person left\|right`) | **Yes, structurally** — the Tasks API's `PoseLandmarker` supports `num_poses>1` per frame | But MediaPipe does not track identity across frames; you'd need to reimplement `side_track()`'s screen-side/nearest-centroid logic against MediaPipe's per-frame detections instead of GVHMR's YOLO tracker — portable, not free |

## The gap that isn't in the table: depth quality

The GVHMR module docstring says the quiet part out loud (`estimate_pose_gvhmr.py:31-32`):

> GVHMR gives real depth, plausible proportions and no per-frame Z
> spikes; it still does not know Mixamo bone lengths...

That property — temporally consistent depth from a full parametric body
fit — is exactly what a lightweight per-frame 2.5D landmark regressor
like BlazePose (MediaPipe's underlying model) is weakest at. This
pipeline's whole beat-detection scheme keys off Z depth (`analyze_landmarks.py`:
"wrist z dips below −0.32" for punches) and this codebase's own hard-won
lesson, repeated throughout `docs/PITFALLS.md`, is that Z jitter and
mismatched proxies are the #1 source of bad retargets. This is a real
accuracy risk, not just missing plumbing — punches and kicks *toward the
camera* are this pipeline's core content and MediaPipe's known weak axis.

## Licensing — confirmed

- MediaPipe (Apache-2.0), no SMPL/SMPL-X, no registration gate.
- Pretrained pose_landmarker model files ship under the same Apache-2.0
  terms as the library (Google's own model card).
- This is the only candidate of the ones surveyed that actually clears
  the non-commercial trap 4D-Humans/WHAM/NLF all fall into via SMPL.

## What would actually be needed — phased

1. **Estimator swap (`estimate_pose_mediapipe.py`)** — direct mapping of
   `world`/`image`/`visibility`/`presence`, naive `gaze`. *Smallest piece,
   prototyped below.*
2. **`pelvis_height` — a real metric-scale solve, not a heuristic patch.**
   As found above, this needs absolute vertical motion recovered from 2D
   image-space displacement plus a depth/scale estimate (apparent limb
   length vs camera intrinsics, or similar) — closer in kind to step 5's
   `incam` solve than to a simple ground-height lookup. The prototype's
   ankle-height heuristic is included only as a stand-in that produces
   plausible numbers for crouches/weight-shifts; it will misread jumps.
   Nothing downstream (`plant` solving, airborne integration) can be
   trusted on jump/flight content until this is done properly.
3. **`gaze` quality pass** — optionally run Face Landmarker alongside
   Pose Landmarker and fuse, instead of trusting Pose's coarse face points.
4. **Multi-person tracking** — port `side_track()`'s screen-side/nearest-
   centroid assignment onto per-frame `num_poses=2` detections. Needed
   only for duel-style specs.
5. **`incam`/`incam_root`** — camera-space solve for root motion. Needed
   only for duel-style specs with `root_motion: true`. Realistically the
   largest single piece of new work if two-performer plates matter to you;
   skippable if you only care about solo clips.
6. **Validate against a real plate** — re-run `analyze_landmarks.py`,
   `qa_clip.py`, `compare_reference.py` on a MediaPipe-sourced
   `landmarks.json` for one of the existing `action_specs/` and diff the
   beat numbers against the GVHMR-sourced one this repo was tuned against.

Steps 1 and 3 are genuinely small. Step 2 is required before anything
downstream is usable and has no existing analog to port. Step 5 is
required only for the duo pipeline and is a project of its own.

## Prototype

`pipeline/estimate_pose_mediapipe.py` in this branch implements phase 1
only: it stands up the MediaPipe Tasks `PoseLandmarker`, remaps its output
into this repo's `world`/`image` convention, and computes a naive `gaze`
from the pose landmarks' own nose/ear points. `pelvis_height` is left at
a **flagged placeholder** (a same-signature ground-height heuristic,
clearly marked as unvalidated) and `incam`/`incam_root` are emitted as
zeros with a printed warning — so the file is honest about being solo-
clip-only and pre-`plant`-validation. It has not been run against real
footage in this environment.

## Bottom line

Licensing-wise MediaPipe is the clean answer. Engineering-wise the swap
is contained to one file at the interface level (the schema was already
shaped for this), but `pelvis_height` and `incam`/`incam_root` both turn
out to need the same kind of thing: a metric-scale, camera-space solve
that recovers absolute position from MediaPipe's per-frame hip-centered
landmarks — neither is a small patch, and there's no prior art for it in
this pipeline to port from. On top of that there's a real open question
on whether BlazePose's depth quality holds up on the toward-camera
strikes this pipeline is built around. This isn't a weekend swap: solo
clips without jumps could plausibly land once the metric-scale solve
exists (it also unblocks jumps and the whole duo pipeline at once, so it
is the one piece of new work actually worth doing first), but calling it
"contained" would undersell what's missing.
