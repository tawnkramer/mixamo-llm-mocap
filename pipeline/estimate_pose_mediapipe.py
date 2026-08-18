"""SPIKE / PROTOTYPE — not runtime-tested, not wired into the pipeline.

Investigates swapping the GVHMR estimator (non-commercial license, see
docs/MEDIAPIPE_INVESTIGATION.md) for MediaPipe Pose Landmarker
(Apache-2.0, no SMPL/SMPL-X dependency). Read that doc first — this file
implements only the directly-mappable part of the landmarks.json
contract; two fields are stubbed and clearly marked:

  pelvis_height  — MediaPipe's world landmarks are hip-centered
                   INDEPENDENTLY every frame, with no frame shared across
                   the clip (GVHMR's pelvis_height instead comes from one
                   ground-fixed frame computed once over the whole clip).
                   That makes this structurally hard, not just missing:
                   a hip-centered-per-frame representation cannot see
                   whole-body rigid vertical translation (a jump) at all
                   — see estimate_ground_height()'s docstring for the
                   full argument. The heuristic below only catches
                   crouches/weight-shifts (real joint articulation); it
                   will misread jumps. Every downstream stage that plants
                   a foot or integrates a jump depends on this being right.
  incam/incam_root — no camera-space translation solve exists here at
                   all. Left as zeros. Only matters for `root_motion`
                   specs (today: the two-fighter duel specs). Solo clips
                   never read these fields.

Multi-person (`--person left|right`) uses MediaPipe's native
`num_poses=2` per-frame detection plus a simplified port of
estimate_pose_gvhmr.py's screen-side/nearest-centroid tracking — MediaPipe
does not track identity across frames on its own.

Requires a MediaPipe Tasks pose_landmarker `.task` model file (download
separately, Apache-2.0, e.g. `pose_landmarker_heavy.task` from Google's
MediaPipe model index) and the `mediapipe` pip package. Neither is
installed in this sandbox; the MediaPipe Tasks API calls below are
written from the public API surface, not verified against a live import.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Same 33-name convention as estimate_pose_gvhmr.py / the rest of the
# pipeline. This is MediaPipe's own BlazePose landmark order, so no
# remapping is needed here (unlike GVHMR, which had to synthesize it).
MP_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def load_landmarker(model_path: Path, num_poses: int):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def mp_to_world(p) -> np.ndarray:
    """MediaPipe world-landmark axes -> this repo's convention (y down,
    z negative toward camera, hip-centered). MediaPipe's pose_world_landmarks
    are approximately: origin at hip midpoint, x toward the subject's
    left, y up, z toward the camera. That is a 180 degree flip on y and z
    versus this repo's convention (same shape of remap as
    estimate_pose_gvhmr.py's `ayfz_to_mp`, different source axes) —
    UNVERIFIED against a live model output; check against a real plate
    before trusting sign conventions here.
    """
    return np.array([p.x, -p.y, -p.z])


def side_track_frame(hip_mids: list[np.ndarray], prev_slots_x: list[float] | None):
    """Assign this frame's detected poses to stable left/right slots by
    hip-center x, nearest to the previous frame's slot positions.
    Simplified port of estimate_pose_gvhmr.py's side_track(): no YOLO
    track ids to lean on here, MediaPipe's per-frame detections are the
    only signal, so this is pure nearest-centroid with no interpolation
    or smoothing pass. Good enough for a spike; a real port should reuse
    the moving-average smoothing GVHMR's version applies.
    """
    order = sorted(range(len(hip_mids)), key=lambda i: hip_mids[i][0])
    if prev_slots_x is None:
        return order, [hip_mids[i][0] for i in order]
    slots = []
    used = set()
    for sx in prev_slots_x:
        j = min((i for i in order if i not in used), key=lambda i: abs(hip_mids[i][0] - sx))
        used.add(j)
        slots.append(j)
    return slots, [hip_mids[j][0] for j in slots]


def estimate_ground_height(ankle_y_by_frame: np.ndarray, ankle_speed_by_frame: np.ndarray) -> float:
    """UNVALIDATED heuristic, and — worked through below — structurally
    broken for the one thing pelvis_height matters most for: jumps.

    MediaPipe's pose_world_landmarks are hip-centered INDEPENDENTLY each
    frame (no shared global frame across the clip). GVHMR's pelvis_height
    instead comes from a single consistent ground-fixed frame computed
    ONCE over the whole clip (see estimate_pose_gvhmr.py's smpl_joints():
    pelvis_height is read before the per-frame hip-centering is applied).

    That ordering matters: if the whole body translates rigidly upward
    (a two-footed jump, joint angles unchanged), a hip-centered frame
    shows ZERO change in ankle-relative-to-hip height, because hip-
    centering removes exactly the signal a jump produces. This function
    can only see articulation (knee/hip bend — crouches, weight shifts),
    never whole-body vertical translation. It will silently miss jump
    height and probably read a jump as a flat "both planted" stretch.

    A real fix needs the same kind of camera-space/metric solve flagged
    for incam in docs/MEDIAPIPE_INVESTIGATION.md: recover absolute
    vertical motion from 2D image-space displacement of the hip point
    plus a depth/scale estimate (e.g. from apparent limb length vs
    camera intrinsics), not from anything in pose_world_landmarks alone.
    Kept here only so `plant: "both"` clips with real crouches produce
    plausible-looking numbers; treat any jump/flight window as unverified
    until that solve exists.
    """
    if len(ankle_y_by_frame) < 5:
        return float(np.min(ankle_y_by_frame)) if len(ankle_y_by_frame) else 0.0
    speed_thresh = np.percentile(ankle_speed_by_frame, 25)
    height_thresh = np.percentile(ankle_y_by_frame, 25)
    candidates = ankle_y_by_frame[(ankle_speed_by_frame <= speed_thresh) & (ankle_y_by_frame <= height_thresh)]
    if len(candidates) < 3:
        return float(np.min(ankle_y_by_frame))
    return float(np.median(candidates))


def main() -> None:
    ap = argparse.ArgumentParser(description="[SPIKE] MediaPipe Pose -> landmarks.json (see docs/MEDIAPIPE_INVESTIGATION.md)")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=REPO / "tools" / "mediapipe" / "pose_landmarker_heavy.task")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--person", default=None, help="'left', 'right', or a 0-based slot index; omit for a solo plate")
    args = ap.parse_args()

    warnings.warn(
        "estimate_pose_mediapipe.py is an unvalidated spike (docs/MEDIAPIPE_INVESTIGATION.md). "
        "pelvis_height uses an unverified heuristic; incam/incam_root are zeros — "
        "duo/root_motion specs will not work.",
        stacklevel=1,
    )

    def rp(p: Path) -> Path:
        return p if p.is_absolute() else (REPO / p)

    video = rp(args.video)
    out_path = rp(args.out)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    if not args.model.exists():
        raise SystemExit(
            f"missing MediaPipe model: {args.model}\n"
            "download a pose_landmarker .task file (Apache-2.0) from Google's "
            "MediaPipe model index and pass --model, or place it at the default path."
        )

    cap = cv2.VideoCapture(str(video))
    fps = args.fps or (cap.get(cv2.CAP_PROP_FPS) or 24.0)
    n_people = 2 if args.person is not None else 1
    landmarker = load_landmarker(args.model, n_people)

    frames = []
    hip_mid_world_by_frame = []
    prev_slots_x = None
    frame_idx = 0
    import mediapipe as mp

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(frame_idx * 1000.0 / fps)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        if not result.pose_world_landmarks:
            frames.append({"frame": frame_idx + 1, "t": frame_idx / fps, "ok": False,
                           "world": {}, "image": {}, "incam": {}})
            frame_idx += 1
            continue

        if args.person is not None:
            hip_mids = []
            for lm_set in result.pose_world_landmarks:
                lh = mp_to_world(lm_set[MP_NAMES.index("left_hip")])
                rh = mp_to_world(lm_set[MP_NAMES.index("right_hip")])
                hip_mids.append(0.5 * (lh + rh))
            slots, prev_slots_x = side_track_frame(hip_mids, prev_slots_x)
            order = {"left": 0, "right": n_people - 1}
            slot = order.get(args.person, int(args.person) if str(args.person).isdigit() else 0)
            pose_idx = slots[slot] if slot < len(slots) else 0
        else:
            pose_idx = 0

        world_lm = result.pose_world_landmarks[pose_idx]
        img_lm = result.pose_landmarks[pose_idx]

        world_pts = {n: mp_to_world(world_lm[i]) for i, n in enumerate(MP_NAMES)}
        hip_mid = 0.5 * (world_pts["left_hip"] + world_pts["right_hip"])
        hip_mid_world_by_frame.append(hip_mid)

        ear_mid = 0.5 * (world_pts["left_ear"] + world_pts["right_ear"])
        gaze = world_pts["nose"] - ear_mid
        gn = float(np.linalg.norm(gaze)) or 1.0
        gaze = gaze / gn

        rec = {
            "frame": frame_idx + 1, "t": frame_idx / fps, "ok": True,
            # placeholder — filled from estimate_ground_height() after the
            # loop, once we've seen the whole clip's ankle trajectory.
            "pelvis_height": None,
            "root": [round(float(hip_mid[0]), 5), round(float(hip_mid[2]), 5)],
            "incam_root": [0.0, 0.0, 0.0],  # NOT IMPLEMENTED — see module docstring
            "gaze": [round(float(x), 5) for x in gaze],
            "image": {}, "world": {}, "incam": {},
        }
        for i, n in enumerate(MP_NAMES):
            w = world_pts[n] - hip_mid
            lm = img_lm[i]
            rec["world"][n] = {"x": round(float(w[0]), 6), "y": round(float(w[1]), 6), "z": round(float(w[2]), 6),
                               "visibility": round(float(lm.visibility), 4), "presence": round(float(lm.presence), 4)}
            rec["image"][n] = {"x": round(float(lm.x), 6), "y": round(float(lm.y), 6), "z": round(float(lm.z), 6),
                               "visibility": round(float(lm.visibility), 4), "presence": round(float(lm.presence), 4)}
            rec["incam"][n] = {"x": 0.0, "y": 0.0, "z": 0.0}  # NOT IMPLEMENTED — see module docstring
        frames.append(rec)
        frame_idx += 1

    cap.release()

    # Second pass: ground height from the whole clip's ankle trajectory,
    # then absolute pelvis_height per frame. See estimate_ground_height()
    # docstring — this is the unvalidated heuristic.
    ok_frames = [f for f in frames if f["ok"]]
    if ok_frames:
        ankle_y = np.array([0.5 * (f["world"]["left_ankle"]["y"] + f["world"]["right_ankle"]["y"]) for f in ok_frames])
        ankle_speed = np.zeros_like(ankle_y)
        ankle_speed[1:] = np.abs(np.diff(ankle_y)) * fps
        ground_y = estimate_ground_height(ankle_y, ankle_speed)
        # world is hip-centered every frame, so the hip's own world y is
        # 0 by construction — pelvis_height collapses to ground_y itself,
        # which is exactly the blind-to-vertical-translation problem the
        # estimate_ground_height() docstring works through.
        for f in ok_frames:
            f["pelvis_height"] = round(ground_y, 5)

    payload = {
        "source": "mediapipe_pose_landmarker_SPIKE",
        "person": args.person,
        "model": str(args.model),
        "fps": fps,
        "frame_count": len(frames),
        "ok_count": len(ok_frames),
        "landmark_names": MP_NAMES,
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {out_path} frames={len(frames)} ok={len(ok_frames)} fps={fps:.3f}")
    print("SPIKE: pelvis_height is an unvalidated ground-plane heuristic; incam/incam_root are zeros.")
    print("See docs/MEDIAPIPE_INVESTIGATION.md before trusting this output.")


if __name__ == "__main__":
    main()
