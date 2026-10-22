"""
Two-player pose detection and identity tracking for IPCV Task 3.2.3.

Features
--------
- Detects multiple people and estimates 17 COCO body landmarks with YOLO pose.
- Uses Ultralytics' ByteTrack tracker to associate detections across frames.
- Assigns persistent display names: PLAYER 1 and PLAYER 2.
- Draws landmarks, skeleton connections, bounding boxes, and a label above each head.
- Smooths displayed landmark positions and tolerates brief missing detections.
- Prints each player's normalized image position and visible keypoint coordinates.

Tested design target: Python 3.10+ with a webcam.
The first run downloads the pretrained YOLO pose model if it is not cached.
"""

from collections import deque
import time

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------- Configuration -----------------------------

CAMERA_INDEX = 0
MODEL_NAME = "yolov8n-pose.pt"  # Lightweight model; try yolov8s-pose.pt for better accuracy.
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.50
MAX_PLAYERS = 2
SMOOTHING_ALPHA = 0.60  # Higher = more responsive; lower = smoother.
LOST_PLAYER_SECONDS = 3.0
IDENTITY_MEMORY_SECONDS = 15.0  # Keep identity features for re-entry matching.
SHOW_ALL_DETECTIONS = False  # False: only label the two assigned players.
WINDOW_NAME = "IPCV - Two-player pose and identity tracking"

# COCO keypoint indices used by YOLO pose:
# 0 nose, 5/6 shoulders, 7/8 elbows, 9/10 wrists,
# 11/12 hips, 13/14 knees, 15/16 ankles.
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]



def appearance_descriptor(frame, box):
    """Describe the upper-body clothing with a normalized HSV histogram."""
    x1, y1, x2, y2 = np.round(box).astype(int)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 12:
        return None

    # Approximate torso/clothing region; avoid most of the face and background.
    crop = frame[
        y1 + int(0.20 * bh): y1 + int(0.72 * bh),
        x1 + int(0.15 * bw): x1 + int(0.85 * bw)
    ]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def appearance_distance(hist_a, hist_b):
    """Bhattacharyya distance: 0 is similar, 1 is very different."""
    if hist_a is None or hist_b is None:
        return 0.5  # Neutral rather than treating missing appearance as a mismatch.
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def pose_descriptor(points, box):
    """Represent visible keypoints relative to the person's bounding box."""
    x1, y1, x2, y2 = box
    bw, bh = max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)
    normalized = points.copy().astype(float)
    valid = np.isfinite(normalized).all(axis=1)
    normalized[valid, 0] = (normalized[valid, 0] - x1) / bw
    normalized[valid, 1] = (normalized[valid, 1] - y1) / bh
    normalized[~valid] = np.nan
    return normalized


def pose_distance(pose_a, pose_b):
    """Compare shared visible landmarks, returning a bounded distance."""
    if pose_a is None or pose_b is None:
        return 0.5
    valid = np.isfinite(pose_a).all(axis=1) & np.isfinite(pose_b).all(axis=1)
    if int(valid.sum()) < 4:
        return 0.5
    distances = np.linalg.norm(pose_a[valid] - pose_b[valid], axis=1)
    return float(np.clip(np.mean(distances), 0.0, 1.0))


class PlayerIdentityManager:
    """
    Maintains persistent Player 1/2 records independently of ByteTrack IDs.
    Matches detections using clothing colour/appearance, predicted motion and pose.
    """

    def __init__(self, max_players=2, lost_seconds=3.0, memory_seconds=15.0):
        self.max_players = max_players
        self.lost_seconds = lost_seconds
        self.memory_seconds = memory_seconds
        self.player_state = {
            i: {
                "center": None, "velocity": (0.0, 0.0), "appearance": None,
                "pose": None, "track_id": None, "last_seen": 0.0,
                "last_box": None
            }
            for i in range(1, max_players + 1)
        }

    def _cost(self, player, detection, now, frame_diagonal):
        state = self.player_state[player]
        center = np.array(detection["center"], dtype=float)
        old_center = state["center"]

        if old_center is None:
            motion_cost = 0.5
        else:
            dt = max(0.0, min(now - state["last_seen"], 1.0))
            predicted = np.array(old_center) + np.array(state["velocity"]) * dt
            motion_cost = float(np.clip(
                np.linalg.norm(center - predicted) / max(frame_diagonal * 0.35, 1.0),
                0.0, 1.0
            ))

        app_cost = appearance_distance(state["appearance"], detection["appearance"])
        pose_cost = pose_distance(state["pose"], detection["pose"])

        # Appearance is deliberately the strongest cue to help through crossings.
        cost = 0.55 * app_cost + 0.30 * motion_cost + 0.15 * pose_cost
        if state["track_id"] is not None and detection["track_id"] == state["track_id"]:
            cost -= 0.08  # Helpful continuity cue, but never the main identity signal.
        return float(np.clip(cost, 0.0, 1.0))

    def _update_record(self, player, detection, now):
        state = self.player_state[player]
        center = np.array(detection["center"], dtype=float)

        if state["center"] is not None:
            dt = max(now - state["last_seen"], 1e-3)
            measured_velocity = (center - np.array(state["center"])) / dt
            # Smooth velocity estimates to avoid overreacting to detector jitter.
            old_velocity = np.array(state["velocity"], dtype=float)
            velocity = 0.65 * old_velocity + 0.35 * measured_velocity
            state["velocity"] = tuple(np.clip(velocity, -1200, 1200))
        else:
            state["velocity"] = (0.0, 0.0)

        if detection["appearance"] is not None:
            if state["appearance"] is None or now - state["last_seen"] > self.lost_seconds:
                state["appearance"] = detection["appearance"].copy()
            else:
                # Blend histograms so a single lighting change has limited impact.
                state["appearance"] = (
                    0.8 * state["appearance"] + 0.2 * detection["appearance"]
                )
                cv2.normalize(state["appearance"], state["appearance"], 1.0, 0, cv2.NORM_L1)

        if detection["pose"] is not None:
            state["pose"] = detection["pose"].copy()
        state["center"] = tuple(center)
        state["track_id"] = detection["track_id"]
        state["last_seen"] = now
        state["last_box"] = detection["box"].copy()

    def assign(self, detections, now, frame_shape):
        """
        Return {detection_index: player_number}. Ambiguous matches are withheld
        for that frame instead of silently swapping player identities.
        """
        h, w = frame_shape[:2]
        diagonal = float(np.hypot(w, h))

        # Expire very old records. A later arrival is treated as a new assignment.
        for player, state in self.player_state.items():
            if state["center"] is not None and now - state["last_seen"] > self.memory_seconds:
                state.update({
                    "center": None, "velocity": (0.0, 0.0), "appearance": None,
                    "pose": None, "track_id": None, "last_seen": 0.0, "last_box": None
                })

        assigned = {}
        active_players = [
            p for p, s in self.player_state.items() if s["center"] is not None
        ]
        unused_players = [
            p for p, s in self.player_state.items() if s["center"] is None
        ]

        if not active_players:
            # Initial calibration: Player 1 is the leftmost detected person.
            for player, det_idx in zip(
                range(1, self.max_players + 1),
                sorted(range(len(detections)), key=lambda i: detections[i]["center"][0])
            ):
                assigned[det_idx] = player
                self._update_record(player, detections[det_idx], now)
            return assigned

        costs = {
            (p, i): self._cost(p, detections[i], now, diagonal)
            for p in active_players for i in range(len(detections))
        }

        if len(active_players) >= 2 and len(detections) >= 2:
            # Evaluate both possible two-person assignments globally.
            p1, p2 = active_players[:2]
            best = None
            for i in range(len(detections)):
                for j in range(len(detections)):
                    if i == j:
                        continue
                    total = costs[(p1, i)] + costs[(p2, j)]
                    if best is None or total < best[0]:
                        best = (total, i, j)
            if best is not None:
                _, i, j = best
                direct = costs[(p1, i)] + costs[(p2, j)]
                swapped = costs[(p1, j)] + costs[(p2, i)]
                # If the alternatives are nearly indistinguishable during a close
                # crossing, don't emit uncertain controls for this frame.
                centers_close = np.linalg.norm(
                    np.array(detections[i]["center"]) - np.array(detections[j]["center"])
                ) < 0.12 * diagonal
                if centers_close and abs(direct - swapped) < 0.10:
                    return assigned
                for player, det_idx in ((p1, i), (p2, j)):
                    if costs[(player, det_idx)] < 0.88:
                        assigned[det_idx] = player
        else:
            # With one visible detection and two known players, compare it against
            # BOTH identities. If the match is ambiguous, withhold controls this frame.
            ranked_players = sorted(
                (costs[(p, i)], p, i)
                for p in active_players
                for i in range(len(detections))
            )
            if ranked_players:
                best_cost, best_player, best_det = ranked_players[0]
                second_cost = next(
                    (cost for cost, p, i in ranked_players
                     if p != best_player or i != best_det),
                    float("inf")
                )
                if best_cost < 0.88 and second_cost - best_cost > 0.06:
                    assigned[best_det] = best_player

        # If one identity has never been initialized, assign the unmatched detection
        # to it. Never steal an initialized identity just because a track ID changed.
        used_detections = set(assigned)
        used_players = set(assigned.values())
        available_slots = [p for p in unused_players if p not in used_players]
        remaining_detections = [i for i in range(len(detections)) if i not in used_detections]
        for player, det_idx in zip(
            available_slots,
            sorted(remaining_detections, key=lambda i: detections[i]["center"][0])
        ):
            assigned[det_idx] = player

        for det_idx, player in assigned.items():
            self._update_record(player, detections[det_idx], now)

        return assigned


def smooth_points(previous, current, alpha):
    """Exponential smoothing of a (17, 2) landmark array."""
    if previous is None or previous.shape != current.shape:
        return current.copy()
    result = current.copy()
    valid = np.isfinite(current).all(axis=1)
    prev_valid = np.isfinite(previous).all(axis=1)
    both = valid & prev_valid
    result[both] = alpha * current[both] + (1.0 - alpha) * previous[both]
    result[~valid] = np.nan
    return result


def draw_pose(frame, points, color):
    """Draw keypoints and the skeleton where the model has reasonable confidence."""
    for a, b in SKELETON:
        if np.isfinite(points[[a, b]]).all():
            pa = tuple(np.round(points[a]).astype(int))
            pb = tuple(np.round(points[b]).astype(int))
            cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

    for x, y in points:
        if np.isfinite([x, y]).all():
            cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)


def main():
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Try changing CAMERA_INDEX to 1 if you have multiple cameras."
        )

    # Keep the input reasonably small for real-time performance.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    identities = PlayerIdentityManager(MAX_PLAYERS, LOST_PLAYER_SECONDS, IDENTITY_MEMORY_SECONDS)
    smoothed_points = {}       # player number -> smoothed keypoints
    last_player_points = {}    # player number -> last keypoints (for short dropouts)
    last_player_box = {}       # player number -> last box
    last_player_seen = {}
    player_colors = {
        1: (40, 220, 40),       # green
        2: (255, 160, 40),      # blue/orange in BGR
    }

    previous_time = time.perf_counter()
    fps = 0.0

    print("Press Q or ESC to quit.")
    print("Keypoint coordinates are in webcam pixels; player x/y are normalized 0..1.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a webcam frame.")
                break

            # Mirror the image so movement feels natural. Coordinates below match this display.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.time()

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
            )[0]

            current_players = {}
            boxes = result.boxes
            keypoints = result.keypoints
            detections = []

            if boxes is not None and keypoints is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(xyxy)
                )
                all_points = keypoints.xy.cpu().numpy()
                all_conf = (
                    keypoints.conf.cpu().numpy()
                    if keypoints.conf is not None
                    else np.ones(all_points.shape[:2], dtype=float)
                )

                for i, box in enumerate(xyxy):
                    tid = track_ids[i]
                    if tid is None:
                        continue

                    points = all_points[i].astype(float)
                    points[all_conf[i] < 0.35] = np.nan
                    detections.append({
                        "box": box.copy(),
                        "track_id": int(tid),
                        "center": ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
                        "points": points,
                        "appearance": appearance_descriptor(frame, box),
                        "pose": pose_descriptor(points, box),
                    })

            assignment = identities.assign(detections, now, frame.shape)

            for det_idx, player in assignment.items():
                detection = detections[det_idx]
                points = smooth_points(
                    smoothed_points.get(player), detection["points"], SMOOTHING_ALPHA
                )
                smoothed_points[player] = points
                last_player_points[player] = points.copy()
                last_player_box[player] = detection["box"].copy()
                last_player_seen[player] = now
                current_players[player] = (
                    detection["box"], points, detection["track_id"]
                )


            # Draw current detections.
            for player, (box, points, tid) in current_players.items():
                color = player_colors[player]
                x1, y1, x2, y2 = np.round(box).astype(int)
                draw_pose(frame, points, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Prefer the nose landmark as the head anchor; fall back to box top-centre.
                if np.isfinite(points[0]).all():
                    label_x = int(points[0, 0])
                    label_y = int(points[0, 1] - 50)
                else:
                    label_x = int((x1 + x2) / 2)
                    label_y = max(25, y1 - 35)

                label = f"PLAYER {player}"
                cv2.rectangle(
                    frame,
                    (max(0, label_x - 8), max(0, label_y - 25)),
                    (min(width - 1, label_x + 125), min(height - 1, label_y + 7)),
                    color,
                    -1,
                )
                cv2.putText(
                    frame, label, (max(0, label_x - 4), max(20, label_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA
                )

                # Spatial information for Task 4: normalized torso/body-box center.
                nx = float(np.clip(((x1 + x2) / 2) / width, 0, 1))
                ny = float(np.clip(((y1 + y2) / 2) / height, 0, 1))
                print_line = (
                    f"PLAYER {player}: x_norm={nx:.3f}, y_norm={ny:.3f}, "
                    f"box_px=({x1},{y1},{x2},{y2}), tracker_id={tid}"
                )
                cv2.putText(
                    frame, f"x={nx:.2f}, y={ny:.2f}",
                    (x1, min(height - 8, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
                )
                # Uncomment if you want a console line every frame:
                # print(print_line)

            # During a short detection gap, show a dashed/ghost-style box and identity label.
            # This is visual grace only; no pose is claimed to be currently detected.
            for player in range(1, MAX_PLAYERS + 1):
                if player in current_players:
                    continue
                last_seen = last_player_seen.get(player)
                if last_seen is None or now - last_seen > LOST_PLAYER_SECONDS:
                    continue
                box = last_player_box.get(player)
                if box is None:
                    continue
                x1, y1, x2, y2 = np.round(box).astype(int)
                color = player_colors[player]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                cv2.putText(
                    frame, f"PLAYER {player} (temporarily lost)",
                    (max(5, x1), max(22, y1 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                )

            # Show missing-player warning at the bottom centre of the frame.
            # A player is considered visible only when assigned a detection this frame.
            missing_players = [
                player for player in range(1, MAX_PLAYERS + 1)
                if player not in current_players
            ]

            if missing_players:
                message = "  |  ".join(
                    f"PLAYER {player} MISSING" for player in missing_players
                )
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    message, font, font_scale, thickness
                )
                text_x = max(8, (width - text_width) // 2)
                text_y = height - 20

                # Dark background keeps the warning readable over the camera image.
                cv2.rectangle(
                    frame,
                    (text_x - 12, text_y - text_height - 10),
                    (min(width - 1, text_x + text_width + 12),
                     min(height - 1, text_y + baseline + 5)),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame, message, (text_x, text_y), font, font_scale,
                    (0, 0, 255), thickness, cv2.LINE_AA
                )

            # FPS estimate.
            current_time = time.perf_counter()
            instant_fps = 1.0 / max(current_time - previous_time, 1e-6)
            fps = instant_fps if fps == 0 else 0.85 * fps + 0.15 * instant_fps
            previous_time = current_time
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, "Q / ESC: quit", (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
