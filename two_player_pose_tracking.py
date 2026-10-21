"""
DEPRECATED: superseded by game/player_tracker.py (built on this file, cut down to what the game needs)
and game/vision.py (owns it inside the fused camera game loop). Kept for reference only; see
game/README.md for the old-to-new mapping. Nothing in game/ imports this file.

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
import warnings

warnings.warn(
    "two_player_pose_tracking.py is superseded by game/player_tracker.py + game/vision.py; see game/README.md",
    DeprecationWarning, stacklevel=2)

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
LOST_PLAYER_SECONDS = 2.0
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


class PlayerIdentityManager:
    """Maps tracker IDs to stable PLAYER 1 / PLAYER 2 display identities."""

    def __init__(self, max_players=2, lost_seconds=2.0):
        self.max_players = max_players
        self.lost_seconds = lost_seconds
        self.track_to_player = {}
        self.player_state = {
            i: {"track_id": None, "center": None, "last_seen": 0.0}
            for i in range(1, max_players + 1)
        }

    def update(self, track_id, center, now):
        # Existing tracker ID: keep its identity.
        if track_id in self.track_to_player:
            player = self.track_to_player[track_id]
            state = self.player_state[player]
            state["center"] = center
            state["last_seen"] = now
            state["track_id"] = track_id
            return player

        # First, reuse a player slot that has been unseen long enough.
        free_players = [
            p for p, s in self.player_state.items()
            if s["track_id"] is None or now - s["last_seen"] > self.lost_seconds
        ]

        # If a slot is free, prefer the slot whose last position is closest.
        if free_players:
            candidates = []
            for p in free_players:
                old_center = self.player_state[p]["center"]
                if old_center is None:
                    distance = float("inf")
                else:
                    distance = float(np.linalg.norm(np.array(center) - np.array(old_center)))
                candidates.append((distance, p))
            # Unused slots (no old center) should be assigned before expired slots.
            unused = [p for p in free_players if self.player_state[p]["center"] is None]
            player = min(unused) if unused else min(candidates)[1]
        else:
            # Both identities are considered active. Avoid assigning a third person.
            return None

        old_track = self.player_state[player]["track_id"]
        if old_track in self.track_to_player:
            del self.track_to_player[old_track]

        self.track_to_player[track_id] = player
        self.player_state[player] = {
            "track_id": track_id,
            "center": center,
            "last_seen": now,
        }
        return player

    def mark_missing(self, visible_track_ids, now):
        """Keep identity slots reserved briefly after a detection disappears."""
        visible_track_ids = set(visible_track_ids)
        for player, state in self.player_state.items():
            tid = state["track_id"]
            if tid is not None and tid not in visible_track_ids:
                # Do not erase immediately; the slot expires in update().
                pass


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

    identities = PlayerIdentityManager(MAX_PLAYERS, LOST_PLAYER_SECONDS)
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

            visible_track_ids = []
            current_players = {}

            boxes = result.boxes
            keypoints = result.keypoints

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
                        # ByteTrack normally provides IDs with persist=True.
                        # Skip untracked detections rather than inventing unstable IDs.
                        continue

                    visible_track_ids.append(int(tid))
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    player = identities.update(int(tid), center, now)
                    if player is None:
                        continue

                    points = all_points[i].astype(float)
                    # Suppress low-confidence points so unreliable joints are not drawn.
                    points[all_conf[i] < 0.35] = np.nan
                    points = smooth_points(smoothed_points.get(player), points, SMOOTHING_ALPHA)
                    smoothed_points[player] = points
                    last_player_points[player] = points.copy()
                    last_player_box[player] = box.copy()
                    last_player_seen[player] = now
                    current_players[player] = (box, points, int(tid))

            identities.mark_missing(visible_track_ids, now)

            # Draw current detections.
            for player, (box, points, tid) in current_players.items():
                color = player_colors[player]
                x1, y1, x2, y2 = np.round(box).astype(int)
                draw_pose(frame, points, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Prefer the nose landmark as the head anchor; fall back to box top-centre.
                if np.isfinite(points[0]).all():
                    label_x = int(points[0, 0])
                    label_y = int(points[0, 1] - 100)
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
                    (max(5, x1), max(22, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
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
