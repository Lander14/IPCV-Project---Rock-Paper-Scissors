import itertools
import math
import os
import time
from dataclasses import dataclass
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL = Path(r"C:\cvdata\face_landmarker.task")

CAMERA_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720

EXPECTED_FACES = 2
MAX_FACES = 2

DETECTION_CONFIDENCE = 0.4
PRESENCE_CONFIDENCE = 0.4
TRACKING_CONFIDENCE = 0.4

ACCEL_STILL = 300.0
ACCEL_FAST = 5000.0
ACCEL_SIZE = 400.0
ROLL_ACCEL_STILL = 150.0
ROLL_ACCEL_FAST = 8000.0

MEAS_NOISE_FRAC = 0.025
MEAS_NOISE_MIN = 1.5
ROLL_MEAS_NOISE = 1.0

MIN_HITS = 2
LOST_TIMEOUT = 0.8
COAST_DAMPING = 0.85
GATE_FACTOR = 1.2

ROI_SCALE = 6.0
ROI_MIN_SIZE = 160
SEARCH_EVERY = 3
SEARCH_OVERLAP = 0.15

APPEARANCE_WEIGHT = 1.0
APPEARANCE_RATE = 0.05
HIST_BINS = (16, 16)

CIRCLE_MARGIN = 1.10
EYE_A, EYE_B = 33, 263

MAX_HP = 100
TEST_DAMAGE = 10
LOW_HP = 0.25
HIT_FLASH = 0.3

GLASSES = {1: ("sunglasses_picture.jpg", 0.49), 2: ("sunglasses_picture2.jpg", 0.45)}
GLASSES_WIDTH = 1.6
EYE_SMOOTHING = 0.3

TRACK_COLORS = [(255, 120, 0), (0, 140, 255)]


@dataclass
class Detection:
    cx: float
    cy: float
    r: float
    roll: float
    hist: np.ndarray = None
    eyes: tuple = None


def face_histogram(rgb, cx, cy, r):
    h, w = rgb.shape[:2]
    half = int(0.8 * r)
    x0, x1 = max(0, int(cx) - half), min(w, int(cx) + half)
    y0, y1 = max(0, int(cy) - half), min(h, int(cy) + half)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    return hist / max(hist.sum(), 1e-6)


def appearance_distance(h1, h2):
    if h1 is None or h2 is None:
        return 0.5
    return cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32),
                           cv2.HISTCMP_BHATTACHARYYA)


def detection_from_landmarks(landmarks, x0, y0, w, h, rgb):
    pts = np.array([[x0 + p.x * w, y0 + p.y * h] for p in landmarks],
                   dtype=np.float32)

    (cx, cy), radius = cv2.minEnclosingCircle(pts)

    a, b = pts[EYE_A], pts[EYE_B]
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    roll = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))

    r = radius * CIRCLE_MARGIN
    mx, my = (left[0] + right[0]) / 2 - cx, (left[1] + right[1]) / 2 - cy
    c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    eyes = ((mx * c + my * s) / r, (-mx * s + my * c) / r,
            math.hypot(right[0] - left[0], right[1] - left[1]) / r)
    return Detection(cx, cy, r, roll, face_histogram(rgb, cx, cy, r), eyes)


class FaceDetector:
    def __init__(self, model_path):
        vision = mp.tasks.vision

        def make(mode):
            return vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mode,
                    num_faces=MAX_FACES,
                    min_face_detection_confidence=DETECTION_CONFIDENCE,
                    min_face_presence_confidence=PRESENCE_CONFIDENCE,
                    min_tracking_confidence=TRACKING_CONFIDENCE,
                )
            )

        self.video = make(vision.RunningMode.VIDEO)
        self.image = make(vision.RunningMode.IMAGE)
        self.start = time.monotonic()
        self.last_ts = -1
        self.rgb = None

    def detect_full(self, rgb):
        self.rgb = rgb
        h, w = rgb.shape[:2]

        ts = int((time.monotonic() - self.start) * 1000)
        ts = max(ts, self.last_ts + 1)
        self.last_ts = ts

        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        result = self.video.detect_for_video(image, ts)
        return [detection_from_landmarks(f, 0, 0, w, h, rgb)
                for f in result.face_landmarks]

    def detect_region(self, x0, y0, x1, y1):
        crop = np.ascontiguousarray(self.rgb[y0:y1, x0:x1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        result = self.image.detect(image)
        return [detection_from_landmarks(f, x0, y0, x1 - x0, y1 - y0, self.rgb)
                for f in result.face_landmarks]

    def close(self):
        self.video.close()
        self.image.close()


class FaceKalman:
    N = 4
    H = np.hstack([np.eye(N), np.zeros((N, N))])
    MOTION_DIMS = [0, 1, 3]

    def __init__(self, det):
        self.x = np.array([det.cx, det.cy, det.r, det.roll, 0, 0, 0, 0], float)
        s = self._meas_std()

        self.P = np.diag([v**2 for v in s] + [800.0**2, 800.0**2, 200.0**2, 200.0**2])
        self.motion = 1.0

    def _meas_std(self):
        s = max(MEAS_NOISE_MIN, MEAS_NOISE_FRAC * self.x[2])
        return [s, s, 1.5 * s, ROLL_MEAS_NOISE]

    def predict(self, dt):
        n = self.N
        F = np.eye(2 * n)
        F[:n, n:] = np.eye(n) * dt

        m = self.motion
        q = [ACCEL_STILL + m * (ACCEL_FAST - ACCEL_STILL),
             ACCEL_STILL + m * (ACCEL_FAST - ACCEL_STILL),
             ACCEL_SIZE,
             ROLL_ACCEL_STILL + m * (ROLL_ACCEL_FAST - ROLL_ACCEL_STILL)]
        Q = np.zeros((2 * n, 2 * n))
        for i in range(n):
            q2 = q[i] ** 2
            Q[i, i] = q2 * dt**4 / 4
            Q[i, i + n] = Q[i + n, i] = q2 * dt**3 / 2
            Q[i + n, i + n] = q2 * dt**2

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, det):
        z = np.array([det.cx, det.cy, det.r, det.roll])
        R = np.diag(np.array(self._meas_std()) ** 2)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R

        d_idx = self.MOTION_DIMS
        d = math.sqrt(y[d_idx] @ np.linalg.inv(S[np.ix_(d_idx, d_idx)]) @ y[d_idx])
        level = min(1.0, max(0.0, (d - 2.8) / 2.5))
        self.motion = level if level > self.motion else 0.85 * self.motion + 0.15 * level

        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2 * self.N) - K @ self.H) @ self.P

    def damp_velocity(self, factor):
        self.x[self.N:] *= factor

    @property
    def center(self):
        return self.x[0], self.x[1]

    @property
    def radius(self):
        return max(5.0, self.x[2])

    @property
    def roll(self):
        return self.x[3]

    @property
    def speed(self):
        return math.hypot(self.x[4], self.x[5])


class FaceTrack:
    def __init__(self, track_id, det, t):
        self.id = track_id
        self.kf = FaceKalman(det)
        self.hist = det.hist
        self.new_hist = None
        self.eyes = np.array(det.eyes, float)
        self.hits = 1
        self.missing_frames = 0
        self.last_seen = t
        self.updated = True

    def predict(self, dt):
        self.kf.predict(dt)
        self.updated = False

    def update(self, det, t):
        self.kf.update(det)
        self.new_hist = det.hist
        self.eyes = (1 - EYE_SMOOTHING) * self.eyes + EYE_SMOOTHING * np.array(det.eyes, float)
        self.hits += 1
        self.missing_frames = 0
        self.last_seen = t
        self.updated = True

    def learn_appearance(self):
        if self.new_hist is not None:
            if self.hist is None:
                self.hist = self.new_hist
            else:
                self.hist = (1 - APPEARANCE_RATE) * self.hist + APPEARANCE_RATE * self.new_hist
        self.new_hist = None

    def mark_missed(self):
        self.missing_frames += 1
        self.kf.damp_velocity(COAST_DAMPING)

    @property
    def confirmed(self):
        return self.hits >= MIN_HITS

    def gate(self, det, dt):
        return (GATE_FACTOR * (self.kf.radius + det.r)
                + self.kf.speed * dt
                + 10 * self.missing_frames)


def optimal_assignment(cost, max_cost):
    n_t, n_d = cost.shape
    if n_t == 0 or n_d == 0:
        return []

    if n_t <= n_d:
        combos = [[(i, perm[i]) for i in range(n_t)]
                  for perm in itertools.permutations(range(n_d), n_t)]
    else:
        combos = [[(perm[j], j) for j in range(n_d)]
                  for perm in itertools.permutations(range(n_t), n_d)]

    best, best_pairs = None, []
    for combo in combos:
        pairs = [(i, j) for i, j in combo if cost[i, j] < max_cost]
        total = sum(cost[i, j] for i, j in pairs) + max_cost * (len(combo) - len(pairs))
        if best is None or total < best:
            best, best_pairs = total, pairs
    return best_pairs


def is_duplicate(det, others):
    return any(math.hypot(det.cx - o.cx, det.cy - o.cy) < max(det.r, o.r)
               for o in others)


class FaceTracker:
    def __init__(self):
        self.tracks = []
        self.memory = {}

    def _id_cost(self, pid, det, width):
        hist, x = self.memory.get(pid, (None, (pid - 0.5) / MAX_FACES * width))
        return appearance_distance(hist, det.hist) + abs(det.cx - x) / width

    def step(self, detections, dt, t, frame_size, detect_region=None, frame_index=0):
        width, height = frame_size

        for tr in self.tracks:
            tr.predict(dt)

        unmatched = self._associate(self.tracks, detections, dt, t)
        used = [d for d in detections if d not in unmatched]

        if detect_region is not None:
            for tr in self.tracks:
                if tr.updated:
                    continue
                cx, cy = tr.kf.center
                if not (0 <= cx < width and 0 <= cy < height):
                    continue
                half = max(ROI_MIN_SIZE, ROI_SCALE * tr.kf.radius) / 2
                x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
                x1, y1 = int(min(width, cx + half)), int(min(height, cy + half))
                if x1 - x0 < 32 or y1 - y0 < 32:
                    continue

                found = [d for d in detect_region(x0, y0, x1, y1)
                         if not is_duplicate(d, used)]
                if found:
                    d = min(found, key=lambda d: math.hypot(d.cx - cx, d.cy - cy))
                    if math.hypot(d.cx - cx, d.cy - cy) < tr.gate(d, dt):
                        tr.update(d, t)
                        used.append(d)

        if (detect_region is not None
                and len(self.tracks) + len(unmatched) < EXPECTED_FACES
                and frame_index % SEARCH_EVERY == 0):
            overlap = int(SEARCH_OVERLAP * width)
            for x0, x1 in [(0, width // 2 + overlap), (width // 2 - overlap, width)]:
                for d in detect_region(x0, 0, x1, height):
                    if is_duplicate(d, used + unmatched):
                        continue
                    lost = [tr for tr in self.tracks if not tr.updated]
                    if self._associate(lost, [d], dt, t):
                        unmatched.append(d)
                    else:
                        used.append(d)

        new = sorted(unmatched, key=lambda d: -d.r)[:MAX_FACES - len(self.tracks)]
        free = [i for i in range(1, MAX_FACES + 1) if i not in {tr.id for tr in self.tracks}]
        if new and free:
            cost = np.array([[self._id_cost(i, d, width) for d in new] for i in free])
            for i, j in optimal_assignment(cost, 1e9):
                self.tracks.append(FaceTrack(free[i], new[j], t))

        for tr in self.tracks:
            if not tr.updated:
                tr.mark_missed()
        kept = []
        for tr in self.tracks:
            if t - tr.last_seen <= LOST_TIMEOUT and (tr.confirmed or tr.missing_frames == 0):
                kept.append(tr)
            elif tr.confirmed:
                self.memory[tr.id] = (tr.hist, tr.kf.center[0])
        self.tracks = kept

        for tr in self.tracks:
            overlapping = any(
                o is not tr and math.hypot(o.kf.center[0] - tr.kf.center[0],
                                           o.kf.center[1] - tr.kf.center[1])
                < o.kf.radius + tr.kf.radius
                for o in self.tracks)
            if overlapping:
                tr.new_hist = None
            tr.learn_appearance()

        return [tr for tr in self.tracks if tr.confirmed]

    def _associate(self, tracks, detections, dt, t):
        if not tracks or not detections:
            return list(detections)

        big = 1e9
        cost = np.full((len(tracks), len(detections)), big)
        for i, tr in enumerate(tracks):
            px, py = tr.kf.center
            for j, d in enumerate(detections):
                dist = math.hypot(d.cx - px, d.cy - py)
                if dist < tr.gate(d, dt):
                    cost[i, j] = dist + APPEARANCE_WEIGHT * tr.kf.radius * \
                        appearance_distance(tr.hist, d.hist)

        pairs = optimal_assignment(cost, big)
        for i, j in pairs:
            tracks[i].update(detections[j], t)
        matched = {j for _, j in pairs}
        return [d for j, d in enumerate(detections) if j not in matched]


def new_game_state():
    return {pid: {"hp": MAX_HP, "max_hp": MAX_HP, "last_hit": -1.0}
            for pid in range(1, MAX_FACES + 1)}


def damage(game_state, pid, amount, t):
    p = game_state[pid]
    if p["hp"] > 0 and amount > 0:
        p["hp"] = max(0, p["hp"] - amount)
        p["last_hit"] = t


def make_transparent(img):
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    alpha = np.clip((sat - 30) / 50, 0, 1)
    solid = cv2.morphologyEx((alpha > 0.5).astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid)
    if n < 2:
        raise ValueError("No glasses found in the image")
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    alpha[labels != biggest] = 0
    ys, xs = np.where(alpha > 0.05)
    img = np.dstack([img, (alpha * 255).astype(np.uint8)])
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def load_glasses():
    glasses = {}
    for pid, (name, eye_y) in GLASSES.items():
        path = Path(__file__).with_name(name)
        if not path.exists():
            raise FileNotFoundError(f"Glasses image not found: {path}")
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read glasses image: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            img = make_transparent(img)
        gh, gw = img.shape[:2]
        half = gw / (2 * GLASSES_WIDTH)
        src = np.float32([[gw / 2 - half, eye_y * gh], [gw / 2 + half, eye_y * gh]])
        glasses[pid] = (img, src)
    return glasses


def draw_sunglasses(frame, tracks, glasses):
    height, width = frame.shape[:2]
    for tr in tracks:
        if tr.id not in glasses:
            continue
        img, src = glasses[tr.id]
        cx, cy = tr.kf.center
        r = tr.kf.radius
        c, s = math.cos(math.radians(tr.kf.roll)), math.sin(math.radians(tr.kf.roll))
        ox, oy, d = tr.eyes

        mx = cx + r * (ox * c - oy * s)
        my = cy + r * (ox * s + oy * c)
        hx, hy = 0.5 * d * r * c, 0.5 * d * r * s
        dst = np.float32([[mx - hx, my - hy], [mx + hx, my + hy]])

        M, _ = cv2.estimateAffinePartial2D(src, dst)
        if M is None:
            continue

        gh, gw = img.shape[:2]
        corners = cv2.transform(np.float32([[[0, 0], [gw, 0], [0, gh], [gw, gh]]]), M)[0]
        x0, y0 = np.floor(corners.min(axis=0)).astype(int)
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            continue

        M[:, 2] -= (x0, y0)
        warped = cv2.warpAffine(img, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        roi = frame[y0:y1, x0:x1]
        roi[:] = (alpha * warped[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)


def draw_hp_bars(frame, tracks, game_state, t):
    height, width = frame.shape[:2]
    for tr in tracks:
        player = game_state.get(tr.id)
        if player is None:
            continue
        cx, cy = tr.kf.center
        r = tr.kf.radius
        frac = player["hp"] / player["max_hp"]

        bar_w = int(min(300, max(60, 1.6 * r)))
        bar_h = int(min(22, max(6, 0.12 * r)))
        x0 = int(min(max(cx - bar_w / 2, 0), width - bar_w))
        y0 = int(cy - 1.25 * r - bar_h)
        if y0 < 60:
            y0 = int(min(cy + 1.25 * r + 20, height - bar_h - 2))

        if frac > 0.5:
            fill = (0, 200, 0)
        elif frac > LOW_HP:
            fill = (0, 220, 255)
        else:
            fill = (0, 0, 255)

        if 0 <= t - player["last_hit"] < HIT_FLASH:
            cv2.rectangle(frame, (x0 - 3, y0 - 3), (x0 + bar_w + 3, y0 + bar_h + 3),
                          (255, 255, 255), -1)

        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * frac), y0 + bar_h), fill, -1)
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (0, 0, 0), 1)

        color = TRACK_COLORS[(tr.id - 1) % len(TRACK_COLORS)]
        text = f"P{tr.id}  KO" if player["hp"] <= 0 else f"P{tr.id}  {player['hp']}/{player['max_hp']}"
        cv2.putText(frame, text, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if player["hp"] <= 0 else color, 2)


def draw_text(frame, text, y):
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def open_camera():
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def hide_mediapipe_logs():
    sys.stderr.flush()
    python_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    sys.stderr = os.fdopen(python_stderr, "w", buffering=1)


def main():
    hide_mediapipe_logs()
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    camera = open_camera()
    detector = FaceDetector(MODEL)
    tracker = FaceTracker()
    game_state = new_game_state()
    glasses = load_glasses()

    fps = 0.0
    prev_t = time.perf_counter()
    frame_index = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Failed to capture frame")
                break

            frame_start = time.perf_counter()
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            dt = min(max(frame_start - prev_t, 1 / 120), 0.2)
            prev_t = frame_start

            detections = detector.detect_full(rgb)
            tracks = tracker.step(detections, dt, frame_start, (width, height),
                                  detector.detect_region, frame_index)
            frame_index += 1

            draw_sunglasses(frame, tracks, glasses)
            draw_hp_bars(frame, tracks, game_state, frame_start)

            process_ms = (time.perf_counter() - frame_start) * 1000
            fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
            draw_text(frame, f"Faces: {len(tracks)}   FPS: {fps:4.1f}   "
                             f"Processing: {process_ms:5.1f} ms", 30)

            cv2.imshow("IPCV - Face Tracking v3", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("1"), ord("2")):
                damage(game_state, key - ord("0"), TEST_DAMAGE, frame_start)
            if key in (ord("r"), ord("R")):
                game_state = new_game_state()

    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
