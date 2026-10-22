"""
Task 2 prototype: hand pose -> smooth, stable control signals (2 hands, real time).

Pipeline per frame:
  webcam -> MediaPipe Hands (21 landmarks per hand)
         -> slot assignment (placeholder for Task 3 identity tracking)
         -> reject unreliable detections (low confidence / sudden jump)
         -> One Euro filter on every landmark (jitter vs lag trade-off)
         -> finger states (wrist distance, rotation-robust) -> raw gesture
         -> hold-to-confirm -> stable gesture
         -> palm velocity (motion signal)
         -> one signal dict per hand, ready for Tasks 3 and 4

Keys:  Esc = quit,  f = toggle smoothing filter (for before/after evidence)
Run:   python task2_hands.py [--camera 0] [--log metrics.csv]
"""
import argparse
import csv
import math
import time

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

# ---------------------------------------------------------------- parameters
FRAME_W, FRAME_H = 640, 480   # low resolution = faster, good enough for hands
MIN_CUTOFF = 1.5              # One Euro: lower = smoother when still (more lag)
BETA = 0.02                   # One Euro: higher = less lag when moving fast
D_CUTOFF = 1.0                # One Euro: cutoff for the speed estimate
EXTEND_RATIO = 1.1            # finger "up" if tip is this much farther from wrist than PIP joint
HOLD_FRAMES = 5               # raw gesture must persist this many frames to be confirmed
GRACE_FRAMES = 10             # keep last state (flagged stale) this long when hand is lost
MIN_CONF = 0.6                # ignore detections below this handedness score
JUMP_FRAC = 0.35              # wrist moving > this fraction of frame width in 1 frame = glitch

# MediaPipe landmark indices
WRIST, PALM = 0, 9
TIPS = [8, 12, 16, 20]        # index, middle, ring, pinky fingertips
PIPS = [6, 10, 14, 18]        # middle joint of each of those fingers

COLORS = [(0, 200, 255), (255, 140, 0)]   # P1, P2 (BGR)
GREY = (150, 150, 150)


# ------------------------------------------------------------- One Euro filter
class OneEuro:
    """Adaptive low-pass: smooth when slow (kills jitter), responsive when fast."""

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# ------------------------------------------------------------ gesture logic
def finger_states(pts):
    """True per finger if extended. Uses distance from wrist, so it works at any hand angle."""
    wx, wy = pts[WRIST]
    states = []
    for tip, pip in zip(TIPS, PIPS):
        d_tip = math.hypot(pts[tip][0] - wx, pts[tip][1] - wy)
        d_pip = math.hypot(pts[pip][0] - wx, pts[pip][1] - wy)
        states.append(d_tip > d_pip * EXTEND_RATIO)
    return states


def classify(states):
    if not any(states):
        return "FIRE"          # fist / rock
    if all(states):
        return "WATER"         # open hand / paper
    if states == [True, True, False, False]:
        return "LIGHTNING"     # index + middle / scissors
    return "?"                 # in-between pose, ignored


# ------------------------------------------------------------- per-hand state
class HandSlot:
    """Holds smoothed state for one hand. Slot 0/1 = placeholder Player 1/2 (Task 3 replaces this)."""

    def __init__(self, idx):
        self.idx = idx
        self.reset()

    def reset(self):
        self.filters = [OneEuro() for _ in range(42)]   # x and y for 21 landmarks
        self.pts = None            # smoothed keypoints (pixels)
        self.raw_wrist = None
        self.conf = 0.0
        self.raw = "NONE"          # unconfirmed gesture this frame
        self.gesture = "NONE"      # confirmed, stable gesture
        self.candidate, self.cand_count = "NONE", 0
        self.velocity = (0.0, 0.0)  # palm velocity, px/s
        self.prev_center, self.prev_t = None, None
        self.missing = 0

    # -- unreliable / missing data: hold last state briefly, then drop to neutral
    def mark_missing(self):
        self.missing += 1
        self.velocity = (self.velocity[0] * 0.5, self.velocity[1] * 0.5)
        if self.missing > GRACE_FRAMES:
            self.reset()

    def update(self, raw_pts, conf, t, use_filter, frame_w):
        if conf < MIN_CONF:
            self.mark_missing()
            return
        if self.pts is not None and self.missing == 0:
            jump = math.hypot(raw_pts[WRIST][0] - self.pts[WRIST][0],
                              raw_pts[WRIST][1] - self.pts[WRIST][1])
            if jump > JUMP_FRAC * frame_w:      # one-frame glitch: treat as missing
                self.mark_missing()
                return

        self.missing = 0
        self.conf = conf
        self.raw_wrist = raw_pts[WRIST]

        if use_filter:
            self.pts = [(self.filters[2 * i](x, t), self.filters[2 * i + 1](y, t))
                        for i, (x, y) in enumerate(raw_pts)]
        else:
            self.pts = list(raw_pts)

        # gesture: raw -> hold-to-confirm
        self.raw = classify(finger_states(self.pts))
        if self.raw != "?":
            if self.raw == self.candidate:
                self.cand_count += 1
            else:
                self.candidate, self.cand_count = self.raw, 1
            if self.cand_count >= HOLD_FRAMES:
                self.gesture = self.candidate

        # motion signal: palm centre velocity, lightly smoothed
        cx = (self.pts[WRIST][0] + self.pts[PALM][0]) / 2
        cy = (self.pts[WRIST][1] + self.pts[PALM][1]) / 2
        if self.prev_center is not None:
            dt = max(t - self.prev_t, 1e-3)
            vx = (cx - self.prev_center[0]) / dt
            vy = (cy - self.prev_center[1]) / dt
            self.velocity = (0.6 * self.velocity[0] + 0.4 * vx,
                             0.6 * self.velocity[1] + 0.4 * vy)
        self.prev_center, self.prev_t = (cx, cy), t

    @property
    def speed(self):
        return math.hypot(*self.velocity)

    def signal(self):
        """What Tasks 3 and 4 should consume. Check 'stale' before triggering any action."""
        return {
            "slot": self.idx,
            "present": self.pts is not None,
            "stale": self.missing > 0,
            "confidence": self.conf,
            "raw_gesture": self.raw,
            "gesture": self.gesture,
            "velocity": self.velocity,
            "speed": self.speed,
            "keypoints": self.pts,
        }


# ----------------------------------------------------- slot assignment (temp)
def assign(detected, slots, frame_w):
    """Placeholder identity logic: left-most hand = slot 0. Task 3 should replace this."""
    if len(detected) >= 2:
        detected = sorted(detected, key=lambda d: d[0][WRIST][0])[:2]
        return {0: detected[0], 1: detected[1]}
    if len(detected) == 1:
        pts, conf = detected[0]
        active = [s for s in slots if s.pts is not None]
        if active:
            s = min(active, key=lambda s: math.hypot(pts[WRIST][0] - s.pts[WRIST][0],
                                                     pts[WRIST][1] - s.pts[WRIST][1]))
            return {s.idx: detected[0]}
        return {(0 if pts[WRIST][0] < frame_w / 2 else 1): detected[0]}
    return {}


# --------------------------------------------------------------------- drawing
def draw_slot(frame, slot):
    if slot.pts is None:
        return
    color = GREY if slot.missing > 0 else COLORS[slot.idx]
    pts = [(int(x), int(y)) for x, y in slot.pts]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 3, color, -1)
    label = "P%d %s%s" % (slot.idx + 1, slot.gesture, " (lost)" if slot.missing else "")
    x, y = pts[WRIST]
    cv2.putText(frame, label, (max(x - 50, 5), max(y - 25, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--log", default=None, help="CSV file for per-frame metrics")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    hands = mp_hands.Hands(max_num_hands=2, model_complexity=0,
                           min_detection_confidence=0.7, min_tracking_confidence=0.5)
    slots = [HandSlot(0), HandSlot(1)]

    use_filter = True
    fps, lat_ema, latencies = 0.0, 0.0, []
    prev_t, frame_i = None, 0

    log_f, writer = None, None
    if args.log:
        log_f = open(args.log, "w", newline="")
        writer = csv.writer(log_f)
        writer.writerow(["t", "frame", "slot", "present", "stale", "conf", "raw_gesture",
                         "gesture", "wrist_raw_x", "wrist_raw_y", "wrist_sm_x", "wrist_sm_y",
                         "speed_px_s", "latency_ms", "filter_on"])

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)

            detected = []
            if res.multi_hand_landmarks:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    pts = [(p.x * w, p.y * h) for p in lm.landmark]
                    detected.append((pts, hd.classification[0].score))

            assigned = assign(detected, slots, w)
            for s in slots:
                if s.idx in assigned:
                    pts, conf = assigned[s.idx]
                    s.update(pts, conf, t0, use_filter, w)
                elif s.pts is not None:
                    s.mark_missing()

            # ---- draw + HUD
            for s in slots:
                draw_slot(frame, s)

            lat = (time.perf_counter() - t0) * 1000.0   # processing time only
            latencies.append(lat)
            lat_ema = lat if lat_ema == 0 else 0.9 * lat_ema + 0.1 * lat
            if prev_t is not None:
                inst = 1.0 / max(t0 - prev_t, 1e-6)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            prev_t = t0

            cv2.putText(frame, "FPS %.0f  proc %.1f ms  filter %s" %
                        (fps, lat_ema, "ON" if use_filter else "OFF"),
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for s in slots:
                sig = s.signal()
                txt = "P%d: %s (raw %s)  v=%4.0f px/s  conf %.2f" % (
                    s.idx + 1, sig["gesture"], sig["raw_gesture"], sig["speed"], sig["confidence"])
                cv2.putText(frame, txt, (10, 55 + 25 * s.idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[s.idx], 2)

            if writer:
                for s in slots:
                    sig = s.signal()
                    sm = s.pts[WRIST] if s.pts else (None, None)
                    rw = s.raw_wrist if s.raw_wrist else (None, None)
                    writer.writerow([round(t0, 4), frame_i, s.idx, int(sig["present"]),
                                     int(sig["stale"]), round(sig["confidence"], 3),
                                     sig["raw_gesture"], sig["gesture"], rw[0], rw[1],
                                     sm[0], sm[1], round(sig["speed"], 1), round(lat, 2),
                                     int(use_filter)])
            frame_i += 1

            cv2.imshow("Task 2 - hand control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("f"):
                use_filter = not use_filter
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        if log_f:
            log_f.close()
        if latencies:
            srt = sorted(latencies)
            print("frames: %d | mean proc latency: %.1f ms | p95: %.1f ms | last FPS: %.0f" %
                  (len(srt), sum(srt) / len(srt), srt[int(0.95 * (len(srt) - 1))], fps))


if __name__ == "__main__":
    main()
