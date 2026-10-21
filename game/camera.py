"""Webcam reader for the game: the newest picture is always ready (own thread), and a 'fast' mode keeps 30 FPS
in a dim room.

Many webcams lower their frame rate in dim light: auto exposure makes every picture longer (we measured 10 FPS
on our laptop in the evening, in every format and size). fast=True fixes the exposure at 30 ms (the longest that
still allows 30 FPS) and raises the gain instead until the picture is about as bright as with auto exposure.
V4L2 settings stay on the camera after a program ends, so release() puts the camera back as it was.

    cam = Camera(0, fast=True)
    frame, n = cam.latest()          # newest picture (None until the first one) and its number
    cam.fps                          # pictures per second the camera really delivers
    cam.release()
"""
import atexit
import threading
import time

import cv2

PROPS = (cv2.CAP_PROP_GAIN, cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_AUTO_EXPOSURE)     # restored in this order


class Camera:
    def __init__(self, index=0, size=(1280, 720), fast=True, target=110):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():                                   # not Linux / no V4L2: default backend
            self.cap = cv2.VideoCapture(index)
        c = self.cap
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # compressed: 30 FPS at 1280x720 on USB 2
        c.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        c.set(cv2.CAP_PROP_FPS, 30)
        # no CAP_PROP_BUFFERSIZE = 1: with a single buffer the camera waits for us and drops to ~18 FPS (measured).
        # The reader thread empties the queue all the time, so the picture is fresh anyway.
        self.saved = {p: c.get(p) for p in PROPS}
        self.fast, self.target, self.gain = fast, target, max(0.0, c.get(cv2.CAP_PROP_GAIN))
        if fast:
            c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)                      # V4L2: 1 = manual, 3 = automatic
            c.set(cv2.CAP_PROP_EXPOSURE, 300)                         # in 100 us units: 30 ms
        self.frame, self.n, self.fps = None, 0, 0.0
        self.ok = c.isOpened()
        self._lock, self._run = threading.Lock(), self.ok
        self._thread = threading.Thread(target=self._loop, daemon=True)
        if self.ok:
            self._thread.start()
        atexit.register(self.release)                                 # also after a crash or Ctrl+C

    def _loop(self):
        last = t_gain = time.perf_counter()
        while self._run:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            rate = 1 / max(1e-3, now - last)
            self.fps = 0.9 * self.fps + 0.1 * rate if self.fps else rate
            last = now
            with self._lock:
                self.frame, self.n = f, self.n + 1
            if self.fast and now - t_gain > 0.5:                      # slowly adjust the gain: picture stays bright
                t_gain = now
                b = cv2.cvtColor(cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).mean()
                g = min(128.0, max(0.0, self.gain + 0.5 * (self.target - b)))
                if abs(self.target - b) > 12 and abs(g - self.gain) >= 1:     # dead band: no constant fiddling
                    self.gain = g
                    self.cap.set(cv2.CAP_PROP_GAIN, g)

    def latest(self):
        with self._lock:
            return self.frame, self.n

    def release(self):
        """Stop and put the camera settings back. Safe to call twice (it is also registered with atexit)."""
        if getattr(self, "_released", False):
            return
        self._released = True
        self._run = False
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        if self.fast and self.ok:
            for p in PROPS:                                           # the camera as we found it
                self.cap.set(p, self.saved[p])
        self.cap.release()
