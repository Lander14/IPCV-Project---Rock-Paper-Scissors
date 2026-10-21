"""Visual feedback of the rock-paper-scissors duel (T4-07 / T4-08), drawn ON THE CAMERA IMAGE (augmented reality).

    hud = Hud()
    events = game.update(t_ms, inputs)
    snap = game.snapshot()
    hud.add(events, t_ms, snap)
    hud.draw(frame, snap, t_ms)                 # frame = the webcam image; positions come from the game state

Draws the three lanes between the players, the rock / paper / scissors travelling along them (with a glow),
shields as barriers in their lane, a ring on each casting hand with the recognised gesture, impacts, HP / mana,
the countdown and a short text feed of what happened.
"""
import math
import random
import time
from collections import defaultdict

import cv2
import numpy as np

from game_logic import (FIGHT, LOBBY, MATCH_OVER, PAPER, PAUSED, POTION, RESUME, ROCK, ROUND_END, ROUND_START,
                        SCISSORS, SHIELD)

PLAYER_COL = {1: (230, 110, 250), 2: (110, 230, 110), 3: (240, 210, 70), 4: (70, 210, 240)}   # BGR: pink, green, ...
SPELL_COL = {ROCK: (60, 150, 240), PAPER: (255, 255, 255), SCISSORS: (255, 210, 90),
             SHIELD: (255, 225, 150), POTION: (120, 230, 120)}
FONT = cv2.FONT_HERSHEY_DUPLEX
SHOW_MS, FEED_MS = 1300, 3500        # impact texts; messages in the feed
_ROCK = np.array([1.0, 0.82, 0.95, 0.78, 0.93, 1.0, 0.85, 0.97, 0.8, 0.9])


# ------------------------------------------------------------------ drawing helpers
def text(img, s, org, scale, col, thick=2, center=False):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = int(org[0] - w / 2) if center else int(org[0]), int(org[1])
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, col, thick, cv2.LINE_AA)
    return w, h


def _u(a):
    return np.array([math.cos(a), math.sin(a)])


def draw_rock(img, c, r, ang=0.0):
    c = np.asarray(c, float)
    th = ang + np.linspace(0, 2 * np.pi, len(_ROCK), endpoint=False)
    pts = np.stack([c[0] + r * _ROCK * np.cos(th), c[1] + r * _ROCK * np.sin(th)], 1)
    cv2.fillPoly(img, [pts.astype(np.int32)], (105, 115, 128), cv2.LINE_AA)
    hi = (pts - c) * 0.55 + c - np.array([0.18 * r, 0.22 * r])
    cv2.fillPoly(img, [hi.astype(np.int32)], (150, 160, 172), cv2.LINE_AA)
    for k in (0.4, 2.6, 4.4):                                            # cracks turn with the rock
        p0, p1 = c + 0.15 * r * _u(ang + k), c + 0.8 * r * _u(ang + k + 0.35)
        cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), (55, 60, 68), max(1, int(r / 10)), cv2.LINE_AA)
    cv2.polylines(img, [pts.astype(np.int32)], True, (45, 50, 58), max(1, int(r / 11)), cv2.LINE_AA)


def draw_paper(img, c, r, t=0.0):
    ang = 14 * math.sin(t * 7)                                           # the sheet flutters
    box = cv2.boxPoints(((float(c[0]), float(c[1])), (1.5 * r, 1.15 * r), ang))
    cv2.fillPoly(img, [box.astype(np.int32)], (245, 245, 240), cv2.LINE_AA)
    a, b, cc, d = box
    for f in (0.3, 0.5, 0.7):                                            # lines of writing
        p0, p1 = a + (d - a) * f + (b - a) * 0.15, a + (d - a) * f + (b - a) * 0.8
        cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), (175, 175, 175), max(1, int(r / 14)), cv2.LINE_AA)
    fold = np.array([cc, cc + (b - cc) * 0.3, cc + (d - cc) * 0.3])
    cv2.fillPoly(img, [fold.astype(np.int32)], (205, 205, 200), cv2.LINE_AA)
    cv2.polylines(img, [box.astype(np.int32)], True, (120, 120, 120), max(1, int(r / 14)), cv2.LINE_AA)


def draw_scissors(img, c, r, t=0.0, d=1):
    c = np.asarray(c, float)
    base = 0.0 if d > 0 else math.pi                                     # blades point where it flies
    opening = math.radians(14 + 16 * abs(math.sin(t * 9)))             # snip, snip
    for s in (1, -1):
        a = base + s * opening / 2
        tip, back, perp = c + 1.15 * r * _u(a), c - 0.2 * r * _u(a), _u(a + math.pi / 2)
        blade = np.array([tip, c + perp * 0.13 * r, back, c - perp * 0.13 * r])
        cv2.fillPoly(img, [blade.astype(np.int32)], (215, 215, 222), cv2.LINE_AA)
        cv2.polylines(img, [blade.astype(np.int32)], True, (110, 110, 120), max(1, int(r / 14)), cv2.LINE_AA)
        ring = c - 0.65 * r * _u(a)                                      # handle on the other side of the pivot
        cv2.circle(img, tuple(ring.astype(int)), int(0.27 * r), (40, 40, 210), max(2, int(r / 7)), cv2.LINE_AA)
    cv2.circle(img, tuple(c.astype(int)), max(2, int(r / 9)), (70, 70, 70), -1, cv2.LINE_AA)


def icon(img, action, c, r, t=0.0, d=1):
    """Small symbol of an action (nothing for None)."""
    if action == ROCK:
        draw_rock(img, c, r)
    elif action == PAPER:
        draw_paper(img, c, r * 0.9, t)
    elif action == SCISSORS:
        draw_scissors(img, c, r, t, d)
    elif action == SHIELD:
        cv2.circle(img, (int(c[0]), int(c[1])), int(r), SPELL_COL[SHIELD], max(2, int(r / 5)), cv2.LINE_AA)
    elif action == POTION:
        x, y = c
        cv2.circle(img, (int(x), int(y + 0.35 * r)), int(0.65 * r), (90, 200, 90), -1, cv2.LINE_AA)
        cv2.rectangle(img, (int(x - 0.2 * r), int(y - r)), (int(x + 0.2 * r), int(y - 0.2 * r)), (90, 200, 90), -1)
        cv2.rectangle(img, (int(x - 0.3 * r), int(y - r)), (int(x + 0.3 * r), int(y - 0.8 * r)), (60, 90, 140), -1)


def bar(img, x, y, w, h, frac, col, back=(40, 40, 40)):
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), back, -1)
    cv2.rectangle(img, (int(x), int(y)), (int(x + w * max(0.0, min(1.0, frac))), int(y + h)), col, -1)
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1)


def shade(img, x0, y0, x1, y1, alpha=0.45, col=(0, 0, 0)):
    """Darken (or tint) a rectangle of the camera image, in place, so text and bars stay readable.
    8-bit OpenCV maths: ~1 ms for a 1280x720 frame (the numpy float version took 13 ms)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    if col == (0, 0, 0) and (x0, y0, x1, y1) == (0, 0, W, H) and img.flags["C_CONTIGUOUS"]:
        cv2.convertScaleAbs(img, img, 1 - alpha)                         # whole frame: in place
    elif col == (0, 0, 0):
        img[y0:y1, x0:x1] = cv2.convertScaleAbs(img[y0:y1, x0:x1], alpha=1 - alpha)
    else:
        roi = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, col), alpha, 0)


class Glow:
    """Light layer at half resolution. What is drawn here is blurred and ADDED to the picture: a cheap glow.
    Coordinates and thicknesses are given in full-resolution pixels."""
    F = 0.5

    def __init__(self, w, h):
        self.w, self.h, self.sc, self.used = w, h, h / 720, False
        self.img = np.zeros((int(h * self.F), int(w * self.F), 3), np.uint8)

    def _p(self, p):
        return int(p[0] * self.F), int(p[1] * self.F)

    def line(self, p0, p1, col, th):
        self.used = True
        cv2.line(self.img, self._p(p0), self._p(p1), col, max(1, int(th * self.F)), cv2.LINE_AA)

    def circle(self, c, r, col, th):
        self.used = True
        cv2.circle(self.img, self._p(c), max(1, int(r * self.F)), col, max(1, int(th * self.F)), cv2.LINE_AA)

    def add_to(self, img, sigma=3):
        if not self.used:
            return
        g = self.img
        blur = cv2.GaussianBlur(cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), (0, 0), sigma)
        mix = cv2.addWeighted(cv2.resize(blur, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR), 1.0, g, 0.5, 0)
        cv2.add(img, cv2.resize(mix, (self.w, self.h), interpolation=cv2.INTER_LINEAR), img)


def hp_col(f):
    return (60, 200, 60) if f > 0.5 else (40, 200, 230) if f > 0.25 else (50, 50, 230)


def sentence(e, n=lambda pid: f"P{pid}"):
    """One line of the text feed for an event (None = not worth a line). n(pid) = short player name."""
    k, a = e["type"], (e.get("action") or "").lower()
    if k == "hit":
        return f"{a.title()} hits {n(e['target'])}: -{e['amount']}"
    if k == "blocked":
        return f"{n(e['target'])}'s shield stops the {a}"
    if k == "counter":
        return f"{e['why']}: {n(e['player'])} counters {n(e['target'])}"
    if k == "clash":
        return f"{a.title()} meets {a}: both gone"
    if k == "heal":
        return f"{n(e['player'])} drinks a potion: +{e['amount']} HP"
    if k == "invalid":
        return f"{n(e['player'])}: {e['reason']}"
    if k == "ko":
        return f"{n(e['player'])} is K.O.!"
    return None


# ------------------------------------------------------------------ the HUD
class Hud:
    def __init__(self, names=None, profile=False):
        self.names = names or {}         # e.g. {2: "CPU"} in single-player mode
        self.prof = defaultdict(list) if profile else None     # ms per drawing step, for profiling (T5)
        self._lanes_key, self._lane_segs = None, []
        self.fx = []                     # animations: (start_ms, end_ms, glow_fn, draw_fn); fn(img, u), u = 0..1
        self.feed = []                   # (t_ms, text) of the last messages

    def short(self, pid):
        return self.names.get(pid, f"P{pid}")

    def long(self, pid):
        return self.names.get(pid, f"PLAYER {pid}").upper()

    def _fx(self, t0, dur, glow=None, draw=None):
        self.fx.append((t0, t0 + dur, glow, draw))

    # ---------------------------------------------------------------- events -> animations
    def add(self, events, t_ms, snap):
        def body(pid):
            b = snap["players"][pid]["body_xy"]
            return None if b is None else np.array(b, float)

        slot = {}                                                 # texts on the same spot are stacked

        def stack(key):
            slot[key] = slot.get(key, -1) + 1
            return slot[key]

        for e in events:
            k, a = e["type"], e.get("action")
            s = sentence(e, self.short)
            if s:
                self.feed = (self.feed + [(t_ms, s)])[-3:]
            xy = np.array(e["xy"], float) if e.get("xy") is not None else None
            if k == "cast" and xy is not None:                    # a flash where the spell leaves the hand
                self._fx(t_ms, 350, glow=lambda g, u, p=xy, a=a: g.circle(
                    p, (20 + 70 * u) * g.sc, SPELL_COL[a], max(2, 10 * (1 - u))))
            elif k == "hit":
                self._fx(t_ms, 500, glow=lambda g, u, p=xy: g.circle(
                    p, (30 + 110 * u) * g.sc, (60, 60, 255), max(2, 16 * (1 - u))))
                self._float(t_ms, xy, f"-{e['amount']}", (80, 80, 255), 1.6, k=stack(("p", e["target"])))
            elif k in ("blocked", "counter", "clash"):
                col = SPELL_COL[SHIELD] if k == "blocked" else SPELL_COL[a]
                self._fx(t_ms, 600, glow=lambda g, u, p=xy, col=col: self._burst(g.line, g.sc, p, u, col, 1.4),
                         draw=lambda img, u, p=xy: self._burst(
                             lambda a, b, c, t: cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), c,
                                                         max(1, int(t)), cv2.LINE_AA), img.shape[0] / 720,
                             p, u, (255, 255, 255), 0.6))
                label = {"blocked": "BLOCKED", "clash": "CLASH", "counter": e.get("why", "").upper()}[k]
                self._float(t_ms, xy, label, (255, 255, 255), 0.9, k=stack(("xy", tuple(xy.astype(int)))))
            elif k in ("heal", "invalid", "ko") and body(e["player"]) is not None:
                p = body(e["player"])
                kk = stack(("p", e["player"]))
                if k == "heal":
                    self._float(t_ms, p, f"+{e['amount']} HP", (120, 240, 120), 1.3, k=kk)
                elif k == "invalid":
                    self._float(t_ms, p, e["reason"], (60, 170, 255), 0.75, k=kk)
                else:
                    self._float(t_ms, p, "K.O.", (40, 40, 255), 2.5, dur=2500, k=kk)

    def _float(self, t0, p, s, col, scale, dur=SHOW_MS, k=0):
        """Text that rises above a point; the k-th text on the same spot is drawn k lines higher."""
        def fn(img, u):
            sc = img.shape[0] / 720
            text(img, s, (p[0], p[1] - (55 + 50 * k) * sc - 35 * sc * u), scale * sc, col, max(2, int(3 * sc)),
                 center=True)
        self._fx(t0, dur, draw=fn)

    @staticmethod
    def _burst(line, sc, p, u, col, width):
        """12 rays flying out of a point; line(p0, p1, colour, thickness) draws one ray."""
        for k in range(12):
            d = _u(2 * math.pi * k / 12)
            line(p + d * (15 + 70 * u) * sc, p + d * (40 + 120 * u) * sc, col, max(1, 6 * width * sc * (1 - u)))

    # ---------------------------------------------------------------- per frame
    def draw(self, img, snap, t_ms):
        H, W = img.shape[:2]
        sc = H / 720
        ph = snap["phase"]
        tp = [time.perf_counter()]

        def mark(name):                                            # profiling checkpoints (Hud(profile=True))
            if self.prof is not None:
                now = time.perf_counter()
                self.prof[name].append((now - tp[0]) * 1000)
                tp[0] = now

        if ph in (PAUSED, MATCH_OVER):
            shade(img, 0, 0, W, H, 0.5)
        elif ph == ROUND_END:
            shade(img, 0, 0, W, H, 0.3)
        players = snap["players"]
        xs = sorted(p["body_xy"][0] for p in players.values() if p["body_xy"] is not None)
        x0, x1 = (xs[0] + 0.05 * W, xs[-1] - 0.05 * W) if len(xs) >= 2 else (0.15 * W, 0.85 * W)
        fighting = ph in (FIGHT, ROUND_START, RESUME)
        active = [(t0, t1, g, d) for t0, t1, g, d in self.fx if t0 <= t_ms < t1]
        self.fx = [f for f in self.fx if t_ms < f[1]]

        # 1. the lanes, painted on the camera image
        half = snap["lane_half"]
        key = (int(x0) // 16, int(x1) // 16, tuple(int(y) for y in snap["lanes"]), int(half), W)
        if key != self._lanes_key:                                 # dashed borders: computed once, drawn in one call
            self._lanes_key, self._lane_segs = key, [
                np.array([[xa, yy], [min(xa + 18 * sc, x1), yy]], np.int32)
                for y in snap["lanes"] for yy in (y - half, y + half) for xa in np.arange(x0, x1, 34 * sc)]
        for y in snap["lanes"]:
            shade(img, x0, y - half, x1, y + half, 0.22)
        cv2.polylines(img, self._lane_segs, False, (235, 235, 235), 1, cv2.LINE_AA)
        mark("lanes")

        # 2. light: everything that glows goes into one layer (half resolution), blurred and added to the image
        glow = Glow(W, H)
        if fighting:
            for pid, p in players.items():                         # the lane each player is aiming at
                if p["present"] and p["alive"] and p["lane"] is not None and p["body_xy"] is not None:
                    y = snap["lanes"][p["lane"]]
                    a = x0 if p["body_xy"][0] <= (x0 + x1) / 2 else x1
                    b = a + (0.12 if a == x0 else -0.12) * (x1 - x0)
                    col = tuple(int(0.45 * c) for c in PLAYER_COL.get(pid, (255, 255, 255)))
                    glow.line((a, y), (b, y), col, max(2, 0.2 * half))
        for s in snap["spells"]:                                   # glowing trail behind each spell
            x, y = s["xy"]
            r = 0.8 * half
            glow.line((x - s["dir"] * 2.6 * r, y), (x, y), SPELL_COL[s["action"]], max(2, r))
        for pid, p in players.items():
            if p["shield_up"] and p["lane"] is not None and p["body_xy"] is not None:
                sx = self._shield_x(p, x0, x1, W)
                y = snap["lanes"][p["lane"]]
                glow.line((sx, y - half), (sx, y + half), SPELL_COL[SHIELD], max(3, 14 * sc))
        for t0, t1, g, _ in active:
            if g:
                g(glow, (t_ms - t0) / (t1 - t0))
        glow.add_to(img, 3)
        mark("glow")

        # 3. solid things on top of the light
        for pid, p in players.items():
            if p["shield_up"] and p["lane"] is not None and p["body_xy"] is not None:
                sx, y = self._shield_x(p, x0, x1, W), snap["lanes"][p["lane"]]
                cv2.ellipse(img, (int(sx), int(y)), (int(14 * sc), int(half)), 0, 0, 360, (255, 245, 220), 2, cv2.LINE_AA)
        for s in snap["spells"]:
            t = s["age_ms"] / 1000
            r = 0.8 * half
            if s["action"] == ROCK:
                draw_rock(img, s["xy"], r, s["dir"] * t * 5)        # it rolls
            elif s["action"] == PAPER:
                draw_paper(img, s["xy"], r, t)
            else:
                draw_scissors(img, s["xy"], r, t, s["dir"])
        mark("spells + shields")
        self._hands(img, snap, sc)
        mark("hands")
        self._panels(img, snap, sc)
        mark("panels")
        self._centre(img, snap, sc)
        mark("centre texts")
        self._feed(img, t_ms, sc)
        for t0, t1, _, d in active:
            if d:
                d(img, (t_ms - t0) / (t1 - t0))
        mark("feed + effects")
        return img

    @staticmethod
    def _shield_x(p, x0, x1, W):
        """A shield stands just in front of its player, on the lane."""
        return p["body_xy"][0] + (0.06 if p["body_xy"][0] <= (x0 + x1) / 2 else -0.06) * W

    def _hands(self, img, snap, sc):
        """A ring on each casting hand with the gesture recognised right now (T4-07), at the player (T4-08)."""
        ph = snap["phase"]
        for pid, p in snap["players"].items():
            col = PLAYER_COL.get(pid, (255, 255, 255))
            b = p["body_xy"]
            if not p["present"]:
                if b is not None:
                    text(img, f"{self.short(pid)} OUT OF VIEW", (b[0], b[1] - 100 * sc), 0.9 * sc, (60, 60, 255), 2, center=True)
                continue
            if ph in (LOBBY, MATCH_OVER) and b is not None:
                ok = p["ready"]
                text(img, f"{self.short(pid)} READY" if ok else f"{self.short(pid)}: open palm = ready", (b[0], b[1] - 100 * sc),
                     (0.9 if ok else 0.6) * sc, (100, 255, 100) if ok else col, 2, center=True)
            elif ph in (FIGHT, ROUND_START, RESUME) and p["alive"] and p["hand_xy"] is not None:
                h = (int(p["hand_xy"][0]), int(p["hand_xy"][1]))
                cv2.circle(img, h, int(30 * sc), col, max(2, int(3 * sc)), cv2.LINE_AA)
                if p["cooldown"] > 0:                              # ring that empties while the cooldown runs
                    cv2.ellipse(img, h, (int(38 * sc), int(38 * sc)), -90, 0, 360 * p["cooldown"], (255, 255, 255),
                                max(2, int(3 * sc)), cv2.LINE_AA)
                if p["gesture"]:
                    d = 1 if b is None or b[0] < img.shape[1] / 2 else -1
                    icon(img, p["gesture"], (h[0], h[1] - 62 * sc), 22 * sc, d=d)
                text(img, self.short(pid), (h[0], h[1] + 58 * sc), 0.6 * sc, col, 2, center=True)

    def _panels(self, img, snap, sc):
        """HP, mana, rounds won, shield and potion of each player, in the top corners."""
        H, W = img.shape[:2]
        ids = list(snap["players"])
        pw = min(0.28 * W, 380 * sc)
        for i, pid in enumerate(ids):
            p = snap["players"][pid]
            x = 16 * sc + i * (W - 32 * sc - pw) / max(1, len(ids) - 1)
            y = 14 * sc
            col = PLAYER_COL.get(pid, (255, 255, 255))
            shade(img, x, y, x + pw, y + 100 * sc, 0.5)
            cv2.rectangle(img, (int(x), int(y)), (int(x + pw), int(y + 100 * sc)), col, 2)
            text(img, self.long(pid), (x + 10 * sc, y + 26 * sc), 0.7 * sc, col, 2)
            for r in range(snap["rounds_to_win"]):                 # rounds won
                c = (int(x + pw - 24 * sc - r * 24 * sc), int(y + 19 * sc))
                cv2.circle(img, c, int(8 * sc), (60, 220, 255) if r < p["rounds_won"] else (90, 90, 90), -1, cv2.LINE_AA)
            f = p["hp"] / p["max_hp"]
            bar(img, x + 10 * sc, y + 36 * sc, pw - 20 * sc, 20 * sc, f, hp_col(f))
            text(img, f"{p['hp']}", (x + pw / 2, y + 52 * sc), 0.5 * sc, (255, 255, 255), 1, center=True)
            bar(img, x + 10 * sc, y + 60 * sc, pw - 20 * sc, 7 * sc, p["mana"] / p["max_mana"], (255, 160, 60))
            st, st_col = ("shield UP", (255, 245, 210)) if p["shield_up"] else \
                ("shield ready", (255, 225, 150)) if p["shield_in_s"] == 0 else \
                (f"shield {p['shield_in_s']:.1f} s", (140, 140, 140))
            text(img, st, (x + 10 * sc, y + 88 * sc), 0.5 * sc, st_col, 1)
            text(img, f"potion x{p['potions']}", (x + pw * 0.55, y + 88 * sc), 0.5 * sc,
                 (120, 230, 120) if p["potions"] else (140, 140, 140), 1)

    def _centre(self, img, snap, sc):
        H, W = img.shape[:2]
        ph, cx, el = snap["phase"], W / 2, snap["phase_ms"]
        if ph == LOBBY:
            text(img, "ROCK  PAPER  SCISSORS", (cx, 0.2 * H), 1.5 * sc, (255, 255, 255), max(2, int(3 * sc)), center=True)
            text(img, "DUEL", (cx, 0.29 * H), 1.5 * sc, (60, 220, 255), max(2, int(3 * sc)), center=True)
            text(img, "Both players: hold an open palm to start", (cx, 0.88 * H), 0.8 * sc, (255, 255, 255), 2,
                 center=True)
            if snap["ready_progress"] > 0:
                bar(img, cx - 0.15 * W, 0.91 * H, 0.3 * W, 12 * sc, snap["ready_progress"], (100, 255, 100))
        elif ph in (ROUND_START, RESUME):
            frac = (el % 1000) / 1000
            title = f"ROUND {snap['round']}" if ph == ROUND_START else "GET READY"
            text(img, title, (cx, 0.2 * H), 1.1 * sc, (255, 255, 255), 2, center=True)
            text(img, str(snap["countdown"]), (cx, 0.62 * H), (4.5 - 1.6 * frac) * sc, (255, 255, 255),
                 max(3, int(8 * sc)), center=True)
        elif ph == FIGHT:
            if el < 700:
                text(img, "FIGHT!", (cx, 0.56 * H), (3.0 - el / 700) * sc, (60, 220, 255), max(3, int(7 * sc)),
                     center=True)
            if snap["round_left_s"] is not None:
                text(img, f"{snap['round_left_s']:.0f}", (cx, 48 * sc), 1.1 * sc, (255, 255, 255), 2, center=True)
        elif ph == ROUND_END:
            w = snap["round_winner"]
            text(img, snap["end_reason"] + "!", (cx, 0.45 * H), 2.6 * sc, (60, 60, 255), max(3, int(6 * sc)), center=True)
            msg = f"{self.names.get(w, f'Player {w}')} wins round {snap['round']}" if w else "Draw - nobody wins this round"
            text(img, msg, (cx, 0.56 * H), 0.9 * sc, PLAYER_COL.get(w, (255, 255, 255)), 2, center=True)
        elif ph == PAUSED:
            who = " and ".join(self.names.get(pid, f"Player {pid}") for pid in snap["lost"]) or "A player"
            text(img, "PAUSED", (cx, 0.42 * H), 2.0 * sc, (255, 255, 255), max(2, int(4 * sc)), center=True)
            text(img, f"{who}: come back into the picture!", (cx, 0.52 * H), 0.9 * sc, (60, 200, 255), 2, center=True)
        elif ph == MATCH_OVER:
            w = snap["match_winner"]
            text(img, f"{self.long(w)} WINS THE MATCH!", (cx, 0.45 * H), 1.6 * sc, PLAYER_COL.get(w, (255, 255, 255)),
                 max(2, int(4 * sc)), center=True)
            text(img, "Hold an open palm to play again", (cx, 0.56 * H), 0.8 * sc, (230, 230, 230), 2, center=True)

    def _feed(self, img, t_ms, sc):
        """The last few things that happened, at the bottom (what the gestures caused: T4-07)."""
        H, W = img.shape[:2]
        lines = [s for t0, s in self.feed if t_ms - t0 < FEED_MS]
        for i, s in enumerate(reversed(lines)):
            text(img, s, (W / 2, H - (18 + 30 * i) * sc), (0.7 if i == 0 else 0.55) * sc,
                 (255, 255, 255) if i == 0 else (200, 200, 200), 2, center=True)
