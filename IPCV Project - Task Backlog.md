# IPCV Project · Task Backlog

*Game: real-time **magic duel** = rock–paper–scissors with spells, 2 players, one webcam · Deadline **Sun 1 Nov 23:59** (video + final package) · Exhibition **Thu 5 – Fri 6 Nov** (to be confirmed)*
*Requirement IDs (SYS-, T1-…T5-, DEL-) refer to [[IPCV Project - Requirements Specification]].*

> [!abstract] Where we are (8 Oct)
> **Done:** a script that detects hands in images and classifies rock / paper / scissors.
> **Missing:** webcam + real time, two players with stable identities, face effects, game rules, scene, integration, measurements, and all deliverables.
> **About 3.5 weeks left.** The plan below gets a **playable two-player version by 19 Oct**, so the last two weeks are for robustness, measurements and deliverables.

---

## 0 · Decide this week (whole team)

> [!important] These block everything else. Agree on them in one meeting.

- [ ] **Owners** — fill in the table below (one task per person; the T5 owner is the team leader).
- [ ] **Game rules** — agree on the proposal in section 1, or change it.
- [ ] **Shared Git repository** — one repo, one folder per module, `requirements.txt`, `README.md`, and the existing hand/RPS script moved into it.
- [ ] **Interfaces** — T5 writes down what each module receives and returns (section 2). Everyone codes against that.
- [ ] **Demo hardware** — which laptop and webcam you will use at the exhibition. Develop and test on that one. Fix the resolution (e.g. 1280×720) and the target speed (**≥ 20 FPS**).

| Task | Module | Owner |
|---|---|---|
| T1 | Face tracking + face effects | |
| T2 | Hands + gestures (+ optional body pose) | Lander |
| T3 | Players: detection, identity, position | |
| T4 | Game rules, actions, feedback | |
| T5 | Scene, integration, performance · **team leader** | |

---

## 1 · Game rules (proposal)

| | |
|---|---|
| **Spells** | Rock = **Earth** · Paper = **Water** · Scissors = **Lightning** |
| **Who wins** | Earth grounds Lightning · Lightning electrifies Water · Water erodes Earth (same as rock–paper–scissors) |
| **Round** | countdown 3-2-1 → both cast **at the same moment** → winner deals damage → effects → next round |
| **Life** | 100 HP each; a win deals 25; a tie deals nothing (or both 10) |
| **2nd action** | **Shield**: both palms open towards the camera (or arms crossed). Blocks the next hit; one use per 3 rounds. Satisfies "≥ 2 distinguishable actions per player" (T4-02). — **Dropped (team decision, 22 Oct 2026).** Countering in the same lane is the only defence now. |
| **Optional extra** | **Potion**: hold a cup or bottle to your mouth → +15 HP, once per game (real-world object interaction, T4). — **Kept, camera trigger pending** (object detector or a dedicated gesture); keyboard-only (P1 G, P2 N) until then. |
| **End** | a player at 0 HP loses; a "WINNER" screen; restart with both hands up |

> [!tip] Why a countdown?
> Both gestures are read at the same instant, so nobody can react to the other. It also gives a clear moment to confirm the gesture (T4-04/05) and a natural place for the "charging" effects.

---

## 2 · Interfaces (T5 defines, everyone uses)

```python
# what each module produces every frame (proposal)
Hand   = {"wrist_xy": (x, y), "landmarks": 21x(x, y, z), "gesture": "rock|paper|scissors|None", "conf": 0-1}
Face   = {"bbox": (x0, y0, x1, y1), "center": (x, y), "size": px, "roll_deg": a, "landmarks": [...]}
Player = {"id": 1|2, "status": "ok|lost|reentered", "pos_cm": (x, y), "face": Face, "hands": [Hand], "hp": 100, ...}

hands   = hand_module.process(frame, t_ms)         # T2 -> list[Hand]
faces   = face_module.process(frame, t_ms)         # T1 -> list[Face]
players = player_tracker.update(faces, hands)      # T3 -> dict[id, Player]
events  = game.update(players, t_ms)               # T4 -> spells, hits, state
frame   = scene.render(frame, players, events)     # T5 -> image shown on screen
```

> [!important] Rule for everyone
> Code for **N players from day one**: no `hands[0]`, no global `hp`. Everything lives inside `players[id]`.

---

## 3 · Milestones

| Week | Dates | Goal | Done when |
|---|---|---|---|
| **W1** | Thu 8 – Sun 11 Oct | Foundations | repo + interfaces agreed; every module runs **live on the webcam** for **one** person |
| **W2** | Mon 12 – Sun 18 Oct | **First playable** | two players, identity by **screen halves**, a full round works end to end |
| **W3** | Mon 19 – Sun 25 Oct | Robust + measured | real identity tracking (crossing, leave/re-enter); effects polished; **all measurements done**; report draft |
| **W4** | Mon 26 Oct – Sun 1 Nov | Freeze + deliver | **feature freeze Tue 27**; video recorded **Thu 29**; report, poster, ZIP submitted **before Sun 1, 23:59** |
| Exhibition | Thu 5 – Fri 6 Nov | Live demo | 3-min demo + 1 min per person of questions rehearsed |

---

## 4 · Tasks per module

### T2 · Hands and gestures — Lander

**W1**
- [ ] Move the existing hand + RPS script into the repo and wrap it as `hand_module.process(frame, t_ms) -> list[Hand]`.
- [ ] Switch to **webcam + video mode** (MediaPipe `VIDEO` mode, frame timestamps); live window with FPS counter.
- [ ] Check **mirroring**: if the image is flipped for display, does Left/Right still come out right? Decide where to flip.

**W2**
- [ ] **Smoothing** of the landmarks (EMA or One Euro filter) — less jitter without too much delay (T2-04, T2-07).
- [ ] **Stable gesture**: report a gesture only if it is the same for N frames in a row (e.g. 5) (T2-04, T4-04).
- [ ] **Uncertain hands**: low confidence or a lost hand → gesture `None`, never a random spell (T2-05, T2-06).
- [x] ~~**Shield gesture** (both palms open) or other 2nd action, agreed with T4.~~ **Dropped (team decision, 22 Oct 2026).**

**W3**
- [ ] **Evaluation** (report T2 section):
  - [ ] gesture accuracy: ≥ 30 attempts per gesture, at least 2 people → confusion matrix
  - [ ] jitter: landmark std (px) with the hand still, raw vs smoothed
  - [ ] delay added by the smoothing (ms / frames)
  - [ ] ms per frame of the module
  - [ ] failure cases with images (touching hands, side view, fast motion, bad light)
- [ ] *(Optional)* compare your rule with MediaPipe's GestureRecognizer (`Closed_Fist` / `Open_Palm` / `Victory`).

### T1 · Face tracking and face effects

**W1**
- [ ] Face detection/landmarks on the webcam for **2 faces** (e.g. MediaPipe Face Landmarker, `num_faces=2`, video mode) (T1-01).
- [ ] Per face: position, size and tilt angle (from the eye line) (T1-02).

**W2**
- [ ] **Face effect** (e.g. a wizard hat in the player's colour) as a PNG with transparency, placed with an affine transform from position + size + angle (T1-04, T1-05).
- [ ] **Smoothing** of position, size and angle (T1-03).
- [ ] **Extra game-state effect**: name + HP bar (+ mana) floating above each head (T1-08).
- [ ] Short loss of the face → keep the last position for ~0.3 s, then fade out (T1-07).

**W3**
- [ ] Effects that **mean something in the game** (T1-09): glowing eyes while casting; frost/burn/mud on the face when hit by Water / Lightning / Earth.
- [ ] Effects follow the **player ID from T3**, so they never swap between players (T1-06).
- [ ] **Evaluation:** detection rate, effect jitter (px) with vs without smoothing, swaps per minute, ms per frame.

### T3 · Players: detection, identity and position

**W1**
- [ ] Decide the **anchor** of each player (face, body pose, or both) and how hands are attached to a player (e.g. hand wrist ↔ closest body wrist or face).
- [ ] Temporary version: **screen halves** — left = Player 1, right = Player 2 (T3-01, T3-02).

**W2**
- [ ] Real tracker: predicted position (velocity) + **Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) (T3-05).
- [ ] Attach each face and each hand to the right player (T3-03, T3-04).

**W3**
- [ ] **Calibration at game start**: each player alone on their side for 2 s → store a colour histogram of the torso (and/or a face descriptor) (T3-07).
- [ ] Use appearance in the matching cost → **crossing and stopping**, **leave and re-enter** (T3-05, T3-06, T3-07).
- [ ] Behaviour when a player is lost: round paused, "Player 2, come back!" (T3-07, T5-10).
- [ ] **Position in physical units** (cm) — e.g. from face width or a reference marker; optional depth (T3-08, T3-09).
- [ ] **Evaluation:** ID swaps over 20 crossings, recovery time after occlusion, % correct re-entries, ms per frame (T3-10).

### T4 · Game rules, actions and feedback

**W1**
- [ ] Game **state machine**: `CALIBRATION → COUNTDOWN → CAST → RESOLVE → (GAME_OVER)`. Test it with the keyboard first (keys = gestures), without any vision.
- [ ] Rules table from section 1 in code: who beats whom, damage, ties (T4-01).

**W2**
- [ ] Connect to the real gestures: read each player's **stable** gesture at the end of the countdown (T4-04, T4-05).
- [x] **≥ 2 actions per player**: spell + lane choice + potion (T4-02, T4-03). (Shield dropped, team decision, 22 Oct 2026; potion kept, camera trigger pending.)
- [ ] **Cooldowns** and rules for simultaneous / conflicting actions (T4-05, T4-09).
- [ ] HP / mana updated **per player ID** (T4-06).

**W3**
- [ ] **Visual feedback**: gesture icon over each hand, the spell flying from the caster's hand to the opponent, a "HIT!" / "BLOCKED!" text (T4-07, T4-08).
- [ ] **Evaluation:** action accuracy over N rounds, false actions per minute, gesture-to-feedback delay (ms).

### T5 · Scene, integration and performance — team leader

**W1**
- [ ] Main application: camera loop → modules → game → render; one config file (camera index, resolution, thresholds).
- [ ] Interfaces of section 2 written down and shared (T5-01).

**W2**
- [ ] **First integrated version** with all modules, screen-half identity, a full round (T5-02, T5-05).
- [ ] **Scene / background**: arena look (e.g. background replacement with a person-segmentation model, or a darkened, tinted background) (T5-03).
- [ ] **FPS and latency** shown on screen and logged (T5-06).

**W3**
- [ ] **Scene reacts to spells**: Earth = rocks and a shaking screen, Water = a wave across the screen, Lightning = a white flash + bolt (T5-04).
- [ ] **Profiling**: time of each module per frame → bottlenecks → optimisations (lower resolution for detection, skip frames, threads) (T5-07, T5-08).
- [ ] **Failure handling**: player lost, no hands, a module crashing → the game pauses instead of breaking (T5-10).
- [ ] **Full tests with 2 players** on the demo laptop, in different lighting (T5-09).
- [ ] **Evaluation:** end-to-end FPS (mean/min) and latency, profiling table before vs after optimising.

---

## 5 · Deliverables

### Peer-review video — due Sun 1 Nov (DEL-1)
- [ ] **Script of the 60 s**, about 10 s per task: calibration → round with spells → crossing (T3) → face effects close up → scene reacting → winner. (Shield step removed - dropped, team decision, 22 Oct 2026.)
- [ ] Record in **one take**: no cuts, no speed-up; webcam image and game output both visible.
- [ ] Check: MP4, ≤ 60 s, ≤ 100 MB. One member submits on Canvas.

### Technical report — ≤ 5 pages (DEL-2.1 – 2.5)
- [ ] Template + system design section (pipeline diagram, interfaces) — **T5**
- [ ] One section per task, **with the owner's name**, each with: brief pipeline, design choices + alternatives, **measured** results (FPS / latency, accuracy), failure cases — **each owner**
- [ ] Responsibilities table
- [ ] Appendix (does not count for the page limit): **AI and external-tools statement** — models, libraries, coding agents, what each did, how you adapted and tested it, its limits (AI-01 … AI-04)

### Poster — 1 page A4 (DEL-2.6 – 2.8)
- [ ] Game concept + rules, pipeline diagram, the 5 tasks with owners, main tools, **performance numbers**, limitations.

### Final ZIP — ≤ 500 MB (DEL-2.9 – 2.13)
- [ ] All code, report, poster, README, `requirements.txt`, assets (images, sounds), AI statement.
- [ ] README: file structure, **tested** setup steps, how to run, owners of T1–T5 (same names as the report).
- [ ] Model files: included, or download instructions with **exact version / hash**.
- [ ] Final test: unzip on **another laptop**, follow the README from zero, play one game.

### Exhibition — Thu 5 / Fri 6 Nov (DEMO-01 … 07)
- [ ] Rehearse the **3-minute** live demo, including one **difficult tracking moment** (crossing or someone leaving the frame).
- [ ] Each owner rehearses a **1-minute** explanation of their task: pipeline, choices, numbers, failures.
- [ ] Poster printed (or on a second screen); spare webcam; everything installed on the demo laptop.
- [ ] **All five present** for the whole assigned half-day.

---

## 6 · Risks to watch

| Risk | Plan B |
|---|---|
| Too slow with all models running | lower detection resolution, run face/pose every 2nd frame, skip the background segmentation |
| Hands of the two players touch or overlap | game rule: stay on your side; keep about 1.5–2 m from the camera |
| Lighting at the exhibition differs | test in a dim and in a bright room; tune thresholds; bring a small lamp if allowed |
| Identity swaps at crossings | appearance calibration; in the worst case keep the screen-half rule as a game rule and **report** the crossing failures honestly |
| Someone's module is late | T5 integrates a simple placeholder (e.g. screen halves, static hat) so the game always runs |
