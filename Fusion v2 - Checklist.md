# Fusion v2 – Checklist

Review of the second fusion (Claude Code, 22 Oct 2026). Teammates' files are integrated unchanged, 62 tests pass, solo fix and frame dedup work. What is left, in order.

## 1. Code fixes (Claude Code – prompt in section 5)

- [ ] **Critical – player 2's hand is dropped.** `HandAdapter` uses `task2_hands.assign()` (marked *placeholder* by its author): with 3 or 4 hands visible it keeps only the 2 left-most, so P2 gets no gesture while P1 shows both hands. Reproduced: 3 hands → `{1: FIRE}`, P2 missing.
  *Done when:* hands are assigned to players first (pose raised wrist), one `HandSlot` per player id; tests for 2 / 3 / 4 visible hands give both players their gesture.
- [ ] **Faces at playing distance.** Small faces are not found on the full frame (sample `04_arms_raised_stage.jpg`: no glasses; v1 with pose eyes had them).
  *Done when:* face search is seeded from each player's pose head (`detect_region` around `Player.head`); sample 04 gets a face track.
- [ ] **Kalman `dt`** clamped to [1/120, 0.2] s, as the original `face_tracking_final.main()` does.
- [ ] **Speed with two people.** Measured: pose 4 ms (CUDA) + hands ≈25 ms + faces ≈20 ms (p95 40) ≈ 50 ms/frame ≈ 20 FPS. README says ≈38 FPS (measured with fewer people).
  *Done when:* ≤ 33 ms per camera frame with 2 people and ≥ 3 hands, or a clear explanation of why not.
- [ ] **Lobby:** the over-head "P1 100/100" label overlaps the title → hide the over-head bar during LOBBY.
- [ ] **README** updated with the 2-person numbers and the fixes; one commit per fix.
- [ ] **Shield removed** from the game and from the meeting material; potion kept (prompt in section 6).

## 2. Review after Claude Code (ask Claude in chat)

- [ ] Re-run the 3- and 4-hand check (P2 must get a gesture)
- [ ] Re-run the still-photo check with timings (samples 03, 04, 05, 06)
- [ ] Look at the glasses / HP bar picture again

## 3. Live test with two people

- [ ] P1 is the one on the left at the start
- [ ] Both players cast rock / paper / scissors and change lane; **P2 casts while P1 has both hands visible**
- [ ] Glasses and HP bar stay on the right player when crossing slowly
- [ ] One hides behind the other for 1 s → same player afterwards
- [ ] One leaves the frame 4 s and comes back → same player
- [ ] A third person walks behind → ignored
- [ ] FPS on screen ≥ 25
- [ ] Record a 30 s clip of all the above for offline comparison:
  `ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 -t 30 clip_2p.mp4`
- [ ] One minute with `python play_camera.py --log session.csv` → T2 jitter metric (raw vs filtered wrist)

## 4. Team

- [ ] **T2:** `task2_hands.py` does not run on its own with Python 3.14 / MediaPipe 1.0.1 (`mp.solutions` was removed). His standalone demo needs his own environment or the Tasks API.
- [ ] **T1:** the face detector misses small faces on the full frame; the game now searches around the pose head (once fix 2 is in).
- [ ] **T3:** validate the colour re-ID option (`appearance=True`) on the two-person clip.
- [ ] Where do the sunglasses images come from (licence)?
- [x] Shield: **dropped** (team decision, 22 Oct 2026).
- [ ] **Potion: kept, still pending.** Decide how the camera triggers it (e.g. a real bottle or cup at the mouth: YOLO's COCO classes `bottle` / `cup` near the pose head; or a dedicated gesture) and who builds it. Until then it stays on the keyboard: P1 G, P2 N.
- [ ] Still open: lane heights, demo laptop and its environment, shared git repo, owner name per module.

## 5. Prompt for Claude Code

```
Fusion v2 follow-up in game/. Keep everything that works: the teammates' files in face_tracking/,
hand_gestures/ and pose_reid/ stay byte-identical; game_logic.py and hud.py stay untouched; all current
tests keep passing. First commit the current working tree as "Fusion v2: teammates' modules integrated",
then make one commit per fix below.

1. CRITICAL - player 2's hand is dropped. HandAdapter feeds HandSlots through task2_hands.assign(), which
   its author marked as a placeholder ("Task 3 should replace this"): with 2+ detections it keeps only the
   two left-most hands. With P1 showing both hands and P2 one raised hand, P2 gets no gesture (reproduced
   with a fake model: 3 hands -> {1: FIRE}, P2 missing; 4 hands -> same). Fix: assign the raw MediaPipe
   detections to players FIRST (raised-wrist match against PlayerTracker, i.e. the game-side assign()),
   then feed one unmodified task2_hands.HandSlot per player id (slots keyed by pid, never by left-to-right
   order); detections not assigned to a player are ignored. Tests: with 2, 3 and 4 visible hands both
   players get the gesture of their raised hand, and a player's slot never jumps to another player's hand
   when a resting hand appears or disappears.

2. Faces at playing distance. The short-range FaceLandmarker misses faces of ~100-200 px on the full
   1280-wide frame (README already notes it); hand_detection/samples/04_arms_raised_stage.jpg currently
   gets no face track at all. Seed the search from pose: for each present player without a confirmed face
   track, call FaceDetector.detect_region() on a box around Player.head (size from the shoulder width or
   the person box), and pass those detections into FaceTracker.step(). Keep the half-frame search as a
   fallback. Do this in face_module.py, not in face_tracking_final.py. Report before/after on samples
   03, 04, 05, 06.

3. FaceModule.update(): clamp dt to [1/120, 0.2] s, as face_tracking_final.main() does (the first frame
   currently gets dt=0 and a stall gives an unbounded dt).

4. Performance with TWO people and 3+ hands in the frame (samples 03, 05, 06 letterboxed to 1280x720,
   30 frames each, CPU and CUDA): per-stage mean and p95 for pose, hands, faces and drawing. My own
   measurement: pose 4 ms on CUDA, hands ~25 ms, faces ~20 ms (p95 ~40) -> ~50 ms per camera frame,
   ~20 FPS, not the ~38 FPS in the README. If the total is above 33 ms, make the cheapest change that
   brings it under (e.g. --face-every 2 by default, ROI-only face detection once tracks exist) and measure
   again. Ask me before adding a worker thread.

5. Lobby: the over-head "P1 100/100" label overlaps the title. Hide the over-head HP bar during LOBBY and
   show it from ROUND_START on.

6. game/README.md: replace the performance table with the two-person numbers (state the test
   conditions), document fixes 1-5, keep the known-risks list current.

Finish with: all tests passing, the list of commits, and what is still unvalidated on real two-player
footage.
```

## 6. Prompt for Claude Code – remove the shield, keep the potion

```
Team decision (22 Oct 2026): the SHIELD is dropped completely. The POTION stays (its camera trigger is
still pending; until then it remains on the keyboard: P1 G, P2 N). Actions are now: rock / paper /
scissors spells + lane choice + potion. Work in this order, one commit per step. Do not touch the
teammates' files (face_tracking/, hand_gestures/, pose_reid/). If any potion code was already removed by
an earlier instruction, restore it from commit 643ce6d.

STEP 1 - meeting material first (I am showing it now), commit as soon as it is done:
- docs/team_meeting.md: remove every mention of the shield (gesture decision, keys F/H, any shield
  feature). Under "decisions taken": "Shield dropped (22 Oct)". Under "pending": "Potion kept - decide
  how the camera triggers it (e.g. real bottle / cup at the mouth: YOLO COCO bottle / cup near the pose
  head, or a dedicated gesture) and who builds it; keyboard G / N until then". Update the T4 rows of the
  requirement table: T4-02 (>= 2 actions) = 3 spell gestures + lane choice + potion; T4-03
  (interaction) = spells clashing / countering in the lanes (player-player) + potion with a real object
  (player-object, pending).

STEP 2 - the game (shield only):
- game_logic.py: remove SHIELD and its GESTURE_TO_ACTION entry (an incoming "SHIELD" gesture must simply
  be ignored, like any unknown gesture), Config.shield_cost / shield_max_ms / shield_cooldown_ms, the
  shield fields of PlayerState, the snapshot keys shield_up / shield_in_s and the events shield_up /
  shield_down / blocked. Keep POTION, potion_heal, potions_per_round, the "potions" snapshot key and the
  "heal" event exactly as they are. Defence against a spell is now only countering in the same lane.
- hud.py: remove shield drawing, the "shield ready" panel text and the feed sentences for the removed
  events; keep "potion x1" and the heal feedback. Keep the panel layout tidy.
- bot.py: remove the shield behaviour; keep drinking the potion when hurt; the bot defends by countering.
- play_keyboard.py: remove keys F (P1) and H (P2); keep G / N for the potion; update the help text.
- play_camera.py: keep the keyboard override only for the potion (G / N); remove f / h and the shield
  help text; say in the help that the potion's camera trigger is pending.
- simulate.py: remove the shield from the scripted match, keep the potion.
- test_game_logic.py: delete the shield tests, add a test that a "SHIELD" gesture is ignored and that
  countering in the same lane still works; keep the potion tests. All tests must pass.

STEP 3 - regenerate and document:
- regenerate docs/img/camera_mode_samples.jpg (its HUD panels show "shield ready") and
  game/simulated_game.mp4;
- game/README.md: rules, T4 requirement mapping, keyboard keys, events list, PlayerInput comment, the
  "to decide as a team" list (shield: dropped; potion: camera trigger pending); re-measure the bot win
  rates per level (5 games each, as before) since its defence changed;
- "IPCV Project - Task Backlog.md": do not delete the history: mark the shield rows and tasks as
  "Dropped (team decision, 22 Oct 2026)", mark the potion as "kept - camera trigger pending", and remove
  the shield step from the 60 s demo script.

Finish with: grep -riE "shield" over the project (excluding .venv, .git and the teammates' folders)
showing only the "dropped" notes, all tests passing, and the list of commits.
```
