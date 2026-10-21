# Hand detection · how it works

*Code: `detect_hands.py` · setup and test results: [[UT/A1/Image Processing/Project/hand_detection/README|README]] · figures: `docs/make_figures.py`*

> [!abstract] In one breath
> The script uses Google's **MediaPipe Hand Landmarker**: one file, `hand_landmarker.task`, that contains **two small neural networks**.
> **Net 1** looks at the whole photo shrunk to 192×192 and finds **palms**.
> **Net 2** looks at **each hand separately**, cut out, turned upright and resized to 224×224, and returns **21 points**, "is this really a hand?" and **Left/Right**.
> Our own code prepares the image, calls the model, converts the points to pixels, and draws them.

## Map
1. [[#1 · Which model]]
2. [[#2 · The pipeline in one picture]]
3. [[#3 · Step 1 — prepare the image]]
4. [[#4 · Step 2 — Net 1, the palm detector]]
5. [[#5 · Step 3 — from palm box to hand crop]]
6. [[#6 · Step 4 — Net 2, the landmark model]]
7. [[#7 · Step 5 — back to the photo]]
8. [[#8 · The 21 landmarks]]
9. [[#9 · What our script adds]]
10. [[#10 · From points to rock, paper, scissors]]
11. [[#11 · Settings and scale]]
12. [[#12 · Photo mode vs video mode]]
13. [[#13 · Limits seen in our tests]]
14. [[#14 · Where the course theory is]]
15. [[#15 · Glossary]]

---

## 1 · Which model

> [!info] Idea
> A **pretrained** model from Google. We did not train anything; we use it as a building block and must be able to explain it (requirements SYS-10 and AI-01).

| | |
|---|---|
| Name | MediaPipe **Hand Landmarker** (float16) |
| File | `models/hand_landmarker.task` · 7.8 MB · downloaded 6 Oct 2026 |
| SHA-256 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| Library | `mediapipe` 1.0.1, **Tasks API** (`vision.HandLandmarker`) |
| Runs on | laptop CPU, about 10–30 ms per photo |

A `.task` file is just a **ZIP** with the two networks in TensorFlow Lite format (check it with `unzip -l models/hand_landmarker.task`):

| Inside the `.task` | Internal name (read from the file) | Size |
|---|---|---|
| `hand_detector.tflite` | `palmdetector_full_192x192_2021_09_30` | 2.3 MB |
| `hand_landmarks_detector.tflite` | `handskeleton_handflag_handedness_full_2021_10_07` | 5.5 MB |

> [!tip] Remember
> "One file, two nets: one finds the palm, one draws the skeleton."

---

## 2 · The pipeline in one picture

![[HD_pipeline.jpg|900]]

| Step | Who | What happens |
|---|---|---|
| 1 | our code | load the photo, convert BGR → RGB |
| 2 | **Net 1** | the whole photo at 192×192 → palm boxes |
| 3 | MediaPipe | cut out each hand, rotate it upright, resize to 224×224 |
| 4 | **Net 2** | each crop → 21 points + presence + Left/Right |
| 5 | MediaPipe + our code | points mapped back onto the original photo, converted to pixels, drawn |

*Panels 2 and 3 were rebuilt by `docs/make_figures.py` to show what each net receives. MediaPipe computes the real crop from Net 1's palm box; ours is rebuilt from the final points, so it is approximate.*

> [!info] Why two networks?
> Finding **where** a hand is needs the whole image but little detail. Finding **each finger joint** needs detail but only a small area. Splitting the job keeps both nets small and fast.

---

## 3 · Step 1 — prepare the image

> [!info] Idea
> OpenCV and MediaPipe store colours in a different order, so we convert first.

```python
bgr = cv2.imread(path)                          # OpenCV loads Blue-Green-Red
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)      # MediaPipe expects Red-Green-Blue
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = detector.detect(image)                 # steps 2-5 happen inside this call
```

- **Inside `detect()`**, MediaPipe shrinks the photo to 192×192 **keeping its proportions** and fills the rest with black bars (*letterbox*, panel 2 of the figure).
- **Why not just stretch it to a square?** That would deform the hands, and the net was trained on undeformed ones.

> [!warning] Trap
> Forgetting BGR → RGB does not crash: the model just sees blue skin and detects worse.

---

## 4 · Step 2 — Net 1, the palm detector

> [!info] Idea
> The net does not "search" the image. It looks at it once and gives a score to **2016 fixed candidate boxes** (*anchors*) spread over the image. The good ones are kept.

| | Shape (read from the file) | Meaning |
|---|---|---|
| Input | 1 × **192 × 192** × 3 | the letterboxed photo |
| Output A | 1 × **2016 × 18** | for each candidate: 4 box numbers + 7 palm points × (x, y) |
| Output B | 1 × **2016 × 1** | for each candidate: "palm or not" score |

**Where 2016 comes from.** The net has two output layers, `classifier_palm_8` and `classifier_palm_16` (layer names in the file), which look at the image on two grids:

| Grid | Cell size | Candidates per cell | Total |
|---|---|---|---|
| 24 × 24 | 8 px | 2 | 1152 |
| 12 × 12 | 16 px | 6 | 864 |
| | | | **2016** |

*(Candidates per cell are inferred: they are the split that gives the 2016 in the file.)* The fine grid is for small palms, the coarse one for big palms.

**From 2016 candidates to a few palms:**
1. score → probability (sigmoid) → keep only those above `min_hand_detection_confidence` (0.5);
2. many neighbouring candidates fire on the same palm → **non-maximum suppression (NMS)** keeps the best and removes boxes that overlap it.

**Inside the net:** 35 normal convolutions + 28 *depthwise* convolutions, 1.13 M parameters. *Depthwise separable* convolution = first one small filter **per colour channel**, then a 1×1 convolution that mixes channels. It gives about the same result as a normal convolution with far fewer multiplications, which is why it runs on a laptop CPU in milliseconds.

**Why palms and not whole hands?** A palm is almost rigid and roughly square; fingers bend in every direction. A rigid square is much easier to detect, and two close hands overlap less as palms than as whole hands (Zhang et al., *MediaPipe Hands*, 2020).

> [!tip] Remember
> "2016 guesses → threshold → NMS → a few palm boxes."

> [!warning] Trap
> NMS removes boxes that overlap a lot. **Two touching palms** can be merged into one: that is what happened in photos 02 and 05.

---

## 5 · Step 3 — from palm box to hand crop

> [!info] Idea
> The palm box is turned into a **square, upright crop of the whole hand**. This is a geometric transformation (Lecture 6).

1. **Enlarge.** The palm box only covers the palm. MediaPipe enlarges it (about 2.6×) and shifts it towards the fingers so the whole hand fits.
2. **Rotate.** Net 1 also gave 7 palm points; the line **wrist → base of the middle finger** gives the hand's angle. The crop is rotated so that this line points **up**.
3. **Resize** the crop to 224×224.

All three are one **affine transformation** $M$ (rotation + scale + translation), applied with a warp. MediaPipe keeps $M$ to map the points back later (step 5).

```python
# what docs/make_figures.py does to rebuild panel 3 of the figure
v = pts[9] - pts[0]                                     # wrist -> middle-finger base
angle = np.degrees(np.arctan2(v[1], v[0])) + 90         # angle that makes v point up
M = cv2.getRotationMatrix2D(center, angle, 224 / side)  # rotation + scale (Lecture 6)
crop = cv2.warpAffine(img, M, (224, 224))
```

> [!info] Why rotate?
> Net 2 then only ever sees **upright hands**. It does not have to learn every possible angle, so it can be smaller and more precise.

---

## 6 · Step 4 — Net 2, the landmark model

> [!info] Idea
> For each crop the net directly **outputs numbers**: the coordinates of 21 points, plus two yes/no scores. This is *regression*: no searching, no heatmaps.

| | Shape (read from the file) | Meaning |
|---|---|---|
| Input | 1 × **224 × 224** × 3 | the upright hand crop |
| Output 1 | 1 × **63** | 21 points × (x, y, z) in crop coordinates |
| Output 2 | 1 × 1 | **presence**: is there really a hand in this crop? |
| Output 3 | 1 × 1 | **handedness**: Left or Right |
| Output 4 | 1 × 63 | the same 21 points in **metres** (*world landmarks*) |

- **Inside:** 31 convolutions + 16 depthwise convolutions extract features. A *global average* (one number per feature map) is followed by **4 fully connected "heads"**, one per output. 2.72 M parameters.
- **Presence** below `min_hand_presence_confidence` (0.5) → the hand is thrown away. That is the second filter against false detections.
- **Handedness** gives the label and a confidence. A low value means "not sure" (0.55 in photo 08).
- **Training** (MediaPipe paper): real photos annotated by hand, plus synthetic images rendered from a 3D hand model.

> [!tip] Remember
> "Net 2 does not look for the hand. It already knows it is there, and measures it."

---

## 7 · Step 5 — back to the photo

> [!info] Idea
> The points are in crop coordinates. They are mapped back with the **inverse** of $M$ and then expressed as fractions of the photo size.

| Field our code receives | Meaning |
|---|---|
| `lm.x`, `lm.y` | position as a **fraction** of width and height: 0 = left/top, 1 = right/bottom |
| `lm.z` | depth **relative to the wrist**; negative = closer to the camera than the wrist |
| `handedness[0].category_name`, `.score` | `"Left"`/`"Right"` and its confidence |

Our code then converts to pixels and builds the box:

```python
pts = np.array([[lm.x * w - b, lm.y * h - b] for lm in landmarks])   # b = padding (0 by default)
x0, y0 = pts.min(axis=0)                                             # box = smallest rectangle
x1, y1 = pts.max(axis=0)                                             # around the 21 points
```

> [!example] Worked example — photo 09 (1280 × 855 px), the man's right fist
> | Point | x (px) | y (px) | z |
> |---|---|---|---|
> | 0 · wrist | 378 | 529 | 0 |
> | 8 · index tip | 430 | 482 | −0.029 |
> | 20 · pinky tip | 431 | 536 | −0.023 |
>
> Box: (378, 466) → (472, 535). The fingertips are close together in one column, which is a fist. The negative z means the knuckles are closer to the camera than the wrist.

> [!info] Handedness on normal photos
> In our photos the label matched the person's real hand (04, 09). If you **mirror** the webcam image before detection, Left and Right swap.

---

## 8 · The 21 landmarks

![[HD_landmarks_numbered.jpg|420]]

| Finger | Base → tip | Joint names (base → tip) |
|---|---|---|
| wrist | 0 | |
| thumb | 1 → 4 | CMC, MCP, IP, tip |
| index | 5 → 8 | MCP, PIP, DIP, tip |
| middle | 9 → 12 | MCP, PIP, DIP, tip |
| ring | 13 → 16 | MCP, PIP, DIP, tip |
| pinky | 17 → 20 | MCP, PIP, DIP, tip |

*MCP = knuckle where the finger joins the hand · PIP = middle joint · DIP = joint nearest the tip.*

> [!tip] Remember
> "Tips are multiples of 4: 4, 8, 12, 16, 20. The wrist is 0. Each finger base is a tip + 1: 5, 9, 13, 17."

---

## 9 · What our script adds

| Function in `detect_hands.py` | What it does |
|---|---|
| `make_detector()` | loads the `.task` and sets the options: photo mode, up to 4 hands, thresholds 0.5 |
| `detect()` | optional padding → BGR→RGB → `detector.detect()` → pixels, box, timing |
| `draw()` | skeleton (`HAND_CONNECTIONS`), points (wrist in red), box, "Left 0.98" |
| `overview()` | all results on one sheet, `_overview.jpg` |
| `main()` | reads the arguments, loops over the photos, writes `results.json` |
| top of the file | replaces the `sounddevice` module if there is no sound server: MediaPipe imports audio even for vision |

> [!note] Division of work
> The **model** finds the hands and the points. **Our code** handles input/output, coordinates, drawing, measurements, and the gesture rule (section 10).

---

## 10 · From points to rock, paper, scissors

![[HD_rps_rule.jpg|650]]

> [!info] Idea
> A finger is **extended** if its tip is **farther from the wrist** than its middle joint (PIP). Green line = extended, red = folded, white dot = PIP.

| Gesture | Index | Middle | Ring | Pinky |
|---|---|---|---|---|
| Rock | folded | folded | folded | folded |
| Paper | extended | extended | extended | extended |
| Scissors | extended | extended | folded | folded |
| anything else | → `?` (no decision) | | | |

```python
FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}  # (PIP, tip)
def finger_up(pts, pip, tip):
    wrist = pts[0]
    return np.linalg.norm(pts[tip] - wrist) > np.linalg.norm(pts[pip] - wrist)
```

- **Rotation-proof** — distances do not change when the hand turns. The tutorial rule "tip above the joint" (`tip.y < pip.y`) fails on sideways hands.
- **Size-proof** — it compares two distances of the **same** hand, so distance to the camera does not matter.
- **Thumb ignored** — not needed for these three gestures, and the least reliable finger.

**Tested** on your three rock-paper-scissors photos: **6 of 6 detected hands classified correctly**, including two sideways scissors. *(Prototype run on `results.json`; not yet built into `detect_hands.py`.)*

> [!warning] Trap
> A finger pointing **at the camera** looks short in the image (foreshortening) and counts as folded. Fix: use the world landmarks in metres, or the z values.

---

## 11 · Settings and scale

| Setting | Default | Higher | Lower |
|---|---|---|---|
| `min_hand_detection_confidence` (Net 1) | 0.5 | fewer false palms, more misses | more palms found, more false ones |
| `min_hand_presence_confidence` (Net 2) | 0.5 | same idea, second filter | at 0.1 the handshake photo gave 4 fake "hands" |
| `num_hands` | 4 | — | stops after N hands (2 players × 2 hands = 4) |
| `--pad` (our option) | 0 | close-up hands found | small hands lost |

![[HD_scale_padding.jpg|900]]

> [!info] Why padding works both ways
> Net 1 always sees **192×192**. Padding makes the whole photo smaller inside those 192 pixels. A fist that filled the photo (01) becomes palm-sized and is found; hands that were already small (05) become too small and disappear. This is the **scale problem** of Lecture 5: one fixed scale cannot fit every hand size.

---

## 12 · Photo mode vs video mode

| Mode | Call | Behaviour |
|---|---|---|
| `IMAGE` (now) | `detect(img)` | every photo is independent: Net 1 + Net 2 every time |
| `VIDEO` | `detect_for_video(img, t_ms)` | frames in order: **tracking** |
| `LIVE_STREAM` | `detect_async(img, t_ms)` | like VIDEO, results via a callback, drops frames if too slow |

> [!info] Tracking
> In video a hand barely moves between frames, so **the points of frame t give the crop for frame t+1** and Net 1 can be skipped. Net 1 runs again only when the presence score drops below `min_tracking_confidence`, for example when the hand is lost. This makes the webcam version much faster than running photo mode on every frame.

---

## 13 · Limits seen in our tests

| Situation | Result | Why |
|---|---|---|
| isolated, medium-size hand | **8 / 8 found** | what the model was trained for |
| two palms touching (02, 05) | merged into one | NMS (section 4) |
| interlocked or overlapping hands (06, 11) | missed | Net 1 never sees a clear palm shape |
| hand filling the whole photo (01) | missed, found with `--pad` | scale (section 11) |
| black and white, low contrast (10) | missed | trained on colour photos, skin colour is a cue |
| fist seen from the side (RPS showdown photo) | missed | unusual view, no visible palm |
| squirrel, train | **no false positive** | two filters: Net 1 score + Net 2 presence |

> [!tip] For the game
> Medium-size hands (players about 1.5–2 m from the webcam), hands kept apart, palms or backs of hands facing the camera.

---

## 14 · Where the course theory is

| Lecture | Where it shows up |
|---|---|
| L1 · colour images | BGR vs RGB; colour helps: black-and-white photo 10 failed |
| L2 · convolution | both nets are stacks of learned convolution filters (+ depthwise separable trick) |
| L5 · scale space | fixed 192×192 input → hands too big or too small are missed; two grids (8 px and 16 px cells) for two scales |
| L5 · non-maximum suppression | same idea as in Canny: keep the local maximum, remove its neighbours |
| L6 · geometric transformation | letterbox resize, rotated affine crop for Net 2, inverse transform back; distances are rotation-invariant (gesture rule) |
| L8 · template matching | Net 1 is like matching a learned palm template at every position and scale at once |

---

## 15 · Glossary

| Term | Meaning |
|---|---|
| **Anchor** | a fixed candidate box at a grid position; the net scores it and adjusts it |
| **NMS** | non-maximum suppression: among overlapping boxes keep the best one |
| **Letterbox** | resize keeping proportions, fill the rest with bars |
| **Affine transformation** | rotation + scale + translation (+ shear), as a 2×3 matrix |
| **Regression head** | the last layers that output numbers directly (here: 63 coordinates) |
| **Depthwise separable convolution** | per-channel filter, then 1×1 mixing: cheaper than a normal convolution |
| **Sigmoid** | squashes any number into 0–1, used for the scores |
| **Presence score** | Net 2's confidence that the crop really contains a hand |
| **Normalized coordinates** | position as a fraction of image width and height (0–1) |
| **World landmarks** | the 21 points in metres, relative to the hand's centre |
| **TFLite / float16** | compact model format for phones and laptops; weights stored with 16-bit precision |

---

## Flashcards

Which model does detect_hands.py use?::MediaPipe Hand Landmarker (float16), hand_landmarker.task: a ZIP with two TFLite networks, a palm detector and a hand landmark model.
What does Net 1 receive and return?::The whole photo letterboxed to 192×192; 2016 candidate boxes, each with 4 box values + 7 palm points, and a palm score.
Why exactly 2016 candidates?::Two grids: 24×24 cells × 2 anchors = 1152 and 12×12 cells × 6 anchors = 864.
How are 2016 candidates reduced to a few hands?::Keep scores above min_hand_detection_confidence, then non-maximum suppression removes overlapping boxes.
Why detect palms instead of whole hands?::A palm is nearly rigid and square, much easier to detect than bending fingers, and palms of close hands overlap less.
What happens between Net 1 and Net 2?::The palm box is enlarged to cover the fingers, rotated so wrist → middle finger points up, and resized to 224×224: one affine transformation.
Why rotate the hand crop upright?::Net 2 only has to learn upright hands, so it can be smaller and more precise.
What are Net 2's four outputs?::21×(x,y,z) in the crop, presence score, handedness, and 21×(x,y,z) in metres (world landmarks).
How does Net 2 produce coordinates?::Regression: fully connected heads output the 63 numbers directly.
What do lm.x, lm.y and lm.z mean?::x, y = fraction of image width/height (0–1); z = depth relative to the wrist, negative = closer to the camera.
How does our code get pixel coordinates and the box?::Multiply x by width and y by height; the box is the min and max of the 21 points.
Which landmark indices are the fingertips?::4, 8, 12, 16, 20; wrist = 0; finger bases 5, 9, 13, 17.
What is the rock–paper–scissors rule?::A finger is extended if its tip is farther from the wrist than its PIP joint. Rock = 0 of index–pinky extended, paper = all 4, scissors = index + middle only.
Why is the distance rule better than tip.y < pip.y?::Distances do not change with rotation, so sideways hands work too.
Why does padding help the close-up fist but hurt small hands?::Net 1 always works at 192×192; padding shrinks everything, so big hands fit, small hands become too small (scale problem).
Why is video mode faster than photo mode?::Tracking: the previous frame's points give the next crop, so the palm detector is skipped until the hand is lost.
Why are two touching palms detected as one?::Their boxes overlap strongly and NMS keeps only one.
