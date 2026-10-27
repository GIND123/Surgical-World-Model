# 01 — Data audit: local SurgVU holdings

*Audit date: 2026-10-27. Every number here is produced by `scripts/data/audit_surgvu.py`, `scripts/data/audit_timeline.py` and `scripts/data/audit_frames.py` (outputs in `outputs/audit/`, Git-ignored).*

## 1. What is on disk

| Asset | Location (`data/raw/…`) | Content | Size |
|---|---|---|---|
| SurgVU24 videos | `surgvu24/videos/case_XXX/case_XXX_video_part_00N.mp4` | 155 da Vinci porcine training-lab cases, 280 video files | 0.35 TB |
| SurgVU24 labels (v2) | `surgvu24/labels/case_XXX/{tools,tasks}.csv` | robot-log arm occupancy (incl. endoscope) + task segments | 2.8 MB |
| SurgVU25 labels | `surgvu25/labels/case_XXX/{tools,tasks}.csv` | same cases; adds 46 fine-grained task names and 21 free-text task descriptions | 4.0 MB |
| SurgVU25 Cat-2 sample | `surgvu25/cat2_sample_set_public/` | 11 short clips with a yes/no VQA question and paraphrased answers | 188 MB |
| SurgVU26 Cat-1 public test | `surgvu26/cat1_test_set_public/` | 7 videos at 1 fps, 5,178 frames, 11,323 human-annotated boxes, 14 instrument classes | 168 MB |

The previous project's artefacts (detector runs, pseudo-labels, extracted frames, challenge containers) were moved to `archive/surgvu26_cat1/`; see `docs/legacy.md`.

## 2. Videos

* **280 files, 837.8 h, all decodable.** Every file was opened and decoded at 5 %, 50 % and 95 % of its length — 0 errors.
* **Uniform format:** H.264, 1280×720, 60 fps, yuv420p, ~1 Mb/s, no audio.
* **155 cases:** 30 single-part, 125 two-part. Case length median 5.75 h (IQR 5.18–6.25 h; range 0.44–9.44 h).
* **One missing part:** `case_071` part 1 was never delivered; the folder holds only a zero-filled, undecodable partial upload (`case_video_000_001.mp4_.gstmp`, 1.9 GB). Labels in that part are dropped and intervals that continue into part 2 are clipped to its start (`swm.data.surgvu._clip_missing_parts`).
* **Stray files** (ignored by all loaders): `.DS_Store`, `.cnvrgignore`, the `.gstmp` above, and macOS `__MACOSX/._*` resource forks (these doubled a naive `*.mp4` count to 560; there are 280 real videos).

### Frame content (50,269 keyframes, one per minute of video)

| Class | Share | Rule (`audit_frames.classify`) |
|---|---:|---|
| tissue (usable) | 70.4 % | everything else |
| black | 23.1 % | mean luma < 25 |
| low colour (out of body, smoke, drapes) | 4.5 % | mean saturation < 45 |
| fluorescence (Firefly / ICG) | 1.7 % | > 2 % saturated green pixels |
| blank / blurred | 0.3 % | variance of Laplacian < 15 |

* **≈ 604 h usable** (tissue + fluorescence). The black share is spread evenly (median 24 % per case; only one case is > 50 % black) — docking, idle and camera-out periods, not corrupt files.
* **Geometry:** the endoscope image is pillarboxed to columns 193–1086 (pillar luma ≤ 0.006 at the 99th percentile).
* **Console UI is burned into the video.** On usable frames the bottom arm-panel band (rows 684–718) is present 81 % of the time and the top warning banner ("TRAINING INSTRUMENT -- NOT FOR HUMAN USE", rows 8–50) 45 %; 125 cases show the panels most of the time, 28 rarely. The panels display, per arm, the mounted instrument, the camera arm, **which hand (L/R) currently drives which arm**, and energy state (CUT/COAG) — a frame-synchronous robot-state readout (§4).

A contact sheet of random frames is written locally to `outputs/audit/frames/contact_sheet.jpg`; dataset frames are not committed to this public repository.

## 3. Robot-log labels on one timeline

Label times are *seconds within a video part*; `swm.data.surgvu.timeline` converts them to global case time (`offset[part] + t`). Treating them as global time — or ignoring the part — produces the 771 "negative-duration" tool intervals and 36 task intervals a naive parse reports.

### Arm occupancy (`tools.csv`)

* **12,101 rows → 8,544 usable intervals** after removing 3,210 exact duplicate rows, 348 inverted intervals (uninstall before install) and 57 rows in the missing part. 144 rows are "Unknown Instrument" with no ground-truth name; they are kept as an explicit `unknown` occupant.
* **All four arms (USM1–USM4) are logged, including the endoscope** (`nan(camera in)`: 0° and 30° scopes). The camera arm is logged for a median 99.4 % of each case. USM2 holds it for most of the case in 142 cases and USM3 in 11, but the **endoscope moves between ports during most cases: 211 port changes in 104 of 155 cases** (counting a change when another arm holds the camera for over 60 s; changes usually pass through a gap with no endoscope mounted). *Corrected 2026-10-27: an earlier version counted only direct hand-overs and wrongly reported no changes.*
* **Concurrent instruments** (hours): 0 → 119.5, 1 → 81.4, 2 → 85.7, **3 → 541.4**, 4 → 9.9 (the last is log noise: only three instrument arms exist besides the camera).
* **5,969 instrument exchanges** (7.1 per hour).
* **Instrument hours:** needle driver 546, cadiere forceps 359, bipolar forceps 305, monopolar curved scissors 251, prograsp forceps 128, grasping retractor 106, unknown 73, vessel sealer 45, force bipolar 41, permanent cautery hook/spatula 21, stapler 17, clip applier 13, tip-up fenestrated grasper 4.6, and < 2 h each for SynchroSeal, suction irrigator, bipolar dissector, crocodile grasper, Potts scissors, tenaculum forceps and single-site curved scissors.
* **Known label noise — missing uninstall events.** 2,128 pairs of intervals overlap on the same arm with different occupants; 2,071 (97 %) are *fully contained* inside a much longer interval (median overlap 7.8 min), i.e. an uninstall was not logged and the old instrument's interval runs on. `arm_occupancy` resolves overlaps by "latest install wins", which is correct inside the contained interval but can revert to the stale instrument afterwards. **To do:** validate both resolution policies against the on-screen arm panels before using occupancy as conditioning.

### Tasks (`tasks.csv`)

* SurgVU24 v2: 1,178 segments, 7 tasks. SurgVU25 keeps all of them, adds 672 segments and an `Other` class, and replaces 456 task names; we use SurgVU25.
* **218 h annotated (26 % of video):** Suturing 75.9 h, Uterine horn 43.7, Rectal artery/vein 38.1, Suspensory ligaments 24.2, Skills application 17.0, Retraction & collision avoidance 9.5, Range of motion 5.6, Other 4.1.
* Behind the 7 tasks are **46 curriculum-level names** (e.g. "Training Topic 4 (a) – 3rd arm no bladder", "Training Topic 7 – 1-hand suture", "Skills Drills – luminal closure – interrupted", "Fluorescence imaging demonstration", "undocking") and **21 free-text procedure descriptions** — usable as language conditioning or task context.

## 4. What this means for a world model

1. **Scale.** ~604 usable hours from one embodiment (da Vinci Xi, porcine) is the largest robotic-surgery video corpus we are aware of that carries robot-log state — larger than SurgToolLoc-2022 used by SurgWM (24,695 × 30 s ≈ 206 h, used at 1 fps).
2. **No kinematics, but the embodiment is observable.** For every instant we know the camera arm and the instrument on every other arm (logs), and on 81 % of usable frames the UI additionally shows which arms the surgeon's hands drive and when energy fires. This is the basis of the research hypothesis in `docs/04_research_plan.md`: structure the world model's action space like the robot (camera + per-arm actions) instead of a single entangled latent action.
3. **Shortcut risk.** The UI spells out instrument names and states. Models must see only the endoscope field (crop rows 56–679 of the active area) so that they cannot read the answer off the overlay; the readout is used only to build conditioning/labels.
4. **Quality filtering is mandatory**: drop black, low-colour and blank windows; keep fluorescence as a tagged sub-domain.
5. **Splits must be by case** (frames within a case are highly correlated). The legacy detector was trained on pseudo-labels from all 155 cases, so detector-based *evaluation* metrics will use a re-trained, test-case-free detector.
