---
license: apache-2.0
tags:
  - object-detection
  - surgical-tool-detection
  - semi-supervised-learning
  - weakly-supervised-learning
  - endoscopy
  - yolo
  - medical-imaging
library_name: ultralytics
pipeline_tag: object-detection
---

# SurgVU 2026 — Category 1: Surgical Tool Detection & Localization (Team DiceMed)

Surgical-tool detector for the [MICCAI EndoVis SurgVU 2026](https://surgvu26.grand-challenge.org/)
Category 1 challenge: localize and classify robotic instruments in da Vinci endoscopic video,
training from **noisy tool-presence labels** (which instrument is installed on which robot arm)
plus a **small box-annotated validation set**.

This repository contains the trained weights, the exact Grand Challenge inference container, and
the full training/pseudo-labelling code.

## Results

| Submission | Model | Hidden-test mAP@[.5:.95] |
|---|---|---|
| 1 — `DiceMed SurgeAI` | yolo11m @640, round-1 pseudo-labels | **0.4874** |
| 2 — `DiceMed ImageSurgery` | yolo11l **@832**, round-2 balanced pseudo-labels, 7 videos | **0.5020** (2nd) |
| 3 — final | yolo11l @832, **round-3** pseudo-labels (1.84x the boxes), 7 videos | *submitted* |

All are inferred at **640** with 4-scale + flip TTA — training at 832 but inferring at 640 measured
best (see below).

### Evidence for the final model

There is **no honest leave-one-video-out number for the final model** — the round-3 holdout is
contaminated (see "Validation integrity" below). The evidence is instead an uncontaminated presence
metric (`scripts/presence_eval.py`): 4000 keyframes from the 280 training videos that were never
selected for pseudo-labelling, scored against the installed-instrument multiset from `tools.csv`.
The same seeded frames are used for every checkpoint, so comparisons are paired.

| checkpoint | f1_topk | f1_conf | count_match |
|---|---|---|---|
| `r2_l832` — the model that scored **0.5020** | 0.5339 | 0.5176 | 0.243 |
| round-3 warm-start source | 0.5580 | 0.5250 | 0.260 |
| + 1 epoch on all 7 videos | 0.5663 | 0.5321 | 0.273 |
| **final (4 more epochs, proven schedule)** | **0.5681** | **0.5364** | 0.269 |

**This measures detection + classification only, not box IoU**, so it does not translate into a
predicted hidden-test mAP; the known bottleneck is high-IoU box precision, and no uncontaminated
box-quality signal exists (box ground truth exists only for the 7 public videos, all of which the
round-3 teacher saw). Read it as: the final model is meaningfully better than the 0.5020 model at
finding and naming the right instruments, on data neither has trained on.

Reference: 1st place 0.5376, 3rd place 0.4012. Previous editions' winners: 0.4671 (SurgVU 2025),
0.4669 (SurgToolLoc 2023).

### Held-out validation (leave-one-video-out, video 5, matched yolo11s students)

| Setting | v5 mAP@[.5:.95] |
|---|---|
| public-val-only baseline | 0.1898 |
| + round-1 pseudo-labels | 0.2840 |
| + round-1 pseudo-labels, tuned TTA | 0.2930 |
| + round-2 balanced pseudo-labels | 0.2969 |
| + round-2 balanced, tuned TTA | **0.3122** |

Round 2 wins despite a *smaller* training budget (2 epochs vs 5), so the comparison is biased
against it. `force bipolar`, which appears only in video 5 and therefore has zero labelled examples
in that fold's training data, goes from **AP 0.000 → 0.097** purely through pseudo-labelling.

### Training vs inference resolution (same round-2 data, matched yolo11s students, video 5 held out)

| Trained at | Inferred at | TTA | v5 mAP |
|---|---|---|---|
| 640 | 640 | 4-scale + flip | 0.3122 |
| **832** | **640** | **4-scale + flip** | **0.3187** |
| 832 | 832 | 4-scale + flip | 0.3123 |
| 832 | 832 | 3-scale + flip | 0.3126 |
| ensemble of the two (peers) | 640 | 4-scale + flip | 0.3155 |

Training at higher resolution helps (+2.1% relative), but **inferring at the native 640 is better
than inferring at 832** — the TTA scale ladder (0.8–1.5 × 640 = 512–960 px) already covers the
useful range, and re-basing at 832 overshoots it. Ensembling two comparably strong round-2 models
still lost to the better single model.

## Method

### 1. Domain normalisation

The 280 training videos are 1280×720 @60 fps but the content is **pillarboxed** — the real image
is a ~5:4 region inside the 16:9 frame. Calibrating by maximising normalised cross-correlation of
the static da Vinci UI against the test-set median image gives the crop **x∈[193,1087), y∈[4,717)**,
which resized to **640×512** matches the test frames exactly (NCC 0.851 vs 0.830 for a naive crop).

The da Vinci UI **spells out the tool names in plain text** at the bottom of the frame. The
organisers blur the UI in the hidden test set and forbid using it. A model trained on unmasked
frames can read the label and score well locally while collapsing on the hidden test, so the static
UI bands (rows `0–31` and `488–511`, measured from per-video temporal row-variance) are **blacked
out identically at training and inference**.

Keyframe-only decoding (`skip_frame='NONKEY'`) runs at ~900× realtime with a ~4.17 s keyframe
interval → **725,124 training frames**.

### 2. Presence-constrained pseudo-labelling

`tools.csv` gives, per robot arm (USM1–USM4), which instrument is installed over which time span.
That is a *frame-level multiset*, not boxes. To turn it into box labels:

1. A teacher detector proposes boxes with a 14-way class posterior.
2. The frame's installed instruments form the candidate label set.
3. Solve the best **injective matching** of boxes → installed arms, scored by
   `log p(class | box)` plus a spatial prior (USM1→USM4 run left→right), with
   **logit adjustment** (`log p − τ·log p_train`) so long-tail classes stay competitive.

The key property: when the number of detected boxes equals the number of installed instruments,
the matching is a *perfect* matching, so **every installed arm is forced to receive a box**. A class
the teacher has never seen is then recovered purely **by elimination** — this is how classes absent
from the public validation set (stapler, prograsp forceps, tip-up fenestrated grasper, permanent
cautery hook/spatula, bipolar dissector) acquire training boxes at all. Measured per-class yield of
this branch is **1.00**.

Two passes are merged, elimination labels taking precedence:
- volume pass (`conf 0.25`, partial matching allowed),
- elimination pass (`conf 0.02–0.05`, full matching required).

Frames containing rare classes are then **oversampled** (up to 24×), cutting the head:tail box
imbalance from 172:1 to ~20:1.

### 3. Self-training rounds

- **Round 1** — teacher trained on the public validation boxes only → 62k pseudo frames / 111k boxes.
- **Round 2** — the round-1 student (the 0.4874 model) becomes the teacher → **131k frames / 270k
  boxes**, and after rare-class balancing **153k frames**. Round 2 tripled prograsp forceps (3.9×)
  and roughly doubled most other classes.

- **Round 3** — the round-2 `yolo11l@832` student (the 0.5020 model) becomes the teacher, over an
  expanded 262k-frame pool → after balancing **266,356 frames / 609,921 boxes**, 1.84× round 2's
  332,100 boxes. The elimination full-match rate rose from 22.9% to 31.7%, so a larger share of
  frames carry *exactly* determined labels. Rare classes grew most: bipolar dissector 7.4×,
  cadiere forceps 2.5×, stapler 2.5×, clip applier 2.3×, force bipolar 2.2×, tip-up 2.1×.

Students train on pseudo-labels **plus the public validation GT oversampled 5×**.

### Validation integrity — read this before trusting any round-3 number

Rounds 1 and 2 used a **per-fold teacher**: fold *v*'s teacher never saw video *v*, so its
pseudo-labels carried no information about the held-out video and the leave-one-video-out estimate
was honest. **Round 3 broke this.** It generated all 266k pseudo-labels from a single teacher,
`r2_l832`, which was trained on all seven annotated videos. Every round-3 pseudo-label therefore
carries video-5-derived knowledge, and any student trained on them inherits it.

The fold itself is clean — there are no video-5 ground-truth frames in the round-3 training set —
so the leak is entirely teacher-mediated and invisible from the fold. The tell is arithmetic:
round-3 students score ~0.66-0.68 on video 5 against a round-2 reference of 0.3187, and a
mAP@[.5:.95] of 0.68 is impossible when that reference model's *mAP50* was only 0.6965. The likely
mechanism is the known bottleneck: per-video annotator box convention, learned from video 5's
ground truth by the teacher and baked into every pseudo-box.

Consequently, round-3 video-5 scores are used **only to rank checkpoints within a single run**
(all checkpoints share one identical training set, so the contamination is a common level shift,
while a genuine decline still exposes overfitting to pseudo-label noise). They are **not**
comparable to rounds 1-2 and are **not** an estimate of hidden-test performance. No such number is
reported as a held-out result anywhere in this card.

Because no clean holdout remains among the seven public videos, the final model is trained on
**all seven**, as the 0.5020 model was. Video 5 matters disproportionately: it holds the only
ground-truth `force bipolar` boxes, a class that scores just 0.097 from pseudo-labels alone.

To restore an honest estimate, regenerate pseudo-labels per fold with a teacher that never saw that
fold's video.

### 4. Inference

3-scale (0.8/1.0/1.25) + horizontal-flip TTA → class-agnostic NMS → IoU track linking →
track-level class voting (α=0.35) → top-5 class hypotheses emitted per box.

Track voting exploits the fact that a track is one physical instrument, so its class is constant;
at 1 fps consecutive GT boxes still overlap strongly (median IoU 0.74, only 11% with zero overlap).

The container **measures its own throughput at runtime** and picks the richest TTA/ensemble plan
that fits the 10-minute-per-case limit, degrading gracefully and always emitting valid JSON — one
case over the limit fails the entire submission.

## Evaluation notes (verified against the official evaluator)

Reading [`isi-challenges/surgvu24-category-1-eval`](https://github.com/isi-challenges/surgvu24-category-1-eval):

- **`slice_nr` must be the 0-based frame index.** The evaluator subtracts 1 from *ground-truth*
  slice numbers but not from predictions. The shipped GT `_gc.json` files use `frame+1`, so copying
  their convention shifts every prediction by one frame: measured **0.7304 → 0.2333** on a real video.
- The score is a **macro-average over videos** of per-video COCO mAP@[.5:.95] — short videos count
  as much as long ones.
- COCO AP is per-class and classes absent from a video's GT are dropped, so **false positives of an
  absent class are free**. This makes recall on rare classes asymmetrically valuable.
- Predictions on frames without GT are discarded, so predicting on every frame costs nothing.

## What did not work (measured, so you don't repeat it)

| Idea | Effect on held-out video |
|---|---|
| Multi-scale box emission (hedging annotation looseness) | −0.016 to −0.029 |
| Weighted box fusion instead of NMS for TTA/ensembles | −0.003 (box-only) to −0.032 (mean) |
| GT-only low-LR fine-tune after pseudo pretraining | −0.049 |
| Ensembling the round-1 model with the round-2 model | −0.002 (0.3105 vs 0.3122 for round-2 alone) |
| 5-scale TTA | −0.003 vs 4-scale |
| Training >30 epochs on the 4k public-val frames | overfits (0.433 @ep5 → 0.40 @ep60) |
| Larger backbone on public-val alone | yolo11m ≈ yolo11s — data diversity, not capacity, is the limit |
| Larger backbone at a **short** budget | yolo11m@832 0.3066 < yolo11s@832 0.3187 — undertrained, not overparameterised |
| Lengthening the run to 12 epochs | **model collapsed at epoch 4** (see below) |
| Passing `--lr0` with `optimizer=auto` | silently ignored |

Two failure modes cost most of the final day and are worth stating plainly.

**`close_mosaic` changes meaning when you change `epochs`.** It fires at
`epoch == epochs - close_mosaic`. At the default `close_mosaic=10`, a 4-epoch run never fires it
(negative), but a 12-epoch run closes mosaic at epoch 3 — and the model collapsed at epoch 4:
presence f1_topk 0.5580 → **0.2301**, `count_match` 0.260 → **0.004**, i.e. it stopped emitting
confident detections at all, and had not recovered nine epochs later. **The training loss falls
smoothly straight through the collapse**, because it is an epoch average while the damage is in the
end-of-epoch weights. Train loss is therefore useless as a health check; score checkpoints
externally, every epoch.

**`optimizer=auto` ignores `lr0`.** It logs `ignoring 'lr0=...' and determining best 'optimizer',
'lr0' and 'momentum' automatically` and selects MuSGD(lr=0.01); the realised `lr/pg0` is 0.03 × the
decay factor. Passing `--lr0 0.005` changed nothing. The 0.5020 run stayed in a safe 0.0100–0.0151
band only because its 3-epoch warmup never completed inside a 4-epoch run. Control the realised LR
through `warmup_epochs` relative to `epochs`, or set `optimizer` explicitly.

For the record, high LR was *not* the cause: the checkpoint trained at the highest LR (0.03) scored
the best of its round (0.5663). That wrong diagnosis came from comparing train losses across two
runs that used different folds, where losses are not comparable.

The ensembling result is worth stressing: naively concatenating a weaker model's boxes and running
class-agnostic NMS lets the weaker model win boxes it should not, so a "bigger ensemble" was a
regression. Ensemble only models of comparable strength.

## Known ceiling

Median box area for the **same class** varies up to **4.6×** across the seven public videos (one
video is uniformly loose, others tight) — partly camera zoom, partly annotator convention. This
caps attainable IoU and therefore mAP@[.5:.95], and it is why hedging tricks like multi-scale
emission fail: there is no way to know which convention a given test video follows.

## Usage

```python
from ultralytics import YOLO
m = YOLO("model/det0.pt")            # 14 classes, 640×512 input
```

Class order (COCO category id 0–13):

```
grasping retractor, cadiere forceps, bipolar forceps, force bipolar, clip applier,
stapler, permanent cautery hook spatula, monopolar curved scissor, vessel sealer,
tip up fenestrated grasper, bipolar dissector, needle driver, prograsp forceps,
suction irrigator
```

Full Grand Challenge container:

```bash
./do_build.sh surgvu26-cat1 && ./do_save.sh surgvu26-cat1
```

Inference expects `/input/endoscopic-robotic-surgery-video.mp4` and writes
`/output/surgical-tools.json` in the `Multiple 2D bounding boxes` format.

## Data

Trained on the SurgVU/SurgToolLoc challenge dataset (280 da Vinci training videos, ~840 h, tool
presence labels; plus the 7-video box-annotated public validation set). The dataset is distributed
by the challenge organisers and is not redistributed here.

## Citation

```bibtex
@misc{surgvu2026_dicemed,
  title  = {Presence-Constrained Pseudo-Labelling for Weakly-Supervised Surgical Tool Detection},
  author = {Team DiceMed},
  year   = {2026},
  note   = {SurgVU 2026 Category 1, MICCAI EndoVis}
}
```
