# Surgical Image — SurgVU 2026 Tool Detection

Surgical instrument detection and localization for the **MICCAI EndoVis SurgVU 2026 Category 1** challenge, developed by **Team DiceMed**.

The project trains a 14-class YOLO detector from noisy, frame-level tool-presence labels and a small box-annotated validation set. Its main approach is presence-constrained pseudo-labelling, followed by self-training, test-time augmentation, class-agnostic NMS, and track-level class voting.

## Results

| Submission | Model | Hidden-test mAP@[.5:.95] |
| --- | --- | ---: |
| DiceMed SurgeAI | YOLO11m @ 640, round-1 pseudo-labels | 0.4874 |
| DiceMed ImageSurgery | YOLO11l @ 832, round-2 balanced pseudo-labels | **0.5020** |
| Final | YOLO11l @ 832, round-3 pseudo-labels | Submitted |

See [the model card](work/MODEL_CARD.md) for the full method, validation caveats, ablations, and per-class details.

## Repository layout

```text
work/
├── MODEL_CARD.md       # Method, results, limitations, and model details
├── scripts/            # Data preparation, training, evaluation, and inference
├── submission/         # Submission variant 1
├── submission2/        # Submission variant 2
├── submission3/        # Submission variant 3
├── submission4/        # Submission variant 4
└── submission5/        # Latest container variant
```

Datasets, generated runs, archives, and model weights are intentionally excluded from Git. In particular, `*.pt` files are ignored. Obtain the challenge data from the organizers and place the required weights locally before running training or inference.

## Requirements

- Linux
- Python 3.10+
- NVIDIA GPU with CUDA for practical training and inference
- Docker with NVIDIA Container Toolkit for the Grand Challenge container
- Ultralytics `8.3.155`
- NumPy `1.26.4`

For a lightweight local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ultralytics==8.3.155 numpy==1.26.4 opencv-python-headless==4.10.0.84
```

## Quick model usage

Once you have a trained checkpoint:

```python
from ultralytics import YOLO

model = YOLO("path/to/det0.pt")
results = model.predict("path/to/frame.jpg", imgsz=640)
```

The 14 classes, in category-ID order, are:

```text
grasping retractor
cadiere forceps
bipolar forceps
force bipolar
clip applier
stapler
permanent cautery hook spatula
monopolar curved scissor
vessel sealer
tip up fenestrated grasper
bipolar dissector
needle driver
prograsp forceps
suction irrigator
```

## Build the inference container

The latest submission implementation is in `work/submission5`. Copy a compatible checkpoint into its model directory as `det0.pt`, then build the image:

```bash
cp path/to/checkpoint.pt work/submission5/model/det0.pt
cd work/submission5
./do_build.sh surgvu26-cat1
```

Save the image as a compressed archive for upload:

```bash
./do_save.sh surgvu26-cat1
```

The container expects:

```text
/input/endoscopic-robotic-surgery-video.mp4
```

and writes Grand Challenge `Multiple 2D bounding boxes` JSON to:

```text
/output/surgical-tools.json
```

Inference behavior is controlled by [`work/submission5/model/config.json`](work/submission5/model/config.json), including batch size, tracking, confidence thresholds, TTA plans, and runtime budget adaptation.

## Training pipeline

The scripts under `work/scripts` cover:

- frame extraction and dataset construction;
- presence-label parsing;
- constrained pseudo-label generation;
- rare-class balancing;
- cross-validation and final training;
- checkpoint, COCO, presence, and temporal evaluation;
- container packaging.

The scripts reflect the original experiment environment and some contain absolute paths. Before running them on another machine, update paths such as `ROOT` and `PY`, then arrange the challenge datasets in the locations expected by `work/scripts/ds.py` and related preparation scripts.

The original final training flow is implemented in [`work/scripts/final_train.py`](work/scripts/final_train.py):

```bash
cd work
python scripts/final_train.py --stage teacher
python scripts/final_train.py --stage pseudo
python scripts/final_train.py \
  --stage student \
  --model yolo11l.pt \
  --epochs 5 \
  --imgsz 832 \
  --batch 64 \
  --name final_l
```

These commands are GPU- and data-intensive; review the script paths and parameters before launching a run.

## Important evaluation detail

Predicted `slice_nr` values must be the **zero-based video frame index**. The official evaluator adjusts ground-truth slice numbers but not prediction slice numbers, so using one-based prediction indices causes a severe score drop.

## Data

The project was trained on the SurgVU/SurgToolLoc challenge data: 280 da Vinci training videos with tool-presence annotations and seven public validation videos with bounding boxes. The data is not redistributed in this repository. Refer to the [SurgVU 2026 challenge website](https://surgvu26.grand-challenge.org/) for access and terms.

## Citation

```bibtex
@misc{surgvu2026_dicemed,
  title  = {Presence-Constrained Pseudo-Labelling for Weakly-Supervised Surgical Tool Detection},
  author = {Team DiceMed},
  year   = {2026},
  note   = {SurgVU 2026 Category 1, MICCAI EndoVis}
}
```

## License

This project is described as Apache-2.0 in the model metadata. Add a root `LICENSE` file before distributing the repository as an Apache-2.0-licensed release.
