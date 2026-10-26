# Legacy: the SurgVU 2026 Category-1 detection project

This working directory previously held Team DiceMed's entry to the MICCAI EndoVis **SurgVU 2026 Category 1** (surgical tool detection) challenge — a 14-class YOLO detector trained from frame-level tool-presence labels with presence-constrained pseudo-labelling (best hidden-test mAP@[.5:.95] 0.5020, YOLO11l @ 832). Its full Git history remains at <https://github.com/Ajogeorge29/surgical_Image> and on the local branch `legacy/surgvu26-cat1`.

## What carries over

The detector is reused by the world-model project to localise instruments (arm-region binding) and to score rollouts (Embodiment Consistency Score). Its code is kept in `legacy/surgvu26_cat1_detection/` (scripts, model card, final inference container; the challenge's test video is not redistributed). For evaluation we re-train it without the world-model test cases.

## Where things moved (2026-10-27)

All moves are renames on the same filesystem; nothing was deleted.

| Old path | New path | In Git? |
|---|---|---|
| `surgvu24_videos_only/surgvu24/` | `data/raw/surgvu24/videos/` | no |
| `surgvu24_labels_updated_v2/labels/` | `data/raw/surgvu24/labels/` | no |
| `SURGVU25_cat2_train_labels/SURGVU25_train_labels/` | `data/raw/surgvu25/labels/` | no |
| `SURGVU25_cat_2_sample_set_public/` | `data/raw/surgvu25/cat2_sample_set_public/` | no |
| `cat1_test_set_public/` | `data/raw/surgvu26/cat1_test_set_public/` | no |
| `surgvu26-cat1-*.tar.gz`, `surgvu26-cat1-r3_7v/` | `archive/surgvu26_cat1/release_archives/` | no |
| `work/` (frames, pseudo-labels, runs, env, logs) | `archive/surgvu26_cat1/work/` | no |
| `weights/`, `.tools/` | `archive/surgvu26_cat1/{weights,tools}/` | no |
| empty wrapper folders (only `__MACOSX` left) | `archive/surgvu26_cat1/raw_wrappers/` | no |
| `work/scripts/`, `work/MODEL_CARD.md`, `work/submission5/` | `legacy/surgvu26_cat1_detection/` | yes |
| best detector `work/runs/r2_l832/w/weights/best.pt` | symlink `checkpoints/detector/yolo11l_r2_l832.pt` | no |

Legacy scripts contain absolute paths to the old layout; they are kept for reference, not maintained.
