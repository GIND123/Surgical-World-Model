#!/bin/bash
# 1) wait for model A (yolo11l @640 on round-2 balanced data)
# 2) honest fold-5 check: does round-2+balanced data beat round-1 at a MATCHED sample budget?
#    teacher for fold 5 is st1/f5, which never saw video 5, so the estimate stays clean
# 3) model B (yolo11m @832) for the ensemble
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
export YOLO_CONFIG_DIR=$PWD/.ultra
PY=./env/bin/python

while pgrep -f "[f]inal_train.py" > /dev/null; do sleep 30; done
echo "=== MODEL A DONE $(date) ==="

# ---- fold-5 validation of the round-2 recipe ----
$PY scripts/make_pseudo.py --weights runs/st1/f5/weights/last.pt --out data/pseudo/v5_a \
    --sel selected_frames_big --conf 0.25 --require_full 0 --margin 1.0 \
    --tau 0.5 --prior data/prior_final_m.json --batch 96 --workers 14
$PY scripts/make_pseudo.py --weights runs/st1/f5/weights/last.pt --out data/pseudo/v5_c \
    --sel selected_frames_big --conf 0.02 --require_full 1 --margin 1.0 \
    --tau 0.5 --prior data/prior_final_m.json --batch 96 --workers 14
$PY scripts/build_balanced.py v5_a,v5_c v5_all v5_bal
echo "=== FOLD5 PSEUDO DONE $(date) ==="

$PY scripts/selftrain.py --tag st2 --teacher_tag base_s --pdir v5_bal \
    --model yolo11s.pt --epochs 2 --imgsz 640 --batch 64 --folds 5 --oversample 5
echo "=== FOLD5 VALIDATION DONE $(date) ==="

# ---- model B ----
$PY scripts/final_train.py --stage student --pdir r2_bal \
    --model yolo11m.pt --epochs 5 --imgsz 832 --batch 32 --name r2_m832 --oversample 5 --seed 1
echo "=== PHASE2 DONE $(date) ==="
