#!/bin/bash
# After model B finishes, settle two questions on a properly held-out video (v5):
#   1. does training at 832 beat training at 640 on the same round-2 data?
#   2. does ensembling two PEER round-2 models help? (round1+round2 hurt: 0.3105 vs 0.3122)
# Both fold-5 students are trained from the fold-5-safe teacher, so video 5 stays unseen.
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
export YOLO_CONFIG_DIR=$PWD/.ultra
PY=./env/bin/python

while pgrep -f "[f]inal_train.py" > /dev/null; do sleep 30; done
echo "=== MODEL B DONE $(date) ==="

$PY scripts/selftrain.py --tag st3 --teacher_tag base_s --pdir v5b \
    --model yolo11s.pt --epochs 2 --imgsz 832 --batch 32 --folds 5 --oversample 5
echo "=== 832 FOLD5 STUDENT DONE $(date) ==="
