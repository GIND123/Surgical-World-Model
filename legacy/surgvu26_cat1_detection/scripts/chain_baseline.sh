#!/bin/bash
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
export YOLO_CONFIG_DIR=$PWD/.ultra
while pgrep -f extract_train.py > /dev/null; do sleep 20; done
echo "extraction done at $(date)"
sync
./env/bin/python scripts/frame_index.py > logs/frame_index.log 2>&1
echo "frame index rebuilt"
./env/bin/python scripts/train_cv.py --tag base_m640 --model yolo11m.pt --epochs 70 --imgsz 640 --batch 48
