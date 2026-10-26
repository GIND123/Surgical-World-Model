#!/bin/bash
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
export YOLO_CONFIG_DIR=$PWD/.ultra
while pgrep -f "[t]rain_cv" > /dev/null; do sleep 15; done
echo "=== baseline done $(date) ==="
for v in 1 5 6 7; do
  echo "=== pseudo fold $v ==="
  ./env/bin/python scripts/make_pseudo.py \
     --weights runs/base_s/f$v/weights/last.pt \
     --out data/pseudo/p1_f$v --conf 0.25 --margin 1.0 --batch 96 --workers 12
done
echo "ALL PSEUDO DONE $(date)"
