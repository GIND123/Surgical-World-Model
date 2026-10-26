#!/bin/bash
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
export YOLO_CONFIG_DIR=$PWD/.ultra
T=runs/final_m/w/weights/last.pt
P=data/prior_final_m.json
./env/bin/python scripts/make_pseudo.py --weights $T --out data/pseudo/r2_a --sel selected_frames_big \
   --conf 0.25 --require_full 0 --margin 1.0 --tau 0.5 --prior $P --batch 96 --workers 14
./env/bin/python scripts/make_pseudo.py --weights $T --out data/pseudo/r2_b --sel selected_frames_big \
   --conf 0.05 --require_full 1 --margin 1.0 --tau 0.5 --prior $P --batch 96 --workers 14
echo "R2 PSEUDO DONE $(date)"
