#!/bin/bash
# Wait for the final self-validating run, then score every epoch checkpoint on the
# held-out video with the real container pipeline and sweep inference resolution.
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
export YOLO_CONFIG_DIR=/tmp/ultra_cfg
PY=./env/bin/python

while pgrep -f "[s]elftrain.py --tag fin" > /dev/null; do sleep 60; done
echo "=== FINAL TRAINING DONE $(date) ==="

$PY scripts/eval_ckpts.py runs/fin/f5 5 runs/fin_ckpt_scores.json
echo "=== CKPT EVAL DONE $(date) ==="
