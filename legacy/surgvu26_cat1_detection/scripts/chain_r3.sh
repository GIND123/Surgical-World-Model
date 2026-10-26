#!/bin/bash
# Round 3: teacher is r2_l832 (the 0.5020 leaderboard model), over the expanded 262k frame pool.
# Then the decisive ablation: an EPOCH CURVE on a held-out video, because the measured
# s-vs-m comparison at 2 epochs (0.3187 vs 0.3066) says the larger models are undertrained,
# not overparameterised.
set -e
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
export YOLO_CONFIG_DIR=/tmp/ultra_cfg
mkdir -p $YOLO_CONFIG_DIR
PY=./env/bin/python
T=runs/r2_l832/w/weights/last.pt
P=data/prior_r2_l832.json

echo "=== prior $(date) ==="
$PY - <<'EOF'
import os, json, collections
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
cc=collections.Counter()
A=json.load(open(f'{ROOT}/data/val_ann.json'))
for v in map(str,range(1,8)):
    for bs in A[v]['ann'].values():
        for c,*_ in bs: cc[c]+=5
L=f'{ROOT}/data/pseudo/r2_bal/labels'
for f in os.listdir(L):
    for line in open(f'{L}/{f}'): cc[int(line.split()[0])]+=1
json.dump([cc[i] for i in range(14)], open(f'{ROOT}/data/prior_r2_l832.json','w'))
print('prior written')
EOF

echo "=== round-3 pseudo passes $(date) ==="
$PY scripts/make_pseudo.py --weights $T --out data/pseudo/r3_a --sel selected_frames_xl \
    --conf 0.25 --require_full 0 --margin 1.0 --tau 0.5 --prior $P --batch 96 --workers 14
$PY scripts/make_pseudo.py --weights $T --out data/pseudo/r3_c --sel selected_frames_xl \
    --conf 0.02 --require_full 1 --margin 1.0 --tau 0.5 --prior $P --batch 96 --workers 14
$PY scripts/make_pseudo.py --weights $T --out data/pseudo/r3_b --sel selected_frames_xl \
    --conf 0.05 --require_full 1 --margin 1.0 --tau 0.5 --prior $P --batch 96 --workers 14
$PY scripts/build_balanced.py r3_a,r3_c,r3_b r3_all r3_bal
echo "=== ROUND3 DATA DONE $(date) ==="

ln -sfn "$PWD/data/pseudo/r3_bal" data/pseudo/r3b_f5
echo "=== epoch curve (yolo11s@832, video 5 held out) $(date) ==="
$PY scripts/selftrain.py --tag ec --teacher_tag base_s --pdir r3b \
    --model yolo11s.pt --epochs 12 --imgsz 832 --batch 32 --folds 5 --oversample 5 \
    --save_period 2
echo "=== EPOCH CURVE TRAINED $(date) ==="
