#!/bin/bash
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
export YOLO_CONFIG_DIR=$PWD/.ultra
# wait until fold v6 has been reported, then stop the CV run
while [ "$(grep -ac 'fold v' logs/st1.log)" -lt 2 ]; do sleep 20; done
echo "=== fold6 done, stopping CV $(date) ==="
pkill -9 -f "[s]elftrain" || true
sleep 8
./env/bin/python scripts/final_train.py --stage teacher --tmodel yolo11s.pt --tepochs 25 --batch 64
./env/bin/python scripts/final_train.py --stage pseudo
./env/bin/python scripts/final_train.py --stage student --model yolo11m.pt --epochs 4 --batch 64 --name final_m --oversample 5
echo "=== FINAL PIPELINE DONE $(date) ==="
