# Controlled comparison of action interfaces (size S, 60k steps, batch 32): slots
set -e
.venv/bin/python scripts/train/train_wm.py --name S_slots --action slots --size S  --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/eval_wm.py --run S_slots --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_slots --limit 300
