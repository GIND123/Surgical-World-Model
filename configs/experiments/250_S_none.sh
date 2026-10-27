# Controlled comparison of action interfaces (size S, 60k steps, batch 32): none
set -e
.venv/bin/python scripts/train/train_wm.py --name S_none --action none --size S --no-camera --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/diagnose_actions.py --run S_none --windows 512
.venv/bin/python scripts/eval/eval_wm.py --run S_none --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_none --limit 300
