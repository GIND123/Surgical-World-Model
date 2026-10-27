# Controlled comparison of action interfaces (size S, 60k steps, batch 32): global_nocam
set -e
.venv/bin/python scripts/train/train_wm.py --name S_global_nocam --action global --size S --no-camera --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/diagnose_actions.py --run S_global_nocam --windows 512
.venv/bin/python scripts/eval/eval_wm.py --run S_global_nocam --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_global_nocam --limit 300
