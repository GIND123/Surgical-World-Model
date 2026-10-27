# Controlled comparison of action interfaces (size S, 60k steps, batch 32): state_nocam
set -e
.venv/bin/python scripts/train/train_wm.py --name S_state_nocam --action state --size S --no-camera --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/eval_wm.py --run S_state_nocam --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_state_nocam --limit 300
