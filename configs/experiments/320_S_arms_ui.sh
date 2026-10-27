# Ablation (size S, 60k steps, batch 32): arms_ui
set -e
.venv/bin/python scripts/train/train_wm.py --name S_arms_ui --action arms --size S --use-ui --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/diagnose_actions.py --run S_arms_ui --windows 512
.venv/bin/python scripts/eval/eval_wm.py --run S_arms_ui --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_arms_ui --limit 300
