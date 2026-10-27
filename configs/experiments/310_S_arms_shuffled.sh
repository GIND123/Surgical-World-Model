# Ablation (size S, 60k steps, batch 32): arms_shuffled
set -e
.venv/bin/python scripts/train/train_wm.py --name S_arms_shuffled --action arms --size S --arm-pool shuffled --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/diagnose_actions.py --run S_arms_shuffled --windows 512
.venv/bin/python scripts/eval/eval_wm.py --run S_arms_shuffled --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_arms_shuffled --limit 300
