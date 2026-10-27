# Ablation (size S, 60k steps, batch 32): arms_poolglobal
set -e
.venv/bin/python scripts/train/train_wm.py --name S_arms_poolglobal --action arms --size S --arm-pool global --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/eval_wm.py --run S_arms_poolglobal --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_arms_poolglobal --limit 300
