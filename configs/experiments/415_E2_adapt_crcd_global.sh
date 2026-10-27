# E2: action-free adaptation of S_global to crcd videos (no kinematics used), for probing
set -e
.venv/bin/python scripts/train/train_wm.py --name crcd_adapt_global --data crcd --action global --size S --init S_global --steps 5000 --bs 32 --warmup 500 --kl-warmup 1 --eval-every 2500 --save-every 2500 --resume
