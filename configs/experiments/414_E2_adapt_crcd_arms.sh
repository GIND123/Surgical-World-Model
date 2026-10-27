# E2: action-free adaptation of S_arms to crcd videos (no kinematics used), for probing
set -e
.venv/bin/python scripts/train/train_wm.py --name crcd_adapt_arms --data crcd --action arms --size S --init S_arms --steps 5000 --bs 32 --warmup 500 --kl-warmup 1 --eval-every 2500 --save-every 2500 --resume
