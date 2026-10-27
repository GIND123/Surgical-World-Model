# E2: action-free adaptation of S_slots to suturebot videos (no kinematics used), for probing
set -e
.venv/bin/python scripts/train/train_wm.py --name suturebot_adapt_slots --data suturebot --action slots --size S --init S_slots --steps 5000 --bs 32 --warmup 500 --kl-warmup 1 --eval-every 2500 --save-every 2500 --resume
