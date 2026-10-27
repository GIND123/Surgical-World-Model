# E2: action-free adaptation of S_global_vq to jigsaws videos (no kinematics used), for probing
set -e
.venv/bin/python scripts/train/train_wm.py --name jigsaws_adapt_global_vq --data jigsaws --action global_vq --size S --init S_global_vq --steps 5000 --bs 32 --warmup 500 --kl-warmup 1 --eval-every 2500 --save-every 2500 --resume
