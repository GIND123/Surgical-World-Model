# E2: action-free adaptation of S_global to suturebot videos (no kinematics used), for probing
set -e
.venv/bin/python scripts/train/train_wm.py --name suturebot_adapt_global --data suturebot --action global --size S --init S_global --steps 5000 --bs 32 --warmup 500 --kl-warmup 1 --eval-every 2500 --save-every 2500 --resume
# results: weights on Hugging Face (needs the Git-ignored .env); failures here do not fail the job
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py suturebot_adapt_global || echo "### suturebot_adapt_global not published"
