# E2: action-free adaptation of S_global_vq to suturebot videos (no kinematics used), for probing
set -e
.venv/bin/python scripts/train/train_wm.py --name suturebot_adapt_global_vq --data suturebot --action global_vq --size S --init S_global_vq --steps 5000 --bs 32 --warmup 500 --kl-warmup 1 --eval-every 2500 --save-every 2500 --resume
# results: weights on Hugging Face (needs the Git-ignored .env); failures here do not fail the job
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py suturebot_adapt_global_vq || echo "### suturebot_adapt_global_vq not published"
