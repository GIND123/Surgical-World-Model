# E2: action-free adaptation of S_slots to crcd videos (no kinematics used), for probing
set -e
.venv/bin/python scripts/train/train_wm.py --name crcd_adapt_slots --data crcd --action slots --size S --init S_slots --steps 5000 --bs 32 --warmup 500 --kl-warmup 1 --eval-every 2500 --save-every 2500 --resume
# results: weights on Hugging Face (needs the Git-ignored .env); failures here do not fail the job
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py crcd_adapt_slots || echo "### crcd_adapt_slots not published"
