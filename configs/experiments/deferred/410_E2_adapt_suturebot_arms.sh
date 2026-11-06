# E2: action-free adaptation of S_arms to suturebot videos (no kinematics used), for probing
set -e
.venv/bin/python scripts/train/train_wm.py --name suturebot_adapt_arms --data suturebot --action arms --size S --init S_arms --steps 5000 --bs 32 --warmup 500 --kl-warmup 1 --eval-every 2500 --save-every 2500 --resume
# results: weights on Hugging Face (needs the Git-ignored .env); failures here do not fail the job
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py suturebot_adapt_arms || echo "### suturebot_adapt_arms not published"
