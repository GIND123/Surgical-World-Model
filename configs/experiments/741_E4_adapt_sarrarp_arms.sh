# E4: action-free adaptation of S_arms to the SAR-RARP50 training videos (5k steps), then the test videos
set -e
.venv/bin/python scripts/train/train_wm.py --name SAR_adapt_arms --data sar_rarp50 --action arms --size S --init S_arms --steps 5000 --bs 32 --warmup 500 --kl-warmup 1 --eval-every 2500 --save-every 2500 --resume
.venv/bin/python scripts/eval/eval_wm.py --run SAR_adapt_arms --data sar_rarp50 --fvd
# results: tables, weights on Hugging Face (needs the Git-ignored .env), git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py SAR_adapt_arms || echo "### SAR_adapt_arms not published"
scripts/queue/commit_results.sh SAR_adapt_arms || echo "### SAR_adapt_arms results not pushed"
