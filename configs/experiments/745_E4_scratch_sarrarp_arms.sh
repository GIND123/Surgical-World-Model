# E4: EG-SWM trained on the SAR-RARP50 training videos alone with the adaptation budget (5k steps)
set -e
.venv/bin/python scripts/train/train_wm.py --name SAR_scratch_arms --data sar_rarp50 --action arms --size S --steps 5000 --bs 32 --warmup 500 --kl-warmup 2000 --eval-every 2500 --save-every 2500 --resume
.venv/bin/python scripts/eval/eval_wm.py --run SAR_scratch_arms --data sar_rarp50 --fvd
# results: tables, weights on Hugging Face (needs the Git-ignored .env), git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py SAR_scratch_arms || echo "### SAR_scratch_arms not published"
scripts/queue/commit_results.sh SAR_scratch_arms || echo "### SAR_scratch_arms results not pushed"
