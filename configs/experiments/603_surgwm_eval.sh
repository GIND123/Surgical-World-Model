# Baseline: SurgWM re-implementation on the same 1,000 test windows as every other model (E1)
set -e
.venv/bin/python scripts/eval/eval_surgwm.py --limit 1000 --fvd --bs 16                 # 9 GB at batch 8 (job 304)
# results: tables, weights on Hugging Face, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py surgwm || echo "### surgwm not published"
scripts/queue/commit_results.sh surgwm || echo "### surgwm results not pushed"
