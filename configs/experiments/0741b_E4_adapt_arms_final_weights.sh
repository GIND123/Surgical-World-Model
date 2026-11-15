# E4: SAR_adapt_arms (job 0741) scored again with its final weights, like the other 5k-step SAR-RARP50 runs
# (0742-0745) and the 8k-step E3 runs: after 5k steps the weight average still holds 8 % of the starting weights.
set -e
.venv/bin/python scripts/eval/eval_wm.py --run SAR_adapt_arms --data sar_rarp50 --fvd --weights raw
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh SAR_adapt_arms || echo "### SAR_adapt_arms results not pushed"
