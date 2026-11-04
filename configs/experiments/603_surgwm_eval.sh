# Baseline: SurgWM re-implementation on the same 1,000 test windows as every other model (E1)
set -e
.venv/bin/python scripts/eval/eval_surgwm.py --limit 1000 --fvd
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh surgwm || echo "### surgwm results not pushed"
