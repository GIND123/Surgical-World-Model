# E3: lower and upper bounds on the SRT-H test episodes, copying the last context frame and the VAE reconstruction
set -e
.venv/bin/python scripts/eval/eval_wm.py --reference --data srth --fvd
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh E3_reference_srth || echo "### E3_reference_srth results not pushed"
