# The size-B flagship's 14 s rollouts (as job 333), and the time and memory of one prediction for it and again for
# the S models, each now with only the timed model and the VAE on the GPU (job 302 had every S model loaded). Its
# sampler, context and command-size tests are deferred for the 9 Nov deadline
set -e
.venv/bin/python scripts/eval/eval_long.py --run B_arms
.venv/bin/python scripts/docs/make_rollouts.py --only timing --timing-runs S_arms,S_global,B_arms
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh B_arms_sensitivity || echo "### B_arms_sensitivity results not pushed"
