# Sensitivity: Euler steps and guidance on the actions for EG-SWM and the interfaces it is compared with, on 200
# test windows spread over the 19 cases (job 204 repeated without case_126)
set -e
for run in S_arms S_global S_slots S_global_vq S_arms_scene; do
  for cs in "1.0 20" "1.0 50" "1.0 100" "2.0 50"; do
    set -- $cs
    .venv/bin/python scripts/eval/eval_wm.py --run $run --limit 200 --cfg $1 --steps $2 --tag sampler_cfg$1_s$2
  done
done
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh sensitivity_sampler || echo "### sensitivity_sampler results not pushed"
