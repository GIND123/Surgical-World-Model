# Sensitivity: 1 or 3 context latents (0.1 s or 0.9 s of video; the protocol's 2 latents ran in job 330), same
# 200 test windows
set -e
for run in S_arms S_global S_slots S_global_vq S_arms_scene; do
  for n in 1 3; do
    .venv/bin/python scripts/eval/eval_wm.py --run $run --limit 200 --n-ctx $n --tag ctx$n
  done
done
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh sensitivity_context || echo "### sensitivity_context results not pushed"
