# Intervals over test cases for endoscope control and embodiment consistency: the endoscope and embodiment tests again
# for the runs whose control evaluation (29 Oct-4 Nov) kept only summaries, now recording every window with its case.
# Same 300 windows and the same noise draws as the full evaluation (the locality test, left out, draws its own), so
# the summaries must reproduce (aggregate.py reports any that do not)
set -e
for run in S_arms S_arms_scene S_arms_poolglobal S_global S_global_vq S_slots S_state \
           S_none S_state_nocam S_global_nocam S_arms_nocam; do
  .venv/bin/python scripts/eval/eval_control.py --run $run --limit 300 --tests camera,ecs --tag camera_ecs
done
# results: tables, git push; failures here do not fail the job
{ .venv/bin/python scripts/eval/aggregate.py || echo "### tables not updated"; } | grep "^###" || true
scripts/queue/commit_results.sh camera_ecs_rows || echo "### camera_ecs_rows results not pushed"
