# Locality with intervals over test cases: the locality test again for the runs whose control evaluation (before
# 4 Nov) kept only the median, now recording every window and instrument with its case (same 300 windows and noise)
set -e
for run in S_arms S_arms_scene S_arms_nocam S_arms_poolglobal S_global S_global_nocam S_global_vq S_slots; do
  .venv/bin/python scripts/eval/eval_control.py --run $run --limit 300 --tests locality --tag locality
done
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh locality_rows || echo "### locality_rows results not pushed"
