# H4: 14 s autoregressive rollouts (five 2.8 s chunks, each re-encoding the last 5 generated frames as context) on 95
# clips from the 19 test cases (configs/eval/surgvu_test_long.json): fidelity and instrument hallucination over time
set -e
.venv/bin/python scripts/eval/eval_long.py --reference
for run in S_arms S_global S_slots S_global_vq S_state S_none S_arms_scene; do
  .venv/bin/python scripts/eval/eval_long.py --run $run
done
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh long_rollouts || echo "### long_rollouts results not pushed"
