# Sensitivity: is the realised endoscope motion proportional to the command? Commands at 0.5, 1 and 2 times the
# 99th percentile of real motion on 100 test windows, for the interfaces whose endoscope action is controllable
set -e
for run in S_arms S_global_vq S_state; do
  for sc in 0.5 1.0 2.0; do
    .venv/bin/python scripts/eval/eval_control.py --run $run --limit 100 --tests camera --scale $sc --tag control_scale$sc
  done
done
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh sensitivity_camera_scale || echo "### sensitivity_camera_scale results not pushed"
