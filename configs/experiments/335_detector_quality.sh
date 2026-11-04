# Detector quality (E6): EG-SWM on the 1,000 E1 windows with a weaker instrument detector at test time, 25 / 50 /
# 100 % of the detections removed at random (nested); an arm loses its action in every step without a region
set -e
for p in 0.25 0.5 1.0; do
  .venv/bin/python scripts/eval/eval_wm.py --run S_arms --limit 1000 --drop-detections $p --tag replay_drop$p
done
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh detector_quality || echo "### detector_quality results not pushed"
