# E2: kinematic grounding after action-free adaptation
set -e
for d in suturebot crcd jigsaws; do
  .venv/bin/python scripts/eval/probe_kinematics.py --data $d --runs ${d}_adapt_arms ${d}_adapt_global ${d}_adapt_global_vq ${d}_adapt_slots
  mv outputs/eval/probes/$d.json outputs/eval/probes/${d}_adapted.json
done
.venv/bin/python scripts/eval/validate_camera_action.py --data crcd
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh E2_probe_adapted || echo "### E2_probe_adapted results not pushed"
