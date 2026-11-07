# E2 control for the size of the probed features: each arm's probe gets every arm's action of EG-SWM (32 features),
# as it gets the whole action of a frame-wide interface (monolithic 24, slots 32), instead of its own arm's 8
set -e
for d in suturebot crcd jigsaws; do
  .venv/bin/python scripts/eval/probe_kinematics.py --data $d --runs S_arms:all B_arms:all
  mv outputs/eval/probes/$d.json outputs/eval/probes/${d}_allarms.json
done
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh E2_probe_allarms || echo "### E2_probe_allarms results not pushed"
