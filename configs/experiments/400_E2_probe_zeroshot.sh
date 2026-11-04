# E2: zero-shot kinematic grounding of SurgVU-trained latent actions (+ model-free baselines)
set -e
for d in suturebot crcd jigsaws; do
  .venv/bin/python scripts/eval/probe_kinematics.py --data $d --runs S_arms S_arms_scene S_global S_global_vq S_slots S_arms_poolglobal S_arms_shuffled S_arms_nofwd S_arms_nocam --baselines
  mv outputs/eval/probes/$d.json outputs/eval/probes/${d}_zeroshot.json
done
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh E2_probe_zeroshot || echo "### E2_probe_zeroshot results not pushed"
