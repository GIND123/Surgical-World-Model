# E2: zero-shot kinematic grounding of SurgVU-trained latent actions (+ model-free baselines)
set -e
for d in suturebot crcd jigsaws; do
  .venv/bin/python scripts/eval/probe_kinematics.py --data $d --runs S_arms S_arms_scene S_global S_global_vq S_slots S_arms_poolglobal S_arms_shuffled --baselines
  mv outputs/eval/probes/$d.json outputs/eval/probes/${d}_zeroshot.json
done
