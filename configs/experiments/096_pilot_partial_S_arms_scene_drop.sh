# With a scene or monolithic action the explicit endoscope action matters less, even with the learned actions
# zeroed (as when the endoscope motion is commanded): actions were only ever dropped together. This pilot also
# drops the learned actions alone (p = 0.2, camera kept), on the parts of pilot_partial_S_arms_fwd.
set -e
.venv/bin/python scripts/train/train_wm.py --name pilot_partial_S_arms_scene_drop --scene --p-drop-learned 0.2 --size S --steps 12000 --bs 32 --eval-every 3000 --save-every 3000 --stems configs/splits/surgvu_pilot_parts.json
.venv/bin/python scripts/eval/diagnose_actions.py --run pilot_partial_S_arms_scene_drop --windows 256 --stems configs/splits/surgvu_pilot_parts.json
