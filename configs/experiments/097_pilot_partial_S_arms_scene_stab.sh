# Frame-wide learned actions (monolithic or scene) duplicate the endoscope motion and weaken the explicit
# endoscope action; dropping them alone (096) restored it only partly and cost replay accuracy. This pilot infers
# the learned actions from camera-stabilised latents (z_{t-1} warped by the measured endoscope motion), without
# learned-only dropout, on the parts of pilot_partial_S_arms_fwd.
set -e
.venv/bin/python scripts/train/train_wm.py --name pilot_partial_S_arms_scene_stab --scene --stabilise --p-drop-learned 0 --size S --steps 12000 --bs 32 --eval-every 3000 --save-every 3000 --stems configs/splits/surgvu_pilot_parts.json
.venv/bin/python scripts/eval/diagnose_actions.py --run pilot_partial_S_arms_scene_stab --windows 256 --stems configs/splits/surgvu_pilot_parts.json
