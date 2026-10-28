# The monolithic-action pilot replays futures 17 % more accurately than the arm actions, mostly outside the
# instrument regions. This pilot adds a scene action pooled and written only outside every arm's region, on the
# parts of pilot_partial_S_arms_fwd, and diagnoses it on the same validation windows.
set -e
.venv/bin/python scripts/train/train_wm.py --name pilot_partial_S_arms_scene --scene --p-drop-learned 0 --size S --steps 12000 --bs 32 --eval-every 3000 --save-every 3000 --stems configs/splits/surgvu_pilot_parts.json
.venv/bin/python scripts/eval/diagnose_actions.py --run pilot_partial_S_arms_scene --windows 256 --stems configs/splits/surgvu_pilot_parts.json
