# Separates the forward decoder from the lower KL weight: the first arms pilot used beta = 1e-3 (and fewer parts),
# this one beta = 1e-4 without the decoder on the parts of pilot_partial_S_arms_fwd. Then every pilot is diagnosed
# on the same validation windows.
set -e
.venv/bin/python scripts/train/train_wm.py --name pilot_partial_S_arms_nofwd --fwd 0 --p-drop-learned 0 --size S --steps 12000 --bs 32 --eval-every 3000 --save-every 3000 --stems configs/splits/surgvu_pilot_parts.json
for r in pilot_partial_S_arms pilot_partial_S_arms_fwd pilot_partial_S_arms_nofwd pilot_partial_S_global_fwd; do
  .venv/bin/python scripts/eval/diagnose_actions.py --run $r --windows 256 --stems configs/splits/surgvu_pilot_parts.json
done
