# Monolithic latent action under the same objective, for an early arms-vs-global read before the S comparisons.
# Not a reported result: the training set is partial. Pilots from here on use the parts of pilot_partial_S_arms_fwd
# (configs/splits/surgvu_pilot_parts.json, 97 training parts, 213,985 windows) so that they see the same data.
set -e
.venv/bin/python scripts/train/train_wm.py --name pilot_partial_S_global_fwd --action global --size S --steps 12000 --bs 32 --eval-every 3000 --save-every 3000 --stems configs/splits/surgvu_pilot_parts.json
