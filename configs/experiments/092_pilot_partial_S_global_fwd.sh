# Monolithic latent action under the same objective, for an early arms-vs-global read before the S comparisons.
# Not a reported result: the training set is partial.
set -e
.venv/bin/python scripts/train/train_wm.py --name pilot_partial_S_global_fwd --action global --size S --steps 12000 --bs 32 --eval-every 3000 --save-every 3000
.venv/bin/python scripts/eval/diagnose_actions.py --run pilot_partial_S_global_fwd --windows 256
