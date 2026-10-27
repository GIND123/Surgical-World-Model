# Pilot 090 trained on the diffusion loss alone and its arm actions collapsed to the prior (1/8 active units,
# shuffling them changed the loss by <0.1%). Same pilot with the forward-decoder objective and KL weight 1e-4.
# Not a reported result: the training set is partial.
set -e
.venv/bin/python scripts/eval/diagnose_actions.py --run pilot_partial_S_arms --windows 256
.venv/bin/python scripts/train/train_wm.py --name pilot_partial_S_arms_fwd --action arms --size S --steps 12000 --bs 32 --eval-every 3000 --save-every 3000
.venv/bin/python scripts/eval/diagnose_actions.py --run pilot_partial_S_arms_fwd --windows 256
