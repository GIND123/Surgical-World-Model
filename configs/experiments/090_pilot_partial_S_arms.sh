# Early check of the learning dynamics (KL activity, shuffled-action gap) on the parts encoded so far.
# Not a reported result: the training set is partial.
until [ "$(ls data/processed/surgvu/conditioning/*.npz 2>/dev/null | wc -l)" -ge 100 ]; do sleep 120; done
.venv/bin/python scripts/train/train_wm.py --name pilot_partial_S_arms --action arms --kl 1e-3 --fwd 0 --p-drop-learned 0 --size S --steps 12000 --bs 32 --eval-every 3000 --save-every 3000
