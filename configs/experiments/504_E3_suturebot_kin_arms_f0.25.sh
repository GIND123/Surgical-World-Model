# E3: kinematics-conditioned prediction on SutureBot, init=S_arms, train fraction 0.25
set -e
.venv/bin/python scripts/train/train_wm.py --name SB_kin_arms_f0.25 --data suturebot --action kin --size S --init S_arms --train-fraction 0.25 --steps 8000 --bs 32 --warmup 500 --eval-every 4000 --save-every 4000 --resume
.venv/bin/python scripts/eval/eval_wm.py --run SB_kin_arms_f0.25 --data suturebot --fvd
