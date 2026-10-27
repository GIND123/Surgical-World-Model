# E3: kinematics-conditioned prediction on SutureBot, init=S_global, train fraction 1.0
set -e
.venv/bin/python scripts/train/train_wm.py --name SB_kin_global_f1.0 --data suturebot --action kin --size S --init S_global --train-fraction 1.0 --steps 8000 --bs 32 --warmup 500 --eval-every 4000 --save-every 4000 --resume
.venv/bin/python scripts/eval/eval_wm.py --run SB_kin_global_f1.0 --data suturebot --fvd
