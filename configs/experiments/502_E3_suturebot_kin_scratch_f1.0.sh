# E3: kinematics-conditioned prediction on SutureBot, init=scratch, train fraction 1.0
set -e
.venv/bin/python scripts/train/train_wm.py --name SB_kin_scratch_f1.0 --data suturebot --action kin --size S  --train-fraction 1.0 --steps 8000 --bs 32 --warmup 500 --eval-every 4000 --save-every 4000 --resume
.venv/bin/python scripts/eval/eval_wm.py --run SB_kin_scratch_f1.0 --data suturebot --fvd
