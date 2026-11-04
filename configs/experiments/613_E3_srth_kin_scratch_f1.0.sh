# E3: kinematics-conditioned prediction on SRT-H, init=scratch, train fraction 1.0
set -e
.venv/bin/python scripts/train/train_wm.py --name SRTH_kin_scratch_f1.0 --data srth --action kin --size S --train-fraction 1.0 --steps 8000 --bs 32 --warmup 500 --eval-every 4000 --save-every 4000 --resume
.venv/bin/python scripts/eval/eval_wm.py --run SRTH_kin_scratch_f1.0 --data srth --fvd
# results: tables, weights on Hugging Face (needs the Git-ignored .env), git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py SRTH_kin_scratch_f1.0 || echo "### SRTH_kin_scratch_f1.0 not published"
scripts/queue/commit_results.sh SRTH_kin_scratch_f1.0 || echo "### SRTH_kin_scratch_f1.0 results not pushed"
