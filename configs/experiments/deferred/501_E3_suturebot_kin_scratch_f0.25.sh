# E3: kinematics-conditioned prediction on SutureBot, init=scratch, train fraction 0.25
set -e
.venv/bin/python scripts/train/train_wm.py --name SB_kin_scratch_f0.25 --data suturebot --action kin --size S  --train-fraction 0.25 --steps 8000 --bs 32 --warmup 500 --eval-every 4000 --save-every 4000 --resume
.venv/bin/python scripts/eval/eval_wm.py --run SB_kin_scratch_f0.25 --data suturebot --fvd --weights raw
# results: tables, weights on Hugging Face (needs the Git-ignored .env), git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py SB_kin_scratch_f0.25 || echo "### SB_kin_scratch_f0.25 not published"
scripts/queue/commit_results.sh SB_kin_scratch_f0.25 || echo "### SB_kin_scratch_f0.25 results not pushed"
