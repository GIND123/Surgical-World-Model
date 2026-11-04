# Seed 1 of S_arms (size S, 60k steps, batch 32): run-to-run variation of the main comparison
set -e
.venv/bin/python scripts/train/train_wm.py --name S_arms_seed1 --action arms --size S --seed 1 --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/diagnose_actions.py --run S_arms_seed1 --windows 512
.venv/bin/python scripts/eval/eval_wm.py --run S_arms_seed1 --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_arms_seed1 --limit 300
# results: tables, weights on Hugging Face (needs the Git-ignored .env), git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py S_arms_seed1 || echo "### S_arms_seed1 not published"
scripts/queue/commit_results.sh S_arms_seed1 || echo "### S_arms_seed1 results not pushed"
