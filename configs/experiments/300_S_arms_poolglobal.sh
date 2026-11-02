# Ablation (size S, 60k steps, batch 32): arms_poolglobal
set -e
.venv/bin/python scripts/train/train_wm.py --name S_arms_poolglobal --action arms --size S --arm-pool global --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/diagnose_actions.py --run S_arms_poolglobal --windows 512
.venv/bin/python scripts/eval/eval_wm.py --run S_arms_poolglobal --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_arms_poolglobal --limit 300
# results: tables, weights on Hugging Face (needs the Git-ignored .env), git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py S_arms_poolglobal || echo "### S_arms_poolglobal not published"
scripts/queue/commit_results.sh S_arms_poolglobal || echo "### S_arms_poolglobal results not pushed"
