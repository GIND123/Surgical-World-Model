# Ablation (size S, 60k steps, batch 32): arms trained on the diffusion loss alone (no forward decoder)
set -e
.venv/bin/python scripts/train/train_wm.py --name S_arms_nofwd --action arms --size S --fwd 0 --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/diagnose_actions.py --run S_arms_nofwd --windows 512
.venv/bin/python scripts/eval/eval_wm.py --run S_arms_nofwd --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_arms_nofwd --limit 300
# publish weights, config and evaluation (needs the Git-ignored .env; a failed upload does not fail the job)
{ .venv/bin/python scripts/eval/aggregate.py > /dev/null && CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py S_arms_nofwd; } || echo "### S_arms_nofwd not published"
