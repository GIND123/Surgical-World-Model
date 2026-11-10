# Revision: the monolithic latent at size B, the recipe of B_arms (340) otherwise, so that the size-B comparison is at
# equal size (the paper compares EG-SWM at size B with the monolithic latent at size S). About 20 GPU-hours with the
# evaluation. Before restoring: add ("B_global", "Monolithic latent + camera, size B", "global", True) to SCALE in
# scripts/eval/aggregate.py and the contrast (B_arms, B_global)
set -e
.venv/bin/python scripts/train/train_wm.py --name B_global --action global --size B --drop-no-signal --grad-ckpt --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/diagnose_actions.py --run B_global --windows 512
.venv/bin/python scripts/eval/eval_wm.py --run B_global --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run B_global --limit 300
# results: tables, weights on Hugging Face (needs the Git-ignored .env), git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py B_global || echo "### B_global not published"
scripts/queue/commit_results.sh B_global || echo "### B_global results not pushed"
