# Flagship and model scale (E6): EG-SWM at size B (12 blocks of width 768, 153.7 M parameters) with the recipe of
# size S (60k steps, batch 32, learning rate 1e-4, seed 0), without the recorder's colour-bar windows (0.45 % of the
# training windows). Every block is recomputed in the backward pass, which batch 32 needs at this size on 48 GB
# (same gradients, checked on CPU). Boxes stay as detected: holding them across detector gaps (325) gained 0.05 dB
# and cost endoscope control
set -e
.venv/bin/python scripts/train/train_wm.py --name B_arms --action arms --size B --drop-no-signal --grad-ckpt --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/diagnose_actions.py --run B_arms --windows 512
.venv/bin/python scripts/eval/eval_wm.py --run B_arms --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run B_arms --limit 300
# results: tables, weights on Hugging Face (needs the Git-ignored .env), git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/release/push_weights.py B_arms || echo "### B_arms not published"
scripts/queue/commit_results.sh B_arms || echo "### B_arms results not pushed"
