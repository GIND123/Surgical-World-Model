# SurgWM re-implementation: smoke test of its three training stages (300 steps each; speed and peak memory in
# outputs/runs/surgwm_smoke_<stage>/log.jsonl) and of its evaluation on 32 test windows, before the full runs (600-603)
set -e
.venv/bin/python scripts/train/train_surgwm.py --name surgwm_smoke --stage tokenizer --steps 300 \
  || .venv/bin/python scripts/train/train_surgwm.py --name surgwm_smoke --stage tokenizer --steps 300 --grad-ckpt
.venv/bin/python scripts/train/train_surgwm.py --name surgwm_smoke --stage lam --steps 300
.venv/bin/python scripts/train/train_surgwm.py --name surgwm_smoke --stage dynamics --steps 300 --grad-ckpt
.venv/bin/python scripts/eval/eval_surgwm.py --name surgwm_smoke --limit 32 --fvd
