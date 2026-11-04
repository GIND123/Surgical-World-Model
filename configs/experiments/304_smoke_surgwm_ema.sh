# SurgWM re-implementation as corrected on 4 Nov (the paper's moving-average quantisers, the authors' released
# action model, 16-frame clips, 25 MaskGIT steps): smoke test before the full runs (600-603). Tokenizer and action
# model 2,000 steps each, past the 1,000-step warm-up, so that code use and restarts show (codes_used, perplexity,
# restarts in outputs/runs/surgwm_smoke_<stage>/log.jsonl); dynamics 300 steps without gradient checkpointing and,
# if that runs out of memory, with it; evaluation on 32 test windows (time and peak memory in its summary)
set -e
rm -rf outputs/runs/surgwm_smoke_tokenizer outputs/runs/surgwm_smoke_lam outputs/runs/surgwm_smoke_dynamics \
  outputs/eval/surgwm_smoke
.venv/bin/python scripts/train/train_surgwm.py --name surgwm_smoke --stage tokenizer --steps 2000
.venv/bin/python scripts/train/train_surgwm.py --name surgwm_smoke --stage lam --steps 2000
.venv/bin/python scripts/train/train_surgwm.py --name surgwm_smoke --stage dynamics --steps 300 \
  || .venv/bin/python scripts/train/train_surgwm.py --name surgwm_smoke --stage dynamics --steps 300 --grad-ckpt
.venv/bin/python scripts/eval/eval_surgwm.py --name surgwm_smoke --limit 32 --fvd
