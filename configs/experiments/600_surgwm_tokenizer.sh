# Baseline: SurgWM re-implementation, stage 1 of 3: ST-Transformer VQ-VAE tokenizer (120x180, 1024 codes), 60k steps
set -e
.venv/bin/python scripts/train/train_surgwm.py --stage tokenizer --steps 60000 --resume
