# Baseline: SurgWM re-implementation, stage 2 of 3: latent action model (12 discrete actions), 60k steps
set -e
.venv/bin/python scripts/train/train_surgwm.py --stage lam --steps 60000 --resume
