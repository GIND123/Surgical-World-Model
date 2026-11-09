# Baseline: SurgWM re-implementation, stage 3 of 3: MaskGIT dynamics (62.5 M parameters) on 33-frame windows,
# 60k steps
set -e
.venv/bin/python scripts/train/train_surgwm.py --stage dynamics --steps 60000 --grad-ckpt --resume
