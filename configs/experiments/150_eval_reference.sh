# Reference rows (copy last context frame, VAE reconstruction) on the fixed test windows
set -e
.venv/bin/python scripts/eval/eval_wm.py --reference --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --reference --limit 300
