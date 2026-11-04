# Smoke test of the SurgWM evaluation after its fidelity and control code was shared with eval_control.py: both
# modes on the smoke checkpoints of job 304 (16 windows each, a few minutes); nothing is published
set -e
.venv/bin/python scripts/eval/eval_surgwm.py --name surgwm_smoke --limit 16 --bs 8 --tag replay_check
.venv/bin/python scripts/eval/eval_surgwm.py --name surgwm_smoke --control --limit 16 --bs 8 --tag control_check
echo "### smoke passed"
