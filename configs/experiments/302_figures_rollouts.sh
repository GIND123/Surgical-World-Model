# Figures from model rollouts: the architecture figure's sample output (case_139), interfaces side by side,
# action swaps and endoscope commands, failure cases; time and memory of one prediction (outputs/eval/timing.json)
set -e
.venv/bin/python scripts/docs/make_rollouts.py
