# Controlled comparison of action interfaces (size S, 60k steps, batch 32): arms_scene -- EG-SWM plus a scene action
# pooled and written outside every arm's region (pilots 095-097: replay as accurate as the monolithic action,
# weaker use of the endoscope action).
set -e
.venv/bin/python scripts/train/train_wm.py --name S_arms_scene --action arms --scene --size S --steps 60000 --bs 32 --eval-every 10000 --save-every 5000 --resume
.venv/bin/python scripts/eval/diagnose_actions.py --run S_arms_scene --windows 512
.venv/bin/python scripts/eval/eval_wm.py --run S_arms_scene --limit 1000 --fvd
.venv/bin/python scripts/eval/eval_control.py --run S_arms_scene --limit 300
