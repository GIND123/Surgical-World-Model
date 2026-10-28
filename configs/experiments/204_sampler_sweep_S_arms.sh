# Inference settings on the EG-SWM S checkpoint: Euler steps and classifier-free guidance on actions,
# 200 test windows spread over all 20 test cases (10 per case)
set -e
for cs in "1.0 20" "1.0 50" "1.0 100" "2.0 50"; do
  set -- $cs
  .venv/bin/python scripts/eval/eval_wm.py --run S_arms --limit 200 --cfg $1 --steps $2 --tag sampler_cfg$1_s$2
done
