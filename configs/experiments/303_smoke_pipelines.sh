# Smoke test of the paths that no job has run yet, before jobs 305-731 depend on them: training and evaluation on
# external data (E2 adaptation, E3 kinematics on SutureBot and SRT-H, E4 SAR-RARP50), kinematic probes with model
# features, and the new evaluation options (context length, endoscope-command scale). 30 steps each, 8-16 windows;
# everything written here is removed at the end.
set -e
T="--size S --steps 30 --bs 8 --warmup 10 --eval-every 15 --save-every 30"
.venv/bin/python scripts/train/train_wm.py --name smoke_sb_kin --data suturebot --action kin --init S_arms --train-fraction 0.1 $T
.venv/bin/python scripts/eval/eval_wm.py --run smoke_sb_kin --data suturebot --limit 16 --fvd
.venv/bin/python scripts/train/train_wm.py --name smoke_srth_kin --data srth --action kin $T
.venv/bin/python scripts/eval/eval_wm.py --run smoke_srth_kin --data srth --limit 16
.venv/bin/python scripts/train/train_wm.py --name smoke_sar_arms --data sar_rarp50 --action arms --init S_arms --kl-warmup 1 $T
.venv/bin/python scripts/eval/eval_wm.py --run smoke_sar_arms --data sar_rarp50 --limit 16
.venv/bin/python scripts/train/train_wm.py --name smoke_crcd_vq --data crcd --action global_vq --init S_global_vq --kl-warmup 1 $T
.venv/bin/python scripts/eval/probe_kinematics.py --data suturebot --runs S_arms S_global_vq --baselines
.venv/bin/python scripts/eval/eval_wm.py --run S_arms --limit 8 --n-ctx 1 --tag smoke_ctx1
.venv/bin/python scripts/eval/eval_control.py --run S_arms --limit 8 --tests camera --scale 2.0 --tag smoke_control
.venv/bin/python scripts/eval/eval_control.py --run S_arms --limit 8 --tag smoke_control_all
echo "### smoke test passed"
rm -rf outputs/runs/smoke_sb_kin outputs/runs/smoke_srth_kin outputs/runs/smoke_sar_arms outputs/runs/smoke_crcd_vq \
  outputs/eval/smoke_sb_kin outputs/eval/smoke_srth_kin outputs/eval/smoke_sar_arms outputs/eval/probes/suturebot.json \
  outputs/eval/S_arms/smoke_ctx1.json outputs/eval/S_arms/smoke_ctx1.npz outputs/eval/S_arms/smoke_control.json \
  outputs/eval/S_arms/smoke_control_all.json
