# Open-H SRT-H (dVRK ex-vivo porcine cholecystectomy): 14,836 short episodes at 30 fps with official
# train/val/test splits; then the video-kinematics lag check (pooled rule for short episodes)
set -e
.venv/bin/python scripts/data/prepare_lerobot.py --name srth --root data/external/open_h_srth_chole/Surgical/jhu/imerse/srth_porcine_chole --workers 6 --stride 16
.venv/bin/python scripts/data/check_sync.py srth
