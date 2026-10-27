.venv/bin/python scripts/data/prepare_jigsaws.py --workers 6 --stride 16
# video-kinematics lag per episode; kinematics probes and kin-conditioned runs use synced windows only
.venv/bin/python scripts/data/check_sync.py jigsaws
