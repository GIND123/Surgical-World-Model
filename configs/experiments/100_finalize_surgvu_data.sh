# Block until every SurgVU part has cache, latents, detections and conditioning; then refresh the
# full-corpus window index and camera statistics, and fix the held-out evaluation sets.
P=data/processed/surgvu
until [ "$(ls $P/meta10/*.npz 2>/dev/null | wc -l)" -ge 280 ] && [ "$(ls $P/latents/*.starts.npy 2>/dev/null | wc -l)" -ge 280 ] \
   && [ "$(ls $P/detections/*.npz 2>/dev/null | wc -l)" -ge 280 ] && [ "$(ls $P/conditioning/*.npz 2>/dev/null | wc -l)" -ge 280 ]; do
  sleep 120
done
.venv/bin/python scripts/data/build_windows.py
.venv/bin/python scripts/eval/build_eval_set.py --split test --per-case 100
.venv/bin/python scripts/eval/build_eval_set.py --split val --per-case 25
