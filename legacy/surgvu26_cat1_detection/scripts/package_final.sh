#!/usr/bin/env bash
# One-shot packaging for the final submission.
#   package_final.sh <weights.pt> <docker-tag> <output.tar.gz>
# Builds the container around <weights.pt>, runs it under Grand Challenge's
# constraints, scores the result with the ORGANISERS' own eval code, and only
# then writes the tarball. Existing tarballs are never touched.
set -e
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
W=$1; TAG=$2; OUT=$3
[ -f "$W" ] || { echo "no such weights: $W"; exit 1; }
[ -e "$OUT" ] && { echo "REFUSING: $OUT already exists"; exit 1; }

echo "=== stage model (strip optimizer state: 152MB -> ~51MB, EMA promoted, fp16) $(date) ==="
./env/bin/python - "$W" <<'PY'
import sys, shutil
from ultralytics.utils.torch_utils import strip_optimizer
src, dst = sys.argv[1], 'submission5/model/det0.pt'
shutil.copyfile(src, dst)
strip_optimizer(dst)          # in-place: model <- ema, optimizer None, epoch -1, half
PY
ls -l submission5/model/
./env/bin/python - <<'PY'
import torch
ck = torch.load('submission5/model/det0.pt', map_location='cpu', weights_only=False)
assert ck.get('epoch') == -1 and ck.get('optimizer') is None, 'strip_optimizer did not run'
print('  stripped OK; nc =', ck['model'].yaml.get('nc'))
PY

echo "=== build $TAG $(date) ==="
./submission5/do_build.sh "$TAG"

echo "=== GC-constrained run $(date) ==="
rm -f submission5/test/output/interf0/*.json || true
chmod 777 submission5/test/output/interf0
docker run --rm --network none --memory=32g --memory-swap=32g \
  --cap-drop=ALL --security-opt=no-new-privileges --pids-limit=256 \
  -v "$PWD/submission5/test/input/interf0":/input:ro \
  -v "$PWD/submission5/test/output/interf0":/output \
  "$TAG"

echo "=== organiser-code score (fixture is video 3; it is IN train, so this is a"
echo "    pipeline/convention check, not a quality estimate) $(date) ==="
./env/bin/python scripts/official_check.py \
    submission5/test/output/interf0/surgical-tools.json 3

echo "=== save tarball $(date) ==="
./submission5/do_save.sh "$TAG" "$OUT"
echo "=== PACKAGE DONE $(date) ==="
