#!/usr/bin/env bash
# Full container rehearsal with a PLACEHOLDER model, run while the GPU trains.
# Point is to prove build + GC-constrained run + output validity now, so that at
# packaging time the only thing that changes is model/det0.pt.
set -e
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
TAG=surgvu26-cat1-v5-dryrun
cp submission4/model/det0.pt submission5/model/det0.pt
echo "=== BUILD $(date) ==="
./submission5/do_build.sh $TAG
echo "=== RUN under GC constraints (CPU; no nvidia toolkit here) $(date) ==="
rm -f submission5/test/output/interf0/*.json || true
chmod 777 submission5/test/output/interf0
docker run --rm --network none --memory=32g --memory-swap=32g \
  --cap-drop=ALL --security-opt=no-new-privileges --pids-limit=256 \
  -v "$PWD/submission5/test/input/interf0":/input:ro \
  -v "$PWD/submission5/test/output/interf0":/output \
  $TAG
echo "=== OUTPUT $(date) ==="
ls -la submission5/test/output/interf0/
head -c 400 submission5/test/output/interf0/surgical-tools.json
echo
echo "=== DRYRUN DONE $(date) ==="
