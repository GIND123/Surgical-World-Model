#!/usr/bin/env bash
set -e
SD=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TAG=${1:-surgvu26-cat1}
OUT=${2:-"$SD/../${TAG}.tar.gz"}
docker save "$TAG" | gzip -c > "$OUT"
ls -lh "$OUT"
