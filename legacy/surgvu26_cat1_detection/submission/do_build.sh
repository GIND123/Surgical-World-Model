#!/usr/bin/env bash
set -e
SD=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TAG=${1:-surgvu26-cat1}
cp "$SD/../scripts/infer_core.py" "$SD/../scripts/track_core.py" "$SD/"
docker build --platform=linux/amd64 -t "$TAG" "$SD"
echo "built $TAG"
