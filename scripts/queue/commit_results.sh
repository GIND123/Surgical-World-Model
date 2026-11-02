#!/usr/bin/env bash
# Commit and push the generated result tables after a queue job (no-op when they are unchanged). Only
# docs/results/ and paper/tmrb/tables/ are committed, so work in progress elsewhere in the tree is left alone.
#
#     scripts/queue/commit_results.sh <run>
set -euo pipefail
cd "$(dirname "$0")/../.."
export GIT_TERMINAL_PROMPT=0
paths=(docs/results paper/tmrb/tables)
git add -- "${paths[@]}"
git diff --cached --quiet -- "${paths[@]}" && { echo "results unchanged"; exit 0; }
git commit -q -m "Results: $1" -- "${paths[@]}"
if ! git push -q origin main 2>/dev/null; then                      # the remote moved on: rebase, then push
  git pull -q --rebase --autostash origin main || { git rebase --abort 2>/dev/null || true; exit 1; }
  git push -q origin main
fi
echo "pushed $(git log --oneline -1)"
