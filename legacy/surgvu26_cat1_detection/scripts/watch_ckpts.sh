#!/usr/bin/env bash
# watch_ckpts.sh <weights_dir> <out.json> <train_pattern>
# Score each checkpoint on the uncontaminated presence metric as it appears, so a
# repeat of the epoch-4 collapse is visible within one epoch instead of nine.
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
export YOLO_CONFIG_DIR=/tmp/ultra_cfg
W=$1; OUT=$2; PAT=$3
seen=""
score() {
  ./env/bin/python scripts/presence_eval.py --ckpts "$1" --n 4000 --batch 16 \
      --workers 8 --out "$OUT" 2>&1 | grep -E "f1_topk=|Traceback|Error"
}
while true; do
  for c in $(ls $W/epoch*.pt 2>/dev/null | sort -V); do
    case " $seen " in *" $c "*) continue;; esac
    sleep 20
    score "$c"; seen="$seen $c"
  done
  if ! pgrep -f "$PAT" > /dev/null; then
    for c in $(ls $W/epoch*.pt 2>/dev/null | sort -V); do
      case " $seen " in *" $c "*) continue;; esac
      score "$c"; seen="$seen $c"
    done
    echo "=== WATCH DONE $(date) ==="
    break
  fi
  sleep 60
done
