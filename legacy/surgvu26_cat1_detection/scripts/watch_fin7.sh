#!/usr/bin/env bash
# Score each fin7 checkpoint on the uncontaminated presence metric as soon as it
# appears. The previous run collapsed at epoch 4 and it cost nine epochs to find
# out; this makes a repeat visible within one epoch.
cd /mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work
export YOLO_CONFIG_DIR=/tmp/ultra_cfg
W=runs/fin7/w/weights
seen=""
while true; do
  for c in $(ls $W/epoch*.pt 2>/dev/null | sort -V); do
    case " $seen " in *" $c "*) continue;; esac
    sleep 20   # let the file finish being written
    ./env/bin/python scripts/presence_eval.py --ckpts "$c" --n 4000 --batch 16 \
        --workers 8 --out runs/presence_fin7.json 2>&1 | grep -E "f1_topk=|Traceback|Error"
    seen="$seen $c"
  done
  if ! pgrep -f "[t]rain7.py" > /dev/null; then
    echo "=== fin7 training finished $(date) ==="
    for c in $(ls $W/epoch*.pt 2>/dev/null | sort -V); do
      case " $seen " in *" $c "*) continue;; esac
      ./env/bin/python scripts/presence_eval.py --ckpts "$c" --n 4000 --batch 16 \
          --workers 8 --out runs/presence_fin7.json 2>&1 | grep -E "f1_topk=|Traceback|Error"
      seen="$seen $c"
    done
    echo "=== WATCH DONE $(date) ==="
    break
  fi
  sleep 60
done
