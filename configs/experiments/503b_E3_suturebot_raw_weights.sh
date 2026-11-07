# E3 is scored on the final weights: after 8k steps the weight average (decay 0.9995, time constant 2k steps)
# still lags training and starts from different weights for each initialisation (random or pretrained); on 64
# validation windows it raised the generation error of the from-scratch runs by 19 and 27 % (10 and 100 % of the
# episodes). Re-scores the runs evaluated before the change; their weight-average results stay as replay_ema.*
set -e
for r in SB_kin_scratch_f0.1 SB_kin_scratch_f1.0 SB_kin_arms_f0.1; do
  d=outputs/eval/$r/suturebot
  if [ ! -f $d/replay_ema.json ]; then mv $d/replay.json $d/replay_ema.json; mv $d/replay.npz $d/replay_ema.npz; fi
  .venv/bin/python scripts/eval/eval_wm.py --run $r --data suturebot --fvd --weights raw
done
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh E3_suturebot_raw_weights || echo "### E3_suturebot_raw_weights results not pushed"
