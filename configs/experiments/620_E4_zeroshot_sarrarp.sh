# E4: transfer to clinical surgery. Reference rows and the SurgVU models without adaptation on the SAR-RARP50 test videos
set -e
.venv/bin/python scripts/eval/eval_wm.py --reference --data sar_rarp50 --fvd
for run in S_none S_global S_global_vq S_slots S_arms S_arms_scene; do
  .venv/bin/python scripts/eval/eval_wm.py --run $run --data sar_rarp50 --fvd
done
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh E4_zeroshot_sarrarp || echo "### E4_zeroshot_sarrarp results not pushed"
