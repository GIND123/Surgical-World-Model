# The size-B flagship through the tests of jobs 330-333 (Euler steps and guidance, 1 or 3 context latents, endoscope
# commands at 0.5-2x, 14 s rollouts), and the time and memory of one prediction for it and again for the S models,
# each now with only the timed model and the VAE on the GPU (job 302 had every S model loaded)
set -e
for cs in "1.0 20" "1.0 50" "1.0 100" "2.0 50"; do
  set -- $cs
  .venv/bin/python scripts/eval/eval_wm.py --run B_arms --limit 200 --cfg $1 --steps $2 --tag sampler_cfg$1_s$2
done
for n in 1 3; do
  .venv/bin/python scripts/eval/eval_wm.py --run B_arms --limit 200 --n-ctx $n --tag ctx$n
done
for sc in 0.5 1.0 2.0; do
  .venv/bin/python scripts/eval/eval_control.py --run B_arms --limit 100 --tests camera --scale $sc --tag control_scale$sc
done
.venv/bin/python scripts/eval/eval_long.py --run B_arms
.venv/bin/python scripts/docs/make_rollouts.py --only timing --timing-runs S_arms,S_global,B_arms
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh B_arms_sensitivity || echo "### B_arms_sensitivity results not pushed"
