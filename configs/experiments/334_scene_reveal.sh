# Test of the explanation for the scene action's weak endoscope control: does a frame-wide learned action change
# mainly the content that the moving endoscope reveals? (300 test windows; scripts/eval/test_scene_reveal.py)
set -e
.venv/bin/python scripts/eval/test_scene_reveal.py --limit 300
# results: tables, git push; failures here do not fail the job
.venv/bin/python scripts/eval/aggregate.py > /dev/null || echo "### tables not updated"
scripts/queue/commit_results.sh scene_reveal || echo "### scene_reveal results not pushed"
