# Supplementary Movie S1 for the T-MRB submission: the windows of the rollout figures (302, 307) as clips at half
# speed, and all six endoscope commands for EG-SWM and the monolithic latent (same models and noise; a few minutes).
# Written to outputs/videos/movie_s1.mp4, outside Git (SurgVU frames).
set -e
.venv/bin/python scripts/docs/make_rollouts.py --only video
