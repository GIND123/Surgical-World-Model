# Surgical World Model

World models of robot-assisted surgery whose actions are structured like the robot: an explicit endoscope ego-motion action plus one latent action per instrument arm, learned from unlabeled in-vivo da Vinci video and grounded in the robot's own logs. Target venue: *IEEE Transactions on Medical Robotics and Bionics*.

**Status (2026-10-27):** data audit, external-data survey, literature review and research plan are complete; the data pipeline is next.

| Document | Content |
|---|---|
| [docs/01_data_audit.md](docs/01_data_audit.md) | the local SurgVU corpus: 155 cases, 837.8 h, robot-log arm occupancy incl. endoscope, tasks, quality, label noise |
| [docs/02_external_data.md](docs/02_external_data.md) | external datasets (Open-H-Embodiment, SAR-RARP50, gated ones), PhysioNet verdict |
| [docs/03_literature_review.md](docs/03_literature_review.md) | surgical world models 2024–2026, gap analysis |
| [docs/04_research_plan.md](docs/04_research_plan.md) | hypotheses, method, experiments, baselines, ablations, metrics, timeline |
| [docs/legacy.md](docs/legacy.md) | the previous SurgVU detection project and where its files moved |

## Layout

```text
src/swm/            Python package (data loading, models, evaluation)
scripts/data/       audits, downloads, preprocessing
configs/            experiment and split configs
docs/               research documents
legacy/             previous SurgVU 2026 Cat-1 detection project (reused detector)
data/               raw and external datasets            (Git-ignored)
checkpoints/        pretrained and trained weights        (Git-ignored)
outputs/            runs, logs, audit tables              (Git-ignored)
archive/            old challenge artefacts               (Git-ignored)
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -e ".[dev]"
```

Raw SurgVU data is obtained from the challenge organisers and placed under `data/raw/` (see `docs/legacy.md` for the expected layout). Open external data is fetched with `python scripts/data/download_external.py --list`.

## Reproducing the audit

```bash
.venv/bin/python scripts/data/audit_surgvu.py --workers 20   # video integrity, label tables
.venv/bin/python scripts/data/audit_timeline.py               # robot configuration on the global timeline
.venv/bin/python scripts/data/audit_frames.py --every 60      # frame quality, console-UI presence
```
