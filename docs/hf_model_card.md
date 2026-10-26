---
license: cc-by-nc-sa-4.0
library_name: pytorch
pipeline_tag: image-to-video
tags:
  - surgical-robotics
  - world-model
  - video-prediction
  - medical
  - da-vinci
datasets:
  - nvidia/PhysicalAI-Robotics-Open-H-Embodiment
---

# Surgical World Model (EG-SWM)

Embodiment-grounded world models of robot-assisted surgery: an action-conditioned video world model whose action space mirrors the robot — an explicit **endoscope ego-motion** action plus one latent action per **instrument arm**, learned from unlabeled in-vivo da Vinci video and bound to the arms through the robot's own system logs.

Code, data pipeline, documentation and diagrams: <https://github.com/GIND123/Surgical-World-Model>

![architecture](architecture.png)

## Checkpoints

Checkpoints are added to this repository as training runs finish; each sub-folder holds the weights, the exact config and its evaluation summary.

| Folder | Model | Status |
|---|---|---|
| — | EG-SWM (main) | training not started |

## Training data

* **SurgVU** (Zia et al., *Surgical Visual Understanding (SurgVU) Dataset*, arXiv:2501.09209): 155 da Vinci porcine training-lab cases, 837.8 h, with robot-log instrument and endoscope occupancy per arm.
* For kinematic grounding / benchmarks: Open-H-Embodiment subsets (CC-BY-4.0), JIGSAWS (Gao et al., M2CAI 2014; data-use agreement), SAR-RARP50 (CC BY-NC-SA 4.0).

## Licence and intended use

Weights are released under **CC BY-NC-SA 4.0** for non-commercial research, in line with the licences of the training data. The models simulate endoscopic video; they are **not** medical devices and must not be used for clinical decision-making or to control a robot on a patient.
