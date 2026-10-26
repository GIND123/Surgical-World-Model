# 02 — External data: what we need, what we skip

*Checked 2026-10-27. Sizes and licences were read from the hosting APIs (Hugging Face, figshare), not from papers.*

The core model is trained on our own SurgVU holdings (~604 usable hours, `docs/01_data_audit.md`). SurgVU has robot-log state but **no kinematics**. Reviewers at IEEE T-MRB will ask two questions it cannot answer: *do the learned actions correspond to real robot motion?* and *how does the model compare on the public benchmarks other surgical world models report?* External data is needed for exactly those two questions, plus one clinical (human, non-porcine) generalisation test.

## A. Downloading now (open licences, no registration)

`python scripts/data/download_external.py <name>` → `data/external/<name>/` (resumable; figshare files are MD5-verified).

| Name | Source | Licence | Size | Why we need it |
|---|---|---|---:|---|
| `open_h_suturebot` | Open-H-Embodiment `Surgical/jhu/imerse/suturebot` | CC-BY-4.0 | 15.3 GB | dVRK suturing, video + synchronized kinematics. Benchmark used by SurgCast (2026). |
| `open_h_srth_chole` | Open-H `Surgical/jhu/imerse/srth_porcine_chole` | CC-BY-4.0 | 114.3 GB | dVRK ex-vivo porcine cholecystectomy (SRT-H), video + kinematics. Second SurgCast benchmark; closest tissue domain to SurgVU. |
| `open_h_crcd` | Open-H `Surgical/uic/uic_crcd_lerobot` | CC-BY-4.0 | 17.0 GB | da Vinci porcine cholecystectomy with kinematics (CRCD). |
| `sar_rarp50` | UCL RDR / figshare 24932529 + 24932499 | CC BY-NC-SA 4.0 | 33.6 GB | 50 *clinical* robot-assisted prostatectomies with gesture labels + instrument segmentation: human-anatomy transfer and gesture-level actions. |

Open-H-Embodiment (HF `nvidia/PhysicalAI-Robotics-Open-H-Embodiment`, LeRobot v2.1, 4.54 TB in total, ~770 h across 9 embodiments) has further subsets we can pull later if an experiment needs them: `jhu/imerse/wound_closure` (53 GB), `srt_needle_pickup_handover` (9.9 GB), `srt_tissue_lift` (3.4 GB), `hamlyn` (24 GB, dVRK skills), `ubc` (2.9 GB, JIGSAWS-style tasks on dVRK), `cmr_surgical/*` (2.8 TB, Versius procedures with kinematics).

## B. Optional — need *your* signature (form / data-use agreement)

| Dataset | Access | Value | Priority |
|---|---|---|---|
| **JIGSAWS** (JHU CIRL) | request form + DUA | The classic da Vinci video + kinematics benchmark (suturing, knot tying, needle passing); reviewers often expect it. Overlaps in purpose with SutureBot. | Medium |
| **AutoLaparo** (CUHK, MICCAI 2022) | request form | Has a *laparoscope-motion prediction* subset (labelled camera moves) — a direct external test of our explicit camera-action branch. | Medium |
| **Cholec80 / CholecT50** (CAMMA) | request form | Only if we also want to compare with laparoscopic video-generation papers (Endora, HieraSurg, Surg-UniWorld). Not robotic. | Low |

## C. Pretrained models (open weights, downloaded on demand into `checkpoints/`)

* **Latent tokenizers:** an image VAE (FLUX.1-schnell AE, Apache-2.0) and a video VAE (Wan2.x VAE) — ablated against a SurgWM-style VQ-VAE trained from scratch.
* **Baselines:** Cosmos-Predict2.5-2B and **Cosmos-H-Surgical-Simulator** (NVIDIA Open Model Licence; kinematics-conditioned, trained on Open-H). SurgWM has released no code or weights ("Codes coming soon!"), so we re-implement it from the paper.
* **Measurement models:** optical flow (RAFT / SEA-RAFT) and point tracking (CoTracker3) for camera ego-motion and instrument trajectories; I3D for FVD; our SurgVU instrument detector (re-trained without the test cases) for instrument-consistency metrics.

## D. PhysioNet — nothing to take

We searched PhysioNet's catalogue for surgery-related projects. None contains surgical video paired with robot kinematics, which is what a visual world model consumes.

| PhysioNet project | Access | Why not |
|---|---|---|
| Endoscapes2023 (CVS + segmentation, laparoscopic cholecystectomy) | Restricted (DUA) | 1 fps frames for critical-view-of-safety assessment — no usable dynamics; non-robotic; also distributed by CAMMA directly. |
| EEG + eye-gaze for robot-assisted surgery / FLS tasks | Open | Physiological signals of the operator, no endoscopic video or kinematics. |
| VitalDB, INSPIRE, JSMF ACCESS, multimodal anaesthesia indices | Open / credentialed | Intraoperative vital signs and EEG — patient physiology, no video. A possible *future* multimodal extension (patient-state world model), out of scope here. |

## E. Deliberately not used

* **SurgToolLoc-2022** (SurgWM's training set): same Intuitive porcine-lab source as SurgVU, so overlap/leakage with our test cases cannot be excluded, and SurgWM's weights are unavailable anyway. We reproduce SurgWM on our own case splits instead, which is the fairer comparison.
* **Large web/YouTube surgical corpora**: different embodiments and no robot state; not needed for the hypotheses we test.
