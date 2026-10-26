# 03 — Literature review: surgical world models (state of the field, October 2026)

*Method: systematic arXiv API queries (12 queries over surgical/endoscopic × world model / video generation / video prediction / latent action / kinematics / inverse dynamics, 2024-01 → 2026-10; 74 unique hits, screened by abstract), plus targeted searches for the two reference papers, general world-model methods and datasets. IDs below were verified against the arXiv API.*

## 1. The two reference papers

| | **MeWM — Medical World Model** (Yang et al., ICCV 2025, arXiv 2506.02327; IEEE Xplore 11444416) | **SurgWM — Surgical Vision World Model** (Koju et al., MICCAI-DEMI 2025, LNCS, arXiv 2503.02904) |
|---|---|---|
| World-model framing | policy (GPT-4o proposes TACE protocols) → **dynamics** (3D latent diffusion simulates post-treatment CT) → **inverse dynamics** (survival model scores outcomes) → beam search over actions | Genie recipe on unlabeled video: VQ tokenizer → latent action model (12 codes) → MaskGIT dynamics (62.5 M params) |
| Data | 338 in-house + 78 public liver-TACE CT pairs | SurgToolLoc-2022 (24,695 × 30 s da Vinci porcine clips), 16 frames at **1 fps**, **120×180 px** |
| Evaluation | radiologist Turing test, FID/LPIPS, survival C-index, protocol-selection F1 (+13 % for physicians with MeWM), ablations | PSNR/SSIM (≈ 17–19 dB), FVD, ΔPSNR ≈ 2 dB; **no baselines, no ablations, no downstream task**; code "coming soon" |
| What we adopt | the *policy / dynamics / inverse-dynamics* decomposition and a clinically meaningful downstream test; expert Turing-style evaluation | action-free learning from robotic video of the *same source* as our SurgVU data |
| What we fix | — | entangled latent actions, very low resolution and frame rate, no comparison, no evidence the actions mean anything physically |

## 2. Taxonomy of surgical video generation / world models (2024–2026)

### 2.1 Unconditional or text-conditioned generation
Endora (MICCAI 2024, 2403.11050), SurGen (2408.14028), Bora (2407.08944), Ophora (MICCAI 2025, 2505.07449), HieraSurg (MICCAI 2025, 2506.21287; phase → segmentation → video), hierarchical curriculum for robotic thyroidectomy (2610.17684), **FLAIR** (2610.09800; latent action priors learned from optical flow, injected into a frozen backbone, text-only inference; SurgActionClip-30K + SurgMetrics).
*Limit:* realistic appearance, but no notion of *who* acts or *how*; not usable as a controllable simulator.

### 2.2 Structure-conditioned generation (masks, flow, depth, trajectories, graphs)
VISAGE (action graphs, 2410.17751), interactive laparoscopic generation (masks, 2406.06537), SurgSora (RGB-D + flow + user trajectories, MICCAI 2025, 2412.14018), SurgiFlowVid (RGB + flow dual prediction, 2510.07345), **SAW** (text + reference frame + tissue-affordance mask + **2-D tool-tip trajectories**, 12,044 laparoscopic clips, 2603.13024), **Surg-UniWorld** (edge/depth/flow control experts on Wan2.2, Cholec80-SurgWAM benchmark, 2608.06770), DepthPilot (colonoscopy, 2604.26232), SWoMo (neuro-symbolic simulator → diffusion, cataract, 2605.16530).
*Limit:* conditions require annotation or a simulator at inference; control is over pixels/trajectories, not over the robot.

### 2.3 Kinematics-conditioned (robot-action) world models
**Cosmos-H-Surgical / SurgWorld** (2512.23162; SATA dataset, IDM pseudo-kinematics for VLA training), **Open-H-Embodiment** + **Cosmos-H-Surgical-Simulator** (2604.21017; Cosmos-Predict2.5-2B, 44-D kinematics, 9 embodiments), **Cosmos-Surg-dVRK** (IEEE RA-L 2026; world model as automatic policy evaluator), **Cosmos-H-Dreams** (2608.24199; distilled to ~160 FPS real-time simulator), **KVLR** (NeurIPS 2026, 2605.08712; kinematics lifted to five image-aligned control maps with hierarchical routing), **SurgCast** (2610.15645; predicts future instrument skeletons from robot state + actions; SutureBot and SRT-H porcine cholecystectomy), **Surgical WAM** (2608.11204; Cosmos-Policy world-action model, action-free video pre-training raises sim success 63.5 → 77.8 %), FoMo-FD (2607.27511; flow-matching WM for failure detection on dVRK), suturing world models (2503.12531), EndoWAM (2608.01221), CrossScope (2608.03211), joint visual–trajectory forecasting (2608.20284).
*Limit:* need synchronized kinematics, which exist only for lab platforms (dVRK, research da Vinci, Versius) — a few hundred hours, mostly phantoms/ex-vivo. Clinical and in-vivo robotic video, by far the larger resource, carries none.

### 2.4 Action-free / latent-action surgical models
**SurgWM** (above), FLAIR (above), latent-action-guided video–language features for instrument–tissue interaction recognition (2607.19889), **EndoWorld-JEPA** (2610.27890; JEPA pre-training conditioned on optical motion that is split into *dominant displacement* vs *residual* motion — representation learning, not control), **SurgVIL** (2608.16058; approximate kinematics estimated from open surgical videos as weak supervision for imitation learning).
*Limit:* one monolithic latent action per transition. In surgical video that latent must explain endoscope motion, two (or three) instruments, and tissue response at once.

### 2.5 Long horizon, hierarchy, benchmarks
SurgVista (2606.19889; deformation-consistency regularisation + drift-adaptation training; SurgWorld-Bench), S2-HWM (2608.13103; event-structured hierarchical WM, SurRoL), SurgVeo + Surgical Plausibility Pyramid (2511.01775; Veo-3 zero-shot fails instrument-operation, environment-feedback and intent plausibility as judged by four surgeons), SurgWMBench (2608.08070; argues FVD/CD-FVD are misaligned with instrument-motion planning), UniSWM (unified understanding + trajectory + generation, 1.81 M samples). Surveys: 2608.26214, 2607.25242, 2606.05925, 2606.00133, 2511.16333.

## 3. General world-model methods we build on

* **Latent actions:** Genie (2402.15391), LAPA (2410.11758), AdaWorld (2503.18938), villa-X (2507.23682).
* **Why monolithic latent actions break on surgical video:** *Latent Action Learning Requires Supervision in the Presence of Distractors* (LAOM, ICML 2025, 2502.00379) — action-correlated distractors cripple LAPO-style latent actions; supervising with true actions on as little as 2.5 % of data improves downstream performance 4.2×. *Object-Centric Latent Action Learning* (AAAI 2026, 2502.09680) — slot decomposition halves the distractor penalty. **FLAM** (2602.16229) — factorised latent actions, one per *learned* scene factor, beat monolithic models on multi-entity scenes. **EgoWAM** (2607.08436) — factoring out ego-motion makes motion targets comparable across moving and static cameras.
* **Dynamics backbones and training:** diffusion world models (DIAMOND 2405.12399), Diffusion Forcing (2407.01392), Self Forcing (2506.08009), Cosmos / Cosmos-Predict2.5 (2501.03575, 2511.00062), Wan (2503.20314); latent-feature world models for planning (DINO-WM 2411.04983, V-JEPA 2-AC 2506.09985); UniSim (2310.06114).

## 4. Gap analysis

1. **Embodiment-blind actions.** Action-free surgical WMs (SurgWM, FLAIR) learn one entangled latent; kinematic WMs (Cosmos-H, KVLR, SurgCast) need kinematics. Nobody structures the action space of a world model trained on *unlabeled in-vivo robotic video* like the robot itself — an endoscope plus independently driven instrument arms.
2. **Camera ego-motion is treated as a nuisance or ignored.** In robotic surgery the endoscope is itself a robot arm the surgeon commands, and camera moves happen in a separate control mode. EndoWorld-JEPA separates dominant displacement for *representation learning*; no surgical WM makes ego-motion an explicit, controllable action.
3. **Free robot-state signals are unused.** da Vinci system logs (instrument per arm, camera arm) and the console overlay (which hand drives which arm, energy activation) are available for thousands of hours of video but have not been used to supervise or factorise world-model actions — exactly the "small amount of supervision" LAOM shows is decisive.
4. **Evaluation stops at pixels.** Most papers report FVD/PSNR; SurgWMBench and SurgVeo show this misses instrument behaviour and plausibility. Missing: metrics that check a rollout against the *robot's* state (right instruments on the right arms, idle arms still, commanded camera motion realised) and evidence that learned actions correspond to real kinematics.
5. **Thin baselines and ablations.** SurgWM reports none; several 2026 papers compare only against general-purpose video models.

These gaps motivate the hypotheses and experimental plan in `docs/04_research_plan.md`.
