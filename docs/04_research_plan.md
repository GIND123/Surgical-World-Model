# 04 — Research plan: an embodiment-grounded surgical world model

**Target:** IEEE Transactions on Medical Robotics and Bionics (T-MRB) full paper; NBRCOM-2026 (AIIMS Nagpur, 18–22 Nov 2026, *Health Sciences → Innovation, AI & Medical Robotics*) as an early showcase.
**Working name:** EG-SWM (Embodiment-Grounded Surgical World Model).

## 1. Problem

A surgical world model predicts how the operative scene evolves under the robot's actions. Today such models come in two kinds (`docs/03_literature_review.md`):

* **kinematics-conditioned** models (Cosmos-H-Surgical, KVLR, SurgCast, Surgical WAM) — controllable, but trainable only on the few hundred hours of lab data that record kinematics;
* **action-free** models (SurgWM, FLAIR) — trainable on abundant video, but driven by one *monolithic* latent action that must explain endoscope motion, two or three instruments and tissue response at once. LAOM (ICML 2025) shows such latents collapse in the presence of action-correlated distractors unless a little true supervision is added.

Robot-assisted surgery is unusual among video domains: **the embodiment is observable even when kinematics are not.** In our SurgVU corpus (`docs/01_data_audit.md`) the da Vinci logs tell us, at every instant, which arm holds the endoscope and which instrument is on each other arm; on 81 % of usable frames the console overlay also shows which hand drives which arm and when energy fires; and teleoperation itself is structured — at most two instrument arms move at once, the third is parked, and the camera moves in a separate control mode.

## 2. Hypotheses

* **H1 — Factorisation.** Structuring the action space like the robot — an explicit *endoscope ego-motion* action plus one latent action per *instrument arm*, bound to arms by robot-log instrument identity and image localisation — yields a world model that is more controllable (ΔPSNR, action-effect accuracy), more disentangled (per-arm locality, camera compliance) and more faithful to the robot state than (a) a monolithic latent action (SurgWM/Genie), (b) unsupervised factor slots (FLAM), at matched capacity, data and compute.
* **H2 — Grounding.** The per-arm latent actions inferred from video encode real robot motion: on platforms that record kinematics (Open-H: SutureBot, SRT-H cholecystectomy, CRCD) a linear probe from a *frozen* action encoder to per-arm end-effector displacement reaches substantially higher R² than probes from monolithic or FLAM latents, and the camera action predicts endoscope motion.
* **H3 — Transfer and data efficiency.** Pre-training EG-SWM on ~600 h of unlabeled in-vivo video, then attaching a small kinematics→action adapter, beats training from scratch and matches or beats Cosmos-H-Surgical-Simulator on the SutureBot and SRT-H benchmarks, with the largest margin when only 5–25 % of the kinematic data is used.
* **H4 — Long-horizon embodiment consistency.** Conditioning on the robot state (instrument per arm, active hands) reduces instrument hallucination/disappearance and drift over 8–16 s autoregressive rollouts.

Following MeWM's decomposition: the **dynamics model** is the world model; its action encoder is an **inverse-dynamics model** that turns any robotic video into embodiment-structured pseudo-actions (camera + per-arm), which H2 validates; a learned action prior p(a | history) is the **policy** that lets the model imagine surgeon behaviour, used for anticipation and for downstream evaluation.

## 3. Method (v1 design — details settle with the experiments)

1. **Observation.** Endoscope field only (active columns 193–1086, rows 56–679 → no console UI, so no shortcut to instrument names), 224×320 for development, 288×416 for final runs; 8 fps; training clips of 16 frames (4 context + 12 predicted); rollouts to 16 s.
2. **Tokenizer.** Frozen pretrained continuous VAE (f8, 16 ch); SurgWM-style VQ-VAE trained from scratch as an ablation.
3. **Robot state s_t** (conditioning, never pixels): camera arm, instrument class on each arm (logs), driving hand L/R per arm and energy event (UI readout, when visible).
4. **Actions a_t = (c_t, u_t^1..K, e_t).**
   * **c_t — endoscope ego-motion**, explicit: similarity/homography fitted with RANSAC to dense optical flow on tissue pixels, instruments masked out. Validated on synthetic warps and on Open-H episodes with recorded endoscope kinematics.
   * **u_t^k — per-arm latent action** (low-dimensional, bottlenecked): inferred from consecutive latents restricted to arm k's instrument region. Regions come from the instrument detector + tracker; detections are bound to arms through the instrument identity in the robot log (same-class ties broken by image side and the UI's L/R tag).
   * **e_t — energy activation** (UI readout).
5. **Dynamics.** Spatio-temporal transformer (spatial attention + causal temporal attention) on VAE latents, trained by flow matching with diffusion-forcing noise schedules for stable autoregressive rollout. c_t and e_t enter through AdaLN; each arm's u_t^k is a token carrying arm and instrument embeddings that the spatial tokens cross-attend to. Weak supervision: arms not driven by a hand (UI) are regularised towards zero action.
6. **Training.** Inverse-dynamics encoder and dynamics trained jointly (AdaWorld/LAPA style); a two-stage Genie-style variant is kept for comparability with SurgWM.

## 4. Experiments

| ID | Data | Question | Primary metrics |
|---|---|---|---|
| E1 | SurgVU (case-level test split) | H1, H4: fidelity, controllability, disentanglement, embodiment consistency | PSNR/SSIM/LPIPS per horizon, FVD, ΔPSNR, camera compliance, arm locality, ECS (below), drift curves |
| E2 | Open-H SutureBot, SRT-H, CRCD | H2: do latent actions encode kinematics? | linear-probe R² (per-arm Δpose, endoscope Δpose), inverse-dynamics error |
| E3 | SutureBot, SRT-H (benchmarks of SurgCast) | H3: kinematics-conditioned prediction, data efficiency | FVD, PSNR, SSIM, LPIPS, instrument-tip trajectory error vs commanded kinematics; 5/10/25/100 % data curves |
| E4 | SAR-RARP50 (clinical prostatectomy) | transfer from porcine lab to human surgery; gesture-level control | zero-shot / fine-tuned fidelity, gesture-conditioned controllability |
| E5 | E1 rollouts | expert realism and plausibility (MeWM / SurgVeo style) | surgeon 2-AFC real-vs-generated, Surgical Plausibility Pyramid ratings |
| E6 | E1 | ablations (§6) | E1 metrics |

**Embodiment Consistency Score (ECS)** — new, robot-log-grounded: run an instrument detector (re-trained *without* test cases) on rollouts and score (i) instrument-identity precision/recall against the robot state, (ii) hallucination and disappearance rates over the horizon, (iii) idle-arm stillness, and (iv) commanded vs realised camera motion.

## 5. Baselines

1. **SurgWM** re-implemented from the paper (VQ tokenizer, 12-code LAM, 62.5 M MaskGIT dynamics) on our splits.
2. Same backbone as ours with a **monolithic latent action** (Genie/AdaWorld style).
3. Same backbone with **FLAM-style unsupervised factor slots**.
4. **Unconditional** and **text/task-conditioned** (SurgVU25 descriptions) variants of our backbone.
5. **Large pretrained video model** fine-tuned with LoRA (Cosmos-Predict2.5-2B or Wan2.x) — upper bound on appearance quality, no per-arm control.
6. On E3: **Cosmos-H-Surgical-Simulator** (released weights), training from scratch, and published SurgCast/KVLR numbers where splits match.

## 6. Ablations (E6)

* Action structure: none / monolithic / camera only / arms only / camera + arms / + energy.
* Camera action: explicit (estimated) vs latent; similarity vs homography.
* Arm grounding sources: robot log only / + detector regions / + UI hand assignment; with injected label noise.
* Arm action bottleneck: dimension, continuous vs FSQ codebook size, KL weight; forward-decoder (latent-action-model) objective on/off; idle-arm regulariser on/off.
* Tokenizer: pretrained image VAE vs video VAE vs VQ-VAE from scratch.
* Temporal: 2 / 4 / 8 fps; context 1 / 4 frames; diffusion forcing on/off.
* Scale: 10 / 30 / 100 % of training cases; model S (~60 M, SurgWM-sized) / B (~200 M) / L.
* UI visible vs masked (shortcut check).

## 7. Protocol

* **Splits:** by case, stratified by camera arm (USM2/USM3), case length and task mix — ~123 train / 12 val / 20 test; fixed in `configs/splits/`. Model selection on val only; test touched once per reported number.
* **Windows:** quality filter (no black / low-colour / blank frames inside a window); fluorescence kept as a tagged sub-domain and reported separately.
* **Statistics:** mean ± 95 % bootstrap CI over test cases; paired tests between methods; 3 seeds for the main comparisons.
* **Compute:** one RTX PRO 5000 Blackwell (48 GB). Latents are pre-computed once; action-model comparisons share tokenizer, backbone, data and training budget.

## 8. Timeline (journal-first)

| Weeks (2026–27) | Work |
|---|---|
| 1: 27 Oct – 2 Nov | data pipeline (windows, splits, latents, camera ego-motion, detection + tracking + arm binding, UI readout), metric suite, SurgWM re-implementation |
| 2–3: 3 – 16 Nov | EG-SWM v1, E1 core comparison → NBRCOM showcase (18–22 Nov) |
| 4–6: 17 Nov – 7 Dec | E2, E3 on Open-H; large-model and Cosmos-H baselines |
| 7–9: 8 – 28 Dec | ablations, long horizon, E4, expert study |
| 10–12: Jan 2027 | T-MRB manuscript, supplementary video, code release |

## 9. Risks

| Risk | Mitigation |
|---|---|
| Missing uninstall events corrupt arm occupancy (`01_data_audit.md` §3) | validate against the UI arm panels; use only windows where logs and UI agree |
| Detector misses / identity switches blur arm regions | dilated regions, track smoothing, ablation with oracle-free noise injection |
| Ego-motion estimates fail in smoke / low texture | confidence weighting; drop low-confidence windows |
| Evaluator bias (detector saw test cases) | re-train detector on train cases only for ECS |
| Comparisons judged unfair | shared tokenizer/backbone/budget; re-implement baselines on our splits; report released-model numbers separately |
| SurgVU licence (non-commercial research) | data stays local; only code, aggregate results and (if permitted) weights are released |

## 10. Pre-submission checklist

Status on 4 Nov 2026 of a reviewer-oriented checklist: comparisons, standard datasets, ablations, trustworthy numbers. Jobs are in `configs/experiments/`; tables in `docs/results/` are generated by `scripts/eval/aggregate.py`.

| Item | Status |
|---|---|
| **Comparisons** | |
| Action interfaces at equal size, data and budget: none, robot state, monolithic (Genie-style), 12-code VQ (SurgWM-style), slots (FLAM-style), EG-SWM, EG-SWM + scene; with and without the endoscope action | done, 10 runs (`e1_main`) |
| SurgWM, full re-implementation (own tokenizer, latent action model, MaskGIT dynamics) | queued: smoke test `301`, then `600`–`603` (60k steps per stage, evaluated on the E1 windows) |
| Cosmos-H-Surgical-Simulator on SutureBot and SRT-H | feasible, set up after E3 in its own environment. Weights `nvidia/Cosmos-H-Surgical-Simulator`, code `NVIDIA-Medtech/Cosmos-H-Surgical-Simulator` (Apache-2.0): Cosmos-Predict2.5-2B fine-tuned on 32 Open-H datasets (SutureBot and SRT-H included), 1 context frame + 12 kinematic actions (dVRK: 20-D, padded to 44-D) per 12-frame chunk, 512×288, about 10 fps. It holds out the trailing 5 % of each dataset's samples (highest episode indices); our test splits are also the highest episodes, so 73 of the 146 SutureBot test episodes (783 windows) and 477 of the 951 SRT-H test episodes (692 windows) are unseen by it. Every method is compared on that subset (a sample of it: about 1 min per window for three chunks) |
| Large pretrained video model fine-tuned with LoRA | deferred until after the B flagship: an appearance upper bound without per-arm control |
| Reference rows: copy last frame, VAE reconstruction, actions of another window, no actions | done |
| **Datasets** | |
| E1 SurgVU (19 test cases) | done |
| E2 SutureBot, CRCD, JIGSAWS (synced episodes only) | queued `400`–`450` |
| E2b endoscope action on CRCD and AutoLaparo | done |
| E3 SutureBot / SRT-H | queued `499`–`508` / `610`–`619` |
| E3 on CRCD | dropped: 11 synced episodes (1,963 training windows) are too few for data-efficiency curves |
| E4 SAR-RARP50 | queued `620`–`625`: fidelity zero-shot, after action-free adaptation, and from scratch. Gesture-conditioned control deferred |
| E5 expert study | planned; needs surgeons; material from the E1 rollouts |
| **Ablations (E6)** | |
| Endoscope action | done (`*_nocam`) |
| Whole-frame pooling, no forward decoder, shuffled arm binding, console readout | `300` and `305`–`320` |
| Boxes held across detector gaps (binding coverage 73 % → 79 % of a mounted instrument's steps) | queued `325` |
| KL weight 10⁻³, 10⁻⁵ | queued `720`, `721` |
| Scene action | done |
| Learned-only action dropout, camera-stabilised inference | not run at full size: in the pilots (`096`, `097`) both lowered replay accuracy and neither restored endoscope control |
| Tokenizer (FLUX.2 vs Wan2.2) | reconstruction benchmark only: FLUX.2 latents give 16× more tokens per window, so a world model on them does not fit the S budget |
| Data scale (10 %, 30 % of the training cases) | queued `730`, `731` |
| Model scale | the B flagship is designed once `300`–`320` are in |
| Detector quality | open. EG-SWM's gap to the frame-wide actions sits in windows without bound instruments (`e1_binding`), so detector recall matters; degraded detections would quantify it |
| **Trustworthy numbers** | |
| Seeds | queued: seeds 1 and 2 of EG-SWM, monolithic, slots and VQ (`710`–`717`) |
| Case-bootstrap intervals; cases won | done (`e1_main`, `e1_paired`) |
| Quality over the horizon | done (`docs/figures/e1_horizon.png`) |
| **Sensitivity** | |
| Sampler steps and guidance without `case_126` | queued `330` (job `204` still covered it) |
| Context length | queued `331` |
| Command size (proportionality of endoscope control) | queued `332` |
| Bound vs unbound instruments | done (`e1_binding`) |
| Rollouts of 8–16 s (H4) | queued `333`: 14 s in five 2.8 s chunks (the last 5 generated frames re-encoded as the next context, actions replayed on each chunk's grid) on 95 clips of the 19 test cases (`configs/eval/surgvu_test_long.json`); repeated for the B flagship |
| **Own explanations** | |
| The scene action's weaker endoscope control | worded as a hypothesis (README, paper); direct test queued `334`: the share of the change caused by replacing the scene action (and, as controls, the arm, monolithic and slot actions) that lands in grid cells the moving endoscope reveals, over their area share |
| Claims rest on fidelity, control and locality together | done |
| Detector floor (5.3 %) shown with the hallucination rate | done |
| **Cost** | parameters, training time and memory in `cost`; time and memory of one prediction from `302` |
| **Figures** | architecture output row from a model rollout, interfaces side by side, action swaps, endoscope commands, failure cases: `302` |
| **Release** | weights of every finished run on Hugging Face (automatic); split and evaluation windows in `configs/`; data availability statement in the paper; a tagged code release at submission |
| **Consistency** | SRT-H status corrected (prepared and sync-checked); pilot and full-run numbers kept in separate tables; tables generated, not typed; `case_126` excluded from every test (the reference ECS was re-run by `206`; the sampler sweep by `330`) |
