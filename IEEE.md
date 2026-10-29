# Coherence as a Reward: Label-Free Reinforcement Learning for Rational Belief Revision in Clinical Reasoning Models

**Research plan, draft v0.1 (October 29, 2026)**
**Lead:** Balu Bhasuran (School of Information, Florida State University)
**Collaborators:** [Clinical LLM PI, providing compute resources], [others TBD]
**Target venue:** IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

---

## 1. Summary

Our 2026 coherence paper showed that LLMs' stated diagnostic beliefs violate Bayesian belief-revision axioms (order invariance, update fidelity, redundancy insensitivity, positional anchoring). This project turns those axioms into **training rewards**.

Most of the rewards need **no labels**. A patient's posterior should be unchanged when:
- the findings are shuffled,
- a finding is duplicated, or
- a non-evidential cue (e.g., an attending's hunch) is inserted.

To stop the model from "cheating" by ignoring evidence (a constant output is perfectly invariant), we use a **contrastive coherence reward**. It asks for invariance *within* a patient while keeping posteriors well separated *between* patients. This reuses the floor-to-ceiling anchoring from the coherence paper.

The study has two parts:
1. **A frozen frontier panel.** Frontier models (Claude Opus 5.5, Sonnet 5, GPT-6, Grok 4.6, DeepSeek-R1) are evaluated once on AWS Bedrock before the credits expire.
2. **A fully local method.** All training and ablations run on our own GPU, so every revision requested by reviewers can be done after the credits expire.

---

## 2. Context and constraints

| Item | Status |
|---|---|
| AWS Bedrock credit | $1,100, $0 used, expires ~Dec 24, 2026 |
| Credit eligibility | **Unconfirmed.** Third-party models (Claude, GPT, Grok) are billed via AWS Marketplace, which standard promotional credits often exclude. |
| Review timing | Reviews will arrive months after the credits expire, so **no new API calls can be required during revision** |
| Local compute | GPU available (VRAM: **TBD**). Rough guide: 4B models need ~24 GB, 8B ~48 GB, 14B ~80 GB for GRPO with LoRA. |
| Prior work | Arun, George & Bhasuran (2026), *Probabilistic Coherence of Stated Diagnostic Beliefs in LLMs*: DDXPlus, 11 open-weight configs (4B–32B), 1.12M responses |
| Goal | One paper that serves both LLM reasoning/AGI interests (lead) and clinical LLM interests (PI) |

---

## 3. Background and gap

### 3.1 Key findings from our coherence paper
- Raw permutation divergence **rises** with scale on the Qwen3 ladder (0.128 to 0.291). The order effect as a fraction of each model's floor-to-ceiling range **falls** (0.215 to 0.127).
- Belief-update direction agreement with corpus evidence is near chance (0.454 to 0.549).
- Models name the correct diagnosis **1.8 to 5.6 times** more often than their own stated posterior ranks it first. The deficit is in the probabilistic interface, not in clinical knowledge.

### 3.2 Closest related work

| Work | Contribution | How this project differs |
|---|---|---|
| Qiu et al., *Nature Communications* 2026 (Bayesian teaching) | Supervised imitation of a normative Bayesian model; skills transfer to new tasks | Needs a Bayesian model at training time. Ours is **label-free RL**, usable where no such model exists (real cases). |
| Kim et al., ICML 2026 (*From Drift to Coherence*) | Martingale drift under answer resampling in multiple-choice QA; self-consistency loss | Static QA resampling, not **sequential clinical evidence revision** |
| Damani et al., ICLR 2026 (RLCR) | Brier-score calibration reward added to RLVR | Needs correctness labels. **Used as our strongest baseline.** |
| Andrews 2026 (*Revealed Rationality*) | Argues axiom checks from representation theorems can serve as label-free evaluation and regularization | Conceptual note with no LLM training experiments evident. Ours is an **empirical instantiation** for belief revision. |
| Yamin et al. 2026 (Horvitz, Wilder) | Tests whether LLM elicited beliefs and decisions are mutually consistent in clinical diagnosis | Diagnostic only, no training |
| Paleka et al. 2025; Lee et al. 2025 | Consistency checks and auxiliary rewards for forecasters | Static logical consistency in forecasting, not sequential evidence |
| DDXPlus fine-tuning (e.g., Kang et al. 2025) | LoRA fine-tuning for diagnostic accuracy | Accuracy only, with no coherence objective |

### 3.3 Gap
No published work (as of Oct 2026, to our search) does all of the following:
- trains LLMs with **label-free belief-revision axioms** as rewards,
- includes **anti-collapse** (contrastive) protection,
- works in a **clinical sequential-evidence** setting, and
- tests whether coherence training **improves or harms accuracy** and **transfers** to real cases and non-medical tasks.

**Risk:** this area is moving quickly (Andrews 2026 practically invites the experiment), so speed and an early arXiv preprint matter.

---

## 4. Research questions and hypotheses

**RQ1: Frontier coherence (frozen evaluation).** Do frontier reasoning models satisfy the belief-revision axioms, and does higher reasoning effort improve compliance?
- H1a: Floor-normalized order effects remain significantly above zero for all frontier models.
- H1b: Higher effort reduces order effects (two-sided; prior work on decision axioms suggests it may).

**RQ2: Label-free coherence training.** Can contrastive coherence rewards make small open models coherent?
- H2: Contrastive coherence reward (CCR) reduces floor-normalized order effects by at least 50% relative to the base model, without collapsing to constant outputs.

**RQ3: Coherence vs correctness.** Does coherence training improve, preserve or harm diagnostic accuracy and calibration?
- H3 (two-sided): CCR + accuracy reward matches or beats accuracy-only RLVR on top-1 accuracy, and improves calibration.

**RQ4: Transfer.** Do coherence gains transfer to held-out pathologies, real case reports and non-medical probabilistic tasks?
- H4: Coherence gains persist out of distribution at a reduced magnitude.

**RQ5 (secondary): Knows vs states.** Does training shrink the gap between "names the correct diagnosis" and "ranks it first in the stated posterior"?

---

## 5. Method

### 5.1 Setup
- A patient has an evidence set `E = {e_1, ..., e_n}` (DDXPlus findings).
- The model reports a posterior `p(· | E)` over 49 pathologies as structured output.

### 5.2 Label-free transformations (should leave the posterior unchanged)
- **Permutation:** `π(E)`, the same findings in a different order
- **Redundancy:** `E ∪ {e_i}`, one finding duplicated
- **Null cue:** `E ∪ {c}`, a non-evidential cue inserted. Examples: an attending's opinion, patient insistence, a prior AI suggestion, a demographic mention.

### 5.3 Rewards
Let `D` be the Jensen–Shannon divergence.
- **Within-patient divergence:** `d_w = mean over transformations T of D(p(E), p(T(E)))`
- **Between-patient divergence:** `d_b = mean D(p(E), p(E'))` over other patients in the GRPO batch

**Contrastive coherence reward (label-free):**
`R_coh = 1 − d_w / (d_b + ε)`, clipped to [0, 1]
This is maximized when the model is invariant within a patient **and** discriminative across patients. Constant outputs give `d_b ≈ 0` and are therefore penalized.

**Fidelity reward (DDXPlus only):** agreement between the model's update direction and magnitude and the corpus log-odds of each finding.

**Accuracy reward:** 1 if the top-1 diagnosis is correct.

**Combined reward:**
`R = α·R_acc + β·R_coh + γ·R_fid`
Ablate all combinations.

### 5.4 Formal component (planned)
**Proposition (sketch).** `R_coh` is maximized only by posterior maps that are invariant to evidence order, redundancy and null cues, and injective across distinct evidence multisets. Adding `R_fid` selects the Bayesian posterior under the corpus likelihood model. This gives TNNLS reviewers a formal anchor.

### 5.5 Training
- **Algorithm:** GRPO with LoRA (TRL, verl or Unsloth) plus a KL penalty to the reference model.
- **Base models:** Qwen3-4B and Qwen3-8B (continuity with the coherence paper), plus one other family (Gemma 3 or Llama 3.1 8B). 14B if VRAM allows.
- **Data:** DDXPlus training split. Hold out **entire pathologies** for out-of-distribution tests.
- **Seeds:** 3 per configuration.

---

## 6. Experiments

### 6.1 Frozen frontier panel (Bedrock, finish by Dec 15, 2026)
- **Patients:** ~300 DDXPlus patients (stratified across 49 pathologies) plus ~150 real case reports (MedCaseReasoning, CC-BY).
- **Calls per patient (~11):** base, 4 permutations, 2 redundancy variants, 2 null cues, 2 retests.
- **Models, low effort:** Opus 5.5, Sonnet 5, GPT-6 Sol, GPT-6 Luna, Grok 4.6, DeepSeek-R1.
- **Models, high effort (subset):** Sonnet 5 and GPT-6 Sol on 150 patients.
- **Serving check:** Qwen3-32B and gpt-oss on Bedrock **and** locally, to confirm hosted and local numbers match.
- **Fallback if Marketplace is excluded from the credit:** DeepSeek-R1, GLM 5, Qwen3-235B and gpt-oss-120b only.

### 6.2 Local method experiments (Dec 2026 to Feb 2027)
**Baselines:**
1. Base model with chain-of-thought prompting
2. "Reason like a Bayesian" prompting
3. Supervised fine-tuning on exact corpus posteriors (Bayesian-teaching analogue)
4. Accuracy-only RLVR
5. RLCR (accuracy + Brier)
6. Inference-time neuro-symbolic baseline: the LLM estimates per-finding likelihood ratios and Bayes' rule is applied in code

**Proposed variants:**
- CCR only (fully label-free)
- CCR + accuracy
- CCR + accuracy + fidelity

**Ablations:**
- each transformation type removed in turn
- without the between-patient (contrastive) term, to demonstrate collapse
- KL strength
- group size

### 6.3 Metrics
| Metric | Definition |
|---|---|
| Floor-normalized order effect | `(D_perm − floor) / (ceiling − floor)`, as in the coherence paper |
| Redundancy effect | Divergence after duplicating a finding, floor-normalized |
| Exchange rate | Shift in log-odds caused by a null cue, expressed as an equivalent likelihood ratio (target = 1) |
| Direction agreement | Fraction of updates whose sign matches the corpus log-odds |
| Top-1 / top-3 accuracy | Against the DDXPlus ground truth or the case-report diagnosis |
| Calibration | Brier score, ECE |
| Knows-vs-states ratio | Rate of naming the correct diagnosis vs rate of ranking it first in the stated posterior |
| Capability retention | Small MMLU and GSM8K subsets |
| Collapse check | Posterior entropy and between-patient divergence |

### 6.4 Transfer
- **Held-out pathologies** (DDXPlus)
- **Real case reports** (MedCaseReasoning): the label-free axioms plus accuracy against the gold diagnosis
- **Non-medical probability tasks:** synthetic Bayesian-updating tasks, probability-judgment tasks (Zhu & Griffiths style), and the Qiu et al. tasks if released

### 6.5 Statistics
- Mixed-effects models (random effects for patient and seed)
- Case-clustered bootstrap 95% CIs
- Holm correction for pre-specified contrasts
- Pre-registered analysis plan (OSF) before the main runs

---

## 7. Review-proofing protocol

1. **Bedrock is used only for the frozen frontier panel.** No method claim depends on new API calls.
2. **Over-collect now:** two prompt formats, retests, both effort levels and extra null-cue types. Reviewer "what about X?" questions are then answered offline.
3. **Archive everything:** raw JSON, prompts, model IDs and inference profiles, UTC timestamps, token usage and file hashes.
4. **Release the frozen panel as a dataset.** It's an additional contribution and makes the results transparent.
5. **All method results come from local open models** with 3 seeds. Release code and LoRA weights.
6. **Pre-empt standard objections:**
   - capability retention
   - collapse analysis
   - real-case transfer
   - a clinician (PI) review of a sample of outputs
   - serving equivalence between Bedrock and local
7. **Stop all Bedrock spending by Dec 18, 2026**, ahead of the credit expiry.

---

## 8. Budget (Bedrock credit)

| Item | Est. cost |
|---|---|
| Pilot (20 patients × all models) | ~$40 |
| Frontier panel, low effort (6 models) | ~$420 |
| High-effort subset (2 models) | ~$175 |
| Open-model teacher traces (gpt-oss-120b) | ~$20 |
| **Planned total** | **~$650** |
| **Reserve** | **~$450** |

These are rough estimates. The pilot measures real reasoning-token usage. Use Global cross-region inference (~10% cheaper) and batch inference where supported (open-weight models).

---

## 9. Timeline

| Dates | Tasks |
|---|---|
| Nov 1–4 | Check credit eligibility (Billing → Credits → Applicable products) and run a ~$1 billing test. Set up AWS Budgets alerts. Reuse the axiom code; select patients and cases; run the pilot. |
| Nov 5–11 | Freeze prompts; run the full low-effort frontier panel |
| Nov 12–18 | High-effort subset, over-collection runs, serving check |
| Nov 19–25 | Completeness audit, rerun failures, archive. Start the local GRPO pipeline on Qwen3-4B. |
| Nov 26–Dec 18 | Local training experiments. Any last Bedrock needs (teacher traces). **Bedrock spending stops Dec 18.** |
| Jan–Feb 2027 | Full training grid, ablations, seeds, transfer tests, statistics, writing. PI reviews the clinical sections and real-case outputs. |
| Mar 2027 | Submit to IEEE TNNLS; post on arXiv the same day |
| 2027 (revision) | All revisions run locally from archived data and on the local GPU |

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Credit excludes Marketplace models | Open-weight frontier fallback panel; the method half is unaffected |
| Reward hacking or collapse | Contrastive between-patient term, KL penalty, accuracy term, entropy monitoring |
| Coherence training lowers accuracy | This is itself a reportable finding (RQ3); combined-reward variants |
| Being scooped | Move fast; arXiv preprint at submission |
| Reasoning-token cost overrun | Pilot first, cap max tokens, drop the high-effort subset if the projection exceeds $900 |
| Limited GPU memory | Use 4B models with QLoRA; scale up only if VRAM allows |
| Reviewers ask about real data | MedCaseReasoning transfer plus clinician review included from the start |

---

## 11. Roles

**Lead (LLM reasoning):** method design, reward formulation, proposition, training pipeline, writing.

**PI (clinical LLM):**
- clinical validity of null cues and transformations
- real-case selection and review
- clinical framing of the exchange rate and anchoring results
- clinician sample review

**Co-authors:** pipeline engineering, statistics, frontier panel execution.

---

## 12. Open decisions

- [ ] GPU model and VRAM (decides 4B, 8B or 14B)
- [ ] Whether the credit covers Bedrock third-party models (Marketplace)
- [ ] Exact credit expiry date
- [ ] Final list of non-evidential cue types (with the PI)
- [ ] Choice of non-medical transfer tasks
- [ ] Author list and order

---

## 13. Venue strategy

| Order | Venue | 2025 IF | Notes |
|---|---|---|---|
| 1 | IEEE TNNLS | 8.9 | Main target. Method plus proposition plus ablations. Free on the standard route (check overlength rules). |
| Stretch | Information Fusion | 15.5 | If framed around evidence combination |
| Fallback | Knowledge-Based Systems | 7.6 | Neuro-symbolic and decision-support framing |
| Free alternative | TMLR | (none) | Diamond open access, rolling submissions, presentation route to NeurIPS/ICML/ICLR |

---

## References

- Arun G., George A.B., Bhasuran B. (2026). *Probabilistic Coherence of Stated Diagnostic Beliefs in Large Language Models: Anchored Tests on a Synthetic Clinical Corpus.* [venue/preprint TBD]
- Qiu L., Sha F., Allen K., Kim Y., Linzen T., van Steenkiste S. (2026). Bayesian teaching enables probabilistic reasoning in large language models. *Nature Communications.* https://arxiv.org/abs/2503.17523
- Kim S., Lee S., Fong E., Lee H., Lee J. (2026). From Drift to Coherence: Stabilizing Beliefs in LLMs. ICML 2026. https://arxiv.org/abs/2606.17832
- Damani M., Puri I., Slocum S., Shenfeld I., Choshen L., Kim Y., Andreas J. (2026). Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty. ICLR 2026. https://arxiv.org/abs/2507.16806
- Andrews I. (2026). Revealed Rationality: Label-Free Evaluation and Regularization from Representation Theorems. https://arxiv.org/abs/2608.05015
- Yamin K., Tang J., Cortes-Gomez S., Sharma A., Horvitz E., Wilder B. (2026). Do LLMs Act Like Rational Agents? Measuring Belief Coherence in Probabilistic Decision Making. https://arxiv.org/abs/2602.06286
- Tak A.N. et al. (2026). Sparks of rationality: Do reasoning LLMs align with human judgment and choice? arXiv:2601.22329
- Paleka D. et al. (2025). Consistency checks for language model forecasters. ICLR 2025.
- Zhu J.-Q., Griffiths T.L. (2024). Incoherent probability judgments in large language models. CogSci 2024.
- Kang L. et al. (2025). LLM-Driven Medical Document Analysis: Enhancing Trustworthy Pathology and Differential Diagnosis. https://arxiv.org/abs/2506.19702
- Tchango A.F. et al. (2022). DDXPlus: A New Dataset for Automatic Medical Diagnosis. NeurIPS Datasets & Benchmarks.
- Wu K. et al. (2025). MedCaseReasoning. https://github.com/kevinwu23/stanford-medcasereasoning
