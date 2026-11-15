<!-- Draft cover letter for the T-MRB submission. Fill the bracketed fields; check every number against the final
     main.pdf before submitting. The letter goes to the editor only, so it may name the authors; the manuscript must not. -->

[Date]

The Editor-in-Chief
IEEE Transactions on Medical Robotics and Bionics

Dear Editor-in-Chief,

We submit the manuscript "Embodiment-Grounded Surgical World Models: Learning Arm-Factorized, Camera-Aware Actions from
Unlabeled Robotic Video" for consideration as an Original Article in the IEEE Transactions on Medical Robotics and
Bionics.

Surgical world models, learned simulators that predict endoscopic video in response to the robot's actions, could serve
as simulators, policy evaluators and data generators for robot-assisted surgery. They are trained either on robot
kinematics, which are recorded for only a few hundred hours of laboratory data, or on a single latent action inferred
from video that must explain endoscope motion, several instruments and tissue at once. The manuscript shows that the
robot's own system logs, which record the arm that holds the endoscope and the instrument on every other arm, can give a
world model learned from 476 h of in-vivo da Vinci video without kinematics an action space that mirrors the robot: an
explicit endoscope action and one latent action per instrument arm.

Main findings:

- Among the action interfaces compared at equal capacity, data and training budget, it is the only one that keeps the
  endoscope and each instrument controllable while predicting better than a copied frame. Frame-wide latent actions
  predict 0.46-0.76 dB better under action replay but realise only 5-16 % of a commanded endoscope move (ours: 70 %).
  The trade-off holds at three times the model size.
- Initialising kinematics-conditioned models from it adds 1.03-1.11 dB on two public dVRK benchmarks with 10 % of the
  training episodes.
- The evaluation goes beyond pixel fidelity: controllability, per-arm locality, compliance with commanded endoscope
  motion, consistency with the robot's state, linear probes against real kinematics, and transfer to clinical video,
  with case-level bootstrap intervals, several training seeds, ablations of every component and a re-implementation of
  the closest action-free surgical world model.

We believe the work fits the journal's scope in surgical robotics: it turns signals that every teleoperated surgical robot
records into controllable, learned simulation of the operative scene.

The manuscript is original, has not been published, and is not under consideration elsewhere. All authors approved the
submission. Following the double-anonymous review process, the manuscript and supplementary material carry no author
names, affiliations, acknowledgements or identifying links. The code, configurations and model weights are publicly
available; their links are withheld from the manuscript and can be provided to the editor on request. The work analyses
only previously published datasets of animal and human procedures, under their licences, and collected no new data from
humans or animals. [Conflicts of interest: none / state them.] [Funding: state it.]

The submission consists of the manuscript (12 pages), a supplementary PDF and Supplementary Movie S1.

[Optional: suggested reviewers without conflicts of interest, with affiliations and e-mail addresses.]

Sincerely,

[Name, affiliation, address, e-mail, ORCID], on behalf of all authors
