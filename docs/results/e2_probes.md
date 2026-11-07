| Features | Setting | SutureBot pos | SutureBot rot | SutureBot jaw | CRCD pos | CRCD rot | CRCD jaw | JIGSAWS pos | JIGSAWS rot | JIGSAWS jaw |
|---|---|---|---|---|---|---|---|---|---|---|
| Latent change in the arm region | model-free | 0.40 | 0.07 | 0.07 | 0.00 | -0.00 | -0.01 | 0.05 | -0.00 | 0.00 |
| Latent change over the frame | model-free | 0.20 | 0.04 | 0.03 | 0.02 | 0.00 | 0.00 | 0.09 | -0.01 | 0.01 |
| Endoscope action | model-free | 0.24 | 0.05 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | -0.00 | 0.00 |
| Monolithic latent + camera | zero-shot | 0.28 | 0.05 | 0.02 | 0.05 | 0.01 | 0.00 | 0.10 | -0.01 | 0.02 |
| Monolithic VQ + camera (SurgWM-style) | zero-shot | 0.12 | 0.02 | 0.02 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Slots + camera (FLAM-style) | zero-shot | 0.43 | 0.08 | 0.01 | 0.06 | 0.01 | 0.00 | 0.15 | -0.01 | 0.01 |
| EG-SWM, no camera action | zero-shot | 0.30 | 0.02 | 0.02 | 0.01 | 0.00 | -0.00 | 0.04 | -0.00 | -0.00 |
| EG-SWM | zero-shot | 0.29 | 0.01 | 0.02 | 0.01 | 0.00 | -0.01 | 0.05 | 0.00 | 0.00 |
| EG-SWM + scene action | zero-shot | 0.33 | 0.01 | 0.02 | 0.01 | 0.00 | -0.00 | 0.04 | -0.00 | 0.00 |
| EG-SWM, diffusion loss only | zero-shot | 0.11 | 0.00 | -0.00 | 0.00 | 0.00 | 0.00 | 0.07 | -0.00 | 0.01 |
| EG-SWM, arm actions not grounded | zero-shot | 0.12 | 0.02 | 0.02 | 0.01 | 0.00 | -0.00 | 0.04 | -0.00 | 0.00 |
| EG-SWM, wrong arm binding | zero-shot | 0.02 | -0.01 | 0.01 | 0.00 | 0.00 | -0.00 | -0.01 | -0.00 | 0.00 |
| EG-SWM, size B | zero-shot | 0.28 | 0.00 | 0.02 | 0.01 | 0.00 | -0.01 | 0.05 | 0.00 | 0.00 |
