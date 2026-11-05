| Model | Size | Steps | Training (h) | Windows/s | Training memory (GB) | Prediction (s) | VAE decoding (s) | Inference memory (GB) |
|---|---|---|---|---|---|---|---|---|
| None (unconditional) | S | 60000 | 4.6 | 117 | 26.0 | – | – | – |
| Robot state | S | 60000 | 5.2 | 103 | 29.7 | – | – | – |
| Robot state + camera | S | 60000 | 5.2 | 103 | 29.7 | – | – | – |
| Monolithic latent (Genie) | S | 60000 | 5.6 | 96 | 31.4 | – | – | – |
| Monolithic latent + camera | S | 60000 | 5.6 | 96 | 31.8 | 0.28 | 0.37 | 3.7 |
| Monolithic VQ + camera (SurgWM-style) | S | 60000 | 5.6 | 96 | 31.4 | – | – | – |
| Slots + camera (FLAM-style) | S | 60000 | 5.6 | 96 | 31.5 | – | – | – |
| EG-SWM, no camera action | S | 60000 | 5.6 | 96 | 31.4 | – | – | – |
| EG-SWM | S | 60000 | 5.6 | 96 | 31.4 | 0.28 | 0.37 | 3.7 |
| EG-SWM + scene action | S | 60000 | 5.6 | 96 | 31.4 | – | – | – |
| EG-SWM, diffusion loss only | S | 60000 | 5.4 | 99 | 31.4 | – | – | – |
| EG-SWM, arm actions not grounded | S | 60000 | 5.6 | 96 | 31.4 | – | – | – |
| EG-SWM, wrong arm binding | S | 60000 | 5.6 | 96 | 31.4 | – | – | – |
| EG-SWM + console readout | S | 60000 | 5.6 | 96 | 31.4 | – | – | – |
| EG-SWM, boxes held across detector gaps | S | 60000 | 5.6 | 96 | 31.4 | – | – | – |
