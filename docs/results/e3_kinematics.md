| Data, training episodes | Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | ΔPSNR ↑ |
|---|---|---|---|---|---|---|
| SutureBot, – | Copy last context frame | 20.66 [20.49, 20.85] | 0.770 [0.767, 0.774] | 0.216 [0.211, 0.220] | 702 | – |
| SutureBot, – | VAE reconstruction (ceiling) | 37.58 [37.46, 37.70] | 0.975 [0.975, 0.976] | 0.009 [0.009, 0.010] | 5 | – |
| SutureBot, 10% | None (from scratch) | 21.96 [21.80, 22.14] | 0.777 [0.773, 0.780] | 0.208 [0.204, 0.212] | 848 | 1.55 |
| SutureBot, 100% | None (from scratch) | 21.23 [21.07, 21.40] | 0.749 [0.745, 0.753] | 0.242 [0.239, 0.246] | 1072 | 1.24 |
