### Dice and gap closed by artefact strength and number of unlabeled target images

| artefact strength | method | unlabeled target images | Dice | gap closed | RF-FD x1000 | adaptation s |
|---|---|---|---|---|---|---|
| 0 | oracle | - | 0.748 ± 0.003 | - | 0.00 ± 0.00 | 0 |
| 0 | physics-fitted | 8 | 0.664 ± 0.034 | 0.70 ± 0.13 | 0.07 ± 0.01 | 64 |
| 0 | physics-fitted | 32 | 0.686 ± 0.033 | 0.78 ± 0.13 | 0.02 ± 0.00 | 65 |
| 0 | physics-fitted | 128 | 0.559 ± 0.011 | 0.33 ± 0.02 | 0.08 ± 0.05 | 71 |
| 0 | GAN translator | 8 | 0.443 ± 0.006 | -0.08 ± 0.06 | 0.15 ± 0.05 | 26 |
| 0 | GAN translator | 32 | 0.446 ± 0.003 | -0.07 ± 0.05 | 0.13 ± 0.04 | 27 |
| 0 | GAN translator | 128 | 0.457 ± 0.006 | -0.03 ± 0.04 | 0.13 ± 0.04 | 27 |
| 0 | augment | - | 0.370 ± 0.010 | -0.34 ± 0.09 | 0.56 ± 0.01 | 0 |
| 0 | source-only | - | 0.465 ± 0.016 | - | 0.48 ± 0.00 | 0 |
| 0.5 | oracle | - | 0.639 ± 0.005 | - | 0.00 ± 0.00 | 0 |
| 0.5 | physics-fitted | 8 | 0.409 ± 0.065 | -0.20 ± 0.41 | 0.28 ± 0.03 | 64 |
| 0.5 | physics-fitted | 32 | 0.432 ± 0.051 | -0.08 ± 0.33 | 0.15 ± 0.14 | 66 |
| 0.5 | physics-fitted | 128 | 0.372 ± 0.036 | -0.38 ± 0.22 | 0.25 ± 0.01 | 56 |
| 0.5 | GAN translator | 8 | 0.432 ± 0.007 | -0.07 ± 0.06 | 0.16 ± 0.07 | 28 |
| 0.5 | GAN translator | 32 | 0.440 ± 0.007 | -0.02 ± 0.05 | 0.14 ± 0.02 | 23 |
| 0.5 | GAN translator | 128 | 0.438 ± 0.006 | -0.04 ± 0.04 | 0.14 ± 0.04 | 23 |
| 0.5 | augment | - | 0.361 ± 0.010 | -0.43 ± 0.08 | 0.55 ± 0.01 | 0 |
| 0.5 | source-only | - | 0.444 ± 0.012 | - | 0.50 ± 0.00 | 0 |
| 1 | oracle | - | 0.587 ± 0.013 | - | 0.00 ± 0.00 | 0 |
| 1 | physics-fitted | 8 | 0.367 ± 0.009 | -0.28 ± 0.16 | 0.34 ± 0.02 | 59 |
| 1 | physics-fitted | 32 | 0.405 ± 0.038 | -0.05 ± 0.25 | 0.31 ± 0.08 | 55 |
| 1 | physics-fitted | 128 | 0.435 ± 0.014 | 0.12 ± 0.11 | 0.17 ± 0.02 | 57 |
| 1 | GAN translator | 8 | 0.420 ± 0.010 | 0.04 ± 0.05 | 0.20 ± 0.03 | 32 |
| 1 | GAN translator | 32 | 0.423 ± 0.005 | 0.05 ± 0.07 | 0.18 ± 0.01 | 23 |
| 1 | GAN translator | 128 | 0.420 ± 0.006 | 0.03 ± 0.08 | 0.18 ± 0.03 | 24 |
| 1 | augment | - | 0.347 ± 0.012 | -0.39 ± 0.08 | 0.56 ± 0.01 | 0 |
| 1 | source-only | - | 0.413 ± 0.015 | - | 0.56 ± 0.00 | 0 |

### Ablation and GAN budget check (32 unlabeled images, artefact strength 0.5)

| method | variant | Dice | gap closed | RF-FD x1000 | adaptation s |
|---|---|---|---|---|---|
| GAN translator | long | 0.395 ± 0.008 | -0.26 ± 0.04 | 0.09 ± 0.02 | 100 |
| GAN translator | short | 0.440 ± 0.007 | -0.02 ± 0.05 | 0.14 ± 0.02 | 23 |
| augment | - | 0.361 ± 0.010 | -0.43 ± 0.08 | 0.55 ± 0.01 | 0 |
| oracle | - | 0.639 ± 0.005 | - | 0.00 ± 0.00 | 0 |
| physics-fitted | full | 0.432 ± 0.051 | -0.08 ± 0.33 | 0.15 ± 0.14 | 66 |
| physics-fitted | without charge jumps | 0.314 ± 0.043 | -0.66 ± 0.11 | 0.35 ± 0.01 | 32 |
| physics-fitted | without physics shift | 0.439 ± 0.016 | -0.03 ± 0.13 | 0.32 ± 0.04 | 31 |
| physics-fitted | without sensor peak | 0.358 ± 0.030 | -0.45 ± 0.14 | 0.24 ± 0.04 | 26 |
| physics-fitted | without thermal broadening | 0.361 ± 0.022 | -0.44 ± 0.20 | 0.27 ± 0.02 | 32 |
| source-only | - | 0.444 ± 0.012 | - | 0.50 ± 0.00 | 0 |
