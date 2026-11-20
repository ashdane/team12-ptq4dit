# Pretrained DiT image-quality comparison

256px; 50 steps; CFG 1.5; 250 images/mode.
Exploratory small-sample metrics; not paper-comparable.

| Measured mode | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ |
|---|---:|---:|---:|---:|
| fp32 | 133.0420† | 474.3484† | 101.3942 | 0.8360 |
| fp16 | 133.2728† | 473.8405† | 101.9153 | 0.8160 |
| int8 | 133.4928† | 475.4457† | 99.4484 | 0.7880 |
| int4 | 228.1669† | 594.7252† | 5.6662 | 0.1040 |

† Official ADM sqrtm failed (fewer samples than the 2048-d Inception features make the covariance rank-deficient); value computed with the mathematically equivalent stable eigendecomposition formula. Heavily biased at this sample size: compare modes with each other only.

Published PTQ4DiT reference (different quantization method; not measured here):

| Published method | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ |
|---|---:|---:|---:|---:|
| FP32 | 6.02 | 21.77 | 246.24 | 0.7812 |
| PTQ4DiT W8A8 | 5.45 | 19.50 | 250.68 | 0.7882 |
| PTQ4DiT W4A8 | 9.17 | 24.29 | 179.95 | 0.7052 |

Source: https://arxiv.org/html/2405.16005v2

Quantized modes replace attention/MLP projections according to scope; other operations remain floating point.
Precision is generative precision. Recall is an additional ADM output, not one of the four paper headline metrics.
Deltas in CSV/JSON are measured minus FP32. Lower FID/sFID and higher IS/Precision are better.
Throughput is warmed sampling only; VAE decoding and disk I/O are excluded. Do not infer a statistically stable speedup from one run.
