# Pretrained DiT image-quality comparison

256px; 50 steps; CFG 1.5; 5000 images/mode.
Exploratory small-sample metrics; not paper-comparable.

| Measured mode | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ |
|---|---:|---:|---:|---:|
| fp32 | 10.5926 | 38.2863 | 234.2197 | 0.7826 |
| fp16 | 10.6242 | 38.2590 | 232.9182 | 0.7854 |
| int8 | 10.8773 | 38.8205 | 226.4867 | 0.7726 |
| int8_v2 | 10.9124 | 38.8575 | 226.0248 | 0.7736 |
| int8_v3 | 10.2891 | 37.5351 | 239.9875 | 0.7946 |
| int8_v4 | 10.2386 | 37.5470 | 241.7512 | 0.7872 |
| int8_v4w | 10.3791 | 37.8698 | 234.4083 | 0.7786 |
| int8_v4c | 10.9098 | 38.8381 | 225.5954 | 0.7748 |

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
