# Pretrained DiT image-quality comparison

256px; 50 steps; CFG 1.5; 10000 images/mode.
Paper sample count met or exceeded; method/precision/protocol differences still apply.

| Measured mode | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ |
|---|---:|---:|---:|---:|
| fp32 | 6.4972 | 21.0189 | 242.0947 | 0.7837 |
| fp16 | 6.4975 | 21.0172 | 241.3866 | 0.7843 |
| int8 | 6.7417 | 21.4451 | 235.7480 | 0.7783 |
| int8_v2 | 6.7564 | 21.4563 | 237.0449 | 0.7789 |
| int8_v3 | 6.2297 | 20.3203 | 248.8919 | 0.7907 |
| int8_v4 | 6.2518 | 20.3103 | 248.9574 | 0.7893 |
| int8_v4w | 6.3335 | 20.6213 | 245.3412 | 0.7886 |
| int8_v4c | 6.7497 | 21.4852 | 236.6469 | 0.7785 |

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
