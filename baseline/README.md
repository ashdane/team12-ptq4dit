# Baseline: W8A8 / W4A8 DiT-XL/2 with real INT kernels

Pretrained DiT-XL/2 (ImageNet 256×256, 50 DDPM steps, CFG 1.5) running W8A8 and W4A8 on custom
Triton INT8 / INT4 kernels, with image-quality evaluation, latency benchmarking, PTQ4DiT CSB + SSC,
and Nsight Compute profiling on A100 and L4.

## Implementation

- **Kernels** (`src/fc1bench/kernels.py`, `optimized.py`): dynamic per-token INT8 activation
  quantizer; tiled INT8 and packed-INT4 GEMMs with INT32 accumulation and a fused
  scale/bias epilogue; cuBLASLt (`torch._int_mm`) comparator.
- **Quantized model** (`model_layers.py`): `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` in all 28
  blocks replaced by INT layers; per-output-channel absmax weights (W8 / W4).
- **Tile autotuning** (`tuning.py`): correctness-gated, per GPU and per layer shape.
- **Generation and evaluation** (`generate.py`, `image_quality.py`): matched-seed sampling per mode,
  ADM TensorFlow evaluator (FID, sFID, IS, Precision), stable eigen-decomposition FID below 2,048 images.
- **Latency benchmark** (`latency_bench.py`): CUDA-event timing of one denoising step, median of 50,
  shared scaffolds for all modes: SDPA, CUDA graphs, pure FP16, `torch.compile` (INT layers as a
  custom op, 0 graph breaks), tuned tiles; FP32 accuracy reference per configuration.
- **CSB + SSC** (`csb.py`): PTQ4DiT formulas and calibration recipe (25 timesteps × 4 samples,
  Spearman-weighted salience, s = √(act/weight) on qkv/proj/fc1), folded offline into the adaLN and
  V-projection weights.
- **Profiling**: `torch.profiler` breakdowns, Nsight Compute targets (`ncu_targets.py`,
  `scripts/ncu_profile.sh`, `scripts/ncu_summary.py`), analytical roofline (`scripts/roofline.py`).

## Layout

```
src/fc1bench/   library
scripts/        setup, latency, ncu, roofline, notebook builder
tests/          unit tests (CPU references, GPU kernels)
validation/     RTX 2080 Ti kernel validation
colab/          self-contained Colab notebook, colab.md (CLI runbook), finish_eval.py
docs/           kernel design notes
results/        quality, latency, CSB and ncu results (A100, L4)
```

## Running

Commands for Colab GPU VMs are in [`colab/colab.md`](colab/colab.md).

```bash
# image quality: run colab/Team12_Triton_Kernels_Colab.ipynb (preview = 250 images/mode)

# latency (on the VM)
bash scripts/latency_on_colab.sh /content/latency/run --batches 4 8 16 32 --profile

# CSB + SSC
PYTHONPATH=src python3 -m fc1bench.csb calibrate --dit-repo DiT --checkpoint DiT-XL-2-256x256.pt --output csb_scales.pt
PYTHONPATH=src python3 -m fc1bench.csb evaluate  --dit-repo DiT --checkpoint DiT-XL-2-256x256.pt --scales csb_scales.pt --output eval.json --decode

# Nsight Compute
bash scripts/ncu_profile.sh <out_dir> <tuning_bank.json> all && python3 scripts/ncu_summary.py <out_dir>
```

## Results

### Image quality, 250 images/mode (`results/quality_250img/`)
† FID/sFID via the stable eigen-decomposition (fewer than 2,048 images need more for better result).

| GPU | Mode | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ |
|---|---|---:|---:|---:|---:|
| A100 | fp32 | 133.04† | 474.35† | 101.39 | 0.836 |
| A100 | fp16 | 133.27† | 473.84† | 101.92 | 0.816 |
| A100 | W8A8 | 133.49† | 475.45† | 99.45 | 0.788 |
| A100 | W4A8 | 228.17† | 594.73† | 5.67 | 0.104 |
| L4 | fp32 | 133.04† | 474.35† | 101.39 | 0.836 |
| L4 | fp16 | 133.20† | 473.99† | 101.10 | 0.820 |
| L4 | W8A8 | 133.73† | 475.20† | 98.26 | 0.828 |
| L4 | W4A8 | 228.01† | 594.88† | 5.53 | 0.112 |

### Latency per denoising step (`results/latency/`)
One `forward_with_cfg` (CFG batch 2×B), median of 50; speedup vs the fastest FP16 configuration
(SDPA + CUDA graphs + `torch.compile`) on the same GPU.

| GPU | Batch | FP16 (ms) | W8A8 (ms) | W8A8 speedup | W4A8 speedup |
|---|---:|---:|---:|---:|---:|
| A100 | 4 | 12.73 | 13.39 | 0.95× | 0.71× |
| A100 | 8 | 24.08 | 23.66 | 1.02× | 0.72× |
| A100 | 16 | 47.50 | 44.56 | 1.07× | 0.77× |
| A100 | 32 | 92.79 | 85.37 | 1.09× | 0.78× |
| L4 | 4 | 43.90 | 31.76 | 1.38× | 1.22× |
| L4 | 8 | 99.11 | 68.82 | 1.44× | 1.33× |
| L4 | 16 | 211.62 | 156.43 | 1.35× | 1.25× |
| L4 | 32 | 452.38 | 359.57 | 1.26× | 1.15× |

W8A8 output relative L2 vs FP32 per step: 0.0137 (FP16: 0.0012–0.0016). `sweep1/` is the scaffold
ablation (explicit attention → SDPA → CUDA graphs → FP16 I/O → tuned tiles); `sweep2/` adds pure
FP16 and `torch.compile`.

### CSB + SSC (`results/csb/`)
Folded model matches the unfolded FP32 output to ~3e-7; scales from A100 and L4 agree to 1.3e-4.
Paired evaluation, 16 images, full sampling, identical noise, vs FP32:

| GPU | Mode | latent rel-L2 (no CSB → CSB) | decoded PSNR (no CSB → CSB) |
|---|---|---|---|
| A100 | W8A8 | 0.126 → 0.136 | 29.0 → 29.0 dB |
| A100 | W4A8 | 0.854 → 0.881 | 13.3 → 13.2 dB |
| L4 | W8A8 | 0.132 → 0.132 | 28.7 → 28.8 dB |
| L4 | W4A8 | 0.854 → 0.882 | 13.3 → 13.2 dB |

### Nsight Compute, batch 32 (`results/ncu/`)

| | A100 FP16 (cuBLAS) | A100 INT8 (ours) | L4 FP16 (cuBLAS) | L4 INT8 (ours) |
|---|---:|---:|---:|---:|
| fc1 GEMM (ms, base clock) | 0.94 | 0.82 | 4.03 | 2.27 |
| Tensor pipe busy | 80–93% | 42–48% | 90–95% | 77–86% |
| Top stall | math pipe | `long_scoreboard` | math pipe | math pipe |

Activation quantization is ~15% of a W8A8 step on both GPUs (DRAM-bound at 90% on L4).
W4A8 GEMM tensor pipe: 28% (A100), 54% (L4).
