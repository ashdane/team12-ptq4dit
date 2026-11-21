# Team 12 · Quantized DiT-XL/2 inference

This project uses Triton kernels to run the attention and MLP projections of DiT-XL/2 with 8-bit activations and either 8-bit or 4-bit weights (W8A8 and W4A8). PTQ4DiT is the research baseline. For latency, we compare our kernels with the fastest measured FP16 configuration on each GPU.

## Fused W8A8 latency

Median time for one denoising step at batch 32 on ImageNet 256×256, measured over 50 repetitions with classifier-free guidance set to 1.5. The FP16 baseline uses scaled dot-product attention, CUDA graphs, and `torch.compile`.

| GPU | Compiled FP16 | Fusion v2 W8A8 | Speedup |
| --- | ---: | ---: | ---: |
| A100 | 91.10 ms | 69.08 ms | 1.32× |
| L4 | 474.46 ms | 310.04 ms | 1.53× |

## Code and results

- **Fusion v2** contains the fused kernels, per-GPU measurements, and commands to reproduce the benchmark on Colab. [Fusion guide](fusion/README.md)
- **Baseline experiments** cover W8A8 and W4A8 kernels, profiling, and image-quality comparisons using 250 images per mode. [Baseline guide](baseline/README.md)
- **PTQ4DiT** contains the paper authors’ implementation and instructions for calibration and quantized sampling. [Upstream guide](external/PTQ4DiT-main/README.md)
