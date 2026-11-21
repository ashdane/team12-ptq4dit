# Our approach to W8A8 inference for DiT-XL/2

Sweep 5 measured one classifier-free guidance step of DiT-XL/2 at 256×256 on an A100 and an L4. At batch 32, our approach reduced median step latency from 91.10 to 69.08 ms on A100 and from 474.46 to 310.04 ms on L4. The FP16 comparison used scaled dot-product attention, CUDA graphs, and `torch.compile`.

The attention and MLP projections use per-channel INT8 weights and dynamic per-token INT8 activations. The fused Triton kernels combine activation quantization with its producers, including modulated layer normalization and GELU. Version 2 tunes multi-row tiles by shape, uses reciprocal multiplication for quantization and an approximate tanh for GELU, and batches the adaLN projections across blocks. Sweep 5 also includes updated GEMM tiles and fresh tuning on each GPU.

## Measurements

Each timing is the median of 50 CUDA-event samples after 10 warmups. Batch refers to the number of images. Classifier-free guidance doubles the model input batch. Speedups use the fastest measured FP16 configuration at the same batch on the same GPU.

| Batch | A100 W8A8 (ms) | A100 speedup | L4 W8A8 (ms) | L4 speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 10.46 | 1.20× | 29.44 | 1.68× |
| 8 | 18.47 | 1.27× | 65.02 | 1.77× |
| 16 | 35.38 | 1.33× | 152.75 | 1.61× |
| 32 | 69.08 | 1.32× | 310.04 | 1.53× |

A separate paired 16-image sampling run compared decoded W8A8 outputs with FP32 using the same noise. Median PSNR was 29.16 dB on A100 and 29.50 dB on L4. Each GPU folder contains its latency summary, run manifest, tuning bank, accuracy measurements, and Nsight Systems summary. [A100 results](results/a100/) · [L4 results](results/l4/)

## Reproduce sweep 5

From the `fusion/` directory on a Colab GPU runtime with Nsight Systems installed, run the following commands. Set `gpu=l4` on the L4 runtime. The driver writes the results under `/content/opt2/`.

```bash
gpu=a100
python3 -m pip install -r requirements-colab.txt
python3 scripts/setup_pretrained.py --assets /content/dit_pretrained_eval/assets --image-size 256 --skip-reference
bash scripts/final_opt.sh "$PWD/results/$gpu/csb_scales.pt"
```
