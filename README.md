# Team 12 · Quantized DiT-XL/2 inference

Our approach accelerates image generation with DiT-XL/2 by running its attention and MLP projections in INT8. At batch 32, it reduced median denoising-step latency from **91.10 to 69.08 ms on A100 (1.32×)** and from **474.46 to 310.04 ms on L4 (1.53×)**. These speedups are relative to the fastest compiled FP16 configuration measured on the same GPU. PTQ4DiT is the research baseline. The repository also includes the authors’ released implementation. [PTQ4DiT code and usage](external/PTQ4DiT-main/README.md)

## How our approach works

1. **Quantize the four projections in each of the 28 transformer blocks.** The attention QKV projection, attention output projection, and two MLP linear layers use symmetric INT8 weights with one scale per output channel. At inference, each input token gets its own INT8 activation scale. This is the W8A8 configuration. The attention calculation itself uses floating-point scaled dot-product attention. [Quantized layers](fusion/src/fc1bench/model_layers.py)
2. **Quantize activations where they are produced.** A Triton kernel combines the gated residual update, layer normalization, adaptive shift and scale, and quantization where these operations meet. A second kernel combines GELU and quantization before the last MLP projection. This avoids writing the full intermediate activations to GPU memory and reading them back just to quantize them. All 28 blocks also share one matrix multiplication for their adaptive layer normalization parameters. [DiT block implementation](fusion/src/fc1bench/fused_ops.py)
3. **Tune the integer matrix multiplications and replay the step.** The projection kernels accumulate in INT32, apply the activation and weight scales and bias, and return FP16 outputs. The tuner selects a configuration for each matrix shape on the target GPU. We compile the forward pass with `torch.compile` and replay it with CUDA graphs. [Matrix multiplication kernels](fusion/src/fc1bench/optimized.py) · [Tuner](fusion/src/fc1bench/tuning.py)

## Measured latency and image quality

Each timing covers one classifier-free guidance step on ImageNet 256×256 with guidance scale 1.5. A batch of 32 images gives 64 model inputs because guidance uses conditional and unconditional predictions. The table reports medians over 50 CUDA-event measurements after 10 warmups. Both methods use scaled dot-product attention, CUDA graphs, and compilation. [A100 measurements](fusion/results/a100/summary.csv) · [L4 measurements](fusion/results/l4/summary.csv)

| Image batch | A100 FP16 | A100 our approach | Speedup | L4 FP16 | L4 our approach | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 12.53 ms | 10.46 ms | 1.20× | 49.37 ms | 29.44 ms | 1.68× |
| 8 | 23.43 ms | 18.47 ms | 1.27× | 115.32 ms | 65.02 ms | 1.77× |
| 16 | 46.97 ms | 35.38 ms | 1.33× | 245.46 ms | 152.75 ms | 1.61× |
| 32 | 91.10 ms | 69.08 ms | 1.32× | 474.46 ms | 310.04 ms | 1.53× |

A separate 16-image check ran the full 50-step sampler using matched noise and compared the decoded W8A8 images with FP32. Median PSNR was **29.16 dB on A100** and **29.50 dB on L4**. The earlier W8A8 and W4A8 experiments include a separate 250-image evaluation with FID, spatial FID, Inception Score, precision, and recall. [A100 paired check](fusion/results/a100/accuracy_fused.json) · [L4 paired check](fusion/results/l4/accuracy_fused.json) · [250-image baseline results](baseline/results/quality_250img/)

## Run our approach

Use a **Bash terminal on a Google Colab A100 or L4 runtime**. To reproduce the comparison across GPUs, run these steps independently on each runtime. The commands below share variables, so use the same shell for each step.

### 1. Install dependencies and download the model

Start from a fresh GPU runtime with CUDA-enabled PyTorch, torchvision, and Triton. The requirements file keeps the Colab-provided versions of those three packages. Setup downloads the pretrained DiT-XL/2 checkpoint and checks out the pinned DiT and ADM sources.

```bash
cd /content
git clone https://github.com/ashdane/team12-ptq4dit.git
cd team12-ptq4dit/fusion
python3 -m pip install -r requirements-colab.txt
nvidia-smi --query-gpu=name,memory.total --format=csv

export PYTHONPATH="$PWD/src"
assets=/content/dit_pretrained_eval/assets
checkpoint="$assets/pretrained_models/DiT-XL-2-256x256.pt"
our_approach=sdpa_graph_pure16_compile_tuned_fused
python3 scripts/setup_pretrained.py --assets "$assets" --image-size 256 --skip-reference
```

### 2. Benchmark our approach against compiled FP16

This command runs batch 32 and tunes the integer projections on the current GPU. The final token in `$our_approach` selects our hand-written DiT block. The benchmark stores the resulting tile settings in `tuning_bank.json`, which the image generator can reuse on this GPU.

```bash
bench=/content/dit_bench_b32
python3 -m fc1bench.latency_bench \
  --dit-repo "$assets/DiT" --checkpoint "$checkpoint" --output "$bench" \
  --modes fp16 int8 --batches 32 --warmup 10 --repeats 50 \
  --scaffolds "$our_approach"
cat "$bench/summary.csv"
```

Read `median_ms` for time per step and `speedup_vs_best_fp16` for the W8A8 speedup. To reproduce the full published sweep, including kernel tests, all four batch sizes, the paired image check, and Nsight Systems profiling, use the driver below on a runtime with `nsys` installed. It writes the latency table to `/content/opt2/sweep5/summary.csv`, the paired check to `/content/opt2/accuracy_fused.json`, and profiles to `/content/opt2/nsys/`. [Benchmark driver](fusion/scripts/final_opt.sh)

```bash
gpu=a100  # use l4 on an L4 runtime
bash scripts/final_opt.sh "$PWD/results/$gpu/csb_scales.pt"
```

### 3. Generate a batch of images

After the batch-32 benchmark, run the pretrained model with its tuning bank. The sampler uses 50 DDPM steps, guidance scale 1.5, and a pinned VAE decoder. The generator saves the first four images as `preview_0.png` through `preview_3.png` and the complete batch in `samples.npz`.

```bash
sample_args=(
  --dit-repo "$assets/DiT" --checkpoint "$checkpoint"
  --image-size 256 --batch-size 32 --steps 50 --cfg 1.5 --seed 1
  --vae-revision f04b2c4b98319346dad8c65879f680b1997b204a
)
python3 -m fc1bench.generate "${sample_args[@]}" \
  --output /content/dit_preview --num-samples 32 --mode int8 \
  --scaffold "$our_approach" --bank "$bench/tuning_bank.json"
ls /content/dit_preview/int8/preview_*.png
```

### 4. Evaluate image quality with 10,000 images per mode

Use a separate output directory per GPU. For a long Colab run, set `quality` to a directory on mounted persistent storage so the image batches and metrics survive VM termination. Generation saves batches incrementally and can resume from the same directory with the same arguments, source, and GPU environment.

```bash
quality=/content/dit_quality_10k  # change to your persistent run directory
common=("${sample_args[@]}" --output "$quality" --num-samples 10000)
python3 -m fc1bench.generate "${common[@]}" --mode int8 \
  --scaffold "$our_approach" --bank "$bench/tuning_bank.json"
python3 -m fc1bench.generate "${common[@]}" --mode fp16 --scaffold sdpa_graph_pure16_compile
python3 -m fc1bench.generate "${common[@]}" --mode fp32 --scaffold sdpa_graph_compile --tf32
```

These commands use the same labels, initial noise, and sampling seeds for each mode. The FP32 generation baseline enables TF32 matrix multiplication. Once all three archives are complete, download the ImageNet reference and run OpenAI’s ADM evaluator in a separate Python 3.11 environment.

```bash
python3 scripts/setup_pretrained.py --assets "$assets" --image-size 256
uv venv --python 3.11 /content/dit-adm-env
adm_python=/content/dit-adm-env/bin/python
uv pip install --python "$adm_python" -r ../baseline/requirements-adm.txt
"$adm_python" -m fc1bench.image_quality \
  --output "$quality" --adm-repo "$assets/guided-diffusion" \
  --reference "$assets/references/VIRTUAL_imagenet256_labeled.npz" \
  --modes fp32 fp16 int8
cat "$quality/quality/comparison.csv"
```

The CSV and JSON reports contain FID, spatial FID, Inception Score, precision, and recall for each mode. They also record sampling throughput and peak allocated GPU memory. The JSON identifies the evaluator revision and reference archive checksum. [Image generator](fusion/src/fc1bench/generate.py) · [Metric calculation](fusion/src/fc1bench/image_quality.py)

## Repository layout

| Directory | Contents |
| --- | --- |
| [fusion/](fusion/) | Our approach, GPU kernels, tests, benchmark driver, tuning banks, and measured A100 and L4 results |
| [baseline/](baseline/README.md) | Earlier W8A8 and W4A8 implementations, calibration experiments, quality comparisons, and profiling |
| [external/PTQ4DiT-main/](external/PTQ4DiT-main/README.md) | The paper authors’ calibration and inference code |
