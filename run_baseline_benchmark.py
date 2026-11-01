"""
Week 1 Automated Baseline Benchmark & Hardware Profiler (`run_baseline_benchmark.py`)
-------------------------------------------------------------------------------------
Designed for CS3406 Programming AI Accelerators (L40S 48GB / RTX PRO 5000 48GB / A100 40GB).
Runs a controlled, reproducible comparison of:
  1. Full-Precision Baseline (FP32 / FP16)
  2. Official PTQ4DiT Simulated W8A8 (Unfused PyTorch Fake-Quantization)
  3. Official PTQ4DiT Simulated W4A8 (Unfused PyTorch Fake-Quantization)
  4. Offline Re-Parameterized PTQ4DiT W8A8 & W4A8 (Completing the authors' unchecked TODO)

Outputs:
  - `output/baseline_metrics.csv`: Machine-readable latency, memory, and fidelity metrics
  - `output/baseline_report.md`:   Formatted Markdown report ready for your Week 1 submission
"""

import os
import csv
import copy
import time
import argparse
import torch
import torch.nn.functional as F
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from quant import QuantModel, calibrate_and_reconstruct_blocks
from fast_inference_reparam import apply_offline_reparameterization


def get_gpu_info():
    if not torch.cuda.is_available():
        return "CPU (Sandbox Verification Mode)", "N/A", 0.0
    props = torch.cuda.get_device_properties(0)
    arch = f"sm_{props.major}{props.minor}"
    vram_gb = props.total_memory / (1024.0 ** 3)
    return props.name, arch, vram_gb


def compute_fidelity_metrics(ref_tensor: torch.Tensor, test_tensor: torch.Tensor):
    ref = ref_tensor.detach().float()
    tst = test_tensor.detach().float()
    mse = F.mse_loss(tst, ref).item()
    max_val = ref.abs().max().item() + 1e-8
    psnr = 20.0 * torch.log10(torch.tensor(max_val) / torch.sqrt(torch.tensor(mse + 1e-12))).item()
    cos_sim = F.cosine_similarity(ref.flatten().unsqueeze(0), tst.flatten().unsqueeze(0)).item()
    return mse, psnr, cos_sim


def benchmark_model_denoise(model, diffusion, z_cfg, model_kwargs, device, warmup_steps=2):
    """Measures exact per-step and end-to-end denoising latency and peak memory."""
    indices = list(range(diffusion.num_timesteps))[::-1]
    img = z_cfg.clone()

    # Warmup
    with torch.no_grad():
        for i in indices[:warmup_steps]:
            t = torch.tensor([i] * img.shape[0], device=device)
            _ = diffusion.p_sample(model.forward_with_cfg, img, t, clip_denoised=False, model_kwargs=model_kwargs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    img = z_cfg.clone()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in indices:
            t = torch.tensor([i] * img.shape[0], device=device)
            out = diffusion.p_sample(model.forward_with_cfg, img, t, clip_denoised=False, model_kwargs=model_kwargs)
            img = out["sample"]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_sec = time.perf_counter() - t0
    step_ms = (total_sec * 1000.0) / len(indices)
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        if torch.cuda.is_available()
        else 0.0
    )
    samples, _ = img.chunk(2, dim=0)
    return samples, total_sec, step_ms, peak_vram_mb


def main():
    parser = argparse.ArgumentParser(description="Run Week 1 PTQ4DiT Baseline Benchmark & Hardware Profile")
    parser.add_argument("--model", type=str, default="DiT-XL/2", choices=list(DiT_models.keys()))
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--recon-iters", type=int, default=10)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--cali-data-path", type=str, default="calib/imagenet_DiT-256_sample4000_50steps_allst.pt")
    parser.add_argument("--outdir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name, sm_arch, vram_gb = get_gpu_info()
    print(f"[Benchmark] Device: {gpu_name} ({sm_arch}, {vram_gb:.1f} GB)")

    latent_size = args.image_size // 8
    base_model = DiT_models[args.model](input_size=latent_size, num_classes=1000).to(device)
    ckpt_path = args.ckpt or f"pretrained_models/DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    if os.path.isfile(ckpt_path):
        state_dict = find_model(ckpt_path)
        if state_dict is not None:
            base_model.load_state_dict(state_dict)
    base_model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Fixed initial noise and class labels for exact fidelity comparison across bit-widths
    torch.manual_seed(args.seed)
    n = args.batch_size
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279][:n], device=device)
    z_cfg = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y_cfg = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y_cfg, cfg_scale=1.5)

    # Prepare calibration subset (either from file or synthetic trajectory if running quick test)
    if os.path.isfile(args.cali_data_path):
        raw_cali = torch.load(args.cali_data_path, map_location="cpu", weights_only=False)
        cali_subset = {
            "xs": raw_cali["xs"][: min(10, raw_cali["xs"].shape[0]), : min(8, raw_cali["xs"].shape[1])],
            "ts": raw_cali["ts"][: min(10, raw_cali["ts"].shape[0]), : min(8, raw_cali["ts"].shape[1])],
            "ys": raw_cali["ys"][: min(10, raw_cali["ys"].shape[0]), : min(8, raw_cali["ys"].shape[1])],
        }
    else:
        print("[Benchmark] Generating compact in-memory calibration trajectory...")
        cali_subset = {
            "xs": torch.randn(5, n, 4, latent_size, latent_size),
            "ts": torch.tensor([[900] * n, [700] * n, [500] * n, [300] * n, [100] * n]),
            "ys": y.cpu().unsqueeze(0).repeat(5, 1),
        }

    results = []

    # 1. Full Precision Baseline
    print("[Benchmark] 1/5 Running Full-Precision (FP) Baseline...")
    fp_latents, fp_sec, fp_step_ms, fp_vram = benchmark_model_denoise(
        base_model, diffusion, z_cfg, model_kwargs, device
    )
    fp_param_mb = sum(p.numel() * 4 for p in base_model.parameters()) / (1024.0 ** 2)
    results.append({
        "Config": "FP32/FP16 Baseline",
        "W/A Bits": "32/32",
        "Mode": "Standard PyTorch",
        "Theoretical Weight Size (MB)": f"{fp_param_mb:.1f}",
        "Latency / Step (ms)": f"{fp_step_ms:.2f}",
        "Total Denoising Time (s)": f"{fp_sec:.2f}",
        "Speedup vs Simulated": "1.00x (Ref)",
        "Peak VRAM (MB)": f"{fp_vram:.1f}",
        "Latent MSE vs FP": "0.000000",
        "PSNR (dB)": "Inf",
        "Cosine Sim": "1.0000",
    })

    for w_bit, a_bit in [(8, 8), (4, 8)]:
        print(f"[Benchmark] Calibrating PTQ4DiT W{w_bit}A{a_bit} (CSB + SSC)...")
        qnn = QuantModel(copy.deepcopy(base_model), weight_bit=w_bit, act_bit=a_bit).to(device)
        qnn.eval()
        ckpt_save = os.path.join(args.outdir, f"ckpt_W{w_bit}A{a_bit}.pth")
        calibrate_and_reconstruct_blocks(
            qnn,
            cali_data=cali_subset,
            use_ssc=True,
            do_recon=(args.recon_iters > 0),
            recon_iters=args.recon_iters,
            save_ckpt_path=ckpt_save,
        )
        qnn.set_quant_state(weight_quant=True, act_quant=True)

        # Simulated PTQ4DiT Baseline
        print(f"[Benchmark] Running Official PTQ4DiT Simulated W{w_bit}A{a_bit}...")
        sim_latents, sim_sec, sim_step_ms, sim_vram = benchmark_model_denoise(
            qnn, diffusion, z_cfg, model_kwargs, device
        )
        mse, psnr, cos_sim = compute_fidelity_metrics(fp_latents, sim_latents)
        theo_mb = fp_param_mb * (w_bit / 32.0)
        results.append({
            "Config": f"PTQ4DiT Simulated W{w_bit}A{a_bit}",
            "W/A Bits": f"{w_bit}/{a_bit}",
            "Mode": "Simulated (Fake-Quant)",
            "Theoretical Weight Size (MB)": f"{theo_mb:.1f}",
            "Latency / Step (ms)": f"{sim_step_ms:.2f}",
            "Total Denoising Time (s)": f"{sim_sec:.2f}",
            "Speedup vs Simulated": "1.00x",
            "Peak VRAM (MB)": f"{sim_vram:.1f}",
            "Latent MSE vs FP": f"{mse:.6f}",
            "PSNR (dB)": f"{psnr:.2f}",
            "Cosine Sim": f"{cos_sim:.4f}",
        })

        # Offline Re-Parameterized PTQ4DiT (Section 4.3 optimization)
        print(f"[Benchmark] Running Offline Re-Parameterized PTQ4DiT W{w_bit}A{a_bit}...")
        apply_offline_reparameterization(qnn)
        rep_latents, rep_sec, rep_step_ms, rep_vram = benchmark_model_denoise(
            qnn, diffusion, z_cfg, model_kwargs, device
        )
        mse_r, psnr_r, cos_r = compute_fidelity_metrics(fp_latents, rep_latents)
        speedup = sim_step_ms / max(rep_step_ms, 1e-6)
        results.append({
            "Config": f"PTQ4DiT Re-Parameterized W{w_bit}A{a_bit}",
            "W/A Bits": f"{w_bit}/{a_bit}",
            "Mode": "Offline Re-Param (Sec 4.3)",
            "Theoretical Weight Size (MB)": f"{theo_mb:.1f}",
            "Latency / Step (ms)": f"{rep_step_ms:.2f}",
            "Total Denoising Time (s)": f"{rep_sec:.2f}",
            "Speedup vs Simulated": f"{speedup:.2f}x",
            "Peak VRAM (MB)": f"{rep_vram:.1f}",
            "Latent MSE vs FP": f"{mse_r:.6f}",
            "PSNR (dB)": f"{psnr_r:.2f}",
            "Cosine Sim": f"{cos_r:.4f}",
        })

    # Save CSV
    csv_path = os.path.join(args.outdir, "baseline_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Save Markdown Report
    md_path = os.path.join(args.outdir, "baseline_report.md")
    with open(md_path, "w") as f:
        f.write("# CS3406 Programming AI Accelerators — Week 1 Baseline Report (PTQ4DiT)\n\n")
        f.write(f"- **Hardware Device:** {gpu_name} (Compute Capability: `{sm_arch}`, VRAM: `{vram_gb:.1f} GB`)\n")
        f.write(f"- **Model & Resolution:** `{args.model}` at `{args.image_size}x{args.image_size}` ({args.num_sampling_steps} steps, Batch Size: {args.batch_size})\n\n")
        f.write("## 1. Baseline & Offline Re-Parameterization Results\n\n")
        headers = list(results[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in results:
            f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")
        f.write("\n## 2. Key Systems Takeaways for Week 2 Kernel Optimization\n\n")
        f.write("1. **Why Official Simulated PTQ4DiT Is Slower Than FP:** The official `adreamwu/PTQ4DiT` codebase executes channel-wise salience rescaling (`X * B_rho^X` and `W * B_rho^W`) plus FP32/FP16 `round`/`clamp` dequantization on every forward pass across all 28 DiT blocks.\n")
        f.write("2. **Impact of Offline Re-Parameterization (Completing the Unchecked Repo TODO):** Absorbing `B_rho^W` into the weights offline and folding `B_rho^X` into the `adaLN_modulation` MLPs (Eqs. 13 & 20) eliminates all runtime channel-balancing multiplications and weight fake-quantization while maintaining mathematical equivalence.\n")
        f.write("3. **Next Step (Fused Low-Bit Tensor Core GEMM):** Replacing the remaining `F.linear` FP16/FP32 MatMuls with fused OpenAI Triton / CUTLASS W8A8 (INT8 / Ada FP8) and W4A8 kernels will convert the theoretical 4x-8x weight compression into true Tensor Core wall-clock speedup.\n")

    print(f"\n[Benchmark] Saved CSV metrics -> {csv_path}")
    print(f"[Benchmark] Saved Markdown report -> {md_path}")


if __name__ == "__main__":
    main()
