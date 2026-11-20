import sys
import os
for p in ['src', '../src', 'sim/baseline_experiments', '../sim/baseline_experiments']:
    abs_p = os.path.abspath(p)
    if os.path.exists(abs_p) and abs_p not in sys.path:
        sys.path.insert(0, abs_p)
import torch
import torch.nn.functional as F
import time
import json
import math
from pathlib import Path
from fc1bench.kernels import integer_linear, quantize_activation
from fc1bench.quantization import quantize_rows, pack_int4, metrics
from fc1bench.report import summarize

OPERATORS = [
    ("attn.qkv", 2048, 1152, 3456),
    ("attn.proj", 2048, 1152, 1152),
    ("mlp.fc1", 2048, 1152, 4608),
    ("mlp.fc2", 2048, 4608, 1152),
]

def bench_fn(fn, warmup=25, repeats=50, inner=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / inner)
    return summarize(times)

def run():
    print("=" * 80)
    print("PTQ4DiT TRITON KERNEL BENCHMARK SUITE — NVIDIA RTX 2080 Ti (sm_75)")
    print("=" * 80)
    
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Device: {device_name} (sm_{capability[0]}{capability[1]})")
    
    all_results = []
    
    for op_name, m, k, n in OPERATORS:
        print(f"\n>>> Benchmarking Operator: {op_name} (M={m}, K={k}, N={n})")
        torch.manual_seed(2026)
        x_fp16 = torch.randn(m, k, device="cuda", dtype=torch.float16)
        w_fp16 = (torch.randn(n, k, device="cuda", dtype=torch.float16) / math.sqrt(k)).contiguous()
        b_fp16 = (torch.randn(n, device="cuda", dtype=torch.float16) * 0.01).contiguous()
        b_fp32 = b_fp16.float()
        
        # FP16 Reference
        fp16_ref = F.linear(x_fp16, w_fp16, b_fp16)
        torch.cuda.synchronize()
        
        # 1. FP16 Baseline Timing
        fp16_fn = lambda: F.linear(x_fp16, w_fp16, b_fp16)
        fp16_stats = bench_fn(fp16_fn)
        fp16_ms = fp16_stats["median_ms"]
        print(f"  FP16 Baseline Latency:    {fp16_ms:.4f} ms (p90: {fp16_stats['p90_ms']:.4f} ms)")
        
        # Prepare W8A8
        wq8, sw8 = quantize_rows(w_fp16.float(), 8)
        xq8, sx8 = quantize_activation(x_fp16)
        
        # 2. Activation Quantization Timing
        quant_fn = lambda: quantize_activation(x_fp16)
        quant_stats = bench_fn(quant_fn)
        quant_ms = quant_stats["median_ms"]
        print(f"  Act Quantization Latency: {quant_ms:.4f} ms (p90: {quant_stats['p90_ms']:.4f} ms)")
        
        # 3. W8A8 GEMM-only Timing
        w8_gemm_fn = lambda: integer_linear(xq8, wq8, sx8, sw8, b_fp32, packed=False)
        w8_gemm_stats = bench_fn(w8_gemm_fn)
        w8_gemm_ms = w8_gemm_stats["median_ms"]
        print(f"  W8A8 GEMM-only Latency:   {w8_gemm_ms:.4f} ms (p90: {w8_gemm_stats['p90_ms']:.4f} ms)")
        
        # 4. W8A8 Combined Timing (Quant + GEMM)
        def w8_combined_fn():
            q, s = quantize_activation(x_fp16)
            return integer_linear(q, wq8, s, sw8, b_fp32, packed=False)
        w8_comb_stats = bench_fn(w8_combined_fn)
        w8_comb_ms = w8_comb_stats["median_ms"]
        print(f"  W8A8 Combined Latency:    {w8_comb_ms:.4f} ms (p90: {w8_comb_stats['p90_ms']:.4f} ms)")
        
        # Validate W8 output correctness vs FP16
        w8_out = w8_combined_fn()
        w8_metrics = metrics(fp16_ref.float(), w8_out.float())
        print(f"  W8A8 vs FP16: MSE={w8_metrics['mse']:.4e}, RelL2={w8_metrics['relative_l2']:.4e}, CosSim={w8_metrics['cosine_similarity']:.6f}")
        
        # Prepare W4A8
        wq4, sw4 = quantize_rows(w_fp16.float(), 4)
        w4_packed = pack_int4(wq4)
        
        # 5. W4A8 GEMM-only Timing
        w4_gemm_fn = lambda: integer_linear(xq8, w4_packed, sx8, sw4, b_fp32, packed=True)
        w4_gemm_stats = bench_fn(w4_gemm_fn)
        w4_gemm_ms = w4_gemm_stats["median_ms"]
        print(f"  W4A8 GEMM-only Latency:   {w4_gemm_ms:.4f} ms (p90: {w4_gemm_stats['p90_ms']:.4f} ms)")
        
        # 6. W4A8 Combined Timing (Quant + GEMM)
        def w4_combined_fn():
            q, s = quantize_activation(x_fp16)
            return integer_linear(q, w4_packed, s, sw4, b_fp32, packed=True)
        w4_comb_stats = bench_fn(w4_combined_fn)
        w4_comb_ms = w4_comb_stats["median_ms"]
        print(f"  W4A8 Combined Latency:    {w4_comb_ms:.4f} ms (p90: {w4_comb_stats['p90_ms']:.4f} ms)")
        
        # Validate W4 output correctness vs FP16
        w4_out = w4_combined_fn()
        w4_metrics = metrics(fp16_ref.float(), w4_out.float())
        print(f"  W4A8 vs FP16: MSE={w4_metrics['mse']:.4e}, RelL2={w4_metrics['relative_l2']:.4e}, CosSim={w4_metrics['cosine_similarity']:.6f}")
        
        # Speedups
        w8_gemm_speedup = fp16_ms / w8_gemm_ms
        w8_comb_speedup = fp16_ms / w8_comb_ms
        w4_gemm_speedup = fp16_ms / w4_gemm_ms
        w4_comb_speedup = fp16_ms / w4_comb_ms
        
        print(f"  Speedup W8A8: GEMM-only = {w8_gemm_speedup:.2f}x, Combined = {w8_comb_speedup:.2f}x")
        print(f"  Speedup W4A8: GEMM-only = {w4_gemm_speedup:.2f}x, Combined = {w4_comb_speedup:.2f}x")
        
        # Weight memory
        w_fp16_bytes = w_fp16.numel() * 2
        w_int8_bytes = wq8.numel() * 1 + sw8.numel() * 4
        w_int4_bytes = w4_packed.numel() * 1 + sw4.numel() * 4
        
        result_item = {
            "operator": op_name,
            "m": m, "k": k, "n": n,
            "fp16_latency_ms": fp16_ms,
            "quant_latency_ms": quant_ms,
            "w8a8_gemm_ms": w8_gemm_ms,
            "w8a8_combined_ms": w8_comb_ms,
            "w8a8_gemm_speedup": w8_gemm_speedup,
            "w8a8_combined_speedup": w8_comb_speedup,
            "w4a8_gemm_ms": w4_gemm_ms,
            "w4a8_combined_ms": w4_comb_ms,
            "w4a8_gemm_speedup": w4_gemm_speedup,
            "w4a8_combined_speedup": w4_comb_speedup,
            "w8a8_metrics": w8_metrics,
            "w4a8_metrics": w4_metrics,
            "w_fp16_bytes": w_fp16_bytes,
            "w_int8_bytes": w_int8_bytes,
            "w_int4_bytes": w_int4_bytes,
            "w_int8_ratio": w_fp16_bytes / w_int8_bytes,
            "w_int4_ratio": w_fp16_bytes / w_int4_bytes,
            "fp16_stats": fp16_stats,
            "quant_stats": quant_stats,
            "w8_gemm_stats": w8_gemm_stats,
            "w8_comb_stats": w8_comb_stats,
            "w4_gemm_stats": w4_gemm_stats,
            "w4_comb_stats": w4_comb_stats,
        }
        all_results.append(result_item)
        
    out_path = Path("benchmark_results_rtx2080ti.json")
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved benchmark results to {out_path.resolve()}")

if __name__ == "__main__":
    run()

