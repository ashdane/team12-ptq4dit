#!/usr/bin/env python3
"""Analytical roofline for the four DiT projection shapes from a latency_bench run.

Uses tuning_summary.json (per-layer holdout timings: INT quantize+GEMM layer vs FP16 F.linear).
Peaks are dense datasheet values (no sparsity) and must be checked against NVIDIA datasheets;
power-capped GPUs (e.g. L4, 72 W) usually cannot sustain them. Byte counts assume each tensor
moves through DRAM exactly once (ideal caching), so "floor" is optimistic.

Usage: python3 scripts/roofline.py <result_dir> <a100|l4>
"""
import json
import sys
from pathlib import Path

PEAK = {"a100": dict(fp16=312e12, int8=624e12, bw=1555e9),   # A100-SXM4-40GB
        "l4": dict(fp16=121e12, int8=242e12, bw=300e9)}      # L4
NAMES = {(3456, 1152): "qkv", (1152, 1152): "proj", (4608, 1152): "fc1", (1152, 4608): "fc2"}


def main():
    root, gpu = Path(sys.argv[1]), sys.argv[2].lower()
    peak = PEAK[gpu]
    tuning = json.loads((root / "tuning_summary.json").read_text())
    print(f"{gpu.upper()} ridge point: fp16 {peak['fp16']/peak['bw']:.0f} FLOP/B, int8 {peak['int8']/peak['bw']:.0f} OP/B")
    print(f"{'layer':5} {'M':>6} {'bits':>4} | {'fp16 AI':>7} {'TFLOPS':>7} {'%peak':>6} | "
          f"{'int AI':>6} {'TOPS':>6} {'%peak':>6} {'bound':>7} | {'floor ms':>8} {'meas ms':>8} {'gap':>5}")
    for key, v in sorted(tuning.items(), key=lambda kv: (int(kv[0].split("_")[0]), kv[0])):
        m, n, k = (int(x) for x in key.split("_")[:3])
        bits = int(key.split("_w")[1][0])
        flops = 2 * m * n * k
        bytes16 = 2 * (m * k + n * k + m * n)                       # FP16 A, W, out
        # INT layer as implemented: quantize pass (reads FP16 x twice, writes int8)
        # + GEMM (int8 A, packed/int8 W, FP16 out).
        bytes_int = (2 * 2 * m * k + m * k) + (m * k + n * k * bits / 8 + 2 * m * n)
        t16 = v["holdout_float_linear_median_ms"] / 1e3
        tint = v["holdout_layer_median_ms"] / 1e3
        ai_int = flops / bytes_int
        floor = max(flops / peak["int8"], bytes_int / peak["bw"])
        bound = "memory" if ai_int < peak["int8"] / peak["bw"] else "compute"
        print(f"{NAMES.get((n, k), '?'):5} {m:>6} {bits:>4} | {flops/bytes16:7.0f} {flops/t16/1e12:7.1f} "
              f"{100*flops/t16/peak['fp16']:5.0f}% | {ai_int:6.0f} {flops/tint/1e12:6.1f} "
              f"{100*flops/tint/peak['int8']:5.0f}% {bound:>7} | {floor*1e3:8.3f} {tint*1e3:8.3f} {tint/floor:4.1f}x")


if __name__ == "__main__":
    main()
