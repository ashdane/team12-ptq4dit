#!/usr/bin/env python3
"""Summarise nsys graphed-step profiles from scripts/profile_fused.sh.

Usage: python3 scripts/nsys_summary.py <prof_dir>/nsys
Per target: ms per step (CUDA events), GPU-busy ms per step (sum of kernel durations), idle gap per
step (span of kernels minus busy), and kernel time per group / top kernels, all at real clocks.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def group(name):
    low = name.lower()
    if "_res_ln_mod_quant" in low or "_act_quant" in low:
        return "fused_producer_quant"
    if "_quantize_fused" in low:
        return "int_activation_quant"
    if low.startswith("triton_") and ("amax" in low or "round" in low):
        return "inductor_fused_quant"
    if low.startswith("triton_"):
        return "inductor_elementwise_norm"
    if ("_tiled" in low or "_epilogue" in low or "i16832" in low or "i8816" in low or "_s8" in low
            or "imma" in low or "int8" in low):
        return "int_gemm"
    if "flash" in low or "fmha" in low or "attention" in low:
        return "attention"
    if "gemm" in low or "cutlass" in low or "xmma" in low or "ampere" in low or "sm80" in low or "sm90" in low:
        return "fp16_gemm"
    return "other"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    root = Path(sys.argv[1])
    rows = []
    for log in sorted(root.glob("step_*.log")):
        tag = log.stem
        meta = next((json.loads(l) for l in log.read_text(errors="replace").splitlines()
                     if l.startswith("{") and "ms_per_step" in l), None)
        trace = root / f"{tag}_cuda_gpu_trace.csv"
        if not meta or not trace.exists():
            print(f"{tag}: missing results")
            continue
        n = meta["replays"]
        kern = [r for r in read_csv(trace) if r.get("Name") and not r["Name"].startswith("[CUDA")]
        starts = [int(r["Start (ns)"]) for r in kern]
        ends = [int(r["Start (ns)"]) + int(r["Duration (ns)"]) for r in kern]
        busy = sum(int(r["Duration (ns)"]) for r in kern) / 1e6 / n
        span = (max(ends) - min(starts)) / 1e6 / n if kern else 0
        groups, names = defaultdict(float), defaultdict(float)
        for r in kern:
            ms = int(r["Duration (ns)"]) / 1e6 / n
            groups[group(r["Name"])] += ms
            names[r["Name"]] += ms
        row = dict(tag=tag, ms_per_step_event=meta["ms_per_step_cuda_event"], gpu_busy_ms=busy,
                   idle_gap_ms=span - busy, kernels_per_step=len(kern) / n,
                   **{f"group_{k}_ms": v for k, v in groups.items()})
        rows.append(row)
        print(f"\n== {tag}: step {meta['ms_per_step_cuda_event']:.2f} ms (events) | GPU busy {busy:.2f} ms | "
              f"idle gaps {span - busy:.2f} ms | {len(kern) / n:.0f} kernels/step")
        for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
            print(f"   {k:28s} {v:8.2f} ms  ({100 * v / busy:4.1f}%)")
        for nme, v in sorted(names.items(), key=lambda kv: -kv[1])[:6]:
            print(f"     {v:7.2f} ms  {re.sub(r'\s+', ' ', nme)[:90]}")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "tag", k))
    with (root / "nsys_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {root / 'nsys_summary.csv'}")


if __name__ == "__main__":
    main()
