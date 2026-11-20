"""INT8 GEMM tile exploration beyond the tuner's candidate set (diagnostic for the next GEMM step).

For each DiT projection shape at M = 2048 (batch 4) and 16384 (batch 32), times:
  * every candidate Triton config (current tuned one + larger tiles / deeper pipelines),
    after checking its raw INT32 accumulator is identical to torch._int_mm;
  * cuBLASLt INT8 main loop alone (torch._int_mm, raw INT32), and with our epilogue;
  * cuBLAS FP16 F.linear.
Timing: CUDA events around 10 back-to-back calls, 30 samples, median per call (warm L2).
Usage: PYTHONPATH=src python3 -m fc1bench.gemm_explore --bank tuning_bank.json --output gemm_explore.json
"""
import argparse
import json
import statistics

import torch
import torch.nn.functional as F

from .optimized import Config, gemm, prepare_weight, quantize_native
from .quantization import quantize_rows
from .tuning import key

SHAPES = {"qkv": (3456, 1152), "proj": (1152, 1152), "fc1": (4608, 1152), "fc2": (1152, 4608)}
EXTRA = [(64, 128, 64, 4, 3), (64, 128, 128, 4, 3), (64, 256, 64, 4, 3), (64, 256, 128, 8, 3),
         (128, 64, 128, 4, 3), (128, 128, 64, 4, 3), (128, 128, 64, 8, 3), (128, 128, 64, 8, 4),
         (128, 128, 128, 4, 3), (128, 128, 128, 8, 3), (128, 256, 64, 8, 3), (128, 256, 128, 8, 2),
         (128, 256, 128, 8, 3), (256, 128, 64, 8, 3), (256, 128, 128, 8, 2), (256, 128, 128, 8, 3)]


def timeit(fn, inner=10, samples=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(samples):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner):
            fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / inner)
    return statistics.median(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rows", nargs="+", type=int, default=[2048, 16384])
    args = p.parse_args(argv)
    torch.set_grad_enabled(False)
    entries = json.load(open(args.bank))["entries"]
    results = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for m in args.rows:
        for name, (n, k) in SHAPES.items():
            x = (torch.randn(m, k, generator=g) * 2).cuda().half()
            w = (torch.randn(n, k, generator=g) / k ** 0.5).cuda().float()
            b = (torch.randn(n, generator=g) * 0.01).cuda().float()
            xq, sx = quantize_native(x)
            wq, sw = quantize_rows(w, 8)
            wq = wq.contiguous()
            ref = torch._int_mm(xq, wq.T)
            flops = 2 * m * n * k
            row = dict(layer=name, m=m, n=n, k=k)
            row["fp16_cublas_ms"] = timeit(lambda: F.linear(x, w.half(), b.half()))
            row["int8_cublaslt_mainloop_ms"] = timeit(lambda: torch._int_mm(xq, wq.T))
            lt = Config(backend="torch_int_mm")
            row["int8_cublaslt_with_epilogue_ms"] = timeit(lambda: gemm(xq, wq, sx, sw, b, bits=8, config=lt,
                                                                       dtype=torch.float16))
            tuned = entries.get(key(m, n, k, 8, torch.float16))
            cands = [("tuned", Config(**tuned))] if tuned and Config(**tuned).backend == "triton" else []
            cands += [(f"{bm}x{bn}x{bk}_w{wp}_s{st}", Config(bm, bn, bk, wp, st, 8, "nk")) for bm, bn, bk, wp, st in EXTRA]
            trials = []
            for label, cfg in cands:
                try:
                    wp_ = prepare_weight(wq, 8, cfg)
                    raw = gemm(xq, wp_, sx, sw, b, bits=8, config=cfg, raw=True)
                    torch.cuda.synchronize()
                    if not torch.equal(raw, ref):
                        trials.append(dict(config=label, status="WRONG_RESULT"))
                        continue
                    ms = timeit(lambda: gemm(xq, wp_, sx, sw, b, bits=8, config=cfg, dtype=torch.float16))
                    trials.append(dict(config=label, status="ok", ms=ms, tops=flops / ms / 1e9))
                except Exception as exc:
                    msg = str(exc).lower()
                    if any(s in msg for s in ("illegal memory", "device-side assert", "launch failure")):
                        raise
                    trials.append(dict(config=label, status="unsupported", error=f"{type(exc).__name__}: {str(exc)[:120]}"))
            ok = [t for t in trials if t["status"] == "ok"]
            best = min(ok, key=lambda t: t["ms"]) if ok else None
            tuned_ms = next((t["ms"] for t in ok if t["config"] == "tuned"), None)
            row.update(trials=trials, best_config=best and best["config"], best_ms=best and best["ms"],
                       best_tops=best and best["tops"], tuned_ms=tuned_ms,
                       best_vs_tuned=(tuned_ms / best["ms"]) if (best and tuned_ms) else None,
                       best_vs_fp16=(row["fp16_cublas_ms"] / best["ms"]) if best else None,
                       cublaslt_mainloop_vs_best=(best["ms"] / row["int8_cublaslt_mainloop_ms"]) if best else None)
            results.append(row)
            print(json.dumps({k_: v for k_, v in row.items() if k_ != "trials"}), flush=True)
    json.dump(dict(gpu=torch.cuda.get_device_name(), results=results), open(args.output, "w"), indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
