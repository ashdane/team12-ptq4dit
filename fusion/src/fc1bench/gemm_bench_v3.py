"""Per-shape INT8 GEMM comparison for the DiT-XL/2 projections (diagnostic).

For each projection shape at the given token counts, times (CUDA events, warm L2, median):
  fp16      cuBLAS F.linear
  lt_main   cuBLASLt INT8 main loop only (torch._int_mm, INT32 out, no scales)
  v2        the v2 Triton GEMM with the tuned config from a tuning bank (FP16 out, scales + bias)
  v3_*      fused_gemm.py kernels: plain, residual epilogue, GELU+group-quant epilogue, grouped-A
  cutlass_* (optional) the CUTLASS kernels from cutlass_gemm.py, if they build
Usage: PYTHONPATH=src python3 -m fc1bench.gemm_bench_v3 --bank tuning_bank.json --output out.json
"""
import argparse
import json
import statistics

import torch
import torch.nn.functional as F

SHAPES = {"qkv": (3456, 1152), "proj": (1152, 1152), "fc1": (4608, 1152), "fc2": (1152, 4608)}


def timeit(fn, inner=10, samples=25):
    """Median per call of `inner` back-to-back calls replayed from a CUDA graph (no launch overhead)."""
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(samples):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / inner)
    return statistics.median(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank")
    p.add_argument("--output", required=True)
    p.add_argument("--rows", nargs="+", type=int, default=[2048, 16384])
    p.add_argument("--cutlass", action="store_true")
    args = p.parse_args(argv)
    torch.set_grad_enabled(False)
    from . import fused_gemm as fg
    from .optimized import Config, gemm
    from .quantization import quantize_rows
    from .tuning import key
    entries = json.load(open(args.bank))["entries"] if args.bank else {}
    cg = None
    if args.cutlass:
        from . import cutlass_gemm as cg
    g = torch.Generator(device="cpu").manual_seed(0)
    rows_out = []
    for m in args.rows:
        tokens = 256
        mods = (torch.randn(m // tokens, 28 * 6 * 4608, generator=g) * 0.5).cuda().half()
        for name, (n, k) in SHAPES.items():
            x = (torch.randn(m, k, generator=g) * 2).cuda().half()
            w = (torch.randn(n, k, generator=g) / k ** 0.5).cuda().float()
            b = (torch.randn(n, generator=g) * 0.01).cuda().float()
            xq = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
            sx = (torch.rand(m, device="cuda") * 0.01 + 1e-3).float()
            sxg = (torch.rand(m, k // 128, device="cuda") * 0.01 + 1e-3).float()
            wq, sw = quantize_rows(w, 8)
            wq = wq.contiguous()
            r = torch.randn(m, n, device="cuda").half()
            flops = 2 * m * n * k
            row = dict(layer=name, m=m, n=n, k=k)
            t = {}
            t["fp16"] = timeit(lambda: F.linear(x, w.half(), b.half()))
            t["lt_main"] = timeit(lambda: torch._int_mm(xq, wq.T))
            ent = entries.get(key(m, n, k, 8, torch.float16))
            if ent:
                c = Config(**ent)
                t["v2"] = timeit(lambda: gemm(xq, wq, sx, sw, b, bits=8, config=c, dtype=torch.float16))
                row["v2_config"] = ent
            t["v3_plain"] = timeit(lambda: fg.gemm_plain(xq, sx, wq, sw, b))
            row["v3_plain_cfg"] = str(fg._gemm_v3.best_config)
            if name in ("proj", "fc2"):
                t["v3_res"] = timeit(lambda: fg.gemm_res(xq, sx, wq, sw, b, r, mods, 0, tokens, False))
            if name == "fc2":
                t["v3_res_grouped"] = timeit(lambda: fg.gemm_res(xq, sxg, wq, sw, b, r, mods, 0, tokens, True))
                row["v3_res_grouped_cfg"] = str(fg._gemm_v3.best_config)
                sxg2 = sxg[:, ::2].contiguous()
                t["v3_res_grouped256"] = timeit(lambda: fg.gemm_res(xq, sxg2, wq, sw, b, r, mods, 0, tokens, True))
                row["v3_res_grouped256_cfg"] = str(fg._gemm_v3.best_config)
            if name == "fc1":
                t["v3_gelu_gq"] = timeit(lambda: fg.gemm_gelu_gq(xq, sx, wq, sw, b))
                row["v3_gelu_gq_cfg"] = str(fg._gemm_v3.best_config)
                t["v3_gelu_gq256"] = timeit(lambda: fg.gemm_gelu_gq(xq, sx, wq, sw, b, 256))
                row["v3_gelu_gq256_cfg"] = str(fg._gemm_v3.best_config)
            if cg is not None:
                for label, fn in cg.bench_variants(name, xq, sx, sxg, wq, sw, b, r, mods, tokens).items():
                    try:
                        t[label] = timeit(fn)
                    except Exception as exc:  # report and continue
                        row[label + "_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            row["ms"] = t
            row["tops"] = {kk: flops / v / 1e9 for kk, v in t.items()}
            rows_out.append(row)
            print(f"M={m:5d} {name:4s} " + "  ".join(f"{kk} {v * 1e3:7.1f}us/{flops / v / 1e9:4.0f}T"
                                                    for kk, v in t.items()), flush=True)
    json.dump(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, results=rows_out),
              open(args.output, "w"), indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
