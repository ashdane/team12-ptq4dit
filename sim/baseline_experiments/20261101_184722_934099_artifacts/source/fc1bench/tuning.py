"""Correctness-gated, randomized tuning with a separate holdout measurement."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
import triton

from .eval_utils import write_csv, write_json
from .optimized import Config, candidates, gemm, prepare_weight, quantize_native
from .quantization import quantize_rows
from .report import summarize


def fingerprint():
    root = Path(__file__).parent
    return dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                sources={n: hashlib.sha256((root/n).read_bytes()).hexdigest()
                         for n in ("optimized.py", "kernels.py", "quantization.py", "tuning.py")})


def key(m, n, k, bits, dtype):
    return f"{m}_{n}_{k}_w{bits}_{str(dtype).split('.')[-1]}"


def interleaved(functions, *, rounds=20, inner=5, warmup=10, seed=2026):
    """Eager event timings; each sample averages inner calls. No cache flushing."""
    if min(rounds, inner, warmup) < 1:
        raise ValueError("Timing counts must be positive")
    for fn in functions.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    rng = random.Random(seed)
    samples = {name: [] for name in functions}
    orders = []
    for _ in range(rounds):
        names = list(functions)
        rng.shuffle(names)
        orders.append(names)
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner):
                functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / inner)
    return dict(timings={name: summarize(values) for name, values in samples.items()},
                randomized_order=orders, rounds=rounds, inner=inner, warmup=warmup, seed=seed)


def tune(x, w, bias, bits, out, *, rounds=10, inner=5):
    """Compile/check every candidate before measuring. Fail closed on wrong math.

    Compile/resource failures are recorded and excluded; numerical mismatches
    abort the experiment rather than silently selecting another kernel.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    x = x.contiguous()
    xq, sx = quantize_native(x)
    qr, sr = quantize_rows(x)
    torch.testing.assert_close(xq, qr, rtol=0, atol=0)
    torch.testing.assert_close(sx, sr, rtol=0, atol=0)
    wq, sw = quantize_rows(w, bits)
    n, k = w.shape
    m = x.shape[0]
    ident = key(m, n, k, bits, x.dtype)
    expected_int = F.linear(xq.double(), wq.double()).to(torch.int32)
    expected = expected_int.float() * sx[:, None] * sw[None, :]
    if bias is not None:
        expected += bias
    expected = expected.to(x.dtype)
    report = dict(key=ident, environment=fingerprint(), shape=[m, n, k], bits=bits,
                  output_dtype=str(x.dtype), candidates=[], status="running")
    valid, funcs = {}, {}
    weights = {}
    t0 = time.perf_counter()
    for i, config in enumerate(candidates(bits)):
        name = f"c{i:02d}"
        row = dict(name=name, config=asdict(config))
        print(f"Tuning {ident}: {name} {asdict(config)}", flush=True)
        if config.layout not in weights:
            weights[config.layout] = prepare_weight(wq, bits, config)
        weight = weights[config.layout]
        try:
            raw = gemm(xq, weight, sx, sw, bias, bits=bits, config=config, raw=True)
            y, kernel = gemm(xq, weight, sx, sw, bias, bits=bits, config=config,
                             dtype=x.dtype, return_kernel=True)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            # No launch-time device faults are treated as harmless tuning failures.
            message = str(exc).lower()
            if any(s in message for s in ("illegal memory", "device-side assert", "launch failure")):
                raise
            row.update(status="unsupported", error=f"{type(exc).__name__}: {exc}")
            report["candidates"].append(row)
            (out/f"{ident}.json").write_text(json.dumps(report, indent=2))
            continue
        try:
            torch.testing.assert_close(raw, expected_int, rtol=0, atol=0)
            rtol = {torch.float32: 2e-5, torch.float16: 1e-3, torch.bfloat16: 1e-2}[x.dtype]
            torch.testing.assert_close(y.float(), expected.float(), rtol=rtol, atol=2e-5)
        except AssertionError:
            row["status"] = "numerical_failure"
            report["candidates"].append(row)
            report["status"] = "failed"
            (out/f"{ident}.json").write_text(json.dumps(report, indent=2))
            raise
        row["status"] = "validated"
        row["epilogue_max_abs_error"] = (y.float() - expected.float()).abs().max().item()
        if kernel is not None:
            row["registers"] = getattr(kernel, "n_regs", None)
            row["spills"] = getattr(kernel, "n_spills", None)
        report["candidates"].append(row)
        valid[name] = config
        funcs[name] = lambda c=config, wt=weight: gemm(xq, wt, sx, sw, bias, bits=bits, config=c, dtype=x.dtype)
        del raw, y
    if not valid:
        raise RuntimeError(f"No valid kernel for {ident}; inspect tuning logs")
    report["compile_and_validation_wall_s"] = time.perf_counter() - t0
    report["search"] = interleaved(funcs, rounds=rounds, inner=inner)
    winner = min(valid, key=lambda name: report["search"]["timings"][name]["median_ms"])
    config = valid[winner]
    wd, bd = w.to(x.dtype), None if bias is None else bias.to(x.dtype)

    def layer():
        q, scale = quantize_native(x)
        return gemm(q, weights[config.layout], scale, sw, bias, bits=bits, config=config, dtype=x.dtype)

    from .kernels import integer_linear
    original_weight = prepare_weight(wq, bits, Config())
    holdout = {"selected_core": funcs[winner], "selected_layer": layer,
               "native_float_linear": lambda: F.linear(x, wd, bd),
               "original_core_fp32_output": lambda: integer_linear(
                   xq, original_weight, sx, sw, bias, packed=bits == 4)}
    report["holdout"] = interleaved(holdout, rounds=20, inner=inner, seed=2027)
    report.update(selected=winner, config=asdict(config), status="ok")
    _, compiled = gemm(xq, weights[config.layout], sx, sw, bias, bits=bits,
                       config=config, dtype=x.dtype, return_kernel=True)
    if compiled is not None:
        # For torch_int_mm these artifacts describe only its separate epilogue.
        prefix = out / (ident + ("_epilogue" if config.backend == "torch_int_mm" else "_gemm"))
        prefix.with_suffix(".ptx").write_text(compiled.asm.get("ptx", ""))
        if "cubin" in compiled.asm:
            prefix.with_suffix(".cubin").write_bytes(compiled.asm["cubin"])
    (out/f"{ident}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return config, report


class Bank:
    def __init__(self, path):
        self.path = Path(path)
        self.data = dict(environment=fingerprint(), entries={})
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data["environment"] != fingerprint():
                raise ValueError("Tuning bank GPU/software/source mismatch; tune a new bank")

    def get(self, m, n, k, bits, dtype):
        name = key(m, n, k, bits, dtype)
        if name not in self.data["entries"]:
            raise KeyError(f"Missing tuning entry {name}; tune before inference")
        return Config(**self.data["entries"][name])

    def add(self, m, n, k, bits, dtype, config):
        self.data["entries"][key(m, n, k, bits, dtype)] = asdict(config)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, indent=2))
        temp.replace(self.path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", nargs="+", default=["2048x4608x1152", "2048x1152x4608"],
                   help="MxNxK shapes to tune (defaults to DiT-XL/2 FC1 and FC2 at batch=4 CFG)")
    p.add_argument("--bits", nargs="+", type=int, choices=[4, 8], default=[8, 4])
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default="results/tuning")
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        p.error("A CUDA GPU is required for kernel tuning")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    bank = Bank(out / "tuning_bank.json")
    rows = []
    for spec in args.shapes:
        m, n, k = (int(v) for v in spec.lower().split("x"))
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        x = torch.randn(m, k, generator=gen).cuda().to(dtype).contiguous()
        w = (torch.randn(n, k, generator=gen) / math.sqrt(k)).cuda().float().contiguous()
        b = (torch.randn(n, generator=gen) * 0.01).cuda().float().contiguous()
        for bits in args.bits:
            config, report = tune(x, w, b, bits, out, rounds=args.rounds, inner=args.inner)
            bank.add(m, n, k, bits, dtype, config)
            holdout = report["holdout"]["timings"]
            rows.append(dict(
                key=report["key"], m=m, n=n, k=k, bits=bits, dtype=args.dtype,
                selected=report["selected"], backend=config.backend, layout=config.layout,
                bm=config.bm, bn=config.bn, bk=config.bk, warps=config.warps, stages=config.stages,
                selected_core_median_ms=holdout["selected_core"]["median_ms"],
                selected_layer_median_ms=holdout["selected_layer"]["median_ms"],
                native_float_linear_median_ms=holdout["native_float_linear"]["median_ms"],
                original_core_fp32_median_ms=holdout["original_core_fp32_output"]["median_ms"],
            ))
        del x, w, b
        torch.cuda.empty_cache()
    write_csv(out / "summary.csv", rows)
    write_json(out / "summary.json", rows)


if __name__ == "__main__":
    main()
