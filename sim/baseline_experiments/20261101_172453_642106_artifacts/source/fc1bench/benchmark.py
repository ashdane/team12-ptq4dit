"""python -m fc1bench.benchmark --help"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from .quantization import metrics, pack_int4, quantize_rows
from .report import summarize


MODES = ("fp32", "fp16", "bf16", "fake_w8a8", "fake_w4a8", "int8", "int4")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timing(fn, args, device):
    for _ in range(args.warmup):
        fn()
    synchronize(device)
    gpu_samples, wall_samples = [], []
    for _ in range(args.repeats):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        wall_start = time.perf_counter()
        for _ in range(args.inner):
            fn()
        if device.type == "cuda":
            end.record()
        synchronize(device)
        wall_samples.append((time.perf_counter() - wall_start) * 1000 / args.inner)
        if device.type == "cuda":
            gpu_samples.append(start.elapsed_time(end) / args.inner)
    return dict(cuda_event=summarize(gpu_samples) if gpu_samples else None,
                synchronized_wall=summarize(wall_samples))


def memory(fn, device):
    if device.type != "cuda":
        return None
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = fn()
    synchronize(device)
    result = dict(baseline_allocated_bytes=allocated, baseline_reserved_bytes=reserved,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    result["incremental_peak_allocated_bytes"] = result["peak_allocated_bytes"] - allocated
    result["output_bytes"] = output.numel() * output.element_size()
    del output
    return result


def environment(device):
    packages = {}
    for name in ("torch", "triton", "numpy", "nvidia-cublas-cu12", "nvidia-cublas-cu13"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = dict(python=sys.version, platform=platform.platform(), packages=packages,
                  cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  torch_build=torch.__config__.show(), device=str(device),
                  tf32=False, float32_matmul_precision=torch.get_float32_matmul_precision())
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, compute_capability=list(torch.cuda.get_device_capability(device)),
                      gpu_total_memory_bytes=props.total_memory)
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,pstate,clocks.sm,clocks.mem",
             "--format=csv"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = None
    return result


def load_data(args, device):
    if args.input:
        data = torch.load(args.input, map_location="cpu", weights_only=True)
        x, w, b = data["x"], data["weight"], data.get("bias")
        if x.ndim < 2 or w.ndim != 2 or x.shape[-1] != w.shape[1]:
            raise ValueError("Input must contain x [...,K], weight [N,K], optional bias [N]")
        x = x.reshape(-1, x.shape[-1])
        if b is not None and b.shape != (w.shape[0],):
            raise ValueError("bias must have shape [N]")
        provenance = dict(kind="captured", file=str(Path(args.input).resolve()),
                          sha256=file_hash(args.input), metadata=data.get("metadata", {}))
    else:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        x = torch.randn(args.m, args.k, generator=generator)
        w = torch.randn(args.n, args.k, generator=generator) / math.sqrt(args.k)
        b = torch.randn(args.n, generator=generator) * .01
        provenance = dict(kind="synthetic", seed=args.seed, distribution="X~N(0,1), W~N(0,1/K), b~N(0,0.0001)")
    values = [t.to(device=device, dtype=torch.float32).contiguous() if t is not None else None
              for t in (x, w, b)]
    if any(not bool(torch.isfinite(t).all()) for t in values if t is not None):
        raise ValueError("Input tensors must be finite")
    if min(values[0].shape + values[1].shape) < 1:
        raise ValueError("Empty dimensions are unsupported")
    return *values, provenance


def save_assembly(kernel, out, name):
    ptx = kernel.asm.get("ptx", "")
    (out / f"{name}.ptx").write_text(ptx)
    cubin = kernel.asm.get("cubin")
    if cubin is not None:
        (out / f"{name}.cubin").write_bytes(cubin)
    instructions = sorted(set(line.strip() for line in ptx.splitlines()
                              if re.search(r"mma\..*\.s32\.s8\.s8\.s32", line)))
    return dict(ptx_file=f"{name}.ptx", integer_mma_instructions=instructions,
                integer_mma_in_ptx=bool(instructions),
                hardware_counter_verification="not performed; use Nsight Compute")


def run_mode(name, x, w, b, reference, args, out):
    device = x.device
    synchronize(device)
    prep_start = time.perf_counter()
    metadata, state = {}, None
    if name in ("fp32", "fp16", "bf16"):
        dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[name]
        if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            return dict(mode=name, status="skipped", reason="BF16 not supported")
        xd, wd = x.to(dtype), w.to(dtype)
        bd = b.to(dtype) if b is not None else None
        core = lambda: F.linear(xd, wd, bd)
        # Common layer boundary: FP32 input -> native GEMM -> FP32 output.
        layer = lambda: F.linear(x.to(dtype), wd, bd).float()
        weight_bytes = wd.numel() * wd.element_size()
        meta_bytes = 0
        bias_bytes = 0 if bd is None else bd.numel() * bd.element_size()
        metadata.update(operands=f"{dtype} x {dtype}", core="F.linear including bias; pre-cast operands")
    elif name.startswith("fake_"):
        if not args.ptq4dit_repo:
            return dict(mode=name, status="skipped", reason="Supply --ptq4dit-repo for actual upstream operator")
        from .upstream import isolated_module
        bits = 8 if name == "fake_w8a8" else 4
        module, xd, wd, metadata, state = isolated_module(args.ptq4dit_repo, x, w, b, bits, args.upstream_scale_method)
        core = lambda: F.linear(xd, wd, b)
        layer = lambda: module(x)
        weight_bytes = w.numel() * w.element_size()
        meta_bytes = sum(v.numel() * v.element_size() for item in state.values()
                         for v in item.values() if torch.is_tensor(v))
        bias_bytes = 0 if b is None else b.numel() * b.element_size()
        metadata.update(operands="dequantized FP32 x FP32", core="F.linear on precomputed dequantized operands")
    else:
        if device.type != "cuda":
            return dict(mode=name, status="skipped", reason="Integer kernels require CUDA")
        from .kernels import integer_linear, quantize_activation
        packed = name == "int4"
        wq, sw = quantize_rows(w, 4 if packed else 8)
        weight = pack_int4(wq) if packed else wq
        xq, sx = quantize_activation(x)
        core = lambda: integer_linear(xq, weight, sx, sw, b, packed=packed)

        def layer():
            q, scale = quantize_activation(x)
            return integer_linear(q, weight, scale, sw, b, packed=packed)

        weight_bytes = weight.numel() * weight.element_size()
        meta_bytes = sw.numel() * sw.element_size()
        bias_bytes = 0 if b is None else b.numel() * b.element_size()
        metadata.update(operands="INT8 x (packed INT4 unpacked to INT8 in registers)" if packed else "INT8 x INT8",
                        core="integer GEMM + fused scale/bias/FP32 store",
                        quantization="experimental symmetric absmax; per-token A8, per-output W; zero-points=0",
                        activation_range=[-127, 127], weight_range=[-7, 7] if packed else [-127, 127],
                        activation_scale_bytes=sx.numel() * sx.element_size(),
                        ptq4dit_equivalent=False)
    synchronize(device)
    prep_ms = (time.perf_counter() - prep_start) * 1000
    # Includes first use/JIT where applicable; it is intentionally NOT steady state.
    observed = layer()
    correctness = metrics(reference, observed)
    implementation = None
    if name in ("int8", "int4"):
        from .kernels import integer_linear
        from .quantization import unpack_int4
        expected_q, expected_s = quantize_rows(x)
        torch.testing.assert_close(xq, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(sx, expected_s, rtol=0, atol=0)
        # FP64 accumulates integer products exactly for these bounded dimensions.
        # This correctness-only work is outside every timed region.
        integer_reference = F.linear(xq.double(), wq.double()).to(torch.int32)
        raw = integer_linear(xq, weight, sx, sw, packed=packed, raw=True)
        torch.testing.assert_close(raw, integer_reference, rtol=0, atol=0)
        if packed:
            torch.testing.assert_close(unpack_int4(weight, x.shape[1]), wq, rtol=0, atol=0)
        quant_reference = integer_reference.float() * sx[:, None] * sw[None, :]
        if b is not None:
            quant_reference += b
        torch.testing.assert_close(observed, quant_reference, rtol=2e-5, atol=2e-5)
        implementation = metrics(quant_reference, observed)
        assembly_output, kernel = integer_linear(xq, weight, sx, sw, b, packed=packed, return_kernel=True)
        metadata["assembly"] = save_assembly(kernel, out, name)
        del assembly_output
        torch.save(dict(weight=weight.cpu(), weight_scale=sw.cpu(), activation_scale=sx.cpu(),
                        weight_bits=4 if packed else 8, k=x.shape[1], zero_points=0), out / f"{name}_quantization.pt")
        del integer_reference, raw, quant_reference, expected_q, expected_s, wq
    elif name.startswith("fake_"):
        torch.testing.assert_close(observed, core(), rtol=1e-5, atol=1e-5)
        torch.save(state, out / f"{name}_quantization.pt")
    del observed
    result = dict(mode=name, status="ok", metadata=metadata, preparation_wall_ms=prep_ms,
                  correctness_vs_fp32=correctness, implementation_error_vs_same_quantization=implementation,
                  weight_storage_bytes=weight_bytes, weight_metadata_bytes=meta_bytes,
                  bias_storage_bytes=bias_bytes,
                  core_timing=timing(core, args, device), layer_timing=timing(layer, args, device),
                  layer_memory=memory(layer, device))
    if args.profile and device.type == "cuda":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as profiler:
            for _ in range(3):
                with torch.profiler.record_function(name + "_layer"):
                    layer()
            synchronize(device)
        profiler.export_chrome_trace(str(out / f"{name}_trace.json"))
        (out / f"{name}_operators.txt").write_text(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
    return result


def write_results(out, report):
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    rows = []
    for result in report["results"]:
        row = dict(mode=result["mode"], status=result["status"], reason=result.get("reason", ""))
        if result["status"] == "ok":
            row.update(result["correctness_vs_fp32"])
            row.update(weight_storage_bytes=result["weight_storage_bytes"],
                       weight_metadata_bytes=result["weight_metadata_bytes"],
                       preparation_wall_ms=result["preparation_wall_ms"])
            for field in ("core", "layer"):
                for timer in ("cuda_event", "synchronized_wall"):
                    stats = result[f"{field}_timing"][timer]
                    if stats:
                        row.update({f"{field}_{timer}_{key}": value for key, value in stats.items() if key != "samples_ms"})
            if result["layer_memory"]:
                row.update(result["layer_memory"])
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=1152)
    parser.add_argument("--n", type=int, default=4608)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--output", default="results/fc1")
    parser.add_argument("--input", help="torch.save dictionary with x, weight, bias and optional metadata")
    parser.add_argument("--ptq4dit-repo")
    parser.add_argument("--upstream-scale-method", choices=["max", "mse"], default="max")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    if min(args.m, args.k, args.n, args.warmup, args.repeats, args.inner) < 1:
        parser.error("Dimensions, warmup, repeats and inner must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Select a GPU runtime; --device cpu is for smoke tests only.")
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        parser.error("Output already contains results; choose a new --output to preserve the previous run")
    x, w, b, provenance = load_data(args, device)
    reference = F.linear(x, w, b)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(root)): file_hash(p) for p in sorted((root / "fc1bench").glob("*.py"))}
    report = dict(schema_version=1, config=vars(args), environment=environment(device),
                  source_sha256=source_hashes, input=provenance,
                  actual_shape=dict(m=x.shape[0], k=x.shape[1], n=w.shape[0]),
                  protocol=dict(layer_boundary="FP32 input to FP32 output; weight prep excluded",
                                timing="warm eager calls; CUDA event and synchronized wall per call; no CUDA graphs",
                                memory="full harness working set retains FP32 reference/input/weights; incremental peak is measured separately",
                                reference="FP32 F.linear; TF32 disabled",
                                cpu="smoke test only; not GPU performance data"), results=[])
    failures = False
    for name in args.modes:
        print(f"Running {name}...", flush=True)
        try:
            result = run_mode(name, x, w, b, reference, args, out)
        except Exception as exc:
            result = dict(mode=name, status="failed", reason=f"{type(exc).__name__}: {exc}")
            failures = True
        report["results"].append(result)
        write_results(out, report)
        print(json.dumps({k: v for k, v in result.items() if k in ("mode", "status", "reason", "correctness_vs_fp32")}))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Saved {out / 'results.json'} and {out / 'results.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
