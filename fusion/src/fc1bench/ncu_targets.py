"""Isolated kernel targets for Nsight Compute (ncu).

Run under ncu with --profile-from-start off: only the region between
torch.cuda.profiler.start()/stop() is profiled, after warmup/compilation.

  layer     one projection op at a real DiT-XL/2 shape with real block weights:
            fp16 (cuBLAS F.linear) | quant (our activation quantizer) | int8 (our Triton GEMM,
            tuned config) | int8lt (cuBLASLt torch._int_mm + epilogue) | int4 (our packed W4 GEMM)
  block     one eager DiT block (block 13) inside a full forward: fp16 | int8 | int4
  compiled  one torch.compile'd forward (no CUDA graph): fp16 | int8 | int4

Examples (see scripts/ncu_profile.sh):
  ncu --profile-from-start off --set full -o fc1_int8 \
      python -m fc1bench.ncu_targets layer --layer fc1 --op int8 --rows 16384 --bank tuning_bank.json ...
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

SHAPES = {"qkv": (3456, 1152), "proj": (1152, 1152), "fc1": (4608, 1152), "fc2": (1152, 4608)}
SUFFIX = {"qkv": "attn.qkv", "proj": "attn.proj", "fc1": "mlp.fc1", "fc2": "mlp.fc2"}


def bank_config(bank_path, m, n, k, bits):
    from .optimized import Config
    from .tuning import key
    entries = json.loads(Path(bank_path).read_text())["entries"]
    name = key(m, n, k, bits, torch.float16)
    if name not in entries:
        raise KeyError(f"{name} not in {bank_path}")
    cfg = Config(**entries[name])
    if cfg.backend != "triton":
        # The tuner picked cuBLASLt for this shape; profile our best *Triton* kernel separately
        # by falling back to the default portable tiles, and say so.
        print(f"[ncu_targets] bank picked {cfg.backend} for {name}; using default Triton tiles", flush=True)
        cfg = Config(bm=64, bn=128, bk=64, warps=4, stages=1, layout="kn_packed" if bits == 4 else "nk")
    return cfg


def run_layer(args):
    from .optimized import gemm, prepare_weight, quantize_native
    from .quantization import quantize_rows
    n, k = SHAPES[args.layer]
    m = args.rows
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("ema", state)
    name = f"blocks.13.{SUFFIX[args.layer]}"
    w = state[name + ".weight"].cuda().float()
    b = state[name + ".bias"].cuda().float()
    g = torch.Generator(device="cpu").manual_seed(0)
    x16 = (torch.randn(m, k, generator=g) * 2.0).cuda().half().contiguous()
    w16, b16 = w.half(), b.half()
    bits = 4 if args.op == "int4" else 8
    wq, sw = quantize_rows(w, bits)
    xq, sx = quantize_native(x16)
    if args.op == "fp16":
        fn = lambda: F.linear(x16, w16, b16)
    elif args.op == "quant":
        fn = lambda: quantize_native(x16)
    elif args.op in ("int8", "int4"):
        cfg = bank_config(args.bank, m, n, k, bits) if args.bank else None
        from .optimized import Config
        cfg = cfg or Config(bm=64, bn=128, bk=64, warps=4, stages=1, layout="kn_packed" if bits == 4 else "nk")
        wp = prepare_weight(wq, bits, cfg)
        fn = lambda: gemm(xq, wp, sx, sw, b, bits=bits, config=cfg, dtype=torch.float16)
        print(f"[ncu_targets] {args.op} config {cfg}", flush=True)
    elif args.op == "int8lt":
        from .optimized import Config
        cfg = Config(backend="torch_int_mm")
        wp = wq.contiguous()
        fn = lambda: gemm(xq, wp, sx, sw, b, bits=8, config=cfg, dtype=torch.float16)
    else:
        raise ValueError(args.op)
    profile(fn, args.warmup)
    print(json.dumps(dict(target="layer", layer=args.layer, op=args.op, m=m, n=n, k=k,
                          flops=2 * m * n * k)), flush=True)


def profile(fn, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push("ncu_target")
    fn()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def run_fusedk(args):
    """One fused producer+quant kernel (fused_ops.py) at a real DiT shape."""
    from . import fused_ops as f
    g = torch.Generator(device="cpu").manual_seed(0)
    m, tokens, c = args.rows, 256, 1152
    n_img = m // tokens
    x = (torch.randn(m, c, generator=g) * 3).cuda().half()
    d = (torch.randn(m, c, generator=g) * 2).cuda().half()
    mod = (torch.randn(n_img, 6 * c, generator=g) * 0.5).cuda().half()
    h = (torch.randn(m, 4 * c, generator=g) * 2).cuda().half()
    fns = {"ln": lambda: f.ln_mod_quant(x, mod, 0, c, tokens, 1e-6),
           "resln": lambda: f.res_ln_mod_quant(x, d, mod, 5 * c, mod, 3 * c, 4 * c, tokens, 1e-6),
           "gelu": lambda: f.gelu_quant(h),
           "row": lambda: f.row_quant(x)}
    profile(fns[args.kernel], max(args.warmup, 5))   # warmup also finishes Triton autotuning
    bytes_moved = {"ln": m * c * (2 + 1) + m * 4, "resln": m * c * (2 + 2 + 2 + 1) + m * 4,
                   "gelu": m * 4 * c * (2 + 1) + m * 4, "row": m * c * (2 + 1) + m * 4}[args.kernel]
    print(json.dumps(dict(target="fusedk", kernel=args.kernel, m=m, ideal_bytes=bytes_moved)), flush=True)


def run_fusedblock(args):
    """Block 13 of the hand-fused forward (eager, no graph), real activations from blocks 0-12."""
    from .fused_ops import fused_forward_with_cfg
    model, _, (x, t, y) = model_setup(args, compile_=False, fused=True)
    fwd = lambda hook=None: fused_forward_with_cfg(model, x.half(), t, y, 1.5, hook)

    def hook(i, phase):
        if i == args.block and phase == "start":
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
        elif i == args.block and phase == "end":
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
    with torch.inference_mode():
        for _ in range(max(args.warmup, 3)):
            fwd()
        torch.cuda.synchronize()
        fwd(hook)
        torch.cuda.synchronize()
    print(json.dumps(dict(target="fusedblock", mode=args.mode, batch=args.batch, block=args.block)), flush=True)


def model_setup(args, compile_, fused=False):
    from .full_model import load_checkpoint, load_official
    from .latency_bench import Scaffold, build, make_forward, patch_timestep_embedding, make_inputs
    from .tuning import Bank
    models, _ = load_official(args.dit_repo)
    patch_timestep_embedding(models)
    state = load_checkpoint(args.checkpoint)
    if args.csb:
        from .csb import load_folded
        state, _ = load_folded(state, args.csb)
    fused = fused or getattr(args, "fused", False)
    sc = Scaffold("ncu", attention="sdpa", fp16io=True, tuned=bool(args.bank), pure16=True, compile=compile_,
                  fused=fused)
    ns = SimpleNamespace(image_size=256, scope="all")
    bank = Bank(args.bank) if args.bank else None
    model, dtype, _, _ = build(models, state, ns, args.mode, args.batch, sc, bank)
    fwd = make_forward(model, dtype, 1.5, sc)
    x, t, y = make_inputs(args.batch, 256, 1)
    return model, fwd, (x, t, y)


def run_block(args):
    model, fwd, (x, t, y) = model_setup(args, compile_=False)
    with torch.inference_mode():
        for _ in range(args.warmup):
            fwd(x, t, y)
        torch.cuda.synchronize()
        block = model.blocks[args.block]

        # Hooks must return None: a non-None return value would replace the block's
        # inputs (pre-hook) or output (forward hook).
        def start(module, inputs):
            torch.cuda.profiler.start()
            torch.cuda.nvtx.range_push(f"block{args.block}")

        def stop(module, inputs, output):
            torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.stop()
        pre = block.register_forward_pre_hook(start)
        post = block.register_forward_hook(stop)
        fwd(x, t, y)
        torch.cuda.synchronize()
        pre.remove(), post.remove()
    print(json.dumps(dict(target="block", mode=args.mode, batch=args.batch, block=args.block)), flush=True)


def run_compiled(args):
    model, fwd, (x, t, y) = model_setup(args, compile_=True)
    with torch.inference_mode():
        profile(lambda: fwd(x, t, y), max(args.warmup, 3))
    print(json.dumps(dict(target="compiled", mode=args.mode, batch=args.batch)), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="target", required=True)
    for name in ("layer", "block", "compiled", "fusedk", "fusedblock"):
        s = sub.add_parser(name)
        s.add_argument("--checkpoint", required=name != "fusedk")
        s.add_argument("--bank")
        s.add_argument("--warmup", type=int, default=3)
        if name == "fusedk":
            s.add_argument("--kernel", choices=["ln", "resln", "gelu", "row"], required=True)
            s.add_argument("--rows", type=int, required=True)
            continue
        if name == "layer":
            s.add_argument("--layer", choices=sorted(SHAPES), required=True)
            s.add_argument("--op", choices=["fp16", "quant", "int8", "int8lt", "int4"], required=True)
            s.add_argument("--rows", type=int, required=True, help="M = 2 * batch * 256")
        else:
            s.add_argument("--dit-repo", required=True)
            s.add_argument("--mode", choices=["fp16", "int8", "int4"], required=True)
            s.add_argument("--batch", type=int, required=True)
            s.add_argument("--csb", help="Optional CSB scales file to fold")
            if name in ("block", "fusedblock"):
                s.add_argument("--block", type=int, default=13)
            if name == "compiled":
                s.add_argument("--fused", action="store_true", help="Use the hand-fused DiT block forward")
    args = p.parse_args(argv)
    torch.set_grad_enabled(False)
    {"layer": run_layer, "block": run_block, "compiled": run_compiled, "fusedk": run_fusedk,
     "fusedblock": run_fusedblock}[args.target](args)


if __name__ == "__main__":
    main()
