"""Nsight Systems target: replay one CUDA-graphed denoising step N times inside a capture range.

Run under:
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
      -t cuda,nvtx -o <out> python -m fc1bench.nsys_step --mode int8 --scaffold sdpa_graph_pure16_compile_tuned_fused ...

Unlike ncu, nsys does not lock clocks or replay kernels, so per-kernel durations and the idle
gaps between kernels are those of the real timed configuration.
"""
import argparse
import json
from types import SimpleNamespace

import torch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit-repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bank")
    p.add_argument("--mode", choices=["fp16", "int8", "int4"], required=True)
    p.add_argument("--scaffold", required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--replays", type=int, default=20)
    args = p.parse_args(argv)
    from .full_model import load_checkpoint, load_official
    from .latency_bench import Scaffold, build, make_forward, make_graphed, make_inputs, patch_timestep_embedding
    from .tuning import Bank
    torch.set_grad_enabled(False)
    models, _ = load_official(args.dit_repo)
    patch_timestep_embedding(models)
    state = load_checkpoint(args.checkpoint)
    sc = Scaffold.parse(args.scaffold)
    ns = SimpleNamespace(image_size=256, scope="all")
    with torch.inference_mode():
        model, dtype, _, _ = build(models, state, ns, args.mode, args.batch, sc, Bank(args.bank) if args.bank else None)
        fwd = make_forward(model, dtype, 1.5, sc)
        x, t, y = make_inputs(args.batch, 256, 1)
        fwd(x, t, y)
        replay, _ = make_graphed(fwd, x, t, y) if sc.graphs else (lambda: fwd(x, t, y), None)
        for _ in range(10):
            replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.profiler.start()
        start.record()
        for i in range(args.replays):
            torch.cuda.nvtx.range_push(f"step{i}")
            replay()
            torch.cuda.nvtx.range_pop()
        end.record()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    print(json.dumps(dict(mode=args.mode, scaffold=args.scaffold, batch=args.batch, replays=args.replays,
                          ms_per_step_cuda_event=start.elapsed_time(end) / args.replays)), flush=True)


if __name__ == "__main__":
    main()
