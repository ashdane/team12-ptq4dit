"""Per-step latency of pretrained DiT-XL/2 with one shared scaffold per comparison.

Each *scaffold* (attention implementation, CUDA graphs, quantized-layer I/O dtype,
INT kernel tiles) is applied identically to every mode, so within a scaffold only the
attention/MLP projection layers differ. Speedups are reported both against FP16 in the
same scaffold and against the fastest FP16 scaffold at the same batch (strongest baseline).

Timed region: one `forward_with_cfg` call at the real CFG batch shape (2 x batch).
Excluded: model construction, weight conversion, tuning, Triton compilation, warmup.
Diagnostic passes (module attribution, profiler) are separate and never timed as latency.

Scaffold names are '_'-joined tokens:
  baseline  explicit attention, eager, native projection I/O, default INT tiles (= image runs)
  sdpa      fused scaled_dot_product_attention (FlashAttention / memory-efficient kernels)
  graph     capture the whole forward in a CUDA graph and replay it
  fp16io    quantized projections take and return FP16 (instead of autocast's FP32)
  tuned     INT tile configs tuned on this GPU from captured real-model activations
  pure16    whole model natively in FP16, no autocast (removes FP32<->FP16 casts); implies fp16io
  compile   torch.compile the forward (INT projections are an opaque custom op, so both
            float and quantized models compile without graph breaks)
  fuseq     activation quantization written as PyTorch ops outside a GEMM-only custom op, so
            torch.compile can fuse it into the producer kernel (use together with compile)
  fused     hand-fused DiT block (fused_ops.py): residual+LayerNorm+modulate+quant, GELU+quant and
            row-quant Triton kernels around the INT GEMMs; implies pure16 (INT modes only)
  v3        fused block v3 (fused_gemm.py): residual adds in the proj/fc2 GEMM epilogues, GELU and
            128-column group quantization in the fc1 epilogue; implies fused (INT modes only)
  tattn     v3 with the single-pass Triton attention (fused_attn.py) instead of SDPA; implies v3
  v4        fused block with CUTLASS INT8 GEMMs (cutlass_gemm.py) for qkv and proj(+residual), v3 MLP
            (fc1 GELU + 128-column group quant, grouped fc2 + residual); implies fused
  v4w       as v4 with 256-column groups in the MLP
  v4c       CUTLASS for all four projections with per-token scales (v2 numerics) + GELU-quant kernel
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import time

import torch

from .benchmark import environment
from .eval_utils import git_record, model_storage, write_csv, write_json
from .full_model import load_checkpoint, load_official
from .generate import DefaultBank
from .model_layers import TunedLinear, convert, projection_names
from .quantization import metrics
from .report import summarize
from .tuning import Bank, key as bank_key, tune

FLOAT_MODES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
QUANT_MODES = {"int8": 8, "int4": 4}
TOKENS = {"baseline", "sdpa", "graph", "fp16io", "tuned", "pure16", "compile", "fuseq", "fused", "v3", "tattn", "v4", "v4w", "v4c"}


@dataclass(frozen=True)
class Scaffold:
    name: str
    attention: str = "explicit"
    graphs: bool = False
    fp16io: bool = False
    tuned: bool = False
    pure16: bool = False
    compile: bool = False
    fuseq: bool = False
    fused: bool = False
    v3: bool = False
    tattn: bool = False
    v4: str = ""

    @staticmethod
    def parse(name):
        tokens = name.split("_")
        unknown = set(tokens) - TOKENS
        if unknown:
            raise ValueError(f"Unknown scaffold token(s) {sorted(unknown)} in {name!r}; use {sorted(TOKENS)}")
        tattn = "tattn" in tokens
        v4s = [tok for tok in tokens if tok in ("v4", "v4w", "v4c")]
        if len(v4s) > 1:
            raise ValueError("use at most one of v4, v4w, v4c")
        v4 = v4s[0] if v4s else ""
        v3 = "v3" in tokens or tattn
        fused = "fused" in tokens or v3 or bool(v4)
        pure16 = "pure16" in tokens or fused
        if fused and "sdpa" not in tokens:
            raise ValueError("'fused' uses scaled_dot_product_attention; add the 'sdpa' token")
        return Scaffold(name, attention="sdpa" if "sdpa" in tokens else "explicit",
                        graphs="graph" in tokens, fp16io="fp16io" in tokens or pure16,
                        tuned="tuned" in tokens, pure16=pure16, compile="compile" in tokens,
                        fuseq="fuseq" in tokens, fused=fused, v3=v3, tattn=tattn, v4=v4)

    def effective(self, mode):
        """Knobs that actually change this mode (float modes ignore INT-only knobs)."""
        if mode in FLOAT_MODES:
            return (self.attention, self.graphs, self.pure16, self.compile)
        return (self.attention, self.graphs, self.fp16io, self.tuned, self.pure16, self.compile, self.fuseq,
                self.fused, self.v3, self.tattn, self.v4)


def frequency_table(dim, max_period=10000):
    """Upstream's exact CPU expression (models.py:52-54) for the sinusoidal frequencies."""
    import math
    half = dim // 2
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)


def patch_timestep_embedding(models):
    """Keep DiT's sinusoidal frequency table on the GPU (bit-identical, applied to all modes).

    Upstream builds the table on the CPU and copies it to the GPU on every forward
    (models.py:53-54). That per-step host->device copy is illegal during CUDA graph
    capture and is pure overhead. build() stores the identical table once as a non-
    persistent FP32 buffer; the pinned DiT checkout is not modified.

    The embedding is cast to the MLP weight dtype before the MLP. Under autocast this is
    exactly the cast autocast performs at the Linear; in pure FP16 it is required.
    """
    cls = models.TimestepEmbedder
    if getattr(cls, "_fc1bench_patched", False):
        return

    def forward(self, t):
        dim = self.frequency_embedding_size
        args = t[:, None].float() * self._fc1bench_freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))

    cls.forward = forward
    cls._fc1bench_patched = True


def install_frequency_buffer(model):
    te = model.t_embedder
    te.register_buffer("_fc1bench_freqs", frequency_table(te.frequency_embedding_size).cuda(),
                       persistent=False)


def make_inputs(batch, image_size, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(batch, 4, image_size // 8, image_size // 8, generator=g).cuda()
    x = torch.cat([z, z])
    y = torch.cat([torch.arange(batch, device="cuda") % 1000,
                   torch.full((batch,), 1000, device="cuda", dtype=torch.long)])
    t = torch.full((2 * batch,), 500, device="cuda", dtype=torch.long)
    return x, t, y


def build(models, state, args, mode, batch, scaffold, bank):
    model_dtype = FLOAT_MODES.get(mode, torch.float16)
    model = models.DiT_models["DiT-XL/2"](input_size=args.image_size // 8).eval()
    model.load_state_dict(state, strict=True)
    for block in model.blocks:
        if not hasattr(block.attn, "fused_attn"):
            raise RuntimeError("This timm version has no Attention.fused_attn switch; cannot select attention")
        block.attn.fused_attn = scaffold.attention == "sdpa"
    model = model.to(device="cuda", dtype=model_dtype)
    install_frequency_buffer(model)  # after .to(): table stays FP32 like upstream
    names = projection_names(model, args.scope)
    conversions = []
    if mode in QUANT_MODES:
        rows = batch * 2 * (args.image_size // 16) ** 2
        conversions = convert(model, names, state, bank if scaffold.tuned else DefaultBank(), rows,
                              QUANT_MODES[mode], model_dtype,
                              io_dtype=torch.float16 if scaffold.fp16io else None)
        for name in names:
            model.get_submodule(name).use_custom_op = scaffold.compile
            model.get_submodule(name).quant_mode = "torch" if scaffold.fuseq else "kernel"
        if scaffold.v4:
            # Pick CUTLASS tile configs for every v4 GEMM shape on real activations, before any
            # torch.compile tracing or CUDA graph capture.
            from .fused_ops import fused_forward_with_cfg
            x, t, y = make_inputs(batch, args.image_size, getattr(args, "seed", 1))
            with torch.inference_mode():
                fused_forward_with_cfg(model, x.half(), t, y, getattr(args, "cfg", 1.5), version=4,
                                       v4=scaffold.v4, tuning=True)
            torch.cuda.synchronize()
    return model, model_dtype, names, conversions


def make_forward(model, dtype, cfg, scaffold=None, compile_mode="default"):
    pure = bool(scaffold and scaffold.pure16) and dtype != torch.float32
    fused = bool(scaffold and scaffold.fused) and any(isinstance(m, TunedLinear) for m in model.modules())
    if fused:
        from .fused_ops import prepare_fused
        prepare_fused(model)          # batched adaLN weights registered before tracing/compilation

    def forward(x, t, y):
        if fused:
            from .fused_ops import fused_forward_with_cfg
            version = 4 if scaffold.v4 else 3 if scaffold.v3 else 2
            return fused_forward_with_cfg(model, x.to(dtype), t, y, cfg, version=version,
                                          triton_attn=scaffold.tattn, v4=scaffold.v4 or "v4")
        if pure:
            # Native FP16 end to end: only the input latent needs casting (autocast would
            # cast it at the patch-embedding conv anyway). Output is FP16 as under autocast.
            return model.forward_with_cfg(x.to(dtype), t, y, cfg_scale=cfg)
        # cache_enabled=False keeps autocast valid inside CUDA graph capture.
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
            return model.forward_with_cfg(x, t, y, cfg_scale=cfg)

    if scaffold is not None and scaffold.compile:
        return torch.compile(forward, mode=compile_mode, dynamic=False)
    return forward


def make_graphed(forward, x, t, y):
    sx, st, sy = x.clone(), t.clone(), y.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            forward(sx, st, sy)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = forward(sx, st, sy)

    def replay(x=None, t=None, y=None):
        if x is not None:
            sx.copy_(x), st.copy_(t), sy.copy_(y)
        graph.replay()
        return out
    return replay, graph


def time_calls(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return summarize(samples)


def attribution(model, call):
    """Eager, hook-instrumented CUDA-event attribution by module. Diagnostic only."""
    events, handles = {}, []
    for i in range(len(model.blocks)):
        for suffix in ("", ".attn", ".attn.qkv", ".attn.proj", ".mlp", ".mlp.fc1", ".mlp.fc2", ".adaLN_modulation"):
            name = f"blocks.{i}{suffix}"
            module = model.get_submodule(name)

            def pre(m, inputs, name=name):
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                events[name] = [e]

            def post(m, inputs, output, name=name):
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                events[name].append(e)
            handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    try:
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()
    ms = {n: a.elapsed_time(b) for n, (a, b) in events.items()}
    total = start.elapsed_time(end)
    out = defaultdict(float)
    for i in range(len(model.blocks)):
        p = f"blocks.{i}"
        for part in ("qkv", "proj"):
            out[f"linear_{part}"] += ms[f"{p}.attn.{part}"]
        for part in ("fc1", "fc2"):
            out[f"linear_{part}"] += ms[f"{p}.mlp.{part}"]
        out["attention_core"] += ms[f"{p}.attn"] - ms[f"{p}.attn.qkv"] - ms[f"{p}.attn.proj"]
        out["mlp_gelu"] += ms[f"{p}.mlp"] - ms[f"{p}.mlp.fc1"] - ms[f"{p}.mlp.fc2"]
        out["adaLN_modulation_float"] += ms[f"{p}.adaLN_modulation"]
        out["block_norm_modulate_residual"] += (ms[p] - ms[f"{p}.attn"] - ms[f"{p}.mlp"]
                                                - ms[f"{p}.adaLN_modulation"])
        out["_blocks"] += ms[p]
    out["outside_blocks_embed_final_cfg"] = total - out.pop("_blocks")
    out["total_instrumented"] = total
    return dict(out)


def kernel_profile(call, out_path, iters=3):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        for _ in range(iters):
            call()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out_path))
    trace = json.loads(Path(out_path).read_text())
    totals = defaultdict(float)
    for e in trace["traceEvents"]:
        if e.get("cat") == "kernel":
            totals[e["name"]] += e.get("dur", 0) / 1000 / iters
    groups = defaultdict(float)
    for name, ms in totals.items():
        low = name.lower()
        if "_res_ln_mod_quant" in low or "_act_quant" in low:
            group = "fused_producer_quant"     # fused_ops.py: LN/modulate/GELU + INT8 quantization
        elif "_quantize_fused" in low:
            group = "int_activation_quant"
        elif low.startswith("triton_") and ("amax" in low or "round" in low):
            group = "inductor_fused_quant"     # torch.compile kernels that include quantization (fuseq)
        elif low.startswith("triton_"):
            group = "norm" if "norm" in low else "elementwise_other"   # other torch.compile kernels
            group = "layernorm" if group == "norm" else group
        elif ("_tiled" in low or "_epilogue" in low or "_gemm_v3" in low or "i8" in low or "int8" in low
              or "imma" in low or "_s8" in low or "s8s8" in low or "i16832" in low or "i8816" in low
              or "signed char" in low or "gemmshape<16, 8, 32>" in low):
            # cuBLASLt INT8 kernels are named e.g. cutlass_80_tensorop_i16832gemm_s8_128x64_...
            group = "int_gemm"
        elif "flash" in low or "fmha" in low or "attention" in low or "softmax" in low:
            group = "attention_kernels"
        elif "gemm" in low or "cutlass" in low or "xmma" in low or "sm80" in low or "sm90" in low or "ampere" in low:
            group = "float_gemm"
        elif "norm" in low:
            group = "layernorm"
        elif "copy" in low or "cast" in low or "to_copy" in low:
            group = "copy_cast"
        else:
            group = "elementwise_other"
        groups[group] += ms
    top = sorted(totals.items(), key=lambda kv: -kv[1])[:25]
    return dict(groups_ms_per_forward=dict(groups), kernel_sum_ms_per_forward=sum(totals.values()),
                top_kernels_ms_per_forward=top)


def ensure_tuned(models, state, args, mode, batch, scaffold, bank, root):
    """Tune each projection shape on activations captured from the real model at this scaffold."""
    bits, rows = QUANT_MODES[mode], batch * 2 * (args.image_size // 16) ** 2
    model_dtype = torch.float16
    shapes = {}
    # Eager, untuned probe with the same attention/precision so captured activations match
    # what the timed configuration feeds its projections.
    probe = Scaffold(scaffold.name, scaffold.attention, False, scaffold.fp16io, False, scaffold.pure16, False)
    model, _, names, _ = build(models, state, args, mode, batch, probe, bank)
    captured, handles = {}, []
    middle = len(model.blocks) // 2
    for name in names:
        if not name.startswith(f"blocks.{middle}."):
            continue
        layer = model.get_submodule(name)

        def hook(m, inputs, name=name):
            x = inputs[0].reshape(-1, m.in_features)
            if m.io_dtype is not None:
                x = x.to(m.io_dtype)
            captured[name] = x.detach().contiguous().clone()
        handles.append(layer.register_forward_pre_hook(hook))
    x, t, y = make_inputs(batch, args.image_size, args.seed)
    try:
        make_forward(model, model_dtype, args.cfg, probe)(x, t, y)
        torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()
    for name, act in captured.items():
        n, k = state[name + ".weight"].shape
        ident = bank_key(rows, n, k, bits, model_dtype)
        if ident in bank.data["entries"]:
            continue
        w = state[name + ".weight"].cuda().float().contiguous()
        b = state.get(name + ".bias")
        b = None if b is None else b.cuda().float().contiguous()
        print(f"  tuning {ident} on captured {name} input ({act.dtype})", flush=True)
        config, report = tune(act, w, b, bits, root / "tuning", rounds=args.tune_rounds)
        bank.add(rows, n, k, bits, model_dtype, config)
        shapes[ident] = dict(layer=name, input_dtype=str(act.dtype), config=asdict(config),
                             holdout_layer_median_ms=report["holdout"]["timings"]["selected_layer"]["median_ms"],
                             holdout_float_linear_median_ms=report["holdout"]["timings"]["native_float_linear"]["median_ms"])
    del model, captured
    gc.collect()
    torch.cuda.empty_cache()
    return shapes


def run_config(models, state, args, mode, batch, scaffold, bank, root, reference_outputs):
    model, dtype, names, conversions = build(models, state, args, mode, batch, scaffold, bank)
    if scaffold.compile:
        torch._dynamo.reset()
        from torch._dynamo.utils import counters
        counters.clear()
    forward = make_forward(model, dtype, args.cfg, scaffold, args.compile_mode)
    # Uncompiled twin (same precision/attention) for module-hook attribution only.
    eager_twin = make_forward(model, dtype, args.cfg, Scaffold(scaffold.name, scaffold.attention, False,
                                                               scaffold.fp16io, scaffold.tuned, scaffold.pure16, False))
    x, t, y = make_inputs(batch, args.image_size, args.seed)
    result = dict(mode=mode, batch=batch, cfg_model_batch=2 * batch, scaffold=scaffold.name,
                  attention=scaffold.attention, cuda_graphs=scaffold.graphs,
                  quant_io=("fp16" if scaffold.fp16io else "native") if mode in QUANT_MODES else "n/a",
                  int_tiles=("tuned" if scaffold.tuned else "default") if mode in QUANT_MODES else "n/a",
                  precision="pure16" if scaffold.pure16 and mode != "fp32" else "autocast",
                  compiled=scaffold.compile,
                  model_storage_bytes=model_storage(model), conversions=conversions, status="ok")
    try:
        t0 = time.perf_counter()
        eager_out = forward(x, t, y).float()
        torch.cuda.synchronize()
        result["first_call_s_including_compile"] = time.perf_counter() - t0
        if not bool(torch.isfinite(eager_out).all()):
            raise ValueError("Nonfinite forward output")
        if scaffold.compile:
            from torch._dynamo.utils import counters
            result["compile_graph_breaks"] = sum(counters["graph_break"].values())
            result["compile_unique_graphs"] = counters["stats"].get("unique_graphs", None)
        # FP16 runs first in every scaffold (see main), so its same-scaffold output exists.
        fp16_key = (batch,) + scaffold.effective("fp16")
        if mode == "fp16":
            reference_outputs[fp16_key] = eager_out.cpu()
        else:
            result["output_vs_fp16_same_attention"] = metrics(reference_outputs[fp16_key], eager_out.cpu())
        if ("fp32", batch) in reference_outputs:
            result["output_vs_fp32"] = metrics(reference_outputs[("fp32", batch)], eager_out.cpu())
        call = lambda: forward(x, t, y)
        if scaffold.graphs:
            replay, graph = make_graphed(forward, x, t, y)
            graphed_out = replay().float()
            torch.cuda.synchronize()
            result["graph_vs_eager_max_abs"] = (graphed_out - eager_out).abs().max().item()
            call = replay
        torch.cuda.reset_peak_memory_stats()
        result["timing"] = time_calls(call, args.warmup, args.repeats)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["median_ms"] = result["timing"]["median_ms"]
        result["p90_ms"] = result["timing"]["p90_ms"]
        result["ms_per_image_per_step"] = result["median_ms"] / batch
        result["projected_sampling_ms"] = result["median_ms"] * args.steps
        if scaffold.fused and mode in QUANT_MODES:
            result["attribution_eager_ms"] = {}   # fused forward bypasses the per-module hooks
        else:
            result["attribution_eager_ms"] = attribution(model, lambda: eager_twin(x, t, y))
        if args.profile:
            tag = f"{mode}_b{batch}_{scaffold.name}"
            result["kernels_eager"] = kernel_profile(lambda: forward(x, t, y), root / "traces" / f"{tag}.json")
        if args.sampling:
            result["sampling"] = time_sampling(call, x, y, args)
    except Exception as exc:  # keep the sweep going, but record the failure explicitly
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"  FAILED {mode} b{batch} {scaffold.name}: {result['error']}", flush=True)
    finally:
        del model, forward, eager_twin
        if scaffold.compile:
            torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    return result


def fp32_reference(models, state, args, batch):
    """Eager FP32, explicit attention: the accuracy reference for every configuration."""
    ref = Scaffold("fp32_reference")
    model, dtype, _, _ = build(models, state, args, "fp32", batch, ref, None)
    x, t, y = make_inputs(batch, args.image_size, args.seed)
    out = make_forward(model, dtype, args.cfg, ref)(x, t, y).float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def time_sampling(replay, x, y, args):
    """Full DDPM sampling loop (no VAE) driving the graphed forward. First run is warmup."""
    import importlib
    diffusion = importlib.import_module("diffusion").create_diffusion(str(args.steps))
    # Clone: the graph's static output buffer is overwritten by the next replay.
    model_fn = lambda xx, tt, y=None, cfg_scale=None: replay(xx, tt, y).clone()
    samples = []
    for _ in range(args.sampling_repeats + 1):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        diffusion.p_sample_loop(model_fn, x.shape, noise=x.clone(), clip_denoised=False,
                                model_kwargs=dict(y=y, cfg_scale=args.cfg), device=x.device, progress=False)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return summarize(samples[1:])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit-repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--modes", nargs="+", default=["fp16", "int8", "int4"],
                   choices=sorted(FLOAT_MODES) + sorted(QUANT_MODES))
    p.add_argument("--batches", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--scaffolds", nargs="+",
                   default=["sdpa_graph_fp16io_tuned", "sdpa_graph_pure16_tuned",
                            "sdpa_graph_compile_fp16io_tuned", "sdpa_graph_pure16_compile_tuned"])
    p.add_argument("--compile-mode", default="default",
                   help="torch.compile mode (default | max-autotune-no-cudagraphs); graphs come from the 'graph' token")
    p.add_argument("--bank-from", help="Seed the tuning bank from an earlier run on the same GPU/software")
    p.add_argument("--no-fp32-reference", action="store_true", help="Skip the FP32 accuracy reference")
    p.add_argument("--scope", choices=["all", "mlp"], default="all")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=50, help="For projected/measured full-sampling latency")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--tune-rounds", type=int, default=5)
    p.add_argument("--profile", action="store_true", help="Also save an eager kernel trace per configuration")
    p.add_argument("--sampling", action="store_true", help="Also time the full sampling loop (graph scaffolds)")
    p.add_argument("--sampling-repeats", type=int, default=2)
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        p.error("A CUDA GPU is required")
    if "fp16" not in args.modes:
        p.error("Include fp16: it is the baseline for every speedup")
    args.modes = ["fp16"] + [m for m in args.modes if m != "fp16"]  # baseline/reference first
    scaffolds = [Scaffold.parse(s) for s in args.scaffolds]
    if args.sampling and not all(s.graphs for s in scaffolds):
        p.error("--sampling currently supports graph scaffolds only")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "traces").mkdir(exist_ok=True)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    models, _ = load_official(args.dit_repo)
    patch_timestep_embedding(models)
    state = load_checkpoint(args.checkpoint)
    if args.bank_from and not (root / "tuning_bank.json").exists():
        import shutil
        shutil.copy(args.bank_from, root / "tuning_bank.json")
    bank = Bank(root / "tuning_bank.json")  # refuses a bank from a different GPU/software/source
    manifest = dict(config=vars(args), environment=environment(torch.device("cuda")),
                    dit_commit=git_record(args.dit_repo), scaffolds=[asdict(s) for s in scaffolds],
                    timed_region="one forward_with_cfg at batch 2xB (CFG); CUDA events; median of repeats",
                    excluded="construction, conversion, tuning, compilation, warmup, diagnostics",
                    patches="TimestepEmbedder frequency table cached on GPU (bit-identical; all modes)",
                    quantization="symmetric absmax per-token A8, per-channel W8/W4; not calibrated PTQ4DiT",
                    status="running")
    write_json(root / "manifest.json", manifest)

    results, tuning, done = [], {}, {}
    reference_outputs = {}
    with torch.inference_mode():
        for batch in args.batches:
            if not args.no_fp32_reference:
                print(f"[ref ] fp32 b{batch}", flush=True)
                reference_outputs[("fp32", batch)] = fp32_reference(models, state, args, batch)
            for scaffold in scaffolds:
                for mode in args.modes:
                    eff = (mode, batch, scaffold.effective(mode))
                    if eff in done:
                        # Identical work for this mode (e.g. FP16 ignores fp16io/tuned): reuse, label it.
                        copy = dict(done[eff], scaffold=scaffold.name, reused_from=done[eff]["scaffold"])
                        results.append(copy)
                        continue
                    if mode in QUANT_MODES and scaffold.tuned:
                        print(f"[tune] {mode} b{batch} {scaffold.name}", flush=True)
                        tuning.update(ensure_tuned(models, state, args, mode, batch, scaffold, bank, root))
                        write_json(root / "tuning_summary.json", tuning)
                    print(f"[run ] {mode:5s} b{batch:<3d} {scaffold.name}", flush=True)
                    r = run_config(models, state, args, mode, batch, scaffold, bank, root, reference_outputs)
                    done[eff] = r
                    results.append(r)
                    if r["status"] == "ok":
                        print(f"       median {r['median_ms']:.2f} ms  p90 {r['p90_ms']:.2f} ms", flush=True)
                    write_json(root / "results.json", results)

    # Speedups: same scaffold, and vs the fastest FP16 scaffold at the same batch.
    fp16 = {(r["batch"], r["scaffold"]): r["median_ms"] for r in results
            if r["mode"] == "fp16" and r["status"] == "ok"}
    best_fp16 = {}
    for (b, s), ms in fp16.items():
        if b not in best_fp16 or ms < best_fp16[b][0]:
            best_fp16[b] = (ms, s)
    rows = []
    for r in results:
        row = {k: r.get(k) for k in ("mode", "batch", "scaffold", "attention", "cuda_graphs", "quant_io",
                                     "int_tiles", "precision", "compiled", "compile_graph_breaks",
                                     "status", "median_ms", "p90_ms", "ms_per_image_per_step",
                                     "projected_sampling_ms", "peak_allocated_bytes", "model_storage_bytes")}
        if r["status"] == "ok":
            same = fp16.get((r["batch"], r["scaffold"]))
            row["speedup_vs_fp16_same_scaffold"] = same / r["median_ms"] if same else None
            if r["batch"] in best_fp16:
                row["speedup_vs_best_fp16"] = best_fp16[r["batch"]][0] / r["median_ms"]
                row["best_fp16_scaffold"] = best_fp16[r["batch"]][1]
            row["output_rel_l2_vs_fp16"] = r.get("output_vs_fp16_same_attention", {}).get("relative_l2")
            row["output_rel_l2_vs_fp32"] = r.get("output_vs_fp32", {}).get("relative_l2")
            k = r.get("kernels_eager", {}).get("groups_ms_per_forward", {})
            for g, v in k.items():
                row[f"kern_{g}_ms"] = v
            for k, v in r.get("attribution_eager_ms", {}).items():
                row[f"attr_{k}_ms"] = v
            if "sampling" in r:
                row["sampling_median_ms"] = r["sampling"]["median_ms"]
        else:
            row["error"] = r.get("error")
        rows.append(row)
    write_csv(root / "summary.csv", rows)
    write_json(root / "results.json", results)
    manifest["status"] = "ok"
    write_json(root / "manifest.json", manifest)

    print("\nmode   batch scaffold                          median_ms   p90_ms  x_same  x_bestfp16  relL2_fp16  relL2_fp32")
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['mode']:6s} {row['batch']:<5d} {row['scaffold']:33s} FAILED: {row.get('error')}")
            continue
        fmt = lambda v, f: (f % v) if isinstance(v, (int, float)) else "-"
        print(f"{row['mode']:6s} {row['batch']:<5d} {row['scaffold']:33s} {row['median_ms']:9.2f} {row['p90_ms']:8.2f} "
              f"{fmt(row.get('speedup_vs_fp16_same_scaffold'), '%6.2f')}  {fmt(row.get('speedup_vs_best_fp16'), '%9.2f')}  "
              f"{fmt(row.get('output_rel_l2_vs_fp16'), '%10.4f')}  {fmt(row.get('output_rel_l2_vs_fp32'), '%10.4f')}")
    print(f"\nSaved {root/'summary.csv'}")


if __name__ == "__main__":
    main()
