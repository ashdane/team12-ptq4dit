"""Full pretrained DiT forward and matched ancestral-sampling evaluation.

Run in a fresh process with the official facebookresearch/DiT checkout. No random
model fallback and no implicit conversion of calibrated PTQ4DiT checkpoints.
"""
import argparse
import gc
import importlib
import json
import logging
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F

from .benchmark import environment, file_hash
from .eval_utils import (git_record, layer_timings, matched_rng, memory, model_storage,
                         precision, timed, write_csv, write_json)
from .model_layers import TunedLinear, convert, projection_names
from .quantization import metrics
from .tuning import Bank, tune


MODES = ("fp32", "bf16", "fp16", "fake8", "int8", "fake4", "int4")


def load_official(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    models = importlib.import_module("models")
    diffusion = importlib.import_module("diffusion")
    if Path(models.__file__).resolve() != repo / "models.py":
        raise RuntimeError("Wrong models.py imported; use a fresh subprocess")
    # A PTQ4DiT models.py would mutate salience state during inference and is not
    # interchangeable with the official DiT baseline.
    import inspect
    if "calib" in inspect.signature(models.DiT.forward).parameters:
        raise ValueError("Use facebookresearch/DiT here, not PTQ4DiT's modified models.py")
    return models, diffusion


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    if not isinstance(checkpoint, dict) or "final_layer.linear.weight" not in checkpoint:
        raise ValueError("Expected an official DiT state_dict or training checkpoint with ema")
    if not bool(checkpoint["final_layer.linear.weight"].abs().max() > 0):
        raise ValueError("Zero-initialized output layer: refusing an untrained-model experiment")
    if any(not torch.isfinite(v).all() for v in checkpoint.values() if torch.is_tensor(v)):
        raise ValueError("Checkpoint contains nonfinite tensors")
    return checkpoint


def model_forward(model, dtype, x, t, labels, cfg):
    with precision(dtype):
        return model.forward_with_cfg(x, t, labels, cfg_scale=cfg)


def diagnostics(model, names, x, t, labels, args, mode, dtype, state, refdir, out, captures):
    """Full-tensor, per-layer errors in an explicitly untimed pass."""
    handles, rows = [], []
    refdir.mkdir(parents=True, exist_ok=True)
    captured_shapes = set()
    if captures.exists():
        captured_shapes = {p.stem for p in captures.glob("*.pt")}
    captures.mkdir(parents=True, exist_ok=True)

    def hook(name):
        def observe(module, inputs, output):
            a = inputs[0].detach()
            y = output.detach()
            path = refdir / (name + ".pt")
            if mode == "fp32":
                torch.save(y.cpu(), path)
                shape = f"{a.numel()//a.shape[-1]}_{y.shape[-1]}_{a.shape[-1]}"
                if shape not in captured_shapes:
                    torch.save(dict(x=a.reshape(-1, a.shape[-1]).float().cpu(), layer=name,
                                    timestep=int(t[0].item())), captures/(shape+".pt"))
                    captured_shapes.add(shape)
                row = dict(layer=name, input_shape=str(list(a.shape)), output_shape=str(list(y.shape)))
                row.update({"propagated_"+k: v for k, v in metrics(y, y).items()})
            else:
                reference = torch.load(path, map_location="cpu", weights_only=True)
                row = dict(layer=name, input_shape=str(list(a.shape)), output_shape=str(list(y.shape)))
                row.update({"propagated_"+k: v for k, v in metrics(reference, y.cpu()).items()})
            if isinstance(module, TunedLinear):
                # Isolate quantization on THIS candidate layer input from error
                # propagated by all preceding layers.
                w = state[name+".weight"].cuda().float()
                bias = state.get(name+".bias")
                bias = None if bias is None else bias.cuda().float()
                with torch.autocast("cuda", enabled=False):
                    fp = F.linear(a.float(), w, bias)
                fake = module.reference(a)
                row.update({"local_vs_fp32_"+k: v for k, v in metrics(fp, y).items()})
                row.update({"implementation_vs_qdq_"+k: v for k, v in metrics(fake, y).items()})
            rows.append(row)
        return observe

    for name in names:
        handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
    try:
        model_forward(model, dtype, x, t, labels, args.cfg)
    finally:
        for handle in handles:
            handle.remove()
    write_csv(out/"layer_errors.csv", rows)
    write_json(out/"layer_errors.json", rows)


def trajectory(model, diffusion, dtype, latent, labels, args, mode, root, out):
    """Diagnostic trajectory: CPU copies/metrics are excluded from speed runs."""
    selected = {diffusion.timestep_map[i] for i in (0, len(diffusion.timestep_map)//2,
                                                  len(diffusion.timestep_map)-1)}
    case_dir = root/"forward_inputs"
    case_dir.mkdir(exist_ok=True)

    def call(x, t, y, cfg_scale):
        if mode == "fp32" and int(t[0].item()) in selected:
            torch.save(dict(x=x.detach().cpu(), t=t.detach().cpu(), y=y.detach().cpu()),
                       case_dir/f"t{int(t[0].item()):04d}.pt")
        return model_forward(model, dtype, x, t, y, cfg_scale)

    reference = None if mode == "fp32" else torch.load(root/"fp32"/"trajectory.pt", weights_only=True)
    fake_path = root/("fake8" if mode == "int8" else "fake4")/"trajectory.pt"
    fake = torch.load(fake_path, weights_only=True) if mode in ("int8", "int4") else None
    latents, rows, events = [], [], []
    with matched_rng(args.seed + 1):
        iterator = diffusion.p_sample_loop_progressive(
            call, latent.shape, noise=latent.clone(), clip_denoised=False,
            model_kwargs=dict(y=labels, cfg_scale=args.cfg), device=latent.device, progress=False)
        for index in range(args.steps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            value = next(iterator)
            b.record()
            # Comparison is on the conditional half; full CFG batch is saved.
            current = value["sample"].detach().cpu()
            if not bool(torch.isfinite(current).all()):
                raise ValueError(f"Nonfinite latent in {mode} at step {index}")
            latents.append(current)
            row = dict(step=index, training_timestep=diffusion.timestep_map[args.steps-1-index])
            if reference is not None:
                row.update({"vs_fp32_"+k: v for k, v in metrics(reference[index][:args.batch], current[:args.batch]).items()})
            if fake is not None:
                row.update({"vs_same_quantization_fake_"+k: v for k, v in metrics(fake[index][:args.batch], current[:args.batch]).items()})
            rows.append(row)
            events.append((a, b))
    torch.cuda.synchronize()
    for row, (a, b) in zip(rows, events):
        row["instrumented_step_cuda_ms"] = a.elapsed_time(b)
    torch.save(latents, out/"trajectory.pt")
    torch.save(latents[-1][:args.batch], out/"latents.pt")
    write_csv(out/"steps.csv", rows)
    return rows[-1]


def sample_call(model, diffusion, dtype, latent, labels, args):
    with matched_rng(args.seed+1):
        def call(x, t, y, cfg_scale):
            return model_forward(model, dtype, x, t, y, cfg_scale)
        return diffusion.p_sample_loop(call, latent.shape, noise=latent.clone(), clip_denoised=False,
                                       model_kwargs=dict(y=labels, cfg_scale=args.cfg),
                                       device=latent.device, progress=False)


def run_mode(mode, models, diffusion, state, latent, labels, bank, args, root):
    out = root/mode
    out.mkdir(exist_ok=True)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(
        mode, getattr(torch, args.quant_dtype))
    t0 = time.perf_counter()
    model = models.DiT_models[args.model](input_size=args.image_size//8).eval()
    model.load_state_dict(state, strict=True)
    # All modes use the same explicit attention path, avoiding an attention
    # algorithm change during the linear-kernel ablation.
    for block in model.blocks:
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False
    names = projection_names(model, args.scope)
    model = model.to(device="cuda", dtype=dtype)
    conversions = []
    if mode in ("fake8", "int8", "fake4", "int4"):
        bits = int(mode[-1])
        conversions = convert(model, names, state, bank, args.batch*2*(args.image_size//16)**2,
                              bits, dtype, fake=mode.startswith("fake"))
    torch.cuda.synchronize()
    setup_s = time.perf_counter()-t0
    try:
        # Compilation and allocator initialization precede all timed runs.
        ts = torch.full((args.batch*2,), 999, device="cuda", dtype=torch.long)
        for _ in range(args.warmup):
            model_forward(model, dtype, latent, ts, labels, args.cfg)
        torch.cuda.synchronize()
        last_step = trajectory(model, diffusion, dtype, latent, labels, args, mode, root, out)
        cases = sorted((root/"forward_inputs").glob("*.pt"))
        if len(cases) != 3:
            raise RuntimeError("Expected three matched real trajectory forward inputs")
        forward_rows, forward_details = [], []
        for path in cases:
            case = torch.load(path, weights_only=True)
            x, t, y = (case[n].cuda() for n in ("x", "t", "y"))
            call = lambda: model_forward(model, dtype, x, t, y, args.cfg)
            result = call()
            if not bool(torch.isfinite(result).all()):
                raise ValueError(f"Nonfinite forward output in {mode} / {path.stem}")
            torch.save(result.cpu(), out/(path.stem+"_output.pt"))
            reference = result.cpu() if mode == "fp32" else torch.load(root/"fp32"/(path.stem+"_output.pt"), weights_only=True)
            error = metrics(reference, result.cpu())
            same = None
            if mode in ("int8", "int4"):
                fake = torch.load(root/("fake"+mode[-1])/(path.stem+"_output.pt"), weights_only=True)
                same = metrics(fake, result.cpu())
            del result
            stats = timed(call, args.warmup, args.repeats)
            mem = memory(call)
            forward_details.append(dict(case=path.stem, timing=stats, memory=mem,
                                        vs_fp32=error, vs_same_quantization_fake=same))
            forward_rows.append(dict(case=path.stem, median_cuda_ms=stats["cuda_event"]["median_ms"],
                                     median_wall_ms=stats["synchronized_wall"]["median_ms"], **error, **mem))
            if path == cases[len(cases)//2]:
                diagnostics(model, names, x, t, y, args, mode, dtype, state,
                            root/"reference_layer_outputs", out, root/"kernel_inputs")
                write_csv(out/"layer_timings.csv", layer_timings(model, names, call))
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=True, profile_memory=True) as profiler:
                        call()
                        torch.cuda.synchronize()
                    profiler.export_chrome_trace(str(out/"forward_trace.json"))
                    (out/"operators.txt").write_text(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))
                    trace = json.loads((out/"forward_trace.json").read_text())
                    from collections import Counter
                    counts = Counter(e["name"] for e in trace["traceEvents"] if e.get("cat") == "kernel")
                    write_json(out/"kernel_counts.json", dict(counts))
        write_csv(out/"forwards.csv", forward_rows)
        write_json(out/"forwards.json", forward_details)
        del x, t, y, call
        torch.cuda.empty_cache()
        sampler = lambda: sample_call(model, diffusion, dtype, latent, labels, args)
        sample_stats = timed(sampler, warmup=1, repeats=args.sampling_repeats)
        sample_mem = memory(sampler)
        # Verify the clean timed path agrees with its separately logged trajectory.
        clean = sampler()[:args.batch].cpu()
        logged = torch.load(out/"latents.pt", weights_only=True)
        torch.testing.assert_close(clean, logged, rtol=1e-4, atol=1e-4)
        exact_match = bool(torch.equal(clean, logged))
        max_diff = (clean.float() - logged.float()).abs().max().item()
        result = dict(mode=mode, status="ok", dtype=str(dtype), setup_wall_s=setup_s,
                      model_parameter_and_buffer_bytes=model_storage(model), conversions=conversions,
                      forward=forward_details, sampling=sample_stats, sampling_memory=sample_mem,
                      images_per_second=args.batch/(sample_stats["synchronized_wall"]["median_ms"]/1000),
                      final_step=last_step, sampling_vs_logged_exact_match=exact_match,
                      sampling_vs_logged_max_abs_error=max_diff)
        write_json(out/"summary.json", result)
        return result
    finally:
        # Reference/checkpoint tensors live on CPU; retain one GPU model at a time.
        del model
        gc.collect()
        torch.cuda.empty_cache()


def tune_captures(root, state, bank, args):
    dtype = getattr(torch, args.quant_dtype)
    paths = sorted((root/"kernel_inputs").glob("*.pt"))
    if not paths:
        raise RuntimeError("No real-model activation captures were produced")
    rows = []
    for path in paths:
        data = torch.load(path, weights_only=True)
        x = data["x"].cuda().to(dtype).contiguous()
        w = state[data["layer"]+".weight"].cuda().float().contiguous()
        b = state.get(data["layer"]+".bias")
        b = None if b is None else b.cuda().float().contiguous()
        for bits in (8, 4):
            config, report = tune(x, w, b, bits, root/"tuning", rounds=args.tuning_rounds)
            bank.add(x.shape[0], w.shape[0], w.shape[1], bits, dtype, config)
            holdout = report["holdout"]["timings"]
            rows.append(dict(
                key=report["key"], layer=data["layer"], m=x.shape[0], n=w.shape[0], k=w.shape[1],
                bits=bits, dtype=args.quant_dtype, selected=report["selected"],
                backend=config.backend, layout=config.layout,
                bm=config.bm, bn=config.bn, bk=config.bk, warps=config.warps, stages=config.stages,
                selected_core_median_ms=holdout["selected_core"]["median_ms"],
                selected_layer_median_ms=holdout["selected_layer"]["median_ms"],
                native_float_linear_median_ms=holdout["native_float_linear"]["median_ms"],
                original_core_fp32_median_ms=holdout["original_core_fp32_output"]["median_ms"],
            ))
        del x, w, b
        torch.cuda.empty_cache()
    write_csv(root/"tuning"/"summary.csv", rows)


def decode(root, args):
    from diffusers import AutoencoderKL
    from torchvision.utils import save_image
    vae = AutoencoderKL.from_pretrained(args.vae).eval().cuda()
    reference = None
    records = []
    for mode in MODES:
        latents = torch.load(root/mode/"latents.pt", weights_only=True).cuda().float()
        fn = lambda: vae.decode(latents/0.18215).sample
        stats = timed(fn, warmup=2, repeats=5)
        images = fn().float().cpu()
        if not torch.isfinite(images).all():
            raise ValueError(f"Nonfinite decoded images for {mode}")
        image_dir = root/mode/"images"
        image_dir.mkdir(exist_ok=True)
        for i, image in enumerate(images):
            save_image(image, image_dir/f"{i:05d}.png", normalize=True, value_range=(-1, 1))
        save_image(images, root/mode/"grid.png", nrow=min(4, args.batch), normalize=True, value_range=(-1, 1))
        if mode == "fp32":
            reference = images
        records.append(dict(mode=mode, decode_timing=stats, decoded_pixel_error_vs_fp32=metrics(reference, images)))
    write_json(root/"decode.json", dict(vae=args.vae, results=records,
                                        distribution_quality="FID/sFID/IS/precision/recall not estimated from this small batch"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dit-repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="DiT-XL/2")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--batch", type=int, default=4, help="Images; effective CFG model batch is twice this")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--scope", choices=["mlp", "all"], default="mlp")
    p.add_argument("--quant-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--sampling-repeats", type=int, default=3)
    p.add_argument("--tuning-rounds", type=int, default=10)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--decode", action="store_true")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        p.error("A CUDA GPU is required")
    if min(args.batch, args.warmup, args.repeats, args.sampling_repeats, args.tuning_rounds) < 1 or args.steps < 3:
        p.error("Counts must be positive and sampling steps >=3")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/"manifest.json").exists():
        p.error("Choose a fresh output directory")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(root/"evaluation.log")])
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    models, diffusion_module = load_official(args.dit_repo)
    state = load_checkpoint(args.checkpoint)
    diffusion = diffusion_module.create_diffusion(str(args.steps))
    if len(diffusion.timestep_map) != args.steps:
        raise ValueError("Sampler did not produce the requested step count")
    with matched_rng(args.seed):
        z = torch.randn(args.batch, 4, args.image_size//8, args.image_size//8, device="cuda")
    latent = torch.cat([z, z], dim=0)
    labels = torch.cat([torch.arange(args.batch, device="cuda") % 1000,
                        torch.full((args.batch,), 1000, device="cuda", dtype=torch.long)])
    del z
    torch.save(dict(initial_latent=latent.cpu(), labels=labels.cpu(), seed=args.seed,
                    sampler_noise_seed=args.seed+1), root/"inputs.pt")
    manifest = dict(config=vars(args), environment=environment(torch.device("cuda")),
                    checkpoint_sha256=file_hash(args.checkpoint), upstream=git_record(args.dit_repo),
                    source_sha256={f.name: file_hash(f) for f in sorted(Path(__file__).parent.glob("*.py"))},
                    timestep_map=diffusion.timestep_map, modes=list(MODES),
                    quantization="symmetric absmax per-token A8 / per-output W8 or W4; not calibrated PTQ4DiT",
                    quantized_scope=args.scope, attention="explicit timm attention, fused_attn=False in all modes",
                    cfg="official forward_with_cfg, guidance on first three channels",
                    protocol="One GPU model at a time. CPU checkpoint/reference. Timings exclude hooks, profiling, downloads, conversion and tuning.",
                    status="running", completed=[])
    write_json(root/"manifest.json", manifest)
    bank = Bank(root/"tuning_bank.json")
    summaries = []
    try:
        with torch.inference_mode():
            for mode in MODES:
                if mode == "fake8":
                    logging.info("Tuning on saved real model activations")
                    tune_captures(root, state, bank, args)
                logging.info("Evaluating %s", mode)
                result = run_mode(mode, models, diffusion, state, latent, labels, bank, args, root)
                summaries.append(result)
                manifest["completed"].append(mode)
                write_json(root/"manifest.json", manifest)
                write_json(root/"results.json", summaries)
            if args.decode:
                decode(root, args)
        manifest["status"] = "ok"
        write_csv(root/"summary.csv", [dict(mode=r["mode"], dtype=r["dtype"],
                  generation_cuda_ms=r["sampling"]["cuda_event"]["median_ms"],
                  generation_wall_ms=r["sampling"]["synchronized_wall"]["median_ms"],
                  images_per_second=r["images_per_second"],
                  model_storage_bytes=r["model_parameter_and_buffer_bytes"],
                  peak_allocated_bytes=r["sampling_memory"]["peak_allocated_bytes"],
                  final_latent_relative_l2=r["final_step"].get("vs_fp32_relative_l2", 0)) for r in summaries])
    except Exception:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        logging.exception("Evaluation stopped; completed artifacts are preserved")
        raise
    finally:
        write_json(root/"manifest.json", manifest)
    logging.info("Complete: %s", root)


if __name__ == "__main__":
    main()
