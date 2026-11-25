"""PTQ4DiT Channel-wise Salience Balancing (CSB) + Spearman-guided Salience Calibration (SSC).

Faithful to the official implementation (github.com/adreamwu/PTQ4DiT, commit fc27e977):
  * calibration trajectories as in get_calibration_set.py: seed 1, CFG 1.5, 50 DDPM steps,
    class c -> batch [z1, z2, z1, z2] with labels [c, c, null, null]; the model *input* x and the
    mapped (0..999) timestep are logged at every step;
  * initialisation batch as in quant_sample.py (cali_st=25, n_per_t=4): steps range(0, 50, 2),
    first 4 samples per step (= class 0), conditional rows first, then null rows; run through
    forward_with_cfg in FP32 with TF32 enabled and quantisation off;
  * models.py Attention/Mlp: for qkv, proj and fc1 (fc2 is NOT balanced)
        w_scale = |W|.max(dim=0)                      # per input channel
        x_scale = |x|.max(dim=1)                      # per sample, max over tokens -> [B, C]
        tau_i   = spearmanr(x_scale[i], w_scale)
        x_scale = sum_i softmax(-tau)_i * x_scale[i]  # SSC
        s       = sqrt(x_scale / (w_scale + 1e-8))    # CSB
        W <- W * s ;  x <- x / s

Differences from the original, by design (see README of this repo for the rationale):
  * x / s is folded offline (the original divides at runtime; its README lists the
    "re-parameterization for fast inference" as unreleased):
      qkv/fc1 inputs: into the adaLN shift/scale rows feeding modulate(); proj input: into the
      V rows of qkv (attention is linear in V). Folding is exact in exact arithmetic.
  * No BRECQ/AdaRound reconstruction, and our quantizers (dynamic per-token A8, per-channel
    absmax W8/W4, real INT kernels) instead of their static per-tensor asymmetric MSE
    activation quantizers and quantised attention matmuls.

Commands:
  calibrate  generate the calibration batch and compute s for every balanced layer
  fold       check that the folded FP32 model reproduces the original output
  evaluate   paired accuracy vs FP32 (full sampling, matched noise) with and without CSB
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .eval_utils import git_record, matched_rng, write_json
from .full_model import load_checkpoint, load_official

BALANCED = ("attn.qkv", "attn.proj", "mlp.fc1")   # fc2 is not balanced in the official code
CALI_ST, N_PER_T, STEPS, CFG, SEED = 25, 4, 50, 1.5, 1


# --------------------------------------------------------------------------- calibration data
def calibration_trajectory(model, diffusion, image_size, classes, seed=SEED, cfg=CFG):
    """get_calibration_set.py: log (model input x, mapped t, y) at every sampling step."""
    latent = image_size // 8
    torch.manual_seed(seed)   # CUDA generator state matches the original (model init uses CPU RNG)
    logs = []
    for c in classes:
        z = torch.randn(2, 4, latent, latent, device="cuda")
        z = torch.cat([z, z], 0)
        y = torch.tensor([c, c, 1000, 1000], device="cuda")
        steps = []

        def model_fn(x, t, y=None, cfg_scale=None):
            steps.append((x.detach().cpu(), t.detach().cpu()))
            return model.forward_with_cfg(x, t, y, cfg_scale)
        diffusion.p_sample_loop(model_fn, z.shape, z, clip_denoised=False,
                                model_kwargs=dict(y=y, cfg_scale=cfg), progress=False, device="cuda")
        logs.append(dict(xs=torch.stack([s[0] for s in steps]), ts=torch.stack([s[1] for s in steps]),
                         y=y.cpu()))
    xs = torch.cat([l["xs"] for l in logs], dim=1)                    # [steps, 4*classes, 4, h, w]
    ts = torch.cat([l["ts"] for l in logs], dim=1)
    ys = torch.cat([l["y"].unsqueeze(0).repeat(xs.shape[0], 1) for l in logs], dim=1)
    return xs, ts, ys


def init_batch(xs, ts, ys, cali_st=CALI_ST, n_per_t=N_PER_T):
    """quant_sample.py lines 95-117 (get_train_samples + first n_per_t per timestep + reorder)."""
    nsteps = xs.shape[0]
    chosen = list(range(0, nsteps, nsteps // cali_st))
    bx = torch.cat([xs[i][:n_per_t] for i in chosen])
    bt = torch.cat([ts[i][:n_per_t] for i in chosen])
    by = torch.cat([ys[i][:n_per_t] for i in chosen])
    normal, null = torch.where(by != 1000)[0], torch.where(by == 1000)[0]
    order = torch.cat([normal, null])
    return bx[order], bt[order], by[order], chosen


# --------------------------------------------------------------------------- scales
def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def compute_scales(model, batch, cfg=CFG):
    """Run the official forward once (FP32, quant off) and compute s per balanced layer."""
    names = [f"blocks.{i}.{s}" for i in range(len(model.blocks)) for s in BALANCED]
    salience, handles = {}, []
    for name in names:
        def hook(module, inputs, name=name):
            salience[name] = inputs[0].detach().abs().amax(dim=1).float()   # [B, C]
        handles.append(model.get_submodule(name).register_forward_pre_hook(hook))
    x, t, y = (v.cuda() for v in batch)
    try:
        model.forward_with_cfg(x, t, y, cfg)
        torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()
    scales, details = {}, {}
    for name in names:
        w = model.get_submodule(name).weight.detach().float()
        w_scale = w.abs().max(dim=0)[0]
        x_scale = salience[name]
        tau = torch.tensor([spearman(x_scale[i].cpu().numpy(), w_scale.cpu().numpy())
                            for i in range(x_scale.shape[0])], device=x_scale.device)
        weights = F.softmax(-tau, dim=0)
        x_bal = (x_scale * weights.view(-1, 1)).sum(dim=0).float()
        s = (x_bal / (w_scale + 1e-8)) ** 0.5
        if not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
            raise ValueError(f"Non-finite or non-positive CSB scale in {name}")
        scales[name] = s.cpu()
        details[name] = dict(s_min=s.min().item(), s_max=s.max().item(), s_median=s.median().item(),
                             tau_mean=tau.mean().item(), tau_min=tau.min().item(), tau_max=tau.max().item(),
                             ssc_weight_max=weights.max().item())
    return scales, details


# --------------------------------------------------------------------------- folding
def fold(state, scales, hidden=None):
    """Return a new FP32 state dict with x/s folded into producers and W*s into consumers."""
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}
    blocks = sorted({int(k.split(".")[1]) for k in scales})
    for i in blocks:
        p = f"blocks.{i}"
        aw, ab = out[f"{p}.adaLN_modulation.1.weight"].float(), out[f"{p}.adaLN_modulation.1.bias"].float()
        C = hidden or aw.shape[1]
        # adaLN chunks: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        for layer, (shift, scale) in ((f"{p}.attn.qkv", (0, 1)), (f"{p}.mlp.fc1", (3, 4))):
            if layer not in scales:
                continue
            s = scales[layer].float()
            sh, sc = slice(shift * C, (shift + 1) * C), slice(scale * C, (scale + 1) * C)
            # modulate(x, shift, scale) / s = x*((1+scale)/s) + shift/s
            aw[sh] /= s[:, None]
            ab[sh] /= s
            aw[sc] /= s[:, None]
            ab[sc] = ab[sc] / s + 1.0 / s - 1.0
            out[f"{layer}.weight"] = out[f"{layer}.weight"].float() * s[None, :]
        out[f"{p}.adaLN_modulation.1.weight"], out[f"{p}.adaLN_modulation.1.bias"] = aw, ab
        proj = f"{p}.attn.proj"
        if proj in scales:
            s = scales[proj].float()
            qw, qb = out[f"{p}.attn.qkv.weight"].float(), out[f"{p}.attn.qkv.bias"].float()
            v = slice(2 * C, 3 * C)            # timm: reshape(B,N,3,H,D) -> v channel c = row 2C+c
            qw[v] /= s[:, None]
            qb[v] /= s
            out[f"{p}.attn.qkv.weight"], out[f"{p}.attn.qkv.bias"] = qw, qb
            out[f"{proj}.weight"] = out[f"{proj}.weight"].float() * s[None, :]
    return out


def load_folded(state, path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    return fold(state, data["scales"]), hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- commands
def fp32_model(models, state, image_size):
    model = models.DiT_models["DiT-XL/2"](input_size=image_size // 8).eval()
    model.load_state_dict(state, strict=True)
    return model.cuda()


def cmd_calibrate(args):
    models, diffusion_module = load_official(args.dit_repo)
    state = load_checkpoint(args.checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = True     # as in the official scripts
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.perf_counter()
    with torch.no_grad():
        model = fp32_model(models, state, args.image_size)
        diffusion = diffusion_module.create_diffusion(str(STEPS))
        xs, ts, ys = calibration_trajectory(model, diffusion, args.image_size, range(args.classes))
        bx, bt, by, chosen = init_batch(xs, ts, ys)
        scales, details = compute_scales(model, (bx, bt, by))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(scales=scales, init_t=bt, init_y=by, chosen_steps=chosen), out)
    write_json(out.with_suffix(".json"), dict(
        method="PTQ4DiT CSB+SSC (official formulas, commit fc27e977); folded offline",
        balanced_layers=list(BALANCED), cali_st=CALI_ST, n_per_t=N_PER_T, steps=STEPS, cfg=CFG, seed=SEED,
        classes=args.classes, init_batch_rows=int(bx.shape[0]), init_timesteps=bt.tolist(),
        tf32=True, wall_s=time.perf_counter() - t0, dit_commit=git_record(args.dit_repo),
        scales_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), layers=details))
    print(f"Saved {len(scales)} CSB scale vectors to {out} ({time.perf_counter()-t0:.0f}s)")


def cmd_fold(args):
    """Folded FP32 model must reproduce the unfolded FP32 output (TF32 off)."""
    models, _ = load_official(args.dit_repo)
    state = load_checkpoint(args.checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folded, digest = load_folded(state, args.scales)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(8, 4, args.image_size // 8, args.image_size // 8, generator=g).cuda()
    t = torch.tensor([999, 999, 500, 500, 100, 100, 20, 20]).cuda()
    y = torch.tensor([1, 2, 3, 4, 1000, 1000, 1000, 1000]).cuda()
    with torch.no_grad():
        a = fp32_model(models, state, args.image_size)(x, t, y)
        b = fp32_model(models, folded, args.image_size)(x, t, y)
    rel = ((a - b).norm() / a.norm()).item()
    print(json.dumps(dict(folded_vs_original_relative_l2=rel, max_abs=(a - b).abs().max().item(),
                          scales_sha256=digest)))
    if rel > 1e-4:
        raise SystemExit("Folding is not output-preserving; refusing to use these scales")


def cmd_evaluate(args):
    """Paired accuracy: same noise, full sampling; INT modes with/without CSB vs FP32."""
    from .latency_bench import Scaffold, build, make_forward, patch_timestep_embedding
    from .generate import DefaultBank
    models, diffusion_module = load_official(args.dit_repo)
    patch_timestep_embedding(models)
    state = load_checkpoint(args.checkpoint)
    folded, digest = load_folded(state, args.scales)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    diffusion = diffusion_module.create_diffusion(str(STEPS))
    ns = SimpleNamespace(image_size=args.image_size, scope="all")
    n, latent = args.images, args.image_size // 8
    g = torch.Generator().manual_seed(args.seed)
    z = torch.randn(n, 4, latent, latent, generator=g).cuda()
    labels = torch.randint(0, 1000, (n,), generator=g).cuda()
    noise = torch.cat([z, z])
    y = torch.cat([labels, torch.full_like(labels, 1000)])
    v3 = args.v3 or args.tattn
    fused = args.fused or v3 or bool(args.v4)
    scaffold = Scaffold("sdpa_pure16" + ("_fused" if fused else "") + ("_v3" if v3 else "")
                        + ("_tattn" if args.tattn else "") + (f"_{args.v4}" if args.v4 else ""), attention="sdpa",
                        fp16io=True, pure16=True, fused=fused, v3=v3, tattn=args.tattn, v4=args.v4 or "")

    def sample(model, dtype, sc):
        fwd = make_forward(model, dtype, CFG, sc)
        fn = lambda x, t, y=None, cfg_scale=None: fwd(x, t, y).float()
        with matched_rng(args.seed + 1):
            out = diffusion.p_sample_loop(fn, noise.shape, noise=noise.clone(), clip_denoised=False,
                                          model_kwargs=dict(y=y, cfg_scale=CFG), device="cuda", progress=False)
        return out[:n].float().cpu()

    rows, ref = [], None
    configs = [("fp32", False), ("fp16", False)] + [(m, c) for m in args.modes
                                                     for c in ((False,) if args.no_csb else (False, True))]
    with torch.inference_mode():
        for mode, use_csb in configs:
            st = folded if use_csb else state
            sc = Scaffold("fp32_ref") if mode == "fp32" else scaffold
            model, dtype, _, _ = build(models, st, ns, mode, n, sc, DefaultBank())
            t0 = time.perf_counter()
            final = sample(model, dtype, sc)
            del model
            torch.cuda.empty_cache()
            if mode == "fp32":
                ref = final
            rel = ((final - ref).norm() / ref.norm()).item()
            per_image = ((final - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1))
            row = dict(mode=mode, csb=use_csb, final_latent_rel_l2_vs_fp32=rel,
                       per_image_rel_l2_median=per_image.median().item(),
                       per_image_rel_l2_max=per_image.max().item(), wall_s=time.perf_counter() - t0)
            if args.decode:
                row.update(decode_psnr(final, ref))
            rows.append(row)
            print(json.dumps(row), flush=True)
    write_json(Path(args.output), dict(images=n, seed=args.seed, scales_sha256=digest,
                                       scaffold="sdpa, pure FP16 (FP32 reference: explicit attention, FP32)",
                                       note="Paired, matched noise; latent-space and decoded-pixel error vs FP32",
                                       results=rows))


_VAE = {}


def decode_psnr(latents, ref_latents):
    from diffusers import AutoencoderKL
    if "vae" not in _VAE:
        _VAE["vae"] = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").eval().cuda()
    vae = _VAE["vae"]
    dec = lambda l: vae.decode(l.cuda() / 0.18215).sample.clamp(-1, 1).float().cpu()
    a, b = dec(latents), dec(ref_latents)
    mse = ((a - b) ** 2).flatten(1).mean(1)
    psnr = 10 * torch.log10(4.0 / mse.clamp_min(1e-12))       # pixel range [-1, 1] -> peak^2 = 4
    return dict(decoded_psnr_db_median=psnr.median().item(), decoded_psnr_db_min=psnr.min().item())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "fold", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--dit-repo", required=True)
        s.add_argument("--checkpoint", required=True)
        s.add_argument("--image-size", type=int, choices=[256, 512], default=256)
        if name == "calibrate":
            s.add_argument("--output", required=True)
            s.add_argument("--classes", type=int, default=1,
                           help="Classes in the calibration trajectories (official init batch uses class 0 only)")
        else:
            s.add_argument("--scales", required=True)
        if name == "evaluate":
            s.add_argument("--output", required=True)
            s.add_argument("--modes", nargs="+", default=["int8", "int4"], choices=["int8", "int4"])
            s.add_argument("--images", type=int, default=16)
            s.add_argument("--seed", type=int, default=2)
            s.add_argument("--decode", action="store_true")
            s.add_argument("--fused", action="store_true",
                           help="INT modes use the hand-fused block forward (fused_ops.py, fast quantization)")
            s.add_argument("--v3", action="store_true",
                           help="INT modes use the v3 fused block (fused_gemm.py epilogues); implies --fused")
            s.add_argument("--tattn", action="store_true",
                           help="v3 block with the single-pass Triton attention (fused_attn.py); implies --v3")
            s.add_argument("--v4", choices=["v4", "v4w", "v4c"],
                           help="INT modes use a v4 CUTLASS fused block variant")
            s.add_argument("--no-csb", action="store_true", help="Skip the CSB-folded variants")
    args = p.parse_args(argv)
    torch.set_grad_enabled(False)
    {"calibrate": cmd_calibrate, "fold": cmd_fold, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
