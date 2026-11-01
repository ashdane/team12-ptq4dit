"""
PTQ4DiT Quantization & Sampling Script (`quant_sample.py`)
----------------------------------------------------------
Matches the exact CLI of `adreamwu/PTQ4DiT`:
  python quant_sample.py --model DiT-XL/2 --image-size 256 \
    --ckpt pretrained_models/DiT-XL-2-256x256.pt --num-sampling-steps 50 \
    --weight_bit 8 --act_bit 8 --cali_st 25 --cali_n 64 --cali_batch_size 32 --sm_abit 8 \
    --cali_data_path calib/imagenet_DiT-256_sample4000_50steps_allst.pt --outdir output/ \
    --cfg-scale 1.5 --seed 1 --ptq --recon

Added features for CS3406 AI Accelerator Project:
  - `--fast-reparam`: Runs offline Re-Parameterization (Eqs. 12-21, completing the
    unfinished TODO in the official PTQ4DiT repo) to eliminate runtime channel
    salience balancing overhead.
  - Automatic GPU latency & peak VRAM logging on every sampling run.
  - Slot-safe atomic checkpointing (`output/<size>_<w><a >_<steps>/ckpt.pth`).
"""

import os
import time
import argparse
import torch
from torchvision.utils import save_image
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from quant import QuantModel, calibrate_and_reconstruct_blocks
from fast_inference_reparam import apply_offline_reparameterization
from utils.slot_guard import SlotGuard, atomic_torch_save


def decode_latents_to_rgb(latents: torch.Tensor) -> torch.Tensor:
    """
    Decodes 4-channel DiT latents (B, 4, H/8, W/8) to RGB images in [0, 1].
    Uses HuggingFace `stabilityai/sd-vae-ft-mse` if cached locally; otherwise uses
    the standard SD-VAE linear latent-to-RGB approximation matrix so inference
    never stalls or fails if HuggingFace Hub is unreachable during a GPU slot.
    """
    vae_local_dir = os.environ.get("SD_VAE_LOCAL_DIR", "pretrained_models/sd-vae-ft-mse")
    if os.path.isdir(vae_local_dir):
        try:
            from diffusers.models import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(vae_local_dir, local_files_only=True).to(latents.device)
            vae.eval()
            with torch.no_grad():
                imgs = vae.decode(latents / 0.18215).sample
                return torch.clamp(0.5 * imgs + 0.5, 0.0, 1.0)
        except Exception as e:
            print(f"[quant_sample] Local VAE decode fallback ({e}); using fast latent-to-RGB projection.")

    # Standard SD 1.x 4-channel latent -> RGB linear projection + bilinear upsample
    v1_4_rgb_factors = torch.tensor(
        [
            [0.298, 0.207, 0.208],
            [0.187, 0.286, 0.173],
            [-0.158, 0.189, 0.264],
            [-0.184, -0.271, -0.473],
        ],
        device=latents.device,
        dtype=latents.dtype,
    )
    rgb = torch.einsum("bchw,cd->bdhw", latents, v1_4_rgb_factors)
    rgb = torch.nn.functional.interpolate(rgb, scale_factor=8, mode="bilinear", align_corners=False)
    rgb = torch.clamp((rgb + 1.0) * 0.5, 0.0, 1.0)
    return rgb


def main():
    parser = argparse.ArgumentParser(description="PTQ4DiT Calibration & Quantized Inference")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="output/")

    # PTQ4DiT quantization flags (identical to official repo README)
    parser.add_argument("--ptq", action="store_true", help="Enable Post-Training Quantization")
    parser.add_argument("--weight_bit", type=int, default=8)
    parser.add_argument("--act_bit", type=int, default=8)
    parser.add_argument("--sm_abit", type=int, default=8)
    parser.add_argument("--cali_st", type=int, default=25, help="Number of calibration timesteps")
    parser.add_argument("--cali_n", type=int, default=64, help="Number of calibration samples per step")
    parser.add_argument("--cali_batch_size", type=int, default=32)
    parser.add_argument("--cali_data_path", type=str, default="calib/imagenet_DiT-256_sample4000_50steps_allst.pt")
    parser.add_argument("--recon", action="store_true", help="Enable block-wise scale calibration")
    parser.add_argument("--recon-iters", type=int, default=25, help="Iterations per block during reconstruction")
    parser.add_argument("--no-ssc", action="store_true", help="Disable Spearman rho calibration (CSB-only ablation)")
    parser.add_argument("--resume", action="store_true", help="Resume from calibrated checkpoint")
    parser.add_argument("--cali_ckpt", type=str, default=None, help="Path to calibrated checkpoint")
    parser.add_argument("--inference", action="store_true", help="Run quantized image generation")
    parser.add_argument("--n_c", type=int, default=1, help="Samples per class batch to generate during inference")
    parser.add_argument("--num-sample-classes", type=int, default=8, help="Number of classes to sample for quick evaluation")
    parser.add_argument("--fast-reparam", action="store_true", help="Apply offline Re-Parameterization (Section 4.3) for fast inference")
    parser.add_argument("--slot-minutes", type=float, default=120.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_tag = f"{args.image_size}_{args.weight_bit}{args.act_bit}_{args.num_sampling_steps}"
    exp_dir = os.path.join(args.outdir, exp_tag)
    os.makedirs(exp_dir, exist_ok=True)
    default_ckpt_out = os.path.join(exp_dir, "ckpt.pth")

    slot_guard = SlotGuard(slot_minutes=args.slot_minutes, checkpoint_dir=exp_dir)

    latent_size = args.image_size // 8
    base_model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    if state_dict is not None:
        base_model.load_state_dict(state_dict)
    base_model.eval()

    if args.ptq:
        qnn = QuantModel(base_model, weight_bit=args.weight_bit, act_bit=args.act_bit).to(device)
        qnn.eval()

        resume_path = args.cali_ckpt or (default_ckpt_out if args.resume else None)
        if args.resume and resume_path and os.path.isfile(resume_path) and args.inference:
            print(f"[quant_sample] Loading calibrated PTQ4DiT checkpoint from {resume_path} ...")
            ckpt_data = torch.load(resume_path, map_location=device, weights_only=False)
            state_to_load = ckpt_data["model_state"] if "model_state" in ckpt_data else ckpt_data
            qnn.load_state_dict(state_to_load, strict=False)
            qnn.set_quant_state(weight_quant=True, act_quant=True)
        else:
            if not os.path.isfile(args.cali_data_path):
                raise FileNotFoundError(
                    f"Calibration dataset not found at '{args.cali_data_path}'. "
                    f"Run get_calibration_set.py first!"
                )
            raw_cali = torch.load(args.cali_data_path, map_location="cpu", weights_only=False)
            total_t = raw_cali["xs"].shape[0]
            stride = max(1, total_t // min(total_t, args.cali_st))
            t_indices = list(range(0, total_t, stride))[: args.cali_st]
            n_sub = min(raw_cali["xs"].shape[1], args.cali_batch_size)
            cali_subset = {
                "xs": raw_cali["xs"][t_indices, :n_sub],
                "ts": raw_cali["ts"][t_indices, :n_sub],
                "ys": raw_cali["ys"][t_indices, :n_sub],
            }
            calibrate_and_reconstruct_blocks(
                qnn,
                cali_data=cali_subset,
                use_ssc=(not args.no_ssc),
                do_recon=args.recon,
                recon_iters=args.recon_iters,
                resume_ckpt_path=resume_path,
                save_ckpt_path=default_ckpt_out,
                slot_guard=slot_guard,
            )
            qnn.set_quant_state(weight_quant=True, act_quant=True)

        if args.fast_reparam:
            print("[quant_sample] Applying PTQ4DiT Offline Re-Parameterization (Section 4.3 & Appendix D)...")
            apply_offline_reparameterization(qnn)

        active_model = qnn
    else:
        active_model = base_model

    if args.inference or not args.ptq:
        torch.set_grad_enabled(False)
        diffusion = create_diffusion(str(args.num_sampling_steps))
        # Deterministic ImageNet class labels (e.g., golden retriever, volcano, macaw, etc.)
        default_classes = [207, 360, 387, 974, 88, 979, 417, 279][: args.num_sample_classes]
        all_samples = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for round_idx in range(args.n_c):
            n = len(default_classes)
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = torch.tensor(default_classes, device=device)
            z_cfg = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            y_cfg = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_cfg, cfg_scale=args.cfg_scale)

            samples = diffusion.p_sample_loop(
                active_model.forward_with_cfg,
                z_cfg.shape,
                z_cfg,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device,
            )
            samples, _ = samples.chunk(2, dim=0)
            all_samples.append(samples)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mem_mb = (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            if torch.cuda.is_available()
            else 0.0
        )

        latents_cat = torch.cat(all_samples, dim=0)
        imgs = decode_latents_to_rgb(latents_cat)
        mode_str = "reparam" if args.fast_reparam else ("ptq" if args.ptq else "fp")
        img_out_path = os.path.join(exp_dir, f"sample_{mode_str}_W{args.weight_bit}A{args.act_bit}.png")
        save_image(imgs, img_out_path, nrow=4)
        atomic_torch_save(latents_cat.cpu(), os.path.join(exp_dir, f"latents_{mode_str}.pt"))

        total_imgs = latents_cat.shape[0]
        print(
            f"[quant_sample] Generated {total_imgs} images in {elapsed:.2f}s "
            f"({elapsed / total_imgs:.3f}s/img, Peak VRAM: {peak_mem_mb:.1f} MB) -> {img_out_path}"
        )


if __name__ == "__main__":
    main()
