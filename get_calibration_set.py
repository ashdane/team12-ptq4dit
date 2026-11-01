"""
Calibration Dataset Generator for PTQ4DiT (`get_calibration_set.py`)
--------------------------------------------------------------------
Matches the official CLI of `adreamwu/PTQ4DiT`:
  python get_calibration_set.py --model DiT-XL/2 --image-size 256 \
    --ckpt pretrained_models/DiT-XL-2-256x256.pt --num-sampling-steps 50 \
    --outdir calib/ --filename imagenet_DiT-256_sample4000_50steps_allst.pt \
    --cfg-scale 1.5 --seed 1

Enhanced with CS3406 Slot-Safe Incremental Checkpointing:
  - Saves progress after every batch via `atomic_torch_save`.
  - Automatically resumes if interrupted by a GPU slot expiration.
"""

import os
import argparse
import torch
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from utils.slot_guard import SlotGuard, atomic_torch_save


def main():
    parser = argparse.ArgumentParser(description="Generate multi-timestep calibration dataset for PTQ4DiT.")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a DiT checkpoint.")
    parser.add_argument("--outdir", type=str, default="calib/")
    parser.add_argument("--filename", type=str, default="imagenet_DiT-256_sample4000_50steps_allst.pt")
    parser.add_argument("--batch-size", type=int, default=16, help="Samples per batch.")
    parser.add_argument("--num-batches", type=int, default=2, help="Number of batches to collect (default 2 -> 32 samples across all timesteps).")
    parser.add_argument("--slot-minutes", type=float, default=120.0, help="CS3406 GPU slot duration in minutes.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[get_calibration_set] Using device: {device}")

    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, args.filename)
    partial_path = f"{save_path}.partial"

    slot_guard = SlotGuard(slot_minutes=args.slot_minutes, checkpoint_dir=args.outdir)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Check if already completed or partially completed
    if os.path.isfile(save_path):
        print(f"[get_calibration_set] Calibration file already exists at {save_path}. Skipping generation.")
        return

    collected_xs = []
    collected_ts = []
    collected_ys = []
    start_batch = 0

    if os.path.isfile(partial_path):
        part = torch.load(partial_path, map_location="cpu", weights_only=False)
        collected_xs = part["xs"]
        collected_ts = part["ts"]
        collected_ys = part["ys"]
        start_batch = part["completed_batches"]
        print(f"[get_calibration_set] Resumed partial calibration from batch {start_batch}/{args.num_batches}.")

    def _flush_partial():
        if len(collected_xs) > 0:
            atomic_torch_save(
                {
                    "xs": collected_xs,
                    "ts": collected_ts,
                    "ys": collected_ys,
                    "completed_batches": len(collected_xs),
                },
                partial_path,
            )

    slot_guard.register_emergency_save(_flush_partial)

    for b_idx in range(start_batch, args.num_batches):
        if slot_guard.should_stop_safely():
            _flush_partial()
            return

        n = args.batch_size
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance
        z_cfg = torch.cat([z, z], 0)
        y_null = torch.tensor([args.num_classes] * n, device=device)
        y_cfg = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y_cfg, cfg_scale=args.cfg_scale)

        step_xs = []
        step_ts = []
        step_ys = []

        # Collect trajectory across all sampling timesteps
        indices = list(range(diffusion.num_timesteps))[::-1]
        img = z_cfg
        for i in indices:
            t = torch.tensor([i] * img.shape[0], device=device)
            scaled_t = diffusion._scale_timesteps(t)
            step_xs.append(img[:n].detach().cpu())
            step_ts.append(scaled_t[:n].detach().cpu())
            step_ys.append(y.detach().cpu())
            out = diffusion.p_sample(
                model.forward_with_cfg,
                img,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            img = out["sample"]

        # Stack over timesteps: (T, B, ...)
        collected_xs.append(torch.stack(step_xs, dim=0))
        collected_ts.append(torch.stack(step_ts, dim=0))
        collected_ys.append(torch.stack(step_ys, dim=0))
        _flush_partial()
        print(f"  [get_calibration_set] Completed batch {b_idx + 1}/{args.num_batches} ({args.num_sampling_steps} steps)")

    # Concatenate across batches along dim=1 -> shape: (T, Total_N, ...)
    final_data = {
        "xs": torch.cat(collected_xs, dim=1),
        "ts": torch.cat(collected_ts, dim=1),
        "ys": torch.cat(collected_ys, dim=1),
        "num_sampling_steps": args.num_sampling_steps,
        "image_size": args.image_size,
    }
    atomic_torch_save(final_data, save_path)
    if os.path.isfile(partial_path):
        os.remove(partial_path)
    print(
        f"[get_calibration_set] Saved calibration dataset -> {save_path} "
        f"(xs shape: {tuple(final_data['xs'].shape)})"
    )


if __name__ == "__main__":
    main()
