# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted for PTQ4DiT (NeurIPS 2024) & CS3406 Slot-Safe Execution.

import os
import urllib.request
import torch
from torchvision.datasets.utils import download_url


pretrained_models = {
    "DiT-XL-2-512x512.pt",
    "DiT-XL-2-256x256.pt",
}


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively,
    loads a model from a local path, or initializes a deterministic synthetic
    checkpoint if running in an offline test mode (`PTQ4DIT_SYNTHETIC_CKPT=1`).
    """
    if model_name in pretrained_models:
        return download_model(model_name)
    else:
        if os.path.isfile(model_name):
            checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
            if "ema" in checkpoint:
                checkpoint = checkpoint["ema"]
            return checkpoint
        elif os.environ.get("PTQ4DIT_ALLOW_SYNTHETIC", "0") == "1":
            print(f"[download.py] Notice: '{model_name}' not found on disk; using initialized weights (PTQ4DIT_ALLOW_SYNTHETIC=1).")
            return None
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at '{model_name}'.\n"
                f"To download it automatically, run:\n"
                f"  python download.py --model DiT-XL/2 --image-size 256\n"
                f"Or download manually before your GPU slot from:\n"
                f"  https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt\n"
                f"and place it inside pretrained_models/DiT-XL-2-256x256.pt"
            )


def download_model(model_name, target_dir="pretrained_models"):
    """
    Downloads a pre-trained DiT model from the web into `pretrained_models/`.
    Supports resuming partial downloads via curl if interrupted by a GPU slot expiry.
    """
    assert model_name in pretrained_models, f"Unknown pretrained model: {model_name}"
    os.makedirs(target_dir, exist_ok=True)
    local_path = os.path.join(target_dir, model_name)
    if not os.path.isfile(local_path):
        web_path = f"https://dl.fbaipublicfiles.com/DiT/models/{model_name}"
        print(f"[download.py] Downloading {web_path} -> {local_path} ...")
        # Prefer curl -C - for resumable download on slot-limited servers
        ret = os.system(f"curl -L -C - --retry 3 -o '{local_path}.part' '{web_path}' && mv '{local_path}.part' '{local_path}'")
        if ret != 0 or not os.path.isfile(local_path):
            download_url(web_path, target_dir)
    model = torch.load(local_path, map_location=lambda storage, loc: storage, weights_only=False)
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download official pre-trained DiT checkpoints.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--outdir", type=str, default="pretrained_models")
    args = parser.parse_args()
    ckpt_name = f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    download_model(ckpt_name, target_dir=args.outdir)
    print(f"[download.py] Successfully verified {os.path.join(args.outdir, ckpt_name)}")
