#!/usr/bin/env bash
# Phase-0 latency benchmark on a Colab GPU VM.
# Usage (on the VM, from the repo root):
#   bash scripts/latency_on_colab.sh <output_dir> [extra fc1bench.latency_bench args...]
# Installs model deps, fetches DiT code + weights (no ImageNet reference / ADM env needed),
# then runs the benchmark with live output.
set -euo pipefail
cd "$(dirname "$0")/.."
ASSETS=/content/dit_pretrained_eval/assets
OUT=${1:-/content/latency/run_$(date +%Y%m%d_%H%M%S)}
shift || true

python3 -m pip install -q -r requirements-colab.txt
python3 scripts/setup_pretrained.py --assets "$ASSETS" --image-size 256 --skip-reference
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv

PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 python3 -m fc1bench.latency_bench \
  --dit-repo "$ASSETS/DiT" \
  --checkpoint "$ASSETS/pretrained_models/DiT-XL-2-256x256.pt" \
  --output "$OUT" "$@"
