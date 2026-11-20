#!/usr/bin/env bash
# Nsight Compute profiling of the DiT projection kernels, one DiT block, and the compiled forward.
# Run on the GPU VM from the repo root:
#   bash scripts/ncu_profile.sh <out_dir> <tuning_bank.json> [layers|block|compiled|all]
# Produces <tag>.ncu-rep (open in the Nsight Compute GUI for roofline charts), <tag>.csv
# (raw metrics) and <tag>.log per target; summarise with scripts/ncu_summary.py <out_dir>.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?out_dir}; BANK=${2:?tuning_bank.json}; WHAT=${3:-all}
ASSETS=/content/dit_pretrained_eval/assets
CK=$ASSETS/pretrained_models/DiT-XL-2-256x256.pt
mkdir -p "$OUT"
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv > "$OUT/gpu.csv"
ncu --version | tail -1 > "$OUT/ncu_version.txt"

run() {  # run <tag> <ncu section args...> -- <python args...>
  local tag=$1; shift
  local sections=(); while [ "$1" != "--" ]; do sections+=("$1"); shift; done; shift
  if [ -f "$OUT/$tag.ncu-rep" ]; then echo "skip $tag (exists)"; return; fi
  ncu --profile-from-start off "${sections[@]}" --import-source no -f -o "$OUT/$tag" \
      python3 -m fc1bench.ncu_targets "$@" > "$OUT/$tag.log" 2>&1 || { echo "FAILED $tag (see $tag.log)"; return; }
  ncu --import "$OUT/$tag.ncu-rep" --csv --page raw > "$OUT/$tag.csv" 2>/dev/null
  echo "done $tag"
}

if [ "$WHAT" = layers ] || [ "$WHAT" = all ]; then
  for rows in 2048 16384; do                 # batch 4 and batch 32 (CFG-doubled x 256 tokens)
    for layer in qkv proj fc1 fc2; do
      for op in fp16 quant int8 int8lt int4; do
        run "layer_${layer}_${op}_m${rows}" --set full -- \
            layer --checkpoint "$CK" --bank "$BANK" --layer $layer --op $op --rows $rows
      done
    done
  done
fi
if [ "$WHAT" = block ] || [ "$WHAT" = all ]; then
  for b in 4 32; do for mode in fp16 int8; do
    run "block13_${mode}_b$b" --set full -- \
        block --dit-repo "$ASSETS/DiT" --checkpoint "$CK" --bank "$BANK" --mode $mode --batch $b
  done; done
fi
if [ "$WHAT" = compiled ] || [ "$WHAT" = all ]; then
  # Whole compiled forward = hundreds of kernels: lighter sections keep replay time reasonable.
  for b in 4 32; do for mode in fp16 int8; do
    run "compiled_${mode}_b$b" --section SpeedOfLight --section MemoryWorkloadAnalysis \
        --section LaunchStats --section Occupancy -- \
        compiled --dit-repo "$ASSETS/DiT" --checkpoint "$CK" --bank "$BANK" --mode $mode --batch $b
  done; done
fi
echo "ALL DONE"
