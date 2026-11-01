#!/usr/bin/env bash
# ==============================================================================
# CS3406 Slot-Safe Baseline Runner (`slot_safe_runner.sh`)
# ==============================================================================
# Tailored for:
#   - Login: ssh teamNN@10.4.25.63 (only during your assigned slot)
#   - Home directory: /data/cs3406/teamNN
#   - Automatic termination at slot end (with 10-minute terminal warning)
#
# Usage inside your SSH session on 10.4.25.63:
#   cd /data/cs3406/$USER/PTQ4DiT_CS3406_Baseline
#   bash slot_safe_runner.sh
# ==============================================================================

set -euo pipefail

# 1. Keep all caches and checkpoints strictly inside /data/cs3406/$USER
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

export TORCH_HOME="${PROJECT_DIR}/.cache/torch"
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"
mkdir -p "${TORCH_HOME}" "${HF_HOME}" "${TRITON_CACHE_DIR}" pretrained_models calib output checkpoints

echo "=================================================================="
echo " CS3406 AI Accelerator Project: PTQ4DiT Slot-Safe Baseline Runner"
echo " Working Directory : ${PROJECT_DIR}"
echo " Start Time        : $(date)"
echo "=================================================================="

if command -v nvidia-smi &> /dev/null; then
    echo "[GPU Check] Detected NVIDIA GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv
else
    echo "[GPU Check] nvidia-smi not found; running in CPU verification mode."
fi

# 2. Verify or download DiT-XL-2-256x256.pt (resumable via curl -C -)
CKPT_PATH="pretrained_models/DiT-XL-2-256x256.pt"
if [ ! -f "${CKPT_PATH}" ]; then
    echo "[Step 1/4] Downloading pre-trained DiT-XL-2-256x256.pt (resumable)..."
    python3 download.py --image-size 256 --outdir pretrained_models
else
    echo "[Step 1/4] Found pre-trained checkpoint at ${CKPT_PATH}"
fi

# 3. Generate Calibration Dataset (resumable per batch)
CALI_FILE="calib/imagenet_DiT-256_sample4000_50steps_allst.pt"
echo "[Step 2/4] Generating / verifying multi-timestep calibration dataset..."
python3 get_calibration_set.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --ckpt "${CKPT_PATH}" \
    --num-sampling-steps 50 \
    --outdir calib/ \
    --filename "imagenet_DiT-256_sample4000_50steps_allst.pt" \
    --batch-size 16 \
    --num-batches 2 \
    --cfg-scale 1.5 \
    --seed 1

# 4. Run Official PTQ4DiT W8A8 & W4A8 Calibration + Inference (with per-block atomic resume)
echo "[Step 3/4] Running PTQ4DiT W8A8 and W4A8 Calibration + Sampling..."
python3 quant_sample.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --ckpt "${CKPT_PATH}" \
    --num-sampling-steps 50 \
    --weight_bit 8 --act_bit 8 --cali_st 25 --cali_n 64 --cali_batch_size 32 --sm_abit 8 \
    --cali_data_path "${CALI_FILE}" \
    --outdir output/ \
    --cfg-scale 1.5 --seed 1 \
    --ptq --recon --recon-iters 20 --resume --inference --n_c 1

python3 quant_sample.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --ckpt "${CKPT_PATH}" \
    --num-sampling-steps 50 \
    --weight_bit 4 --act_bit 8 --cali_st 25 --cali_n 64 --cali_batch_size 32 --sm_abit 8 \
    --cali_data_path "${CALI_FILE}" \
    --outdir output/ \
    --cfg-scale 1.5 --seed 1 \
    --ptq --recon --recon-iters 20 --resume --inference --n_c 1

# 5. Run Automated Week 1 Benchmark & Hardware Comparison (FP vs Simulated vs Re-Parameterized)
echo "[Step 4/4] Generating Week 1 Baseline Latency & Fidelity Report..."
python3 run_baseline_benchmark.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --num-sampling-steps 50 \
    --batch-size 4 \
    --recon-iters 15 \
    --ckpt "${CKPT_PATH}" \
    --cali-data-path "${CALI_FILE}" \
    --outdir output

echo "=================================================================="
echo " ALL BASELINE TASKS COMPLETED SUCCESSFULLY!"
echo " Checkpoints & Images : ${PROJECT_DIR}/output/"
echo " Week 1 Report (MD)   : ${PROJECT_DIR}/output/baseline_report.md"
echo " Week 1 Metrics (CSV) : ${PROJECT_DIR}/output/baseline_metrics.csv"
echo "=================================================================="
