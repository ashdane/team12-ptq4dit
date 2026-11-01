# PTQ4DiT Baseline & Hardware Acceleration Package (`CS3406`)

Complete baseline codebase, slot-safe execution scripts, offline re-parameterization pass, and reference paper PDF for **PTQ4DiT: Post-training Quantization for Diffusion Transformers** (NeurIPS 2024, [arXiv:2405.16005v3](https://arxiv.org/abs/2405.16005), [Official GitHub](https://github.com/adreamwu/PTQ4DiT)).

Built specifically for the **CS3406 Programming AI Accelerators** GPU cluster (`L40S 48GB` / `RTX PRO 5000 48GB` / `A100 40GB`).

---

## 1. Package Directory Structure

```text
PTQ4DiT_CS3406_Baseline/
├── paper/
│   ├── PTQ4DiT_NeurIPS2024_Paper.pdf   # Full NeurIPS 2024 paper (Eqs. 1-21, Alg. 1, Tables 1-3, Appendices)
│   └── PTQ4DiT_NeurIPS2024_Paper.tex   # LaTeX source of the paper
├── README.md                           # This step-by-step implementation & slot execution guide
├── slot_safe_runner.sh                 # Turnkey 1-command runner for your CS3406 GPU slot
├── run_baseline_benchmark.py           # Automated Week 1 benchmark (FP vs Simulated vs Re-Param W8A8/W4A8)
├── get_calibration_set.py              # Multi-timestep calibration dataset generator (resumable per batch)
├── quant_sample.py                     # PTQ4DiT calibration & quantized sampling CLI (resumable per block)
├── fast_inference_reparam.py           # Offline Re-Parameterization (completes the authors' unchecked TODO)
├── models.py                           # Self-contained DiT-XL/2 architecture (100% checkpoint-compatible)
├── download.py                         # Resumable pre-trained DiT checkpoint downloader (curl -C -)
├── environment.yml                     # Conda environment specification
├── requirements.txt                    # Fast pip requirements for virtualenv setup
├── diffusion/
│   ├── __init__.py                     # create_diffusion() factory
│   └── gaussian_diffusion.py           # DDPM/DDIM respaced sampler (50/100/250 steps)
├── quant/
│   ├── __init__.py
│   ├── csb_ssc.py                      # Channel-wise Salience Balancing (CSB) & Spearman's rho (SSC)
│   ├── quant_layer.py                  # UniformAffineQuantizer & QuantModule (Simulated + Re-Param modes)
│   ├── quant_model.py                  # DiTBlock wrapper for attn.qkv, attn.proj, mlp.fc1, mlp.fc2
│   └── block_recon.py                  # Per-block scale calibration with atomic checkpointing & auto-resume
└── utils/
    ├── __init__.py
    └── slot_guard.py                   # SIGTERM/SIGHUP signal catcher & atomic file saver for CS3406 slots
```

---

## 2. Understanding the CS3406 GPU Slot Rules & How This Code Protects You

1. **Login Window (`ssh teamNN@10.4.25.63`):**
   - You can only SSH in during your team's assigned slot.
   - Outside your slot, SSH refuses login with `"Your account has expired"`—this simply means your slot is not currently active.
2. **Strict Home Directory (`/data/cs3406/teamNN`):**
   - Store **all** code, virtual environments, HuggingFace/PyTorch caches, and checkpoints inside `/data/cs3406/teamNN`.
   - `slot_safe_runner.sh` automatically sets `TORCH_HOME`, `HF_HOME`, and `TRITON_CACHE_DIR` inside `/data/cs3406/teamNN/PTQ4DiT_CS3406_Baseline/.cache` so you never hit permission errors or fill up `/tmp`.
3. **Hard Logout & Process Termination at Slot End:**
   - When your slot ends (or if you log out early), **all running processes are killed immediately** (`nohup`, `tmux`, and `screen` will **not** survive after logout).
   - You receive a terminal warning **10 minutes before** your slot ends.
4. **Built-in Slot Protection (`utils/slot_guard.py`):**
   - **Atomic Checkpointing (`atomic_torch_save`):** Every calibration batch and every reconstructed DiT block (`0..27`) is written to a temporary `.tmp` file first and atomically renamed via `os.replace()`. Even if the server cuts power mid-second, your `.pth` checkpoint is never corrupted.
   - **Signal Interception (`SIGTERM` / `SIGHUP` / `SIGINT`):** When the slot expires or SSH drops, `SlotGuard` catches the OS signal and flushes the current calibration/block state before exiting.
   - **Instant Resume (`--resume`):** Re-running the exact same command in your next slot automatically detects completed batches/blocks and resumes from the exact block where it stopped.

---

## 3. Step-by-Step Execution Guide (Finish Baseline in ~25 Minutes of Slot Time)

### Step 0: Offline Preparation *Before* Your GPU Slot Starts (0 GPU Minutes)

Do not waste precious GPU slot minutes downloading the 2.7 GB `DiT-XL-2-256x256.pt` checkpoint over the cluster's shared internet if you can pre-download it on your laptop before your slot begins:

```bash
# Run on your local laptop BEFORE your slot starts:
curl -L -C - -O https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
```

### Step 1: Log In When Your Slot Opens & Upload the Package (~2 Minutes)

Replace `teamNN` with your actual team username (e.g., `team04`):

```bash
# 1. From your laptop, copy the zip and (optional) pre-downloaded checkpoint to /data/cs3406/teamNN/
scp PTQ4DiT_CS3406_Baseline_Package.zip teamNN@10.4.25.63:/data/cs3406/teamNN/
# If you pre-downloaded DiT-XL-2-256x256.pt on your laptop:
scp DiT-XL-2-256x256.pt teamNN@10.4.25.63:/data/cs3406/teamNN/

# 2. SSH into the GPU node
ssh teamNN@10.4.25.63

# 3. Go to your private team directory and unpack
cd /data/cs3406/$(whoami)
unzip -q PTQ4DiT_CS3406_Baseline_Package.zip
cd PTQ4DiT_CS3406_Baseline

# If you copied DiT-XL-2-256x256.pt to /data/cs3406/teamNN/, move it into pretrained_models/:
mkdir -p pretrained_models
if [ -f ../DiT-XL-2-256x256.pt ]; then
    mv ../DiT-XL-2-256x256.pt pretrained_models/
fi
```

### Step 2: Set Up the Python Environment (~2 Minutes)

**Fastest Option (Reuses the cluster's pre-installed CUDA PyTorch via `venv`):**
```bash
python3 -m venv --system-site-packages /data/cs3406/$(whoami)/venv_dit
source /data/cs3406/$(whoami)/venv_dit/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*(Alternative if you prefer Conda: `conda env create -f environment.yml && conda activate DiT`)*

### Step 3: Run the Complete Baseline with One Command (~20 Minutes)

Keep your SSH terminal open (remember: logging out stops GPU jobs!) and run:

```bash
bash slot_safe_runner.sh
```

What `slot_safe_runner.sh` executes automatically:
1. Checks your assigned GPU (`L40S 48GB`, `RTX PRO 5000 48GB`, or `A100 40GB`) via `nvidia-smi`.
2. Verifies `pretrained_models/DiT-XL-2-256x256.pt` (or downloads it with `curl -C -` resume support if you didn't upload it in Step 1).
3. Runs `get_calibration_set.py` to collect multi-timestep calibration trajectories across 50 denoising steps (`calib/imagenet_DiT-256_sample4000_50steps_allst.pt`).
4. Runs `quant_sample.py` for **W8A8** and **W4A8** using Channel-wise Salience Balancing (CSB) + Spearman's $\rho$-guided Salience Calibration (SSC) + Block Reconstruction, saving calibrated checkpoints (`output/256_88_50/ckpt.pth` and `output/256_48_50/ckpt.pth`) and sample image grids (`sample_ptq_W8A8.png` and `sample_ptq_W4A8.png`).
5. Runs `run_baseline_benchmark.py` to generate your Week 1 benchmark table in `output/baseline_report.md` and `output/baseline_metrics.csv`.

---

## 4. Manual Step-by-Step Commands (If You Prefer Running Step-by-Step)

### A. Generate the Multi-Timestep Calibration Dataset
```bash
python3 get_calibration_set.py \
  --model DiT-XL/2 --image-size 256 \
  --ckpt pretrained_models/DiT-XL-2-256x256.pt \
  --num-sampling-steps 50 \
  --outdir calib/ \
  --filename imagenet_DiT-256_sample4000_50steps_allst.pt \
  --batch-size 16 --num-batches 2 \
  --cfg-scale 1.5 --seed 1
```

### B. Calibrate & Sample W8A8 (8-bit Weights, 8-bit Activations)
```bash
python3 quant_sample.py \
  --model DiT-XL/2 --image-size 256 \
  --ckpt pretrained_models/DiT-XL-2-256x256.pt \
  --num-sampling-steps 50 \
  --weight_bit 8 --act_bit 8 --cali_st 25 --cali_n 64 --cali_batch_size 32 --sm_abit 8 \
  --cali_data_path calib/imagenet_DiT-256_sample4000_50steps_allst.pt \
  --outdir output/ \
  --cfg-scale 1.5 --seed 1 \
  --ptq --recon --recon-iters 20 --resume --inference --n_c 1
```

### C. Calibrate & Sample W4A8 (4-bit Weights, 8-bit Activations)
```bash
python3 quant_sample.py \
  --model DiT-XL/2 --image-size 256 \
  --ckpt pretrained_models/DiT-XL-2-256x256.pt \
  --num-sampling-steps 50 \
  --weight_bit 4 --act_bit 8 --cali_st 25 --cali_n 64 --cali_batch_size 32 --sm_abit 8 \
  --cali_data_path calib/imagenet_DiT-256_sample4000_50steps_allst.pt \
  --outdir output/ \
  --cfg-scale 1.5 --seed 1 \
  --ptq --recon --recon-iters 20 --resume --inference --n_c 1
```

### D. Test Offline Re-Parameterization (`--fast-reparam`)
```bash
python3 quant_sample.py \
  --model DiT-XL/2 --image-size 256 \
  --ckpt pretrained_models/DiT-XL-2-256x256.pt \
  --num-sampling-steps 50 \
  --weight_bit 4 --act_bit 8 \
  --outdir output/ \
  --cfg-scale 1.5 --seed 1 \
  --ptq --resume --cali_ckpt output/256_48_50/ckpt.pth \
  --fast-reparam --inference --n_c 1
```

---

## 5. What If You See the "10 Minutes Remaining" Slot Warning?

- **Do not panic!** Both `get_calibration_set.py` and `quant_sample.py` save atomic checkpoints after every batch and every DiT block (`output/256_88_50/ckpt.pth`).
- If you want to stop manually before the server disconnects you, press `Ctrl+C` once: `SlotGuard` will catch `SIGINT`, flush the latest block checkpoint to `/data/cs3406/teamNN/PTQ4DiT_CS3406_Baseline/output/`, and exit cleanly.
- Before logging out, copy your `output/baseline_report.md`, `output/baseline_metrics.csv`, and generated `.png` images to your laptop:
  ```bash
  # Run from your laptop while your slot is still active:
  scp -r teamNN@10.4.25.63:/data/cs3406/teamNN/PTQ4DiT_CS3406_Baseline/output ./local_output
  ```

---

## 6. Roadmap for Weeks 2–4 (Your Accelerator Optimization Contribution)

1. **Completed Step 1 (Included in `fast_inference_reparam.py`):**
   - The official `adreamwu/PTQ4DiT` repository left `[ ] Release implementation of Re-Parameterization for fast inference` unfinished.
   - `fast_inference_reparam.py` implements Section 4.3 & Appendix D (Eqs. 12–21), absorbing $\mathbf{B}_\rho^{\mathbf{W}}$ into the linear weights offline and folding $\mathbf{B}_\rho^{\mathbf{X}}$ into the `adaLN_modulation` MLP weights/biases.
2. **Week 2–3 Step 2 (Fused Low-Bit Tensor Core GEMM in Triton / CUTLASS):**
   - Inside `quant/quant_layer.py` (`QuantModule.forward`), replace `F.linear(x_sim, self.weight, self.bias)` under `self.reparameterized == True` with:
     - **On L40S (Ada) & RTX PRO 5000 (Blackwell):** Native FP8 (`torch._scaled_mm` with `torch.float8_e4m3fn` or a custom OpenAI Triton `tl.dot` FP8/INT8 kernel) + fused dynamic activation scale reduction inside `AdaLN-Zero`.
     - **On A100 (Ampere):** Packed INT8 (`W8A8`) and INT4-weight / INT8-activation (`W4A8`) Triton or CUTLASS GEMM kernels.
3. **Week 4 Step 3 (Cross-Architecture Roofline Evaluation):**
   - Run `run_baseline_benchmark.py` across **A100 (HBM2)**, **L40S (GDDR6)**, and **RTX PRO 5000 (Blackwell)** to compare memory-bandwidth savings vs. Tensor Core compute speedup.
