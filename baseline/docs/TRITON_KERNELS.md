# PTQ4DiT Custom Triton Kernel Suite — Team 12
**High-Performance Low-Bit Linear Operators for Diffusion Transformers (DiT-XL/2)**  
**Target Architecture:** NVIDIA Turing (`sm_75`, tested on RTX 2080 Ti) and newer  
**Primary Deliverable:** Standalone Audited Kernel Package  

---

## 1. Executive Summary

This package provides custom, high-performance Triton kernels implementing low-bit linear operations tailored for Post-Training Quantization of Diffusion Transformers (**PTQ4DiT**). The kernels are specifically optimized for DiT-XL/2 linear projections (`attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`) and execute directly on NVIDIA integer Tensor Cores (`mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32`).

Key accomplishments verified on NVIDIA GeForce RTX 2080 Ti hardware:
- **32 / 32 passing unit tests** (0 failures, 0 errors, 0 skips).
- **Exact INT32 accumulator parity** (Raw integer difference = 0 across all DiT operator shapes).
- **W8A8 GEMM Speedup:** Up to **1.91x faster** than cuBLAS FP16 linear.
- **W8A8 Combined Speedup:** Up to **1.58x faster** including online dynamic activation quantization.
- **W4A8 GEMM Speedup:** Up to **1.30x faster** than cuBLAS FP16 linear with on-the-fly register unpacking.
- **Weight Footprint Reduction:** **2.00x** (W8) and **4.00x** (W4) raw tensor compression.

---

## 2. Archive Directory Structure

```text
Team12_Triton_Kernels_Final.zip
├── README_TRITON_KERNELS.md                     # Comprehensive documentation & reproduction guide
├── src/
│   └── fc1bench/
│       ├── __init__.py                          # Package initialization
│       ├── kernels.py                           # Fused activation quantizer & base integer GEMM
│       ├── optimized.py                         # Tiled GEMM engine with L2 swizzling & autotuned dispatch
│       ├── quantization.py                      # Exact banker's rounding, nibble packing, & references
│       ├── benchmark.py                         # Benchmark CLI and evaluation harness
│       ├── tuning.py                            # Candidate autotuner with randomized holdout verification
│       ├── eval_utils.py                        # Output CSV/JSON serialization utilities
│       ├── report.py                            # Statistical summary reporter (percentiles, p90, stdev)
│       ├── model_layers.py                      # DiT block integration abstractions
│       ├── integration.py                       # Layer-level replacement hooks
│       ├── full_model.py                        # DiT model-level wrapping utilities
│       └── upstream.py                          # Upstream fake-quant comparator utilities
├── tests/
│   ├── test_cli.py                              # CLI end-to-end smoke test
│   ├── test_numerics.py                         # Golden math parity, boundary rounding, odd K, CUDA tests
│   ├── test_optimized.py                        # Candidate generator, packing roundtrips, tuning bank tests
│   └── test_report.py                           # Percentile and statistical distribution tests
├── validation/
│   ├── run_tests.sh                             # Shell script to run the 32-test unit test suite
│   ├── run_full_benchmarks.py                   # Automated benchmark harness across all 4 DiT shapes
│   ├── validate_full_shapes.py                  # Full-size (M=2048) numerical parity verification script
│   ├── inspect_mma.py                           # Hardware PTX assembly inspection tool
│   └── RTX2080Ti_Triton_Kernel_Validation_Report.md # Formal GPU audit and validation report
└── benchmarks/
    ├── benchmark_results_rtx2080ti.json         # Raw benchmark latency distributions (500 samples/op)
    └── tuning/
        ├── tuning_bank.json                     # Validated optimal tile configurations for sm_75
        ├── summary.json                         # Complete candidate search metrics
        ├── summary.csv                          # Tabular search results
        ├── 2048_3456_1152_w8_float16_gemm.ptx   # Compiled PTX assembly for attn.qkv W8A8
        ├── 2048_3456_1152_w8_float16_gemm.cubin # Compiled CUBIN binary for attn.qkv W8A8
        ├── 2048_3456_1152_w4_float16_gemm.ptx   # Compiled PTX assembly for attn.qkv W4A8
        ├── 2048_3456_1152_w4_float16_gemm.cubin # Compiled CUBIN binary for attn.qkv W4A8
        ├── 2048_1152_1152_w8_float16_gemm.ptx   # Compiled PTX assembly for attn.proj W8A8
        ├── 2048_1152_1152_w8_float16_gemm.cubin # Compiled CUBIN binary for attn.proj W8A8
        ├── 2048_1152_1152_w4_float16_gemm.ptx   # Compiled PTX assembly for attn.proj W4A8
        ├── 2048_1152_1152_w4_float16_gemm.cubin # Compiled CUBIN binary for attn.proj W4A8
        ├── 2048_4608_1152_w8_float16_gemm.ptx   # Compiled PTX assembly for mlp.fc1 W8A8
        ├── 2048_4608_1152_w8_float16_gemm.cubin # Compiled CUBIN binary for mlp.fc1 W8A8
        ├── 2048_4608_1152_w4_float16_gemm.ptx   # Compiled PTX assembly for mlp.fc1 W4A8
        ├── 2048_4608_1152_w4_float16_gemm.cubin # Compiled CUBIN binary for mlp.fc1 W4A8
        ├── 2048_1152_4608_w8_float16_gemm.ptx   # Compiled PTX assembly for mlp.fc2 W8A8
        ├── 2048_1152_4608_w8_float16_gemm.cubin # Compiled CUBIN binary for mlp.fc2 W8A8
        ├── 2048_1152_4608_w4_float16_gemm.ptx   # Compiled PTX assembly for mlp.fc2 W4A8
        └── 2048_1152_4608_w4_float16_gemm.cubin # Compiled CUBIN binary for mlp.fc2 W4A8
```

---

## 3. Kernel Components & Implementation Details

### A. Fused Dynamic Activation Quantization (`fc1bench/kernels.py`)
- **Kernel:** `_quantize_fused` / Python wrapper `quantize_activation`
- **Supported Inputs:** Contiguous FP16, BF16, or FP32 matrices of arbitrary shape `[M, K]`.
- **CSB Channel Scaling:** Fuses channel-wise scaling vector $BX [K]$ directly in register memory prior to reduction, preventing intermediate floating-point global memory writes.
- **Two-Pass Tile Reductions:** Supports arbitrary and non-power-of-two $K$ (e.g., $K=4608$ in `mlp.fc2`) using chunked reductions (`BLOCK_K` between 256 and 2048).
- **Rounding:** Strictly adheres to IEEE-754 round-to-nearest-even ("banker's rounding") via `libdevice.nearbyint` and `tl.div_rn`.
- **Output Modes:**
  - *Symmetric INT8 ($zx = 0$):* Produces $Q \in [-127, 127]$ and per-row scale $S [M]$.
  - *Affine Asymmetric INT8:* Produces $Q \in [-128, 127]$, scale $S [M]$, and zero-point $ZX [M] \in [-128, 127]$.

### B. Tiled INT8 / Packed INT4 GEMM (`fc1bench/optimized.py`)
- **Kernel:** `_tiled` / Python dispatch `gemm`
- **Arithmetic:** Pure INT8 $\times$ INT8 $\to$ INT32 accumulation on Tensor Cores (`tl.dot(..., out_dtype=tl.int32)`).
- **Register-Level INT4 Unpacking:** 4-bit weights packed into `uint8` are unpacked on-the-fly in hardware registers via bitwise shifts and sign-extension (`((nibble ^ 8) - 8)`), entirely avoiding global memory FP16 dequantization.
- **Supported Layouts:**
  - `nk`: Row-major per output channel `[N, (K+1)//2]`.
  - `kn_packed`: Interleaved column-major `[K, (N+1)//2]` with pair contiguous $N$ elements for optimized memory coalescing.
- **L2 Cache Swizzling:** Uses thread-block grouping (`GROUP_M = 8`) to maximize L2 data reuse of activation tiles across adjacent output channel tiles.
- **Fused Epilogue:** Computes zero-point correction, activation scale multiplication, weight scale multiplication, and bias addition before converting to the requested output floating-point format (FP32 or FP16):
  $$\text{out}[m, n] = \left( \text{raw\_acc}[m, n] - z_x[m] \cdot W_{red}[n] \right) \cdot s_x[m] \cdot s_w[n] + \text{bias}[n]$$
  where $W_{red}[n] = \sum_{k} W_q[n, k]$ is precomputed from unpacked signed INT4/8 weights.

---

## 4. Supported Operator Shapes

Optimized for the DiT-XL/2 architecture with patch size $p=2$, latent channels $C=4$, and image resolution $512 \times 512$ (token count $M = 2048$):

| Operator | Context | $M$ (Tokens) | $K$ (In Dim) | $N$ (Out Dim) | Compute (GOPs) |
|---|---|---|---|---|---|
| `attn.qkv` | Self-Attention QKV Projection | 2048 | 1152 | 3456 | 16.30 |
| `attn.proj` | Self-Attention Output Projection | 2048 | 1152 | 1152 | 5.43 |
| `mlp.fc1` | Feed-Forward Expansion (4x) | 2048 | 1152 | 4608 | 21.73 |
| `mlp.fc2` | Feed-Forward Projection (Post-GELU) | 2048 | 4608 | 1152 | 21.73 |

*Arbitrary and irregular dimensions (e.g. odd $K$, non-multiples of block sizes) are fully supported via boundary masking.*

---

## 5. Software & Hardware Dependencies

- **Python:** 3.9+ (tested with 3.9.12)
- **PyTorch:** $\ge 2.1$ (tested with `2.5.1+cu121`)
- **Triton:** $\ge 3.0$ (tested with `3.1.0`)
- **CUDA Runtime:** $\ge 12.0$ (tested with CUDA 12.1 / Driver 570.211.01)
- **Hardware Requirement:** NVIDIA GPU with Tensor Cores supporting INT8 MMA instructions (Turing `sm_75`, Ampere `sm_80`/`sm_86`, Ada `sm_89`, Hopper `sm_90`).

---

## 6. How to Run Tests & Reproduce Validation

### A. Environment Setup
```bash
# Add src to PYTHONPATH
export PYTHONPATH="src:$PYTHONPATH"
```

### B. Run the Complete 32-Test Suite
```bash
# Execute unit tests across CPU references and GPU kernels
bash validation/run_tests.sh

# Or directly via unittest discovery:
python -m unittest discover -s tests -v
```
**Expected Output:**
```text
Ran 32 tests in ~11s
OK
```

### C. Verify Full-Size DiT Shapes ($M=2048$)
```bash
python validation/validate_full_shapes.py
```
**Expected Output:**
```text
Testing attn.qkv: M=2048, K=1152, N=3456...
  attn.qkv W8A8: RawIntDiff=0, MaxAbsErr=4.77e-07, RelL2=4.39e-08, CosSim=1.000000
  attn.qkv W4A8: RawIntDiff=0, MaxAbsErr=4.77e-07, RelL2=4.40e-08, CosSim=1.000000
...
ALL FOUR DIT SHAPES FULLY VALIDATED AT FULL SIZE (M=2048) WITH ZERO ERRORS!
```

### D. Verify Integer Tensor Core Assembly
```bash
python validation/inspect_mma.py
```
**Expected Output:**
```text
W8A8 MMA instructions in PTX: 32
  mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 ...
W4A8 MMA instructions in PTX: 32
  mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 ...
```

### E. Run Full Empirical Benchmark Suite
```bash
python validation/run_full_benchmarks.py
```

### F. Run Autotuner
```bash
python -m fc1bench.tuning --dtype float16 --rounds 10 --inner 5 --output benchmarks/tuning
```

---

## 7. Measured Benchmark Results on RTX 2080 Ti (`sm_75`)

### A. Autotuned Kernel Performance vs cuBLAS FP16 Baseline

Measured using randomized round interleaving to eliminate thermal drift and clock variation:

| Operator Shape ($M \times N \times K$) | Mode | Winning Autotuned Config | Tuned GEMM Latency | cuBLAS FP16 Latency | GEMM Speedup vs FP16 | Tuned Layer (Quant+GEMM) | Layer Speedup vs FP16 |
|---|---|---|---|---|---|---|---|
| `attn.qkv` ($2048 \times 3456 \times 1152$) | **W8A8** | `BM=64, BN=128, BK=128, w=8, s=1, nk` | **0.1889 ms** | 0.3470 ms | **1.84x faster** | **0.2240 ms** | **1.55x faster** |
| `attn.qkv` ($2048 \times 3456 \times 1152$) | **W4A8** | `BM=64, BN=128, BK=64, w=4, s=3, kn_packed` | **0.2737 ms** | 0.3397 ms | **1.24x faster** | **0.3090 ms** | **1.10x faster** |
| `attn.proj` ($2048 \times 1152 \times 1152$) | **W8A8** | `BM=64, BN=128, BK=128, w=8, s=2, nk` | **0.0973 ms** | 0.1856 ms | **1.91x faster** | **0.1661 ms** | **1.12x faster** |
| `attn.proj` ($2048 \times 1152 \times 1152$) | **W4A8** | `BM=64, BN=128, BK=64, w=4, s=2, kn_packed` | **0.1114 ms** | 0.1448 ms | **1.30x faster** | 0.1702 ms | 0.85x (1.18x slowdown) |
| `mlp.fc1` ($2048 \times 4608 \times 1152$) | **W8A8** | `BM=64, BN=128, BK=128, w=8, s=3, nk` | **0.2409 ms** | 0.4428 ms | **1.84x faster** | **0.2794 ms** | **1.58x faster** |
| `mlp.fc1` ($2048 \times 4608 \times 1152$) | **W4A8** | `BM=64, BN=128, BK=64, w=4, s=2, kn_packed` | **0.3595 ms** | 0.4363 ms | **1.21x faster** | **0.3961 ms** | **1.10x faster** |
| `mlp.fc2` ($2048 \times 1152 \times 4608$) | **W8A8** | `BM=64, BN=128, BK=128, w=8, s=1, nk` | **0.2669 ms** | 0.4440 ms | **1.66x faster** | **0.3385 ms** | **1.31x faster** |
| `mlp.fc2` ($2048 \times 1152 \times 4608$) | **W4A8** | `BM=64, BN=128, BK=64, w=4, s=3, kn_packed` | **0.4048 ms** | 0.4349 ms | **1.07x faster** | 0.4776 ms | 0.91x (1.10x slowdown) |

---

## 8. W8A8 vs W4A8 Trade-Offs & Memory Footprint

### A. Memory Savings (`numel() * element_size()`)
- **Raw Tensor Compression:** INT8 weights achieve an exact **2.0000x (50.00%)** memory reduction; packed INT4 weights achieve an exact **4.0000x (75.00%)** memory reduction.
- **Total Module Storage (Weights + FP32 Scale Vector):**
  - `attn.qkv`: FP16 = 7.96 MB $\to$ W8A8 = 4.00 MB (**1.993x**); W4A8 = 2.00 MB (**3.972x**).
  - `mlp.fc1`: FP16 = 10.62 MB $\to$ W8A8 = 5.33 MB (**1.993x**); W4A8 = 2.67 MB (**3.972x**).

### B. Computational Trade-Offs
1. **W8A8 (Optimal Performance & Accuracy):** Operates natively on hardware INT8 Tensor Cores without bit-manipulation in ALU registers. Delivers consistent **1.66x–1.91x GEMM speedups** and **1.12x–1.58x combined layer speedups** over FP16, with minimal relative L2 quantization distortion (~1.12%).
2. **W4A8 (Optimal Memory Footprint):** Maximizes VRAM compression (4x reduction), enabling deployment on memory-constrained GPUs. The register-level bit-unpacking competes with Tensor Core issue slots; however, with autotuned `kn_packed` layout and $BM=64, BN=128$, the GEMM core achieves **1.07x–1.30x speedups** over FP16.
3. **Quantization Amortization:** Dynamic activation quantization requires ~0.054 ms (for $K=1152$) to 0.080 ms (for $K=4608$). For large matrices (`attn.qkv`, `mlp.fc1`), this cost is easily amortized (yielding 1.10x combined speedup for W4A8). For narrower GEMMs (`attn.proj` $N=1152$), the quantization overhead represents a larger fraction of total layer runtime, resulting in a net slowdown for W4A8 relative to cuBLAS FP16.

---

## 9. Architectural Constraints & Limitations

1. **Turing `sm_75` BF16 Hardware Limitation:** The RTX 2080 Ti (`sm_75`) lacks native hardware `.bf16` tensor arithmetic instructions (introduced in Ampere `sm_80`). Triton kernels targeting BF16 fail in `ptxas`. FP16 and FP32 outputs are fully supported.
2. **Triton 3.1.0 Range Unroll Factor:** Triton 3.1.0 does not support `loop_unroll_factor` in `tl.range`. Standard `range(0, tl.cdiv(K, BK))` is used.
3. **Blackwell (`sm_120`) Untested:** Validation on newer microarchitectures such as Blackwell (`sm_120`) was not conducted due to cluster hardware provisioning (`sm_75` RTX 2080 Ti was the allocated target). This is a hardware-specific environment limitation, not an outstanding implementation task.

---

## 10. Team 12 Completion Statement

My individual Triton-kernel implementation and validation work is complete on the NVIDIA RTX 2080 Ti (sm_75). Blackwell (sm_120) validation remains untested and is a hardware-specific limitation, not an outstanding implementation task.
