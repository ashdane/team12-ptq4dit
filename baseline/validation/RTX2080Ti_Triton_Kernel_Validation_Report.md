# PTQ4DiT Custom Triton Kernel GPU Validation & Benchmark Report
**Target Hardware:** NVIDIA GeForce RTX 2080 Ti (`sm_75`)  
**Team:** 12 — PTQ4DiT Triton Kernel Engineering  
**Date:** November 2026  

---

## 1. Environment Verification

| Attribute | Specification / Value | Verification Method |
|---|---|---|
| **GPU Model** | NVIDIA GeForce RTX 2080 Ti | `torch.cuda.get_device_name(0)` |
| **Compute Capability** | `7.5` (`sm_75`, Turing Architecture) | `torch.cuda.get_device_capability(0)` |
| **Total Memory** | 11,264 MiB (11.0 GiB) | `torch.cuda.get_device_properties(0).total_memory` |
| **NVIDIA Driver** | 570.211.01 | `nvidia-smi` |
| **Host CUDA Driver** | 12.8 | `nvidia-smi` |
| **PyTorch CUDA Runtime** | 12.1 (`torch 2.5.1+cu121`) | `torch.version.cuda` |
| **Python Version** | 3.9.12 (64-bit) | `sys.version` |
| **Triton Version** | 3.1.0 | `triton.__version__` |
| **CUDA & Triton Available** | **Yes** (fully operational) | Live JIT kernel compilation & execution |
| **Integer Dot MMA Hardware** | **Confirmed** (`mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32`) | PTX assembly inspection from compiled cubin |

---

## 2. Test Suite Execution Summary (All 32 Tests Passed)

Execution Command:
```bash
PYTHONPATH="sim/baseline_experiments" python -m unittest discover -s sim/baseline_experiments/tests -v
```
**Result:** Ran 32 tests in **10.871s** on NVIDIA GeForce RTX 2080 Ti — **32 PASSED, 0 FAILED, 0 ERRORED, 0 SKIPPED**.

| Test Module | Test Case | Target / Focus | Status |
|---|---|---|---|
| `test_cli.CliTests` | `test_cpu_smoke_artifacts` | CLI end-to-end output artifacts | **PASS** |
| `test_numerics.CudaKernelTests` | `test_all_four_dit_shapes_triton_gemm` | Live Triton GEMM across 4 DiT shapes | **PASS** |
| `test_numerics.CudaKernelTests` | `test_integer_accumulator_and_epilogue` | INT32 accumulator & fused epilogue | **PASS** |
| `test_numerics.CudaKernelTests` | `test_quantizer_matches_torch` | Triton quantizer matches PyTorch | **PASS** |
| `test_numerics.CudaKernelTests` | `test_quantizer_with_csb_scaling` | Channel-wise CSB scaling fusion | **PASS** |
| `test_numerics.NumericsTests` | `test_affine_activation_dtype_and_range` | Signed INT8 range `[-128, 127]` | **PASS** |
| `test_numerics.NumericsTests` | `test_all_four_dit_shapes_integer_gemm_ref` | PyTorch reference on 4 DiT shapes | **PASS** |
| `test_numerics.NumericsTests` | `test_all_signed_nibbles_and_odd_padding` | Packed 4-bit signed two's-complement `[-8, 7]` and odd-$K$ padding | **PASS** |
| `test_numerics.NumericsTests` | `test_fused_activation_quantization_ref` | Dynamic fused quant math parity | **PASS** |
| `test_numerics.NumericsTests` | `test_invalid_dtypes_and_unsupported_combinations` | Error gating on invalid types | **PASS** |
| `test_numerics.NumericsTests` | `test_invalid_int4` | Out-of-bounds INT4 weight detection | **PASS** |
| `test_numerics.NumericsTests` | `test_kn_packed_int4_roundtrip` | `kn_packed` pack/unpack roundtrip | **PASS** |
| `test_numerics.NumericsTests` | `test_metrics` | Numerical metric calculations | **PASS** |
| `test_numerics.NumericsTests` | `test_odd_k_dimensions` | Odd $K$ dimension handling & masking | **PASS** |
| `test_numerics.NumericsTests` | `test_rounding_boundaries_nearest_even` | Nearest-even boundary rounding | **PASS** |
| `test_numerics.NumericsTests` | `test_zero_point_correction_actual_operand_representation` | Mathematical zero-point correction | **PASS** |
| `test_numerics.NumericsTests` | `test_zero_scales_and_rounding` | Degenerate/zero scale handling | **PASS** |
| `test_numerics.NumericsTests` | `test_zero_valued_rows` | Zero-valued input row stability | **PASS** |
| `test_optimized.CPUOptimizedTests` | `test_candidates_generator` | Autotuning candidate enumeration | **PASS** |
| `test_optimized.CPUOptimizedTests` | `test_prepare_weight_shapes` | Layout transformation & packing | **PASS** |
| `test_optimized.CPUOptimizedTests` | `test_unsupported_configurations_raise` | Invalid candidate parameter gating | **PASS** |
| `test_optimized.CPUOptimizedTests` | `test_w_red_precomputation_for_int4_and_int8` | $W_{red}$ precomputation from unpacked signed INT4/8 | **PASS** |
| `test_optimized.CPUOptimizedTests` | `test_zero_point_correction_parity` | Affine correction numerical parity | **PASS** |
| `test_optimized.OptimizedTests` | `test_every_triton_candidate_with_tails_and_groups` | Tail tiles & L2 cache swizzling | **PASS** |
| `test_optimized.OptimizedTests` | `test_native_dtypes_quantization_and_store` | FP32, FP16, BF16 native dtypes | **PASS** |
| `test_optimized.OptimizedTests` | `test_optional_torch_integer_backend` | Optional PyTorch integer backend | **PASS** |
| `test_optimized.OptimizedTests` | `test_rng_restored_and_replayed` | RNG determinism across search | **PASS** |
| `test_optimized.OptimizedTests` | `test_tune_and_bank_roundtrip` | Autotuner search & bank persistence | **PASS** |
| `test_optimized.OptimizedTests` | `test_tuned_linear_fake_weight_layout_and_shape_gate` | Shape & layout validation gates | **PASS** |
| `test_report.ReportTests` | `test_invalid` | Benchmark report validation | **PASS** |
| `test_report.ReportTests` | `test_percentiles` | Summary statistic percentiles | **PASS** |
| `test_report.ReportTests` | `test_single_sample` | Single-sample statistical safety | **PASS** |

---

## 3. Full-Size DiT Operator Numerical Validation ($M=2048$)

Tested with genuine full-scale batch activations and weights ($M=2048$ tokens, DiT-XL/2 patch grid $32 \times 32 \times 2$):

### A. Implementation Error (Triton Output vs Golden Quantized Reference)

Verified using exact 64-bit integer dot product reference `(x.cpu().long() @ w.cpu().long().T).cuda().int()`:

| Operator | Dimensions ($M \times K \times N$) | Precision | Max Raw Int Diff | Relative L2 Error | Cosine Similarity | Max Abs Error | NaNs / Infs |
|---|---|---|---|---|---|---|---|
| `attn.qkv` | $2048 \times 1152 \times 3456$ | W8A8 | **0** | $4.39 \times 10^{-8}$ | **1.000000** | $4.77 \times 10^{-7}$ | 0 / 0 |
| `attn.qkv` | $2048 \times 1152 \times 3456$ | W4A8 | **0** | $4.40 \times 10^{-8}$ | **1.000000** | $4.77 \times 10^{-7}$ | 0 / 0 |
| `attn.proj` | $2048 \times 1152 \times 1152$ | W8A8 | **0** | $4.40 \times 10^{-8}$ | **1.000000** | $4.77 \times 10^{-7}$ | 0 / 0 |
| `attn.proj` | $2048 \times 1152 \times 1152$ | W4A8 | **0** | $4.40 \times 10^{-8}$ | **1.000000** | $4.77 \times 10^{-7}$ | 0 / 0 |
| `mlp.fc1` | $2048 \times 1152 \times 4608$ | W8A8 | **0** | $4.39 \times 10^{-8}$ | **1.000000** | $4.77 \times 10^{-7}$ | 0 / 0 |
| `mlp.fc1` | $2048 \times 1152 \times 4608$ | W4A8 | **0** | $4.40 \times 10^{-8}$ | **1.000000** | $4.77 \times 10^{-7}$ | 0 / 0 |
| `mlp.fc2` | $2048 \times 4608 \times 1152$ | W8A8 | **0** | $4.36 \times 10^{-8}$ | **1.000000** | $4.77 \times 10^{-7}$ | 0 / 0 |
| `mlp.fc2` | $2048 \times 4608 \times 1152$ | W4A8 | **0** | $4.36 \times 10^{-8}$ | **1.000000** | $4.77 \times 10^{-7}$ | 0 / 0 |

*Raw accumulator parity is bitwise exact across all elements (RawIntDiff = 0). Zero accumulator overflows detected.*

### B. Quantization Error (Triton Quantized Output vs Unquantized FP16 Baseline)

Error introduced purely by discrete representation of continuous floating-point weights and activations:

| Operator | Precision | MSE vs FP16 | Max Abs Error vs FP16 | Relative L2 vs FP16 | Cosine Similarity vs FP16 |
|---|---|---|---|---|---|
| `attn.qkv` | W8A8 | $1.26 \times 10^{-4}$ | 0.0731 | $1.12 \times 10^{-2}$ (1.12%) | **0.999937** |
| `attn.qkv` | W4A8 | $2.08 \times 10^{-2}$ | 0.8497 | $1.44 \times 10^{-1}$ (14.4%) | **0.989775** |
| `attn.proj` | W8A8 | $1.25 \times 10^{-4}$ | 0.0608 | $1.12 \times 10^{-2}$ (1.12%) | **0.999937** |
| `attn.proj` | W4A8 | $2.06 \times 10^{-2}$ | 0.7451 | $1.44 \times 10^{-1}$ (14.4%) | **0.989819** |
| `mlp.fc1` | W8A8 | $1.26 \times 10^{-4}$ | 0.0730 | $1.12 \times 10^{-2}$ (1.12%) | **0.999937** |
| `mlp.fc1` | W4A8 | $2.07 \times 10^{-2}$ | 0.8505 | $1.44 \times 10^{-1}$ (14.4%) | **0.989804** |
| `mlp.fc2` | W8A8 | $1.52 \times 10^{-4}$ | 0.0660 | $1.23 \times 10^{-2}$ (1.23%) | **0.999924** |
| `mlp.fc2` | W4A8 | $2.51 \times 10^{-2}$ | 0.8311 | $1.59 \times 10^{-1}$ (15.9%) | **0.987670** |

---

## 4. Kernel-Level Benchmark Comparison (RTX 2080 Ti)

### A. Default Baseline Kernels vs PyTorch FP16 Baseline (Sequential Protocol)

Warm eager execution; 25 warmup calls, 50 repeats $\times$ 10 inner calls (500 measurements) per operator:

| Operator | FP16 Base (ms) | Act Quant (ms) | W8A8 GEMM (ms) | W8A8 Comb (ms) | W8A8 Speedup (GEMM / Comb) | W4A8 GEMM (ms) | W4A8 Comb (ms) | W4A8 Speedup (GEMM / Comb) |
|---|---|---|---|---|---|---|---|---|
| `attn.qkv` | 0.4045 | 0.0553 | 0.3459 | 0.3701 | **1.17x** / **1.09x** | 0.5896 | 0.6177 | 0.69x / 0.65x |
| `attn.proj` | 0.1440 | 0.0543 | 0.1116 | 0.1519 | **1.29x** / 0.95x | 0.2105 | 0.2311 | 0.68x / 0.62x |
| `mlp.fc1` | 0.4399 | 0.0538 | 0.4612 | 0.4824 | 0.95x / 0.91x | 0.7898 | 0.8178 | 0.56x / 0.54x |
| `mlp.fc2` | 0.4486 | 0.0799 | 0.4875 | 0.5550 | 0.92x / 0.81x | 0.8868 | 0.9597 | 0.51x / 0.47x |

---

### B. Autotuned Kernels on sm_75 (`tuning.py` Interleaved Protocol)

Autotuned search space: $BM \in \{16, 32, 64\}, BN \in \{64, 128\}, BK \in \{64, 128\}$, warps $\in \{4, 8\}$, stages $\in \{1, 2, 3\}$, layouts $\in \{\text{nk}, \text{kn\_packed}\}$.

*Protocol note on FP16 baselines: In Table 4B, each row reports the native FP16 baseline measured inside the identical holdout loop with randomized round interleaving. This eliminates GPU clock frequency drift and thermal throttling bias between candidate measurements.*

| Operator | Mode | Winning Config (BM, BN, BK, warps, stages, layout) | Tuned GEMM Latency | Default GEMM Latency | Speedup vs Default Baseline | Interleaved cuBLAS FP16 | Tuned GEMM vs FP16 Speedup | Tuned Layer (Quant+GEMM) vs FP16 |
|---|---|---|---|---|---|---|---|---|
| `attn.qkv` | **W8A8** | `64, 128, 128, w8, s1, nk` | **0.1889 ms** | 0.3404 ms | **1.80x** | 0.3470 ms | **1.84x faster** | **1.55x faster** (0.224 ms) |
| `attn.qkv` | **W4A8** | `64, 128, 64, w4, s3, kn_packed` | **0.2737 ms** | 0.5689 ms | **2.08x** | 0.3397 ms | **1.24x faster** | **1.10x faster** (0.309 ms) |
| `attn.proj` | **W8A8** | `64, 128, 128, w8, s2, nk` | **0.0973 ms** | 0.1505 ms | **1.55x** | 0.1856 ms | **1.91x faster** | **1.12x faster** (0.166 ms) |
| `attn.proj` | **W4A8** | `64, 128, 64, w4, s2, kn_packed` | **0.1114 ms** | 0.1942 ms | **1.74x** | 0.1448 ms | **1.30x faster** | **0.85x (1.18x slowdown)** (0.170 ms) |
| `mlp.fc1` | **W8A8** | `64, 128, 128, w8, s3, nk` | **0.2409 ms** | 0.4479 ms | **1.86x** | 0.4428 ms | **1.84x faster** | **1.58x faster** (0.279 ms) |
| `mlp.fc1` | **W4A8** | `64, 128, 64, w4, s2, kn_packed` | **0.3595 ms** | 0.7610 ms | **2.12x** | 0.4363 ms | **1.21x faster** | **1.10x faster** (0.396 ms) |
| `mlp.fc2` | **W8A8** | `64, 128, 128, w8, s1, nk` | **0.2669 ms** | 0.4784 ms | **1.79x** | 0.4440 ms | **1.66x faster** | **1.31x faster** (0.339 ms) |
| `mlp.fc2` | **W4A8** | `64, 128, 64, w4, s3, kn_packed` | **0.4048 ms** | 0.8568 ms | **2.12x** | 0.4349 ms | **1.07x faster** | **0.91x (1.10x slowdown)** (0.478 ms) |

**Key Takeaways:**
1. **Autotuning impact:** Tuning yields **1.55x to 2.12x speedup** across all shapes over the default tile baseline.
2. **W8A8 superiority:** Tuned W8A8 GEMM is **1.66x to 1.91x faster** than native cuBLAS FP16 linear, and combined layer latency is **1.12x to 1.58x faster**.
3. **W4A8 GEMM-only vs Combined:** Tuned W4A8 GEMM core is **1.07x to 1.30x faster** than cuBLAS FP16 linear. However, when dynamic activation quantization is fused with W4A8, combined speedups occur on large matrix shapes (`attn.qkv`: 1.10x, `mlp.fc1`: 1.10x), while narrower or large-$K$ shapes (`attn.proj`: 0.85x, `mlp.fc2`: 0.91x) experience a slowdown due to activation quantization runtime and register bit-shifting overhead.

---

## 5. Integer Tensor Core & Hardware Verification

Live compilation inspection on RTX 2080 Ti confirms:
- **PTX Assembly Instruction:** Emits 32 instances of `mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32` per loop unroll in both W8A8 and W4A8 kernels.
- **Hardware Execution:** Directly targets Turing `sm_75` INT8 Tensor Cores with INT32 accumulators (`satfinite`), rather than software integer emulation or CUDA ALU scalar pipelines.
- **Accumulator Precision:** Genuine INT32 accumulation prevents saturation/overflow across all DiT dimensions ($K=1152$ and $K=4608$).

---

## 6. Weight Memory Footprint Savings

Recalculated using exact `numel() * element_size()` on actual tensors:

### A. Raw Weight Tensor Storage (Excluding Auxiliary Scales)
- FP16 weight matrix ($N \times K$ float16): $2 \times N \times K$ bytes.
- INT8 weight matrix ($N \times K$ int8): $1 \times N \times K$ bytes $\to$ **exactly 2.0000x reduction (50.00% saving)**.
- Packed INT4 weight matrix ($N \times \lceil K/2 \rceil$ uint8): $1 \times N \times \lceil K/2 \rceil$ bytes $\to$ **exactly 4.0000x reduction (75.00% saving)** for all even $K$. For odd $K$ (e.g., $K=1151$), 1 padded nibble per row yields $2302 / 576 = 3.9965\text{x}$.

### B. Total Module Storage (Including FP32 Per-Channel Scales $sw [N]$)

| Operator | FP16 Storage | W8A8 Raw + Scale | W8A8 Total Ratio | W4A8 Raw + Scale | W4A8 Total Ratio |
|---|---|---|---|---|---|
| `attn.qkv` ($1152 \times 3456$) | 7,962,624 B (7.59 MiB) | 3,981,312 B + 13,824 B = 3,995,136 B | **1.9931x** (49.83% saving) | 1,990,656 B + 13,824 B = 2,004,480 B | **3.9724x** (74.83% saving) |
| `attn.proj` ($1152 \times 1152$) | 2,654,208 B (2.53 MiB) | 1,327,104 B + 4,608 B = 1,331,712 B | **1.9931x** (49.83% saving) | 663,552 B + 4,608 B = 668,160 B | **3.9724x** (74.83% saving) |
| `mlp.fc1` ($1152 \times 4608$) | 10,616,832 B (10.12 MiB) | 5,308,416 B + 18,432 B = 5,326,848 B | **1.9931x** (49.83% saving) | 2,654,208 B + 18,432 B = 2,672,640 B | **3.9724x** (74.83% saving) |
| `mlp.fc2` ($4608 \times 1152$) | 10,616,832 B (10.12 MiB) | 5,308,416 B + 4,608 B = 5,313,024 B | **1.9983x** (49.96% saving) | 2,654,208 B + 4,608 B = 2,658,816 B | **3.9931x** (74.96% saving) |

---

## 7. Architectural Constraints & Known Limitations

1. **Turing `sm_75` BF16 Limitation:** RTX 2080 Ti hardware lacks native hardware `.bf16` tensor arithmetic instructions (first introduced on Ampere `sm_80`). Triton kernels attempting to compile `.bf16` produce fatal `ptxas` errors (`Feature '.bf16' requires .target sm_80 or higher`). FP16 and FP32 are fully supported; tests and tuning automatically guard BF16 on `sm_75`.
2. **Triton 3.1.0 Loop Parameter:** Triton 3.1.0 does not support `loop_unroll_factor` in `tl.range`. The kernels utilize standard `range(0, tl.cdiv(K, BK))` which compiles and unrolls cleanly.
3. **Register-Level Unpacking in W4A8:** 4-bit weights are packed into UINT8 bytes and unpacked in register memory at runtime (`(byte >> shift) & 15`). Because the extra bitwise ALU instructions compete with Tensor Core issue slots, tuning with `kn_packed` and larger tiles ($BM=64, BN=128$) is essential to deliver speedups over FP16.
4. **Quantization Amortization on Small Matrix Shapes:** On narrower GEMMs like `attn.proj` ($N=1152$), activation quantization latency (~0.054 ms) is a higher fraction of runtime than on large GEMMs like `mlp.fc1` ($N=4608$).
5. **Blackwell (`sm_120`) Untested:** Validation on newer microarchitectures such as Blackwell (`sm_120`) was not conducted due to cluster hardware provisioning (`sm_75` RTX 2080 Ti allocated). This is a hardware-specific environment limitation, not an outstanding implementation task.

---

## 8. Final Direct Verdict

### **COMPLETE**

All custom Triton kernels (`quantize_activation`, `integer_linear`, `gemm`, `prepare_weight`) for PTQ4DiT Team 12 are fully implemented, verified on NVIDIA GeForce RTX 2080 Ti hardware, confirmed with 32/32 passing tests, numerically validated across all four DiT operator shapes with exact INT32 integer parity, benchmarked, and autotuned to achieve up to **1.91x speedup over cuBLAS FP16**. All artifacts, benchmarks, configuration banks, and reports are committed and synced in the repository.

My individual Triton-kernel implementation and validation work is complete on the NVIDIA RTX 2080 Ti (sm_75). Blackwell (sm_120) validation remains untested and is a hardware-specific limitation, not an outstanding implementation task.
