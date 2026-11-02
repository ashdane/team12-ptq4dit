# Analysis of the A100 FC1 experiment

INT8 is the fastest measured layer under the FP32-input/FP32-output contract, but BF16 has the fastest prepared-input GEMM. Packed INT4 reduces weight storage while losing both speed and numerical fidelity against INT8/BF16. These are synthetic FC1 findings, not full-model PTQ4DiT results.

## Scope and integrity

- GPU: NVIDIA A100-SXM4-40GB, compute capability 8.0; no L40 result is present.
- PyTorch 2.11.0+cu130, Triton 3.6.0, CUDA 13.0; TF32 disabled.
- M=2048, K=1152, N=4608; synthetic Gaussian data, seed 2026.
- 20 warmup calls; 50 samples averaging 10 calls each. Profiling is separate from timing.
- All five main modes and five upstream-run modes report status=ok. Integer modes passed the benchmark’s exact accumulator and output-reference checks. A standalone validation.log is absent from this archive, so the complete unittest outcome cannot be independently recovered.
- All seven saved implementation files match the SHA256 hashes in the main results.json. The upstream checkout reports dirty=true; a commit SHA alone does not fully describe it. The quant_layer.py hash is recorded, but the checkout/diff is not included.

## Main run

Times below are median CUDA-event milliseconds per invocation. Weight storage includes scale metadata but excludes bias. Relative L2 error is expressed as a percentage.

| Mode | Core ms | Layer ms | Layer speedup vs FP32 | Relative L2 error | Weights + scales MiB | Incremental layer peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 1.196237 | 1.196134 | 1.00x | 0.0000% | 20.250 | 36.000 |
| fp16 | 0.147187 | 0.227995 | 5.25x | 0.0388% | 10.125 | 54.000 |
| bf16 | 0.111043 | 0.181266 | 6.60x | 0.2875% | 10.125 | 54.000 |
| int8 | 0.153245 | 0.171896 | 6.96x | 1.1194% | 5.080 | 38.258 |
| int4 | 0.401134 | 0.385446 | 3.10x | 14.4197% | 2.549 | 38.883 |

INT8 layer latency is 5.17% lower than BF16 (0.171896 vs 0.181266 ms), while its core latency is 38.0% higher (0.153245 vs 0.111043 ms). This is a boundary-conversion result, not evidence that the integer GEMM beats BF16 GEMM. FP16 has the lowest error among reduced-precision modes; BF16 is the fastest floating-point mode in these runs.

The separate three-call profiler supports the explanation: BF16 spends approximately 106.015 us/call in GEMM and 69.653 us/call in two conversion kernels. INT8 spends approximately 137.503 us/call in GEMM and 22.944 us/call in activation quantization. Those are profiled timings, not replacements for the unprofiled benchmark medians. Keeping activations in FP16/BF16 across real model layers removes the benchmark’s forced FP32 boundaries, so the INT8 advantage may disappear.

INT4 layer latency is 2.24x INT8 and 2.13x BF16. Its profiled GEMM takes approximately 354.227 us/call and accounts for 94.77% of summed integer-path GPU kernel time; quantization is 19.541 us/call. Optimizing activation quantization alone cannot address this main bottleneck.

## Numerical meaning

Both integer modes record zero MSE and zero maximum error against the same-quantization implementation reference. The INT32 accumulator check is exact. Their errors against FP32 therefore arise from the selected quantization scheme on these inputs, not an observed integer arithmetic mismatch.

INT8 relative L2 error is 1.1194%, maximum absolute error 0.06548, cosine similarity 0.999937. INT4 relative L2 error is 14.4197%, maximum absolute error 0.93481, cosine similarity 0.989764. INT4’s high cosine similarity does not erase its substantial magnitude error. Neither FC1 metric establishes latent fidelity or FID/sFID across 50 diffusion steps.

## Upstream fake-quant comparison

| Mode | Core ms | Layer ms | Relative L2 error |
|---|---:|---:|---:|
| fake_w8a8 | 1.195725 | 1.607322 | 1.3815% |
| fake_w4a8 | 1.196032 | 1.607475 | 12.7349% |

Both fake modes store 20.25 MiB of FP32 weights before metadata. Their ~1.196 ms cores are effectively the FP32 baseline; full fake-quant layers are ~1.607 ms, about 34.3% slower than that run’s FP32 layer. This demonstrates the overhead of the isolated fake-quant path and the absence of compressed weight storage in that operator.

The raw cross-run layer ratios are ~9.35x (fake W8A8 / experimental INT8) and ~4.17x (fake W4A8 / experimental packed INT4). These are comparisons of different quantization configurations: the upstream path uses affine quantization initialized with max scales; the integer path uses symmetric per-token activation quantization. No CSB/SSC/AdaRound checkpoint is loaded. Do not report these ratios as equivalent-quality acceleration of calibrated PTQ4DiT.

## Memory

Including scale metadata, integer weight storage is 5.080 MiB (INT8) and 2.549 MiB (INT4), versus 20.25 MiB FP32 and 10.125 MiB FP16/BF16. This is approximately 4x/8x compression relative to FP32 weights, or 2x/4x relative to 16-bit weights.

Every layer returns a 36 MiB FP32 output. Incremental allocation is 36 MiB for FP32, 54 MiB for FP16/BF16, 38.258 MiB for INT8, 38.883 MiB for INT4, and 90 MiB for each upstream fake mode. The 16-bit modes temporarily retain a 16-bit result while converting it to FP32. These numbers concern this boundary contract and allocator behavior, not an all-BF16 deployment.

Total harness peaks are 110.39 MiB FP32, 143.03 MiB FP16/BF16, 120.61 MiB INT8, and 118.36 MiB INT4. FP32 source tensors, reference outputs and prepared operands remain resident. Consequently this experiment establishes compressed weight storage, not a 4x/8x reduction in peak model memory.

## Instruction and resource evidence

The saved PTX contains mma.sync ... s32.s8.s8.s32 for both integer modes. Local cuobjdump disassembly of the saved cubins additionally confirms IMMA.16832.S8.S8.SAT instructions. See int8_sass.txt and int4_sass.txt generated during this analysis. These binaries and the profiler traces support that the experiment executes an integer matrix-instruction kernel; no runtime Tensor Core utilization percentage has been measured.

INT4 still multiplies INT8 operands after in-kernel unpacking; it does not execute native INT8-by-INT4 multiplication. Resource inspection reports 56 registers/thread for INT8 and 167 for INT4, with zero reported local memory/stack for both. The much larger INT4 register footprint makes occupancy/register pressure a strong profiling candidate, alongside unpacking and layout overhead. It does not by itself prove the cause of slowdown. Static instruction-site counts differ because code generation/loop structure differs; they are not dynamic operation counts.

## Timing limitations

- FP16 changes between runs: main core/layer medians are 0.1472/0.2280 ms, versus 0.1157/0.1911 ms in the upstream run. Within the main layer measurement, the first ten samples have median 0.2284 ms and the last ten 0.1924 ms. Avoid a strong FP16-versus-BF16 ranking from one sequential pass.
- INT4’s core median (0.4011 ms) exceeds its full-layer median (0.3854 ms), although the layer performs extra quantization. The core’s first-ten median is 0.4018 ms and last-ten median 0.3623 ms. Independent timings occurred under changing conditions; this is not negative quantization overhead. Do not subtract these medians to derive a component cost.
- Main BF16 and INT8 layer p10–p90 intervals are 0.180905–0.182176 ms and 0.171386–0.173548 ms, respectively. The INT8 advantage is clear within this pass, but independent repeated/interleaved runs are needed to establish reproducibility.
- GPU clock snapshots differ between the two runs (1095 vs 1185 MHz). These snapshots do not show clocks throughout timing or establish why measurements drifted.
- Profiling tables contain nested/overlapping attribution, with some percentages exceeding 100%. Do not sum operator rows. Use unprofiled timing for latency and individual GPU kernel events for attribution.
- One-time INT8 preparation (~1.249 s) includes first-use work; it is not a per-denoising-step cost or a clean isolated packing benchmark.

## Next experiments

1. Repeat and interleave modes after a longer stabilization period; preserve every run and compare native 16-bit layer boundaries as well as FP32 boundaries.
2. Use the FP16/BF16 paths as the practical baseline and the INT8 path as the primary optimization experiment. Capture real FC1 inputs at multiple blocks/timesteps, then test FC2 separately.
3. Profile INT4 register pressure, occupancy, unpacking and layout. Try tile/pipeline variants or another supported integer GEMM implementation, changing one factor at a time. Keep INT4 as a compression experiment until speed and error improve.
4. Preserve the exact upstream checkout/diff and calibration state before making a faithful PTQ4DiT comparison. Use the same quantization semantics in fake and integer implementations.
5. Only after those gates, measure matched 50-step model runs, latent error, image quality and deployment memory. Repeat the experiment on L40 to obtain an L40 conclusion.

Supported claim: On this A100 synthetic FC1 workload, the experimental INT8 implementation is numerically correct for its quantization scheme, emits integer matrix instructions, compresses weights by about 4x relative to FP32, and slightly reduces layer latency under FP32 boundaries. It does not beat BF16 prepared-input GEMM. Packed INT4 compresses weights further but currently trades away substantial speed and numerical fidelity.
