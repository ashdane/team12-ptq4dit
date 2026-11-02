# Accelerating PTQ4DiT on NVIDIA Blackwell

## Research plan, optimization opportunities, and baseline preparation

> **Current scope:** Prepare a reproducible baseline and a focused FC1
> experiment before making claims about low-bit Tensor Core
> acceleration. The immediate objective is faster DiT inference. Reduced
> weight/activation memory is an additional benefit to measure, not an
> assumed result.

------------------------------------------------------------------------

## 1. Project objective

PTQ4DiT applies post-training quantization to Diffusion Transformers
(DiT), targeting W8A8 and W4A8 configurations. The project aims to
improve the *actual GPU execution* of the quantized model on NVIDIA
Blackwell.

The central research question is:

> Can we execute PTQ4DiT's low-bit linear operations efficiently on
> Blackwell Tensor Cores---avoiding unnecessary dequantization and
> intermediate memory traffic---while preserving image-generation
> quality?

The project should distinguish three outcomes:

1.  **Numerical quantization:** values are rounded to a low-bit grid.
2.  **Compressed representation:** weights are physically stored in
    INT8/INT4 form.
3.  **Low-bit hardware execution:** GEMM consumes low-bit operands using
    an appropriate GPU instruction path.

These are not interchangeable. A model can simulate INT4 quantization
while storing and multiplying FP32 tensors.

## 2. Current PTQ4DiT approach

### 2.1 Quantization method

PTQ4DiT (Post-Training Quantization for Diffusion Transformers)
introduces techniques intended to make low-bit quantization work well
for DiT:

-   **CSB (Channel-wise Salience Balancing):** balances activation and
    weight channel salience to reduce the effect of outlier channels.
-   **SSC (Spearman's rho-guided Salience Calibration):** accounts for
    changes in activation salience across diffusion timesteps.
-   **Offline re-parameterization:** folds suitable balancing
    transformations into model parameters/preceding operations so that
    the transformations need not be applied as separate runtime
    operations.

The configurations discussed in this project are:

-   **W8A8:** 8-bit weights and 8-bit activations.
-   **W4A8:** 4-bit weights and 8-bit activations.

### 2.2 Quantize-dequantize execution in the released PyTorch path

In the released `QuantModule` path discussed so far, the quantizer
computes quantized values, reconstructs floating-point values, and
passes those values to `F.linear`.

Conceptually:

``` text
FP activation and FP weight
          |
     fake quantize
          |
   dequantize to FP
          |
     floating-point
       F.linear
          |
        output
```

For example, the quantizer's conceptual operations are:

``` python
q = clamp(round(x / scale) + zero_point, qmin, qmax)
x_dequant = (q - zero_point) * scale
return x_dequant
```

The returned tensor is floating point. This simulates the numerical
effect of quantization; it does not by itself establish that the GEMM
uses packed INT8/INT4 operands or integer Tensor Core instructions.

**Important:** This describes the released PyTorch quantization path
inspected in the discussion. Before finalizing a reproduction, verify
the exact repository revision, configuration, and runtime path being
benchmarked. Do not generalize this implementation detail to every
possible implementation of the paper.

### 2.3 Why this creates an optimization opportunity

The simulated path can incur: - quantization/dequantization
operations; - intermediate tensor reads and writes; - additional kernel
launches; - floating-point GEMM rather than low-bit integer GEMM; -
storage of weights in floating-point form, despite simulating low-bit
values.

The precise overhead must be measured with the chosen repository
revision and profiler. Do not assume every pointwise operation becomes a
separate kernel or that every layer follows an identical path.

------------------------------------------------------------------------

## 3. Target workload and hardware

The presentation discussed the following target setup:

  Item                 Configuration
  -------------------- -----------------------------------------------------
  Model                DiT-XL/2
  Parameters           Approximately 675M
  Transformer depth    28 blocks
  Hidden dimension     1152
  Image resolution     256 × 256
  Latent shape         `[B, 4, 32, 32]`
  Patch size           2
  Tokens per image     256
  Sampling             50 denoising steps in the discussed setup
  GPU                  NVIDIA RTX PRO 5000 Blackwell
  Architecture         Blackwell, `sm_120` (as stated in the presentation)
  Quantization modes   FP32 reference, W8A8, W4A8

For the presentation's CFG example, batch size `B=4` becomes an
effective model batch of `2B=8`. Flattening batch and token dimensions
gives an activation matrix of shape:

``` text
[8 * 256, 1152] = [2048, 1152]
```

The four projection types highlighted for an initial analysis are QKV,
attention output projection, MLP FC1, and MLP FC2. Confirm actual
invocation counts and shapes from the executed model rather than
assuming these are all GEMMs in the complete network.

### FC1 shape used for the first microbenchmark

For the discussed DiT-XL/2 setup:

``` text
X:  [2048, 1152]
W:  [1152, 4608]   # logical matrix orientation for Y = X @ W
Y:  [2048, 4608]
```

PyTorch `F.linear(input, weight)` commonly stores the weight as
`[out_features, in_features]` and computes `input @ weight.T`. Keep this
storage convention explicit when comparing implementations.

------------------------------------------------------------------------

## 4. Optimization opportunities

The optimizations below are candidates, not completed results. Each
should be implemented and evaluated as an isolated ablation before
combining them.

### 4.1 Actual W8A8 integer GEMM

**Priority: first low-bit GEMM target**

Represent activations and weights as INT8 and execute integer matrix
multiplication:

``` text
FP activation -> quantize to INT8 ----\
                                       +--> INT8 GEMM --> INT32 accumulator
FP weight     -> quantize to INT8 ----/                       |
                                                        scale + bias
                                                             |
                                                        output dtype
```

Mathematically, for simple per-tensor scales:

\[ A\_{32} = X_q W_q \]

\[ Y `\approx `{=tex}s_x s_w A\_{32} + b \]

Real implementations must correctly handle zero-points, signedness,
per-channel/group scales, padding, and accumulator correction terms
where applicable.

Why start here: - INT8 GEMM is a simpler first step than a custom mixed
INT8-by-INT4 kernel. - It establishes the integer Tensor Core execution
path. - It helps separate GEMM performance from INT4 packing/unpacking
overhead.

Use a supported Blackwell-compatible integer GEMM implementation and
verify generated instructions/profiler evidence. Do not infer Tensor
Core use solely from a high-level API call.

### 4.2 Actual W4A8 GEMM

**Priority: second low-bit GEMM target**

The target representation is INT8 activations and INT4 weights:

``` text
X: INT8
W: packed INT4 (two 4-bit values per byte)
```

Potential implementation paths:

1.  **Native low-bit instruction path:** use a supported
    instruction/library path that directly consumes the required operand
    formats, if available for the exact architecture, layout and
    operation.
2.  **Packed INT4 with on-the-fly unpacking:** load packed bytes,
    extract/sign-extend INT4 values to a supported integer operand
    format, and multiply with INT8 activations.

The second approach can still avoid reconstructing a full FP32 weight
matrix, but unpacking has a cost. It must be benchmarked against native
INT8 and FP16/BF16 alternatives.

Do not call a path "W4A8 Tensor Core GEMM" if it expands weights to
FP16, BF16 or FP8 before multiplication. That would be a different
execution format and should be reported separately.

### 4.3 Fused quantization and GEMM-related operations

**Priority: high after a correct GEMM baseline exists**

Potential pipeline:

``` text
Load FP activation
       |
Quantize activation in the kernel
       |
Load packed low-bit weights
       |
GEMM / Tensor Core computation
       |
Scale + bias + output conversion
       |
Store output
```

Potential benefits: - fewer intermediate tensors; - fewer global-memory
reads and writes; - fewer kernel launches; - less overhead between
quantization and GEMM.

Fusion should be designed around the actual GEMM implementation. A fused
kernel is not automatically faster: large GEMMs may be compute-bound,
while small operations may benefit more from launch reduction. Measure
each component.

Possible fusion opportunities include: - activation scaling +
quantization; - GEMM epilogue + scale + bias; - bias + GELU for the
first MLP projection, where numerically and structurally valid; - output
conversion/quantization for the next layer.

### 4.4 Offline re-parameterization and weight preparation

**Priority: early, because weights are static during inference**

Weights do not change from one diffusion timestep to the next during
ordinary inference. The activation tensor changes at each timestep; the
trained model weights remain fixed.

Therefore, perform reusable weight-side work offline or once when
loading the model:

-   CSB-related transformations that are mathematically valid to fold;
-   weight quantization;
-   INT4 packing;
-   precomputation of scales/zero-points and metadata;
-   layout conversion/reordering for the target GEMM kernel.

This avoids repeating weight preparation for every denoising step.

Do not claim thousands of per-step weight quantization kernels are
removed unless a profiler trace proves they were actually being
launched. Weight preparation should generally be counted as
offline/one-time work, not per-step work.

**Correctness requirement:** A transformation may only be folded across
operations when algebraically valid. In particular, arbitrary channel
scaling cannot generally be moved through a nonlinear operation such as
GELU:

\[ GELU(XD) `\ne `{=tex}GELU(X)D \]

The project should preserve the exact PTQ4DiT transformation semantics
and verify equivalence numerically.

### 4.5 Attention optimization

DiT repeatedly performs QKV projection, attention, and output
projection.

Candidate improvements: - use a well-optimized FlashAttention
implementation as an initial attention baseline; - avoid materializing
the full attention score matrix where possible; - fuse compatible
attention operations; - evaluate lower-precision Q/K/V projections
separately from the attention softmax; - investigate quantized attention
only after projection GEMMs are validated.

Attention softmax and its numerical sensitivity make this a separate
workstream from linear-layer GEMM. Do not assume a quantized QK/AV
implementation preserves quality without measurement.

### 4.6 MLP/FFN optimization

The MLP has two large linear projections:

\[ H = GELU(XW_1\^T + b_1) \]

\[ Y = HW_2\^T + b_2 \]

These are promising targets for low-bit GEMM.

Candidate improvements: - INT8 or W4A8 GEMM for FC1 and FC2; - fuse bias
and GELU where valid; - fuse output scaling/quantization for the next
layer; - tune GEMM tile sizes, layouts and pipeline stages; - reuse
packed weights across all diffusion timesteps.

Start with FC1 as a standalone experiment, then test FC2. FC2 follows
GELU, so activation scaling and quantization must be handled without
incorrectly moving transformations through GELU.

### 4.7 LayerNorm and AdaLN fusion

Candidate improvements: - fuse normalization reductions and affine
operations where practical; - fuse AdaLN modulation with a following
operation or quantization when algebraically valid; - avoid unnecessary
intermediate tensors; - use efficient warp-level reductions.

This is a secondary optimization. Profile first; do not prioritize it
over large GEMMs without evidence that it is a significant bottleneck.

### 4.8 CUDA Graphs and runtime overhead

Diffusion inference repeatedly executes a largely repeated computation
graph. CUDA Graphs may reduce CPU launch/orchestration overhead when
shapes, memory addresses and execution structure are compatible.

Other runtime work: - preallocate reusable activation buffers; - avoid
repeated allocations; - reuse packed weights; - avoid CPU-GPU
synchronization in the timed loop; - separate compilation/warmup from
steady-state timing.

CUDA Graphs reduce launch overhead; they do not reduce the arithmetic
performed by GEMMs.

### 4.9 Sampling algorithm improvements

Fewer denoising steps can reduce total inference time, for example by
using a faster sampler. This changes the sampling algorithm and
potentially image quality, so treat it as a separate experimental track.

First establish kernel-level acceleration with the same sampler and
number of steps. Only then evaluate faster samplers as an additional
end-to-end optimization.

------------------------------------------------------------------------

## 5. Which algorithms/optimizations should we start with?

Recommended order:

  --------------------------------------------------------------------------------------------
                         Order Work item                                 Reason
  ---------------------------- ----------------------------------------- ---------------------
                             1 Reproduce and profile original FC1        Establish a
                                                                         trustworthy baseline

                             2 FP16/BF16 FC1 baseline                    Fair comparison
                                                                         against modern
                                                                         floating-point Tensor
                                                                         Core execution

                             3 W8A8 integer FC1 GEMM                     Simplest actual
                                                                         low-bit GEMM
                                                                         experiment

                             4 W4A8 packed-weight FC1 GEMM               Test the main low-bit
                                                                         research direction

                             5 Fused activation quantization + GEMM      Reduce intermediate
                               epilogue                                  memory traffic and
                                                                         launch overhead

                             6 Offline weight                            Remove repeatable
                               quantization/packing/reparameterization   weight-side work from
                                                                         inference

                             7 Integrate into FC1/FC2, then              Expand to the
                               QKV/projection                            complete DiT

                             8 FlashAttention, AdaLN fusion, CUDA Graphs Optimize remaining
                                                                         bottlenecks based on
                                                                         profiling

                             9 Faster sampling algorithms                Separate algorithmic
                                                                         speedup from kernel
                                                                         speedup
  --------------------------------------------------------------------------------------------

**Recommended first deliverable:** a correct, reproducible FC1 baseline
on the local machine. Do not claim an INT8/INT4 speedup until the
corresponding implementation exists and is measured.

------------------------------------------------------------------------

## 6. FC1 local baseline: experiment to prepare now

The immediate task is to prepare a local FC1 benchmark harness.
Initially, it should measure existing floating-point implementations
only. It should not claim low-bit acceleration.

### 6.1 Fixed logical problem

Use the FC1 dimensions from the target workload:

``` text
M = 2048
K = 1152
N = 4608

X: [M, K]
W: [N, K]      # PyTorch F.linear storage convention
bias: [N]
Y: [M, N]
```

The mathematical operation is:

\[ Y = XW\^T + b \]

Use the same fixed input and weights for all implementations. For
reproducibility, initialize tensors with a fixed seed. Also test
representative real activations/weights from the model once the
synthetic harness works.

### 6.2 Baselines to measure

Prepare these modes where supported by the local GPU and software stack:

1.  FP32 `torch.nn.functional.linear`.
2.  FP16 `F.linear` / GEMM.
3.  BF16 `F.linear` / GEMM, if supported.
4.  Original PTQ4DiT fake-quantized FC1 path, if it can be isolated
    faithfully.

Do not include INT8/INT4 in the baseline results until those kernels are
implemented. They can be listed as future experiment slots.

### 6.3 Benchmark protocol

-   Record GPU model, compute capability, driver, CUDA, PyTorch and
    library versions.
-   Warm up each operation before timing.
-   Use CUDA events or a reliable GPU timing method.
-   Synchronize appropriately around measurements.
-   Run enough repetitions and report median and variability
    (e.g. p10/p90 or standard deviation).
-   Keep tensor shapes, batch, bias, input data and weight data
    identical across modes.
-   Separate one-time conversion/casting costs from steady-state GEMM
    time.
-   Also report end-to-end layer time including any required
    casts/conversions.
-   Avoid timing model loading, compilation or first-use initialization
    as steady-state execution.
-   Check output correctness against the FP32 reference.

### 6.4 Correctness metrics for the FC1 experiment

For a reference output (Y\_{ref}) and candidate output (Y\_{test}),
report:

\[ MSE = `\frac{1}{n}`{=tex}`\sum`{=tex}*i (Y*{test,i}-Y\_{ref,i})\^2 \]

\[ MaxAbsError = `\max`{=tex}*i \|Y*{test,i}-Y\_{ref,i}\| \]

\[ RelativeError = `\frac{\|Y_{test}-Y_{ref}\|_2}`{=tex}
{\|Y\_{ref}\|\_2+`\epsilon`{=tex}} \]

Also report cosine similarity.

FP16/BF16 outputs should not be expected to be bitwise identical to
FP32. Use tolerances and report the actual errors.

### 6.5 FC1 results table template

  -------------------------------------------------------------------------------
  Mode      Operand           GEMM   End-to-end   Max error        MSE       Peak
            formats        latency        layer                            memory
                                        latency                        
  --------- ----------- ---------- ------------ ----------- ---------- ----------
  FP32      FP32 × FP32        TBD          TBD   Reference          0        TBD

  FP16      FP16 × FP16        TBD          TBD         TBD        TBD        TBD

  BF16      BF16 × BF16        TBD          TBD         TBD        TBD        TBD

  PTQ4DiT   Fake               TBD          TBD         TBD        TBD        TBD
  fake W8A8 quantized                                                  
            FP tensors                                                 

  Actual    INT8 × INT8     Future       Future      Future     Future     Future
  W8A8                                                                 

  Actual    INT8 ×          Future       Future      Future     Future     Future
  W4A8      packed INT4                                                
  -------------------------------------------------------------------------------

This table is a template, not measured data.

------------------------------------------------------------------------

## 7. Inference speed versus memory: what should we optimize?

The immediate objective should be **faster inference**. Memory reduction
is a second objective that may also enable larger batches or
resolutions.

### 7.1 Weights do not change across diffusion timesteps

A key conceptual correction:

-   **Weights:** learned model parameters. They remain constant across
    denoising timesteps during ordinary inference.
-   **Activations:** intermediate tensors computed from the current
    noisy latent. They change at every timestep.

Therefore, do not assume the weights keep changing at every diffusion
timestep.

The same packed INT4/INT8 weights can be reused across all steps. This
makes offline quantization and packing particularly attractive.

### 7.2 Where memory savings can come from

For a model with (P) parameters, theoretical raw weight storage is:

  Format        Bytes per parameter   Theoretical storage relative to FP32
  ----------- --------------------- --------------------------------------
  FP32                            4                                     1×
  FP16/BF16                       2                                   0.5×
  INT8                            1                                  0.25×
  INT4                          0.5                                 0.125×

Thus, INT8 has a theoretical 4× reduction and INT4 an 8× reduction
relative to FP32 weight storage.

Actual memory consumption will be higher because of scales, zero-points,
alignment/padding, metadata, activations, temporary buffers and other
model parameters. If the original FP32 checkpoint remains resident, the
expected savings may not appear.

### 7.3 What to measure

Measure separately: - packed weight storage; - quantization metadata
storage; - peak allocated GPU memory; - peak reserved GPU memory; -
activation memory; - temporary workspace memory; - total model memory
after loading.

A smaller weight representation does not automatically guarantee lower
peak memory or faster inference.

### 7.4 Performance trade-offs

Potential trade-offs include:

-   INT4 reduces storage but may require unpacking or specialized
    instructions.
-   INT8 can be easier to execute efficiently but uses more storage than
    INT4.
-   Fusion reduces intermediate traffic but may increase register
    pressure.
-   Larger GEMM tiles may improve reuse but reduce occupancy.
-   Lower precision can increase numerical error.
-   CUDA Graphs reduce launch overhead but do not accelerate the
    underlying arithmetic.

------------------------------------------------------------------------

## 8. Image-quality and numerical evaluation

Since this project changes numerical precision and inference execution,
it must establish both **implementation correctness** and **generative
quality**.

These are different evaluation levels.

### 8.1 Layer-level numerical correctness

For a custom FC1 kernel, compare its output with a trusted FP32 or
high-precision reference.

Report: - MSE; - maximum absolute error; - relative L2 error; - cosine
similarity; - optionally normalized error.

This tells us whether the kernel computes the intended operation
accurately.

It does not establish image quality.

### 8.2 Latent-level fidelity

Use identical initial latent noise, class labels/conditioning,
timesteps, sampler settings and stochastic noise inputs for all
configurations.

Compare: - final latent MSE; - latent cosine similarity; - latent
relative error; - optionally intermediate denoising-step errors.

Useful comparisons include:

1.  PTQ4DiT W8A8 versus FP32.
2.  PTQ4DiT W4A8 versus FP32.
3.  Optimized W8A8 versus simulated W8A8.
4.  Optimized W4A8 versus simulated W4A8.
5.  Optimized W4A8 versus FP32.

Do not use W8A8-versus-W4A8 similarity as if it were similarity against
FP32. Those are different comparisons.

### 8.3 Image-generation quality

Decode generated latents into images and evaluate with metrics used in
generative-model research.

  -----------------------------------------------------------------------
  Metric                  What it measures        Interpretation
  ----------------------- ----------------------- -----------------------
  FID                     Distribution distance   Lower is better
                          between generated and   
                          real image features     

  sFID                    Spatial feature         Lower is better
                          distribution distance   

  IS                      Class confidence and    Higher is generally
                          diversity proxy         better

  Precision               Fidelity/quality of     Higher is better
                          generated samples       

  Recall                  Coverage/diversity of   Higher is better
                          generated distribution  
  -----------------------------------------------------------------------

For ImageNet class-conditional generation, evaluate using a sufficiently
large and consistent sample set and the same evaluation
pipeline/reference statistics.

FID and sFID are distribution-level metrics; a few visually pleasing
images are not enough to establish them reliably.

### 8.4 Runtime and systems metrics

  Metric                       Purpose
  ---------------------------- ---------------------------------
  FC1/GEMM latency             Isolated kernel performance
  Per-layer latency            Cost across a transformer block
  Per-step denoising latency   Model execution speed
  End-to-end generation time   Actual user-visible speed
  Images per second            Throughput
  Peak GPU memory              Memory efficiency
  DRAM bytes/bandwidth         Data movement
  Tensor Core utilization      Hardware execution evidence
  Kernel launch count          Launch overhead
  Power/energy, if available   Efficiency

Use the same sampling steps, batch size, resolution, CFG settings and
hardware when comparing end-to-end generation time.

### 8.5 Speed-quality trade-off

The main result should be a table or plot that reports speed and quality
together.

  Configuration           Latency      FID     sFID   Peak memory
  --------------------- --------- -------- -------- -------------
  FP32 reference              TBD      TBD      TBD           TBD
  FP16/BF16 reference         TBD      TBD      TBD           TBD
  PTQ4DiT W8A8                TBD      TBD      TBD           TBD
  PTQ4DiT W4A8                TBD      TBD      TBD           TBD
  Optimized W8A8           Future   Future   Future        Future
  Optimized W4A8           Future   Future   Future        Future

A good result is not merely the lowest latency. It is a meaningful
latency or memory improvement with acceptable quality degradation.

------------------------------------------------------------------------

## 9. Blackwell-specific implementation and profiling

The target is NVIDIA Blackwell (`sm_120`, as stated in the project
presentation). Kernel choices must be verified for this architecture and
the installed software stack.

### Candidate implementation tools

-   **CUTLASS:** optimized GEMM building blocks, layouts, epilogues and
    architecture-specific kernel paths.
-   **CUDA C++ / PTX:** low-level control over memory movement and
    supported matrix instructions.
-   **Triton:** custom fused quantization and GPU kernels.
-   **cuBLAS:** FP32, FP16/BF16 and supported integer baselines.
-   **Nsight Systems:** timeline, launch overhead, synchronization and
    CPU/GPU activity.
-   **Nsight Compute:** kernel-level throughput, memory traffic,
    occupancy, registers and Tensor Core instruction activity.
-   **PyTorch Profiler:** operator-level attribution and shape
    information.

### Important verification rule

Do not claim that a kernel uses Tensor Cores just because: - it is
written in Triton; - it calls `tl.dot`; - the input is stored in
INT8/INT4; - weights are packed into bytes.

Verify the generated instructions and/or profiler counters for the
actual kernel and architecture.

Also verify the exact supported operand formats. Do not assume that
every Blackwell Tensor Core instruction accepts arbitrary INT8 × INT4
operands.

------------------------------------------------------------------------

## 10. Proposed ablation plan

Keep each optimization isolated so the source of any improvement is
clear.

  -----------------------------------------------------------------------
  Experiment              Change from previous    Main question
                          experiment              
  ----------------------- ----------------------- -----------------------
  E0                      FP32 FC1 baseline       What is the reference
                                                  latency?

  E1                      FP16/BF16 FC1           How fast is a strong
                                                  floating-point Tensor
                                                  Core baseline?

  E2                      PTQ4DiT fake-quant FC1  What overhead does the
                                                  current simulated path
                                                  add?

  E3                      Actual W8A8 GEMM        Does integer Tensor
                                                  Core GEMM improve the
                                                  layer?

  E4                      Actual W4A8 GEMM        Does lower-bit
                                                  storage/execution
                                                  improve further?

  E5                      Fuse activation         Does fusion reduce
                          quantization and GEMM   total layer time?
                          epilogue                

  E6                      Offline weight          What one-time work and
                          quantization and        memory are removed?
                          packing                 

  E7                      Integrate optimized     Does the improvement
                          FC1/FC2                 survive in a real
                                                  transformer block?

  E8                      Add attention/runtime   What is the end-to-end
                          optimizations           benefit?
  -----------------------------------------------------------------------

For every experiment, preserve: - exact code revision; -
model/checkpoint; - GPU and software versions; - shapes and dtypes; -
quantization scales and zero-points; - benchmark
command/configuration; - raw timing outputs; - correctness results.

------------------------------------------------------------------------

## 11. Risks and questions to resolve

1.  **Exact PTQ4DiT quantization semantics:** Confirm signedness,
    quantization ranges, zero-points, scale granularity, and whether the
    project preserves the original CSB/SSC calibration.
2.  **Blackwell instruction support:** Confirm the exact integer operand
    combinations and supported GEMM paths for `sm_120`.
3.  **INT4 unpacking overhead:** Compare packed storage savings against
    the cost of unpacking and layout conversion.
4.  **Scale handling:** Per-channel or group-wise scales may complicate
    the GEMM epilogue and accumulator correction.
5.  **Numerical error accumulation:** Small errors in individual layers
    may accumulate across 28 blocks and 50 denoising steps.
6.  **Fair baselines:** FP16/BF16 Tensor Core baselines are necessary;
    FP32 alone is insufficient.
7.  **Memory accounting:** Do not present theoretical compressed storage
    as measured peak GPU memory.
8.  **Quality evaluation:** Latent similarity and PSNR are useful
    diagnostics, but do not replace FID/sFID and other generation
    metrics.
9.  **Fusion trade-offs:** A fused kernel may use more registers or
    reduce occupancy; measure rather than assume it is faster.
10. **Sampling changes:** Keep sampler and number of denoising steps
    fixed when measuring kernel-only speedup.

------------------------------------------------------------------------

## 12. Immediate next steps

### Deliverable 1: Prepare the FC1 baseline

-   [ ] Create a standalone local benchmark for
    `M=2048, K=1152, N=4608`.
-   [ ] Implement/reference FP32 `F.linear`.
-   [ ] Add FP16 and BF16 baselines where supported.
-   [ ] Reproduce the fake-quantized FC1 path if possible.
-   [ ] Validate output errors against FP32.
-   [ ] Record warm steady-state latency and end-to-end layer latency.
-   [ ] Record actual memory usage and software/hardware configuration.
-   [ ] Save results in CSV/JSON with the exact benchmark configuration.

### Deliverable 2: Establish the low-bit implementation path

-   [ ] Identify the exact quantization semantics used by PTQ4DiT.
-   [ ] Select a supported Blackwell INT8 GEMM path.
-   [ ] Implement and validate W8A8 FC1.
-   [ ] Investigate packed W4A8 FC1.
-   [ ] Verify actual Tensor Core instruction execution.
-   [ ] Compare speed, memory and numerical error.

### Deliverable 3: Integrate and evaluate

-   [ ] Add fusion only after the standalone GEMM is correct.
-   [ ] Integrate into DiT FC1/FC2.
-   [ ] Evaluate latent-level fidelity with matched random inputs.
-   [ ] Evaluate image-generation quality using FID/sFID and related
    metrics.
-   [ ] Measure full 50-step generation latency.
-   [ ] Run ablations and report trade-offs.

------------------------------------------------------------------------

## 13. Expected initial research claim

At the current stage, do **not** claim that the project has achieved
INT4/INT8 Tensor Core acceleration, a specific speedup, or a particular
memory reduction.

A defensible initial statement is:

> We are investigating hardware-efficient inference for PTQ4DiT on
> NVIDIA Blackwell. The initial work establishes reproducible FC1
> baselines across FP32 and floating-point Tensor Core formats. We will
> then evaluate actual INT8 and packed-INT4 GEMM implementations, fused
> quantization/epilogue kernels, and offline weight preparation.
> Performance will be assessed at both layer and end-to-end DiT levels,
> with numerical and generative-quality metrics to quantify the
> speed-quality-memory trade-off.

**Immediate priority:** get the FC1 baseline correct and reproducible.
It provides the reference needed to judge every subsequent kernel
optimization.