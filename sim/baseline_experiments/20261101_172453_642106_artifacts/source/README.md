# FC1 experiments for Colab A100 / L40

Implementation of the **immediate FC1 deliverable** in `agent.md`, with optional
integer-kernel and ordinary-Linear integration experiments. Default shape:
`X[2048,1152]`, `weight[4608,1152]`, `bias[4608]`, `Y = X @ weight.T + bias`.

**Start with [notebooks/FC1_Colab.ipynb](notebooks/FC1_Colab.ipynb).** Upload this
notebook to Colab, choose a GPU runtime, and run its cells in order. It embeds
the source and tests, so there is no repository URL to edit or source ZIP to
upload. The GPU tests run before the integer benchmark. Download the result
archive from the final cell.

The same notebook can connect to an L40-backed runtime. Colab hardware allocation
is outside this project's control; selecting a runtime does not guarantee an L40.
A100 is compute capability 8.0 and L40 is 8.9, according to
[NVIDIA's GPU list](https://developer.nvidia.com/cuda/gpus).
Triton detects the runtime architecture; no Blackwell `sm_120` build is used.

## What is implemented

| Mode | Storage / computation | Status |
|---|---|---|
| `fp32` | FP32 `F.linear`, TF32 disabled | Default baseline |
| `fp16`, `bf16` | Pre-cast weights, native `F.linear` | Default baselines |
| `fake_w8a8`, `fake_w4a8` | Actual upstream `QuantModule`, floating-point GEMM | Optional upstream checkout required |
| `int8` | INT8 activations/weights, INT32 accumulation | Experimental Triton kernel |
| `int4` | Two signed weights per byte; unpack to INT8 inside GEMM | Experimental Triton kernel |

The integer paths use dynamic symmetric per-token activation scales and static
per-output-channel weight scales. A8 uses `[-127,127]`, W8 `[-127,127]`, and W4
`[-7,7]`, round-to-nearest-even, zero-points zero, scale floor `1e-8`.
The INT4 storage/kernel also handles `-8`, which is covered by the tests.
They fuse scale, bias and FP32 output into the GEMM epilogue. Activation reduction
and quantization run in a separate fused kernel. Weight preparation happens once.
**INT4 is packed storage with INT8 multiplication, not native INT8-by-INT4 MMA.**

This is executable research code, not a measured speedup. The development host
has no usable CUDA driver or PyTorch installation. Local validation covers
Python syntax, notebook packaging and dependency-free tests; numerical CPU and
GPU tests must run in the supplied GPU notebook. No GPU performance or image
quality numbers are fabricated.

## Run outside Colab

Use a Linux environment with CUDA-enabled PyTorch and its matching Triton
dependency. The notebook uses Colab's installed pair rather than upgrading them
independently. `requirements.txt` describes the Python dependency; GPU driver and
CUDA-enabled wheel selection depend on the host.

```bash
python -m unittest discover -s tests -v
python -m fc1bench.benchmark --output results/baseline
python -m fc1bench.benchmark --modes fp32 fp16 bf16 int8 int4 \
  --profile --output results/integer_experiment
```

Use distinct output directories: the CLI refuses to overwrite `results.json`.
Any implemented mode that fails correctness or compilation gets `status=failed`
and the process exits nonzero. Missing upstream data and unsupported BF16 are
explicit skips. Integer tests are skipped on machines without CUDA/Triton, so
passing the CPU suite does **not** establish GPU correctness.

Quick CPU smoke test (not a performance experiment):

```bash
python -m fc1bench.benchmark --device cpu --m 5 --k 7 --n 9 \
  --modes fp32 --warmup 1 --repeats 2 --inner 1 --output results/cpu_smoke
```

## Measurement contract

* Identical seeded CPU-generated FP32 data is used in every mode. Weights are
  scaled by `1/sqrt(K)` to keep output magnitudes representative. Captured inputs
  can replace synthetic data with `--input capture.pt`.
* `core_timing` uses already prepared operands. FP modes include bias in
  `F.linear`; integer modes include scale, bias and output conversion in GEMM.
  Fake-quant core timing uses precomputed dequantized inputs/weights.
* `layer_timing` starts with FP32 input and returns FP32 output. FP16/BF16 include
  activation casts and output conversion; integer modes include dynamic
  quantization; upstream fake modes include their actual per-call quantizers.
  Weight casting/packing is excluded from steady-state timing in all cases.
  These layer boundaries differ from an all-FP16 model; use core timing for the
  native floating-point comparison.
* Warmup defaults to 20 calls. Each of 50 timing samples averages 10 calls.
  Both CUDA events and synchronized wall time are saved, with median, p10/p90,
  standard deviation and every raw sample. This is warm eager execution, without
  CUDA Graphs or cache flushing. Host launch gaps may appear in event intervals.
* Preparation wall time includes conversion, calibration and any first-use JIT
  triggered during preparation. All compilation and correctness checks finish
  before steady-state timing. Profiling is a separate, untimed pass.
* FP32 reference disables TF32. All outputs are checked for finiteness; MSE,
  maximum absolute error, relative L2 and cosine similarity are reported in FP64.
  Integer accumulators are checked exactly against an FP64 integer-product
  reference, separately from quantization error against FP32. The FP64 validation
  can take time on L40 and is excluded from timing.
* The default kernels use fixed `32x64x64` tiles, four warps, three stages.
  There is no tuning claim. Benchmark both GPUs independently.

### Memory interpretation

JSON/CSV report physical weight bytes, scale metadata bytes, bias bytes, allocated
and reserved peak memory, initial allocation, incremental peak allocation, and
output bytes. Activation-scale bytes are recorded for integer modes. Memory is
measured in a separate synchronized call after clearing unused allocator cache.

The harness retains FP32 data/reference and prepared operands for comparisons.
Therefore **absolute peak memory is the harness working set, not deployment model
memory**. Incremental allocation includes output and temporary buffers but excludes
already resident prepared inputs. Reserved memory reflects the PyTorch allocator,
not every driver/library allocation. Weight storage reductions must not be
reported as full-model peak-memory reductions.

## Faithful upstream operator isolation

The optional notebook cell clones the
[official PTQ4DiT repository](https://github.com/adreamwu/PTQ4DiT) and records its
resolved commit. For repeated experiments set `PTQ4DIT_REV` to that commit.
The adapter imports the checkout's `quant/quant_layer.py` directly, without
substituting a generic fake quantizer. It logs revision, dirty state, source hash,
quantizer settings, scales and zero-points.

```bash
git clone https://github.com/adreamwu/PTQ4DiT.git third_party/PTQ4DiT
python -m fc1bench.benchmark --ptq4dit-repo third_party/PTQ4DiT \
  --modes fp32 fp16 bf16 fake_w8a8 fake_w4a8 \
  --upstream-scale-method max --output results/upstream
```

The default `max` initialization is a quick isolated-operator experiment.
`--upstream-scale-method mse` selects upstream's more expensive initialization.
Both initialize on the benchmark tensor, not a representative calibration set.
The inspected [upstream sampling setup](https://github.com/adreamwu/PTQ4DiT/blob/main/quant_sample.py)
uses asymmetric activation quantization, per-channel weight quantization and
calibration/reconstruction. **An isolated QuantModule is not calibrated PTQ4DiT
with CSB/SSC/AdaRound.** The symmetric integer kernels cannot replace that affine
quantizer without further scale/zero-point and calibration work. No channel
transformation is moved through GELU here.

## Real FC1/FC2 inputs and model experiments

Capture an ordinary layer before replacing it, during your existing DiT run:

```python
from fc1bench.integration import capture_next_input

handle = capture_next_input(
    model.blocks[0].mlp.fc1, "fc1_real.pt",
    metadata={"checkpoint": "your-checkpoint-and-hash", "block": 0,
              "seed": 2026, "sampling_steps": 50, "cfg_scale": 1.5,
              "timestep": "record-the-actual-timestep", "revision": "your-commit"},
)
# Run your existing model/sampler once; the hook saves the next layer invocation.
# Call handle.remove() if you abandon the run before the layer executes.
```

Then run `python -m fc1bench.benchmark --input fc1_real.pt --output results/real`.
Captured `[B,T,K]` input is flattened to `[B*T,K]`. Actual dimensions, input file
hash and supplied model metadata are recorded. Use separate captures across
timesteps/blocks; a single FC1 sample does not characterize the entire model.

For an **uncalibrated symmetric-quantization ablation** on ordinary DiT linears:

```python
from fc1bench.integration import replace_linears

model.eval().cuda()
replace_linears(model, ["blocks.0.mlp.fc1", "blocks.0.mlp.fc2"], bits=8)
with torch.inference_mode():
    output = model(x, t, y)
```

The wrapper preserves leading input dimensions and returns the input dtype.
It refuses calibrated `QuantModule` conversion. Explicit layer names prevent
accidentally quantizing sensitive conditioning/final layers. Do not apply
`.half()`/`.bfloat16()` to the model after replacement: the kernel requires FP32
scales and bias. Integration is inference-only and adds boundary conversions;
measure actual per-step latency before claiming a benefit.

## Profiling and reproducibility artifacts

`results.json` records software/GPU/driver configuration, source hashes, seed or
capture hash, shapes, settings, raw timings and numerical errors. `results.csv`
is the flat summary. Quantization artifacts retain packed weights, scales and
upstream zero-points. Optional traces include operator shapes and memory.
The notebook also saves `pip freeze` and a copy of the implementation.

Each integer mode saves generated `.ptx` and `.cubin`. JSON reports whether
signed INT8 `mma` instructions appear in PTX. This provides compiler evidence,
not profiler-counter verification. The
[Triton dot documentation](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
lists INT8 operands; an API call alone is not hardware evidence.

On a host with NVIDIA profiling tools and counter access, inspect the cubin with
`cuobjdump --dump-sass results/integer_experiment/int8.cubin`, and profile a
short run, for example:

```bash
ncu --target-processes all --set full --kernel-name-base function \
  --kernel-name regex:_gemm --launch-count 1 -o fc1_integer \
  python -m fc1bench.benchmark --modes int8 --warmup 1 --repeats 1 --inner 1 \
  --output results/ncu
```

Inspect integer matrix instructions and the architecture's Tensor Core counters.
Profilers may not have the required permissions in hosted Colab. A missing
counter result is recorded as unverified; no Tensor Core utilization is invented.

## Remaining research gates from agent.md

The runnable deliverable is the FC1 baseline plus experimental integer kernels
and integration helpers. GPU execution must be validated on the allocated GPU.
Full calibrated PTQ4DiT equivalence requires a specific checkpoint, calibration
artifacts and faithful CSB/SSC/AdaRound conversion. No such artifacts were supplied.
Matched 50-step sampling, latent comparisons, FID/sFID/IS/precision/recall and
full-model speed/memory measurements remain experiments to run on that model.
FlashAttention, AdaLN fusion, CUDA Graphs and sampler changes are subsequent
profile-driven ablations, as specified in `agent.md`.

Rebuild the standalone notebook/ZIP after source edits:

```bash
python scripts/build_colab.py
```
