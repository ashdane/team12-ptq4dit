"""A100/L40 experiments: INT8 dot, optional in-register INT4 unpacking.

No FP weight expansion, zero-points, or PTQ4DiT calibration is implied.
Fixed conservative tiles avoid including autotuning in measurements.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _quantize(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) * (1.0 / 127.0), 1e-8)
    q = libdevice.nearbyint(tl.div_rn(x, scale))
    q = tl.minimum(tl.maximum(q, -127.), 127.).to(tl.int8)
    tl.store(Q + row * K + k, q, k < K)
    tl.store(S + row, scale)


@triton.jit
def _gemm(X, W, SX, SW, BIAS, Y,
          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          PACKED: tl.constexpr, RAW: tl.constexpr, HAS_BIAS: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + ks
        a = tl.load(X + m[:, None] * K + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K), 0)
        if PACKED:
            byte = tl.load(W + n[None, :] * tl.cdiv(K, 2) + k[:, None] // 2,
                           (n[None, :] < N) & (k[:, None] < K), 0).to(tl.int32)
            nibble = (byte >> ((k[:, None] % 2) * 4)) & 15
            w = ((nibble ^ 8) - 8).to(tl.int8)
        else:
            w = tl.load(W + n[None, :] * K + k[:, None],
                        (n[None, :] < N) & (k[:, None] < K), 0)
        # Match the explicit INT32 accumulator on Triton versions whose dot
        # output dtype otherwise defaults to FP32.
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    if RAW:
        result = acc
    else:
        sx = tl.load(SX + m, m < M, 0)
        sw = tl.load(SW + n, n < N, 0)
        result = acc.to(tl.float32) * sx[:, None] * sw[None, :]
        if HAS_BIAS:
            result += tl.load(BIAS + n, n < N, 0)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], result,
             (m[:, None] < M) & (n[None, :] < N))


def quantize_activation(x):
    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous() or x.dtype != torch.float32:
        raise ValueError("Expected contiguous CUDA FP32 [M,K] input")
    if not 0 < x.shape[1] <= 16384:
        raise ValueError("Activation quantizer supports 1 <= K <= 16384")
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    _quantize[(x.shape[0],)](x, q, s, x.shape[1], triton.next_power_of_2(x.shape[1]))
    return q, s


def integer_linear(xq, weight, sx, sw, bias=None, *, packed=False, raw=False,
                   return_kernel=False):
    m, k = xq.shape
    n = weight.shape[0]
    if k > 131071:
        raise ValueError("K is too large for worst-case signed INT8 accumulation")
    tensors = [xq, weight, sx, sw] + ([] if bias is None else [bias])
    if any(not t.is_cuda or not t.is_contiguous() or t.device != xq.device for t in tensors):
        raise ValueError("Operands must be contiguous and on the same CUDA device")
    if xq.dtype != torch.int8 or weight.dtype != (torch.uint8 if packed else torch.int8):
        raise ValueError("Wrong integer operand dtypes")
    if weight.shape != (n, (k + 1) // 2 if packed else k):
        raise ValueError("Weight shape does not match activation K")
    if sx.shape != (m,) or sw.shape != (n,) or sx.dtype != torch.float32 or sw.dtype != torch.float32:
        raise ValueError("Expected FP32 per-row activation/per-output weight scales")
    if bias is not None and (bias.shape != (n,) or bias.dtype != torch.float32):
        raise ValueError("Expected FP32 bias [N]")
    y = torch.empty((m, n), device=xq.device, dtype=torch.int32 if raw else torch.float32)
    kernel = _gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
        xq, weight, sx, sw, bias if bias is not None else sw, y, m, n, k,
        packed, raw, bias is not None, 32, 64, 64, num_warps=4, num_stages=3,
        enable_fp_fusion=False)
    return (y, kernel) if return_kernel else y
