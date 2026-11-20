"""Triton kernel implementations for low-bit PTQ4DiT acceleration.

Includes:
1. Fused dynamic activation quantization with channel-wise CSB scaling (supporting arbitrary K and post-GELU).
2. INT8 x INT8 GEMM with INT32 accumulation, fused scale/bias epilogue, and optional zero-point correction.
3. Packed INT4 weight execution with on-the-fly register-level unpacking and INT8 Tensor Core math.
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    libdevice = None
    HAS_TRITON = False


def _triton_jit(fn):
    if HAS_TRITON:
        return triton.jit(fn)
    return fn


@_triton_jit
def _quantize_fused(
    X, Q, S, ZX_OUT, BX,
    M, K,
    HAS_BX: tl.constexpr,
    AFFINE: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    """
    Fused dynamic activation quantization kernel.
    - Loops over K in chunks of BLOCK_K to support arbitrary and large K (e.g. K=4608 in fc2).
    - If HAS_BX is True, multiplies by channel scaling vector BX [K] (CSB balancing / post-GELU).
    - Computes dynamic per-row scale in registers without intermediate global-memory tensors.
    - If AFFINE is True, computes scale and zero-point; otherwise computes symmetric scale.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    # Pass 1: Row-wise reduction to find max_abs (or min and max for affine)
    max_abs = 0.0
    min_val = 1e30
    max_val = -1e30

    for start_k in range(0, K, BLOCK_K):
        k_offs = start_k + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(X + row * K + k_offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_BX:
            bx = tl.load(BX + k_offs, mask=mask, other=1.0).to(tl.float32)
            x = x * bx

        if AFFINE:
            x_for_max = tl.where(mask, x, -1e30)
            x_for_min = tl.where(mask, x, 1e30)
            max_val = tl.maximum(max_val, tl.max(x_for_max, 0))
            min_val = tl.minimum(min_val, tl.min(x_for_min, 0))
        else:
            abs_x = tl.abs(x)
            max_abs = tl.maximum(max_abs, tl.max(abs_x, 0))

    if AFFINE:
        x_min = tl.minimum(min_val, 0.0)
        x_max = tl.maximum(max_val, 0.0)
        scale = tl.maximum((x_max - x_min) * (1.0 / 255.0), 1e-8)
        # Round-to-nearest-even zero point mapping x_min to -128, 0 to zp, x_max to 127
        zp = libdevice.nearbyint(-128.0 - x_min / scale)
        zp = tl.minimum(tl.maximum(zp, -128.0), 127.0)
        tl.store(S + row, scale)
        tl.store(ZX_OUT + row, zp.to(tl.int32))
    else:
        scale = tl.maximum(max_abs * (1.0 / 127.0), 1e-8)
        tl.store(S + row, scale)

    # Pass 2: Quantize and store
    for start_k in range(0, K, BLOCK_K):
        k_offs = start_k + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(X + row * K + k_offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_BX:
            bx = tl.load(BX + k_offs, mask=mask, other=1.0).to(tl.float32)
            x = x * bx

        if AFFINE:
            q_f = libdevice.nearbyint(tl.div_rn(x, scale)) + zp
            q = tl.minimum(tl.maximum(q_f, -128.0), 127.0).to(tl.int8)
            tl.store(Q + row * K + k_offs, q, mask=mask)
        else:
            q_f = libdevice.nearbyint(tl.div_rn(x, scale))
            q = tl.minimum(tl.maximum(q_f, -127.0), 127.0).to(tl.int8)
            tl.store(Q + row * K + k_offs, q, mask=mask)


# Retain alias for backwards compatibility
_quantize = _quantize_fused


@_triton_jit
def _gemm(
    X, W, SX, SW, BIAS, Y, ZX, W_RED,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    PACKED: tl.constexpr, RAW: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_ZX: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """
    INT8 / packed INT4 GEMM kernel with INT32 accumulation and fused epilogue.
    """
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)

    for start in range(tl.cdiv(K, BK)):
        k = start * BK + ks
        a = tl.load(X + m[:, None] * K + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K), 0)
        if PACKED:
            # nk layout: row-major per output channel [N, cdiv(K, 2)]
            byte = tl.load(W + n[None, :] * tl.cdiv(K, 2) + k[:, None] // 2,
                           (n[None, :] < N) & (k[:, None] // 2 < tl.cdiv(K, 2)), 0).to(tl.int32)
            nibble = (byte >> ((k[:, None] % 2) * 4)) & 15
            w = ((nibble ^ 8) - 8).to(tl.int8)
        else:
            w = tl.load(W + n[None, :] * K + k[:, None],
                        (n[None, :] < N) & (k[:, None] < K), 0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)

    if RAW:
        tl.store(Y + m[:, None] * N + n[None, :], acc,
                 (m[:, None] < M) & (n[None, :] < N))
    else:
        sx = tl.load(SX + m, m < M, 0.0)
        sw = tl.load(SW + n, n < N, 0.0)
        if HAS_ZX:
            zx = tl.load(ZX + m, m < M, 0.0)
            w_red = tl.load(W_RED + n, n < N, 0.0)
            acc_corr = acc.to(tl.float32) - zx[:, None] * w_red[None, :]
            result = acc_corr * sx[:, None] * sw[None, :]
        else:
            result = acc.to(tl.float32) * sx[:, None] * sw[None, :]

        if HAS_BIAS:
            result += tl.load(BIAS + n, n < N, 0.0)[None, :]
        tl.store(Y + m[:, None] * N + n[None, :], result,
                 (m[:, None] < M) & (n[None, :] < N))


def quantize_activation(x, bx=None, affine=False, block_k=1024):
    """
    Fused dynamic activation quantization supporting FP16, BF16, and FP32 inputs,
    optional channel-wise scaling vector bx [K], and arbitrary K dimensions.
    Uses IEEE-754 round-to-nearest-even (banker's rounding).

    Returns:
        - If affine=False: (q, scale)
            q: [M, K] signed int8 in [-127, 127]
            scale: [M] float32
        - If affine=True:  (q, scale, zx)
            q: [M, K] signed int8 in [-128, 127]
            scale: [M] float32
            zx: [M] int32 in [-128, 127]
            Reconstruction: x ~= (q - zx) * scale
    """
    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("Expected contiguous CUDA 2D activation matrix [M, K]")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Activation dtype must be FP16, BF16, or FP32")
    m, k = x.shape
    if min(m, k) < 1:
        raise ValueError("Matrix dimensions must be positive")

    has_bx = bx is not None
    if has_bx:
        if not bx.is_cuda or bx.ndim != 1 or bx.shape[0] != k or not bx.is_contiguous():
            raise ValueError(f"Expected contiguous CUDA channel vector bx of shape [{k}] on same device")

    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty(m, device=x.device, dtype=torch.float32)
    zx_out = torch.empty(m, device=x.device, dtype=torch.int32) if affine else scale

    # Pick power-of-two BLOCK_K between 256 and 2048
    chosen_block = min(2048, max(256, triton.next_power_of_2(min(k, block_k))))
    _quantize_fused[(m,)](
        x, q, scale, zx_out, bx if has_bx else x,
        m, k,
        has_bx, affine, chosen_block
    )
    if affine:
        return q, scale, zx_out
    return q, scale


def integer_linear(xq, weight, sx, sw, bias=None, *, packed=False, raw=False,
                   zx=None, w_red=None, return_kernel=False):
    """
    INT8 / packed INT4 linear operator with INT32 accumulation and fused epilogue.
    Operates strictly on signed INT8 activations and weights (or packed UINT8 nibbles).
    Weights are strictly symmetric around zero (zw = 0).

    Zero-point correction:
        corrected_acc = raw_acc - zx * sum(Wq, dim=K)
        out = corrected_acc * sx[:, None] * sw[None, :] + (bias if bias is not None else 0)
    """
    m, k = xq.shape
    n = sw.numel()
    if k > 131071:
        raise ValueError("K is too large for worst-case signed INT8 accumulation")
    tensors = [xq, weight, sx, sw] + ([] if bias is None else [bias]) + ([] if zx is None else [zx])
    if any(not t.is_cuda or not t.is_contiguous() or t.device != xq.device for t in tensors):
        raise ValueError("Operands must be contiguous and on the same CUDA device")
    if xq.dtype != torch.int8:
        raise TypeError(f"integer_linear requires signed int8 activations, got {xq.dtype}")
    if weight.dtype != (torch.uint8 if packed else torch.int8):
        raise TypeError(f"Expected {'packed uint8' if packed else 'signed int8'} weights, got {weight.dtype}")
    expected_w_cols = (k + 1) // 2 if packed else k
    if weight.shape != (n, expected_w_cols):
        raise ValueError(f"Weight shape {weight.shape} does not match (N={n}, K_cols={expected_w_cols})")
    if sx.shape != (m,) or sw.shape != (n,) or sx.dtype != torch.float32 or sw.dtype != torch.float32:
        raise ValueError("Expected FP32 per-row activation and per-output weight scales")
    if bias is not None and (bias.shape != (n,) or bias.dtype != torch.float32):
        raise ValueError("Expected FP32 bias [N]")

    has_zx = zx is not None
    if has_zx:
        if zx.shape != (m,) or zx.dtype not in (torch.int32, torch.float32):
            raise ValueError("Expected [M] int32/float32 activation zero-points zx")
        if w_red is None:
            if packed:
                from .quantization import unpack_int4
                w_unpacked = unpack_int4(weight, k)
                w_red = w_unpacked.float().sum(dim=1)
            else:
                w_red = weight.float().sum(dim=1)
        if w_red.shape != (n,) or w_red.dtype != torch.float32:
            raise ValueError("Expected [N] float32 weight reduction buffer w_red")

    y = torch.empty((m, n), device=xq.device, dtype=torch.int32 if raw else torch.float32)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 64))
    kernel = _gemm[grid](
        xq, weight, sx, sw, bias if bias is not None else sw, y,
        zx if has_zx else sx, w_red if has_zx else sw,
        m, n, k,
        packed, raw, bias is not None, has_zx,
        32, 64, 64, num_warps=4, num_stages=3
    )
    return (y, kernel) if return_kernel else y
